from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from transformers import GPT2ForSequenceClassification
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import BertTokenizer, BertModel
from einops import rearrange
from models.embed import DataEmbedding, DataEmbedding_wo_time

try:
    from mamba_ssm import Mamba2

    MAMBA2_AVAILABLE = True
    print("✓ Using official mamba-ssm (CUDA accelerated)")
except ImportError:
    MAMBA2_AVAILABLE = False
    from models.mamba_simple import Mamba2Block

    print("✓ Using pure PyTorch Mamba-2 (no installation needed)")


# ==========================================
# 创新点2：平行 Mamba-Attention 适配器模块
# ==========================================
class ParallelMambaAdapter(nn.Module):
    def __init__(self, gpt_layer, d_model, use_official_mamba: bool):
        super().__init__()
        self.gpt_layer = gpt_layer  # 冻结的原始 GPT 层
        self.use_official_mamba = use_official_mamba

        # 可训练的 Mamba 模块（兼容官方/简易版）
        if use_official_mamba:
            self.mamba = Mamba2(
                d_model=d_model,
                d_state=16,
                d_conv=4,
                expand=2,
                headdim=128
            )
        else:
            self.mamba = Mamba2Block(
                d_model=d_model,
                d_state=16,
                d_conv=4,
                expand=2,
                headdim=128
            )

        # 创新：可学习的门控权重，初始偏向 GPT (0.1表示 90% GPT, 10% Mamba)
        self.gate = nn.Parameter(torch.ones(1) * 0.1)
        self.mamba_ln = nn.LayerNorm(d_model)

    def forward(self, hidden_states, *args, **kwargs):
        # 1. 走冻结的 GPT 层路线 (不需要梯度)
        with torch.no_grad():
            gpt_out = self.gpt_layer(hidden_states, *args, **kwargs)[0]  # 只取hidden_states

        # 2. 走可训练的 Mamba 路线
        mamba_out = self.mamba(self.mamba_ln(hidden_states))
        if isinstance(mamba_out, tuple):  # 兼容简易版Mamba2Block的输出格式
            mamba_out = mamba_out[0]

        # 3. 动态融合
        output = gpt_out + self.gate * mamba_out

        # ✅ 关键修复：返回两个元素的tuple，完全匹配GPT2层输出格式
        # present=None表示不使用缓存，不影响训练和推理
        return (output, None)

class gpt4ts(nn.Module):
    def __init__(self, config, data):
        super(gpt4ts, self).__init__()
        self.pred_len = 0
        self.seq_len = data.max_seq_len
        self.max_len = data.max_seq_len
        self.patch_size = config['patch_size']
        self.stride = config['stride']
        self.gpt_layers = config.get('gpt_layers', 6)  # 支持配置，默认6层
        self.feat_dim = data.feature_df.shape[1]
        self.num_classes = len(data.class_names)
        self.d_model = config['d_model']

        # ✅ 修复1：添加断言，确保d_model与GPT2兼容
        assert self.d_model == 768, "GPT2-small requires d_model=768"

        self.patch_num = (self.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1  # padding后patch_num+1
        self.enc_embedding = DataEmbedding(self.feat_dim * self.patch_size, config['d_model'], config['dropout'])

        # ✅ 修复2：关闭不必要的输出，节省显存
        self.gpt2 = GPT2Model.from_pretrained('gpt2')

        # ==========================================
        # 创新点1：时序统计特征 Prompt（支持消融）
        # ==========================================
        self.use_stat_prompt = not config.get('no_stat_prompt', False)
        if self.use_stat_prompt:
            self.prompt_generator = nn.Sequential(
                nn.Linear(self.feat_dim * 4, self.d_model // 2),
                nn.GELU(),
                nn.Linear(self.d_model // 2, self.d_model)
            )
            self.patch_num += 1  # Prompt token使序列长度+1
            print("✓ Innovation 1: Stat Prompt enabled")
        else:
            print("✓ Innovation 1: Stat Prompt disabled")

        # ==========================================
        # 创新点2：替换原有 Mamba 直接替换逻辑为平行适配器
        # 支持通过配置控制替换的层数（用于消融实验）
        # ==========================================
        replace_last_n_layers = config.get('num_mamba_layers', 2)  # 支持配置，默认替换最后2层
        
        if replace_last_n_layers > 0:
            for i in range(replace_last_n_layers):
                layer_idx = self.gpt_layers - 1 - i
                original_gpt_layer = self.gpt2.h[layer_idx]
                # 包装为平行结构（区分官方/简易版Mamba）
                self.gpt2.h[layer_idx] = ParallelMambaAdapter(
                    gpt_layer=original_gpt_layer,
                    d_model=config['d_model'],
                    use_official_mamba=MAMBA2_AVAILABLE
                )
            print(f"✓ Replaced last {replace_last_n_layers} layers with Parallel Mamba-Attention Adapters")
        else:
            print(f"✓ Using pure GPT2 without Mamba adapters (baseline)")

        self.gpt2.h = self.gpt2.h[:self.gpt_layers]

        # ✅ 修复3：正确解冻Mamba2层 + 创新点2的gate参数
        for name, param in self.gpt2.named_parameters():
            if 'ln' in name or 'wpe' in name or 'mamba' in name.lower() or 'gate' in name.lower():
                param.requires_grad = True
            else:
                param.requires_grad = False

        # ✅ 修复4：删除硬编码的设备移动（让main.py统一处理）
        self.act = F.gelu
        self.dropout = nn.Dropout(config['dropout'])

        # ==========================================
        # 创新点3：Attention Pooling（支持消融）
        # ==========================================
        self.use_attn_pooling = not config.get('no_attn_pooling', False)
        if self.use_attn_pooling:
            self.cls_query = nn.Parameter(torch.randn(1, 1, self.d_model))
            self.pool_attention = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=8, batch_first=True)
            self.pool_ln = nn.LayerNorm(self.d_model)
            self.out_layer = nn.Linear(self.d_model, self.num_classes)
            print("✓ Innovation 3: Attn Pooling enabled")
        else:
            self.ln_proj = nn.LayerNorm(config['d_model'] * self.patch_num)
            self.out_layer = nn.Linear(config['d_model'] * self.patch_num, self.num_classes)
            print("✓ Innovation 3: Attn Pooling disabled, using Flatten")

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, mask=None):
        B, L, M = x_enc.shape

        # 创新点1：统计特征 Prompt
        if self.use_stat_prompt:
            x_mean = x_enc.mean(dim=1)
            x_std = x_enc.std(dim=1)
            x_max = x_enc.max(dim=1)[0]
            x_min = x_enc.min(dim=1)[0]
            stats = torch.cat([x_mean, x_std, x_max, x_min], dim=-1)
            prompt_token = self.prompt_generator(stats).unsqueeze(1)

        # Patch处理
        input_x = rearrange(x_enc, 'b l m -> b m l')
        input_x = self.padding_patch_layer(input_x)
        input_x = input_x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        input_x = rearrange(input_x, 'b m n p -> b n (p m)')

        outputs = self.enc_embedding(input_x, None)
        outputs = self.dropout(outputs)

        # 拼接Prompt Token
        if self.use_stat_prompt:
            outputs = torch.cat([prompt_token, outputs], dim=1)

        # GPT2主干
        outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state
        outputs = self.dropout(outputs)
        outputs = self.act(outputs)

        # 创新点3：分类头
        if self.use_attn_pooling:
            query = self.cls_query.expand(B, -1, -1)
            pooled_out, _ = self.pool_attention(query, outputs, outputs)
            pooled_out = pooled_out.squeeze(1)
            pooled_out = self.pool_ln(pooled_out)
            outputs = self.out_layer(pooled_out)
        else:
            outputs = outputs.reshape(B, -1)
            outputs = self.ln_proj(outputs)
            outputs = self.out_layer(outputs)

        return outputs