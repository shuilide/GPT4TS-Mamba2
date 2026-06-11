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

# ==========================================
# 纯 Mamba 层（实验五：Pure Mamba架构）
# ==========================================
class PureMambaLayer(nn.Module):
    """
    纯Mamba架构：不使用GPT层，直接处理输入
    数据流：input → Mamba → output
    """

    def __init__(self, d_model, use_official_mamba: bool):
        super().__init__()
        self.use_official_mamba = use_official_mamba

        # 可训练的 Mamba 模块
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

        self.mamba_ln = nn.LayerNorm(d_model)

    def forward(self, hidden_states, *args, **kwargs):
        # 纯Mamba：直接处理输入，不使用GPT
        mamba_out = self.mamba(self.mamba_ln(hidden_states))
        if isinstance(mamba_out, tuple):
            mamba_out = mamba_out[0]

        # 返回Mamba的输出（保持与GPT2层相同的输出格式）
        return (mamba_out, None)


# ==========================================
# 串行 Mamba 层（实验五：真正的串行架构）
# ==========================================
class SequentialMambaLayer(nn.Module):
    """
    串行架构：先过GPT-2层，再过Mamba层，没有门控融合
    数据流：input → GPT → Mamba → output
    """

    def __init__(self, gpt_layer, d_model, use_official_mamba: bool):
        super().__init__()
        self.gpt_layer = gpt_layer  # 冻结的原始 GPT 层
        self.use_official_mamba = use_official_mamba

        # 可训练的 Mamba 模块
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

        self.mamba_ln = nn.LayerNorm(d_model)

    def forward(self, hidden_states, *args, **kwargs):
        # 串行：先过GPT，再过Mamba
        with torch.no_grad():
            gpt_out = self.gpt_layer(hidden_states, *args, **kwargs)[0]

        # GPT输出直接送入Mamba（无门控）
        mamba_out = self.mamba(self.mamba_ln(gpt_out))
        if isinstance(mamba_out, tuple):
            mamba_out = mamba_out[0]

        # 返回Mamba的输出
        return (mamba_out, None)


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
        # 默认：平行融合架构
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

    def get_gate_value(self):
        """获取当前 gate 参数的值（用于实验二追踪）"""
        return self.gate.item()

    def set_gate_initial(self, initial_value=0.1):
        """设置 gate 的初始值（用于实验）"""
        with torch.no_grad():
            self.gate.fill_(initial_value)


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
        # 实验3：支持随机初始化对比（Pre-trained vs From Scratch）
        use_pretrained = not config.get('no_pretrained', False)
        if use_pretrained:
            self.gpt2 = GPT2Model.from_pretrained('gpt2', output_hidden_states=False, output_attentions=False)
            print("✓ Loading pre-trained GPT-2 weights (Frozen)")
        else:
            from transformers import GPT2Config
            gpt2_config = GPT2Config(
                n_positions=1024,
                n_embd=768,
                n_layer=12,
                n_head=12,
                n_inner=3072
            )
            self.gpt2 = GPT2Model(gpt2_config)
            print("✓ Using randomly initialized GPT-2 (From Scratch, Frozen)")

        # ✅ 修复5：扩展GPT-2位置编码以支持长序列（使用线性插值）
        # 计算最大可能的patch数量（考虑padding和prompt token）
        max_patch_num = (self.max_len - self.patch_size) // self.stride + 2  # +1 padding + 1 prompt

        if max_patch_num > self.gpt2.config.n_positions:
            old_n_positions = self.gpt2.config.n_positions

            # 获取原始位置编码权重
            old_wpe_weight = self.gpt2.wpe.weight.data.clone()  # [1024, 768]

            # 使用线性插值创建新的位置编码
            new_wpe_weight = torch.zeros(max_patch_num, self.d_model)
            with torch.no_grad():
                for i in range(max_patch_num):
                    # 计算在原始编码中的对应位置（可能是小数）
                    original_pos = i * (old_n_positions - 1) / (max_patch_num - 1)

                    # 找到相邻的两个整数位置
                    low = int(original_pos)
                    high = min(low + 1, old_n_positions - 1)

                    # 计算插值权重
                    fraction = original_pos - low

                    # 线性插值
                    new_wpe_weight[i] = (1 - fraction) * old_wpe_weight[low] + fraction * old_wpe_weight[high]

            # 创建新的位置编码层并替换
            new_wpe = nn.Embedding(max_patch_num, self.d_model)
            new_wpe.weight.data = new_wpe_weight

            # 替换模型中的位置编码
            self.gpt2.wpe = new_wpe

            # 重要：同时更新两个配置项
            self.gpt2.config.n_positions = max_patch_num
            self.gpt2.config.max_position_embeddings = max_patch_num

            print(f"✓ Extended GPT2 position encoding with interpolation: {old_n_positions} -> {max_patch_num}")

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

        # 实验五：根据arch_mode决定架构
        arch_mode = config.get('arch_mode', 'parallel')

        if arch_mode == 'pure_gpt':
            # 纯GPT-2：不替换任何层，不使用Mamba
            print("✓ Architecture: Pure GPT-2 (No Mamba adapters)")

        elif arch_mode == 'pure_mamba':
            # 纯Mamba：将所有GPT层替换为纯Mamba层
            print("✓ Architecture: Pure Mamba Network (Replacing all GPT layers with Pure Mamba)")
            for layer_idx in range(self.gpt_layers):
                # 使用PureMambaLayer（不使用GPT，无门控）
                self.gpt2.h[layer_idx] = PureMambaLayer(
                    d_model=config['d_model'],
                    use_official_mamba=MAMBA2_AVAILABLE
                )

        elif arch_mode == 'sequential':
            # 串行架构：前几层GPT-2，后几层Mamba（无门控）
            print(
                f"✓ Architecture: Sequential (First {self.gpt_layers - replace_last_n_layers} GPT layers → Last {replace_last_n_layers} Mamba layers)")
            for i in range(replace_last_n_layers):
                layer_idx = self.gpt_layers - 1 - i
                original_gpt_layer = self.gpt2.h[layer_idx]
                # 使用SequentialMambaLayer（无门控的串行架构）
                self.gpt2.h[layer_idx] = SequentialMambaLayer(
                    gpt_layer=original_gpt_layer,
                    d_model=config['d_model'],
                    use_official_mamba=MAMBA2_AVAILABLE
                )

        else:  # parallel（默认）
            # 平行融合架构：你的方法
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

        # ==========================================
        # 冻结策略（支持三种模式，与 exp6 实验一致）
        # ==========================================
        freeze_strategy = config.get('freeze_strategy', 'partial_frozen')  # 默认使用部分冻结

        if freeze_strategy == 'all_frozen':
            # 策略1：全冻结 - 只微调 Mamba、Gate 和分类头，冻结所有 GPT2 参数
            for name, param in self.gpt2.named_parameters():
                if 'mamba' in name.lower() or 'gate' in name.lower():
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            print(f"✓ Freeze Strategy: ALL FROZEN")
            print(f"  - GPT2 (Attention + FFN + LN + WPE): Frozen")
            print(f"  - Mamba + Gate: Trainable")

        elif freeze_strategy == 'partial_frozen':
            # 策略2：部分冻结（与 Classification2 一致）- 微调 LN + WPE + Mamba + Gate
            for name, param in self.gpt2.named_parameters():
                if 'ln' in name or 'wpe' in name or 'mamba' in name.lower() or 'gate' in name.lower():
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            print(f"✓ Freeze Strategy: PARTIAL FROZEN (Classification2 style)")
            print(f"  - GPT2 (Attention + FFN): Frozen")
            print(f"  - GPT2 (LN + WPE) + Mamba + Gate: Trainable")

        elif freeze_strategy == 'all_finetune':
            # 策略3：全微调 - 所有参数都可训练
            for param in self.gpt2.parameters():
                param.requires_grad = True
            print(f"✓ Freeze Strategy: ALL FINETUNE")
            print(f"  - All GPT2 + Mamba + Gate: Trainable")

        else:
            raise ValueError(
                f"Unknown freeze_strategy: {freeze_strategy}. Choose from ['all_frozen', 'partial_frozen', 'all_finetune']")

        # ✅ 统计可训练参数数量
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"✓ Total Parameters: {total_params:,} | Trainable: {trainable_params:,} ({trainable_params / total_params * 100:.2f}%)")

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
            # 实验4：默认关闭属性捕获
            self.return_attn = False
        else:
            # ✅ 修复6：使用自适应池化支持任意长度序列
            self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
            self.ln_proj = nn.LayerNorm(self.d_model)
            self.out_layer = nn.Linear(self.d_model, self.num_classes)
            print("✓ Innovation 3: Attn Pooling disabled, using Adaptive Pooling")
            self.return_attn = False

        # ✅ 关键修复：统一初始化中间结果存储属性（无论哪种 pooling 模式）
        # 这样可以避免 extract_data.py 访问不存在的属性
        self.last_attention_weights = None
        self.last_pooled_feature = None

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
            # 获取注意力权重
            pooled_out, attn_weights = self.pool_attention(query, outputs, outputs)

            # ✅ 实验4：保存注意力权重 (Shape: [B, 1, N])
            # 关键修复：无论 return_attn 是否为 True，都保存以便 extract_data.py 使用
            self.last_attention_weights = attn_weights

            pooled_out = pooled_out.squeeze(1)
            pooled_out = self.pool_ln(pooled_out)

            # ✅ 实验4：保存池化后的特征 (Shape: [B, D])
            self.last_pooled_feature = pooled_out

            outputs = self.out_layer(pooled_out)
        else:
            # ✅ 使用自适应池化支持任意长度
            # outputs shape: [B, N, D] -> [B, D, N] for AdaptiveAvgPool1d
            outputs = outputs.transpose(1, 2)  # [B, D, N]
            outputs = self.adaptive_pool(outputs).squeeze(-1)  # [B, D]
            outputs = self.ln_proj(outputs)

            # ✅ 实验4：保存池化后的特征 (Shape: [B, D])
            self.last_pooled_feature = outputs

            outputs = self.out_layer(outputs)

        return outputs

    def get_all_gate_values(self):
        """获取所有 ParallelMambaAdapter 层的 gate 值（用于实验二）"""
        gate_values = {}
        for name, module in self.gpt2.named_modules():
            if isinstance(module, ParallelMambaAdapter):
                gate_values[name] = module.get_gate_value()
        return gate_values