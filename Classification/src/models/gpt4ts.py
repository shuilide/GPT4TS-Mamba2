from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from transformers import GPT2Model
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


class gpt4ts(nn.Module):

    def __init__(self, config, data):
        super(gpt4ts, self).__init__()
        self.pred_len = 0
        self.seq_len = data.max_seq_len
        self.max_len = data.max_seq_len
        self.patch_size = config['patch_size']
        self.stride = config['stride']
        self.gpt_layers = 6  # ✅ 完整保留6层GPT2，不修改
        self.feat_dim = data.feature_df.shape[1]
        self.num_classes = len(data.class_names)
        self.d_model = config['d_model']

        assert self.d_model == 768, "GPT2-small requires d_model=768"

        self.patch_num = (self.seq_len - self.patch_size) // self.stride + 1
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1
        self.enc_embedding = DataEmbedding(self.feat_dim * self.patch_size, config['d_model'], config['dropout'])

        # ✅ 加载完整的6层GPT2，不替换任何一层
        self.gpt2 = GPT2Model.from_pretrained('gpt2')
        self.gpt2.h = self.gpt2.h[:self.gpt_layers]

        # ✅ 【核心修改】在GPT2之后追加2层独立的Mamba模块
        self.num_mamba_layers = 2
        self.mamba_layers = nn.ModuleList()

        for _ in range(self.num_mamba_layers):
            if MAMBA2_AVAILABLE:
                self.mamba_layers.append(Mamba2(
                    d_model=self.d_model,
                    d_state=16,
                    d_conv=4,
                    expand=2,
                    headdim=128
                ))
            else:
                self.mamba_layers.append(Mamba2Block(
                    d_model=self.d_model,
                    d_state=16,
                    d_conv=4,
                    expand=2,
                    headdim=128
                ))

        print(f"✓ 完整保留6层GPT2，在输出后追加 {self.num_mamba_layers} 层Mamba-2")

        # ✅ 参数解冻策略：GPT2保持原始冻结方式，Mamba层全部可训练
        for name, param in self.gpt2.named_parameters():
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # ✅ 所有Mamba层自动解冻，不需要手动设置
        for name, param in self.mamba_layers.named_parameters():
            param.requires_grad = True

        self.act = F.gelu
        self.dropout = nn.Dropout(config['dropout'])
        self.ln_proj = nn.LayerNorm(config['d_model'] * self.patch_num)
        self.out_layer = nn.Linear(config['d_model'] * self.patch_num, self.num_classes)

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, mask=None):
        B, L, M = x_enc.shape

        input_x = rearrange(x_enc, 'b l m -> b m l')
        input_x = self.padding_patch_layer(input_x)
        input_x = input_x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        input_x = rearrange(input_x, 'b m n p -> b n (p m)')

        outputs = self.enc_embedding(input_x, None)
        outputs = self.dropout(outputs)

        # ✅ 第一步：完整经过原始6层GPT2，和基线完全一致
        outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state
        outputs = self.dropout(outputs)

        # ✅ 第二步：经过追加的2层Mamba模块
        for mamba_layer in self.mamba_layers:
            outputs = mamba_layer(outputs)[0]  # 兼容官方和纯PyTorch版接口
            outputs = self.dropout(outputs)

        # ✅ 分类头和原始代码完全一致
        outputs = self.act(outputs).reshape(B, -1)
        outputs = self.ln_proj(outputs)
        outputs = self.out_layer(outputs)

        return outputs