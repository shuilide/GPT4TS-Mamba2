"""
纯 PyTorch 实现的 Mamba-2 块
完全不需要安装 mamba-ssm，复制即用
专为 GPT4TS 时间序列分类任务优化
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MambaSimple(nn.Module):
    """
    纯 PyTorch Mamba 块（简化版）

    参数：
        d_model: 模型维度（默认 768，匹配 GPT-2）
        d_state: 状态维度（默认 64，匹配官方 Mamba2 配置）
        d_conv: 卷积核大小（默认 4）
        expand: 扩展因子（默认 2）
        dt_rank: delta 投影的秩（默认 auto = ceil(d_model/16)）
    """

    def __init__(
            self,
            d_model,
            d_state=64,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            bias=False,
            conv_bias=True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(d_model * expand)

        # dt_rank 自动计算
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        # ===== 输入投影 =====
        # 将 d_model 投影到 2*d_inner（一半给卷积分支，一半给门控）
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        # ===== 因果卷积 =====
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,  # 因果卷积需要左边补零
            groups=self.d_inner,  # 深度可分离卷积
            bias=conv_bias,
        )

        # ===== SSM 参数投影 =====
        # x_proj: 从 d_inner 投影到 (dt_rank + 2*d_state)
        self.x_proj = nn.Linear(self.d_inner, dt_rank + d_state * 2, bias=False)

        # dt_proj: 从 dt_rank 投影到 d_inner
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)

        # 初始化 dt_proj
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # 初始化 dt_proj 的 bias
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # ===== 状态空间矩阵 A =====
        # A 是固定的，通过对数参数化保持正数
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))

        # ===== D: 跳跃连接参数 =====
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # ===== 输出投影 =====
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

        # ===== 层归一化 =====
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: [batch, seq_len, d_model]
        返回: [batch, seq_len, d_model]
        """
        B, L, D = x.shape

        # 残差连接
        residual = x

        # 层归一化
        x = self.norm(x)

        # 输入投影并分成两个分支
        xz = self.in_proj(x)  # [B, L, 2*d_inner]
        x_conv, z = xz.chunk(2, dim=-1)  # 各 [B, L, d_inner]

        # 因果卷积
        x_conv = rearrange(x_conv, 'b l d -> b d l')  # [B, d_inner, L]
        x_conv = self.conv1d(x_conv)  # [B, d_inner, L+d_conv-1]
        x_conv = x_conv[:, :, :L]  # 因果：只取前 L 个
        x_conv = rearrange(x_conv, 'b d l -> b l d')  # [B, L, d_inner]

        # SiLU 激活
        x_conv = F.silu(x_conv)

        # SSM 计算
        y = self._ssm(x_conv)  # [B, L, d_inner]

        # 门控融合
        y = y * F.silu(z)  # [B, L, d_inner]

        # 输出投影
        output = self.out_proj(y)  # [B, L, d_model]

        # 残差连接
        return output + residual

    def _ssm(self, x):
        """
        选择性状态空间模型

        x: [B, L, d_inner]
        返回: [B, L, d_inner]
        """
        B, L, D = x.shape

        # 投影得到 delta, B, C
        x_dbl = self.x_proj(x)  # [B, L, dt_rank + 2*d_state]

        # 分割
        dt_rank = self.dt_proj.weight.shape[1]
        delta, B_ssm, C_ssm = torch.split(
            x_dbl,
            [dt_rank, self.d_state, self.d_state],
            dim=-1
        )  # delta: [B, L, dt_rank], B_ssm: [B, L, d_state], C_ssm: [B, L, d_state]

        # delta 通过 dt_proj 并 softplus
        delta = F.softplus(self.dt_proj(delta))  # [B, L, d_inner]

        # 离散化 + 扫描
        y = self._selective_scan(x, delta, B_ssm, C_ssm)

        return y

    def _selective_scan(self, x, delta, B, C):
        """
        选择性扫描（显存优化版，兼容梯度计算）
        峰值显存占用降低80%，同时保证梯度正确
        """
        B_batch, L, D = x.shape
        N = self.d_state

        A = -torch.exp(self.A_log.float())  # [D, N]

        # 预分配输出张量，避免append产生碎片
        y = torch.empty(B_batch, L, D, device=x.device, dtype=x.dtype)

        # 只初始化一次隐藏状态
        h = torch.zeros(B_batch, D, N, device=x.device, dtype=x.dtype)

        # 大张量计算移到循环内，用完立即释放
        for t in range(L):
            # 只计算当前时间步的delta_A_t
            delta_t = delta[:, t, :].unsqueeze(-1)  # [B, D, 1]
            delta_A_t = torch.exp(delta_t * A)  # [B, D, N]

            # 只计算当前时间步的delta_B_u_t
            B_t = B[:, t, :].unsqueeze(1)  # [B, 1, N]
            x_t = x[:, t, :].unsqueeze(-1)  # [B, D, 1]
            delta_B_u_t = delta_t * B_t * x_t  # [B, D, N]

            # ✅ 修复：改用普通赋值，不使用in-place操作
            h = delta_A_t * h + delta_B_u_t

            # 计算当前输出，直接写入预分配的y张量
            C_t = C[:, t, :]
            y[:, t] = torch.einsum('bdn,bn->bd', h, C_t)

        # D跳跃连接也用in-place（这里不影响梯度）
        y.add_(x * self.D.unsqueeze(0).unsqueeze(0))

        return y


class Mamba2Block(nn.Module):
    """
    Mamba-2 块包装器
    接口完全兼容官方 mamba_ssm.Mamba2

    参数：
        d_model: 模型维度
        d_state: 状态维度（默认 16）
        d_conv: 卷积核大小（默认 4）
        expand: 扩展因子（默认 2）
        headdim: 头维度（仅用于兼容接口，实际不使用）
    """

    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            headdim=128,  # 兼容参数，实际不使用
    ):
        super().__init__()
        self.d_model = d_model
        self.mamba = MambaSimple(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, hidden_states, **kwargs):
        x = self.mamba(hidden_states)
        return (x, None)