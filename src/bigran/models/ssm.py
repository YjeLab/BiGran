"""层2 备选 · 对角状态空间编码器（S4D 家族）—— 替换 Transformer 的"新主干"。

动机：TSC 顶会证据（MS4, arXiv 2605.27406）显示**对角 SSM(S4D) 稳定胜过 Mamba 系且更省**；
而我们自己的消融又证明 CNN 层的"物理核宽/多尺度"立不住、band 偏置是唯一有效物理项 ——
所以把物理先验（band 偏置 + 绝对 cm⁻¹ 正弦编码）**移植到一个更现代的主干**上。

实现要点（对角 SSM 的**卷积核**形式，稳定且可复现）：
    离散化 Ā = exp(Δt·A)，B̄ = (Ā−1)/A·B（ZOH）
    冲激响应 K_t = Σ_s C_s·B̄_s·Ā_s^t          （t=0..L−1）
    y = causal_conv(x, K) + D·x
K 只依赖参数、与 batch 无关，故每次前向算一次即可，开销小。

⚠️ 这里**不是**逐字复刻 S4D（那需要 HiPPO 初始化与特定参数化）；本实现是对角 SSM 家族
的一个干净实例（log-均匀负实 A 初始化）。引用时按"对角 SSM（S4D 家族）"表述，别说成 S4D 原版。
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bigran.models.encoder import sine_positional_encoding


class DiagonalSSM(nn.Module):
    """[B, L, H] -> [B, L, H]，逐通道对角 SSM（深度可分离）。"""

    def __init__(self, d_model: int, d_state: int = 16) -> None:
        super().__init__()
        self.d_model, self.d_state = d_model, d_state
        # A: 负实、log-均匀初始化（跨时间尺度），参数化在 log 空间
        a = torch.exp(torch.linspace(math.log(1.0), math.log(d_state), d_state))
        self.A_log = nn.Parameter(torch.log(a).repeat(d_model, 1))          # [H, S]
        self.B = nn.Parameter(torch.ones(d_model, d_state))                 # [H, S]
        self.C = nn.Parameter(torch.ones(d_model, d_state))                 # [H, S]
        # Δt 逐通道（softplus 参数化）
        dt = torch.exp(torch.rand(d_model) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
        self.dt_bias = nn.Parameter(torch.log(torch.expm1(dt)))            # softplus^-1
        self.D = nn.Parameter(torch.ones(d_model))

    def kernel(self, L: int, device, dtype) -> torch.Tensor:
        """冲激响应 K [H, L]（与 batch 无关）。"""
        A = -torch.exp(self.A_log)                                          # [H,S] 负实
        dt = F.softplus(self.dt_bias)                                       # [H]
        Abar = torch.exp(dt[:, None] * A)                                   # [H,S]
        Bbar = (Abar - 1.0) / A * self.B                                    # ZOH: [H,S]
        logAbar = (dt[:, None] * A)                                         # [H,S]
        t = torch.arange(L, device=device, dtype=logAbar.dtype)             # [L]
        # Ā^t = exp(t·logĀ) -> [L,H,S]
        powA = torch.exp(t[:, None, None] * logAbar[None].to(dtype))
        K = (powA * (Bbar * self.C)[None].to(dtype)).sum(-1)                # [L,H]
        return K.transpose(0, 1).contiguous()                              # [H,L]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, H = x.shape
        K = self.kernel(L, x.device, x.dtype)                               # [H,L]
        xc = x.transpose(1, 2)                                              # [B,H,L]
        y = F.conv1d(F.pad(xc, (L - 1, 0)), K[:, None, :], groups=H)        # 因果卷积 [B,H,L]
        return (y + self.D[None, :, None] * xc).transpose(1, 2)


class SSMBlock(nn.Module):
    """pre-norm 残差块：SSM + FFN（结构与 TransformerEncoderLayer 对齐，便于对照）。"""

    def __init__(self, d_model: int, dim_feedforward: int, d_state: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.ssm = DiagonalSSM(d_model, d_state)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.ssm(self.norm1(x)))
        return x + self.drop(self.ff(self.norm2(x)))


class SSMEncoder(nn.Module):
    """与 `SpectrumEncoder` **同接口**：forward(x[B,T,d], wn[T]) -> [B,T//downsample,d]。

    保留**绝对 cm⁻¹ 正弦编码**（物理量位置）—— 这是把物理先验带进新主干的关键，
    也是对"同一条谱重采样后同波数仍落同一编码"这一可复现性主张的延续。
    """

    def __init__(self, d_model: int = 128, n_layers: int = 3, nhead: int = 4,
                 dim_feedforward: int = 256, dropout: float = 0.1, downsample: int = 2,
                 d_state: int = 16) -> None:
        super().__init__()
        self.downsample = downsample
        self.proj = (nn.Conv1d(d_model, d_model, 3, stride=downsample, padding=1)
                     if downsample > 1 else nn.Identity())
        self.blocks = nn.ModuleList(
            [SSMBlock(d_model, dim_feedforward, d_state, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, wn: np.ndarray) -> torch.Tensor:
        if self.downsample > 1:
            x = self.proj(x.transpose(1, 2)).transpose(1, 2)
        wn_sub = np.asarray(wn)[:: self.downsample][: x.shape[1]]
        x = x + sine_positional_encoding(wn_sub, x.shape[-1]).to(x.device).unsqueeze(0)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)