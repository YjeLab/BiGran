"""层2 · Transformer 层（跨波段长程）—— encoder-only，★ 固定非可选。

位置编码用**绝对 cm⁻¹ 的正弦编码**（物理量），不是可学 index。
这点是可复现性的关键：因为位置是物理波数，同一条谱重采样到别的点数后，
"同一波数"仍落在同一位置编码上 —— 而可学 index 编码做不到。
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

WN_SCALE = 1000.0          # cm⁻¹ -> 无量纲，使正弦参数落在合理区间


def sine_positional_encoding(wn: np.ndarray, d_model: int,
                             scale: float = WN_SCALE) -> torch.Tensor:
    """wn: [T] 绝对波数 -> [T, d_model]。

    `scale` 决定相邻 token 的相位差：pos = wn/scale。默认 1000 使 500–4100 cm⁻¹
    只映射到 [0.5, 4.1]，相邻 token（8 cm⁻¹）相位差仅 ~0.002 ⇒ 位置信息很弱。
    调小 scale 可让正弦在波数轴上展开得更充分（消融用）。
    """
    if d_model % 2 != 0:
        raise ValueError(f"d_model 必须为偶数，得到 {d_model}")
    pos = np.asarray(wn, dtype=np.float64) / scale
    div = np.exp(-math.log(10000.0) * np.arange(0, d_model, 2, dtype=np.float64) / d_model)
    pe = np.zeros((pos.size, d_model), dtype=np.float64)
    pe[:, 0::2] = np.sin(pos[:, None] * div[None, :])
    pe[:, 1::2] = np.cos(pos[:, None] * div[None, :])
    return torch.as_tensor(pe, dtype=torch.float32)


class SpectrumEncoder(nn.Module):
    """[B, T, d] -> [B, T//downsample, d]。"""

    def __init__(self, d_model: int = 128, n_layers: int = 3, nhead: int = 4,
                 dim_feedforward: int = 256, dropout: float = 0.1, downsample: int = 2,
                 pe_scale: float = WN_SCALE, pe_mode: str = "absolute",
                 pe_max_len: int = 1024) -> None:
        super().__init__()
        self.downsample = downsample
        self.pe_scale = float(pe_scale)
        self.pe_mode = pe_mode
        self.proj = (nn.Conv1d(d_model, d_model, 3, stride=downsample, padding=1)
                     if downsample > 1 else nn.Identity())
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.body = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        if pe_mode == "learnable":
            self.pe_table = nn.Parameter(torch.zeros(pe_max_len, d_model))
            nn.init.normal_(self.pe_table, std=0.02)

    def forward(self, x: torch.Tensor, wn: np.ndarray) -> torch.Tensor:
        if self.downsample > 1:
            x = self.proj(x.transpose(1, 2)).transpose(1, 2)
        if self.pe_mode == "learnable":
            x = x + self.pe_table[: x.shape[1]].unsqueeze(0)
        else:
            wn_sub = np.asarray(wn)[:: self.downsample][: x.shape[1]]
            x = x + sine_positional_encoding(wn_sub, x.shape[-1], self.pe_scale).to(x.device).unsqueeze(0)
        return self.norm(self.body(x))


class MambaEncoder(nn.Module):
    """消融 #3 的对照臂：编码器换成 Mamba。

    需要 `pip install mamba-ssm`（含 CUDA 扩展）。未安装时**明确报错**，
    不做静默降级 —— 消融结果不能是"假 Mamba"。
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        try:
            from mamba_ssm import Mamba  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "消融 #3 的 Mamba 臂需要 mamba-ssm：pip install mamba-ssm causal-conv1d"
            ) from e
        raise NotImplementedError("Mamba 臂待接入（先跑通主干，消融在后面补）")