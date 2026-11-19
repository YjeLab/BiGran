"""层1 · CNN 层（局部峰形）—— 多尺度【物理核宽】卷积 + 分支级 SE 重标定。

核宽**不是**任意 dilation，而是按真实红外带宽设定：
    122 cm⁻¹  宽峰：O–H / N–H 伸缩、氢键宽包
     50 cm⁻¹  中等
     18 cm⁻¹  尖峰：C=O / C≡N / C=C
（消融 #1 会对照 任意 dilation(1/2/4) 与 单尺度。）
"""
from __future__ import annotations

import torch
import torch.nn as nn

from bigran.constants import WN_STEP


def width_cm_to_kernel(width_cm: float, wn_step: float = WN_STEP) -> int:
    """物理带宽 (cm⁻¹) -> 奇数卷积核点数。"""
    k = int(round(width_cm / wn_step))
    k = max(3, k)
    return k if k % 2 == 1 else k + 1


class SEBlock1d(nn.Module):
    """通道注意力（squeeze-excitation），按分支对通道重标定。"""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, channels), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x: [B, C, T]
        s = x.mean(dim=-1)                                    # [B, C]
        return x * self.fc(s).unsqueeze(-1)


class PhysicsKernelStem(nn.Module):
    """[B, in_ch, N] -> [B, out_channels, N/stride]。"""

    def __init__(self, in_channels: int = 2, widths_cm=(122.0, 50.0, 18.0),
                 wn_step: float = WN_STEP, branch_channels: int = 64,
                 out_channels: int = 128, stride: int = 2) -> None:
        super().__init__()
        self.kernels = [width_cm_to_kernel(w, wn_step) for w in widths_cm]
        self.branches = nn.ModuleList()
        self.ses = nn.ModuleList()
        for k in self.kernels:
            self.branches.append(nn.Sequential(
                nn.Conv1d(in_channels, branch_channels, k, stride=stride, padding=k // 2),
                nn.BatchNorm1d(branch_channels),
                nn.GELU(),
            ))
            self.ses.append(SEBlock1d(branch_channels))
        self.fuse = nn.Sequential(
            nn.Conv1d(branch_channels * len(self.kernels), out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [se(br(x)) for br, se in zip(self.branches, self.ses)]
        return self.fuse(torch.cat(feats, dim=1))


class MultiScaleStem(nn.Module):
    """消融 #1 的对照臂：把物理核宽换成 任意 dilation / 单尺度。"""

    def __init__(self, in_channels: int = 2, mode: str = "dilation",
                 branch_channels: int = 64, out_channels: int = 128, stride: int = 2) -> None:
        super().__init__()
        self.mode = mode
        if mode == "dilation":
            self.kernels = [3, 3, 3]
            self.dilations = [1, 2, 4]
        elif mode == "single":
            self.kernels = [25]
            self.dilations = [1]
        else:
            raise ValueError(f"未知 mode: {mode}")
        self.branches = nn.ModuleList()
        self.ses = nn.ModuleList()
        for k, d in zip(self.kernels, self.dilations):
            self.branches.append(nn.Sequential(
                nn.Conv1d(in_channels, branch_channels, k, stride=stride,
                          padding=d * (k - 1) // 2, dilation=d),
                nn.BatchNorm1d(branch_channels),
                nn.GELU(),
            ))
            self.ses.append(SEBlock1d(branch_channels))
        self.fuse = nn.Sequential(
            nn.Conv1d(branch_channels * len(self.kernels), out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )
        self.kernels = [f"k{k}d{d}" for k, d in zip(self.kernels, self.dilations)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [se(br(x)) for br, se in zip(self.branches, self.ses)]
        return self.fuse(torch.cat(feats, dim=1))