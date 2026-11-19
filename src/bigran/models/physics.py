"""层3 · 物理信息层 —— 可微的谱证据提取 + band-anchor 掩码构造。

P1 谱证据损失用的 `e_k(x)`：标签 k 的锚点窗口 W_k 内，吸光度相对**局部基线**的极大。
必须是**可微**的，否则不能进损失 —— 用 softmax/softmin 池化代替真 max/min。

只对**尖锐特征带**的标签有意义；宽峰/氢键/重叠类（不 sharp）在损失侧跳过。
"""
from __future__ import annotations

import numpy as np
import torch


def masked_softmax_pool(x: torch.Tensor, mask: torch.Tensor, tau: float,
                        largest: bool = True) -> torch.Tensor:
    """在 mask 内做软池化。

    x:    [B, T] 吸光度
    mask: [L, T] 0/1（同一掩码对所有 batch 共享）
    ->    [B, L]
    """
    if tau <= 0:
        raise ValueError("tau 必须为正")
    m = mask.to(dtype=torch.bool)
    xb = x.unsqueeze(1)                                   # [B,1,T]
    if largest:
        score = xb / tau
    else:
        score = -xb / tau
    score = score + torch.where(m, torch.zeros((), device=x.device),
                                torch.full((), -1e9, device=x.device)).unsqueeze(0)
    w = torch.softmax(score, dim=-1)                      # [B,L,T]
    return (w * xb).sum(dim=-1)


def soft_evidence(x_abs: torch.Tensor, anchor_mask: torch.Tensor, tau: float = 0.05,
                  subtract_local_baseline: bool = True) -> torch.Tensor:
    """e_k(x)：窗口内吸光度峰值（软）。返回 [B, L]。

    相对局部基线 = 窗口内软极大 − 窗口内软极小。
    这样对"整段抬高"的基线漂移不敏感，只对**窗口内真的有个峰**有响应。
    """
    peak = masked_softmax_pool(x_abs, anchor_mask, tau, largest=True)
    if not subtract_local_baseline:
        return peak
    base = masked_softmax_pool(x_abs, anchor_mask, tau, largest=False)
    return peak - base


def build_anchor_mask(lengths: torch.Tensor | None, band_shape: tuple[int, int]) -> torch.Tensor:
    """占位：真实掩码由 band_table 预计算成 buffer 后传入，这里只做形状校验。"""
    if lengths is None:
        return torch.ones(band_shape, dtype=torch.float32)
    raise ValueError("锚点掩码应由 BandTable.window_mask 预计算；见 bigran.py")


def window_mask_for_grid(band_table, wn: np.ndarray, pad: float = 0.0) -> torch.Tensor:
    """把 BandTable 的锚点窗口铺到 **编码器输出** 的波数网格上 -> [L, T] float。"""
    m = band_table.window_mask(np.asarray(wn, dtype=np.float64), pad=pad)
    return torch.as_tensor(m, dtype=torch.float32)