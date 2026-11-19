"""标签 → 特征波数锚点（层3 物理先验的底座）。

72 类的标签名自带化学特征波数，形如：
    "C─H alkyl methyl 1260 strong/1380 weak/2870 medium to strong/2960 medium to strong"
    "C─H vinyl cis-disubstituted alkenes 670–700 strong"
    "C─N R─N─C (isocyanides) any 2165–2110"
    "C─O alcohols secondary ~ 1100 strong"
本模块把名字里的波数解析成【锚点】，供两处使用：
    P2/P3  band-anchor 偏置（query 只关注自己锚点窗口）
    P1     谱证据损失（窗口内必须有吸收）

17 类的名字里没有波数（就叫 "esters"），它的锚点由 72 子类的并集给出 —— 见
`BandTable.from_72_union()`，依赖 `label_map.M`（many-to-many，故取并集而非树继承）。
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
LABEL_NAMES_JSON = os.path.join(_HERE, "label_names.json")

# 有效波数搜索范围（超出即认为是分子式里的数字，如 1,1-disub. / 6-membered）
WN_MIN, WN_MAX = 400.0, 4100.0

# 区间写法：670–700(en dash) / 1440-1460(ascii hyphen)
_RANGE_RE = re.compile(r"(\d{3,4})\s*[–\-]\s*(\d{3,4})")
_BARE_RE = re.compile(r"(?<![\d.,])(\d{3,4})(?![\d.,])")

# 这些词说明该谱带宽/被掩盖 —— 谱证据损失对它们不施加（06 文档 §5）
_BROAD_MARKERS = ("broad", "two bands", "often overlapped", "associated")


@dataclass(frozen=True)
class Anchor:
    """一个特征波数窗口。center 为中心波数，half 为半宽（cm⁻¹）。"""

    center: float
    half: float

    @property
    def lo(self) -> float:
        return self.center - self.half

    @property
    def hi(self) -> float:
        return self.center + self.half

    def as_tuple(self) -> tuple[float, float]:
        return (self.lo, self.hi)


def _in_range(v: float) -> bool:
    return WN_MIN <= v <= WN_MAX


def parse_anchors(name: str, point_half: float = 15.0) -> List[Anchor]:
    """从标签名解析锚点。单点 -> half=point_half；区间 -> half=跨度一半。"""
    anchors: List[Anchor] = []
    consumed = [False] * len(name)

    for m in _RANGE_RE.finditer(name):
        a, b = float(m.group(1)), float(m.group(2))
        if not (_in_range(a) and _in_range(b)):
            continue
        lo, hi = min(a, b), max(a, b)
        anchors.append(Anchor(center=(lo + hi) / 2.0, half=(hi - lo) / 2.0))
        for i in range(m.start(), m.end()):
            consumed[i] = True

    for m in _BARE_RE.finditer(name):
        if any(consumed[m.start():m.end()]):
            continue
        v = float(m.group(1))
        if not _in_range(v):
            continue
        anchors.append(Anchor(center=v, half=point_half))

    anchors.sort(key=lambda x: x.center)
    return anchors


def is_sharp(name: str) -> bool:
    """是否带尖锐特征带。宽峰/氢键类噪声大，谱证据损失不施加。"""
    low = name.lower()
    return not any(mark in low for mark in _BROAD_MARKERS)


# 强度词 -> 期望的相对峰高（谱已逐谱 max 归一化，故这里是"相对本谱最强吸收"的比例）
# 顺序敏感：从弱到强逐级判断，命中即返回。
_INTENSITY_TABLE = (
    ("very weak", 0.005),
    ("weak to strong", 0.02),
    ("weak to medium", 0.02),
    ("weak", 0.015),
    ("medium to strong", 0.08),
    ("strong to medium", 0.08),
    ("medium", 0.04),
    ("very strong", 0.20),
    ("strong", 0.12),
)
_DEFAULT_INTENSITY = 0.04


def expected_intensity(name: str) -> float:
    """从标签名解析【期望峰高】—— 谱证据损失据此给每个类设各自的证据下限。

    为什么必须逐类：标签名自己写着 weak / very weak，弱峰又常落在大包络的尾巴上，
    "峰 − 局部基线"天然趋近 0。用统一的强阈值会**惩罚正确的预测**。
    实测（D3 test，sharp 标签）：标签为真时窗口内 e>0.05 的只有 62.7% ——
    其中 `C─H 芳香苯环 3070 weak`（n=5202）真时的平均 e 仅 0.007。
    """
    low = name.lower()
    for word, v in _INTENSITY_TABLE:
        if word in low:
            return v
    return _DEFAULT_INTENSITY


def intensity_floor(labels: Sequence[str], scale: float = 1.0) -> np.ndarray:
    """逐类的证据下限向量，直接喂给 SpectralEvidenceLoss 的 e_min。"""
    return np.asarray([expected_intensity(n) * scale for n in labels], dtype=np.float32)


@dataclass
class BandTable:
    labels: Sequence[str]
    anchors: Sequence[Sequence[Anchor]]
    sharp: Sequence[bool]

    def __len__(self) -> int:
        return len(self.labels)

    @classmethod
    def build(cls, labels: Sequence[str], point_half: float = 15.0) -> "BandTable":
        return cls(
            labels=list(labels),
            anchors=[parse_anchors(n, point_half) for n in labels],
            sharp=[is_sharp(n) for n in labels],
        )

    @classmethod
    def from_72_union(cls, labels17, labels72, m72_to_17, point_half: float = 15.0) -> "BandTable":
        """17 类锚点 = 其 72 子类窗口的并集（06 文档 §3「粗类也有物理锚」）。

        `m72_to_17` 是 many-to-many 的软映射；这里只取稀疏支持集合，做窗口并集。
        """
        t72 = cls.build(labels72, point_half)
        anchors = []
        sharp = []
        for i in range(len(labels17)):
            kids = [j for j in m72_to_17[i]]
            merged: List[Anchor] = []
            for j in kids:
                merged.extend(t72.anchors[j])
            anchors.append(_merge_overlapping(merged))
            sharp.append(any(t72.sharp[j] for j in kids) and bool(kids))
        return cls(labels=list(labels17), anchors=anchors, sharp=sharp)

    @classmethod
    def for_89(cls, classes_17, classes_72, children_of_17, point_half: float = 15.0) -> "BandTable":
        """89 类合并表：前 17 行来自 72 子类并集，后 72 行直接解析名字。

        行序必须与 89-query 解码器一致（17 粗 + 72 细）。
        `children_of_17` 形如 [[72 类索引...] × 17]，由 LabelMap.children_for_anchor 给出。
        """
        coarse = cls.from_72_union(classes_17, classes_72, children_of_17, point_half)
        fine = cls.build(classes_72, point_half)
        return cls(labels=list(classes_17) + list(classes_72),
                   anchors=list(coarse.anchors) + list(fine.anchors),
                   sharp=list(coarse.sharp) + list(fine.sharp))

    def window_mask(self, grid: np.ndarray, pad: float = 0.0) -> np.ndarray:
        """[L, N] 0/1 掩码：第 l 行的锚点窗口覆盖了 grid 的哪些点。"""
        grid = np.asarray(grid, dtype=np.float64)
        out = np.zeros((len(self.labels), grid.size), dtype=bool)
        for l, anch in enumerate(self.anchors):
            for a in anch:
                out[l] |= (grid >= a.lo - pad) & (grid <= a.hi + pad)
        return out.astype(np.float32)

    def distance_to_anchors(self, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """[L, N] 每个 grid 点到该标签最近锚点窗口的距离（cm⁻¹，窗口内为 0）。

        与 `window_mask` 的区别：掩码是二值的，距离保留了"离窗口多远"的信息，
        供 **软** band 偏置（高斯衰减）使用。返回 (距离, has_anchor[L])。
        has_anchor=False 的行（如某粗类无 72 子类、锚点为空）不施加偏置。
        """
        grid = np.asarray(grid, dtype=np.float64)
        out = np.zeros((len(self.labels), grid.size), dtype=np.float64)
        has = np.zeros(len(self.labels), dtype=bool)
        for l, anch in enumerate(self.anchors):
            if not anch:
                continue
            has[l] = True
            best = np.full(grid.size, np.inf)
            for a in anch:
                d = np.where(grid < a.lo, a.lo - grid,
                             np.where(grid > a.hi, grid - a.hi, 0.0))
                best = np.minimum(best, d)
            out[l] = best
        return out, has

    def anchor_centers_padded(self, n_anchors: int) -> np.ndarray:
        """[L, n_anchors] 锚点中心（不足补 NaN），供 band-anchor 注意力偏置用。"""
        out = np.full((len(self.labels), n_anchors), np.nan, dtype=np.float64)
        for l, anch in enumerate(self.anchors):
            for k, a in enumerate(anch[:n_anchors]):
                out[l, k] = a.center
        return out

    def to_json(self) -> dict:
        return {
            "labels": list(self.labels),
            "sharp": [bool(s) for s in self.sharp],
            "anchors": [[[a.center, a.half] for a in row] for row in self.anchors],
        }


def _merge_overlapping(anchors: Iterable[Anchor]) -> List[Anchor]:
    items = sorted(anchors, key=lambda a: a.lo)
    if not items:
        return []
    merged = [items[0]]
    for a in items[1:]:
        last = merged[-1]
        if a.lo <= last.hi:                      # 有重叠 -> 并在一个窗口里
            lo, hi = last.lo, max(last.hi, a.hi)
            merged[-1] = Anchor(center=(lo + hi) / 2.0, half=(hi - lo) / 2.0)
        else:
            merged.append(a)
    return merged


def load_label_names(kind: str = "72", path: str = LABEL_NAMES_JSON) -> list[str]:
    """kind ∈ {'17','72'}。名字顺序 = 标签索引顺序。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    key = f"classes_{kind}"
    if key not in data:
        raise KeyError(f"{path} 里没有 {key}")
    return list(data[key])