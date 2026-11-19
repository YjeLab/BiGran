"""M(72→17) —— 细粒度标签到粗粒度官能团的**经验**映射。

⚠️ 这个映射是 **many-to-many**，不是树、不是层次（实测：51% 单父 / 26% 多父 / 22% 无父）
⇒ 只能用作**软 OR**，绝不能当层级继承。

定义：M[i, j] = P(17_i = 1 | 72_j = 1)，按分子去重后统计。
`support[i, j] = M[i, j] >= thr`（默认 0.99）给出"硬支持"集合，用于 17 类锚点并集。

产物 `label_map_d3.json` 随仓库分发（一次性离线算好）。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from bigran.physics.band_table import load_label_names

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(_HERE, "label_map_d3.json")
DEFAULT_THR = 0.99


@dataclass
class LabelMap:
    """prob: [17, 72] 条件概率；support: [17, 72] 布尔硬支持；thr: 生成时的阈值。"""

    prob: np.ndarray
    thr: float = DEFAULT_THR

    def __post_init__(self) -> None:
        self.prob = np.asarray(self.prob, dtype=np.float64)
        if self.prob.shape != (17, 72):
            raise ValueError(f"prob 形状应为 (17, 72)，得到 {self.prob.shape}")

    @property
    def support(self) -> np.ndarray:
        return self.prob >= self.thr

    def parents_of_72(self, j: int) -> list[int]:
        return [i for i in range(17) if self.support[i, j]]

    def children_of_17(self, i: int) -> list[int]:
        return [j for j in range(72) if self.support[i, j]]

    def children_at(self, i: int, thr: float) -> list[int]:
        return [j for j in range(72) if self.prob[i, j] >= thr]

    def children_for_anchor(self, i: int, fallback_thr: float = 0.90) -> list[int]:
        """17 类锚点并集要用的子类集合。

        硬支持（thr=0.99）为空时回退到 `fallback_thr`。实测只有 `ketones` 会触发：
        它的最佳子类是 `C=O aldehyde/ketone α,β-unsaturated 1685`（P=0.95）—— 但这
        个 72 类名里同时写着 aldehyde，所以 P(ketones|·) 到不了 0.99。
        回退阈值取 0.90 而非 0.50：0.50 会把 1600(C═C) 之类噪声类拉进来，
        且对 alkane/aromatics 这类"泛类"会一次吃进 50+ 个子类，窗口失去物理意义。
        """
        hard = self.children_of_17(i)
        return hard if hard else self.children_at(i, fallback_thr)

    def soft_or(self, p72: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """noisy-OR：由 72 类概率推出 17 类概率。p72: [..., 72] -> [..., 17]。

        用条件概率当"致病强度"：p17_i = 1 - Π_j (1 - p72_j · M_ij)。
        与硬 OR 的区别：子类概率低时不会因为"存在这个子类"就强行置 1。
        """
        p72 = np.asarray(p72, dtype=np.float64)
        m = np.clip(self.prob, 0.0, 1.0)               # [17,72]
        contrib = 1.0 - p72[..., None, :] * m          # [...,17,72]
        return 1.0 - np.clip(contrib, eps, 1.0).prod(axis=-1)

    def to_json(self) -> dict:
        return {
            "_note": "M[i,j] = P(17_i=1 | 72_j=1)，D3 分子级去重统计；many-to-many，只可软 OR",
            "thr": self.thr,
            "classes_17": load_label_names("17"),
            "classes_72": load_label_names("72"),
            "prob": self.prob.tolist(),
        }

    @classmethod
    def from_json(cls, path: str = DEFAULT_PATH) -> "LabelMap":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(prob=np.asarray(d["prob"], dtype=np.float64), thr=float(d.get("thr", DEFAULT_THR)))

    def save(self, path: str = DEFAULT_PATH) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, ensure_ascii=False, indent=1)

    def summary(self) -> str:
        single = multi = orphan = 0
        for j in range(72):
            k = len(self.parents_of_72(j))
            if k == 0:
                orphan += 1
            elif k == 1:
                single += 1
            else:
                multi += 1
        no_kid = sum(1 for i in range(17) if not self.children_of_17(i))
        return (f"M(72→17)@{self.thr}: 单父 {single} / 多父 {multi} / 无父 {orphan}；"
                f"17 类中无 72 子类的有 {no_kid} 个")


def load_label_map(path: str = DEFAULT_PATH) -> LabelMap:
    return LabelMap.from_json(path)