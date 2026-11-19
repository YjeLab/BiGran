"""化学族划分 —— 长尾建模 L1「化学族共享 query 基座 + 每类低秩增量」的底座。

族的来源**不是人为编的**，而是标签名里本来就有的成键描述符：
    72 类名以 "C─H / C═C / C=O / O─H / N─H / C─O / C─N / C─X / N─O / P─C / P─O" 开头
17 类名里没有这个前缀，用下面这张显式表映射到同一套族。

族共享的意义（06 文档 L1）：让支撑稀少的类借同族常见类的统计强度，
例如 C─X 族里 iodoalkanes 只有 37 个、chloroalkanes 有 471、bromoalkanes 有 289。
"""
from __future__ import annotations

from typing import Sequence

# 72 类的族 = 名字的第一个 token（成键描述符）
FINE_PREFIXES = ("C─H", "C═C", "C=O", "O─H", "N─H", "C─O", "C─N", "C─X", "N─O", "P─C", "P─O")
# 名字里出现但不在上表的（C≡C 被写成 "C═C C≡C ..."）需要单独判
_ALIASES = {"C─N C≡N": "C≡N"}

# 17 类 -> 族（显式、可审阅；同族即"共享基座"的那一组）
COARSE_FAMILY = {
    "alkane": "C─H", "methyl": "C─H",
    "alkene": "C═C", "aromatics": "C═C",
    "alkyne": "C≡C", "nitriles": "C≡N",
    "alcohols": "O─H", "amines": "N─H",
    "alkyl halides": "C─X",
    "esters": "C=O", "ketones": "C=O", "aldehydes": "C=O",
    "carboxylic acids": "C=O", "amides": "C=O", "acyl halides": "C=O",
    "ether": "C─O",
    "nitro": "N─O",
}


def family_of_fine(name: str) -> str:
    """从 72 类名推断族。名字形如 'C─H alkyl methyl ...' / 'C─N C≡N (nitriles) ...'。"""
    for alias, fam in _ALIASES.items():
        if name.startswith(alias):
            return fam
    for p in FINE_PREFIXES:
        if name.startswith(p):
            return p
    return "other"


def family_of_coarse(name: str) -> str:
    if name not in COARSE_FAMILY:
        raise KeyError(f"17 类 {name!r} 不在族表里，请补 COARSE_FAMILY")
    return COARSE_FAMILY[name]


def build_families(classes_17: Sequence[str], classes_72: Sequence[str]) -> dict:
    """返回 {'names': [族名...], 'index': {族名: id}, 'of_89': [89 个标签各自的族 id]}。

    顺序与 89-query 解码器一致：前 17 为粗类，后 72 为细类。
    """
    coarse = [family_of_coarse(n) for n in classes_17]
    fine = [family_of_fine(n) for n in classes_72]
    names = sorted(set(coarse) | set(fine))
    index = {n: i for i, n in enumerate(names)}
    of_89 = [index[f] for f in coarse] + [index[f] for f in fine]
    return {"names": names, "index": index, "of_89": of_89}


def families_for(granularity: str, classes_17: Sequence[str],
                 classes_72: Sequence[str]) -> dict:
    """单粒度（17-only / 72-only）的族划分 —— 与 `build_families` 同一套族名。

    联合模型的 89 列里，粗类与细类共用一组族 id；单粒度模型只需要自己那一半，
    但族**定义**完全一样（否则"族共享"在两种设置下不可比）。
    """
    if granularity == "coarse":
        groups = [family_of_coarse(n) for n in classes_17]
    elif granularity == "fine":
        groups = [family_of_fine(n) for n in classes_72]
    else:
        raise ValueError(f"单粒度族划分只支持 coarse/fine，得到 {granularity!r}")
    names = sorted(set(groups))
    index = {n: i for i, n in enumerate(names)}
    return {"names": names, "index": index, "of_q": [index[g] for g in groups]}


def family_sizes(of_89: Sequence[int], n_families: int) -> list[int]:
    sizes = [0] * n_families
    for f in of_89:
        sizes[f] += 1
    return sizes