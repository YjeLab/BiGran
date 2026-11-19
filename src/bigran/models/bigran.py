"""BiGran-IR 组装 —— 层1 CNN → 层2 Transformer →（层3 物理量）→ 层4 89-query 解码器。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from bigran.constants import N_COARSE, N_FINE, N_POINTS, WN_START, WN_STEP, wavenumber_grid
from bigran.models.decoder import NEG, QueryDecoder
from bigran.models.encoder import SpectrumEncoder
from bigran.models.physics import window_mask_for_grid
from bigran.models.stem import MultiScaleStem, PhysicsKernelStem
from bigran.physics.band_table import BandTable, intensity_floor
from bigran.physics.families import build_families, families_for
from bigran.utils.registry import MODELS


@dataclass
class BiGranConfig:
    # 层1
    in_channels: int = 2
    stem_mode: str = "physics"              # physics | dilation | single   （消融 #1）
    widths_cm: tuple = (122.0, 50.0, 18.0)
    branch_channels: int = 64
    stem_stride: int = 2
    # 层2（stem 输出通道恒等于 d_model —— 不再单独设，避免两个必须一致的旋钮）
    d_model: int = 128
    n_encoder_layers: int = 3
    nhead: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.1
    encoder_downsample: int = 2
    encoder_type: str = "transformer"       # transformer | ssm（对角 SSM / S4D 家族）
    encoder_d_state: int = 16               # 仅 ssm 用
    pe_scale: float = 1000.0                # 绝对波数正弦编码的尺度（小 => 相位展开更充分）
    pe_mode: str = "absolute"               # absolute=波数正弦（默认）| learnable=可学位置表（R1-5a 消融）
    # 层3
    band_bias: bool = True                  # P3（消融 #2）
    band_bias_mode: str = "hard"            # hard = 窗口外硬掩码；soft = 高斯衰减 + 全局地板
    band_bias_sigma_cm: float = 40.0        # soft 模式的高斯宽度（cm⁻¹）
    band_bias_floor: float = 1e-2           # soft 模式的全局地板（>0 才保留上下文通道）
    anchor_pad: float = 0.0
    evidence_tau: float = 0.05
    # 层4
    family_rank: int = 8
    decoder_layers: int = 1
    use_families: bool = True               # L1（消融 #5）
    head_mode: str = "linear"               # linear | mlp | per_query_mlp
    head_hidden: int = 256
    # 任务粒度：both = 89 联合（默认，我们的主张）；coarse/fine = 单粒度独立训
    # （用于量化"一次训练同时出两个粒度"相对于"各训一次"的代价）；
    # two_heads = **同一个骨干 + 两个独立解码头**（17 一个、72 一个，各自一套 band 锚点，
    #   彼此无共享 query、无跨粒度 self-attn）——E5 对照，用来回答"这不就是多任务两个头吗"。
    granularity: str = "both"               # both | coarse | fine | two_heads
    # 解码器跨粒度 self-attention 开关（E6）。false 时 Q 之间不再互相看，
    # 只保留 band-anchor 的 cross-attention —— 用来检验"细类借粗类信号"是不是真起作用的。
    self_attn: bool = True
    # 17↔72 的结构关系做进解码器：细类 query 额外读一路"它的粗类父节点"表征。
    # 关系来自标签定义（children_for_anchor），去掉这条通路细类就少一路输入（承重，非装饰）。
    relational_cross: bool = False
    relational_shuffle: bool = False     # 等容量对照：同样的通路与参数，关系被打乱
    relational_seed: int = 0
    # 可分辨性作一等输出：额外一个共享可靠性头，输出逐样本逐标签的"该判定靠不靠得住"。
    # 用途是**选择性预测**（低可靠性可弃权），评测看风险-覆盖率；macro-F1 预期不动。
    use_reliability: bool = False
    # 可靠性头梯度是否回传共享表示。false（现有实现）：L_rel 经 q 正则化编码器/解码器；
    # true：头只在自己参数上学（审稿 R1-1f 的对照变体）。
    rel_detach: bool = False
    # 结构自审（输出侧）：细类 query 额外预测"按标签结构，我这条判定的粗类证据在不在"。
    # 与 R1（结构当输入，净亏）相反；与 R3（输出侧辅助监督）同一思路。
    use_structure_aux: bool = False
    structure_aux_shuffle: bool = False   # 等容量对照：同样的头，结构关系被打乱
    # R5 · 解耦重校准（cRT 式）：冻结骨干（stem+encoder），只重训解码器/查询库。
    # 依据：E3 的线性探针证明联合骨干已经"对两个粒度都好"（表示练好了），
    # 而 Kang 2020 / BS-cRT 2026 表明长尾的瓶颈在**分类器**——所以把两者解耦。
    freeze_backbone: bool = False
    init_from: str = ""                   # 非空时从该 run 目录的 best.pt 初始化（二次阶段用）


def build_band_table_89(classes_17: list[str], classes_72: list[str], label_map) -> BandTable:
    """89 类锚点表：17 粗（72 子类并集）+ 72 细（名字解析）。行序 = query 顺序。"""
    children = [label_map.children_for_anchor(i) for i in range(N_COARSE)]
    return BandTable.for_89(classes_17, classes_72, children)


def bipartite_children_relation(label_map, n_coarse: int = N_COARSE, n_fine: int = N_FINE,
                                shuffle: bool = False, seed: int = 0) -> np.ndarray:
    """[n_fine, n_coarse] 二部关联：细类 j 是否挂在粗类 i 下（来自标签定义，与数据无关）。

    many-to-many：20 个细类挂在多个粗类下、16 个细类无父节点（该行全 0）。
    `shuffle=True` 是**等容量对照**：保持每行的度数不变，把连边随机重排 ——
    参数与通路完全相同，只有"结构对不对"这一件事被破坏。
    """
    R = np.zeros((n_fine, n_coarse), dtype=np.float32)
    for i in range(n_coarse):
        for j in label_map.children_for_anchor(i):
            if 0 <= int(j) < n_fine:
                R[int(j), i] = 1.0
    if not shuffle:
        return R
    rng = np.random.default_rng(seed)
    Rs = np.zeros_like(R)
    for j in range(n_fine):
        k = int(R[j].sum())
        if k:
            Rs[j, rng.choice(n_coarse, size=min(k, n_coarse), replace=False)] = 1.0
    return Rs


class BiGranIR(nn.Module):
    def __init__(self, cfg: BiGranConfig, classes_17: list[str], classes_72: list[str],
                 label_map, band_table: BandTable | None = None,
                 wn_step: float = WN_STEP, wn_start: float = WN_START,
                 n_points: int = N_POINTS) -> None:
        super().__init__()
        self.cfg = cfg
        self.granularity = cfg.granularity
        self.two_heads = self.granularity == "two_heads"
        if self.granularity in ("both", "two_heads"):
            self.n_coarse, self.n_fine = N_COARSE, N_FINE
            band_table = band_table or build_band_table_89(classes_17, classes_72, label_map)
        elif self.granularity == "coarse":
            self.n_coarse, self.n_fine = N_COARSE, 0
            children = [label_map.children_for_anchor(i) for i in range(N_COARSE)]
            band_table = band_table or BandTable.from_72_union(classes_17, classes_72, children)
        elif self.granularity == "fine":
            self.n_coarse, self.n_fine = 0, N_FINE
            band_table = band_table or BandTable.build(classes_72)
        else:
            raise ValueError(f"未知 granularity: {self.granularity!r}"
                             f"（应为 both/coarse/fine/two_heads）")
        self.wn_in = wavenumber_grid(n_points, wn_start, wn_step)
        n_q = self.n_coarse + self.n_fine
        if len(band_table) != n_q:
            raise ValueError(f"band_table 应有 {n_q} 行，得到 {len(band_table)}")

        # 层1
        if cfg.stem_mode == "physics":
            self.stem = PhysicsKernelStem(cfg.in_channels, cfg.widths_cm, wn_step,
                                          cfg.branch_channels, cfg.d_model, cfg.stem_stride)
        else:
            self.stem = MultiScaleStem(cfg.in_channels, cfg.stem_mode, cfg.branch_channels,
                                       cfg.d_model, cfg.stem_stride)
        # 层2
        if cfg.encoder_type == "ssm":
            from bigran.models.ssm import SSMEncoder
            self.encoder = SSMEncoder(cfg.d_model, cfg.n_encoder_layers, cfg.nhead,
                                      cfg.dim_feedforward, cfg.dropout, cfg.encoder_downsample,
                                      cfg.encoder_d_state)
        else:
            self.encoder = SpectrumEncoder(cfg.d_model, cfg.n_encoder_layers, cfg.nhead,
                                           cfg.dim_feedforward, cfg.dropout, cfg.encoder_downsample,
                                           pe_scale=cfg.pe_scale, pe_mode=cfg.pe_mode)

        # 波数网格逐级下采样（层1 stride -> 层2 downsample）
        wn_mid = self.wn_in[:: cfg.stem_stride][: n_points // cfg.stem_stride]
        self.wn_mid = wn_mid
        self.wn_enc = wn_mid[:: cfg.encoder_downsample][: len(wn_mid) // cfg.encoder_downsample]

        # 层3：两套锚点掩码 —— 输入分辨率（谱证据）与编码器分辨率（band 偏置）
        self.register_buffer("mask_in", window_mask_for_grid(band_table, self.wn_in, cfg.anchor_pad),
                             persistent=False)
        self.register_buffer("mask_enc", window_mask_for_grid(band_table, self.wn_enc, cfg.anchor_pad),
                             persistent=False)
        self.register_buffer("sharp", torch.as_tensor(band_table.sharp, dtype=torch.bool),
                             persistent=False)
        # 软 band 偏置用：每个 query 到最近锚点窗口的距离（cm⁻¹）与"有无锚点"
        dist, has = band_table.distance_to_anchors(self.wn_enc)
        self.register_buffer("dist_enc", torch.as_tensor(dist, dtype=torch.float32),
                             persistent=False)
        self.register_buffer("anchor_has", torch.as_tensor(has, dtype=torch.bool),
                             persistent=False)
        # 逐类的证据下限：由标签名里自带的 weak/strong 决定（见 band_table.expected_intensity）
        self.register_buffer("e_min_default",
                             torch.as_tensor(intensity_floor(band_table.labels), dtype=torch.float32),
                             persistent=False)

        # 层4
        if self.two_heads:
            # 两个**完全独立**的解码头：各自一套 band 锚点（89 行表的前 17 行 / 后 72 行），
            # 各自查各自的 query，彼此不共享参数、不做跨粒度 self-attn。
            fam_c = families_for("coarse", classes_17, classes_72)
            fam_f = families_for("fine", classes_17, classes_72)
            self.family_names = fam_c["names"]
            of_c = fam_c["of_q"] if cfg.use_families else list(range(N_COARSE))
            of_f = fam_f["of_q"] if cfg.use_families else list(range(N_FINE))
            self.decoder = None
            self.decoder_coarse = self._make_decoder(N_COARSE, 0, of_c)
            self.decoder_fine = self._make_decoder(0, N_FINE, of_f)
        else:
            fam = (build_families(classes_17, classes_72) if self.granularity == "both"
                   else families_for(self.granularity, classes_17, classes_72))
            self.family_names = fam["names"]
            of_q = fam["of_89" if self.granularity == "both" else "of_q"] if cfg.use_families \
                else list(range(n_q))
            self.decoder_coarse = self.decoder_fine = None
            rel = None
            if cfg.relational_cross and self.granularity == "both":
                rel = bipartite_children_relation(label_map, self.n_coarse, self.n_fine,
                                                  cfg.relational_shuffle, cfg.relational_seed)
            self.decoder = self._make_decoder(self.n_coarse, self.n_fine, of_q, rel)

        if cfg.freeze_backbone:
            # 解耦重校准：骨干只前向不更新（分类器/查询库仍在学）
            for mod in (self.stem, self.encoder):
                for p in mod.parameters():
                    p.requires_grad_(False)

    def _make_decoder(self, n_coarse: int, n_fine: int, family_of_query,
                      relation: np.ndarray | None = None) -> QueryDecoder:
        c = self.cfg
        return QueryDecoder(c.d_model, n_coarse, n_fine, family_of_query, c.family_rank,
                            c.decoder_layers, c.nhead, c.dim_feedforward, c.dropout, c.band_bias,
                            band_bias_mode=c.band_bias_mode,
                            band_bias_sigma_cm=c.band_bias_sigma_cm,
                            band_bias_floor=c.band_bias_floor,
                            head_mode=c.head_mode, head_hidden=c.head_hidden,
                            use_self_attn=c.self_attn, relation=relation,
                            use_reliability=c.use_reliability,
                            rel_detach=c.rel_detach,
                            use_structure_aux=c.use_structure_aux)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.stem(x)                                 # [B, d, T_mid]
        return self.encoder(z.transpose(1, 2), self.wn_mid)

    def forward(self, x: torch.Tensor, return_evidence: bool = False, return_attn: bool = False) -> dict:
        memory = self.encode(x)
        if self.two_heads:
            out_c = self.decoder_coarse(memory, self.wn_enc, self.mask_enc[:N_COARSE],
                                        self.dist_enc[:N_COARSE], self.anchor_has[:N_COARSE],
                                        return_attn=return_attn)
            out_f = self.decoder_fine(memory, self.wn_enc, self.mask_enc[N_COARSE:],
                                      self.dist_enc[N_COARSE:], self.anchor_has[N_COARSE:],
                                      return_attn=return_attn)
            logits, coarse, fine = torch.cat([out_c.logits, out_f.logits], dim=1), \
                out_c.logits, out_f.logits
            attn = out_f.attn_weights
            rel = (torch.cat([out_c.reliability, out_f.reliability], dim=1)
                   if out_c.reliability is not None else None)
            struct = None
            qrepr = torch.cat([out_c.repr, out_f.repr], dim=1)
        else:
            out = self.decoder(memory, self.wn_enc, self.mask_enc, self.dist_enc, self.anchor_has,
                               return_attn=return_attn)
            logits, coarse, fine = out.logits, out.coarse, out.fine
            attn, rel, struct = out.attn_weights, out.reliability, out.structure
            qrepr = out.repr
        res = {
            "logits": logits,
            "coarse": coarse,
            "fine": fine,
            "anc_mask_in": self.mask_in,
            "query_repr": qrepr,
            # 谱证据损失要用它；只是一次切片，恒定返回，免得开了 λ₂ 却拿不到 key
            "absorbance": x[:, 0, :],
        }
        if rel is not None:
            res["reliability"] = rel
        if struct is not None:
            res["structure"] = struct
        if return_attn:
            res["attn"] = attn
        return res

    @property
    def n_logits(self) -> int:
        return self.n_coarse + self.n_fine


@MODELS.register("bigran")
def build_bigran(cfg: BiGranConfig | None = None, classes_17=None, classes_72=None,
                 label_map=None, band_table=None, **kw) -> BiGranIR:
    cfg = cfg or BiGranConfig(**kw)
    if label_map is None:
        from bigran.physics.label_map import load_label_map
        label_map = load_label_map()
    if classes_17 is None or classes_72 is None:
        from bigran.physics.band_table import load_label_names
        classes_17 = classes_17 or load_label_names("17")
        classes_72 = classes_72 or load_label_names("72")
    return BiGranIR(cfg, classes_17, classes_72, label_map, band_table)