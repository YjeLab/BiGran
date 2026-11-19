"""层4 · 89 标签查询解码器 —— 17 粗 + 72 细 **同一组 query**，不是两个头。

四件套（06 文档 §3）：
  同一解码空间   17 与 72 是同一组 89 query
  跨粒度信息流   89 query 之间 self-attention（细类借粗类的稳定信号）
  粗类也有物理锚 17 类用其 72 子类波数并集（在 bigran.py 构造 QueryDecoder 时传入）
  band-anchor 偏置（P3 软）query 只关注自己的锚点窗口

长尾 L1：化学族共享基座 + 每类低秩增量。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

NEG = -1e9


@dataclass
class DecoderOutput:
    logits: torch.Tensor          # [B, 89]
    coarse: torch.Tensor          # [B, 17]
    fine: torch.Tensor            # [B, 72]
    attn_weights: torch.Tensor | None = None
    reliability: torch.Tensor | None = None   # [B, n_q] 逐样本逐标签的"这个判定靠不靠得住"
    structure: torch.Tensor | None = None     # [B, n_fine] 结构自审：我这条细类判定的粗类证据在吗
    repr: torch.Tensor | None = None          # [B, n_q, d] 解码器输出处的 query 表征（原型对比用）


class DecoderBlock(nn.Module):
    """解码器的一层：query 间 self-attn（跨粒度）→ 对谱 cross-attn（band 偏置）→ FFN。

    每层**独立权重**（不是权重共享的迭代）——这样 `decoder_layers` 才真正增加容量。
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float,
                 use_self_attn: bool = True) -> None:
        super().__init__()
        # 关掉时**不建**这个子模块（消融要的是真删，不是留一堆死权重）
        self.self_attn = (nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
                          if use_self_attn else None)
        self.cross = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model) if use_self_attn else None
        self.norm3 = nn.LayerNorm(d_model)

    def self_block(self, q: torch.Tensor) -> torch.Tensor:
        if self.self_attn is None:                                   # E6：跨粒度自注意力关
            return q
        s, _ = self.self_attn(q, q, q, need_weights=False)
        return self.norm2(q + s)

    def ff_block(self, q: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        q = self.norm1(q + c)
        return self.norm3(q + self.ff(q))


class QueryDecoder(nn.Module):
    def __init__(self, d_model: int = 128, n_coarse: int = 17, n_fine: int = 72,
                 family_of_query: list[int] | None = None, family_rank: int = 8,
                 n_layers: int = 1, nhead: int = 4, dim_feedforward: int = 256,
                 dropout: float = 0.1, use_band_bias: bool = True,
                 band_bias_mode: str = "hard", band_bias_sigma_cm: float = 40.0,
                 band_bias_floor: float = 1e-2,
                 head_mode: str = "linear", head_hidden: int = 256,
                 use_self_attn: bool = True,
                 relation: np.ndarray | None = None,
                 use_reliability: bool = False,
                 rel_detach: bool = False,
                 use_structure_aux: bool = False) -> None:
        super().__init__()
        self.n_coarse, self.n_fine = n_coarse, n_fine
        self.n_queries = n_coarse + n_fine
        self.use_band_bias = use_band_bias
        self.use_self_attn = use_self_attn
        # 可分辨性作一等输出：额外一个共享的可靠性头，输出 [B, n_q] 的"该判定靠不靠得住"。
        # 它与主头**并行**（同一份 query 表征），训练时用"这个预测对不对"当目标。
        # 关键用法是选择性预测：低可靠性的判定可以弃权；评测看风险-覆盖率，不只看 macro-F1。
        self.use_reliability = use_reliability
        self.rel_detach = rel_detach
        # 结构自审（输出侧）：细类 query 额外预测"按标签结构，我这条判定的粗类证据在不在"。
        # 与 R1（把结构当**输入**喂进去，净亏）相反，这是**输出侧辅助监督**——
        # 与 R3 同一个思路（R3 已证明输出侧辅助监督有效），只是监督量换成结构一致性。
        self.use_structure_aux = use_structure_aux
        self.band_bias_mode = band_bias_mode
        self.band_bias_sigma_cm = float(band_bias_sigma_cm)
        self.band_bias_floor = float(band_bias_floor)
        if band_bias_mode not in ("hard", "soft"):
            raise ValueError(f"未知 band_bias_mode: {band_bias_mode!r}（hard | soft）")

        if family_of_query is None:
            family_of_query = list(range(self.n_queries))
        self.register_buffer("family_of_query",
                             torch.as_tensor(family_of_query, dtype=torch.long), persistent=False)
        n_families = int(max(family_of_query)) + 1

        # 化学族共享基座 + 每类低秩增量
        self.family_base = nn.Embedding(n_families, d_model)
        self.delta_u = nn.Parameter(torch.zeros(self.n_queries, family_rank))
        self.delta_v = nn.Parameter(torch.zeros(family_rank, d_model))
        nn.init.normal_(self.delta_u, std=0.02)
        nn.init.normal_(self.delta_v, std=0.02 / max(1, family_rank) ** 0.5)
        self.q_norm = nn.LayerNorm(d_model)

        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, nhead, dim_feedforward, dropout, use_self_attn)
            for _ in range(n_layers)
        ])
        self.n_layers = n_layers
        self.head = build_head(head_mode, d_model, self.n_queries, head_hidden, dropout)
        self.reliability_head = nn.Linear(d_model, 1) if use_reliability else None
        self.structure_head = nn.Linear(d_model, 1) if use_structure_aux else None

        # 17↔72 的结构关系（many-to-many）做进解码器：细类 query 额外读一次"它的粗类邻居"的表征。
        # 关系矩阵 R[fine, coarse] 来自**标签定义**（children_for_anchor），与数据无关。
        # 去掉它（R 全 0 / 不打这条通路）细类就少一路输入 —— 所以消融是**承重**的，不是加性装饰。
        self.relational_cross = relation is not None
        if self.relational_cross:
            R = np.asarray(relation, dtype=np.float32)
            if R.shape != (n_fine, n_coarse):
                raise ValueError(f"关系矩阵应为 [{n_fine},{n_coarse}]，得到 {R.shape}")
            R = R / np.maximum(R.sum(axis=1, keepdims=True), 1.0)      # 行归一化
            self.register_buffer("rel_norm", torch.as_tensor(R), persistent=False)
            self.rel_proj = nn.Linear(d_model, d_model)

    def query_embeddings(self) -> torch.Tensor:
        base = self.family_base(self.family_of_query)
        return self.q_norm(base + self.delta_u @ self.delta_v)

    def build_band_bias(self, wn: np.ndarray, anchor_mask: torch.Tensor | None,
                        anchor_dist: torch.Tensor | None = None,
                        anchor_has: torch.Tensor | None = None) -> torch.Tensor | None:
        """[89, T] 加性偏置：query 只关注自己锚点窗口附近的 token。

        hard 模式：窗口外 -inf（等价硬掩码），需要 `anchor_mask` [89,T] 0/1。
        soft 模式：以到最近锚点窗口的距离做高斯衰减，乘性门 = floor + (1-floor)·exp(-d²/2σ²)，
                  加性偏置取其对数。`floor` 保留全局上下文通道（≠0 时才不是硬掩码）。
        两种模式下，"无锚点"的 query（例如该粗类没有 72 子类）都**不施加偏置**，否则整行 -inf 出 NaN。
        """
        if not self.use_band_bias:
            return None
        if self.band_bias_mode == "hard":
            if anchor_mask is None:
                return None
            m = anchor_mask.to(dtype=torch.bool)                     # [89, T]
            has_anchor = m.any(dim=1, keepdim=True)                  # [89, 1]
            m = torch.where(has_anchor, m, torch.ones_like(m))       # 无锚点的 query 放开
            return torch.where(m, torch.zeros((), dtype=torch.float32),
                               torch.full((), NEG, dtype=torch.float32))
        # soft
        if anchor_dist is None:
            return None
        d = anchor_dist.to(dtype=torch.float32)                      # [89, T] cm⁻¹
        gate = self.band_bias_floor + (1.0 - self.band_bias_floor) * torch.exp(
            -0.5 * (d / self.band_bias_sigma_cm) ** 2)
        bias = torch.log(gate.clamp_min(1e-12))
        if anchor_has is not None:
            has = anchor_has.to(dtype=torch.bool).unsqueeze(1)       # [89,1]
            bias = torch.where(has, bias, torch.zeros_like(bias))
        return bias

    def _apply_relation(self, q: torch.Tensor) -> torch.Tensor:
        """细类 query 加一路"粗类邻居"的上下文：h_f += W · (R_norm @ h_c)。

        R_norm[f, c] 由标签定义给出（细类 f 是否挂在粗类 c 下，行归一化）。
        没有粗类父节点的细类（16 个）该行为 0，不受影响。
        """
        hc = q[:, : self.n_coarse]                                # [B, 17, d]
        hf = q[:, self.n_coarse:]                                 # [B, 72, d]
        ctx = torch.einsum("fc,bcd->bfd", self.rel_norm.to(q.dtype), hc)
        return torch.cat([hc, hf + self.rel_proj(ctx)], dim=1)

    def forward(self, memory: torch.Tensor, wn: np.ndarray,
                anchor_mask: torch.Tensor | None = None,
                anchor_dist: torch.Tensor | None = None,
                anchor_has: torch.Tensor | None = None,
                return_attn: bool = False) -> DecoderOutput:
        """memory: [B, T, d]（层2 输出）；wn: [T] 波数网格。"""
        q = self.query_embeddings().unsqueeze(0).expand(memory.shape[0], -1, -1)   # [B,89,d]
        attn_cross = None
        bias = self.build_band_bias(wn, anchor_mask, anchor_dist, anchor_has)
        if bias is not None:
            bias = bias.to(memory.device)

        for blk in self.blocks:
            # 跨粒度信息流：89 query 之间互相看
            q = blk.self_block(q)
            # 看谱（band 偏置）
            c, w = blk.cross(q, memory, memory, attn_mask=bias, need_weights=return_attn,
                             average_attn_weights=False)
            if return_attn:
                attn_cross = w
            q = blk.ff_block(q, c)

        if self.relational_cross:
            q = self._apply_relation(q)

        logits = self.head(q).squeeze(-1)                        # [B, 89]
        rel = (self.reliability_head(q.detach() if self.rel_detach else q).squeeze(-1)
               if self.use_reliability else None)
        struct = (self.structure_head(q[:, self.n_coarse:]).squeeze(-1)
                  if self.use_structure_aux and self.n_fine else None)
        return DecoderOutput(
            logits=logits,
            coarse=logits[:, : self.n_coarse],
            fine=logits[:, self.n_coarse:],
            attn_weights=attn_cross,
            reliability=rel,
            structure=struct,
            repr=q,
        )


class PerQueryMLPHead(nn.Module):
    """每类独立的两层 MLP 决策头 —— 给 89 个标签各自的决策边界。

    动机：原实现是 89 类**共用**一个 Linear(d,1)（仅 192 个参数），标签间唯一的差异
    来自 query 嵌入。粗类边界少无所谓；72 个细类边界多，分类容量明显不足。
    这里每个 query 有自己的 (d->h->1)，容量随标签数线性增长。
    """

    def __init__(self, d_model: int, n_queries: int, hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(n_queries, hidden, d_model))
        self.b1 = nn.Parameter(torch.zeros(n_queries, hidden))
        self.w2 = nn.Parameter(torch.empty(n_queries, 1, hidden))
        self.b2 = nn.Parameter(torch.zeros(n_queries, 1))
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor) -> torch.Tensor:          # [B, Q, d] -> [B, Q, 1]
        h = self.act(torch.einsum("bqd,qhd->bqh", q, self.w1) + self.b1)
        return torch.einsum("bqh,qoh->bqo", self.drop(h), self.w2) + self.b2


def build_head(mode: str, d_model: int, n_queries: int, hidden: int, dropout: float) -> nn.Module:
    """分类头。linear = 现状（共用单层）；mlp = 共用两层；per_query_mlp = 每类独立两层。"""
    if mode == "linear":
        return nn.Linear(d_model, 1)
    if mode == "mlp":
        return nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout),
                             nn.Linear(hidden, 1))
    if mode == "per_query_mlp":
        return PerQueryMLPHead(d_model, n_queries, hidden, dropout)
    raise ValueError(f"未知 head_mode: {mode!r}（linear | mlp | per_query_mlp）")