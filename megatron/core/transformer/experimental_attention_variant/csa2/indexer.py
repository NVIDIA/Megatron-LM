# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DeepSeek-V4.1 indexer: learned top-k over compressed positions with shared keys.

Two kinds of layers own an indexer:

* a KV-source (``full``) layer additionally owns the key path ``k = k_norm(wk(latent))``
  and publishes the rotated keys for every later indexer that shares its compressed KV;
* a ``reindex`` layer owns only the query path (``wq_b`` and ``weights_proj``) and scores
  the shared keys.

The candidate-source layer turns its scores into a block mask that later indexers apply
before their own top-k (two-level selection). Semantics follow the official ``Indexer`` /
``select_candidate_blocks`` in ``inference/model.py``; independent implementation.
"""

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import torch

from megatron.core.fp8_utils import get_fp8_disabled_context
from megatron.core.transformer.experimental_attention_variant.csa2.candidate_kernels import (
    candidate_kernel_supported,
    candidate_scores,
    candidate_scores_available,
    candidate_topk_ids,
)
from megatron.core.transformer.experimental_attention_variant.csa2.reference import (
    candidate_blocks_to_mask,
    indexer_scores,
    indexer_topk_indices,
    select_candidate_block_ids,
)
from megatron.core.transformer.experimental_attention_variant.csa2.roles import CSA2LayerPlan
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig

# Applies RoPE to the trailing ``qk_pos_emb_head_dim`` features of ``[n, b, h, d]`` for
# positions ``0, stride, 2*stride, ...``.
RopeFn = Callable[[torch.Tensor, int], torch.Tensor]

# Budget for the per-chunk fp32 score matrix of the packed indexer (scores, masked copy and
# top-k workspace, about three [rows, n_comp] buffers). 1 GiB keeps a 128K-key layer at
# ~680 query rows per chunk.
INDEXER_SCORE_BUDGET_BYTES = 1 << 30


def _rows_within_budget(n_keys: int, budget_bytes: int = INDEXER_SCORE_BUDGET_BYTES) -> int:
    """Query rows whose dense fp32 scores (three buffers) fit the budget; at least one."""
    return max(1, budget_bytes // (3 * 4 * max(1, n_keys)))


_FUSED_INDEXER_SCORE = None  # resolved lazily: the merged cuDNN DSA dense indexer wrapper


def _fused_indexer_score_fn():
    """The merged ``_compute_dense_indexer_score`` (cuDNN DSA) or ``None`` when unavailable."""
    global _FUSED_INDEXER_SCORE
    if _FUSED_INDEXER_SCORE is None:
        try:
            from megatron.core.transformer.experimental_attention_variant.csa_utils import (
                fused_sparse_attention as _fsa,
            )

            _fsa._ensure_dsa_namespace()
            _FUSED_INDEXER_SCORE = _fsa._compute_dense_indexer_score
        except Exception:  # noqa: BLE001 - missing cudnn-frontend DSA namespace
            _FUSED_INDEXER_SCORE = False
    return _FUSED_INDEXER_SCORE or None


def fused_indexer_scores_rows(
    q_rows: torch.Tensor,
    keys: torch.Tensor,
    head_weights: torch.Tensor,
    head_weight_scale: float,
    ratio: int,
    first_position: int,
) -> Optional[torch.Tensor]:
    """``sum_h relu(q_h . k) * w_h`` for ``rows`` consecutive query positions through the merged
    cuDNN DSA dense indexer kernel (one THD segment, all keys).

    Args:
        q_rows: ``[rows, h_i, d_i]`` rotated indexer queries (bf16).
        keys: ``[n, d_i]`` rotated index keys of the segment (bf16).
        head_weights: ``[rows, h_i]`` raw head weights (unscaled; the scale goes to the kernel,
            which applies it in fp32 before the ReLU -- equivalent for a positive scale).
        head_weight_scale: ``(d_i * h_i) ** -0.5``.
        ratio: compress ratio (causal limit ``(position + 1) // ratio`` inside the kernel).
        first_position: sequence position of ``rows[0]``; rows must be consecutive positions.

    Returns:
        ``[rows, n]`` fp32 scores (positions beyond the causal limit are not meaningful and
        must be masked by the caller), or ``None`` when the kernel is unavailable.
    """
    fn = _fused_indexer_score_fn()
    if fn is None or not q_rows.is_cuda:
        return None
    rows, n_keys = q_rows.size(0), keys.size(0)
    device = q_rows.device
    cu_q = torch.tensor([0, rows], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, n_keys], dtype=torch.int32, device=device)
    offsets = torch.tensor([first_position], dtype=torch.int32, device=device)
    scores, _ = fn(
        q_rows.contiguous(),
        keys.unsqueeze(1).contiguous(),
        head_weights.to(q_rows.dtype).contiguous(),
        qhead_per_kv_head=q_rows.size(1),
        indexer_softmax_scale=float(head_weight_scale),
        ratio=int(ratio),
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_k,
        max_seqlen_q=rows,
        max_seqlen_kv=n_keys,
        q_causal_offsets=offsets,
    )
    return scores[:, :n_keys].float()


def indexer_scores_rows(
    q_rows: torch.Tensor, keys_t: torch.Tensor, head_weights: torch.Tensor
) -> torch.Tensor:
    """``sum_h relu(q_h . k) * w_h`` for flat rows, reducing the heads one at a time.

    Args:
        q_rows: ``[rows, h_i, d_i]`` indexer queries (RoPE applied).
        keys_t: ``[d_i, n]`` transposed fp32 keys.
        head_weights: ``[rows, h_i]`` mixing weights (already scaled).

    Returns:
        ``[rows, n]`` float32. Peak temporaries are two ``[rows, n]`` buffers instead of the
        ``[rows, h_i, n]`` tensor of :func:`indexer_scores`.
    """
    q = q_rows.float()
    w = head_weights.float()
    total = None
    for h in range(q.size(1)):
        contribution = torch.relu(q[:, h] @ keys_t) * w[:, h : h + 1]
        total = contribution if total is None else total.add_(contribution)
    return total


@dataclass
class CSA2IndexerSubmodules:
    """Submodule specs for :class:`CSA2Indexer`."""

    linear_wq_b: Union[ModuleSpec, type] = None
    linear_weights_proj: Union[ModuleSpec, type] = None
    linear_wk: Union[ModuleSpec, type] = None
    k_norm: Union[ModuleSpec, type] = None


class CSA2Indexer(MegatronModule):
    """Query-side indexer, plus the key path when the layer is a KV source."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: CSA2IndexerSubmodules,
        plan: CSA2LayerPlan,
        candidate_topk_blocks: int = 0,
        candidate_block_size: int = 0,
        name: str | None = None,
    ) -> None:
        super().__init__(config=config)
        if not plan.runs_indexer:
            raise ValueError(f"model layer {plan.layer_id} ({plan.mode}) does not run an indexer")
        self.plan = plan
        self.n_heads = config.dsa_indexer_n_heads
        self.head_dim = config.dsa_indexer_head_dim
        self.topk = config.dsa_indexer_topk
        self.rope_dim = config.qk_pos_emb_head_dim
        self.candidate_topk_blocks = candidate_topk_blocks
        self.candidate_block_size = candidate_block_size
        q_lora_rank = config.q_lora_rank if config.q_lora_rank is not None else config.hidden_size

        # Reference: scores are relu(q.k) summed over heads with weights
        # ``weights_proj(x) * head_dim^-0.5 * n_heads^-0.5``.
        self.head_weight_scale = self.head_dim**-0.5 * self.n_heads**-0.5

        self.linear_wq_b = build_module(
            submodules.linear_wq_b,
            q_lora_rank,
            self.n_heads * self.head_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
            name=(name + ".linear_wq_b") if name is not None else None,
        )
        # Kept in the checkpoint dtype (bf16) even under FP8 training, as in the reference.
        with get_fp8_disabled_context(config, is_init=True):
            self.linear_weights_proj = build_module(
                submodules.linear_weights_proj,
                config.hidden_size,
                self.n_heads,
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                skip_weight_param_allocation=False,
                parallel_mode="duplicated",
                name=(name + ".linear_weights_proj") if name is not None else None,
            )

        self.owns_keys = plan.runs_compressor
        if self.owns_keys:
            with get_fp8_disabled_context(config, is_init=True):
                self.linear_wk = build_module(
                    submodules.linear_wk,
                    config.v_head_dim,
                    self.head_dim,
                    config=config,
                    init_method=config.init_method,
                    bias=False,
                    skip_bias_add=False,
                    skip_weight_param_allocation=False,
                    parallel_mode="duplicated",
                    name=(name + ".linear_wk") if name is not None else None,
                )
            self.k_norm = submodules.k_norm(
                hidden_size=self.head_dim, config=config, eps=config.layernorm_epsilon
            )
        else:
            self.linear_wk = None
            self.k_norm = None

        # Only integer selections leave the indexer, so without the indexer distillation loss
        # its parameters have no gradient path; keep them frozen so DDP's gradient accounting
        # stays consistent. The loss and unfreezing are a later milestone.
        self.frozen = bool(getattr(config, "csa2_indexer_frozen", True))
        if self.frozen:
            for param in self.parameters():
                param.requires_grad_(False)

    def build_index_keys(self, latent: torch.Tensor, rope_fn: RopeFn) -> torch.Tensor:
        """Indexer keys ``[n_compressed, b, head_dim]`` from the pre-RoPE compressed latent."""
        if not self.owns_keys:
            raise RuntimeError("only a KV-source indexer can build index keys")
        # BF16 contract for the key path also at run time (not only at construction).
        with get_fp8_disabled_context(self.config):
            keys, _ = self.linear_wk(latent)
        keys = self.k_norm(keys)
        # Entry j stands for token j * ratio: rotate with that position.
        keys = rope_fn(keys.unsqueeze(2), self.plan.compress_ratio)
        return keys.squeeze(2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_compressed: torch.Tensor,
        index_keys: torch.Tensor,
        visible_counts: torch.Tensor,
        rope_fn: RopeFn,
        candidate_blocks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Select compressed positions for every query.

        Args:
            hidden_states: ``[s, b, hidden]`` normalised attention input (for ``weights_proj``).
            q_compressed: ``[s, b, q_lora_rank]`` normalised low-rank query.
            index_keys: ``[n_compressed, b, head_dim]`` shared, RoPE applied.
            visible_counts: ``[s]`` compressed entries visible to each query position.
            rope_fn: rotates the trailing rope dims at token positions (stride 1).
            candidate_blocks: ``[s, b, topk_blocks]`` int32 block ids from the candidate
                source (``-1`` padded), or None.

        Returns:
            ``(topk_indices [s, b, k] int32, candidate_blocks_or_None)``; the block ids are only
            produced by the candidate-source layer.
        """
        s, b, _ = hidden_states.shape
        n_compressed = index_keys.size(0)

        q, _ = self.linear_wq_b(q_compressed)
        q = q.view(s, b, self.n_heads, self.head_dim)
        q = rope_fn(q, 1)

        with get_fp8_disabled_context(self.config):
            head_weights, _ = self.linear_weights_proj(hidden_states)
        # Scale in fp32: the factor (head_dim * n_heads)^-0.5 is not a power of two, so applying
        # it to the bf16 projection output would add one rounding to every head weight.
        head_weights = head_weights.float() * self.head_weight_scale

        scores = indexer_scores(q, index_keys, head_weights)  # [s, b, n]
        reachable = torch.arange(n_compressed, device=scores.device).view(1, 1, -1) < (
            visible_counts.view(s, 1, 1)
        )
        scores = scores.masked_fill(~reachable, float("-inf"))

        produced_candidates = None
        if self.plan.is_candidate_source:
            produced_candidates = select_candidate_block_ids(
                scores,
                visible_counts.view(s, 1),
                self.candidate_topk_blocks,
                self.candidate_block_size,
            )
        elif self.plan.uses_candidates:
            if candidate_blocks is None:
                raise RuntimeError(
                    f"model layer {self.plan.layer_id} needs the candidate blocks from layer "
                    f"{self.plan.kv_source}"
                )
            mask = candidate_blocks_to_mask(
                candidate_blocks, n_compressed, self.candidate_block_size
            )
            scores = scores.masked_fill(~mask, float("-inf"))

        topk = indexer_topk_indices(scores, visible_counts, self.topk)
        return topk, produced_candidates

    # ---- packed (THD) path -------------------------------------------------------------------

    def build_index_keys_packed(self, latent: torch.Tensor, rope_rows_fn) -> torch.Tensor:
        """Indexer keys ``[n, head_dim]`` for packed latents; ``rope_rows_fn`` rotates rows at
        caller-supplied positions."""
        if not self.owns_keys:
            raise RuntimeError("only a KV-source indexer can build index keys")
        with get_fp8_disabled_context(self.config):
            keys, _ = self.linear_wk(latent)
        keys = self.k_norm(keys)
        return rope_rows_fn(keys.unsqueeze(1)).squeeze(1)

    def forward_packed(
        self,
        hidden_rows: torch.Tensor,
        q_rows: torch.Tensor,
        index_keys: torch.Tensor,
        positions: torch.Tensor,
        segment_ids: torch.Tensor,
        valid: torch.Tensor,
        cu_comp: torch.Tensor,
        rope_rows_fn,
        candidate_blocks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Top-k over the compressed entries of each row's own segment (packed layout).

        Args:
            hidden_rows: ``[n_q, hidden]`` local rows; ``q_rows``: ``[n_q, q_lora_rank]``.
            index_keys: ``[n_comp_total, head_dim]`` global sequence-major shared keys.
            positions / segment_ids / valid: per local row (see ``thd.row_metadata``).
            cu_comp: ``[B + 1]`` compressed cumulative lengths.
            rope_rows_fn: rotates ``[n, h, d]`` rows at the local rows' positions.
            candidate_blocks: ``[n_q, topk_blocks]`` int32 segment-local block ids per row from
                the candidate source (``-1`` padded), or None.

        Returns:
            ``(topk [n_q, k] int32 segment-local ids (-1 invalid), candidate_blocks_or_None)``.
        """
        n_q = hidden_rows.size(0)
        q, _ = self.linear_wq_b(q_rows)
        q = rope_rows_fn(q.view(n_q, self.n_heads, self.head_dim))
        with get_fp8_disabled_context(self.config):
            raw_head_weights, _ = self.linear_weights_proj(hidden_rows)
        head_weights = raw_head_weights.float() * self.head_weight_scale  # fp32, see forward()
        use_fused = getattr(self.config, "csa2_indexer_impl", "reference") == "fused"

        k_width = min(self.topk, int(cu_comp[-1])) if cu_comp.numel() > 1 else 0
        topk = torch.full((n_q, k_width), -1, dtype=torch.int32, device=hidden_rows.device)
        produced = None
        if self.plan.is_candidate_source:
            produced = torch.full(
                (n_q, self.candidate_topk_blocks), -1, dtype=torch.int32, device=hidden_rows.device
            )
        cu_comp_list = cu_comp.tolist()
        max_chunk_rows = max(1, int(getattr(self.config, "csa2_indexer_chunk_rows", 4096)))
        # The selection is integer valued; with frozen indexer weights nothing downstream needs
        # the score graph, so scoring runs without autograd in row chunks. The chunk is bounded
        # by a memory budget on the dense fp32 [rows, n_comp] score matrix and its masks, and
        # the heads are reduced one at a time so no [rows, heads, n_comp] tensor exists.
        score_ctx = torch.no_grad() if self.frozen else nullcontext()
        with score_ctx:
            for segment in torch.unique(segment_ids[valid]).tolist():
                in_segment = (segment_ids == segment) & valid
                seg_rows = torch.nonzero(in_segment, as_tuple=False).squeeze(1)
                start, end = cu_comp_list[segment], cu_comp_list[segment + 1]
                n_comp = end - start
                if n_comp == 0 or seg_rows.numel() == 0:
                    continue
                keys_t = None if use_fused else index_keys[start:end].float().t()  # [d, n_comp]
                chunk_rows = min(max_chunk_rows, _rows_within_budget(n_comp))
                for c0 in range(0, seg_rows.numel(), chunk_rows):
                    rows = seg_rows[c0 : c0 + chunk_rows]
                    if (
                        use_fused
                        and self.plan.uses_candidates
                        and candidate_blocks is not None
                        and candidate_scores_available()
                        and candidate_kernel_supported(
                            q.size(1),
                            q.size(2),
                            self.candidate_block_size,
                            candidate_blocks.size(1),
                        )
                    ):
                        # Reindex layer: score only the candidate blocks (fused Triton kernel
                        # with the block / range / causal masks inside), top-k over the
                        # candidate width, ids mapped back to the segment-local axis.
                        visible = ((positions[rows] + 1) // self.plan.compress_ratio).clamp_max(
                            n_comp
                        )
                        cand = candidate_scores(
                            q[rows],
                            index_keys[start:end],
                            raw_head_weights[rows],
                            self.head_weight_scale,
                            candidate_blocks[rows],
                            self.candidate_block_size,
                            visible,
                        )
                        local = candidate_topk_ids(
                            cand, candidate_blocks[rows], self.candidate_block_size, self.topk
                        )
                        # The destination is min(topk, total compressed entries) wide, the
                        # candidate result min(topk, candidate width): with fewer compressed
                        # entries than topk in the whole pack the result is wider than the
                        # destination and its tail holds only -1 (valid ids <= n_comp).
                        width = min(local.size(1), topk.size(1))
                        topk[rows, :width] = local[:, :width]
                        continue
                    scores = None
                    if use_fused:
                        # Rows of one segment chunk are consecutive positions; the kernel
                        # derives the causal limit from the first position.
                        pos0, pos1 = int(positions[rows[0]]), int(positions[rows[-1]])
                        if pos1 - pos0 == rows.numel() - 1:
                            scores = fused_indexer_scores_rows(
                                q[rows],
                                index_keys[start:end],
                                raw_head_weights[rows],
                                self.head_weight_scale,
                                self.plan.compress_ratio,
                                pos0,
                            )
                    if scores is None:
                        if keys_t is None:
                            keys_t = index_keys[start:end].float().t()
                        scores = indexer_scores_rows(q[rows], keys_t, head_weights[rows])
                    scores = scores.unsqueeze(1)  # [rows, 1, n_comp]
                    visible = (positions[rows] + 1) // self.plan.compress_ratio
                    visible = visible.clamp_max(n_comp)
                    reachable = torch.arange(n_comp, device=scores.device).view(1, 1, -1)
                    reachable = reachable < visible.view(-1, 1, 1)
                    scores = scores.masked_fill(~reachable, float("-inf"))
                    if self.plan.is_candidate_source:
                        produced[rows] = select_candidate_block_ids(
                            scores,
                            visible.view(-1, 1),
                            self.candidate_topk_blocks,
                            self.candidate_block_size,
                        ).squeeze(1)
                    elif self.plan.uses_candidates:
                        if candidate_blocks is None:
                            raise RuntimeError(
                                f"model layer {self.plan.layer_id} needs the candidate blocks "
                                f"from layer {self.plan.kv_source}"
                            )
                        mask = candidate_blocks_to_mask(
                            candidate_blocks[rows].unsqueeze(1), n_comp, self.candidate_block_size
                        )
                        scores = scores.masked_fill(~mask, float("-inf"))
                    local = indexer_topk_indices(scores, visible, self.topk).squeeze(1)
                    topk[rows, : local.size(1)] = local
        return topk, produced
