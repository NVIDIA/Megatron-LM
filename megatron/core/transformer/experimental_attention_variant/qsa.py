# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""QSA (Qwen Sparse Attention) — GQA-based block-sparse attention.

Semantics reference: HuggingFace ``modeling_qwen4_exp.py`` (transformers @ 99e19a9a,
Apache-2.0) ``Qwen4ExpTextQSAIndexer`` / ``Qwen4ExpTextAttention``; parity oracle in
``tests/unit_tests/transformer/experimental_attention_variant/qsa_reference.py``.

Structure:

* :class:`QSAIndexer` — parameter-free 4-token mean-pool compression of a single shared
  index-key head, 4 MQA indexer query heads, ``relu(q@k).sum(heads)/sqrt(head_dim)``
  scoring in fp32, deterministic top-k block selection with ``(score desc, block_id asc)``
  tie-breaking, unconditional retention of the ragged tail (< 1 block) per query.
  The indexer consumes the post-input-layernorm hidden states (same input as the main
  q/k/v projections, matching HF) and is detached from the main gradient path; its
  parameters are replicated across TP (``average_gradients_across_tp_domain``).
* :class:`QSASelfAttention` — a :class:`~megatron.core.transformer.attention.SelfAttention`
  whose attention mask is the indexer's selected-token mask (selection already implies
  causality). The main attention itself is plain GQA composed from existing knobs
  (``attention_output_gate``, ``rotary_percent``, ``qk_layernorm``, ``kv_channels``).

This is the **dense-mask bridge** path (functional correctness; no sparsity speedup):
the selection mask is materialized and fed to dense core attention with
``AttnMaskType.arbitrary``. The sparse kernel path replaces the mask consumption later
without changing selection semantics. BSHD only; packed THD arrives with the sparse path.
"""

import math
from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple, Union

import torch
from torch import Tensor

from megatron.core.fp8_utils import get_fp8_disabled_context
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
)
from megatron.core.transformer.experimental_attention_variant.dsa_layout import (
    build_packed_allgather_cp_query_positions_and_key_reorder,
    build_zigzag_allgather_cp_key_reorder,
    build_zigzag_cp_local_positions,
)
from megatron.core.transformer.experimental_attention_variant.dsa_indexer_loss import (
    normalize_indexer_target_,
)
from megatron.core.transformer.experimental_attention_variant.dsa_masking import (
    masked_log_softmax,
    masked_softmax,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig


class QSASelection(NamedTuple):
    """Selection artifacts for the sparse attention path and the sparse-KL loss.

    q:          indexer queries ``[s, b, n_heads, head_dim]`` (grad into indexer params).
    block_keys: pooled/normed/RoPE'd block keys ``[n_blocks_total, b, head_dim]`` (grad).
                For packed THD inputs this is the per-document concatenation.
    order:      selected block ids ``[b, s, block_topk]`` (deterministic top-k order),
                DOCUMENT-LOCAL under THD.
    picked:     bool validity of each selected slot ``[b, s, block_topk]``.
    own_block:  per-token (document-local) id of the block containing the next token,
                ``[s]`` — appending it to the selection reproduces the unconditional
                ragged tail under in-kernel causal masking.
    block_offsets: ``[s]`` per-token offset of its document's first row in
                ``block_keys`` (zeros for BSHD), or None.
    token_offsets: ``[s]`` per-token offset of its document start in the packed
                sequence (zeros for BSHD), or None.

    Student scores for the KL are recomputed chunked from ``q``/``block_keys`` at the
    selected blocks only, so no ``[s, n_blocks]`` gradient graph is ever materialized.
    """

    q: Tensor
    block_keys: Tensor
    order: Tensor
    picked: Tensor
    own_block: Tensor
    block_offsets: Optional[Tensor] = None
    token_offsets: Optional[Tensor] = None
    # Attention-kernel block space (ceil blocks per document, incl. the partial
    # trailing block): per-token offsets plus per-block token [base, end) ranges.
    attn_block_offsets: Optional[Tensor] = None  # [s]
    attn_block_bases: Optional[Tensor] = None  # [num_attn_blocks] int32
    attn_block_ends: Optional[Tensor] = None  # [num_attn_blocks] int32
    # Context parallelism (allgather CP): global position of each LOCAL query row in
    # the token space the block ranges live in (zigzag positions for BSHD, global
    # packed ids for THD). None means queries cover the full sequence (cp == 1).
    q_positions: Optional[Tensor] = None  # [s_local] int64
    # Permutation restoring a rank-major CP allgather to global token order; computed
    # once per selection and reused by every K/V gather of the layer (attention and
    # the KL teacher).
    cp_key_reorder: Optional[Tensor] = None  # [s_global] int64


@dataclass
class QSAIndexerSubmodules:
    """Submodules of the QSA indexer."""

    index_qk_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    k_layernorm: Union[ModuleSpec, type] = None


class QSAIndexer(MegatronModule):
    """Block-granular token selector for QSA.

    Projects hidden states to 4 MQA indexer query heads plus one shared index-key head
    (``index_qk_proj: hidden -> (n_heads + 1) * head_dim``), mean-pools the raw index
    keys over fixed causal-prefix-aligned ``compress_ratio``-token blocks (fp32,
    parameter-free), RMS-normalizes queries and pooled keys, applies the main
    attention's partial RoPE (queries at token positions; pooled keys at their block's
    first-token position), scores ``relu(q@k).sum(heads) / sqrt(head_dim)`` in fp32,
    and deterministically selects the top ``budget / compress_ratio`` visible blocks
    per query. The ragged tail (tokens past the last complete block in the causal
    prefix) is always kept.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: QSAIndexerSubmodules,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        super().__init__(config=config)
        self.n_heads = config.qsa_indexer_n_heads
        self.head_dim = config.qsa_indexer_head_dim
        self.compress_ratio = config.qsa_indexer_compress_ratio
        self.block_topk = config.qsa_indexer_budget // config.qsa_indexer_compress_ratio

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        self.pg_collection = pg_collection

        # D8: indexer GEMMs stay in high precision under FP8/MXFP8 — the projection is
        # tiny (N = (n_heads + 1) * head_dim) and selection noise amplifies downstream.
        with get_fp8_disabled_context(self.config, is_init=True):
            self.index_qk_proj = build_module(
                submodules.index_qk_proj,
                config.hidden_size,
                (self.n_heads + 1) * self.head_dim,
                config=self.config,
                init_method=self.config.init_method,
                bias=False,
                skip_bias_add=False,
                skip_weight_param_allocation=False,
                parallel_mode="duplicated",
            )
        self.q_layernorm = build_module(
            submodules.q_layernorm,
            config=self.config,
            hidden_size=self.head_dim,
            eps=self.config.layernorm_epsilon,
        )
        self.k_layernorm = build_module(
            submodules.k_layernorm,
            config=self.config,
            hidden_size=self.head_dim,
            eps=self.config.layernorm_epsilon,
        )

        # The indexer is replicated across tensor-parallel ranks (4 query heads do not
        # shard usefully); average its gradients across the TP domain like DSAIndexer.
        for param in self.parameters():
            setattr(param, "average_gradients_across_tp_domain", True)

        # Gathered-global BSHD frequency tables per (s_local, dtype); rotary
        # frequencies are position-deterministic, so one gather per length suffices.
        self._cp_freqs_cache = {}

    def _project(self, hidden_states: Tensor, rotary_pos_emb: Tensor):
        """Shared front-end: detach, SP-gather, project, norm and RoPE the queries."""
        # Selection is not differentiable and (by design) the indexer does not
        # backpropagate into the trunk; its own training signal is the sparse KL loss.
        hidden_states = hidden_states.detach()
        if self.config.sequence_parallel:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states, group=self.pg_collection.tp
            )
        assert rotary_pos_emb.shape[-1] <= self.head_dim, (
            f"QSA indexer reuses the main-attention rotary frequencies (dim "
            f"{rotary_pos_emb.shape[-1]}), which must fit its head dim ({self.head_dim})."
        )
        s, b, _ = hidden_states.shape
        with get_fp8_disabled_context(self.config):
            qk, _ = self.index_qk_proj(hidden_states)
        q, raw_keys = torch.split(
            qk, [self.n_heads * self.head_dim, self.head_dim], dim=-1
        )  # [s, b, n*d], [s, b, d]
        q = self.q_layernorm(q.reshape(s, b, self.n_heads, self.head_dim))
        return q, raw_keys

    def _block_keys(self, raw_keys: Tensor, rotary_pos_emb: Tensor, n_blocks: int) -> Tensor:
        """Parameter-free fp32 mean-pool + norm + block-first-token RoPE."""
        r = self.compress_ratio
        pooled = raw_keys[: n_blocks * r].view(n_blocks, r, raw_keys.shape[1], self.head_dim)
        pooled = pooled.float().mean(dim=1).to(raw_keys.dtype)  # [n_blocks, b, d]
        pooled = self.k_layernorm(pooled)
        # The stride-r frequency slice is materialized contiguously: the fused RoPE
        # kernel path is not guaranteed to accept strided frequency tensors.
        block_freqs = rotary_pos_emb[: n_blocks * r : r].contiguous()
        return apply_rotary_pos_emb(
            pooled.unsqueeze(2),  # [n_blocks, b, 1, d]
            block_freqs,
            config=self.config,
            cp_group=self.pg_collection.cp,
        ).squeeze(2)

    def select(
        self,
        hidden_states: Tensor,
        rotary_pos_emb: Tensor,
        cu_seqlens: Optional[Tensor] = None,
    ) -> QSASelection:
        """Run the indexer and return the deterministic block selection.

        Args:
            hidden_states: ``[sq, b, h]`` post-input-layernorm hidden states.
            rotary_pos_emb: rotary frequencies covering the longest document.
            cu_seqlens: optional ``[n_docs + 1]`` boundaries for packed THD inputs
                (requires b == 1). Blocks never cross documents; positions and block
                ids restart at every boundary. Under CP the boundaries describe the
                GLOBAL pack; the local shard is the config-declared zigzag layout.
        """
        cp_group = self.pg_collection.cp
        if cp_group is not None and cp_group.size() > 1:
            return self._select_cp(hidden_states, rotary_pos_emb, cu_seqlens)
        q_all, raw_keys = self._project(hidden_states, rotary_pos_emb)
        s, b = q_all.shape[0], q_all.shape[1]
        r = self.compress_ratio
        device = q_all.device

        if cu_seqlens is None:
            q = apply_rotary_pos_emb(
                q_all, rotary_pos_emb[:s], config=self.config, cp_group=self.pg_collection.cp
            )
            t = torch.arange(s, device=device)
            m_t = (t + 1) // r
            n_blocks = s // r
            k = max(min(self.block_topk, n_blocks), 1)
            if n_blocks > 0:
                block_keys = self._block_keys(raw_keys, rotary_pos_emb, n_blocks)
                order, picked = self._streaming_topk(q, block_keys, m_t, n_blocks, k)
            else:
                block_keys = raw_keys.new_zeros(0, b, self.head_dim)
                order = torch.zeros(b, s, k, dtype=torch.long, device=device)
                picked = torch.zeros(b, s, k, dtype=torch.bool, device=device)
            # attention-kernel block space: ceil blocks + 1 so the own block of a
            # multiple-of-r position (fully in the future) has a valid, empty range
            nb_attn = s // r + 1
            attn_bases = (torch.arange(nb_attn, device=device, dtype=torch.int32) * r)
            attn_ends = torch.clamp(attn_bases + r, max=s)
            return QSASelection(
                q=q,
                block_keys=block_keys,
                order=order,
                picked=picked,
                own_block=m_t,
                attn_block_offsets=torch.zeros(s, dtype=torch.long, device=device),
                attn_block_bases=attn_bases,
                attn_block_ends=attn_ends,
            )

        assert b == 1, "packed THD inputs must be flattened to batch size 1"
        k = max(self.block_topk, 1)
        q_rot = torch.empty_like(q_all)
        order = torch.zeros(1, s, k, dtype=torch.long, device=device)
        picked = torch.zeros(1, s, k, dtype=torch.bool, device=device)
        own_block = torch.zeros(s, dtype=torch.long, device=device)
        block_offsets = torch.zeros(s, dtype=torch.long, device=device)
        token_offsets = torch.zeros(s, dtype=torch.long, device=device)
        attn_block_offsets = torch.zeros(s, dtype=torch.long, device=device)
        block_keys_parts = []
        attn_bases_parts = []
        attn_ends_parts = []
        blocks_so_far = 0
        attn_blocks_so_far = 0
        # One host sync for all boundaries instead of one per document.
        cu_list = cu_seqlens.tolist()
        for d0 in range(len(cu_list) - 1):
            s0, s1 = int(cu_list[d0]), int(cu_list[d0 + 1])
            sd = s1 - s0
            if sd == 0:
                continue
            q_d = apply_rotary_pos_emb(
                q_all[s0:s1],
                rotary_pos_emb[:sd],
                config=self.config,
                cp_group=self.pg_collection.cp,
            )
            q_rot[s0:s1] = q_d
            t_d = torch.arange(sd, device=device)
            m_t_d = (t_d + 1) // r
            n_blocks_d = sd // r
            own_block[s0:s1] = m_t_d
            block_offsets[s0:s1] = blocks_so_far
            token_offsets[s0:s1] = s0
            attn_block_offsets[s0:s1] = attn_blocks_so_far
            nb_attn_d = sd // r + 1
            bases_d = s0 + torch.arange(nb_attn_d, device=device, dtype=torch.int32) * r
            attn_bases_parts.append(bases_d)
            attn_ends_parts.append(torch.clamp(bases_d + r, max=s1))
            attn_blocks_so_far += nb_attn_d
            if n_blocks_d > 0:
                bk_d = self._block_keys(raw_keys[s0:s1], rotary_pos_emb, n_blocks_d)
                k_d = min(self.block_topk, n_blocks_d)
                order_d, picked_d = self._streaming_topk(q_d, bk_d, m_t_d, n_blocks_d, k_d)
                order[:, s0:s1, :k_d] = order_d
                picked[:, s0:s1, :k_d] = picked_d
                block_keys_parts.append(bk_d)
                blocks_so_far += n_blocks_d
        block_keys = (
            torch.cat(block_keys_parts, dim=0)
            if block_keys_parts
            else raw_keys.new_zeros(0, 1, self.head_dim)
        )
        return QSASelection(
            q=q_rot,
            block_keys=block_keys,
            order=order,
            picked=picked,
            own_block=own_block,
            block_offsets=block_offsets,
            token_offsets=token_offsets,
            attn_block_offsets=attn_block_offsets,
            attn_block_bases=torch.cat(attn_bases_parts),
            attn_block_ends=torch.cat(attn_ends_parts),
        )

    def _rope_at_positions(self, t: Tensor, rotary_pos_emb: Tensor, positions: Tensor) -> Tensor:
        """Apply RoPE at explicit per-row positions from a FULL frequency table.

        Indexing the frequency table by position keeps the BSHD apply path (which
        never slices by CP) exact for any sharding layout, sidestepping the
        per-document assumption in the THD+CP rope helper. Note: under CP the
        model passes a full table only for packed (THD) inputs — for BSHD,
        RotaryEmbedding already emits the rank's zigzag slice."""
        freqs = rotary_pos_emb.index_select(0, positions).contiguous()
        return apply_rotary_pos_emb(t, freqs, config=self.config, cp_group=self.pg_collection.cp)

    def _cp_global_freqs(self, rotary_pos_emb: Tensor, s_local: int, reorder: Tensor) -> Tensor:
        """Rebuild the full BSHD frequency table from each rank's zigzag slice.

        Under CP, ``RotaryEmbedding`` returns only this rank's zigzag rows for
        BSHD inputs; the pooled block keys live in GLOBAL order, so the table is
        all-gathered and restored with the same reorder index as the keys. The
        result is cached per (length, dtype): frequencies are a deterministic
        function of position, so one gather per length suffices."""
        key = (s_local, rotary_pos_emb.dtype)
        cached = self._cp_freqs_cache.get(key)
        if cached is not None:
            return cached
        gathered = gather_from_sequence_parallel_region(
            rotary_pos_emb[:s_local].contiguous(), group=self.pg_collection.cp
        )
        result = gathered.index_select(0, reorder)
        self._cp_freqs_cache[key] = result
        return result

    def _cp_layout(self, s_local: int, cu_seqlens: Optional[Tensor], device):
        """(q_positions, key_reorder_idx) for this rank's allgather-CP shard."""
        cp_group = self.pg_collection.cp
        cp_size, cp_rank = cp_group.size(), cp_group.rank()
        if cu_seqlens is None:
            q_pos = build_zigzag_cp_local_positions(s_local * cp_size, cp_size, cp_rank, device)
            reorder = build_zigzag_allgather_cp_key_reorder(s_local, cp_size, device)
            return q_pos, reorder
        return build_packed_allgather_cp_query_positions_and_key_reorder(
            cu_seqlens,
            cu_seqlens,
            cp_size,
            cp_rank,
            device,
            local_output_size=s_local,
            query_cu_seqlens_cover_output=True,
            key_cu_seqlens_cover_output=True,
            cp_packing_layout=self.config.qsa_cp_packing_layout,
        )

    def _select_cp(
        self,
        hidden_states: Tensor,
        rotary_pos_emb: Tensor,
        cu_seqlens: Optional[Tensor] = None,
    ) -> QSASelection:
        """Allgather-CP selection: local queries, globally ordered block keys.

        Only the single shared raw index-key head is all-gathered across CP (its
        backward is a reduce-scatter, so key projections receive gradient from the
        queries on every rank); queries stay local at their zigzag positions.
        Pooling happens strictly AFTER the gather+reorder — pooling rank-local
        neighbours would combine tokens that are not adjacent in the sequence.
        All returned block/token coordinates live in the GLOBAL token space, and
        ``q_positions`` records each local row's global position.
        """
        cp_group = self.pg_collection.cp
        cp_size = cp_group.size()
        q_all, raw_keys = self._project(hidden_states, rotary_pos_emb)
        s_loc, b = q_all.shape[0], q_all.shape[1]
        r = self.compress_ratio
        device = q_all.device

        q_pos, reorder = self._cp_layout(s_loc, cu_seqlens, device)
        gathered = gather_from_sequence_parallel_region(raw_keys, group=cp_group)
        global_keys = gathered.index_select(0, reorder)  # [S_g, b, d] global order

        if cu_seqlens is None:
            s_g = s_loc * cp_size
            m_t = (q_pos + 1) // r
            n_blocks = s_g // r
            k = max(min(self.block_topk, n_blocks), 1)
            # BSHD under CP: rotary_pos_emb is already this rank's zigzag slice, so
            # it applies to the local queries directly; the block keys need the
            # rebuilt GLOBAL table.
            q = apply_rotary_pos_emb(
                q_all,
                rotary_pos_emb[:s_loc],
                config=self.config,
                cp_group=self.pg_collection.cp,
            )
            if n_blocks > 0:
                global_freqs = self._cp_global_freqs(rotary_pos_emb, s_loc, reorder)
                block_keys = self._block_keys(global_keys, global_freqs, n_blocks)
                order, picked = self._streaming_topk(q, block_keys, m_t, n_blocks, k)
            else:
                block_keys = raw_keys.new_zeros(0, b, self.head_dim)
                order = torch.zeros(b, s_loc, k, dtype=torch.long, device=device)
                picked = torch.zeros(b, s_loc, k, dtype=torch.bool, device=device)
            nb_attn = s_g // r + 1
            attn_bases = torch.arange(nb_attn, device=device, dtype=torch.int32) * r
            attn_ends = torch.clamp(attn_bases + r, max=s_g)
            return QSASelection(
                q=q,
                block_keys=block_keys,
                order=order,
                picked=picked,
                own_block=m_t,
                attn_block_offsets=torch.zeros(s_loc, dtype=torch.long, device=device),
                attn_block_bases=attn_bases,
                attn_block_ends=attn_ends,
                q_positions=q_pos,
                cp_key_reorder=reorder,
            )

        assert b == 1, "packed THD inputs must be flattened to batch size 1"
        cu = cu_seqlens.to(device=device, dtype=torch.int64)
        # Positions restart per document; map each local row to its document.
        doc_idx = torch.searchsorted(cu, q_pos, right=True) - 1
        doc_start = cu[doc_idx]
        pos_doc = q_pos - doc_start
        m_t = (pos_doc + 1) // r
        q_rot = self._rope_at_positions(q_all, rotary_pos_emb, pos_doc)

        k = max(self.block_topk, 1)
        order = torch.zeros(1, s_loc, k, dtype=torch.long, device=device)
        picked = torch.zeros(1, s_loc, k, dtype=torch.bool, device=device)
        block_offsets = torch.zeros(s_loc, dtype=torch.long, device=device)
        attn_block_offsets = torch.zeros(s_loc, dtype=torch.long, device=device)
        block_keys_parts = []
        attn_bases_parts = []
        attn_ends_parts = []
        blocks_so_far = 0
        attn_blocks_so_far = 0
        # q_pos is strictly increasing under both packing layouts, so each document's
        # local rows form one contiguous slice. One host sync for all boundaries.
        bounds = torch.cat([cu, torch.searchsorted(q_pos, cu)]).tolist()
        cu_list, row_bounds = bounds[: cu.numel()], bounds[cu.numel() :]
        for d0 in range(len(cu_list) - 1):
            s0, s1 = int(cu_list[d0]), int(cu_list[d0 + 1])
            sd = s1 - s0
            if sd == 0:
                continue
            r0, r1 = int(row_bounds[d0]), int(row_bounds[d0 + 1])
            attn_block_offsets[r0:r1] = attn_blocks_so_far
            nb_attn_d = sd // r + 1
            bases_d = s0 + torch.arange(nb_attn_d, device=device, dtype=torch.int32) * r
            attn_bases_parts.append(bases_d)
            attn_ends_parts.append(torch.clamp(bases_d + r, max=s1))
            attn_blocks_so_far += nb_attn_d
            n_blocks_d = sd // r
            if n_blocks_d > 0:
                bk_d = self._block_keys(global_keys[s0:s1], rotary_pos_emb, n_blocks_d)
                block_keys_parts.append(bk_d)
                if r1 > r0:
                    k_d = min(self.block_topk, n_blocks_d)
                    order_d, picked_d = self._streaming_topk(
                        q_rot[r0:r1], bk_d, m_t[r0:r1], n_blocks_d, k_d
                    )
                    order[:, r0:r1, :k_d] = order_d
                    picked[:, r0:r1, :k_d] = picked_d
                block_offsets[r0:r1] = blocks_so_far
                blocks_so_far += n_blocks_d
        block_keys = (
            torch.cat(block_keys_parts, dim=0)
            if block_keys_parts
            else raw_keys.new_zeros(0, 1, self.head_dim)
        )
        return QSASelection(
            q=q_rot,
            block_keys=block_keys,
            order=order,
            picked=picked,
            own_block=m_t,
            block_offsets=block_offsets,
            token_offsets=doc_start,
            attn_block_offsets=attn_block_offsets,
            attn_block_bases=torch.cat(attn_bases_parts),
            attn_block_ends=torch.cat(attn_ends_parts),
            q_positions=q_pos,
            cp_key_reorder=reorder,
        )

    def forward(
        self, hidden_states: Tensor, rotary_pos_emb: Tensor, return_selection: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Optional[QSASelection]]]:
        """Compute the selected-token attention mask (dense-mask bridge path, BSHD).

        Args:
            hidden_states: ``[sq, b, h]`` post-input-layernorm hidden states
                (sequence-sharded under sequence parallelism; gathered internally).
            rotary_pos_emb: rotary frequencies ``[s, 1, 1, rotary_dim]`` covering the
                full sequence (the same tensor the main attention consumes).

        Returns:
            Bool mask ``[b, 1, sq, sq]`` in the Megatron/TE convention
            (**True = masked out**). Selection implies causality, so this mask is the
            complete attention mask for the layer.
        """
        assert self.pg_collection.cp is None or self.pg_collection.cp.size() == 1, (
            "The dense-mask bridge is single-rank only; QSA under context parallelism "
            "requires the sparse kernel path (qsa_use_sparse_attention=True)."
        )
        selection = self.select(hidden_states, rotary_pos_emb)
        order, picked = selection.order, selection.picked
        b, s, k = order.shape
        r = self.compress_ratio
        device = order.device
        m_t = selection.own_block
        t = torch.arange(s, device=device)

        visible = torch.zeros(b, s, s, dtype=torch.bool, device=device)
        if selection.block_keys.shape[0] > 0:
            # Expand selected block ids to their tokens and scatter into the mask.
            tok = order.unsqueeze(-1) * r + torch.arange(r, device=device)  # [b, s, k, r]
            tok = tok.reshape(b, s, k * r)
            keep = picked.unsqueeze(-1).expand(-1, -1, -1, r).reshape(b, s, k * r)
            # Absorb dropped slots by scattering them into a sentinel column,
            # then slice the sentinel off (same trick as the HF reference).
            buf = torch.zeros(b, s, s + 1, dtype=torch.bool, device=device)
            buf.scatter_(2, torch.where(keep, tok, s), True)
            visible = buf[..., :s]

        # Unconditional ragged tail: tokens in [m_t * r, t].
        j = torch.arange(s, device=device)
        tail = (j.unsqueeze(0) >= (m_t * r).unsqueeze(1)) & (j.unsqueeze(0) <= t.unsqueeze(1))
        visible = visible | tail.unsqueeze(0)

        mask = ~visible.unsqueeze(1)  # [b, 1, s, s], True = masked out
        if return_selection:
            return mask, selection
        return mask

    @torch.no_grad()
    def _streaming_topk(
        self, q: Tensor, block_keys: Tensor, m_t: Tensor, n_blocks: int, k: int
    ) -> Tuple[Tensor, Tensor]:
        """Chunked K-scan with a deterministic running top-k merge.

        Scores each chunk of blocks (fp32 relu-sum-heads), masks causal visibility,
        and merges into the running top-k under the (score desc, block_id asc)
        lexicographic order — realized as a stable ascending sort by block id
        followed by a stable descending sort by score.

        Returns (order ``[b, s, k]``, picked ``[b, s, k]``).
        """
        s, b = q.shape[0], q.shape[1]
        device = q.device
        chunk = max(k, 2048)

        # Lexicographic (score desc, block_id asc) top-k via UNIQUE integer keys:
        # relu scores are non-negative, so their fp32 bit pattern is order-isomorphic
        # to the value; pack (bits + 1) above an inverted block id. Every valid key is
        # unique, so torch.topk over int64 keys is deterministic (no ties exist), and
        # O(n) selection replaces full sorts. Key 0 marks invisible blocks.
        ID_BITS = 21  # supports up to 2M blocks (8M tokens per document)
        assert n_blocks < (1 << ID_BITS)
        id_code_max = (1 << ID_BITS) - 1

        # On CUDA the scoring + masking + key encoding is one fused Triton kernel
        # (qsa_indexer_keys), avoiding the [s, chunk, heads] einsum intermediate.
        use_kernel = q.is_cuda
        if use_kernel:
            from megatron.core.transformer.experimental_attention_variant.ops.triton_qsa import (
                qsa_indexer_keys,
            )

            q_c = [q[:, bi].contiguous() for bi in range(b)]
            bk_c = [block_keys[:, bi].contiguous() for bi in range(b)]
            m_t_c = m_t.contiguous()

        run_keys = torch.zeros((b, s, k), dtype=torch.int64, device=device)
        for j0 in range(0, n_blocks, chunk):
            j1 = min(j0 + chunk, n_blocks)
            w = j1 - j0
            if use_kernel:
                key = torch.empty(b, s, w, dtype=torch.int64, device=device)
                for bi in range(b):
                    qsa_indexer_keys(q_c[bi], bk_c[bi], m_t_c, j0, w, key[bi])
            else:
                sc = torch.einsum("sbhd,nbd->bshn", q.float(), block_keys[j0:j1].float())
                sc = torch.relu(sc).sum(dim=2) / math.sqrt(self.head_dim)  # [b, s, w]
                ids = torch.arange(j0, j1, device=device)
                vis = ids.unsqueeze(0) < m_t.unsqueeze(1)  # [s, w]
                bits = sc.view(torch.int32).to(torch.int64)
                key = ((bits + 1) << ID_BITS) | (id_code_max - ids)
                key = torch.where(vis.unsqueeze(0), key, torch.zeros_like(key))

            cand = torch.cat([run_keys, key], dim=-1)
            run_keys = cand.topk(k, dim=-1).values  # descending; valid keys are unique

        picked = run_keys > 0
        order = id_code_max - (run_keys & id_code_max)
        order = torch.where(picked, order, torch.zeros_like(order))
        return order, picked


@dataclass
class QSASelfAttentionSubmodules(SelfAttentionSubmodules):
    """Self-attention submodules plus the QSA indexer."""

    indexer: Union[ModuleSpec, type] = None


class QSASelfAttention(SelfAttention):
    """GQA self-attention whose mask is produced by the QSA indexer.

    Dense-mask bridge: the indexer's bool mask is handed to dense core attention with
    ``attn_mask_type=arbitrary``. Everything else (interleaved output gate, partial
    RoPE, per-head qk layernorm, GQA) is the stock :class:`SelfAttention` machinery.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: QSASelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.arbitrary,
        **kwargs,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            **kwargs,
        )
        self.indexer = build_module(
            submodules.indexer, config=self.config, pg_collection=self.pg_collection
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor],
        key_value_states: Optional[Tensor] = None,
        inference_context=None,
        rotary_pos_emb=None,
        packed_seq_params=None,
        **kwargs,
    ):
        """Run the indexer, then self-attention restricted to the selected tokens."""
        assert inference_context is None and kwargs.get("inference_params") is None, (
            "QSA does not implement an inference path (pretrain/continue-pretrain scope)."
        )
        assert key_value_states is None, "QSA is self-attention only."
        freqs = rotary_pos_emb[0] if isinstance(rotary_pos_emb, tuple) else rotary_pos_emb
        assert freqs is not None, "QSA requires rotary position embeddings."

        loss_coeff = self.config.qsa_indexer_loss_coeff or 0.0
        train_indexer = self.training and loss_coeff > 0 and torch.is_grad_enabled()

        if self.config.qsa_use_sparse_attention:
            if attention_mask is not None and attention_mask.dtype != torch.bool:
                raise ValueError(
                    "QSA expects a bool attention mask (True = masked out); additive "
                    f"float masks are not supported (got dtype {attention_mask.dtype})."
                )
            # A bool mask is assumed to be the standard causal/per-document mask, which
            # the block selection already implies, and is therefore not consumed here.
            # Padded batches are NOT supported on the sparse path: pad tokens would be
            # attended and eligible for selection. Use unpadded BSHD or packed THD.
            output, bias, selection = self._sparse_forward(
                hidden_states, freqs, packed_seq_params
            )
        else:
            assert packed_seq_params is None, (
                "QSA dense-mask bridge supports BSHD only; use "
                "qsa_use_sparse_attention=True for packed THD inputs."
            )
            if train_indexer:
                selection_mask, selection = self.indexer(
                    hidden_states, freqs, return_selection=True
                )
            else:
                selection_mask, selection = self.indexer(hidden_states, freqs), None
            if attention_mask is not None:
                if attention_mask.dtype != torch.bool:
                    raise ValueError(
                        "QSA expects a bool attention mask (True = masked out); additive "
                        f"float masks are not supported (got dtype {attention_mask.dtype})."
                    )
                selection_mask = selection_mask | attention_mask

            output, bias = super().forward(
                hidden_states,
                selection_mask,
                key_value_states=None,
                inference_context=None,
                rotary_pos_emb=rotary_pos_emb,
                packed_seq_params=None,
                **kwargs,
            )

        if train_indexer and selection is not None:
            loss_cu_seqlens = (
                packed_seq_params.cu_seqlens_q if packed_seq_params is not None else None
            )
            indexer_loss = self._compute_indexer_loss(
                hidden_states, freqs, selection, loss_coeff, cu_seqlens=loss_cu_seqlens
            )
            output = DSAIndexerLossAutoScaler.apply(output, indexer_loss)
            DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                loss=indexer_loss,
                layer_number=self.layer_number,
                num_layers=self.config.num_layers + (self.config.mtp_num_layers or 0),
            )
        return output, bias

    def _rope_qk(self, query, key, freqs, cu_seqlens):
        """Apply the main-attention partial RoPE to q/k ([s, b, h, d] layout).

        Under packed THD the Megatron rope helpers expect squeezed [t, h, d]
        tensors plus cu_seqlens (per-document positions)."""
        if cu_seqlens is None:
            freqs = freqs[: query.shape[0]]
            query = apply_rotary_pos_emb(
                query, freqs, config=self.config, cp_group=self.pg_collection.cp
            )
            key = apply_rotary_pos_emb(
                key, freqs, config=self.config, cp_group=self.pg_collection.cp
            )
            return query, key
        max_seqlen = freqs.shape[0]
        q2 = apply_rotary_pos_emb(
            query.squeeze(1),
            freqs,
            config=self.config,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            cp_group=self.pg_collection.cp,
        )
        k2 = apply_rotary_pos_emb(
            key.squeeze(1),
            freqs,
            config=self.config,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            cp_group=self.pg_collection.cp,
        )
        return q2.unsqueeze(1), k2.unsqueeze(1)

    def _rope_qk_cp(self, query, key, freqs, selection: QSASelection):
        """CP shard q/k RoPE at the shard's true positions (layout-exact).

        THD: ``freqs`` is a full table — index it at the per-document positions of
        this rank's rows. BSHD: ``freqs`` is already the rank's zigzag slice and
        applies to the local rows directly."""
        if selection.token_offsets is not None:  # THD: positions restart per document
            pos = selection.q_positions - selection.token_offsets
            f = freqs.index_select(0, pos).contiguous()
        else:
            f = freqs[: query.shape[0]]
        query = apply_rotary_pos_emb(
            query, f, config=self.config, cp_group=self.pg_collection.cp
        )
        key = apply_rotary_pos_emb(key, f, config=self.config, cp_group=self.pg_collection.cp)
        return query, key

    def _cp_gather_kv(self, tensors, selection: QSASelection):
        """All-gather CP-sharded [s, b, ...] tensors and restore global order."""
        cp_group = self.pg_collection.cp
        reorder = selection.cp_key_reorder
        return [
            gather_from_sequence_parallel_region(t, group=cp_group).index_select(0, reorder)
            for t in tensors
        ]

    def _sparse_forward(self, hidden_states, freqs, packed_seq_params):
        """Sparse kernel path: indexer selection -> query-tile shared-superset GQA.

        Supports BSHD and packed THD (``packed_seq_params.qkv_format == 'thd'``).
        The kernels take global block ids plus per-block token [base, end) ranges,
        so packed varlen needs no cu_seqlens on the kernel side, and any GQA group
        size works without head padding.
        """
        from megatron.core.transformer.experimental_attention_variant.ops.triton_qsa import (
            qsa_sparse_attention,
        )

        cu_seqlens = None
        if packed_seq_params is not None:
            assert packed_seq_params.qkv_format == 'thd', "QSA supports BSHD or packed THD"
            cu_seqlens = packed_seq_params.cu_seqlens_q
            # QSA is self-attention only, so q and kv boundaries must coincide. A full
            # torch.equal would force a device sync per layer; check identity/shape and
            # rely on PackedSeqParams construction for element equality.
            cu_kv = packed_seq_params.cu_seqlens_kv
            assert cu_kv is None or cu_kv is cu_seqlens or cu_kv.shape == cu_seqlens.shape, (
                "QSA is self-attention only (cu_seqlens_q must equal cu_seqlens_kv)."
            )

        # THD inputs arrive as [t, 1, h]; SelfAttention would squeeze them later, we
        # keep the [s, b, ...] convention throughout and reshape for the kernel.
        selection = self.indexer.select(hidden_states, freqs, cu_seqlens=cu_seqlens)

        qkv = self.get_query_key_value_tensors(
            hidden_states,
            None,
            split_qkv=True,
            output_gate=self.config.attention_output_gate,
        )
        if self.config.attention_output_gate:
            query, key, value, gate = qkv
        else:
            query, key, value = qkv
            gate = None
        cp_size = self.pg_collection.cp.size() if self.pg_collection.cp is not None else 1
        if cp_size > 1:
            # Allgather CP: RoPE locally at the shard's true positions, then gather
            # K/V into global order. The gather is differentiable (backward =
            # reduce-scatter), so dk/dv flow back to every rank's projections.
            query, key = self._rope_qk_cp(query, key, freqs, selection)
            key, value = self._cp_gather_kv([key, value], selection)
        else:
            query, key = self._rope_qk(query, key, freqs, cu_seqlens)

        s, b, np_local, d = query.shape
        q_bt = query.permute(1, 0, 2, 3)
        k_bt = key.permute(1, 0, 2, 3)
        v_bt = value.permute(1, 0, 2, 3)

        # Selection: the query's own block leads; the deterministic top-k follows.
        # In-kernel token [base, end) + causal masking makes the own-block entry
        # reproduce the unconditional ragged tail exactly (and drop it when fully in
        # the future). Ids are translated from document-local to the global
        # attention-block space (ceil blocks per document).
        order, picked = selection.order, selection.picked
        own = selection.own_block.view(1, s, 1).expand(b, s, 1)
        block_indices = torch.cat([own, order], dim=-1)  # [b, s, k + 1] doc-local
        block_indices = block_indices + selection.attn_block_offsets.view(1, s, 1)
        block_counts = 1 + picked.sum(dim=-1)  # [b, s]

        core_out = qsa_sparse_attention(
            q_bt,
            k_bt,
            v_bt,
            block_indices=block_indices,
            block_counts=block_counts,
            block_bases=selection.attn_block_bases,
            block_ends=selection.attn_block_ends,
            scale=self.hidden_size_per_attention_head**-0.5,
            q_positions=selection.q_positions,
        )  # [b, s, np, d_v]
        core_out = core_out.reshape(b, s, np_local * core_out.shape[-1]).permute(1, 0, 2)

        if gate is not None:
            core_out = self._apply_output_gate(core_out, gate)
        output, bias = self.linear_proj(core_out)
        return output, bias, selection

    def _compute_indexer_loss(
        self,
        hidden_states: Tensor,
        freqs: Tensor,
        selection: QSASelection,
        loss_coeff: float,
        cu_seqlens: Optional[Tensor] = None,
    ) -> Tensor:
        """Sparse-KL distillation loss for the indexer (block granularity).

        Teacher (tech-report Eq. 17/20): the main attention distribution over the
        tokens of the *selected* blocks only (bounded memory — nothing outside the
        selected set is materialized), computed from detached, re-projected q/k,
        softmaxed per head in fp32, summed over heads (TP all-reduced — main-attention
        heads are TP-sharded, the indexer is replicated), **max-pooled** to 4-token
        blocks ("a block is worth its most salient token"; sum-pooling would dilute
        peaky signals), and L1-normalized over the selected block set. The report's
        intermediate token-level L1 (Eq. 17) is a per-query scalar that the final
        selected-set L1 absorbs, so it is omitted. Student: log-softmax of the
        indexer's fp32 block scores restricted to the same set (Eq. 20). The indexer
        input is detached, so this loss trains only the indexer parameters.
        """
        r = self.indexer.compress_ratio
        order, picked = selection.order, selection.picked  # [b, s, k]
        b, s, k = order.shape

        # Detached teacher q/k: re-run the QKV projection (+ qk norms inside) and RoPE.
        with torch.no_grad():
            qkv = self.get_query_key_value_tensors(
                hidden_states.detach(),
                None,
                split_qkv=True,
                output_gate=self.config.attention_output_gate,
            )
            query, key = qkv[0], qkv[1]  # [s, b, np, d], [s, b, ng, d]
            cp = self.pg_collection.cp
            if cp is not None and cp.size() > 1 and selection.q_positions is not None:
                # Teacher under allgather CP: local queries at their true positions,
                # keys gathered to global order (the token gather below uses global
                # ids). No grad here, so the gather carries no backward cost.
                query, key = self._rope_qk_cp(query, key, freqs, selection)
                (key,) = self._cp_gather_kv([key], selection)
            else:
                query, key = self._rope_qk(query, key, freqs, cu_seqlens)
        q_bt = query.permute(1, 0, 2, 3)  # [b, s, np, d]
        k_bt = key.permute(1, 0, 2, 3)  # [b, s, ng, d]
        np_local, ng_local = q_bt.shape[2], k_bt.shape[2]
        hpg = np_local // ng_local
        d = q_bt.shape[-1]
        scale = self.hidden_size_per_attention_head**-0.5

        tp_group = self.pg_collection.tp
        tp_size = tp_group.size() if tp_group is not None else 1

        chunk = 256
        arange_r = torch.arange(r, device=order.device)

        # Pass 1 (no grad): local-head teacher, block-max-pooled into a small
        # [b, s, k] fp32 buffer. The Eq.(17) MaxPool is nonlinear, so the TP head-sum
        # must complete BEFORE pooling (deferring a single block-level reduce would
        # be exact only for sum pooling) — but the reduce granularity is free: the
        # token-level mass is staged over several einsum chunks and all-reduced per
        # stage, keeping the collective count low without growing the einsum tile.
        # Normalization still commutes (it happens after the reduce).
        targets = torch.zeros(b, s, k, dtype=torch.float32, device=order.device)
        stage = chunk * 8
        with torch.no_grad():
            for g0 in range(0, s, stage):
                g1 = min(g0 + stage, s)
                mass = torch.zeros(
                    b, g1 - g0, k, r, dtype=torch.float32, device=order.device
                )
                for c0 in range(g0, g1, chunk):
                    c1 = min(c0 + chunk, g1)
                    order_c = order[:, c0:c1]  # [b, c, k]
                    picked_c = picked[:, c0:c1]  # [b, c, k]
                    c = c1 - c0

                    tok = order_c.unsqueeze(-1) * r + arange_r  # [b, c, k, r] doc-local
                    if selection.token_offsets is not None:
                        tok = tok + selection.token_offsets[c0:c1].view(1, c, 1, 1)
                    tok = tok.reshape(b, c * k * r)
                    k_sel = torch.gather(
                        k_bt, 1, tok[..., None, None].expand(-1, -1, ng_local, d)
                    ).view(b, c, k * r, ng_local, d)

                    q_c = q_bt[:, c0:c1].view(b, c, ng_local, hpg, d)
                    teacher = torch.einsum(
                        "bcghd,bctgd->bcght", q_c.float(), k_sel.float()
                    ) * scale  # [b, c, g, h, T]
                    valid_t = picked_c.repeat_interleave(r, dim=-1)  # [b, c, T]
                    probs = masked_softmax(
                        teacher, valid_t[:, :, None, None, :].expand_as(teacher)
                    )
                    # [b, c, T] — head sum (local heads), staged before the TP reduce
                    mass[:, c0 - g0 : c1 - g0] = probs.sum(dim=(2, 3)).view(b, c, k, r)
                if tp_size > 1:
                    torch.distributed.all_reduce(mass, group=tp_group)
                # Eq.(17) block MaxPool; probs are >= 0 and invalid slots are exactly
                # 0 from the masked softmax, so amax over the r token slots is the max
                # over the valid ones.
                targets[:, g0:g1] = mass.amax(dim=-1).masked_fill(~picked[:, g0:g1], 0.0)
            normalize_indexer_target_(targets)

        # Pass 2 (grad into the indexer): student scores recomputed at the selected
        # blocks only, chunked, against the reduced+normalized teacher.
        kl_sum = torch.zeros((), device=order.device, dtype=torch.float32)
        valid_rows = torch.zeros((), device=order.device, dtype=torch.float32)
        bk_bt = selection.block_keys.permute(1, 0, 2)  # [b, n_blocks_total, d_idx]
        d_idx = bk_bt.shape[-1]
        for c0 in range(0, s, chunk):
            c1 = min(c0 + chunk, s)
            order_c = order[:, c0:c1]
            picked_c = picked[:, c0:c1]
            c = c1 - c0

            # Student: recompute indexer scores at the selected blocks only (with grad).
            blk = order_c
            if selection.block_offsets is not None:
                blk = blk + selection.block_offsets[c0:c1].view(1, c, 1)
            bk_sel = torch.gather(
                bk_bt, 1, blk.reshape(b, c * k)[..., None].expand(-1, -1, d_idx)
            ).view(b, c, k, d_idx)
            q_idx = selection.q.permute(1, 0, 2, 3)[:, c0:c1]  # [b, c, nh, d_idx]
            student = torch.einsum("bchd,bckd->bchk", q_idx.float(), bk_sel.float())
            student = torch.relu(student).sum(dim=2) / math.sqrt(d_idx)  # [b, c, k]
            log_probs = masked_log_softmax(student, picked_c)

            target = targets[:, c0:c1]
            terms = target * (torch.log(target.clamp_min(1e-10)) - log_probs)
            terms = terms.masked_fill(~picked_c, 0.0)
            kl_sum = kl_sum + terms.sum()
            valid_rows = valid_rows + picked_c.any(dim=-1).sum()

        loss = kl_sum / valid_rows.clamp_min(1.0) * loss_coeff
        return loss
