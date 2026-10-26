# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

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
from megatron.core.transformer.cuda_graph_config import cuda_graph_captures_attention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
)
from megatron.core.transformer.experimental_attention_variant.dsa_indexer_loss import (
    normalize_indexer_target_,
)
from megatron.core.transformer.experimental_attention_variant.dsa_layout import (
    build_zigzag_allgather_cp_key_reorder,
    build_zigzag_cp_local_positions,
)
from megatron.core.transformer.experimental_attention_variant.dsa_masking import (
    masked_log_softmax,
    masked_softmax,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import nvtx_decorator


class QSASelection(NamedTuple):
    """Selection artifacts for the sparse attention path and the sparse-KL loss.

    q:          indexer queries ``[s, b, n_heads, head_dim]`` (grad into indexer params).
    block_keys: pooled/normed/RoPE'd block keys ``[n_blocks_total, b, head_dim]`` (grad).
                For packed THD inputs this is the per-document concatenation, padded to
                ``s // compress_ratio`` rows by unused (never visible) tail rows.
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

        # When the CUDA-graph scope captures attention, selection and the sparse-kernel
        # metadata are built with device ops at shape-only upper bounds (no host syncs,
        # no data-dependent shapes). Bit-identical to the dynamic path, which stays the
        # default because it scans only each document's own blocks.
        self.static_shapes = cuda_graph_captures_attention(config)

    @nvtx_decorator(message="QSA::indexer.project")
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

    @nvtx_decorator(message="QSA::indexer.block_keys")
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

    @nvtx_decorator(message="QSA::indexer.select")
    def select(
        self, hidden_states: Tensor, rotary_pos_emb: Tensor, cu_seqlens: Optional[Tensor] = None
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
            attn_bases = torch.arange(nb_attn, device=device, dtype=torch.int32) * r
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
        if self.static_shapes:
            return self._select_thd_static(q_all, raw_keys, rotary_pos_emb, cu_seqlens)
        k = max(self.block_topk, 1)
        q_rot = torch.empty_like(q_all)
        order = torch.zeros(1, s, k, dtype=torch.long, device=device)
        picked = torch.zeros(1, s, k, dtype=torch.bool, device=device)
        own_block = torch.zeros(s, dtype=torch.long, device=device)
        block_offsets = torch.zeros(s, dtype=torch.long, device=device)
        token_offsets = torch.zeros(s, dtype=torch.long, device=device)
        attn_block_offsets = torch.zeros(s, dtype=torch.long, device=device)
        pooled_parts = []
        freq_parts = []
        topk_docs = []
        attn_bases_parts = []
        attn_ends_parts = []
        blocks_so_far = 0
        attn_blocks_so_far = 0
        last_doc_blocks = 0
        # One host sync for all boundaries instead of one per document.
        cu_list = cu_seqlens.tolist()
        for d0 in range(len(cu_list) - 1):
            s0, s1 = int(cu_list[d0]), int(cu_list[d0 + 1])
            sd = s1 - s0
            last_doc_blocks = blocks_so_far
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
                pooled_parts.append(
                    raw_keys[s0 : s0 + n_blocks_d * r]
                    .view(n_blocks_d, r, 1, self.head_dim)
                    .float()
                    .mean(dim=1)
                    .to(raw_keys.dtype)
                )
                freq_parts.append(rotary_pos_emb[: n_blocks_d * r : r])
                topk_docs.append((s0, s1, q_d, m_t_d, n_blocks_d, blocks_so_far))
                blocks_so_far += n_blocks_d
        # Block keys of all documents are normalized and rotated as ONE fixed [s // r]
        # buffer laid out exactly like the static (graph) path's: the documents' blocks,
        # then unused tail rows pooled from the pack's first r tokens at the static path's
        # rotary positions. The tail is never visible to any query. Row-wise the values are
        # the per-document ones, and the k-norm weight gradient (a reduction over rows) sees
        # the same rows in the same order as under CUDA-graph capture, so the eager and
        # captured paths stay bit-identical in the backward as well.
        nb_max = s // r
        if nb_max > 0:
            pad = nb_max - blocks_so_far
            if pad > 0:
                pooled_parts.append(
                    raw_keys[:r]
                    .view(1, r, 1, self.head_dim)
                    .float()
                    .mean(dim=1)
                    .to(raw_keys.dtype)
                    .expand(pad, -1, -1)
                )
                tail_pos = (
                    torch.arange(blocks_so_far, nb_max, device=device) - last_doc_blocks
                ) * r
                freq_parts.append(rotary_pos_emb.index_select(0, tail_pos))
            pooled = self.k_layernorm(torch.cat(pooled_parts, dim=0))
            block_keys = apply_rotary_pos_emb(
                pooled.unsqueeze(2),
                torch.cat(freq_parts, dim=0).contiguous(),
                config=self.config,
                cp_group=self.pg_collection.cp,
            ).squeeze(2)
        else:
            block_keys = raw_keys.new_zeros(0, 1, self.head_dim)
        for s0, s1, q_d, m_t_d, n_blocks_d, off in topk_docs:
            k_d = min(self.block_topk, n_blocks_d)
            order_d, picked_d = self._streaming_topk(
                q_d, block_keys[off : off + n_blocks_d], m_t_d, n_blocks_d, k_d
            )
            order[:, s0:s1, :k_d] = order_d
            picked[:, s0:s1, :k_d] = picked_d
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

    @nvtx_decorator(message="QSA::indexer.select_thd_static")
    def _select_thd_static(
        self, q_all: Tensor, raw_keys: Tensor, rotary_pos_emb: Tensor, cu_seqlens: Tensor
    ) -> QSASelection:
        """Packed-THD selection without host synchronization, at fixed shapes.

        Bit-identical to the per-document loop in :meth:`select`, but every quantity that
        loop derives on the host from the ``cu_seqlens`` values is computed on the device
        and every buffer is sized from shapes alone, so the selection can be captured into
        a CUDA graph and replayed on any packing that fits the captured shapes:

        * per-token document id / position / offsets via ``searchsorted`` over the
          boundaries, and per-document block offsets via ``cumsum``;
        * block keys for all documents in one fixed ``[s // r]`` buffer (a document of
          length ``l`` owns ``l // r`` consecutive rows; the unused tail rows are never
          visible to any query);
        * one top-k scan over that buffer with the visible window ``[block_offset,
          block_offset + own_block)`` per query, at the fixed width ``block_topk``;
          slots beyond a short document's block count stay unpicked, as before;
        * attention-kernel blocks in a fixed ``[s // r + n_docs]`` buffer whose unused
          tail entries are empty token ranges.

        Requires ``cu_seqlens[-1] == s`` (the static THD contract pads the pack to the
        capacity); zero-length documents are allowed anywhere.
        """
        s, b = q_all.shape[0], q_all.shape[1]
        r = self.compress_ratio
        device = q_all.device
        if not torch.cuda.is_current_stream_capturing():
            # Eager-only check of the precondition (a device read; skipped under capture).
            assert int(cu_seqlens[-1]) == s, (
                f"static QSA selection needs the pack to fill the buffer: cu_seqlens[-1]="
                f"{int(cu_seqlens[-1])} != {s}"
            )
        cu = cu_seqlens.to(device=device, dtype=torch.int64)
        n_docs = cu.numel() - 1
        lens = cu[1:] - cu[:-1]
        t = torch.arange(s, device=device, dtype=torch.int64)
        # Document of each token (empty documents are skipped by searchsorted).
        doc = torch.searchsorted(cu[1:].contiguous(), t, right=True).clamp(max=n_docs - 1)
        token_offsets = cu[doc]
        pos = t - token_offsets
        own_block = (pos + 1) // r

        q_rot = self._rope_at_positions(q_all, rotary_pos_emb, pos)

        # Pooled block keys: document d owns rows [blk_off[d], blk_off[d] + lens[d] // r).
        blk_off = torch.nn.functional.pad((lens // r).cumsum(0), (1, 0))
        block_offsets = blk_off[doc]
        nb_max = s // r
        k = max(self.block_topk, 1)
        if nb_max > 0:
            block_keys = self._packed_block_keys(raw_keys, rotary_pos_emb, cu, blk_off, nb_max)
            order, picked = self._streaming_topk(
                q_rot, block_keys, block_offsets + own_block, nb_max, k, m_lo=block_offsets
            )
            # Back to document-local ids (unpicked slots stay 0, as in the loop).
            order = torch.where(picked, order - block_offsets.view(1, s, 1), 0)
        else:
            block_keys = raw_keys.new_zeros(0, b, self.head_dim)
            order = torch.zeros(b, s, k, dtype=torch.long, device=device)
            picked = torch.zeros(b, s, k, dtype=torch.bool, device=device)

        # Attention-kernel block space: a non-empty document owns lens // r + 1 blocks.
        n_attn = torch.where(lens > 0, lens // r + 1, 0)
        attn_off = torch.nn.functional.pad(n_attn.cumsum(0), (1, 0))
        na_max = s // r + n_docs
        aslot = torch.arange(na_max, device=device, dtype=torch.int64)
        adoc = torch.searchsorted(attn_off[1:].contiguous(), aslot, right=True).clamp(
            max=n_docs - 1
        )
        in_use = aslot < attn_off[-1]
        bases = cu[adoc] + (aslot - attn_off[adoc]) * r
        ends = torch.minimum(bases + r, cu[adoc + 1])
        attn_bases = torch.where(in_use, bases, 0).to(torch.int32)
        attn_ends = torch.where(in_use, ends, 0).to(torch.int32)

        return QSASelection(
            q=q_rot,
            block_keys=block_keys,
            order=order,
            picked=picked,
            own_block=own_block,
            block_offsets=block_offsets,
            token_offsets=token_offsets,
            attn_block_offsets=attn_off[doc],
            attn_block_bases=attn_bases,
            attn_block_ends=attn_ends,
        )

    def _packed_block_keys(
        self, raw_keys: Tensor, rotary_pos_emb: Tensor, cu: Tensor, blk_off: Tensor, nb_max: int
    ) -> Tensor:
        """Block keys of a packed pack in one fixed ``[nb_max]`` buffer (no host sync).

        Document ``d`` (token range ``[cu[d], cu[d + 1])``) owns rows
        ``[blk_off[d], blk_off[d + 1])``; each block is pooled from its ``r`` tokens and
        rotated at its first token's document-local position. Rows past ``blk_off[-1]``
        are pooled from the pack's first ``r`` tokens and are never visible to a query.
        """
        r = self.compress_ratio
        b = raw_keys.shape[1]
        device = raw_keys.device
        n_docs = cu.numel() - 1
        slot = torch.arange(nb_max, device=device, dtype=torch.int64)
        bdoc = torch.searchsorted(blk_off[1:].contiguous(), slot, right=True).clamp(max=n_docs - 1)
        j = slot - blk_off[bdoc]
        start = torch.where(slot < blk_off[-1], cu[bdoc] + j * r, 0)
        rows = (start.unsqueeze(1) + torch.arange(r, device=device)).reshape(-1)
        pooled = raw_keys.index_select(0, rows).view(nb_max, r, b, self.head_dim)
        pooled = pooled.float().mean(dim=1).to(raw_keys.dtype)  # [nb_max, b, d]
        pooled = self.k_layernorm(pooled)
        # Each block's RoPE position is its first token's document-local position.
        block_freqs = rotary_pos_emb.index_select(0, j * r).contiguous()
        return apply_rotary_pos_emb(
            pooled.unsqueeze(2), block_freqs, config=self.config, cp_group=self.pg_collection.cp
        ).squeeze(2)

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
        # Packed THD: device-only (graph-capturable) builder; equal to
        # build_packed_allgather_cp_query_positions_and_key_reorder under the contract
        # cu_seqlens[-1] == s_local * cp_size, without its boolean-mask / .item() syncs.
        from megatron.core.transformer.experimental_attention_variant.ops.triton_qsa_select import (  # pylint: disable=line-too-long
            qsa_cp_layout_static,
        )

        return qsa_cp_layout_static(
            cu_seqlens.to(device=device),
            s_local,
            cp_size,
            cp_rank,
            self.config.qsa_cp_packing_layout,
        )

    def _select_cp(
        self, hidden_states: Tensor, rotary_pos_emb: Tensor, cu_seqlens: Optional[Tensor] = None
    ) -> QSASelection:
        """Allgather-CP selection: local queries, globally ordered block keys.

        Only the single shared raw index-key head is all-gathered across CP (its
        backward is a reduce-scatter, so key projections receive gradient from the
        queries on every rank); queries stay local at their zigzag positions.
        Pooling happens strictly AFTER the gather+reorder — pooling rank-local
        neighbours would combine tokens that are not adjacent in the sequence.
        All returned block/token coordinates live in the GLOBAL token space, and
        ``q_positions`` records each local row's global position.

        Host-sync free in every mode (no ``.tolist()``/``.item()``, no per-document
        loop): packed THD inputs use fixed ``[S // r]`` block-key and
        ``[S // r + n_docs]`` attention-block buffers, exactly like
        :meth:`_select_thd_static`, so the layer is CUDA-graph capturable under CP.
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
                q_all, rotary_pos_emb[:s_loc], config=self.config, cp_group=self.pg_collection.cp
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
        # Host-sync free and fixed-shape (CUDA-graph capturable), the CP analogue of
        # _select_thd_static: every quantity is a device op over shape-only sizes. The
        # contract is cu_seqlens[-1] == s_loc * cp_size (the pack fills the buffer).
        s_g = s_loc * cp_size
        cu = cu_seqlens.to(device=device, dtype=torch.int64)
        n_docs = cu.numel() - 1
        lens = cu[1:] - cu[:-1]
        # Positions restart per document; map each local row to its document.
        doc_idx = torch.searchsorted(cu, q_pos, right=True) - 1
        doc_start = cu[doc_idx]
        pos_doc = q_pos - doc_start
        m_t = (pos_doc + 1) // r
        q_rot = self._rope_at_positions(q_all, rotary_pos_emb, pos_doc)

        # Block keys of the GLOBAL pack in one fixed [s_g // r] buffer (same layout as
        # the single-rank static path); one top-k scan with a per-row visible window
        # [block_offset, block_offset + own_block) replaces the per-document loop.
        blk_off = torch.nn.functional.pad((lens // r).cumsum(0), (1, 0))
        block_offsets = blk_off[doc_idx]
        nb_max = s_g // r
        k = max(self.block_topk, 1)
        if nb_max > 0:
            block_keys = self._packed_block_keys(global_keys, rotary_pos_emb, cu, blk_off, nb_max)
            order, picked = self._streaming_topk(
                q_rot, block_keys, block_offsets + m_t, nb_max, k, m_lo=block_offsets
            )
            order = torch.where(picked, order - block_offsets.view(1, s_loc, 1), 0)
        else:
            block_keys = raw_keys.new_zeros(0, b, self.head_dim)
            order = torch.zeros(b, s_loc, k, dtype=torch.long, device=device)
            picked = torch.zeros(b, s_loc, k, dtype=torch.bool, device=device)

        # Attention-kernel block space in GLOBAL token coordinates: a non-empty document
        # owns lens // r + 1 blocks; unused tail entries of the fixed
        # [s_g // r + n_docs] buffer are empty token ranges.
        n_attn = torch.where(lens > 0, lens // r + 1, 0)
        attn_off = torch.nn.functional.pad(n_attn.cumsum(0), (1, 0))
        na_max = s_g // r + n_docs
        aslot = torch.arange(na_max, device=device, dtype=torch.int64)
        adoc = torch.searchsorted(attn_off[1:].contiguous(), aslot, right=True).clamp(
            max=n_docs - 1
        )
        in_use = aslot < attn_off[-1]
        bases = cu[adoc] + (aslot - attn_off[adoc]) * r
        ends = torch.minimum(bases + r, cu[adoc + 1])
        return QSASelection(
            q=q_rot,
            block_keys=block_keys,
            order=order,
            picked=picked,
            own_block=m_t,
            block_offsets=block_offsets,
            token_offsets=doc_start,
            attn_block_offsets=attn_off[doc_idx],
            attn_block_bases=torch.where(in_use, bases, 0).to(torch.int32),
            attn_block_ends=torch.where(in_use, ends, 0).to(torch.int32),
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
    @nvtx_decorator(message="QSA::indexer.topk")
    def _streaming_topk(
        self,
        q: Tensor,
        block_keys: Tensor,
        m_t: Tensor,
        n_blocks: int,
        k: int,
        m_lo: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Chunked K-scan with a deterministic running top-k merge.

        Scores each chunk of blocks (fp32 relu-sum-heads), masks causal visibility,
        and merges into the running top-k under the (score desc, block_id asc)
        lexicographic order — realized as a stable ascending sort by block id
        followed by a stable descending sort by score.

        Block ``j`` is visible to query ``t`` iff ``m_lo[t] <= j < m_t[t]`` (``m_lo``
        defaults to 0). A per-query lower bound lets packed documents share one
        block-key buffer: shifting every visible id of a query by the same offset
        preserves the (score desc, block_id asc) order, so the selection is the same
        as scoring that document's blocks alone.

        Returns (order ``[b, s, k]``, picked ``[b, s, k]``); ``order`` holds the ids of
        ``block_keys`` rows.
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

        if q.is_cuda:
            # K4: fp32 scores (the IEEE-dot arithmetic of this merge, with a causal tile
            # skip) and an exact radix top-k -- Triton below 4096 blocks per row, the cuDNN
            # FE k-th value above -- with the same unique-key tie-break: bit-identical keys
            # to the merge below without the [s, chunk] int64 keys and torch.topk.
            from megatron.core.transformer.experimental_attention_variant.ops.triton_qsa_select import (  # pylint: disable=line-too-long
                qsa_select_topk_keys,
            )

            run_keys = torch.stack(
                [
                    qsa_select_topk_keys(
                        q[:, bi].contiguous(),
                        block_keys[:n_blocks, bi].contiguous(),
                        m_t,
                        k,
                        m_lo=m_lo,
                        impl="auto",
                    )
                    for bi in range(b)
                ]
            )
        else:
            run_keys = torch.zeros((b, s, k), dtype=torch.int64, device=device)
            for j0 in range(0, n_blocks, chunk):
                j1 = min(j0 + chunk, n_blocks)
                sc = torch.einsum("sbhd,nbd->bshn", q.float(), block_keys[j0:j1].float())
                sc = torch.relu(sc).sum(dim=2) / math.sqrt(self.head_dim)  # [b, s, w]
                ids = torch.arange(j0, j1, device=device)
                vis = ids.unsqueeze(0) < m_t.unsqueeze(1)  # [s, w]
                if m_lo is not None:
                    vis = vis & (ids.unsqueeze(0) >= m_lo.unsqueeze(1))
                bits = sc.view(torch.int32).to(torch.int64)
                key = ((bits + 1) << ID_BITS) | (id_code_max - ids)
                key = torch.where(vis.unsqueeze(0), key, torch.zeros_like(key))
                cand = torch.cat([run_keys, key], dim=-1)
                run_keys = cand.topk(k, dim=-1).values  # descending; valid keys are unique

        picked = run_keys > 0
        order = id_code_max - (run_keys & id_code_max)
        order = torch.where(picked, order, torch.zeros_like(order))
        return order, picked


def _require_qsa_kernel_support(config: TransformerConfig, tp_size: int) -> None:
    """The QSA kernels (sparse attention, fused indexer loss) are CuTe DSL kernels for
    Blackwell (SM100-class) GPUs.

    Checked once, when a QSA layer that uses them is built, so that an unsupported device or
    shape fails with its cause instead of inside a kernel launch.
    """
    sparse = config.qsa_use_sparse_attention
    fused_loss = config.qsa_indexer_loss_impl == "fused" and bool(config.qsa_indexer_loss_coeff)
    if not (sparse or fused_loss):
        return
    what = "qsa_use_sparse_attention" if sparse else "qsa_indexer_loss_impl='fused'"
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        raise RuntimeError(
            f"{what} requires a Blackwell (SM100-class: B200, GB200, B300, GB300) GPU; the "
            "QSA kernels are CuTe DSL tcgen05 kernels."
        )
    try:
        import cutlass  # noqa: F401  # pylint: disable=unused-import
    except ImportError as e:
        raise RuntimeError(f"{what} requires nvidia-cutlass-dsl (CuTe DSL).") from e
    if fused_loss:
        from megatron.core.transformer.experimental_attention_variant.ops.cute_qsa_indexer_loss import (  # pylint: disable=line-too-long
            teacher_tmem_columns,
        )

        r = config.qsa_indexer_compress_ratio
        ng = max(1, config.num_query_groups // tp_size)
        hpg = config.num_attention_heads // config.num_query_groups
        n_sel = config.qsa_indexer_budget // r
        if not config.bf16:
            raise ValueError("qsa_indexer_loss_impl='fused' requires bf16 training.")
        if config.kv_channels % 64 or 128 % r or hpg > 256 or ng * (config.kv_channels // 64) < 2:
            raise ValueError(
                "qsa_indexer_loss_impl='fused' requires a head dim that is a multiple of 64, "
                "a compress ratio dividing 128, at most 256 query heads per KV head and "
                f"(KV heads per TP rank) x (head dim / 64) >= 2; got kv_channels="
                f"{config.kv_channels}, ratio={r}, heads per group={hpg}, KV heads={ng}. "
                "Use qsa_indexer_loss_impl='layout'."
            )
        d_idx = config.qsa_indexer_head_dim
        if config.qsa_indexer_n_heads > 8 or d_idx % 8 or d_idx > 256 or n_sel > 2048:
            raise ValueError(
                "qsa_indexer_loss_impl='fused' (sparse student kernel) supports at most 8 "
                "indexer heads, an indexer head dim that is a multiple of 8 and at most 256, "
                f"and at most 2048 selected blocks; got {config.qsa_indexer_n_heads} heads, "
                f"head dim {d_idx}, {n_sel} blocks. Use qsa_indexer_loss_impl='layout'."
            )
        if teacher_tmem_columns(n_sel, r, ng, hpg) > 512:
            raise ValueError(
                f"qsa_indexer_loss_impl='fused': the teacher logits of {n_sel} selected blocks "
                f"x {ng} KV heads x {hpg} heads per group do not fit in tensor memory. Use "
                "qsa_indexer_loss_impl='layout'."
            )
    if not sparse:
        return
    if config.qsa_attn_backend == "cudnn":
        _require_qsa_cudnn_support(config, tp_size)
        return
    if not (config.bf16 or config.fp16):
        raise ValueError("qsa_use_sparse_attention requires bf16 or fp16 training.")
    if config.kv_channels not in (128, 256):
        raise ValueError(
            f"qsa_use_sparse_attention supports head dims 128 and 256, got "
            f"kv_channels={config.kv_channels}."
        )
    group = config.num_attention_heads // config.num_query_groups
    if group > 128:
        raise ValueError(
            f"qsa_use_sparse_attention supports at most 128 query heads per KV head, got {group}."
        )
    if config.kv_channels == 256 and group == 1:
        # The dq kernel's 128-query tile at G = 1 needs 1 KB more shared memory than SM100 has.
        raise ValueError(
            "qsa_use_sparse_attention with head dim 256 needs at least 2 query heads per KV head."
        )


def _require_qsa_cudnn_support(config: TransformerConfig, tp_size: int) -> None:
    """``qsa_attn_backend='cudnn'``: cuDNN Frontend's QSA kernels serve one fixed geometry.

    Checked when the layer is built, so an unsupported shape fails with its cause instead of
    inside cuDNN. The device itself was checked by :func:`_require_qsa_kernel_support`.
    """
    from megatron.core.transformer.experimental_attention_variant.ops.cudnn_qsa_attn import (
        cudnn_qsa_support_error,
    )

    try:
        import cudnn.ops as cudnn_ops

        cudnn_ops.qsa_forward, cudnn_ops.qsa_backward  # pylint: disable=pointless-statement
        if config.qsa_cudnn_bwd_impl == "sparse_gqa":
            from cudnn.sdpa.bwd import sparse_gqa_bwd_wrapper_sm100  # noqa: F401
    except (ImportError, AttributeError) as e:
        raise RuntimeError(
            "qsa_attn_backend='cudnn' requires a cuDNN Frontend build with the QSA training "
            "kernels (cudnn.ops.qsa_forward / qsa_backward, NVIDIA/cudnn-frontend PR #1765) "
            "and, for qsa_cudnn_bwd_impl='sparse_gqa', cudnn.sdpa.bwd.sparse_gqa_bwd_wrapper_sm100."
        ) from e
    num_q_heads = config.num_attention_heads // tp_size
    num_kv_heads = max(1, config.num_query_groups // tp_size)
    dtype = torch.bfloat16 if config.bf16 else (torch.float16 if config.fp16 else torch.float32)
    reason = cudnn_qsa_support_error(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=config.kv_channels,
        dtype=dtype,
        block_size=config.qsa_indexer_compress_ratio,
        num_selected_blocks=config.qsa_indexer_budget // config.qsa_indexer_compress_ratio,
        bwd_impl=config.qsa_cudnn_bwd_impl,
        device_capability=(
            tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None
        ),
    )
    if reason is not None:
        raise ValueError(f"qsa_attn_backend='cudnn': {reason}.")


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
        # The indexer's only gradient source is the sparse-KL loss (its input is detached
        # and top-k selection is not differentiable). Without that loss its parameters
        # never receive gradients, so freeze them: otherwise DDP still registers them in
        # its buckets and overlap_grad_reduce asserts at the start of the second step,
        # and the optimizer would carry state for parameters it never updates.
        if not self.config.qsa_indexer_loss_coeff:
            for param in self.indexer.parameters():
                param.requires_grad_(False)
        _require_qsa_kernel_support(self.config, self.pg_collection.tp.size())

    def is_cuda_graph_capturable(self) -> bool:
        """QSA layers are capturable, with or without (allgather) context parallelism.

        With attention in the capture scope the indexer runs its static-shape path
        (:meth:`QSAIndexer._select_thd_static`, or the always host-sync-free
        :meth:`QSAIndexer._select_cp` under CP) and the sparse kernels build their
        fixed-capacity metadata.
        """
        return True

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
        assert (
            inference_context is None and kwargs.get("inference_params") is None
        ), "QSA does not implement an inference path (pretrain/continue-pretrain scope)."
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
            output, bias, selection = self._sparse_forward(hidden_states, freqs, packed_seq_params)
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

    @nvtx_decorator(message="QSA::rope_qk")
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
        query = apply_rotary_pos_emb(query, f, config=self.config, cp_group=self.pg_collection.cp)
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

    @nvtx_decorator(message="QSA::sparse_attn")
    def _sparse_forward(self, hidden_states, freqs, packed_seq_params):
        """Sparse kernel path: indexer selection -> sparse GQA over the selected blocks.

        The attention op is the in-tree CuTe DSL query-tile shared-superset kernel set, or
        cuDNN Frontend's QSA kernels with ``qsa_attn_backend='cudnn'``.

        Supports BSHD and packed THD (``packed_seq_params.qkv_format == 'thd'``).
        The kernels take global block ids plus per-block token [base, end) ranges,
        so packed varlen needs no cu_seqlens on the kernel side, and any GQA group
        size works without head padding.
        """
        from megatron.core.transformer.experimental_attention_variant.ops.cudnn_qsa_attn import (
            cudnn_qsa_sparse_attention,
        )
        from megatron.core.transformer.experimental_attention_variant.ops.cute_qsa_attn import (
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
            assert (
                cu_kv is None or cu_kv is cu_seqlens or cu_kv.shape == cu_seqlens.shape
            ), "QSA is self-attention only (cu_seqlens_q must equal cu_seqlens_kv)."

        # THD inputs arrive as [t, 1, h]; SelfAttention would squeeze them later, we
        # keep the [s, b, ...] convention throughout and reshape for the kernel.
        selection = self.indexer.select(hidden_states, freqs, cu_seqlens=cu_seqlens)

        qkv = self.get_query_key_value_tensors(
            hidden_states, None, split_qkv=True, output_gate=self.config.attention_output_gate
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
        if self._indexer_teacher_reuses_main_qk():
            # The fused teacher reuses these (bit-identical to its own re-projection).
            self._indexer_main_qk = (q_bt.detach(), k_bt.detach())

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

        if self.config.qsa_attn_backend == "cudnn":
            core_out = cudnn_qsa_sparse_attention(
                q_bt,
                k_bt,
                v_bt,
                block_indices=block_indices,
                block_counts=block_counts,
                block_bases=selection.attn_block_bases,
                block_ends=selection.attn_block_ends,
                block_offsets=selection.attn_block_offsets,
                token_offsets=selection.token_offsets,
                cu_seqlens=cu_seqlens,
                scale=self.hidden_size_per_attention_head**-0.5,
                bwd_impl=self.config.qsa_cudnn_bwd_impl,
            )  # [b, s, np, d_v]
        else:
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
                static_shapes=self.indexer.static_shapes,
            )  # [b, s, np, d_v]
        core_out = core_out.reshape(b, s, np_local * core_out.shape[-1]).permute(1, 0, 2)

        if gate is not None:
            core_out = self._apply_output_gate(core_out, gate)
        output, bias = self.linear_proj(core_out)
        return output, bias, selection

    @nvtx_decorator(message="QSA::indexer_loss")
    def _compute_indexer_loss(
        self,
        hidden_states: Tensor,
        freqs: Tensor,
        selection: QSASelection,
        loss_coeff: float,
        cu_seqlens: Optional[Tensor] = None,
        teacher_qk: Optional[Tuple[Tensor, Tensor]] = None,
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

        ``config.qsa_indexer_loss_impl`` selects how the same loss is computed:

        * ``"reference"``: the original eager path (gathers the selected keys and the
          selected block keys, einsum per chunk).
        * ``"layout"``: identical math and rounding, without materializing the
          gathered keys: per chunk, one GEMM of the chunk's queries against the keys of
          its causal key range, then a gather of the bf16 logits at the selected tokens
          (no ``[b, chunk, budget, groups, d]`` key tensor, no einsum transpose copy).
        * ``"fused"``: the CuTe DSL teacher kernel (``ops/cute_qsa_indexer_loss.py``,
          Blackwell) on the main path's detached q/k (``teacher_qk``, or the pair stashed
          by :meth:`_sparse_forward`; no QKV projection or RoPE recompute), gathering the
          selected keys once, and the CuTe DSL sparse student: scores, KL and gradients at
          the selected blocks only, gradients computed in the forward (deterministic, no
          atomics, no GEMM over the chunk's whole causal block range).
        """
        impl = self.config.qsa_indexer_loss_impl
        r = self.indexer.compress_ratio
        order, picked = selection.order, selection.picked  # [b, s, k]
        b, s, k = order.shape
        chunk = max(1, int(self.config.qsa_indexer_loss_chunk))
        # Without CP every selected block of a query lies before it, so a chunk ending at
        # row c1 only reads blocks [0, ceil(c1 / r)). Under allgather CP the local rows
        # are not a prefix of the global token space.
        causal_prefix = selection.q_positions is None

        main_qk, self._indexer_main_qk = getattr(self, "_indexer_main_qk", None), None
        if teacher_qk is None and impl == "fused":
            teacher_qk = main_qk
        if teacher_qk is not None:
            q_bt, k_bt = teacher_qk[0].detach(), teacher_qk[1].detach()
        else:
            q_bt, k_bt = self._indexer_teacher_qk(hidden_states, freqs, selection, cu_seqlens)
        targets = self._indexer_targets(q_bt, k_bt, selection, impl)
        del q_bt, k_bt

        if impl == "fused":
            from megatron.core.transformer.experimental_attention_variant.ops.cute_qsa_indexer_loss import (  # pylint: disable=line-too-long
                qsa_indexer_student_sparse,
            )

            n_blocks_total = selection.block_keys.shape[0]
            return qsa_indexer_student_sparse(
                selection.q.permute(1, 0, 2, 3),
                selection.block_keys.permute(1, 0, 2),
                order,
                picked,
                targets,
                loss_coeff,
                block_offsets=selection.block_offsets,
                chunk=chunk,
                nb_hi_fn=(
                    (lambda c1: min(n_blocks_total, -(-c1 // r)))
                    if causal_prefix
                    else (lambda c1: n_blocks_total)
                ),
            )

        # Pass 2 (grad into the indexer): student scores recomputed at the selected
        # blocks only, chunked, against the reduced+normalized teacher.
        kl_sum = torch.zeros((), device=order.device, dtype=torch.float32)
        valid_rows = torch.zeros((), device=order.device, dtype=torch.float32)
        bk_bt = selection.block_keys.permute(1, 0, 2)  # [b, n_blocks_total, d_idx]
        d_idx = bk_bt.shape[-1]
        q_idx_all = selection.q.permute(1, 0, 2, 3)  # [b, s, nh, d_idx]
        for c0 in range(0, s, chunk):
            c1 = min(c0 + chunk, s)
            order_c = order[:, c0:c1]
            picked_c = picked[:, c0:c1]
            c = c1 - c0

            # Student: recompute indexer scores at the selected blocks only (with grad).
            blk = order_c
            if selection.block_offsets is not None:
                blk = blk + selection.block_offsets[c0:c1].view(1, c, 1)
            q_idx = q_idx_all[:, c0:c1]  # [b, c, nh, d_idx]
            bk_sel = torch.gather(
                bk_bt, 1, blk.reshape(b, c * k)[..., None].expand(-1, -1, d_idx)
            ).view(b, c, k, d_idx)
            # Same policy as the teacher: bf16 operands, fp32 accumulation inside the
            # matmul, promoted before the relu/head-sum so the reduction and the
            # log-softmax below are unchanged. This one carries the indexer gradient,
            # so the promotion happens before any nonlinearity.
            student = torch.einsum("bchd,bckd->bchk", q_idx, bk_sel).float()
            student = torch.relu(student).sum(dim=2) / math.sqrt(d_idx)  # [b, c, k]
            log_probs = masked_log_softmax(student, picked_c)

            target = targets[:, c0:c1]
            terms = target * (torch.log(target.clamp_min(1e-10)) - log_probs)
            terms = terms.masked_fill(~picked_c, 0.0)
            kl_sum = kl_sum + terms.sum()
            valid_rows = valid_rows + picked_c.any(dim=-1).sum()

        loss = kl_sum / valid_rows.clamp_min(1.0) * loss_coeff
        return loss

    def _indexer_teacher_reuses_main_qk(self) -> bool:
        """True when the coming indexer loss will consume the main path's q/k."""
        return (
            self.training
            and (self.config.qsa_indexer_loss_coeff or 0.0) > 0
            and torch.is_grad_enabled()
            and self.config.qsa_indexer_loss_impl == "fused"
        )

    @torch.no_grad()
    def _indexer_teacher_qk(
        self,
        hidden_states: Tensor,
        freqs: Tensor,
        selection: QSASelection,
        cu_seqlens: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Detached teacher q/k ``[b, s, np, d]`` / ``[b, s_kv, ng, d]``: re-run the QKV
        projection (+ qk norms inside) and RoPE."""
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
        return query.permute(1, 0, 2, 3), key.permute(1, 0, 2, 3)

    @torch.no_grad()
    def _indexer_targets(
        self, q_bt: Tensor, k_bt: Tensor, selection: QSASelection, impl: str
    ) -> Tensor:
        """L1-normalized teacher targets ``[b, s, k]`` (fp32) with the chosen impl."""
        scale = self.hidden_size_per_attention_head**-0.5
        tp_group = self.pg_collection.tp
        tp_size = tp_group.size() if tp_group is not None else 1
        if impl == "fused":
            return self._indexer_targets_fused(q_bt, k_bt, selection, scale, tp_group, tp_size)
        # Query-chunk size: pure memory staging. The softmax is over the last (token) axis
        # and `mass` accumulates per chunk, so the chunk size cannot change any result - it
        # only trades peak memory for kernel launches.
        chunk = max(1, int(self.config.qsa_indexer_loss_chunk))
        return self._indexer_targets_chunked(
            q_bt,
            k_bt,
            selection,
            scale,
            tp_group,
            tp_size,
            chunk,
            layout=(impl == "layout"),
            causal_prefix=selection.q_positions is None,
        )

    @torch.no_grad()
    def _indexer_targets_chunked(
        self,
        q_bt: Tensor,
        k_bt: Tensor,
        selection: QSASelection,
        scale: float,
        tp_group,
        tp_size: int,
        chunk: int,
        layout: bool,
        causal_prefix: bool,
    ) -> Tensor:
        """Eager teacher targets ``[b, s, k]`` (normalized), chunked over queries.

        ``layout=False`` is the original reference (gather the selected keys, einsum).
        ``layout=True`` produces the same bf16 logits from one GEMM of the chunk's
        queries against the keys of its causal range followed by a gather at the selected
        tokens; everything after the logits is the same code.
        """
        r = self.indexer.compress_ratio
        order, picked = selection.order, selection.picked  # [b, s, k]
        b, s, k = order.shape
        np_local, ng_local = q_bt.shape[2], k_bt.shape[2]
        hpg = np_local // ng_local
        d = q_bt.shape[-1]
        s_kv = k_bt.shape[1]
        arange_r = torch.arange(r, device=order.device)
        if layout:
            # Group-major contiguous operands, built once per call (queries ~s*np*d,
            # keys ~s*ng*d): every chunk's GEMM operands are then plain views.
            q_g = q_bt.reshape(b, s, ng_local, hpg, d).permute(0, 2, 1, 3, 4).contiguous()
            k_g = k_bt.permute(0, 2, 1, 3).contiguous()  # [b, ng, s_kv, d]

        # Pass 1 (no grad): local-head teacher, block-max-pooled into a small
        # [b, s, k] fp32 buffer. The Eq.(17) MaxPool is nonlinear, so the TP head-sum
        # must complete BEFORE pooling (deferring a single block-level reduce would
        # be exact only for sum pooling) — but the reduce granularity is free: the
        # token-level mass is staged over several einsum chunks and all-reduced per
        # stage, keeping the collective count low without growing the einsum tile.
        # Normalization still commutes (it happens after the reduce).
        targets = torch.zeros(b, s, k, dtype=torch.float32, device=order.device)
        stage = chunk * 8
        for g0 in range(0, s, stage):
            g1 = min(g0 + stage, s)
            mass = torch.zeros(b, g1 - g0, k, r, dtype=torch.float32, device=order.device)
            for c0 in range(g0, g1, chunk):
                c1 = min(c0 + chunk, g1)
                order_c = order[:, c0:c1]  # [b, c, k]
                picked_c = picked[:, c0:c1]  # [b, c, k]
                c = c1 - c0

                tok = order_c.unsqueeze(-1) * r + arange_r  # [b, c, k, r] doc-local
                if selection.token_offsets is not None:
                    tok = tok + selection.token_offsets[c0:c1].view(1, c, 1, 1)
                tok = tok.reshape(b, c * k * r)
                if layout:
                    kv_hi = min(s_kv, c1) if causal_prefix else s_kv
                    # Unpicked slots may point past the causal range; their logits are
                    # masked below, so any in-range index is fine.
                    tok = tok.clamp_max(kv_hi - 1)
                    scores = torch.bmm(
                        q_g[:, :, c0:c1].reshape(b * ng_local, c * hpg, d),
                        k_g[:, :, :kv_hi].reshape(b * ng_local, kv_hi, d).transpose(1, 2),
                    ).view(b, ng_local, c, hpg, kv_hi)
                    logits = torch.gather(
                        scores.permute(0, 2, 1, 3, 4),
                        4,
                        tok.view(b, c, 1, 1, k * r).expand(b, c, ng_local, hpg, k * r),
                    ).contiguous()  # [b, c, g, h, T] bf16
                    teacher = logits.float() * scale
                else:
                    k_sel = torch.gather(
                        k_bt, 1, tok[..., None, None].expand(-1, -1, ng_local, d)
                    ).view(b, c, k * r, ng_local, d)

                    q_c = q_bt[:, c0:c1].view(b, c, ng_local, hpg, d)
                    # Operands stay in the model dtype; cuBLAS accumulates bf16 matmuls
                    # in fp32, and the result is promoted before the softmax so every
                    # reduction below is still fp32. Upcasting the operands instead
                    # doubled the tensor traffic of the two gathers above and pushed this
                    # matmul onto TF32.
                    teacher = (
                        torch.einsum("bcghd,bctgd->bcght", q_c, k_sel).float() * scale
                    )  # [b, c, g, h, T]
                valid_t = picked_c.repeat_interleave(r, dim=-1)  # [b, c, T]
                probs = masked_softmax(teacher, valid_t[:, :, None, None, :].expand_as(teacher))
                # [b, c, T] — head sum (local heads), staged before the TP reduce
                mass[:, c0 - g0 : c1 - g0] = probs.sum(dim=(2, 3)).view(b, c, k, r)
            if tp_size > 1:
                torch.distributed.all_reduce(mass, group=tp_group)
            # Eq.(17) block MaxPool; probs are >= 0 and invalid slots are exactly
            # 0 from the masked softmax, so amax over the r token slots is the max
            # over the valid ones.
            targets[:, g0:g1] = mass.amax(dim=-1).masked_fill(~picked[:, g0:g1], 0.0)
        normalize_indexer_target_(targets)
        return targets

    @torch.no_grad()
    def _indexer_targets_fused(
        self,
        q_bt: Tensor,
        k_bt: Tensor,
        selection: QSASelection,
        scale: float,
        tp_group,
        tp_size: int,
    ) -> Tensor:
        """Teacher targets ``[b, s, k]`` (normalized) from the CuTe DSL teacher kernel."""
        from megatron.core.transformer.experimental_attention_variant.ops.cute_qsa_indexer_loss import (  # pylint: disable=line-too-long
            qsa_indexer_teacher,
        )

        picked = selection.picked
        pool = tp_size == 1
        out = qsa_indexer_teacher(
            q_bt,
            k_bt,
            selection.order,
            picked,
            compress_ratio=self.indexer.compress_ratio,
            scale=scale,
            token_offsets=selection.token_offsets,
            pool=pool,
        )
        if pool:
            targets = out
        else:
            # Token-level head mass is summed over TP ranks before the block max-pool.
            torch.distributed.all_reduce(out, group=tp_group)
            targets = out.amax(dim=-1).masked_fill(~picked, 0.0)
        normalize_indexer_target_(targets)
        return targets
