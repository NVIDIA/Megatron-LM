# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Qwen Sparse Attention (QSA) of Qwen4-Exp / Qwen3.8-Flash-Next.

QSA is grouped-query softmax attention whose key/value set is chosen per query by a lightweight
*indexer*:

* the indexer projects the layer input to ``n_heads`` query heads and one key head of width
  ``head_dim`` (``index_qk_proj``), RMS-normalizes them and applies the attention RoPE;
* keys are mean-pooled over ``compress_ratio`` consecutive tokens (blocks aligned to the start
  of every sequence) *before* the norm and RoPE (the block is rotated with the position of its
  first token);
* every query scores the complete blocks it can see with ``sum_h relu(q_h . k_block) / sqrt(d)``
  and keeps the ``budget / compress_ratio`` best ones; the tokens of those blocks plus the tokens
  of its own, still incomplete, block form the attention set.

The selection is a hard top-k, so no gradient flows into the indexer from the language-model
loss (as in the reference implementation). The attention itself is exact softmax attention over
the selected tokens and is differentiable w.r.t. Q, K and V.

Sequences with at most ``budget + compress_ratio - 1`` tokens select every visible token, so
plain causal attention is exact for them and the dense core attention is used. Longer sequences
run a block-sparse FlexAttention kernel driven by the per-query block selection.
"""

from __future__ import annotations

import functools
import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from megatron.core.models.common.embeddings.rope_utils import (
    _apply_rotary_pos_emb_bshd,
    _rotate_half,
    apply_rotary_pos_emb,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_pg_size

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    HAVE_FLEX_ATTENTION = True
except ImportError:  # pragma: no cover
    HAVE_FLEX_ATTENTION = False

_QSA_SELECT_TILE_BYTES = 64 << 20


@dataclass
class QSASelection:
    """Per-query block selection produced by :class:`QSAIndexer`.

    All tensors are indexed by the flattened token position of the (full, gathered) sequence.
    Positions and block ids are relative to the start of the token's own sequence (document).

    Attributes:
        doc_ids: [b, s] int32 document id of every token (packed sequences) or the batch row.
        positions: [b, s] int32 position of every token inside its document.
        selected_bits: Flat uint8 bitset ``[b * s * nbytes]`` over document-relative block ids:
            for query ``(b, q)`` bit ``j`` of byte ``(b * s + q) * nbytes + i`` is set when block
            ``8 * i + j`` is selected. Flat so that the FlexAttention mask kernel indexes a 1-D
            buffer (its row stride is passed as a tensor scalar to avoid recompiling per shape).
        bits_per_row: ``nbytes`` (python int).
        bits_per_row_t: ``nbytes`` as a 0-dim int64 tensor for the mask kernel.
        compress_ratio: Tokens per block.
        all_selected: True when every query selected all of its visible complete blocks, i.e.
            the attention pattern is plain causal attention.
    """

    doc_ids: Tensor
    positions: Tensor
    selected_bits: Tensor
    bits_per_row: int
    bits_per_row_t: Tensor
    compress_ratio: int
    all_selected: bool


@dataclass
class QSAIndexerSubmodules:
    """Submodules of the QSA indexer."""

    linear_qk: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    k_layernorm: Union[ModuleSpec, type] = None


@dataclass
class QwenSparseSelfAttentionSubmodules(SelfAttentionSubmodules):
    """Self-attention submodules plus the QSA indexer."""

    indexer: Union[ModuleSpec, type] = None


def _sequence_layout(
    batch_size: int, seq_len: int, packed_seq_params: Optional[PackedSeqParams], device
) -> Tuple[Tensor, Tensor, int]:
    """Document ids and in-document positions for every token.

    Returns ``(doc_ids [b, s], positions [b, s], max_doc_len)`` as int32 tensors.
    """
    if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
        assert batch_size == 1, "THD packing flattens the batch into a single row"
        cu = packed_seq_params.cu_seqlens_q
        if packed_seq_params.cu_seqlens_q_padded is not None:
            cu = packed_seq_params.cu_seqlens_q_padded
        cu = cu.to(device=device, dtype=torch.long)
        starts = cu[:-1]
        arange = torch.arange(seq_len, device=device, dtype=torch.long)
        doc = torch.searchsorted(starts, arange, right=True) - 1
        doc = doc.clamp_min(0)
        positions = arange - starts[doc]
        max_doc_len = int((cu[1:] - cu[:-1]).max().item())
        return doc.unsqueeze(0).to(torch.int32), positions.unsqueeze(0).to(torch.int32), max_doc_len
    arange = torch.arange(seq_len, device=device, dtype=torch.int32)
    positions = arange.unsqueeze(0).expand(batch_size, -1).contiguous()
    doc = (
        torch.arange(batch_size, device=device, dtype=torch.int32)
        .unsqueeze(1)
        .expand(batch_size, seq_len)
        .contiguous()
    )
    return doc, positions, seq_len


def _full_sequence_indexer_rope(
    q: Tensor,
    freqs: Tensor,
    positions: Tensor,
    is_thd: bool,
    is_absolute_mrope: bool,
    rotary_interleaved: bool,
) -> Tensor:
    """Rotate CP-reconstructed queries without remapping them as CP-local tokens."""
    if is_thd and not is_absolute_mrope:
        # Standard packed RoPE resets at each document boundary. VL mRoPE already
        # provides one frequency per token in the reconstructed physical order.
        freqs = freqs.index_select(0, positions.reshape(-1).long())
    return _apply_rotary_pos_emb_bshd(q, freqs, rotary_interleaved=rotary_interleaved)


class QSAIndexer(MegatronModule):
    """Weight-light MQA indexer selecting the key blocks every query attends to.

    Parameters are duplicated across tensor-parallel ranks (the projection is tiny); the indexer
    operates on the full, sequence-parallel-gathered sequence because the selection needs every
    key of the sequence.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: QSAIndexerSubmodules,
        layer_number: int,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(config)
        self.config = config
        self.layer_number = layer_number
        self.n_heads = config.qsa_indexer_n_heads
        self.kv_heads = config.qsa_indexer_kv_heads
        self.head_dim = config.qsa_indexer_head_dim
        self.token_budget = config.qsa_indexer_budget
        self.compress_ratio = config.qsa_indexer_compress_ratio
        self.block_topk = self.token_budget // self.compress_ratio
        self.rotary_interleaved = config.rotary_interleaved
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        self.pg_collection = pg_collection
        # Context parallelism is supported via an all-gather of the hidden (and rotary)
        # tensors in ``forward`` (see ``reconstruct_tensor_cp``): every CP rank rebuilds the
        # full sequence so the block selection sees every key, mirroring the CP=1 path. Only
        # the all-gather CP comm type is supported; ring/p2p cannot expose every key.

        self.index_qk_proj = build_module(
            submodules.linear_qk,
            config.hidden_size,
            (self.n_heads + self.kv_heads) * self.head_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        self.q_layernorm = build_module(
            submodules.q_layernorm,
            config=config,
            hidden_size=self.head_dim,
            eps=config.layernorm_epsilon,
        )
        self.k_layernorm = build_module(
            submodules.k_layernorm,
            config=config,
            hidden_size=self.head_dim,
            eps=config.layernorm_epsilon,
        )
        # Every TP rank computes the identical full-sequence indexer, so duplicated-parameter
        # gradients are averaged (not summed) across the TP domain.
        for param in self.parameters():
            setattr(param, "average_gradients_across_tp_domain", True)

    # ------------------------------------------------------------------ helpers
    def _rope_at_positions(
        self, x: Tensor, freqs: Tensor, positions: Tensor, batch_indices: Optional[Tensor] = None
    ) -> Tensor:
        """Rotate ``x`` [N, D] with the frequencies of ``positions`` [N] (first ``rot_dim``
        dims)."""
        if freqs.shape[1] > 1:
            assert batch_indices is not None, "batched mRoPE requires per-key batch indices"
            f = freqs[positions.long(), batch_indices.long(), 0]  # [N, rot_dim]
        else:
            f = freqs.reshape(freqs.shape[0], -1)[positions.long()]
        rot_dim = f.shape[-1]
        x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
        cos_ = torch.cos(f).to(x.dtype)
        sin_ = torch.sin(f).to(x.dtype)
        x4 = x_rot.view(x_rot.shape[0], 1, 1, rot_dim)
        rotated = x4 * cos_.view(-1, 1, 1, rot_dim) + _rotate_half(
            x4, self.rotary_interleaved
        ) * sin_.view(-1, 1, 1, rot_dim)
        return torch.cat([rotated.view_as(x_rot), x_pass], dim=-1)

    def _pool_keys(
        self, raw_keys: Tensor, doc_ids: Tensor, positions: Tensor, max_doc_len: int
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Mean-pool complete blocks of ``compress_ratio`` consecutive tokens per document.

        Args:
            raw_keys: [T, D] key of every token (flattened batch).
            doc_ids / positions: [T] int32.

        Returns:
            ``(pooled [n_docs, n_blocks_max, D], block_positions [n_docs, n_blocks_max],
            block_valid [n_docs, n_blocks_max])``. Only complete blocks are valid.
        """
        R = self.compress_ratio
        n_docs = int(doc_ids.max().item()) + 1
        n_blocks_max = max_doc_len // R
        D = raw_keys.shape[-1]
        block_of_token = positions.long() // R
        in_block = block_of_token < n_blocks_max  # tokens of the incomplete tail are dropped
        # A block is complete when its document has >= (block+1)*R tokens.
        doc_lens = torch.zeros(n_docs, dtype=torch.long, device=raw_keys.device)
        doc_lens.scatter_add_(0, doc_ids.long(), torch.ones_like(doc_ids, dtype=torch.long))
        blocks_per_doc = doc_lens // R
        block_valid = torch.arange(n_blocks_max, device=raw_keys.device).unsqueeze(
            0
        ) < blocks_per_doc.unsqueeze(1)
        flat_index = (doc_ids.long() * n_blocks_max + block_of_token)[in_block]
        pooled = torch.zeros(n_docs * n_blocks_max, D, dtype=torch.float32, device=raw_keys.device)
        pooled.index_add_(0, flat_index, raw_keys[in_block].float())
        pooled = (pooled / R).to(raw_keys.dtype).view(n_docs, n_blocks_max, D)
        pooled = pooled * block_valid.unsqueeze(-1).to(pooled.dtype)
        block_positions = (
            (torch.arange(n_blocks_max, device=raw_keys.device, dtype=torch.long) * R)
            .unsqueeze(0)
            .expand(n_docs, -1)
        )
        return pooled, block_positions, block_valid

    @torch.no_grad()
    def _select_blocks(
        self,
        q: Tensor,
        pooled_keys: Tensor,
        block_valid: Tensor,
        doc_ids: Tensor,
        positions: Tensor,
        uniform_doc_len: Optional[int] = None,
        output_format: str = "bits",
    ) -> Tuple[Tensor, bool]:
        """Top-k complete blocks per query.

        Args:
            q: [T, H, D] normalized + rotated indexer queries.
            pooled_keys: [n_docs, n_blocks_max, D] normalized + rotated block keys.
            block_valid: [n_docs, n_blocks_max].
            doc_ids / positions: [T] int32.
            uniform_doc_len: Length of each document in the non-THD batch layout.
            output_format: ``bits`` retains the existing Flex mask contract;
                ``ids`` retains ``top.indices`` without building a bitset.

        Returns:
            ``(selection, all_selected)``; selection is either
            ``selected_bits [T, nbytes] uint8`` or ``selected_ids [T, K] int32``.
        """
        T, H, D = q.shape
        R = self.compress_ratio
        n_blocks_max = pooled_keys.shape[1]
        if output_format not in ("bits", "ids"):
            raise ValueError(f"unsupported QSA selection format: {output_format}")
        # One extra (always clear) bit for the incomplete tail block: the mask kernels look up
        # ``position // R`` for every key, including keys of the block that is still open.
        nbytes = (n_blocks_max + 1 + 7) // 8
        # Retain compact TopK IDs directly for the selected-ID kernel. This
        # avoids the O(T * n_blocks_max) bitset, including at long lengths.
        if output_format == "ids":
            selected = torch.full((T, self.block_topk), -1, dtype=torch.int32, device=q.device)
        else:
            selected = torch.empty(T, nbytes, dtype=torch.uint8, device=q.device)
        visible = ((positions.long() + 1) // R).clamp_max(n_blocks_max)  # complete blocks per query
        all_selected = bool((visible <= self.block_topk).all().item())
        block_ids = torch.arange(n_blocks_max, device=q.device)
        # Bound scores, converted queries and the temporary scatter buffer together. The
        # selected-ID output remains O(T * block_topk) when requested.
        output_bytes = 4 * self.block_topk if output_format == "ids" else nbytes
        bytes_per_query = 4 * (n_blocks_max * H + H * D + output_bytes)
        tile = max(1, min(T, _QSA_SELECT_TILE_BYTES // max(1, bytes_per_query)))
        if all_selected or n_blocks_max == 0:
            # Every visible block is selected: the pattern is plain causal attention. Still
            # materialize the requested output so forced sparse runs see the selection.
            for start in range(0, T, tile):
                end = min(T, start + tile)
                if output_format == "ids":
                    ids = torch.arange(self.block_topk, device=q.device, dtype=torch.int32)
                    selected[start:end] = torch.where(
                        ids[None, :] < visible[start:end, None], ids[None, :], -1
                    )
                    continue
                bits = torch.zeros(end - start, nbytes, dtype=torch.int32, device=q.device)
                if n_blocks_max > 0:
                    vis = (block_ids.unsqueeze(0) < visible[start:end].unsqueeze(1)).to(torch.int32)
                    weights = (1 << (block_ids & 7)).to(torch.int32).unsqueeze(0)
                    byte_idx = (block_ids >> 3).unsqueeze(0).expand(end - start, -1)
                    bits.scatter_add_(1, byte_idx, vis * weights)
                selected[start:end] = bits.to(torch.uint8)
            return selected, True

        n_docs = pooled_keys.shape[0]
        k = min(self.block_topk, n_blocks_max)
        # Multiple short packed documents make a GEMM and scatter per document slower than
        # one batched GEMM. Permit a gathered-key path only when its query tile, including
        # both the gathered source and fp32 copy, fits the same workspace budget. Long
        # documents keep the shared-key path below and never expand keys per query.
        # Require at least one average document per gathered tile so a few long
        # documents do not trade their shared GEMMs for many gathered-key GEMMs.
        gathered_bytes_per_query = (
            n_blocks_max * D * (pooled_keys.element_size() + 4)
            + n_blocks_max * (H * 4 + 8)
            + H * D * 4
            + output_bytes * 4
            + k * 32
        )
        gathered_tile = max(1, min(T, _QSA_SELECT_TILE_BYTES // gathered_bytes_per_query))
        if n_docs > 1 and gathered_tile >= min(T, 32) and T <= n_docs * gathered_tile:
            for start in range(0, T, gathered_tile):
                end = min(T, start + gathered_tile)
                q_c = q[start:end].float()
                k_c = pooled_keys[doc_ids[start:end].long()].float()
                scores = torch.einsum("thd,tnd->thn", q_c, k_c)
                scores = scores.relu_().sum(dim=1) / math.sqrt(D)
                vis = block_ids.unsqueeze(0) < visible[start:end].unsqueeze(1)
                scores.masked_fill_(~vis, float("-inf"))
                top = torch.topk(scores, k=k, dim=-1)
                valid = torch.isfinite(top.values)
                if output_format == "ids":
                    selected[start:end, :k] = top.indices.masked_fill(~valid, -1).to(torch.int32)
                    continue
                blocks = top.indices.masked_fill(~valid, 0)
                bit_val = ((1 << (blocks & 7)) * valid.to(blocks.dtype)).to(torch.int32)
                bits = torch.zeros(end - start, nbytes, dtype=torch.int32, device=q.device)
                bits.scatter_add_(1, blocks >> 3, bit_val)
                selected[start:end] = bits.to(torch.uint8)
            return selected, False

        # A single document or a non-THD batch has offsets known from tensor shapes.
        # For packed long documents, doc_ids are monotone after CP reconstruction;
        # empty documents give equal adjacent offsets and need no work.
        if n_docs == 1:
            offsets = (0, T)
        elif uniform_doc_len is not None:
            assert T == n_docs * uniform_doc_len
            offsets = tuple(doc * uniform_doc_len for doc in range(n_docs + 1))
        else:
            offsets = torch.searchsorted(
                doc_ids.contiguous(), torch.arange(n_docs + 1, device=q.device, dtype=doc_ids.dtype)
            ).tolist()
        for doc in range(n_docs):
            doc_start, doc_end = offsets[doc], offsets[doc + 1]
            if doc_start == doc_end:
                continue
            key_t = pooled_keys[doc].float().T  # shared [D, n_blocks_max] GEMM operand
            for start in range(doc_start, doc_end, tile):
                end = min(doc_end, start + tile)
                q_c = q[start:end].float().reshape(-1, D)  # [t * H, D]
                scores = (q_c @ key_t).view(end - start, H, n_blocks_max)
                scores = scores.relu_().sum(dim=1) / math.sqrt(D)
                vis = block_ids.unsqueeze(0) < visible[start:end].unsqueeze(1)
                scores.masked_fill_(~vis, float("-inf"))
                top = torch.topk(scores, k=k, dim=-1)
                valid = torch.isfinite(top.values)
                if output_format == "ids":
                    selected[start:end, :k] = top.indices.masked_fill(~valid, -1).to(torch.int32)
                    continue
                blocks = top.indices.masked_fill(~valid, 0)  # compact [t, k] block IDs
                byte_idx = blocks >> 3
                # Top-k blocks are distinct; summing their bit values is equivalent to OR.
                bit_val = ((1 << (blocks & 7)) * valid.to(blocks.dtype)).to(torch.int32)
                bits = torch.zeros(end - start, nbytes, dtype=torch.int32, device=q.device)
                bits.scatter_add_(1, byte_idx, bit_val)
                selected[start:end] = bits.to(torch.uint8)
        return selected, False

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        hidden_states: Tensor,
        rotary_pos_emb: Optional[Tensor],
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> QSASelection:
        """Compute the block selection.

        Args:
            hidden_states: [s, b, H] layer input (sequence-parallel-sharded when SP is on).
            rotary_pos_emb: [max_s, 1, 1, rot_dim] rotary frequencies of the attention layer.
            packed_seq_params: THD packing parameters.

        Returns:
            :class:`QSASelection` over the full (gathered) sequence.
        """
        assert rotary_pos_emb is not None, "QSA requires rotary position embeddings"
        if isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = rotary_pos_emb[0]
        is_absolute_mrope = getattr(self.config, "mrope_section", None) is not None
        if self.config.sequence_parallel and get_pg_size(self.pg_collection.tp) > 1:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states, group=self.pg_collection.tp
            )
        local_seq_len = hidden_states.shape[0]
        cp_size = get_pg_size(self.pg_collection.cp)
        if cp_size > 1:
            # Rebuild the full sequence (true causal order) so the block selection sees every
            # key of the sequence, exactly as in the CP=1 case. The owning
            # ``QwenSparseSelfAttention`` runs the standard CP attention (all-gather q/k/v ->
            # attention over the full sequence -> split), which aligns with this gathered
            # selection; only the all-gather CP comm type is supported.
            from megatron.core.ssm.mamba_context_parallel import reconstruct_tensor_cp

            hidden_states = reconstruct_tensor_cp(hidden_states, packed_seq_params, dim=0)
            is_thd = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
            if (is_absolute_mrope or not is_thd) and rotary_pos_emb.shape[0] == local_seq_len:
                # Non-packed RoPE and VL mRoPE arrive in CP-local zigzag order.
                # Rebuild frequencies in the same full order as hidden_states.
                # Packed standard RoPE instead uses a max-document-length table.
                rotary_pos_emb = reconstruct_tensor_cp(rotary_pos_emb, packed_seq_params, dim=0)
        # The indexer sees the full sequence after CP reconstruction. The public RoPE
        # helper treats cp_group=None as the global CP group, so it would remap these
        # full-sequence queries as if they were still rank-local.
        is_thd = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
        s, b, _ = hidden_states.shape
        with torch.no_grad():
            doc_ids, positions, max_doc_len = _sequence_layout(
                b, s, packed_seq_params, hidden_states.device
            )
        qk, _ = self.index_qk_proj(hidden_states)  # [s, b, (H+1)*D]
        q, raw_keys = torch.split(
            qk, [self.n_heads * self.head_dim, self.kv_heads * self.head_dim], dim=-1
        )
        q = self.q_layernorm(
            q.reshape(s, b, self.n_heads, self.head_dim).reshape(-1, self.head_dim)
        )
        q = q.view(s, b, self.n_heads, self.head_dim)
        if cp_size > 1:
            q = _full_sequence_indexer_rope(
                q, rotary_pos_emb, positions, is_thd, is_absolute_mrope, self.rotary_interleaved
            )
        elif is_thd:
            q = apply_rotary_pos_emb(
                q.squeeze(1),
                rotary_pos_emb,
                config=self.config,
                cu_seqlens=packed_seq_params.cu_seqlens_q,
                cp_group=self.pg_collection.cp,
                max_seqlen=packed_seq_params.max_seqlen_q,
            ).unsqueeze(1)
        else:
            q = apply_rotary_pos_emb(
                q, rotary_pos_emb, config=self.config, cp_group=self.pg_collection.cp
            )
        # Flatten [s, b] -> [b*s] token order (batch-major) to match doc_ids/positions [b, s].
        q_flat = q.transpose(0, 1).reshape(b * s, self.n_heads, self.head_dim)
        keys_flat = raw_keys.transpose(0, 1).reshape(b * s, self.head_dim)
        doc_flat = doc_ids.reshape(-1)
        pos_flat = positions.reshape(-1)

        with torch.no_grad():
            pooled, block_positions, block_valid = self._pool_keys(
                keys_flat.detach(), doc_flat, pos_flat, max_doc_len
            )
            n_docs, n_blocks_max, D = pooled.shape
            if n_blocks_max > 0:
                pooled = self.k_layernorm(pooled.reshape(-1, D))
                rope_block_positions = block_positions
                if is_absolute_mrope and is_thd and rotary_pos_emb.shape[0] == s:
                    cu = packed_seq_params.cu_seqlens_q_padded
                    if cu is None:
                        cu = packed_seq_params.cu_seqlens_q
                    starts = cu[:-1].to(device=block_positions.device, dtype=torch.long)
                    ends = cu[1:].to(device=block_positions.device, dtype=torch.long)
                    rope_block_positions = (block_positions + starts[:, None]).clamp_max(
                        ends[:, None] - 1
                    )
                batch_indices = None
                if is_absolute_mrope and not is_thd and b > 1:
                    batch_indices = torch.arange(n_docs, device=pooled.device).repeat_interleave(
                        n_blocks_max
                    )
                pooled = self._rope_at_positions(
                    pooled, rotary_pos_emb, rope_block_positions.reshape(-1), batch_indices
                )
                pooled = pooled.view(n_docs, n_blocks_max, D)
            selected_bits, all_selected = self._select_blocks(
                q_flat.detach(), pooled, block_valid, doc_flat, pos_flat, None if is_thd else s
            )
        nbytes = selected_bits.shape[1]
        return QSASelection(
            doc_ids=doc_ids,
            positions=positions,
            selected_bits=selected_bits.reshape(-1).contiguous(),
            bits_per_row=nbytes,
            bits_per_row_t=torch.tensor(nbytes, device=selected_bits.device, dtype=torch.int64),
            compress_ratio=self.compress_ratio,
            all_selected=all_selected,
        )


# ---------------------------------------------------------------------------- core attention


def _qsa_mask_mod_factory(selection: QSASelection):
    """Build the FlexAttention ``mask_mod`` for a :class:`QSASelection`."""
    doc_ids, positions = selection.doc_ids, selection.positions
    selected_bits, nbytes_t, ratio = (
        selection.selected_bits,
        selection.bits_per_row_t,
        selection.compress_ratio,
    )
    # Tensor scalar (not a python int) so a new sequence length does not trigger a recompile.
    seq_len = torch.tensor(positions.shape[1], device=positions.device, dtype=torch.int64)

    def mask_mod(b, h, q_idx, kv_idx):
        same_doc = doc_ids[b, q_idx] == doc_ids[b, kv_idx]
        qp = positions[b, q_idx].to(torch.int64)
        kp = positions[b, kv_idx].to(torch.int64)
        causal = kp <= qp
        tail_start = ((qp + 1) // ratio) * ratio
        is_tail = kp >= tail_start
        blk = kp // ratio
        byte = selected_bits[(b * seq_len + q_idx) * nbytes_t + (blk >> 3)].to(torch.int64)
        is_selected = ((byte >> (blk & 7)) & 1) == 1
        return same_doc & causal & (is_tail | is_selected)

    return mask_mod


@functools.lru_cache(maxsize=1)
def _compiled_flex_attention():
    return torch.compile(flex_attention, dynamic=True)


class QSACoreAttention(torch.nn.Module):
    """Core attention of QSA: dense causal attention when the selection is complete, otherwise
    block-sparse FlexAttention over the selected tokens.

    Constructed like the backend core-attention classes (``config, layer_number, attn_mask_type,
    attention_type, cp_comm_type, softmax_scale, pg_collection``) so it plugs into
    :class:`~megatron.core.transformer.attention.Attention`. ``dense_core_attention`` is the
    backend class used for the dense path. The selection is handed over by the owning
    :class:`QwenSparseSelfAttention` through :meth:`set_selection` right before the call.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        dense_core_attention: type,
        cp_comm_type: Optional[str] = None,
        softmax_scale: Optional[float] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type
        self.pg_collection = pg_collection
        self.softmax_scale = (
            softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(config.kv_channels)
        )
        self.dense_core_attention = dense_core_attention(
            config=config,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type=attention_type,
            cp_comm_type=cp_comm_type,
            softmax_scale=softmax_scale,
            pg_collection=pg_collection,
            **kwargs,
        )
        self._selection: Optional[QSASelection] = None
        self.sparse_backend = os.environ.get("MCORE_QSA_SPARSE_BACKEND", "flex")

    def set_selection(self, selection: Optional[QSASelection]) -> None:
        """Register the block selection consumed by the next forward call."""
        self._selection = selection

    def _flex_forward(
        self, query: Tensor, key: Tensor, value: Tensor, selection: QSASelection, is_thd: bool
    ) -> Tensor:
        if not HAVE_FLEX_ATTENTION:
            raise RuntimeError("QSA sparse attention requires torch.nn.attention.flex_attention")
        if is_thd:  # [t, h, d] -> [1, h, t, d]
            q = query.unsqueeze(0).transpose(1, 2)
            k = key.unsqueeze(0).transpose(1, 2)
            v = value.unsqueeze(0).transpose(1, 2)
        else:  # [s, b, h, d] -> [b, h, s, d]
            q = query.permute(1, 2, 0, 3)
            k = key.permute(1, 2, 0, 3)
            v = value.permute(1, 2, 0, 3)
        b, hq, s, d = q.shape
        mask_mod = _qsa_mask_mod_factory(selection)
        block_mask = create_block_mask(mask_mod, b, None, s, s, device=q.device, _compile=True)
        try:
            out = _compiled_flex_attention()(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                block_mask=block_mask,
                scale=self.softmax_scale,
                enable_gqa=k.shape[1] != hq,
            )
        except torch._dynamo.exc.TorchDynamoException as error:  # pragma: no cover
            # Inductor occasionally finds no valid kernel template for tiny, single-block
            # sequences after a dynamic recompile. Those shapes are cheap to run densely.
            if s > 4096:
                raise
            warnings.warn(
                f"FlexAttention compilation failed for QSA (b={b}, s={s}); falling back to the "
                f"dense masked path for this call: {type(error).__name__}"
            )
            return self._dense_masked_forward(query, key, value, selection, is_thd)
        if is_thd:
            return out.transpose(1, 2).squeeze(0).reshape(s, hq * d)
        return out.permute(2, 0, 1, 3).reshape(s, b, hq * d)

    def _dense_masked_forward(
        self, query: Tensor, key: Tensor, value: Tensor, selection: QSASelection, is_thd: bool
    ) -> Tensor:
        """Reference path: materialize the boolean mask and run SDPA (O(s^2) memory)."""
        if is_thd:
            q = query.unsqueeze(0).transpose(1, 2)
            k = key.unsqueeze(0).transpose(1, 2)
            v = value.unsqueeze(0).transpose(1, 2)
        else:
            q = query.permute(1, 2, 0, 3)
            k = key.permute(1, 2, 0, 3)
            v = value.permute(1, 2, 0, 3)
        b, hq, s, d = q.shape
        mask = build_qsa_dense_mask(selection, s)  # [b, s, s] bool
        rep = hq // k.shape[1]
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        # cuDNN SDPA cannot reliably initialize its frontend for packed CP masks.
        with torch.nn.attention.sdpa_kernel(
            [
                torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                torch.nn.attention.SDPBackend.MATH,
            ]
        ):
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask.unsqueeze(1), scale=self.softmax_scale
            )
        if is_thd:
            return out.transpose(1, 2).squeeze(0).reshape(s, hq * d)
        return out.permute(2, 0, 1, 3).reshape(s, b, hq * d)

    def _all_selected_cp_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        packed_seq_params: Optional[PackedSeqParams],
    ) -> Tensor:
        """Causal attention without a quadratic mask for CP-reconstructed Q/K/V."""
        if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
            from flash_attn import flash_attn_varlen_func

            cu_q = packed_seq_params.cu_seqlens_q_padded
            cu_kv = packed_seq_params.cu_seqlens_kv_padded
            if cu_q is None:
                cu_q = packed_seq_params.cu_seqlens_q
            if cu_kv is None:
                cu_kv = packed_seq_params.cu_seqlens_kv
            out = flash_attn_varlen_func(
                query.contiguous(),
                key.contiguous(),
                value.contiguous(),
                cu_q,
                cu_kv,
                packed_seq_params.max_seqlen_q or query.shape[0],
                packed_seq_params.max_seqlen_kv or key.shape[0],
                softmax_scale=self.softmax_scale,
                causal=True,
            )
            return out.reshape(out.shape[0], -1)

        # Each batch row is a separate document in the non-THD layout.
        q = query.permute(1, 2, 0, 3)
        k = key.permute(1, 2, 0, 3)
        v = value.permute(1, 2, 0, 3)
        with torch.nn.attention.sdpa_kernel(
            [
                torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                torch.nn.attention.SDPBackend.MATH,
            ]
        ):
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                scale=self.softmax_scale,
                enable_gqa=q.shape[1] != k.shape[1],
            )
        return out.permute(2, 0, 1, 3).reshape(query.shape[0], query.shape[1], -1)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor],
        attn_mask_type: Optional[AttnMaskType] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tensor:
        """Attention over the selected tokens.

        ``query``/``key``/``value`` are ``[s, b, h, d]`` (sbhd) or ``[t, h, d]`` (thd); the output
        is ``[s, b, h*d]`` / ``[t, h*d]`` like the backend core attention.
        """
        # Keep the selection registered: selective `core_attn` recompute re-runs this forward
        # in the backward pass with the same tensors, and the next attention forward always
        # registers a fresh selection before reaching here.
        selection = self._selection
        if selection is None:
            raise RuntimeError(
                "QSACoreAttention.forward called without a block selection; "
                "QwenSparseSelfAttention must run the indexer first."
            )
        cp_size = get_pg_size(self.pg_collection.cp)
        if selection.all_selected and not self.config.qsa_force_sparse and cp_size == 1:
            return self.dense_core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        assert attention_bias is None, "QSA does not support attention bias"
        is_thd = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
        # The sparse (FlexAttention) and dense-masked (SDPA) paths run their kernel directly on
        # the local q/k/v the owning attention hands them, so under CP>1 they would attend over
        # the local zigzag slice only while the block selection (computed by the indexer over the
        # gathered full sequence) is over the full sequence. Gather q/k/v to the full causal
        # sequence -- the per-token RoPE the owning attention applied already uses each token's
        # correct global position, so it stays aligned after the gather -- run the sparse kernel
        # over the full sequence with the full-seq selection, then split the output back to the
        # local slice. Numerically exact vs CP=1; the sparse kernel is O(global_s^2) here, so the
        # block-sparse triton kernel remains the long-seq production follow-up. For CP>1,
        # the all-selected dense TE kernel returns NaN gradients with packed padding-causal
        # attention, so short sequences use the explicit gather + masked SDPA path too.
        if cp_size > 1:
            from megatron.core.ssm.mamba_context_parallel import (
                reconstruct_tensor_cp,
                split_tensor_cp,
            )

            query = reconstruct_tensor_cp(query, packed_seq_params, dim=0)
            # Only local query outputs survive the split. Remote keys/values, however,
            # contribute to those outputs and need a summed backward across CP ranks.
            kv_heads = key.shape[-2]
            kv = reconstruct_tensor_cp(
                torch.cat((key, value), dim=-2), packed_seq_params, dim=0, differentiable=True
            )
            key, value = kv.split(kv_heads, dim=-2)
        if selection.all_selected and not self.config.qsa_force_sparse and cp_size > 1:
            out = self._all_selected_cp_forward(query, key, value, packed_seq_params)
        elif self.sparse_backend == "dense_masked":
            out = self._dense_masked_forward(query, key, value, selection, is_thd)
        else:
            out = self._flex_forward(query, key, value, selection, is_thd)
        if cp_size > 1:
            out = split_tensor_cp(out, packed_seq_params, dim=0)
        return out


def build_qsa_dense_mask(selection: QSASelection, seq_len: int) -> Tensor:
    """Materialize the ``[b, s, s]`` boolean attention mask of a selection (tests/reference)."""
    doc, pos, R = selection.doc_ids, selection.positions, selection.compress_ratio
    bits = selection.selected_bits.view(doc.shape[0], doc.shape[1], selection.bits_per_row)
    qp = pos.long().unsqueeze(-1)  # [b, s, 1]
    kp = pos.long().unsqueeze(1)  # [b, 1, s]
    same_doc = doc.unsqueeze(-1) == doc.unsqueeze(1)
    causal = kp <= qp
    tail_start = ((qp + 1) // R) * R
    is_tail = kp >= tail_start
    blk = kp // R  # [b, 1, s]
    byte = torch.gather(bits.long(), 2, (blk >> 3).expand(-1, seq_len, -1))
    is_selected = ((byte >> (blk & 7)) & 1) == 1
    return same_doc & causal & (is_tail | is_selected)


class QwenSparseSelfAttention(SelfAttention):
    """Gated GQA self-attention whose keys/values are selected per query by a QSA indexer.

    The indexer runs on the layer input first; its selection is handed to the
    :class:`QSACoreAttention` core-attention module, after which the standard
    :class:`SelfAttention` forward (QKV projection, QK-norm, RoPE, core attention, output gate,
    output projection) proceeds unchanged.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: QwenSparseSelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.causal,
        cp_comm_type: Optional[str] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
            **kwargs,
        )
        assert isinstance(
            self.core_attention, QSACoreAttention
        ), "QwenSparseSelfAttention requires a QSACoreAttention core_attention submodule"
        assert config.qsa_indexer_n_heads is not None, "QSA config (qsa_indexer_*) is not set"
        self.indexer = build_module(
            submodules.indexer,
            config=config,
            layer_number=layer_number,
            pg_collection=self.pg_collection,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        key_value_states: Optional[Tensor] = None,
        inference_context=None,
        rotary_pos_emb: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params=None,
    ):
        """Run the indexer, register its selection, then the regular attention forward."""
        if inference_context is not None or inference_params is not None:
            raise NotImplementedError(
                "QwenSparseSelfAttention does not support inference contexts yet."
            )
        selection = self.indexer(hidden_states, rotary_pos_emb, packed_seq_params)
        self.core_attention.set_selection(selection)
        return super().forward(
            hidden_states,
            attention_mask,
            key_value_states=key_value_states,
            inference_context=None,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )


__all__ = [
    "QSACoreAttention",
    "QSAIndexer",
    "QSAIndexerSubmodules",
    "QSASelection",
    "QwenSparseSelfAttention",
    "QwenSparseSelfAttentionSubmodules",
    "build_qsa_dense_mask",
]
