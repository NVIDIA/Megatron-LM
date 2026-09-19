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

from megatron.core.models.common.embeddings.rope_utils import _rotate_half, apply_rotary_pos_emb
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
        assert get_pg_size(pg_collection.cp) == 1, "QSA does not support context parallelism."

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
    def _rope_at_positions(self, x: Tensor, freqs: Tensor, positions: Tensor) -> Tensor:
        """Rotate ``x`` [N, D] with the frequencies of ``positions`` [N] (first ``rot_dim``
        dims)."""
        freqs = freqs.reshape(freqs.shape[0], -1)  # [max_s, rot_dim]
        rot_dim = freqs.shape[-1]
        f = freqs[positions.long()]  # [N, rot_dim]
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
    ) -> Tuple[Tensor, bool]:
        """Top-k complete blocks per query.

        Args:
            q: [T, H, D] normalized + rotated indexer queries.
            pooled_keys: [n_docs, n_blocks_max, D] normalized + rotated block keys.
            block_valid: [n_docs, n_blocks_max].
            doc_ids / positions: [T] int32.

        Returns:
            ``(selected_bits [T, nbytes] uint8, all_selected)``.
        """
        T, H, D = q.shape
        R = self.compress_ratio
        n_blocks_max = pooled_keys.shape[1]
        # One extra (always clear) bit for the incomplete tail block: the mask kernels look up
        # ``position // R`` for every key, including keys of the block that is still open.
        nbytes = (n_blocks_max + 1 + 7) // 8
        selected_bits = torch.zeros(T, nbytes, dtype=torch.int32, device=q.device)
        visible = ((positions.long() + 1) // R).clamp_max(n_blocks_max)  # complete blocks per query
        all_selected = bool((visible <= self.block_topk).all().item())
        block_ids = torch.arange(n_blocks_max, device=q.device)
        if all_selected or n_blocks_max == 0:
            # Every visible block is selected: the pattern is plain causal attention. Still
            # materialize the bitset so the sparse kernels (qsa_force_sparse) see it.
            if n_blocks_max > 0:
                vis = (block_ids.unsqueeze(0) < visible.unsqueeze(1)).to(
                    torch.int32
                )  # [T, n_blocks]
                weights = (
                    (torch.ones_like(block_ids) << (block_ids & 7)).to(torch.int32).unsqueeze(0)
                )
                byte_idx = (block_ids >> 3).unsqueeze(0).expand(T, -1)
                selected_bits.scatter_add_(1, byte_idx, (vis * weights).to(torch.int32))
            return selected_bits.to(torch.uint8), True
        chunk = max(1, min(T, (256 << 20) // max(1, n_blocks_max * H * 4)))  # ~256MB of scores
        for start in range(0, T, chunk):
            end = min(T, start + chunk)
            q_c = q[start:end].float()  # [t, H, D]
            k_c = pooled_keys[doc_ids[start:end].long()].float()  # [t, n_blocks_max, D]
            scores = torch.einsum("thd,tnd->thn", q_c, k_c).relu_().sum(dim=1) / math.sqrt(D)
            vis = block_ids.unsqueeze(0) < visible[start:end].unsqueeze(1)
            scores = scores.masked_fill(~vis, float("-inf"))
            k = min(self.block_topk, n_blocks_max)
            top = torch.topk(scores, k=k, dim=-1)
            valid = torch.isfinite(top.values)
            blocks = top.indices.masked_fill(~valid, 0)
            byte_idx = blocks >> 3
            # Distinct blocks of one query never share a bit, so summing the bit values ORs them.
            bit_val = ((torch.ones_like(blocks) << (blocks & 7)) * valid.to(blocks.dtype)).to(
                torch.int32
            )
            selected_bits[start:end].scatter_add_(1, byte_idx, bit_val)
        return selected_bits.to(torch.uint8), False

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
        if self.config.sequence_parallel and get_pg_size(self.pg_collection.tp) > 1:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states, group=self.pg_collection.tp
            )
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
        if is_thd:
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
                pooled = self._rope_at_positions(
                    pooled, rotary_pos_emb, block_positions.reshape(-1)
                )
                pooled = pooled.view(n_docs, n_blocks_max, D)
            selected_bits, all_selected = self._select_blocks(
                q_flat.detach(), pooled, block_valid, doc_flat, pos_flat
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
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask.unsqueeze(1), scale=self.softmax_scale
        )
        if is_thd:
            return out.transpose(1, 2).squeeze(0).reshape(s, hq * d)
        return out.permute(2, 0, 1, 3).reshape(s, b, hq * d)

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
        if selection.all_selected and not self.config.qsa_force_sparse:
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
        if self.sparse_backend == "dense_masked":
            return self._dense_masked_forward(query, key, value, selection, is_thd)
        return self._flex_forward(query, key, value, selection, is_thd)


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
