# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""cuDNN frontend kernels for simplified sparse attention."""

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from megatron.core.ops._backends import require
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

from . import dsa_layout, dsa_masking

_HEAD_DIM = 256
_INDEXER_DIM = 128


@dataclass(frozen=True)
class CudnnSparseAttentionKernels:
    """Kernel entry points bound once when the core attention is constructed.

    ``dense_attention`` (FlashAttention-4) and ``dense_indexer_kl`` (cuDNN frontend) are bound
    only for indexer-only training, where a frozen dense teacher replaces sparse attention.
    """

    sparse_attention: Callable
    row_metadata: type
    dense_attention: Optional[Callable] = None
    dense_indexer_kl: Optional[Callable] = None


def bind_cudnn_sparse_attention(*, dense_teacher: bool = False) -> CudnnSparseAttentionKernels:
    """Bind cuDNN frontend simplified sparse attention, failing early when it is absent.

    The kernels ship in ``nvidia-cudnn-frontend`` 1.33 and newer and require CuTe DSL 4.8.
    ``dense_teacher`` additionally binds the dense indexer objective and FlashAttention-4.
    """
    symbols = ["simplified_sparse_attention_forward", "SimplifiedSparseAttentionMetadata"]
    if dense_teacher:
        symbols.append("dense_indexer_kl_forward")
    ops = require(
        "cudnn.ops", *symbols, dist="nvidia-cudnn-frontend", needed_by="Simplified sparse attention"
    )
    dense_attention = None
    if dense_teacher:
        from megatron.core.transformer import attention

        if not attention.HAVE_FA4:
            raise ImportError(
                "Simplified sparse attention indexer-only training computes its dense teacher "
                f"with flash-attn-4 >= {attention._MIN_FA4_VERSION}, which is not installed."
            )
        dense_attention = attention.flash_attn4_varlen_func
    return CudnnSparseAttentionKernels(
        sparse_attention=ops.simplified_sparse_attention_forward,
        row_metadata=ops.SimplifiedSparseAttentionMetadata,
        dense_attention=dense_attention,
        dense_indexer_kl=ops.dense_indexer_kl_forward if dense_teacher else None,
    )


@dataclass
class AttentionLayout:
    """Row metadata shared by the SSA layers of one microbatch.

    Local query rows map to globally ordered, document-local causal key bounds. Under context
    parallelism, ``key_reorder`` puts all-gathered keys back into global order.
    """

    key_starts: torch.Tensor
    key_ends: torch.Tensor
    query_valid: torch.Tensor
    num_real_tokens: torch.Tensor
    key_reorder: Optional[torch.Tensor]
    cu_seqlens: torch.Tensor
    max_seqlen: int
    packed: bool
    cp_size: int
    cp_rank: int
    phase1_segments: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None


def build_attention_layout(
    query: torch.Tensor,
    packed_seq_params: Optional[PackedSeqParams],
    cp_group: Optional[torch.distributed.ProcessGroup],
) -> AttentionLayout:
    """Map local (zigzag) query rows to globally ordered, document-local causal keys.

    Packed metadata carries logical document lengths in ``cu_seqlens_q`` and physical storage in
    ``cu_seqlens_q_padded``, with padding at each document's suffix. Padding rows are invalid
    queries. Language-model loss masks do not describe validity: prompt or image tokens can be
    real inputs without a prediction objective.
    """
    local_rows, device = query.shape[0], query.device
    cp_size = cp_group.size() if cp_group is not None else 1
    cp_rank = cp_group.rank() if cp_size > 1 else 0
    global_rows = local_rows * cp_size
    packed = packed_seq_params is not None
    if packed:
        if packed_seq_params.qkv_format != "thd":
            raise ValueError("Simplified sparse attention requires THD packed metadata.")
        if packed_seq_params.cp_group is not None and packed_seq_params.cp_group is not cp_group:
            raise ValueError("Packed simplified sparse attention must use the attention CP group.")
        if packed_seq_params.local_cp_size not in (None, cp_size):
            raise ValueError("Packed simplified sparse attention metadata has a different CP size.")
        cu_q, cu_k = dsa_layout.get_packed_qk_cu_seqlens(packed_seq_params)
        # Self-attention: Q/K share document boundaries, possibly in distinct equal tensors.
        if cu_q.shape != cu_k.shape:
            raise ValueError(
                "Simplified sparse attention requires matching Q/K document boundaries."
            )
        logical_cu = packed_seq_params.cu_seqlens_q
        if logical_cu is None or logical_cu.shape != cu_q.shape:
            raise ValueError(
                "Packed simplified sparse attention requires matching logical and physical "
                "document lengths."
            )
        if packed_seq_params.max_seqlen_q is None:
            raise ValueError("Packed simplified sparse attention requires max_seqlen_q.")
        cu_seqlens = cu_q
        max_seqlen = packed_seq_params.max_seqlen_q
    else:
        cu_seqlens = torch.tensor((0, global_rows), dtype=torch.int32, device=device)
        max_seqlen = global_rows
        query_valid = torch.ones(local_rows, dtype=torch.bool, device=device)

    if cp_size > 1:
        if packed:
            positions, reorder = (
                dsa_layout.build_packed_allgather_cp_query_positions_and_key_reorder(
                    cu_seqlens,
                    cu_seqlens,
                    cp_size,
                    cp_rank,
                    device,
                    local_output_size=local_rows,
                    global_output_size=global_rows,
                    query_cu_seqlens_cover_output=True,
                    key_cu_seqlens_cover_output=True,
                )
            )
        else:
            positions = dsa_layout.build_zigzag_cp_local_positions(
                global_rows, cp_size, cp_rank, device
            )
            reorder = dsa_layout.build_zigzag_allgather_cp_key_reorder(local_rows, cp_size, device)
    else:
        positions = torch.arange(local_rows, dtype=torch.int64, device=device)
        reorder = None
    starts, ends = dsa_masking.generate_varlen_mask_params_for_positions(cu_seqlens, positions)
    if packed:
        # Real query rows precede each document's padding, so their causal key prefixes stay
        # valid with physical boundaries in both training phases.
        sequence_ids = torch.searchsorted(
            cu_seqlens[1:].to(device=device).contiguous(), positions, right=True
        )
        logical_lengths = logical_cu.to(device=device, dtype=torch.int64).diff()
        query_valid = positions < starts + logical_lengths.index_select(0, sequence_ids)
        num_real_tokens = query_valid.sum(dtype=torch.float32).reshape(1)
        if cp_size > 1:
            torch.distributed.all_reduce(num_real_tokens, group=cp_group)
        # An all-padding microbatch contributes zero auxiliary loss.
        num_real_tokens = num_real_tokens.clamp_min(1)
    else:
        num_real_tokens = torch.full((1,), float(global_rows), dtype=torch.float32, device=device)
    # cuDNN takes int32 key bounds; convert once per microbatch layout.
    return AttentionLayout(
        starts.to(torch.int32),
        ends.to(torch.int32),
        query_valid,
        num_real_tokens,
        reorder,
        cu_seqlens,
        max_seqlen,
        packed,
        cp_size,
        cp_rank,
    )


def _phase1_flash_inputs(
    key: torch.Tensor, value: torch.Tensor, layout: AttentionLayout
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Express each local zigzag query chunk as bottom-right causal FlashAttention.

    A CP rank owns the front chunk and the mirrored back chunk of every document. Each chunk gets
    its causal key prefix, so FlashAttention's rectangular causal mask matches the chunk's
    global query positions.
    """
    if layout.cp_size == 1:
        return (
            key,
            value,
            layout.cu_seqlens,
            layout.cu_seqlens,
            layout.max_seqlen,
            layout.max_seqlen,
        )
    if layout.phase1_segments is None:
        segments = dsa_layout.build_packed_cp_indexer_layout(
            layout.cu_seqlens,
            layout.cu_seqlens,
            cp_size=layout.cp_size,
            cp_rank=layout.cp_rank,
            key_size=key.shape[0],
        )
        layout.phase1_segments = (
            segments.source_indices,
            segments.segment_cu_q.to(dtype=torch.int32),
            segments.segment_cu_k.to(dtype=torch.int32),
        )
    source_indices, cu_q, cu_k = layout.phase1_segments
    divisor = 2 * layout.cp_size
    return (
        key.index_select(0, source_indices),
        value.index_select(0, source_indices),
        cu_q,
        cu_k,
        layout.max_seqlen // divisor,
        layout.max_seqlen * (divisor - layout.cp_rank) // divisor,
    )


def _prepare_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_indexer: torch.Tensor,
    k_indexer: torch.Tensor,
    layout: Optional[AttentionLayout],
    cp_group: Optional[torch.distributed.ProcessGroup],
):
    """Check shapes and return packed-ness, local row count, layout and [T,H,D] kernel views."""
    packed_input = query.ndim == 3
    query, _ = dsa_layout.ensure_sbhd(query, "query")
    key, _ = dsa_layout.ensure_sbhd(key, "key")
    value, _ = dsa_layout.ensure_sbhd(value, "value")
    q_indexer, _ = dsa_layout.ensure_sbhd(q_indexer, "q_indexer")
    k_indexer, _ = dsa_layout.ensure_sbhd(k_indexer, "k_indexer")
    if query.shape[1] != 1:
        raise ValueError("Simplified sparse attention requires batch size 1.")
    rows = query.shape[0]
    if key.shape != (rows, 1, 1, _HEAD_DIM) or value.shape != key.shape:
        raise ValueError(
            "Simplified sparse attention requires local self-attention K/V [S,1,1,256]."
        )
    if q_indexer.shape != (rows, 1, 1, _INDEXER_DIM) or k_indexer.shape != q_indexer.shape:
        raise ValueError("Simplified sparse attention requires local indexer Q/K [S,1,1,128].")
    if layout is None:
        if packed_input:
            raise ValueError("Packed simplified sparse attention requires layout metadata.")
        layout = build_attention_layout(query, None, cp_group)
    return (
        packed_input,
        rows,
        layout,
        query[:, 0],
        key[:, 0],
        value[:, 0],
        q_indexer[:, 0, 0].contiguous(),
        k_indexer[:, 0, 0].contiguous(),
    )


def _gather_keys(layout: AttentionLayout, cp_group, *tensors: torch.Tensor):
    """All-gather CP-local keys into global order; the backward reduce-scatter stays BF16."""
    if layout.cp_size == 1:
        return tensors
    return tuple(
        gather_from_sequence_parallel_region(tensor, group=cp_group).index_select(
            0, layout.key_reorder
        )
        for tensor in tensors
    )


def _row_fields(layout: AttentionLayout, calculate_per_token_loss: bool) -> dict[str, torch.Tensor]:
    """Per-row key bounds, query validity and the global loss denominator."""
    return {
        "key_starts": layout.key_starts,
        "key_ends": layout.key_ends,
        "query_valid": layout.query_valid,
        "loss_denominator": (
            torch.ones_like(layout.num_real_tokens)
            if calculate_per_token_loss
            else layout.num_real_tokens
        ),
    }


def _restore_shape(output: torch.Tensor, rows: int, packed_input: bool) -> torch.Tensor:
    return output.reshape(rows, -1) if packed_input else output.reshape(rows, 1, -1)


def run_sparse_attention(
    kernels: CudnnSparseAttentionKernels,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_indexer: torch.Tensor,
    k_indexer: torch.Tensor,
    *,
    topk: int,
    loss_coeff: float,
    calculate_per_token_loss: bool,
    layout: Optional[AttentionLayout] = None,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run joint training: top-k sparse attention and the selected-support indexer KL.

    Inputs are SBHD with batch size 1 or packed THD, with CP-local queries. Under context
    parallelism, keys, values and indexer keys are all-gathered into global order. The indexer
    selects ``topk`` keys per query within its document, and the returned loss is the KL from
    the mean attention distribution to the indexer distribution over that selection. cuDNN
    applies the coefficient and global sum/mean normalization once, to the loss and gradients.
    """
    packed_input, rows, layout, query, key, value, q_indexer, k_indexer = _prepare_inputs(
        query, key, value, q_indexer, k_indexer, layout, cp_group
    )
    metadata = kernels.row_metadata(
        **_row_fields(layout, calculate_per_token_loss),
        triangular_scores=layout.cp_size == 1,
        mask_invalid_rows=layout.packed,
    )
    key, value, k_indexer = _gather_keys(layout, cp_group, key, value, k_indexer)
    output, loss = kernels.sparse_attention(
        query, key, value, q_indexer, k_indexer, metadata, topk, loss_coeff=loss_coeff
    )
    return _restore_shape(output, rows, packed_input), loss


def run_dense_teacher(
    kernels: CudnnSparseAttentionKernels,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_indexer: torch.Tensor,
    k_indexer: torch.Tensor,
    *,
    loss_coeff: float,
    calculate_per_token_loss: bool,
    layout: Optional[AttentionLayout] = None,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Indexer-only training: frozen dense causal attention and the full-support indexer KL.

    FlashAttention-4 computes the dense attention output and its natural-log LSE without
    materializing probabilities; the teacher is detached because the backbone is frozen. Under
    context parallelism each local query chunk attends to its causal key prefix. Padding query
    rows get zero output and LSE. With a positive ``loss_coeff``, cuDNN evaluates the KL from
    the mean dense attention distribution to the indexer distribution over all causal keys,
    normalized and scaled like the sparse loss.
    """
    packed_input, rows, layout, query, key, value, q_indexer, k_indexer = _prepare_inputs(
        query, key, value, q_indexer, k_indexer, layout, cp_group
    )
    query, key, value = query.detach(), key.detach(), value.detach()
    key, value, k_indexer = _gather_keys(layout, cp_group, key, value, k_indexer)
    with torch.no_grad():
        teacher_key, teacher_value, cu_q, cu_k, max_q, max_k = _phase1_flash_inputs(
            key, value, layout
        )
        output, lse = kernels.dense_attention(
            query.contiguous(),
            teacher_key.contiguous(),
            teacher_value.contiguous(),
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            softmax_scale=_HEAD_DIM**-0.5,
            causal=True,
            return_lse=True,
        )
        # FlashAttention-4 returns the varlen LSE heads-first, [Hq, S]; cuDNN takes [S, Hq].
        lse = lse.transpose(0, 1).contiguous()
        if layout.packed:
            invalid = ~layout.query_valid
            output.masked_fill_(invalid[:, None, None], 0.0)
            lse.masked_fill_(invalid[:, None], 0.0)
    loss = None
    if loss_coeff > 0:
        loss = kernels.dense_indexer_kl(
            query,
            key,
            q_indexer,
            k_indexer,
            **_row_fields(layout, calculate_per_token_loss),
            teacher_softmax_scale=_HEAD_DIM**-0.5,
            student_softmax_scale=_INDEXER_DIM**-0.5,
            loss_coeff=loss_coeff,
            teacher_lse=lse,
            use_kf=True,
        )
    return _restore_shape(output, rows, packed_input), loss
