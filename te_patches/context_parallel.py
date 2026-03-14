# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Context Parallelism."""
import os
import itertools
from typing import List, Union, Tuple
import torch
import transformer_engine_torch as tex

from transformer_engine.pytorch.utils import (
    get_cudnn_version,
    nvtx_range_pop,
    nvtx_range_push,
    get_device_compute_capability,
)
from transformer_engine.pytorch.cpp_extensions.fused_attn import (
    fused_attn_fwd,
    fused_attn_bwd,
    FusedAttnBackend,
)
from transformer_engine.pytorch.quantization import FP8GlobalStateManager
from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor
from transformer_engine.pytorch.quantized_tensor import QuantizedTensorStorage
from transformer_engine.pytorch.jit import jit_fuser
from transformer_engine.pytorch.graph import is_graph_capturing
from transformer_engine.pytorch.constants import (
    dist_group_type,
    TE_DType,
)
from transformer_engine.pytorch.distributed import (
    get_distributed_world_size,
    get_distributed_rank,
    gather_along_first_dim,
    reduce_scatter_along_first_dim,
)

from transformer_engine.pytorch.quantized_tensor import (
    prepare_for_saving,
    restore_from_saved,
)

# Import attention utils
import transformer_engine.pytorch.attention.dot_product_attention.utils as dpa_utils
from transformer_engine.pytorch.attention.dot_product_attention.utils import (
    FlashAttentionUtils as fa_utils,
    combine_and_quantize,
    combine_and_dequantize,
    print_quantizers,
)

_cu_seqlens_info_with_cp_cache = {}
_seq_chunk_ids_cache_for_reordering_before_attn = {}
_seq_chunk_ids_cache_for_reordering_after_attn = {}
_softmax_offset_chunk_ids_cache = {}

# Float8CurrentScaling: fused_attn_bwd takes O in FP8 by default, this flag allows it in F16
_dpa_fp8_cs_o_in_f16 = os.getenv("NVTE_DPA_FP8CS_O_in_F16", "1") == "1"


def flash_attn_p2p_communicate(
    rank, send_tensor, send_dst, recv_tensor, recv_src, cp_group, batch_p2p_comm
):
    """Point-to-point communications of KV and dKV in Attention with context parallelism"""
    send_recv_ops = []

    if batch_p2p_comm:
        if rank % 2 == 0:
            send_op = torch.distributed.P2POp(
                torch.distributed.isend, send_tensor, send_dst, cp_group
            )
            recv_op = torch.distributed.P2POp(
                torch.distributed.irecv, recv_tensor, recv_src, cp_group
            )
            send_recv_ops.append(send_op)
            send_recv_ops.append(recv_op)
        else:
            recv_op = torch.distributed.P2POp(
                torch.distributed.irecv, recv_tensor, recv_src, cp_group
            )
            send_op = torch.distributed.P2POp(
                torch.distributed.isend, send_tensor, send_dst, cp_group
            )
            send_recv_ops.append(recv_op)
            send_recv_ops.append(send_op)
        send_recv_reqs = torch.distributed.batch_isend_irecv(send_recv_ops)
    else:
        if rank % 2 == 0:
            send_op = torch.distributed.isend(send_tensor, send_dst, cp_group)
            recv_op = torch.distributed.irecv(recv_tensor, recv_src, cp_group)
            send_recv_ops.append(send_op)
            send_recv_ops.append(recv_op)
        else:
            recv_op = torch.distributed.irecv(recv_tensor, recv_src, cp_group)
            send_op = torch.distributed.isend(send_tensor, send_dst, cp_group)
            send_recv_ops.append(recv_op)
            send_recv_ops.append(send_op)
        send_recv_reqs = send_recv_ops

    return send_recv_reqs


@jit_fuser
def flash_attn_fwd_out_correction_init(
    out_init_step: torch.Tensor,
    softmax_lse: torch.Tensor,
    softmax_lse_init_step: torch.Tensor,
    seq_dim: int,
):
    """Merge partial outputs of the first step in Attention with context parallelism"""
    softmax_lse_corrected_exp = torch.exp(softmax_lse_init_step - softmax_lse).movedim(2, seq_dim)
    softmax_lse_corrected_exp = softmax_lse_corrected_exp.unsqueeze(-1)
    out_corrected = out_init_step * softmax_lse_corrected_exp
    return out_corrected.to(out_init_step.dtype)


@jit_fuser
def flash_attn_fwd_out_correction(
    out: torch.Tensor,
    out_per_step: torch.Tensor,
    softmax_lse: torch.Tensor,
    softmax_lse_per_step: torch.Tensor,
    seq_dim: int,
):
    """Merge partial outputs of each step in Attention with context parallelism"""
    softmax_lse_corrected_exp = torch.exp(softmax_lse_per_step - softmax_lse).movedim(2, seq_dim)
    softmax_lse_corrected_exp = softmax_lse_corrected_exp.unsqueeze(-1)
    out_corrected = out_per_step * softmax_lse_corrected_exp
    out.add_(out_corrected)


@jit_fuser
def flash_attn_fwd_second_half_out_correction(
    out: torch.Tensor,
    out_per_step: torch.Tensor,
    softmax_lse: torch.Tensor,
    softmax_lse_per_step: torch.Tensor,
    seq_dim: int,
):
    """Merge second half of partial outputs of each step in Attention with context parallelism"""
    out_ = out.select(seq_dim, 1)
    softmax_lse_ = softmax_lse.view(*softmax_lse.shape[:-1], 2, -1)[..., 1, :]
    softmax_lse_corrected_exp = torch.exp(softmax_lse_per_step - softmax_lse_).movedim(2, seq_dim)
    softmax_lse_corrected_exp = softmax_lse_corrected_exp.unsqueeze(-1)
    out_corrected = out_per_step * softmax_lse_corrected_exp
    out_.add_(out_corrected)


@jit_fuser
def flash_attn_fwd_softmax_lse_correction(
    softmax_lse: torch.Tensor,
    softmax_lse_per_step: torch.Tensor,
):
    """Merge softmax stats of each step in Attention with context parallelism"""
    max_scale = torch.max(softmax_lse, softmax_lse_per_step)
    min_scale = torch.min(softmax_lse, softmax_lse_per_step)
    new_scale = max_scale + torch.log1p(torch.exp(min_scale - max_scale))
    softmax_lse.copy_(new_scale)


@jit_fuser
def flash_attn_fwd_second_half_softmax_lse_correction(
    softmax_lse: torch.Tensor,
    softmax_lse_per_step: torch.Tensor,
):
    """Merge second half of softmax stats of each step in Attention with context parallelism"""
    softmax_lse_ = softmax_lse[..., 1, :]
    max_scale = torch.max(softmax_lse_, softmax_lse_per_step)
    min_scale = torch.min(softmax_lse_, softmax_lse_per_step)
    new_scale = max_scale + torch.log1p(torch.exp(min_scale - max_scale))
    softmax_lse_.copy_(new_scale)


@jit_fuser
def get_cu_seqlens_on_cp_rank(
    cu_seqlens: torch.Tensor,
    cu_seqlens_padded_on_cp_rank: torch.Tensor,
    cp_size: int,
    cp_rank: int,
    first_half: bool,
    second_half: bool,
):
    """Compute cu_seqlens of a context parallelism rank"""
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    seqlens_padded = (cu_seqlens_padded_on_cp_rank[1:] - cu_seqlens_padded_on_cp_rank[:-1]) // 2
    zeros = torch.zeros_like(seqlens)
    cu_seqlens_on_cp_rank = torch.zeros_like(cu_seqlens)
    if first_half:
        seqlens_1 = seqlens - cp_rank * seqlens_padded
        seqlens_1 = seqlens_1.clamp(zeros, seqlens_padded)
        cu_seqlens_on_cp_rank[1:].add_(seqlens_1)
    if second_half:
        seqlens_2 = seqlens - (2 * cp_size - cp_rank - 1) * seqlens_padded
        seqlens_2 = seqlens_2.clamp(zeros, seqlens_padded)
        cu_seqlens_on_cp_rank[1:].add_(seqlens_2)
    cu_seqlens_on_cp_rank.cumsum_(dim=0)
    return cu_seqlens_on_cp_rank


@jit_fuser
def get_seq_chunk_ids_for_reordering_before_attn(cp_size, device):
    """
    Context parallelism assigns two discontiguous sequence chunks to each GPU for load balancing.
    To make sure tokens are ordered correctly for compute, we need to reorder sequence chunks to
    be contigupus before attention compute. This function is to compute sequence chunk ids for
    reordering.
    """
    global _seq_chunk_ids_cache_for_reordering_before_attn
    if (cp_size, device) not in _seq_chunk_ids_cache_for_reordering_before_attn:
        chunk_ids = torch.empty(2 * cp_size, dtype=torch.int32, device=device)
        for rank in range(cp_size):
            chunk_ids[rank] = 2 * rank
            chunk_ids[rank + cp_size] = 2 * cp_size - 2 * rank - 1
        _seq_chunk_ids_cache_for_reordering_before_attn[(cp_size, device)] = chunk_ids
    return _seq_chunk_ids_cache_for_reordering_before_attn[(cp_size, device)]


@jit_fuser
def get_seq_chunk_ids_for_reordering_after_attn(cp_size, device):
    """
    Context parallelism assigns two discontiguous sequence chunks to each GPU for load balancing.
    We need to reorder sequence chunks back to discontiguous after attention compute. This function
    is to compute sequence chunk ids for reordering.
    """
    global _seq_chunk_ids_cache_for_reordering_after_attn
    if (cp_size, device) not in _seq_chunk_ids_cache_for_reordering_after_attn:
        chunk_ids = torch.empty(2 * cp_size, dtype=torch.int32, device=device)
        for rank in range(cp_size):
            chunk_ids[2 * rank] = rank
            chunk_ids[2 * rank + 1] = 2 * cp_size - rank - 1
        _seq_chunk_ids_cache_for_reordering_after_attn[(cp_size, device)] = chunk_ids
    return _seq_chunk_ids_cache_for_reordering_after_attn[(cp_size, device)]


@jit_fuser
def reorder_seq_chunks_for_a2a_before_attn(x, chunk_ids_for_a2a, seq_dim, cp_size):
    """Reorder sequence chunk for A2A communication before attention compute."""
    # [cp, b, s, h//cp, d] -> [b, cp, s, h//cp, d]
    # or [cp, s, b, h//cp, d] -> [cp, s, b, h//cp, d]
    x = x.movedim(0, seq_dim).contiguous()
    # [b, cp, s, h//cp, d] -> [b, cp*2, s//2, h//cp, d]
    # or [cp, s, b, h//cp, d] -> [cp*2, s//2, b, h//cp, d]
    x = x.view(*x.shape[:seq_dim], cp_size * 2, -1, *x.shape[(seq_dim + 2) :])
    # reorder the sequence chunks
    x = torch.index_select(x, dim=seq_dim, index=chunk_ids_for_a2a)
    return x


@jit_fuser
def reorder_seq_chunks_for_a2a_after_attn(x, chunk_ids_for_a2a, seq_dim, cp_size):
    """Reorder sequence chunk for A2A communication after attention compute."""
    # [b, cp*2, s//2, h//cp, d] -> [cp*2, b, s//2, h//cp, d]
    # or [cp*2, s//2, b, h//cp, d] -> [cp*2, s//2, b, h//cp, d]
    x = x.movedim(seq_dim, 0).contiguous()
    # reorder the sequence chunks
    x = torch.index_select(x, dim=0, index=chunk_ids_for_a2a)
    # [cp*2, b, s//2, h//cp, d] -> [cp, 2, b, s//2, h//cp, d]
    # or [cp*2, s//2, b, h//cp, d] -> [cp, 2, s//2, b, h//cp, d]
    x = x.view(cp_size, 2, *x.shape[1:])
    return x


def reorder_seq_chunks_before_a2a_after_attn_thd(x, cu_seqlens, cp_size, seq_dim=0):
    """
    Reorder sequence chunks for A2A communication that happens after attention
    compute.

    Args:
        x:              The input tensor to be reordered.
        cu_seqlens:     The cumulative sequence lengths of the input tensor.
        cp_size:        The number of ranks participating in context parallelism.
        seq_dim:        The dimension in which to reorder.

    Returns:
        The reordered tensor.

    Example:
        x:              [ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  0.,  1.,  2.,  3.,  4.,  5.,
                          6.,  7.,  0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  0.,  1.,  2.,  3.,
                          4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14., 15.]
        cu_seqlens:     [ 0, 8, 16, 24, 40]
        cp_size:        4

        Returns:        [ 0.,  7.,  0.,  7.,  0.,  7.,  0.,  1., 14., 15.,  1.,  6.,  1.,  6.,
                          1.,  6.,  2.,  3., 12., 13.,  2.,  5.,  2.,  5.,  2.,  5.,  4.,  5.,
                          10., 11.,  3.,  4.,  3.,  4.,  3.,  4.,  6.,  7.,  8.,  9.]


        This logic is similar to how the DualChunking is done to split the sequence
        for each rank. Here, the indices of sequence chunks for all those ranks
        are concatenated together. So the returned tensor ends up looking like as if
        the chunks from all the ranks are concatenated together.

         e.g. [
                0.,  7.,  0.,  7.,  0.,  7.,  0.,  1., 14., 15.,  # chunk on rank 0
                1.,  6.,  1.,  6.,  1.,  6.,  2.,  3., 12., 13.,  # chunk on rank 1
                2.,  5.,  2.,  5.,  2.,  5.,  4.,  5., 10., 11.,  # chunk on rank 2
                3.,  4.,  3.,  4.,  3.,  4.,  6.,  7.,  8.,  9.   # chunk on rank 3
             ]
    """
    total_slices_of_any_sequence = 2 * cp_size
    slice_sizes = (cu_seqlens[1:] - cu_seqlens[:-1]) // total_slices_of_any_sequence

    indices = [
        (
            # 1st segment
            torch.arange(
                seq_start + (cp_rank * slice_size),
                seq_start + ((cp_rank + 1) * slice_size),
                device=cu_seqlens.device,
            ),
            # 2nd segment
            torch.arange(
                seq_start + ((total_slices_of_any_sequence - cp_rank - 1) * slice_size),
                seq_start + ((total_slices_of_any_sequence - cp_rank) * slice_size),
                device=cu_seqlens.device,
            ),
        )
        for cp_rank in range(cp_size)
        for slice_size, seq_start in zip(slice_sizes, cu_seqlens[:-1])
    ]

    # flatten the list of tuples to a list
    indices = list(itertools.chain(*indices))
    indices = torch.cat(indices)
    return x.index_select(seq_dim, indices)


def reorder_seq_chunks_after_a2a_before_attn_thd(x, cu_seqlens, seq_chunk_ids, cp_size, seq_dim=0):
    """
    Reorder sequence chunks for A2A communication that happens before attention
    compute.

    Args:
        x:              The input tensor to be reordered.
        cu_seqlens:     The cumulative sequence lengths of the input tensor.
        seq_chunk_ids:  The sequence chunk ids of the input `x` which is to be reordered.
        cp_size:        The number of ranks participating in context parallelism.
        seq_dim:        The dimension in which to reorder.

    Returns:
        The reordered tensor.

    Example:
        x:              [ 0.,  7.,  0.,  7.,  0.,  7.,  0.,  1., 14., 15.,  1.,  6.,  1.,  6.,
                          1.,  6.,  2.,  3., 12., 13.,  2.,  5.,  2.,  5.,  2.,  5.,  4.,  5.,
                          10., 11.,  3.,  4.,  3.,  4.,  3.,  4.,  6.,  7.,  8.,  9.]
        cu_seqlens:     [ 0,  8, 16, 24, 40]
        seq_chunk_ids:  [ 0, 2, 4, 6, 7, 5, 3, 1]
        cp_size:        4

        Returns:        [ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  0.,  1.,  2.,  3.,  4.,  5.,
                          6.,  7.,  0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  0.,  1.,  2.,  3.,
                          4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14., 15.]

        Note that the input sequences (x) are arranged after A2A communication as if DualChunked
        chunks on all the ranks are concatenated together in the `seq_dim`.

        e.g. [
                0.,  7.,  0.,  7.,  0.,  7.,  0.,  1., 14., 15.,  # chunk on rank 0
                1.,  6.,  1.,  6.,  1.,  6.,  2.,  3., 12., 13.,  # chunk on rank 1
                2.,  5.,  2.,  5.,  2.,  5.,  4.,  5., 10., 11.,  # chunk on rank 2
                3.,  4.,  3.,  4.,  3.,  4.,  6.,  7.,  8.,  9.   # chunk on rank 3
             ]

        Then the logic to serialize the sequences is:
        1. For every sequence segment on any rank (denoted by `start` and `end`):
            1a. For every chunk (in `chunk_id` and the total of those are twice as many as the number of CP ranks) :
                1aa. The first `cp_size` number of chunks form the first half of the whole sequence. Get those indices.
                1ab. The second `cp_size` number of chunks form the second half of the whole sequence. Get those indices.
            1b. Concatenate the indices of the first half and the second half.
        2. Reorder the entire input tensor by those indices.
    """

    max_cum_seqlen_per_cp_rank = cu_seqlens[-1] // cp_size
    cu_seqlens_on_any_cp_rank = cu_seqlens // cp_size

    # Go through all the sequence segments (the sizes should be the same from all the ranks)
    indices = [
        torch.arange(
            # Calculate 'left' boundary
            (
                start + max_cum_seqlen_per_cp_rank * (chunk_id // 2)
                if loc < cp_size
                else (start + end) // 2 + max_cum_seqlen_per_cp_rank * (chunk_id // 2)
            ),
            # Calculate 'right' boundary
            (
                (start + end) // 2 + max_cum_seqlen_per_cp_rank * (chunk_id // 2)
                if loc < cp_size
                else end + max_cum_seqlen_per_cp_rank * (chunk_id // 2)
            ),
            device=cu_seqlens.device,
        )
        for start, end in zip(cu_seqlens_on_any_cp_rank[:-1], cu_seqlens_on_any_cp_rank[1:])
        for loc, chunk_id in enumerate(seq_chunk_ids)
    ]

    indices = torch.cat(indices)
    return x.index_select(seq_dim, indices)


def flash_attn_a2a_communicate(
    a2a_inputs: Union[torch.Tensor, List[torch.Tensor]],
    chunk_ids_for_a2a: torch.Tensor,
    seq_dim: int,
    cp_size: int,
    cp_group: dist_group_type,
    cp_stream: torch.cuda.Stream,
    before_attn: bool,
    qkv_format: str = "bshd",
    cu_seqlens_padded: torch.Tensor = None,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """A2A communication for context parallelism."""

    assert (
        qkv_format != "thd" or cu_seqlens_padded is not None
    ), "cu_seqlens_padded is required for THD format!"
    a2a_inputs = [a2a_inputs] if not isinstance(a2a_inputs, list) else a2a_inputs
    a2a_outputs, a2a_reqs = [None] * len(a2a_inputs), [None] * len(a2a_inputs)
    if before_attn:
        for i in range(len(a2a_inputs) + 2):
            if 0 < i < len(a2a_inputs) + 1:
                a2a_outputs[i - 1] = torch.empty_like(a2a_inputs[i - 1])
                a2a_reqs[i - 1] = torch.distributed.all_to_all_single(
                    a2a_outputs[i - 1], a2a_inputs[i - 1], group=cp_group, async_op=True
                )
            if i > 1:
                with torch.cuda.stream(cp_stream):
                    a2a_reqs[i - 2].wait()
                    x = a2a_outputs[i - 2]
                    if qkv_format in ["bshd", "sbhd"]:
                        # reorder the sequence chunks
                        x = reorder_seq_chunks_for_a2a_before_attn(
                            x, chunk_ids_for_a2a, seq_dim, cp_size
                        )
                        # [b, cp*2, s//2, np//cp, hn] -> [b, cp*s, np//cp, hn]
                        # or [cp*2, s//2, b, np//cp, hn] -> [cp*s, b, np//cp, hn]
                        a2a_outputs[i - 2] = x.view(
                            *x.shape[:seq_dim], -1, *x.shape[(seq_dim + 2) :]
                        )
                    else:  # qkv_format == "thd"
                        # [cp, t, np//cp, hn] -> [cp*t, np//cp, hn]
                        x = x.view(-1, *x.shape[2:])
                        # reorder the sequence chunks
                        a2a_outputs[i - 2] = reorder_seq_chunks_after_a2a_before_attn_thd(
                            x, cu_seqlens_padded, chunk_ids_for_a2a, cp_size
                        )

            if i < len(a2a_inputs):
                x = a2a_inputs[i]
                # [b, s, np, hn] -> [b, s, cp, np//cp, hn]
                # or [s, b, np, hn] -> [s, b, cp, np//cp, hn]
                # or [t, np, hn] -> [t, cp, np//cp, hn]
                x = x.view(*x.shape[:-2], cp_size, x.shape[-2] // cp_size, x.shape[-1])
                # [b, s, cp, np//cp, hn] -> [cp, b, s, np//cp, hn]
                # or [s, b, cp, np//cp, hn] -> [cp, s, b, np//cp, hn]
                # or [t, cp, np//cp, hn] -> [cp, t, np//cp, hn]
                a2a_inputs[i] = x.movedim(-3, 0).contiguous()
    else:
        for i in range(len(a2a_inputs) + 2):
            if 0 < i < len(a2a_inputs) + 1:
                a2a_outputs[i - 1] = torch.empty_like(a2a_inputs[i - 1])
                a2a_reqs[i - 1] = torch.distributed.all_to_all_single(
                    a2a_outputs[i - 1], a2a_inputs[i - 1], group=cp_group, async_op=True
                )
            if i < len(a2a_inputs):
                x = a2a_inputs[i]
                if qkv_format in ["bshd", "sbhd"]:
                    # [b, cp*s, np//cp, hn] -> [b, cp*2, s//2, np//cp, hn]
                    # or [cp*s, b, np//cp, hn] -> [cp*2, s//2, b, np//cp, hn]
                    x = x.view(*x.shape[:seq_dim], cp_size * 2, -1, *x.shape[(seq_dim + 1) :])
                    # reorder the sequence chunks
                    a2a_inputs[i] = reorder_seq_chunks_for_a2a_after_attn(
                        x, chunk_ids_for_a2a, seq_dim, cp_size
                    )
                else:  # qkv_format == "thd"
                    # reorder the sequence chunks
                    x = reorder_seq_chunks_before_a2a_after_attn_thd(x, cu_seqlens_padded, cp_size)
                    # [cp*t, np//cp, hn] -> [cp, t, np//cp, hn]
                    a2a_inputs[i] = x.view(cp_size, -1, *x.shape[-2:])
            if i > 1:
                with torch.cuda.stream(cp_stream):
                    a2a_reqs[i - 2].wait()
                    x = a2a_outputs[i - 2]
                    # [cp, 2, b, s//2, np//cp, hn] -> [b, 2, s//2, cp, np//cp, hn]
                    # or [cp, 2, s//2, b, np//cp, hn] -> [2, s//2, b, cp, np//cp, hn]
                    # or [cp, t, np//cp, hn] -> [t, cp, np//cp, hn]
                    x = x.movedim(0, -3).movedim(0, seq_dim).contiguous()
                    # [b, 2, s//2, cp, np//cp, hn] -> [b*s, np, hn]
                    # or [2, s//2, b, cp, np//cp, hn] -> [s*b, np, hn]
                    # or [t, cp, np//cp, hn] -> [t, np, hn]
                    a2a_outputs[i - 2] = x.view(-1, x.shape[-3] * x.shape[-2], x.shape[-1])
    torch.cuda.current_stream().wait_stream(cp_stream)
    return a2a_outputs[0] if len(a2a_inputs) == 1 else a2a_outputs


def flash_attn_a2a_communicate_softmax_offset(
    tensor: torch.Tensor,
    h_dim: int,
    cp_size: int,
    cp_group: dist_group_type,
    cp_stream: torch.cuda.Stream,
    before_attn: bool,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Split/AllGather communication for softmax offset."""
    if tensor is None:
        return None

    global _softmax_offset_chunk_ids_cache
    device = tensor.device
    if (cp_size, device) not in _softmax_offset_chunk_ids_cache:
        chunk_ids = torch.arange(cp_size, dtype=torch.int32, device=device)
        _softmax_offset_chunk_ids_cache[(cp_size, device)] = chunk_ids
    else:
        chunk_ids = _softmax_offset_chunk_ids_cache[(cp_size, device)]

    if before_attn:
        # softmax_offset: split round-robin to CP ranks
        # [1, h, 1, 1] -> [1, cp, h//cp, 1, 1]
        shape = tensor.shape
        tensor = tensor.view(
            *shape[:h_dim], cp_size, shape[h_dim] // cp_size, *shape[(h_dim + 1) :]
        )
        rank = get_distributed_rank(cp_group)
        output = torch.index_select(tensor, dim=h_dim, index=chunk_ids[rank])
        output = output.view(*shape[:h_dim], -1, *shape[(h_dim + 1) :])
    else:
        # d_softmax_offset: all-gather from all ranks to all ranks
        # [1, h//cp, 1, 1] -> [1, h, 1, 1]
        inp = tensor.view(-1)
        output = torch.empty(cp_size * inp.shape[0], dtype=tensor.dtype, device=device)
        with torch.cuda.stream(cp_stream):
            torch.distributed.all_gather_into_tensor(
                output,
                inp,
                group=cp_group,
                async_op=False,
            )
        torch.cuda.current_stream().wait_stream(cp_stream)
        output = output.view(
            *tensor.shape[:h_dim], cp_size * tensor.shape[h_dim], *tensor.shape[h_dim + 1 :]
        )
    return output


def _get_cu_seqlens_info_with_cp(
    batch_size: int,
    max_seqlen: int,
    cp_size: int,
    cu_seqlens: torch.Tensor,
):
    """Cumulative sequence lengths with CP being considered."""
    global _cu_seqlens_info_with_cp_cache
    if (batch_size, max_seqlen, cp_size) not in _cu_seqlens_info_with_cp_cache:
        _cu_seqlens_info_with_cp_cache[(batch_size, max_seqlen, cp_size)] = (
            cu_seqlens // cp_size,
            cu_seqlens // (cp_size * 2),
        )
    return _cu_seqlens_info_with_cp_cache[(batch_size, max_seqlen, cp_size)]


def get_fa_args(
    forward: bool,
    use_flash_attn_3: bool,
    qkv_format: str,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    max_seqlen_q=None,
    max_seqlen_kv=None,
    dq=None,
    dk=None,
    dv=None,
):
    """Get forward/backward arguments for flash-attn v2 and v3."""
    if use_flash_attn_3:
        if forward:
            if qkv_format == "thd":
                return [
                    *[None] * 4,  # k_new, v_new, qv, out
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    *[None] * 3,  # cu_seqlens_k_new, seqused_q, seqused_k
                    max_seqlen_q,
                    max_seqlen_kv,
                    *[None]
                    * 9,  # page_table, kv_batch_idx, leftpad_k, rotary_cos, rotary_sin, seqlens_rotary, q_descale, k_descale, v_descale
                ]
            return [
                *[None]
                * 9,  # k_new, v_new, qv, out, cu_seqlens_q, cu_seqlens_kv, cu_seqlens_k_new, seqused_q, seqused_k
                max_seqlen_q,
                max_seqlen_kv,
                *[None]
                * 9,  # page_table, kv_batch_idx, leftpad_k, rotary_cos, rotary_sin, seqlens_rotary, q_descale, k_descale, v_descale
            ]
        if qkv_format == "thd":
            return [
                cu_seqlens_q,
                cu_seqlens_kv,
                None,  # sequed_q
                None,  # sequed_k
                max_seqlen_q,
                max_seqlen_kv,
                dq,
                dk,
                dv,
            ]
        return [
            None,  # cu_seqlens_q
            None,  # cu_seqlens_kv
            None,  # sequed_q
            None,  # sequed_k
            max_seqlen_q,
            max_seqlen_kv,
            dq,
            dk,
            dv,
        ]
    if forward:
        if qkv_format == "thd":
            return [
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
            ]
        return []
    if qkv_format == "thd":
        return [
            dq,
            dk,
            dv,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
        ]
    return [
        dq,
        dk,
        dv,
    ]


def cp_p2p_fwd_prepare_qkv(
    q_part,
    k_part,
    v_part,
    qkv_format,
    pad_between_seqs,
    cu_seqlens_q,
    cu_seqlens_kv,
    cu_seqlens_q_padded,
    cu_seqlens_kv_padded,
    cu_seqlens_q_half,
    cu_seqlens_kv_half,
    rank,
    step,
    cp_size,
    section,
):
    """Prepare q, k, v and cu_seqlens for CP P2P forward"""
    cu_seqlens_q_per_step = None
    cu_seqlens_kv_per_step = None
    if section in ["diagonal", "all"]:
        if pad_between_seqs:
            cu_seqlens_q_per_step = get_cu_seqlens_on_cp_rank(
                cu_seqlens_q, cu_seqlens_q_padded, cp_size, rank, True, True
            )
            rank_ = rank if section == "diagonal" else (rank - step) % cp_size
            cu_seqlens_kv_per_step = get_cu_seqlens_on_cp_rank(
                cu_seqlens_kv, cu_seqlens_kv_padded, cp_size, rank_, True, True
            )
        elif qkv_format == "thd":
            cu_seqlens_q_per_step = cu_seqlens_q // cp_size
            cu_seqlens_kv_per_step = cu_seqlens_kv // cp_size
        else:
            cu_seqlens_q_per_step = cu_seqlens_q
            cu_seqlens_kv_per_step = cu_seqlens_kv

        if qkv_format == "bshd":
            # [b, 2, s//2, h, d] -> [b, s, h, d]
            q_part, k_part, v_part = [
                x.view(x.shape[0], -1, *x.shape[-2:]) for x in [q_part, k_part, v_part]
            ]
        elif qkv_format == "sbhd":
            # [2, s//2, b, h, d] -> [s, b, h, d]
            q_part, k_part, v_part = [x.view(-1, *x.shape[-3:]) for x in [q_part, k_part, v_part]]

    elif section == "lower-triangle":
        if pad_between_seqs:
            cu_seqlens_q_per_step = get_cu_seqlens_on_cp_rank(
                cu_seqlens_q, cu_seqlens_q_padded, cp_size, rank, True, True
            )
            cu_seqlens_kv_per_step = get_cu_seqlens_on_cp_rank(
                cu_seqlens_kv,
                cu_seqlens_kv_padded,
                cp_size,
                (rank - step) % cp_size,
                True,
                False,
            )
        elif qkv_format == "thd":
            cu_seqlens_q_per_step = cu_seqlens_q // cp_size
            cu_seqlens_kv_per_step = cu_seqlens_kv // (cp_size * 2)
        else:
            cu_seqlens_q_per_step = cu_seqlens_q
            cu_seqlens_kv_per_step = cu_seqlens_kv_half

        if qkv_format == "bshd":
            # [b, 2, sq//2, h, d] -> [b, sq, h, d]
            q_part = q_part.view(q_part.shape[0], -1, *q_part.shape[-2:])
            # [b, 2, sk//2, h, d] -> [b, sk//2, h, d]
            k_part = k_part[:, 0, ...]
            v_part = v_part[:, 0, ...]
        elif qkv_format == "sbhd":
            # [2, sq//2, b, h, d] -> [sq, b, h, d]
            q_part = q_part.view(-1, *q_part.shape[-3:])
            # [2, sk//2, b, h, d] -> [sk//2, b, h, d]
            k_part = k_part[0]
            v_part = v_part[0]
        elif qkv_format == "thd":
            # [t, h, d] -> [t/2, h, d]
            k_part = tex.thd_read_half_tensor(k_part, cu_seqlens_kv_padded, 0)
            v_part = tex.thd_read_half_tensor(v_part, cu_seqlens_kv_padded, 0)

    elif section == "upper-triangle":
        if pad_between_seqs:
            cu_seqlens_q_per_step = get_cu_seqlens_on_cp_rank(
                cu_seqlens_q, cu_seqlens_q_padded, cp_size, rank, False, True
            )
            cu_seqlens_kv_per_step = get_cu_seqlens_on_cp_rank(
                cu_seqlens_kv,
                cu_seqlens_kv_padded,
                cp_size,
                (rank - step) % cp_size,
                True,
                True,
            )
        elif qkv_format == "thd":
            cu_seqlens_q_per_step = cu_seqlens_q // (cp_size * 2)
            cu_seqlens_kv_per_step = cu_seqlens_kv // cp_size
        else:
            cu_seqlens_q_per_step = cu_seqlens_q_half
            cu_seqlens_kv_per_step = cu_seqlens_kv

        if qkv_format == "bshd":
            # [b, 2, sq//2, h, d] -> [b, sq//2, h, d]
            q_part = q_part[:, 1, ...]
            # [b, 2, sk//2, h, d] -> [b, sk, h, d]
            k_part, v_part = [x.view(x.shape[0], -1, *x.shape[-2:]) for x in [k_part, v_part]]
        elif qkv_format == "sbhd":
            # [2, sq//2, b, h, d] -> [sq//2, b, h, d]
            q_part = q_part[1]
            # [2, sk//2, b, h, d] -> [sk, b, h, d]
            k_part, v_part = [x.view(-1, *x.shape[-3:]) for x in [k_part, v_part]]
        elif qkv_format == "thd":
            # [t, h, d] -> [t/2, h, d]
            q_part = tex.thd_read_half_tensor(q_part, cu_seqlens_q_padded, 1)

    return q_part, k_part, v_part, cu_seqlens_q_per_step, cu_seqlens_kv_per_step


def cp_p2p_fwd_fused_attn(
    attn_bias,
    attn_bias_,
    is_training,
    max_seqlen_q,
    max_seqlen_kv,
    cu_seqlens_q_padded,
    cu_seqlens_kv_padded,
    fused_attn_backend,
    softmax_scale,
    dropout_p,
    qkv_layout,
    attn_mask_type,
    attn_bias_type,
    fp8,
    q_fp8,
    k_fp8,
    v_fp8,
    fwd_nominal_dtype,
    S_quantizer_per_step,
    O_quantizer_per_step,
    rank,
    step,
    cp_size,
    return_max_logit,
    q_part,
    k_part,
    v_part,
    cu_seqlens_q_per_step,
    cu_seqlens_kv_per_step,
    section,
):
    """Per-tile forward call of CP P2P with FusedAttention backend"""
    attn_bias_inputs = None
    max_seqlen_q_ = None
    max_seqlen_kv_ = None
    cu_seqlens_q_ = None
    cu_seqlens_kv_ = None
    attn_mask_type_ = None
    cu_seqlens_q_padded_ = None
    cu_seqlens_kv_padded_ = None
    if section in ["diagonal", "all"]:
        if attn_bias is not None:
            idx = (rank - step) % cp_size
            attn_bias_inputs = torch.cat(
                (
                    attn_bias[..., idx, :],
                    attn_bias[..., (2 * cp_size - idx - 1), :],
                ),
                dim=-1,
            ).contiguous()
        max_seqlen_q_ = max_seqlen_q
        max_seqlen_kv_ = max_seqlen_kv
        cu_seqlens_q_ = cu_seqlens_q_per_step
        cu_seqlens_kv_ = cu_seqlens_kv_per_step
        attn_mask_type_ = attn_mask_type
        cu_seqlens_q_padded_ = cu_seqlens_q_padded
        cu_seqlens_kv_padded_ = cu_seqlens_kv_padded
    elif section == "lower-triangle":
        k_part = k_part.contiguous()
        v_part = v_part.contiguous()
        if attn_bias is not None:
            idx = (rank - step) % cp_size
            attn_bias_inputs = attn_bias[..., idx, :].contiguous()
        max_seqlen_q_ = max_seqlen_q
        max_seqlen_kv_ = max_seqlen_kv // 2
        cu_seqlens_q_ = cu_seqlens_q_per_step
        cu_seqlens_kv_ = cu_seqlens_kv_per_step
        attn_mask_type_ = "padding" if "padding" in attn_mask_type else "no_mask"
        cu_seqlens_q_padded_ = cu_seqlens_q_padded
        cu_seqlens_kv_padded_ = (
            cu_seqlens_kv_padded // 2 if cu_seqlens_kv_padded is not None else None
        )
    elif section == "upper-triangle":
        q_part = q_part.contiguous()
        if attn_bias is not None:
            idx = (rank - step) % cp_size
            attn_bias_inputs = torch.cat(
                (
                    attn_bias_[..., 1, :, idx, :],
                    attn_bias_[..., 1, :, (2 * cp_size - idx - 1), :],
                ),
                dim=-1,
            ).contiguous()
        max_seqlen_q_ = max_seqlen_q // 2
        max_seqlen_kv_ = max_seqlen_kv
        cu_seqlens_q_ = cu_seqlens_q_per_step
        cu_seqlens_kv_ = cu_seqlens_kv_per_step
        attn_mask_type_ = "padding" if "padding" in attn_mask_type else "no_mask"
        cu_seqlens_q_padded_ = cu_seqlens_q_padded // 2 if cu_seqlens_q_padded is not None else None
        cu_seqlens_kv_padded_ = cu_seqlens_kv_padded

    fp8_meta_kwargs = {}
    if fp8:
        q_part, k_part, v_part = [
            Float8Tensor.make_like(x, data=y, dtype=fwd_nominal_dtype)
            for x, y in zip([q_fp8, k_fp8, v_fp8], [q_part, k_part, v_part])
        ]
        fp8_meta_kwargs["s_quantizer"] = S_quantizer_per_step
        fp8_meta_kwargs["o_quantizer"] = O_quantizer_per_step

    out_per_step, aux_ctx_tensors, *max_logit = fused_attn_fwd(
        is_training,
        max_seqlen_q_,
        max_seqlen_kv_,
        cu_seqlens_q_,
        cu_seqlens_kv_,
        q_part,
        k_part,
        v_part,
        fake_dtype=fwd_nominal_dtype,
        fused_attention_backend=fused_attn_backend,
        attn_scale=softmax_scale,
        dropout=dropout_p,
        qkv_layout=qkv_layout,
        attn_mask_type=attn_mask_type_,
        attn_bias_type=attn_bias_type,
        attn_bias=attn_bias_inputs,
        cu_seqlens_q_padded=cu_seqlens_q_padded_,
        cu_seqlens_kv_padded=cu_seqlens_kv_padded_,
        **fp8_meta_kwargs,
        return_max_logit=return_max_logit,
        cuda_graph=is_graph_capturing(),
    )

    if fp8:
        softmax_lse_per_step, _, rng_states = aux_ctx_tensors
    else:
        softmax_lse_per_step, rng_states, *rest = aux_ctx_tensors
        attn_bias = rest[0] if len(rest) > 0 else None

    if return_max_logit:
        return out_per_step, softmax_lse_per_step, rng_states, attn_bias, *max_logit
    return out_per_step, softmax_lse_per_step, rng_states, attn_bias, None


def cp_p2p_fwd_flash_attn(
    use_flash_attn_3,
    qkv_format,
    fa_forward_kwargs,
    flash_attn_fwd,
    max_seqlen_q,
    max_seqlen_kv,
    q_part,
    k_part,
    v_part,
    cu_seqlens_q_per_step,
    cu_seqlens_kv_per_step,
    section,
):
    """Per-tile forward call of CP P2P with FlashAttention backend"""
    cu_seqlens_q_ = cu_seqlens_q_per_step
    cu_seqlens_kv_ = cu_seqlens_kv_per_step
    max_seqlen_q_ = max_seqlen_q
    max_seqlen_kv_ = max_seqlen_kv
    causal_ = False
    if section in ["diagonal", "all"]:
        causal_ = section == "diagonal"
    elif section == "lower-triangle":
        max_seqlen_kv_ = max_seqlen_kv // 2
    elif section == "upper-triangle":
        max_seqlen_q_ = max_seqlen_q // 2
    if section in ["lower-triangle", "upper-triangle"]:
        if use_flash_attn_3 or (fa_utils.v2_3_plus and not fa_utils.v2_7_0_plus):
            fa_forward_kwargs["window_size"] = (-1, -1)
        elif fa_utils.v2_7_0_plus:
            fa_forward_kwargs["window_size_left"] = -1
            fa_forward_kwargs["window_size_right"] = -1

    fa_forward_args_thd = get_fa_args(
        True,
        use_flash_attn_3,
        qkv_format,
        cu_seqlens_q=cu_seqlens_q_,
        cu_seqlens_kv=cu_seqlens_kv_,
        max_seqlen_q=max_seqlen_q_,
        max_seqlen_kv=max_seqlen_kv_,
    )
    fa_outputs = flash_attn_fwd(
        q_part,
        k_part,
        v_part,
        *fa_forward_args_thd,
        causal=causal_,
        **fa_forward_kwargs,
    )
    rng_states = None
    if not fa_utils.v2_7_0_plus:
        out_per_step = fa_outputs[4]
        softmax_lse_per_step = fa_outputs[5]
        if not use_flash_attn_3:
            rng_states = fa_outputs[7]
    else:
        out_per_step = fa_outputs[0]
        softmax_lse_per_step = fa_outputs[1]
        if not use_flash_attn_3:
            rng_states = fa_outputs[3]

    return out_per_step, softmax_lse_per_step, rng_states


def cp_p2p_bwd_prepare_qkv(
    q_part,
    k_part,
    v_part,
    out_part,
    dout_part,
    qkv_format,
    cu_seqlens_q_padded,
    cu_seqlens_kv_padded,
    section,
):
    """Prepare q, k, v and cu_seqlens for CP P2P backward"""
    if section in ["diagonal", "all"]:
        if qkv_format == "bshd":
            # [b, 2, s//2, h, d] -> [b, s, h, d]
            q_part, k_part, v_part, out_part, dout_part = [
                x.view(x.shape[0], -1, *x.shape[-2:])
                for x in [q_part, k_part, v_part, out_part, dout_part]
            ]
        elif qkv_format == "sbhd":
            # [2, s//2, b, h, d] -> [s, b, h, d]
            q_part, k_part, v_part, out_part, dout_part = [
                x.view(-1, *x.shape[-3:]) for x in [q_part, k_part, v_part, out_part, dout_part]
            ]
    elif section == "lower-triangle":
        if qkv_format == "bshd":
            # [b, 2, sq//2, h, d] -> [b, sq, h, d]
            q_part, out_part, dout_part = [
                x.view(x.shape[0], -1, *x.shape[-2:]) for x in [q_part, out_part, dout_part]
            ]
            # [b, 2, sk//2, h, d] -> [b, sk, h, d]
            k_part = k_part[:, 0]
            v_part = v_part[:, 0]
        elif qkv_format == "sbhd":
            # [2, sq//2, b, h, d] -> [sq, b, h, d]
            q_part, out_part, dout_part = [
                x.view(-1, *x.shape[-3:]) for x in [q_part, out_part, dout_part]
            ]
            # [2, sk//2, b, h, d] -> [sk, b, h, d]
            k_part = k_part[0]
            v_part = v_part[0]
        elif qkv_format == "thd":
            # [t, h, d] -> [t/2, h, d]
            k_part = tex.thd_read_half_tensor(k_part, cu_seqlens_kv_padded, 0)
            v_part = tex.thd_read_half_tensor(v_part, cu_seqlens_kv_padded, 0)
    elif section == "upper-triangle":
        if qkv_format == "bshd":
            # [b, 2, sq//2, h, d] -> [b, sq//2, h, d]
            q_part, out_part, dout_part = q_part[:, 1], out_part[:, 1], dout_part[:, 1]
            # [b, 2, sk//2, h, d] -> [b, sk, h, d]
            k_part, v_part = [x.view(x.shape[0], -1, *x.shape[-2:]) for x in [k_part, v_part]]
        elif qkv_format == "sbhd":
            # [2, sq//2, b, h, d] -> [sq//2, b, h, d]
            q_part, out_part, dout_part = q_part[1], out_part[1], dout_part[1]
            # [2, sk//2, b, h, d] -> [sk, b, h, d]
            k_part, v_part = [x.view(-1, *x.shape[-3:]) for x in [k_part, v_part]]
        elif qkv_format == "thd":
            # [t, h, d] -> [t/2, h, d]
            q_part, out_part, dout_part = [
                tex.thd_read_half_tensor(x, cu_seqlens_q_padded, 1)
                for x in [q_part, out_part, dout_part]
            ]

    return q_part, k_part, v_part, out_part, dout_part


def cp_p2p_bwd_fused_attn(
    fp8,
    fp8_recipe,
    q_fp8,
    kv_fp8,
    out_fp8,
    dout_fp8,
    softmax_lse,
    softmax_lse_,
    rng_states,
    attn_dbias,
    attn_biases,
    max_seqlen_q,
    max_seqlen_kv,
    step,
    cp_size,
    cu_seqlens_q_per_step,
    cu_seqlens_kv_per_step,
    cu_seqlens_q_padded,
    cu_seqlens_kv_padded,
    fused_attn_backend,
    softmax_scale,
    dropout_p,
    qkv_layout,
    attn_mask_type,
    attn_bias_type,
    deterministic,
    fwd_nominal_dtype,
    bwd_nominal_dtype,
    bwd_output_te_dtype,
    S_quantizer,
    dP_quantizer_per_step,
    dQKV_quantizer_per_step,
    q_part,
    k_part,
    v_part,
    out_part,
    dout_part,
    section,
):
    """Per-tile backward call of CP P2P with FusedAttention backend"""
    if fp8:
        aux_tensors = [
            softmax_lse,
            softmax_lse,
            rng_states[cp_size - step - 1],
        ]
    else:
        aux_tensors = [softmax_lse, rng_states[cp_size - step - 1]]

    max_seqlen_q_ = max_seqlen_q
    max_seqlen_kv_ = max_seqlen_kv
    cu_seqlens_q_padded_ = cu_seqlens_q_padded
    cu_seqlens_kv_padded_ = cu_seqlens_kv_padded
    attn_mask_type_ = attn_mask_type

    if section == "lower-triangle":
        k_part = k_part.contiguous()
        v_part = v_part.contiguous()
        max_seqlen_kv_ = max_seqlen_kv // 2
        cu_seqlens_kv_padded_ = None if cu_seqlens_kv_padded is None else cu_seqlens_kv_padded // 2
        attn_mask_type_ = "padding" if "padding" in attn_mask_type else "no_mask"
    elif section == "upper-triangle":
        q_part, out_part, dout_part = [x.contiguous() for x in [q_part, out_part, dout_part]]
        if fp8:
            aux_tensors = [
                softmax_lse_,
                softmax_lse_,
                rng_states[cp_size - step - 1],
            ]
        else:
            aux_tensors = [softmax_lse_, rng_states[cp_size - step - 1]]

        max_seqlen_q_ = max_seqlen_q // 2
        cu_seqlens_q_padded_ = None if cu_seqlens_q_padded is None else cu_seqlens_q_padded // 2
        attn_mask_type_ = "padding" if "padding" in attn_mask_type else "no_mask"

    if attn_dbias is not None:
        aux_tensors += [attn_biases[cp_size - step - 1]]

    fp8_meta_kwargs = {}
    if fp8:
        q_part, k_part, v_part = [
            Float8Tensor.make_like(x, data=y, dtype=fwd_nominal_dtype)
            for x, y in zip(
                [q_fp8, kv_fp8, kv_fp8],
                [q_part, k_part, v_part],
            )
        ]
        if not (fp8_recipe.float8_current_scaling() and _dpa_fp8_cs_o_in_f16):
            out_part = Float8Tensor.make_like(out_fp8, data=out_part, dtype=fwd_nominal_dtype)
        dout_part = Float8Tensor.make_like(dout_fp8, data=dout_part, dtype=bwd_nominal_dtype)
        fp8_meta_kwargs["s_quantizer"] = S_quantizer
        fp8_meta_kwargs["dp_quantizer"] = dP_quantizer_per_step
        fp8_meta_kwargs["dqkv_quantizer"] = dQKV_quantizer_per_step

    dq, dk, dv, dbias, *_ = fused_attn_bwd(
        max_seqlen_q_,
        max_seqlen_kv_,
        cu_seqlens_q_per_step[cp_size - step - 1],
        cu_seqlens_kv_per_step[cp_size - step - 1],
        q_part,
        k_part,
        v_part,
        out_part,
        dout_part,
        bwd_nominal_dtype,
        bwd_output_te_dtype,
        aux_tensors,
        fused_attn_backend,
        cu_seqlens_q_padded=cu_seqlens_q_padded_,
        cu_seqlens_kv_padded=cu_seqlens_kv_padded_,
        attn_scale=softmax_scale,
        dropout=dropout_p,
        qkv_layout=qkv_layout,
        attn_mask_type=attn_mask_type_,
        attn_bias_type=attn_bias_type,
        deterministic=deterministic,
        cuda_graph=is_graph_capturing(),
        **fp8_meta_kwargs,
    )

    return dq, dk, dv, dbias


def cp_p2p_bwd_flash_attn(
    use_flash_attn_3,
    qkv_format,
    max_seqlen_q,
    max_seqlen_kv,
    cu_seqlens_q_per_step,
    cu_seqlens_kv_per_step,
    step,
    cp_size,
    fa_backward_kwargs,
    flash_attn_bwd,
    rng_states,
    softmax_lse,
    softmax_lse_,
    q_part,
    k_part,
    v_part,
    out_part,
    dout_part,
    section,
):
    """Per-tile backward call of CP P2P with FlashAttention backend"""
    dq, dk, dv = [torch.empty_like(x) for x in [q_part, k_part, v_part]]
    if use_flash_attn_3 or (fa_utils.v2_3_plus and not fa_utils.v2_7_0_plus):
        fa_backward_kwargs["window_size"] = (-1, -1)
    elif fa_utils.v2_7_0_plus:
        fa_backward_kwargs["window_size_left"] = -1
        fa_backward_kwargs["window_size_right"] = -1
        if not use_flash_attn_3:
            fa_backward_kwargs["rng_state"] = rng_states[cp_size - step - 1]
    max_seqlen_q_ = max_seqlen_q
    max_seqlen_kv_ = max_seqlen_kv
    softmax_lse__ = softmax_lse
    causal_ = False
    if section == "diagonal":
        if use_flash_attn_3 or (fa_utils.v2_3_plus and not fa_utils.v2_7_0_plus):
            fa_backward_kwargs["window_size"] = (-1, 0)
        elif fa_utils.v2_7_0_plus:
            fa_backward_kwargs["window_size_left"] = -1
            fa_backward_kwargs["window_size_right"] = 0
        causal_ = True
    elif section == "lower-triangle":
        max_seqlen_kv_ = max_seqlen_kv // 2
    elif section == "upper-triangle":
        max_seqlen_q_ = max_seqlen_q // 2
        softmax_lse__ = softmax_lse_

    fa_backward_args_thd = get_fa_args(
        False,
        use_flash_attn_3,
        qkv_format,
        cu_seqlens_q=cu_seqlens_q_per_step[cp_size - step - 1],
        cu_seqlens_kv=cu_seqlens_kv_per_step[cp_size - step - 1],
        max_seqlen_q=max_seqlen_q_,
        max_seqlen_kv=max_seqlen_kv_,
        dq=dq,
        dk=dk,
        dv=dv,
    )
    flash_attn_bwd(
        dout_part,
        q_part,
        k_part,
        v_part,
        out_part,
        softmax_lse__,
        *fa_backward_args_thd,
        causal=causal_,
        **fa_backward_kwargs,
    )

    return dq, dk, dv


class AttnFuncWithCPAndKVP2P(torch.autograd.Function):
    """
    Attention implementation with context parallelism. Exchange KV between CP ranks
    with P2P in ring topology. Split attention compute into multiple steps, and overlap
    current-step compute with next-step communication.

    This implementation also supports hierarchical CP, which parallelizes attention
    heads in low-level CP groups and parallelizes sequence dimension in high-level CP
    groups. For more details, please refer to `LongVILA <https://arxiv.org/abs/2408.10188>`_
    and `USP <https://arxiv.org/abs/2405.07719>`_.
    """

    @staticmethod
    def forward(
        ctx,
        is_training,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
        dropout_p,
        softmax_scale,
        qkv_format,
        attn_mask_type,
        attn_bias_type,
        attn_bias,
        deterministic,
        use_fused_attention,
        return_max_logit,
        fp8,
        fp8_meta,
        cp_group,
        cp_global_ranks,
        cp_stream,
        quantizers,
        pad_between_seqs,
        use_flash_attn_3,
        fp8_output,
        layer_number,
    ):
        # pylint: disable=missing-function-docstring

        # add NVTX range
        nvtx_label = "transformer_engine.AttnFuncWithCPAndKVP2P.forward"
        nvtx_range_push(f"{nvtx_label}")

        # set up CP groups for cp_comm_type = {'p2p', 'a2a+p2p'}
        cp_group_a2a = None
        cp_size_a2a = 1
        rank_a2a = 0
        if isinstance(cp_group, list):
            cp_group_a2a = cp_group[0]
            cp_size_a2a = get_distributed_world_size(cp_group_a2a)
            rank_a2a = get_distributed_rank(cp_group_a2a)
            cp_group = cp_group[1]
        cp_size = get_distributed_world_size(cp_group)
        rank = get_distributed_rank(cp_group)
        send_dst = cp_global_ranks[(rank + 1) % cp_size * cp_size_a2a + rank_a2a]
        recv_src = cp_global_ranks[(rank - 1) % cp_size * cp_size_a2a + rank_a2a]
        device_compute_capability = get_device_compute_capability()
        batch_p2p_comm = int(os.getenv("NVTE_BATCH_MHA_P2P_COMM", "0")) or (
            device_compute_capability < (10, 0) and cp_size == 2
        )

        # set up attention args
        enable_mla = k.shape[-1] != v.shape[-1]
        causal = "causal" in attn_mask_type

        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        batch_dim = None
        seq_dim = None
        cu_seqlens_q_half, cu_seqlens_kv_half = None, None
        qkv_layout = qkv_format + "_" + qkv_format + "_" + qkv_format
        if qkv_format in ["bshd", "sbhd"]:
            seq_dim = qkv_format.index("s")
            cu_seqlens_q_padded, cu_seqlens_kv_padded = None, None
            if use_fused_attention:
                batch_dim = qkv_format.index("b")
                cu_seqlens_q, cu_seqlens_q_half = _get_cu_seqlens_info_with_cp(
                    q.shape[batch_dim], max_seqlen_q, cp_size, cu_seqlens_q
                )
                cu_seqlens_kv, cu_seqlens_kv_half = _get_cu_seqlens_info_with_cp(
                    q.shape[batch_dim], max_seqlen_kv, cp_size, cu_seqlens_kv
                )
        else:
            cu_seqlens_q_padded = cu_seqlens_q_padded // cp_size
            cu_seqlens_kv_padded = cu_seqlens_kv_padded // cp_size

        max_seqlen_q = max_seqlen_q // cp_size
        max_seqlen_kv = max_seqlen_kv // cp_size
        cu_seqlens_q_per_step = [None for _ in range(cp_size)]
        cu_seqlens_kv_per_step = [None for _ in range(cp_size)]

        fused_attn_backend = None
        amax_per_step = None
        S_quantizer_per_step = [None for _ in range(cp_size)]
        O_quantizer_per_step = [None for _ in range(cp_size)]
        max_logit_per_step = [None for _ in range(cp_size)]
        max_logit = None

        assert isinstance(k, q.__class__) and isinstance(
            v, q.__class__
        ), "q, k, v must be of the same class, e.g. torch.Tensor or Float8Tensor."
        fwd_nominal_dtype = q.dtype
        is_input_fp8 = isinstance(q, Float8Tensor)
        is_output_fp8 = fp8_output
        is_bwd_fp8 = int(os.getenv("NVTE_FP8_DPA_BWD", "1"))
        # recipe passed in through autocast or set by NVTE_DPA_FP8_RECIPE;
        # may be different from fp8_meta["recipe"]
        fp8_recipe = FP8GlobalStateManager.get_fp8_recipe()
        if fp8_meta is not None and fp8_meta.get("local_recipes", None) is not None:
            fp8_recipe = fp8_meta["local_recipes"][0]

        (
            QKV_quantizer,
            O_quantizer,
            S_quantizer,
            dQKV_quantizer,
            dO_quantizer,
            dP_quantizer,
        ) = dpa_utils.get_attention_quantizers(fp8, quantizers)

        q_f16 = None
        q_fp8, k_fp8, v_fp8 = (None, None, None)
        # communicate for the 'a2a' part of 'a2a+p2p'
        if cp_size_a2a > 1:
            if fp8 and is_input_fp8:
                QKV_quantizer = q._quantizer
                q_fp8, k_fp8, v_fp8 = q, k, v
                q, k, v = (q._data, k._data, v._data)
            chunk_ids_for_a2a = get_seq_chunk_ids_for_reordering_before_attn(cp_size_a2a, q.device)
            q, k, v = flash_attn_a2a_communicate(
                [q, k, v], chunk_ids_for_a2a, seq_dim, cp_size_a2a, cp_group_a2a, cp_stream, True
            )
            if fp8 and is_input_fp8:
                q_fp8, k_fp8, v_fp8 = [
                    Float8Tensor.make_like(x, data=y, dtype=fwd_nominal_dtype)
                    for x, y in zip([q_fp8, k_fp8, v_fp8], [q, k, v])
                ]
                q, k, v = q_fp8, k_fp8, v_fp8

        # convert qkv to the right type
        if fp8:
            assert use_fused_attention, "FP8 is only supported with Fused Attention!"
            fused_attn_backend = FusedAttnBackend["FP8"]

            if is_input_fp8:
                # q_fp8, k_fp8, v_fp8: Float8Tensor, dtype=fwd_nominal_dtype
                # q, k, v:             torch.Tensor, dtype=torch.uint8
                q_fp8, k_fp8, v_fp8 = q, k, v
                q, k, v = [q_fp8._data, k_fp8._data, v_fp8._data]
            else:
                # q_f16:               torch.Tensor, dtype=fwd_nominal_dtype
                # q_fp8, k_fp8, v_fp8: Float8Tensor, dtype=fwd_nominal_dtype
                # q, k, v:             torch.Tensor, dtype=torch.uint8
                q_f16 = q
                q_fp8, k_fp8, v_fp8 = combine_and_quantize(qkv_layout, q, k, v, QKV_quantizer)
                q, k, v = [q_fp8._data, k_fp8._data, v_fp8._data]

            # print quantizers
            print_quantizers(
                "AttnFuncWithCPAndKVP2P.forward >> before: ",
                layer_number,
                QKV_quantizer,
                O_quantizer,
                S_quantizer,
                dQKV_quantizer,
                dO_quantizer,
                dP_quantizer,
            )

            # amax_per_step[0]: amax_s x cp_size
            # amax_per_step[1]: amax_o x cp_size
            amax_per_step = torch.zeros((2, cp_size), dtype=torch.float32, device=q.device)
            # per_step tensors are not reduced even if Float8CurrentScaling.with_amax_reduction=True;
            # only used to hold temporary scale/amax values (output only, no quantization op)
            for i in range(cp_size):
                S_quantizer_per_step[i] = S_quantizer.copy()
                S_quantizer_per_step[i].amax = amax_per_step[0][i].reshape((1,))
                O_quantizer_per_step[i] = O_quantizer.copy()
                O_quantizer_per_step[i].amax = amax_per_step[1][i].reshape((1,))
        else:
            # q_f16:   torch.Tensor, dtype=fwd_nominal_dtype
            # q, k, v: torch.Tensor, dtype=fwd_nominal_dtype
            q_f16 = q
            if use_fused_attention:
                fused_attn_backend = FusedAttnBackend["F16_arbitrary_seqlen"]
            if return_max_logit:
                max_logit_per_step = [
                    torch.empty(q.shape[-2], dtype=q.dtype, device=q.device) for _ in range(cp_size)
                ]

        # split qkv to two halves and prepare for load balancing
        assert qkv_format == "thd" or (
            q.shape[seq_dim] % 2 == 0 and k.shape[seq_dim] % 2 == 0
        ), "Sequence length per GPU needs to be divisible by 2!"
        if causal:
            if qkv_format == "bshd":
                # [b, s, h, d] -> [b, 2, s//2, h, d]
                q, k, v = [x.view(x.shape[0], 2, x.shape[1] // 2, *x.shape[2:]) for x in [q, k, v]]
            elif qkv_format == "sbhd":
                # [s, b, h, d] -> [2, s//2, b, h, d]
                q, k, v = [x.view(2, x.shape[0] // 2, *x.shape[1:]) for x in [q, k, v]]
        attn_bias_ = None
        if attn_bias is not None:
            assert len(attn_bias.shape) == 4, (
                "Only support bias shape of [b, h, sq, sk] for forward, "
                "and [1, h, sq, sk] for backward!"
            )
            assert (
                attn_bias.shape[-2] % 2 == 0 and attn_bias.shape[-1] % (2 * cp_size) == 0
            ), "Sequence length does not meet divisible requirements!"
            # [b, h, sq, sk] -> [b, h, 2, sq//2, 2*cp, sk//(2*cp)]
            attn_bias_ = attn_bias.view(
                *attn_bias.shape[:-2],
                2,
                attn_bias.shape[-2] // 2,
                2 * cp_size,
                attn_bias.shape[-1] // (2 * cp_size),
            )
            # [b, h, sq, sk] -> [b, h, sq, 2*cp, sk//(2*cp)]
            attn_bias = attn_bias.view(
                *attn_bias.shape[:-1], 2 * cp_size, attn_bias.shape[-1] // (2 * cp_size)
            )

        # stats tensor shape:
        # BHS1 before cuDNN 9.6 or flash-attention v2.6/v3
        # TH1 after cuDNN 9.6 or flash-attention v2.6/v3
        softmax_lse_in_packed_format = False
        if qkv_format == "thd":
            if use_fused_attention:
                softmax_lse_in_packed_format = get_cudnn_version() >= (9, 6, 0)
            else:
                softmax_lse_in_packed_format = fa_utils.v2_6_0_plus or use_flash_attn_3

        # set up args for FlashAttention backend
        flash_attn_fwd = None
        fa_forward_kwargs = {}
        if not use_fused_attention:
            fa_forward_kwargs = {"softmax_scale": softmax_scale}
            if use_flash_attn_3:
                from transformer_engine.pytorch.attention.dot_product_attention.backends import (
                    _flash_attn_fwd_v3,
                )

                flash_attn_fwd = (
                    _flash_attn_fwd_v3  # pylint: disable=possibly-used-before-assignment
                )
                fa_forward_kwargs["window_size"] = (-1, 0) if causal else (-1, -1)
            else:
                if qkv_format == "thd":
                    from transformer_engine.pytorch.attention.dot_product_attention.backends import (
                        _flash_attn_varlen_fwd,
                    )

                    flash_attn_fwd = _flash_attn_varlen_fwd
                else:
                    from transformer_engine.pytorch.attention.dot_product_attention.backends import (
                        _flash_attn_fwd,
                    )

                    flash_attn_fwd = _flash_attn_fwd
                fa_forward_kwargs["dropout_p"] = dropout_p
                fa_forward_kwargs["return_softmax"] = False
                if fa_utils.v2_3_plus and not fa_utils.v2_7_0_plus:
                    fa_forward_kwargs["window_size"] = (-1, 0) if causal else (-1, -1)
                elif fa_utils.v2_7_0_plus:
                    fa_forward_kwargs["window_size_left"] = -1
                    fa_forward_kwargs["window_size_right"] = 0 if causal else -1
                if fa_utils.v2_4_plus:
                    fa_forward_kwargs["alibi_slopes"] = None
                if fa_utils.v2_5_7_plus and qkv_format == "thd":
                    fa_forward_kwargs["block_table"] = None
                if fa_utils.v2_6_0_plus:
                    fa_forward_kwargs["softcap"] = 0.0

        # set up inputs for forward
        q_inputs = [None, None]
        kv_inputs = [None, None]
        out_per_step = [None for _ in range(cp_size)]
        softmax_lse_per_step = [None for _ in range(cp_size)]
        rng_states = [None for _ in range(cp_size)]
        attn_biases = [None for _ in range(cp_size)]

        # create two streams to resolve wave quantization issue of Flash Attn in each step
        flash_attn_streams = [torch.cuda.current_stream(), cp_stream]
        # synchronize fwd results correction across steps
        fwd_results_correction_done = torch.cuda.Event()

        p2p_comm_buffers = [None for _ in range(cp_size)]
        k_shape = k.shape
        k_numel = k.numel()
        v_shape = v.shape
        p2p_comm_buffers[0] = torch.cat((k.view(-1), v.view(-1)), dim=-1)
        send_recv_reqs = [[], []]

        # P2P communication and compute: each rank has cp_size steps
        # f16 attention:    q, k, v: torch.Tensor, dtype=fwd_nominal_dtype
        # fp8 attention:    q, k, v: torch.Tensor, dtype=torch.uint8
        out = None
        for i in range(cp_size + 1):
            if i < cp_size:
                with torch.cuda.stream(flash_attn_streams[i % 2]):
                    # wait until KV is received
                    for req in send_recv_reqs[(i + 1) % 2]:
                        req.wait()

                    if i < (cp_size - 1):
                        p2p_comm_buffers[i + 1] = torch.empty_like(p2p_comm_buffers[i])
                        send_recv_reqs[i % 2] = flash_attn_p2p_communicate(
                            rank,
                            p2p_comm_buffers[i],
                            send_dst,
                            p2p_comm_buffers[i + 1],
                            recv_src,
                            cp_group,
                            batch_p2p_comm,
                        )

                    kv_inputs[i % 2] = p2p_comm_buffers[i]
                    k_part = kv_inputs[i % 2][:k_numel].view(*k_shape)
                    v_part = kv_inputs[i % 2][k_numel:].view(*v_shape)
                    q_part = q

                    prepare_inputs = [
                        q_part,
                        k_part,
                        v_part,
                        qkv_format,
                        pad_between_seqs,
                        cu_seqlens_q,
                        cu_seqlens_kv,
                        cu_seqlens_q_padded,
                        cu_seqlens_kv_padded,
                        cu_seqlens_q_half,
                        cu_seqlens_kv_half,
                        rank,
                        i,
                        cp_size,
                    ]
                    if use_fused_attention:
                        fused_attn_inputs = [
                            attn_bias,
                            attn_bias_,
                            is_training,
                            max_seqlen_q,
                            max_seqlen_kv,
                            cu_seqlens_q_padded,
                            cu_seqlens_kv_padded,
                            fused_attn_backend,
                            softmax_scale,
                            dropout_p,
                            qkv_layout,
                            attn_mask_type,
                            attn_bias_type,
                            fp8,
                            q_fp8,
                            k_fp8,
                            v_fp8,
                            fwd_nominal_dtype,
                            S_quantizer_per_step[i],
                            O_quantizer_per_step[i],
                            rank,
                            i,
                            cp_size,
                            return_max_logit,
                        ]
                    else:
                        flash_attn_inputs = [
                            use_flash_attn_3,
                            qkv_format,
                            fa_forward_kwargs,
                            flash_attn_fwd,
                            max_seqlen_q,
                            max_seqlen_kv,
                        ]

                    # cp_size = 4:
                    #
                    #           step
                    # section | 0  1  2  3
                    # --------------------
                    #    G  0 | d, u, u, u,
                    #    P  1 | l, d, u, u,
                    #    U  2 | l, l, d, u,
                    #       3 | l, l, l, d,
                    #
                    # Each GPU holds a slice of Q and KV. To compute the attention of each Q slice, each GPU
                    # runs cp_size steps to get the partial results of its own Q and all KV slices. KV is communicated
                    # in a point-to-point, ring fashion. For attn_mask_type = causal, there are three attention
                    # patterns in the cp_size x cp_size (i.e. GPU x step) matrix, the diagonal tiles, the lower-triangle
                    # tiles, and the upper-triangle tiles. For attn_mask_type != causal, the pattern is all the same.
                    if causal:
                        if i == 0:
                            section = "diagonal"
                            prepare_outputs = cp_p2p_fwd_prepare_qkv(*prepare_inputs, section)
                            (
                                q_part,
                                k_part,
                                v_part,
                                cu_seqlens_q_per_step[i],
                                cu_seqlens_kv_per_step[i],
                            ) = prepare_outputs
                            q_inputs[i % 2] = q_part
                            if use_fused_attention:
                                (
                                    out_per_step[i],
                                    softmax_lse_per_step[i],
                                    rng_states[i],
                                    attn_biases[i],
                                    max_logit_per_step[i],
                                ) = cp_p2p_fwd_fused_attn(
                                    *fused_attn_inputs, *prepare_outputs, section
                                )
                            else:
                                out_per_step[i], softmax_lse_per_step[i], rng_states[i] = (
                                    cp_p2p_fwd_flash_attn(
                                        *flash_attn_inputs, *prepare_outputs, section
                                    )
                                )
                        elif i <= rank:
                            section = "lower-triangle"
                            prepare_outputs = cp_p2p_fwd_prepare_qkv(*prepare_inputs, section)
                            (
                                q_part,
                                k_part,
                                v_part,
                                cu_seqlens_q_per_step[i],
                                cu_seqlens_kv_per_step[i],
                            ) = prepare_outputs
                            q_inputs[i % 2] = q_part
                            if use_fused_attention:
                                (
                                    out_per_step[i],
                                    softmax_lse_per_step[i],
                                    rng_states[i],
                                    attn_biases[i],
                                    max_logit_per_step[i],
                                ) = cp_p2p_fwd_fused_attn(
                                    *fused_attn_inputs, *prepare_outputs, section
                                )
                            else:
                                out_per_step[i], softmax_lse_per_step[i], rng_states[i] = (
                                    cp_p2p_fwd_flash_attn(
                                        *flash_attn_inputs, *prepare_outputs, section
                                    )
                                )
                        else:
                            section = "upper-triangle"
                            prepare_outputs = cp_p2p_fwd_prepare_qkv(*prepare_inputs, section)
                            (
                                q_part,
                                k_part,
                                v_part,
                                cu_seqlens_q_per_step[i],
                                cu_seqlens_kv_per_step[i],
                            ) = prepare_outputs
                            q_inputs[i % 2] = q_part
                            if use_fused_attention:
                                (
                                    out_per_step[i],
                                    softmax_lse_per_step[i],
                                    rng_states[i],
                                    attn_biases[i],
                                    max_logit_per_step[i],
                                ) = cp_p2p_fwd_fused_attn(
                                    *fused_attn_inputs, *prepare_outputs, section
                                )
                            else:
                                out_per_step[i], softmax_lse_per_step[i], rng_states[i] = (
                                    cp_p2p_fwd_flash_attn(
                                        *flash_attn_inputs, *prepare_outputs, section
                                    )
                                )
                    else:
                        # all tiles
                        section = "all"
                        prepare_outputs = cp_p2p_fwd_prepare_qkv(*prepare_inputs, section)
                        (
                            q_part,
                            k_part,
                            v_part,
                            cu_seqlens_q_per_step[i],
                            cu_seqlens_kv_per_step[i],
                        ) = prepare_outputs
                        q_inputs[i % 2] = q_part
                        if use_fused_attention:
                            (
                                out_per_step[i],
                                softmax_lse_per_step[i],
                                rng_states[i],
                                attn_biases[i],
                                max_logit_per_step[i],
                            ) = cp_p2p_fwd_fused_attn(*fused_attn_inputs, *prepare_outputs, section)
                        else:
                            out_per_step[i], softmax_lse_per_step[i], rng_states[i] = (
                                cp_p2p_fwd_flash_attn(*flash_attn_inputs, *prepare_outputs, section)
                            )

            # softmax_lse correction
            if i > 0:
                # wait until fwd results correction of last step is done
                if i > 1:
                    flash_attn_streams[(i - 1) % 2].wait_event(fwd_results_correction_done)

                with torch.cuda.stream(flash_attn_streams[(i - 1) % 2]):
                    if use_fused_attention:
                        # [b, h, sq, 1] -> [b, h, sq] or
                        # [t, h, 1] -> [t, np]
                        softmax_lse_per_step[i - 1].squeeze_(-1)
                        if softmax_lse_in_packed_format:
                            softmax_lse_per_step[i - 1] = (
                                softmax_lse_per_step[i - 1].transpose(0, 1).contiguous()
                            )
                    if fp8:
                        # dequantize out_per_step to torch.float32
                        if fp8_recipe.delayed():
                            out_per_step[i - 1] = out_per_step[i - 1].dequantize(
                                dtype=torch.float32
                            )
                        if fp8_recipe.float8_current_scaling():
                            out_per_step[i - 1] = out_per_step[i - 1].to(dtype=torch.float32)

                    if i == 1:
                        softmax_lse = torch.clone(softmax_lse_per_step[0])
                        if qkv_format == "thd":
                            if enable_mla:
                                out = torch.zeros_like(v if not fp8 else out_per_step[0]).view(
                                    v_shape
                                )
                            else:
                                # MHA or GQA
                                out = torch.zeros_like(q if not fp8 else out_per_step[0]).view(
                                    q.shape
                                )
                    elif (i - 1) <= rank or not causal:
                        flash_attn_fwd_softmax_lse_correction(
                            softmax_lse, softmax_lse_per_step[i - 1]
                        )
                    else:
                        if qkv_format == "thd":
                            tex.thd_second_half_lse_correction(
                                softmax_lse,
                                softmax_lse_per_step[i - 1],
                                cu_seqlens_q_padded,
                                softmax_lse_in_packed_format,
                            )
                        else:
                            flash_attn_fwd_second_half_softmax_lse_correction(
                                softmax_lse.view(*softmax_lse.shape[:-1], 2, -1),
                                softmax_lse_per_step[i - 1],
                            )
                    if return_max_logit:
                        if i == 1:
                            max_logit = torch.clone(max_logit_per_step[0])
                        else:
                            max_logit = torch.maximum(max_logit, max_logit_per_step[i - 1])

                if i < cp_size:
                    flash_attn_streams[(i - 1) % 2].record_event(fwd_results_correction_done)

        torch.cuda.current_stream().wait_stream(flash_attn_streams[1])
        if return_max_logit:
            torch.distributed.all_reduce(
                max_logit, op=torch.distributed.ReduceOp.MAX, group=cp_group
            )

        second_half_lse_seqlen = None
        if causal and rank < (cp_size - 1):
            second_half_lse_seqlen = softmax_lse_per_step[-1].shape[-1]

        # fwd output correction: out in torch.float32
        for i in range(cp_size):
            if i <= rank or not causal:
                if qkv_format in ["bshd", "sbhd"]:
                    if i == 0:
                        out = flash_attn_fwd_out_correction_init(
                            out_per_step[0],
                            softmax_lse,
                            softmax_lse_per_step[0],
                            seq_dim,
                        )
                        if enable_mla:
                            out = out.view(v_shape)
                        else:
                            out = out.view(q.shape)
                    else:
                        flash_attn_fwd_out_correction(
                            out.view(*out_per_step[i].shape),
                            out_per_step[i],
                            softmax_lse,
                            softmax_lse_per_step[i],
                            seq_dim,
                        )
                elif qkv_format == "thd":
                    tex.thd_out_correction(
                        out,
                        out_per_step[i],
                        softmax_lse,
                        softmax_lse_per_step[i],
                        cu_seqlens_q_padded,
                        False,
                        softmax_lse_in_packed_format,
                    )
            else:
                if qkv_format in ["bshd", "sbhd"]:
                    flash_attn_fwd_second_half_out_correction(
                        out,
                        out_per_step[i],
                        softmax_lse,
                        softmax_lse_per_step[i],
                        seq_dim,
                    )
                elif qkv_format == "thd":
                    tex.thd_out_correction(
                        out,
                        out_per_step[i],
                        softmax_lse,
                        softmax_lse_per_step[i],
                        cu_seqlens_q_padded,
                        True,
                        softmax_lse_in_packed_format,
                    )

        if qkv_format == "bshd":
            out = out.view(out.shape[0], -1, *out.shape[-2:])
            ctx.batch_size = out.shape[0]
        elif qkv_format == "sbhd":
            out = out.view(-1, *out.shape[-3:])
            ctx.batch_size = out.shape[1]

        if cp_size_a2a > 1:
            chunk_ids_for_a2a = get_seq_chunk_ids_for_reordering_after_attn(cp_size_a2a, out.device)
            out = flash_attn_a2a_communicate(
                out, chunk_ids_for_a2a, seq_dim, cp_size_a2a, cp_group_a2a, cp_stream, False
            )
            if use_fused_attention:
                if qkv_format == "bshd":
                    # [b*s, h, d] -> [b, s, h, d]
                    out = out.view(ctx.batch_size, -1, *out.shape[-2:])
                elif qkv_format == "sbhd":
                    # [s*b, h, d] -> [s, b, h, d]
                    out = out.view(-1, ctx.batch_size, *out.shape[-2:])
            if return_max_logit:
                max_logit = flash_attn_a2a_communicate_softmax_offset(
                    max_logit, 0, cp_size_a2a, cp_group_a2a, cp_stream, False
                )
        elif not use_fused_attention:
            out = out.view(-1, *out.shape[-2:])

        # update FP8 quantizers: amax across cp_size steps
        if fp8 and use_fused_attention:
            amax_cp_fwd = amax_per_step.amax(dim=1)
            S_quantizer.amax.copy_(amax_cp_fwd[0])
            O_quantizer.amax.copy_(amax_cp_fwd[1])

        if fp8:
            # print quantizers
            print_quantizers(
                "AttnFuncWithCPAndKVP2P.forward >> after:  ",
                layer_number,
                QKV_quantizer,
                O_quantizer,
                S_quantizer,
                dQKV_quantizer,
                dO_quantizer,
                dP_quantizer,
            )

        # prepare for return and ctx saves
        out_fp8 = None
        out_f16 = out.to(fwd_nominal_dtype)
        if fp8 and (
            is_output_fp8
            or (is_bwd_fp8 and not (fp8_recipe.float8_current_scaling() and _dpa_fp8_cs_o_in_f16))
        ):
            out_fp8 = O_quantizer(out_f16)
        out_ret = out_fp8 if (fp8 and is_output_fp8) else out_f16

        ctx.layer_number = layer_number
        ctx.fp8_recipe = fp8_recipe
        ctx.fp8 = fp8 and is_bwd_fp8

        kv_fp8 = None
        kv = p2p_comm_buffers[-1]
        if fp8:
            q_fp8, kv_fp8 = [
                Float8Tensor.make_like(x, data=y, dtype=fwd_nominal_dtype)
                for x, y in zip([q_fp8, k_fp8], [q, kv])
            ]
        # q, kv, out
        fp8_tensors = (None, None, None)
        f16_tensors = (None, None, None)
        if ctx.fp8:
            # fwd: fp8, bwd: fp8, save all fp8
            fp8_tensors = (q_fp8, kv_fp8, out_fp8)
            if fp8_recipe.float8_current_scaling() and _dpa_fp8_cs_o_in_f16:
                f16_tensors = (None, None, out_f16)
        elif fp8 and is_input_fp8:
            # fwd: fp8, bwd: f16, save all f16
            # dequantize fp8 inputs
            q_f16 = q_fp8.dequantize()
            kv_f16 = kv_fp8.dequantize()
            f16_tensors = (q_f16, kv_f16, out_f16)
        elif fp8:
            # fwd: fp8, bwd: f16, save all f16
            # inputs are already in f16
            q_f16 = q_f16.view(q.shape)
            kv_f16 = kv_fp8.dequantize()
            f16_tensors = (q_f16, kv_f16, out_f16)
        else:
            # fwd: f16, bwd: f16, save all f16
            # inputs and kernels are both f16
            q_f16 = q_f16.view(q.shape)
            kv_f16 = kv
            f16_tensors = (q_f16, kv_f16, out_f16)

        tensors_to_save, tensor_objects = prepare_for_saving(
            *fp8_tensors,
            *f16_tensors,
            softmax_lse,
            cu_seqlens_q_padded,
            cu_seqlens_kv_padded,
            *cu_seqlens_q_per_step,
            *cu_seqlens_kv_per_step,
            *rng_states,
            *attn_biases,
        )
        ctx.save_for_backward(*tensors_to_save)
        ctx.tensor_objects = tensor_objects

        ctx.cp_group_a2a = cp_group_a2a
        ctx.cp_size_a2a = cp_size_a2a
        ctx.rank_a2a = rank_a2a
        ctx.cp_group = cp_group
        ctx.cp_global_ranks = cp_global_ranks
        ctx.cp_stream = cp_stream
        ctx.dropout_p = dropout_p
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_kv = max_seqlen_kv
        ctx.softmax_scale = softmax_scale
        ctx.qkv_format = qkv_format
        ctx.attn_mask_type = attn_mask_type
        ctx.attn_bias_type = attn_bias_type
        ctx.attn_bias_shape = None if attn_bias is None else attn_bias.shape
        ctx.deterministic = deterministic
        ctx.use_fused_attention = use_fused_attention
        ctx.softmax_lse_in_packed_format = softmax_lse_in_packed_format
        ctx.second_half_lse_seqlen = second_half_lse_seqlen
        ctx.fp8_meta = fp8_meta
        ctx.is_input_fp8 = is_input_fp8
        ctx.is_output_fp8 = is_output_fp8
        ctx.use_flash_attn_3 = use_flash_attn_3

        ctx.enable_mla = enable_mla
        ctx.k_numel = k_numel
        ctx.k_shape = k_shape
        ctx.v_shape = v_shape

        ctx.fwd_nominal_dtype = fwd_nominal_dtype
        ctx.dQKV_quantizer = dQKV_quantizer
        ctx.dO_quantizer = dO_quantizer
        ctx.dP_quantizer = dP_quantizer
        ctx.QKV_quantizer = QKV_quantizer
        ctx.O_quantizer = O_quantizer
        ctx.S_quantizer = S_quantizer
        if ctx.fp8:
            ctx.QKV_quantizer = QKV_quantizer.copy()
            ctx.QKV_quantizer.scale = QKV_quantizer.scale.clone()
            ctx.O_quantizer = O_quantizer.copy()
            ctx.O_quantizer.scale = O_quantizer.scale.clone()
            ctx.S_quantizer = S_quantizer.copy()
            ctx.S_quantizer.scale = S_quantizer.scale.clone()

        nvtx_range_pop(f"{nvtx_label}")

        if return_max_logit:
            return out_ret, max_logit
        return out_ret

    @staticmethod
    def backward(ctx, dout, *_args):
        # pylint: disable=missing-function-docstring

        # add NVTX range
        nvtx_label = "transformer_engine.AttnFuncWithCPAndKVP2P.backward"
        nvtx_range_push(f"{nvtx_label}")

        # dout is expected to be in FP8 if is_output_fp8=True,
        # but in the case it's not, convert it to FP8 before any operation
        if ctx.fp8 and ctx.is_output_fp8 and not isinstance(dout, QuantizedTensorStorage):
            dout = ctx.dO_quantizer(dout)
            if ctx.use_fused_attention:
                dout._data = dout._data.contiguous()
        elif ctx.use_fused_attention:
            dout = dout.contiguous()

        # set up CP groups for cp_comm_type = {'p2p', 'a2a+p2p'}
        cp_size_a2a = ctx.cp_size_a2a
        rank_a2a = ctx.rank_a2a
        cp_size = get_distributed_world_size(ctx.cp_group)
        rank = get_distributed_rank(ctx.cp_group)
        send_dst = ctx.cp_global_ranks[(rank - 1) % cp_size * cp_size_a2a + rank_a2a]
        recv_src = ctx.cp_global_ranks[(rank + 1) % cp_size * cp_size_a2a + rank_a2a]
        device_compute_capability = get_device_compute_capability()
        batch_p2p_comm = int(os.getenv("NVTE_BATCH_MHA_P2P_COMM", "0")) or (
            device_compute_capability < (10, 0) and cp_size == 2
        )

        # get saved tensors
        (
            q_fp8,
            kv_fp8,
            out_fp8,
            q,
            kv,
            out,
            softmax_lse,
            cu_seqlens_q_padded,
            cu_seqlens_kv_padded,
            *other_tensors,
        ) = restore_from_saved(ctx.tensor_objects, ctx.saved_tensors)
        cu_seqlens_q_per_step = other_tensors[:cp_size]
        cu_seqlens_kv_per_step = other_tensors[cp_size : cp_size * 2]
        rng_states = other_tensors[cp_size * 2 : cp_size * 3]
        attn_biases = other_tensors[cp_size * 3 : cp_size * 4]

        # set up attention args
        causal = "causal" in ctx.attn_mask_type
        seq_dim = None
        qkv_layout = ctx.qkv_format + "_" + ctx.qkv_format + "_" + ctx.qkv_format
        if ctx.qkv_format in ["bshd", "sbhd"]:
            seq_dim = ctx.qkv_format.index("s")

        # set up attention bias
        if attn_biases[0] is not None:
            # [b, h, sq, 2*cp, sk//(2*cp)]
            attn_dbias = torch.zeros(
                *ctx.attn_bias_shape, dtype=attn_biases[0].dtype, device=attn_biases[0].device
            )
            # [b, h, sq, 2*cp, sk//(2*cp)] -> [b, h, 2, sq//2, 2*cp, sk//(2*cp)]
            attn_dbias_ = attn_dbias.view(
                *attn_dbias.shape[:-3], 2, attn_dbias.shape[-3] // 2, *attn_dbias.shape[-2:]
            )
        else:
            attn_dbias = None
            attn_dbias_ = None

        # set up softmax_lse
        softmax_lse_ = None
        if causal and ctx.second_half_lse_seqlen is not None:
            if ctx.qkv_format == "thd":
                softmax_lse_ = tex.thd_read_second_half_lse(
                    softmax_lse,
                    cu_seqlens_q_padded,
                    ctx.softmax_lse_in_packed_format,
                    ctx.second_half_lse_seqlen,
                )
            else:
                # [b, h, sq] -> [b, h, 2, sq//2]
                softmax_lse_ = softmax_lse.view(*softmax_lse.shape[:-1], 2, -1)
                softmax_lse_ = softmax_lse_[..., 1, :].contiguous()
            if ctx.use_fused_attention:
                if ctx.softmax_lse_in_packed_format:
                    softmax_lse_ = softmax_lse_.transpose(0, 1).contiguous()
                # [b, h, sq//2] -> [b, h, sq//2, 1] or
                # [t//2, np] -> [t//2, h, 1]
                softmax_lse_.unsqueeze_(-1)
        if ctx.use_fused_attention:
            if ctx.softmax_lse_in_packed_format:
                softmax_lse = softmax_lse.transpose(0, 1).contiguous()
            # [b, h, sq] -> [b, h, sq, 1] or
            # [t, np] -> [t, h, 1]
            softmax_lse.unsqueeze_(-1)

        # assume fwd and bwd always use the same high precision, i.e. torch.float16 or torch.bfloat16
        # used when some tensors are base tensors and loose the "dtype" attribute
        bwd_nominal_dtype = ctx.fwd_nominal_dtype

        # convert out, dout to the right type
        fused_attn_backend = None
        amax_per_step = None
        dP_quantizer_per_step = [None for _ in range(cp_size)]
        dQKV_quantizer_per_step = [None for _ in range(cp_size)]
        buffer_dtype = torch.uint8
        dq_buffer = None
        dout_fp8 = None
        bwd_output_te_dtype = None
        dkv_buffer = None
        if ctx.fp8:
            assert ctx.use_fused_attention, "FP8 is only supported with Fused Attention!"
            fused_attn_backend = FusedAttnBackend["FP8"]
            q, kv, out = (
                q_fp8._data,
                kv_fp8._data,
                (
                    out
                    if ctx.fp8_recipe.float8_current_scaling() and _dpa_fp8_cs_o_in_f16
                    else out_fp8._data
                ),
            )

            # dout_fp8: Float8Tensor, dtype=bwd_nominal_dtype
            # dout:     torch.Tensor, dtype=torch.uint8
            if ctx.is_output_fp8:
                dout_fp8 = dout
            else:
                dout_fp8 = ctx.dO_quantizer(dout)
            dout = dout_fp8._data

            # print quantizers
            print_quantizers(
                "AttnFuncWithCPAndKVP2P.backward >> before: ",
                ctx.layer_number,
                ctx.QKV_quantizer,
                ctx.O_quantizer,
                ctx.S_quantizer,
                ctx.dQKV_quantizer,
                ctx.dO_quantizer,
                ctx.dP_quantizer,
            )

            # dout_fp8._fp8_dtype
            bwd_output_te_dtype = ctx.dO_quantizer.dtype

            # create buffers for reduction in float32
            if ctx.fp8_recipe.delayed():
                dq_buffer = torch.empty(
                    (cp_size, *q.shape),
                    dtype=buffer_dtype,
                    device=q.device,
                )
            if ctx.fp8_recipe.float8_current_scaling():
                dq_buffer = torch.empty(
                    q.shape,
                    dtype=torch.float32,
                    device=q.device,
                )
            kv_recv_buffer = torch.empty_like(kv)
            dkv_send_buffer = torch.empty(
                (cp_size, *kv.shape),
                dtype=buffer_dtype,
                device=kv.device,
            )
            dkv_recv_buffer = torch.empty_like(dkv_send_buffer)
            p2p_comm_buffers = [[kv, dkv_send_buffer], [kv_recv_buffer, dkv_recv_buffer]]
            if ctx.fp8_recipe.float8_current_scaling():
                dkv_buffer = torch.zeros(
                    kv.shape,
                    dtype=torch.float32,
                    device=kv.device,
                )

            # amax_per_step[0]: amax_dp x cp_size
            # amax_per_step[1]: amax_dqkv x cp_size
            amax_per_step = torch.zeros((2, cp_size), dtype=torch.float32, device=q.device)
            # per_step tensors are not reduced even if Float8CurrentScaling.with_amax_reduction=True;
            # only used to hold temporary scale/amax values (output only, no quantization op)
            for i in range(cp_size):
                dP_quantizer_per_step[i] = ctx.dP_quantizer.copy()
                dP_quantizer_per_step[i].amax = amax_per_step[0][i].reshape((1,))
                dQKV_quantizer_per_step[i] = ctx.dQKV_quantizer.copy()
                dQKV_quantizer_per_step[i].amax = amax_per_step[1][i].reshape((1,))
        else:
            if isinstance(dout, QuantizedTensorStorage):
                dout = dout.dequantize(dtype=bwd_nominal_dtype)
            dq_buffer = torch.empty_like(q)
            p2p_comm_buffers = [
                torch.empty((2, *kv.shape), dtype=kv.dtype, device=kv.device),
                torch.empty((2, *kv.shape), dtype=kv.dtype, device=kv.device),
            ]
            p2p_comm_buffers[0][0].copy_(kv)
            if ctx.use_fused_attention:
                bwd_output_te_dtype = TE_DType[bwd_nominal_dtype]
                fused_attn_backend = FusedAttnBackend["F16_arbitrary_seqlen"]

        # communicate for the 'a2a' part of 'a2a+p2p'
        if cp_size_a2a > 1:
            if not ctx.use_fused_attention:
                out = out.view(ctx.batch_size, -1, *out.shape[-2:])
                dout = dout.view(*out.shape)
            chunk_ids_for_a2a = get_seq_chunk_ids_for_reordering_before_attn(
                cp_size_a2a, out.device
            )
            out, dout = flash_attn_a2a_communicate(
                [out, dout],
                chunk_ids_for_a2a,
                seq_dim,
                cp_size_a2a,
                ctx.cp_group_a2a,
                ctx.cp_stream,
                True,
            )

        if ctx.enable_mla:
            out = out.view(*ctx.v_shape)
            dout = dout.view(*ctx.v_shape)
        else:
            # MHA or GQA
            out = out.view(*q.shape)
            dout = dout.view(*q.shape)

        flash_attn_bwd = None
        if not ctx.use_fused_attention:
            fa_backward_kwargs = {"softmax_scale": ctx.softmax_scale}
            if ctx.use_flash_attn_3:
                from transformer_engine.pytorch.attention.dot_product_attention.backends import (
                    _flash_attn_bwd_v3,
                )

                flash_attn_bwd = (
                    _flash_attn_bwd_v3  # pylint: disable=possibly-used-before-assignment
                )
                fa_backward_kwargs["deterministic"] = ctx.deterministic
            else:
                if ctx.qkv_format == "thd":
                    from transformer_engine.pytorch.attention.dot_product_attention.backends import (
                        _flash_attn_varlen_bwd,
                    )

                    flash_attn_bwd = _flash_attn_varlen_bwd
                else:
                    from transformer_engine.pytorch.attention.dot_product_attention.backends import (
                        _flash_attn_bwd,
                    )

                    flash_attn_bwd = _flash_attn_bwd
                fa_backward_kwargs["dropout_p"] = ctx.dropout_p
                if fa_utils.v2_4_plus:
                    fa_backward_kwargs["alibi_slopes"] = None
                if fa_utils.v2_4_1_plus:
                    fa_backward_kwargs["deterministic"] = ctx.deterministic
                if fa_utils.v2_6_0_plus:
                    fa_backward_kwargs["softcap"] = 0.0

        send_recv_reqs = []
        for i in range(cp_size):
            # wait until KV is received
            for req in send_recv_reqs:
                req.wait()

            send_tensor = p2p_comm_buffers[i % 2]
            recv_tensor = p2p_comm_buffers[(i + 1) % 2]
            if ctx.fp8:
                if i < cp_size - 1:
                    send_recv_reqs = flash_attn_p2p_communicate(
                        rank,
                        send_tensor[0],
                        send_dst,
                        recv_tensor[0],
                        recv_src,
                        ctx.cp_group,
                        batch_p2p_comm,
                    )
                else:
                    dkv_a2a_req = torch.distributed.all_to_all_single(
                        dkv_send_buffer,
                        dkv_recv_buffer,
                        group=ctx.cp_group,
                        async_op=True,
                    )
                    send_recv_reqs = [dkv_a2a_req]
            else:
                if i == 0:
                    send_tensor = send_tensor[0]
                    recv_tensor = recv_tensor[0]
                if i == (cp_size - 1):
                    send_tensor = send_tensor[1]
                    recv_tensor = recv_tensor[1]
                send_recv_reqs = flash_attn_p2p_communicate(
                    rank, send_tensor, send_dst, recv_tensor, recv_src, ctx.cp_group, batch_p2p_comm
                )

            kv = p2p_comm_buffers[i % 2][0]
            dq_, dk_, dv_ = None, None, None
            k_part = kv[: ctx.k_numel].view(*ctx.k_shape)
            v_part = kv[ctx.k_numel :].view(*ctx.v_shape)
            q_part, out_part, dout_part = q, out, dout

            prepare_inputs = [
                q_part,
                k_part,
                v_part,
                out_part,
                dout_part,
                ctx.qkv_format,
                cu_seqlens_q_padded,
                cu_seqlens_kv_padded,
            ]
            if ctx.use_fused_attention:
                fused_attn_inputs = [
                    ctx.fp8,
                    ctx.fp8_recipe,
                    q_fp8,
                    kv_fp8,
                    (
                        out
                        if ctx.fp8_recipe.float8_current_scaling() and _dpa_fp8_cs_o_in_f16
                        else out_fp8
                    ),
                    dout_fp8,
                    softmax_lse,
                    softmax_lse_,
                    rng_states,
                    attn_dbias,
                    attn_biases,
                    ctx.max_seqlen_q,
                    ctx.max_seqlen_kv,
                    i,
                    cp_size,
                    cu_seqlens_q_per_step,
                    cu_seqlens_kv_per_step,
                    cu_seqlens_q_padded,
                    cu_seqlens_kv_padded,
                    fused_attn_backend,
                    ctx.softmax_scale,
                    ctx.dropout_p,
                    qkv_layout,
                    ctx.attn_mask_type,
                    ctx.attn_bias_type,
                    ctx.deterministic,
                    ctx.fwd_nominal_dtype,
                    bwd_nominal_dtype,
                    bwd_output_te_dtype,
                    ctx.S_quantizer,
                    dP_quantizer_per_step[i],
                    dQKV_quantizer_per_step[i],
                ]
            else:
                flash_attn_inputs = [
                    ctx.use_flash_attn_3,
                    ctx.qkv_format,
                    ctx.max_seqlen_q,
                    ctx.max_seqlen_kv,
                    cu_seqlens_q_per_step,
                    cu_seqlens_kv_per_step,
                    i,
                    cp_size,
                    fa_backward_kwargs,
                    flash_attn_bwd,
                    rng_states,
                    softmax_lse,
                    softmax_lse_,
                ]

            # Reverse the steps in forward. In the cp_size x cp_size (i.e. GPU x step) matrix,
            # there are still three sections in these tiles based on their attention pattern
            # for attn_mask_type = causal, and one for attn_mask_type != causal.
            if causal:
                if i == (cp_size - 1):
                    section = "diagonal"
                    prepare_outputs = cp_p2p_bwd_prepare_qkv(*prepare_inputs, section)
                    if ctx.use_fused_attention:
                        dq_, dk_, dv_, dbias_ = cp_p2p_bwd_fused_attn(
                            *fused_attn_inputs, *prepare_outputs, section
                        )
                    else:
                        dq_, dk_, dv_ = cp_p2p_bwd_flash_attn(
                            *flash_attn_inputs, *prepare_outputs, section
                        )
                elif i >= (cp_size - rank - 1):
                    section = "lower-triangle"
                    prepare_outputs = cp_p2p_bwd_prepare_qkv(*prepare_inputs, section)
                    if ctx.use_fused_attention:
                        dq_, dk_, dv_, dbias_ = cp_p2p_bwd_fused_attn(
                            *fused_attn_inputs, *prepare_outputs, section
                        )
                    else:
                        dq_, dk_, dv_ = cp_p2p_bwd_flash_attn(
                            *flash_attn_inputs, *prepare_outputs, section
                        )
                else:
                    section = "upper-triangle"
                    prepare_outputs = cp_p2p_bwd_prepare_qkv(*prepare_inputs, section)
                    if ctx.use_fused_attention:
                        dq_, dk_, dv_, dbias_ = cp_p2p_bwd_fused_attn(
                            *fused_attn_inputs, *prepare_outputs, section
                        )
                    else:
                        dq_, dk_, dv_ = cp_p2p_bwd_flash_attn(
                            *flash_attn_inputs, *prepare_outputs, section
                        )
            else:
                section = "all"
                prepare_outputs = cp_p2p_bwd_prepare_qkv(*prepare_inputs, section)
                if ctx.use_fused_attention:
                    dq_, dk_, dv_, dbias_ = cp_p2p_bwd_fused_attn(
                        *fused_attn_inputs, *prepare_outputs, section
                    )
                else:
                    dq_, dk_, dv_ = cp_p2p_bwd_flash_attn(
                        *flash_attn_inputs, *prepare_outputs, section
                    )

            # dq, dk, dv are reduced across steps in higher precision
            # DelayedScaling: collect all results in uint8 to one tensor, dequantize to float32, then reduce
            # CurrentScaling: dequantize partial results from each step to float32, then reduce
            if ctx.fp8 and ctx.use_fused_attention:
                if ctx.fp8_recipe.delayed():
                    dq_, dk_, dv_ = [x._data for x in [dq_, dk_, dv_]]
                if ctx.fp8_recipe.float8_current_scaling():
                    dq_, dk_, dv_ = [x.to(torch.float32) for x in [dq_, dk_, dv_]]

            # copy dq_ into the right buffer position
            # buffer is cp_size x dq_size for DelayedScaling and the same size as dq for CurrentScaling
            if ctx.fp8 and ctx.fp8_recipe.delayed():
                dq = dq_buffer[(rank + i + 1) % cp_size]
            else:
                dq = dq_buffer
            if causal and ctx.qkv_format in ["bshd", "sbhd"] and i >= (cp_size - rank - 1):
                # [b, sq, h, d] -> [b, 2, sq//2, h, d] or
                # [sq, b, h, d] -> [2, sq//2, b, h, d]
                dq_ = dq_.view(*dq.shape)
            if ctx.fp8 and ctx.fp8_recipe.delayed():
                if i >= (cp_size - rank - 1) or not causal:
                    dq.copy_(dq_)
                else:
                    if ctx.qkv_format == "bshd":
                        dq[:, 0, ...].fill_(0)
                        dq[:, 1, ...].copy_(dq_)
                    elif ctx.qkv_format == "sbhd":
                        dq[0].fill_(0)
                        dq[1].copy_(dq_)
                    else:
                        dq.copy_(dq_)
            elif causal:
                if i > (cp_size - rank - 1):
                    dq.add_(dq_)
                elif i == (cp_size - rank - 1):
                    if rank == (cp_size - 1):
                        dq.copy_(dq_)
                    else:
                        if ctx.qkv_format == "bshd":
                            dq[:, 0, ...].copy_(dq_[:, 0, ...])
                            dq[:, 1, ...].add_(dq_[:, 1, ...])
                        elif ctx.qkv_format == "sbhd":
                            dq[0].copy_(dq_[0])
                            dq[1].add_(dq_[1])
                        elif ctx.qkv_format == "thd":
                            tex.thd_grad_correction(dq, dq_, cu_seqlens_q_padded, "copy", "add")
                elif i > 0:
                    if ctx.qkv_format == "bshd":
                        dq[:, 1, ...].add_(dq_)
                    elif ctx.qkv_format == "sbhd":
                        dq[1].add_(dq_)
                    elif ctx.qkv_format == "thd":
                        tex.thd_grad_correction(dq, dq_, cu_seqlens_q_padded, "none", "add")
                else:
                    if ctx.qkv_format == "bshd":
                        dq[:, 1, ...].copy_(dq_)
                    elif ctx.qkv_format == "sbhd":
                        dq[1].copy_(dq_)
                    elif ctx.qkv_format == "thd":
                        tex.thd_grad_correction(dq, dq_, cu_seqlens_q_padded, "none", "copy")
            else:
                if i == 0:
                    dq.copy_(dq_)
                else:
                    dq.add_(dq_)

            # dbias correction
            if attn_dbias is not None:
                idx = (rank + i + 1) % cp_size
                if i == (cp_size - 1) or not causal:
                    # [b, h, sq, sk//cp] -> [b, h, sq, 2, sk//(2*cp)]
                    dbias_ = dbias_.view(*dbias_.shape[:-1], 2, dbias_.shape[-1] // 2)
                    attn_dbias[..., idx, :].copy_(dbias_[..., 0, :])
                    attn_dbias[..., (2 * cp_size - idx - 1), :].copy_(dbias_[..., 1, :])
                elif i >= (cp_size - rank - 1):
                    # [b, h, sq, sk//(2*cp)]
                    attn_dbias[..., idx, :].copy_(dbias_)
                else:
                    # [b, h, sq//2, sk//cp] -> [b, h, sq//2, 2, sk//(2*cp)]
                    dbias_ = dbias_.view(*dbias_.shape[:-1], 2, dbias_.shape[-1] // 2)
                    attn_dbias_[..., 1, :, idx, :].copy_(dbias_[..., 0, :])
                    attn_dbias_[..., 1, :, (2 * cp_size - idx - 1), :].copy_(dbias_[..., 1, :])

            # wait until dKV is received
            for req in send_recv_reqs:
                req.wait()

            # dkv correction
            if ctx.fp8 and ctx.fp8_recipe.delayed():
                dkv = dkv_recv_buffer[(rank + i + 1) % cp_size]
            elif ctx.fp8 and ctx.fp8_recipe.float8_current_scaling():
                dkv = dkv_buffer
            else:
                dkv = p2p_comm_buffers[(i + 1) % 2][1]

            # [b, 2, sk//2, h, d] or
            # [2, sk//2, b, h, d]
            dk = dkv[: ctx.k_numel].view(*ctx.k_shape)
            dv = dkv[ctx.k_numel :].view(*ctx.v_shape)
            if causal and (i < (cp_size - rank - 1) or i == (cp_size - 1)):
                dk_ = dk_.view(*ctx.k_shape)
                dv_ = dv_.view(*ctx.v_shape)

            if ctx.fp8 and ctx.fp8_recipe.delayed():
                # fp8
                if causal and i >= (cp_size - rank - 1) and i != (cp_size - 1):
                    if ctx.qkv_format == "bshd":
                        dk[:, 0, ...].copy_(dk_)
                        dk[:, 1, ...].fill_(0)
                        dv[:, 0, ...].copy_(dv_)
                        dv[:, 1, ...].fill_(0)
                    elif ctx.qkv_format == "sbhd":
                        dk[0].copy_(dk_)
                        dk[1].fill_(0)
                        dv[0].copy_(dv_)
                        dv[1].fill_(0)
                    else:
                        dk.copy_(dk_)
                        dv.copy_(dv_)
                else:
                    dk.copy_(dk_)
                    dv.copy_(dv_)
            elif causal:
                # not fp8 and causal
                if i == (cp_size - 1):
                    if rank == 0:
                        if ctx.qkv_format == "bshd":
                            dk[:, 0, ...].add_(dk_[:, 0, ...])
                            dk[:, 1, ...].copy_(dk_[:, 1, ...])
                            dv[:, 0, ...].add_(dv_[:, 0, ...])
                            dv[:, 1, ...].copy_(dv_[:, 1, ...])
                        elif ctx.qkv_format == "sbhd":
                            dk[0, ...].add_(dk_[0, ...])
                            dk[1, ...].copy_(dk_[1, ...])
                            dv[0, ...].add_(dv_[0, ...])
                            dv[1, ...].copy_(dv_[1, ...])
                        elif ctx.qkv_format == "thd":
                            tex.thd_grad_correction(dk, dk_, cu_seqlens_kv_padded, "add", "copy")
                            tex.thd_grad_correction(dv, dv_, cu_seqlens_kv_padded, "add", "copy")
                    else:
                        dk.add_(dk_)
                        dv.add_(dv_)
                elif i >= (cp_size - rank - 1):
                    if i == 0 and rank == (cp_size - 1):
                        if ctx.qkv_format == "bshd":
                            dk[:, 0, ...].copy_(dk_)
                            dv[:, 0, ...].copy_(dv_)
                        elif ctx.qkv_format == "sbhd":
                            dk[0, ...].copy_(dk_)
                            dv[0, ...].copy_(dv_)
                        elif ctx.qkv_format == "thd":
                            tex.thd_grad_correction(dk, dk_, cu_seqlens_kv_padded, "copy", "none")
                            tex.thd_grad_correction(dv, dv_, cu_seqlens_kv_padded, "copy", "none")
                    else:
                        if ctx.qkv_format == "bshd":
                            dk[:, 0, ...].add_(dk_)
                            dv[:, 0, ...].add_(dv_)
                        elif ctx.qkv_format == "sbhd":
                            dk[0, ...].add_(dk_)
                            dv[0, ...].add_(dv_)
                        elif ctx.qkv_format == "thd":
                            tex.thd_grad_correction(dk, dk_, cu_seqlens_kv_padded, "add", "none")
                            tex.thd_grad_correction(dv, dv_, cu_seqlens_kv_padded, "add", "none")
                elif i > 0:
                    dk.add_(dk_)
                    dv.add_(dv_)
                else:  # i == 0
                    dk.copy_(dk_)
                    dv.copy_(dv_)
            else:
                # not fp8 and not causal
                if i == 0:
                    dk.copy_(dk_)
                    dv.copy_(dv_)
                else:  # i > 0
                    dk.add_(dk_)
                    dv.add_(dv_)

        # sum up all cp_size for dq, dk, dv
        if ctx.fp8 and ctx.use_fused_attention:
            amax_cp_bwd = amax_per_step.amax(dim=1)
            ctx.dP_quantizer.amax.copy_(amax_cp_bwd[0])
            ctx.dQKV_quantizer.amax.copy_(amax_cp_bwd[1])

            dq = dq_buffer
            if ctx.fp8_recipe.delayed():
                # [cp, b, 2, sk//2, h, d] or [cp, 2, sk//2, b, h, d]
                dk = dkv_recv_buffer[:, : ctx.k_numel].view(cp_size, *ctx.k_shape)
                dv = dkv_recv_buffer[:, ctx.k_numel :].view(cp_size, *ctx.v_shape)
                dq, dk, dv = [
                    ctx.dQKV_quantizer.create_tensor_from_data(
                        x, fake_dtype=bwd_nominal_dtype, internal=ctx.dQKV_quantizer.internal
                    )
                    for x in [dq, dk, dv]
                ]
                dq, dk, dv = combine_and_dequantize(
                    qkv_layout,
                    dq,
                    dk,
                    dv,
                    src_nominal_dtype=bwd_nominal_dtype,
                    des_nominal_dtype=torch.float32,
                )
                dq, dk, dv = [x.sum(dim=0).to(bwd_nominal_dtype) for x in [dq, dk, dv]]

            if ctx.fp8_recipe.float8_current_scaling():
                dk = dkv[: ctx.k_numel].view(ctx.k_shape)
                dv = dkv[ctx.k_numel :].view(ctx.v_shape)

        if causal and ctx.qkv_format in ["bshd", "sbhd"]:
            # [b, 2, s//2, h, d] -> [b, s, h, d]
            # [2, s//2, b, h, d] -> [s, b, h, d]
            dim = ctx.qkv_format.index("s")
            dq, dk, dv = [x.view(*x.shape[:dim], -1, *x.shape[dim + 2 :]) for x in [dq, dk, dv]]

        if ctx.qkv_format == "thd" and not ctx.use_fused_attention:
            # Avoid GPU→CPU sync (forbidden in CUDA graph backward capture stream).
            # torch.cuda.is_current_stream_capturing() is a CPU-side check, safe to call.
            # At capture time: cu_seqlens_q_padded[-1] == dq.shape[0] (dummy PSP), so
            # fill is a no-op. At replay, FA backward inits dq to zero for unused positions.
            if torch.cuda.is_current_stream_capturing():
                _q_end, _kv_end = dq.shape[0], dk.shape[0]
            else:
                _q_end = cu_seqlens_q_padded[-1]
                _kv_end = cu_seqlens_kv_padded[-1]
            dq[_q_end:].fill_(0)
            dk[_kv_end:].fill_(0)
            dv[_kv_end:].fill_(0)

        if ctx.fp8 and ctx.is_input_fp8:
            dq, dk, dv = combine_and_quantize(qkv_layout, dq, dk, dv, ctx.dQKV_quantizer)

        if ctx.fp8:
            # print quantizers
            print_quantizers(
                "AttnFuncWithCPAndKVP2P.backward >> after:  ",
                ctx.layer_number,
                ctx.QKV_quantizer,
                ctx.O_quantizer,
                ctx.S_quantizer,
                ctx.dQKV_quantizer,
                ctx.dO_quantizer,
                ctx.dP_quantizer,
            )

        if cp_size_a2a > 1:
            if ctx.fp8 and ctx.is_input_fp8:
                dq_fp8, dk_fp8, dv_fp8 = dq, dk, dv
                dq, dk, dv = (dq_fp8._data, dk_fp8._data, dv_fp8._data)
            chunk_ids_for_a2a = get_seq_chunk_ids_for_reordering_after_attn(cp_size_a2a, q.device)
            dq, dk, dv = flash_attn_a2a_communicate(
                [dq, dk, dv],
                chunk_ids_for_a2a,
                seq_dim,
                cp_size_a2a,
                ctx.cp_group_a2a,
                ctx.cp_stream,
                False,
            )
            if ctx.fp8 and ctx.is_input_fp8:
                dq, dk, dv = [
                    Float8Tensor.make_like(x, data=y, dtype=bwd_nominal_dtype)
                    for x, y in zip([dq_fp8, dk_fp8, dv_fp8], [dq, dk, dv])
                ]
            if ctx.qkv_format == "bshd":
                dq, dk, dv = [x.view(ctx.batch_size, -1, *x.shape[-2:]) for x in [dq, dk, dv]]
            elif ctx.qkv_format == "sbhd":
                dq, dk, dv = [x.view(-1, ctx.batch_size, *x.shape[-2:]) for x in [dq, dk, dv]]

        if attn_dbias is not None:
            # [b, h, sq, 2*cp, sk//(2*cp)] -> [b, h, sq, sk]
            attn_dbias = attn_dbias.view(*attn_dbias.shape[:-2], -1)

        nvtx_range_pop(f"{nvtx_label}")

        return (
            None,
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            attn_dbias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def get_kv_seq_info_after_all_gather(
    local_chunk_id, cp_size, max_seqlen_q, max_seqlen_kv, window_size, causal
):
    """Compute KV sequence index range and update window size after all-gather."""
    local_chunk_end_idx = (local_chunk_id + 1) * max_seqlen_kv
    full_seq_end_idx = max_seqlen_kv * cp_size * 2

    if window_size is None:
        window_size = (-1, 0) if causal else (-1, -1)

    if window_size[1] == -1:
        seq_end_idx = full_seq_end_idx
        window_size_right = -1
    else:
        seq_end_idx = min(full_seq_end_idx, local_chunk_end_idx + window_size[1])
        window_size_right = local_chunk_end_idx + window_size[1] - seq_end_idx

    if window_size[0] == -1:
        seq_start_idx = 0
        window_size_left = -1
    else:
        seq_start_idx = max(0, local_chunk_end_idx - max_seqlen_q - window_size[0])
        window_size_left = window_size[0] + seq_end_idx - local_chunk_end_idx

    return (seq_start_idx, seq_end_idx), (window_size_left, window_size_right)


class AttnFuncWithCPAndKVAllGather(torch.autograd.Function):
    """
    Attention implementation with context parallelism. KV all-gather between CP ranks is exposed.
    Refer section 3.3.2 of `The Llama 3 Herd of Models <https://arxiv.org/abs/2407.21783>`_.
    """

    @staticmethod
    def forward(
        ctx,
        is_training,
        q,
        k,
        v,
        cu_seqlens_q,
        max_seqlen_q,
        max_seqlen_kv,
        cu_seqlens_q_padded,
        dropout_p,
        softmax_scale,
        qkv_format,
        attn_mask_type,
        attn_bias_type,
        attn_bias,
        deterministic,
        use_fused_attention,
        return_max_logit,
        window_size,
        cp_group,
        cp_stream,
        use_flash_attn_3,
    ):
        # pylint: disable=missing-function-docstring
        nvtx_range_push("transformer_engine.AttnFuncWithCPAndKVAllGather.forward")
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        cp_size = get_distributed_world_size(cp_group)
        rank = get_distributed_rank(cp_group)

        qkv_dtype = q.dtype

        causal = "causal" in attn_mask_type
        padding = "padding" in attn_mask_type
        assert not padding, f"{attn_mask_type} mask type is not supported!"
        if use_fused_attention and causal and "bottom_right" not in attn_mask_type:
            attn_mask_type = attn_mask_type + "_bottom_right"
        assert attn_bias_type == "no_bias", f"{attn_bias_type} bias type is not supported!"
        assert q.shape[-1] % 8 == 0, "Hidden size per attention head should be multiple of 8!"
        assert (
            use_fused_attention or fa_utils.v2_3_plus
        ), "Sliding window attention only can work with FusedAttention or FlashAttention >= 2.3!"

        flash_attn_fwd = None
        if not use_fused_attention:
            fa_forward_kwargs = {"softmax_scale": softmax_scale}
            if use_flash_attn_3:
                from transformer_engine.pytorch.attention.dot_product_attention.backends import (
                    _flash_attn_fwd_v3,
                )

                flash_attn_fwd = _flash_attn_fwd_v3
            else:
                if qkv_format == "thd":
                    from transformer_engine.pytorch.attention.dot_product_attention.backends import (
                        _flash_attn_varlen_fwd,
                    )

                    flash_attn_fwd = _flash_attn_varlen_fwd
                else:
                    from transformer_engine.pytorch.attention.dot_product_attention.backends import (
                        _flash_attn_fwd,
                    )

                    flash_attn_fwd = _flash_attn_fwd
                fa_forward_kwargs["dropout_p"] = dropout_p
                fa_forward_kwargs["return_softmax"] = False
                if fa_utils.v2_4_plus:
                    fa_forward_kwargs["alibi_slopes"] = None
                if fa_utils.v2_5_7_plus and qkv_format == "thd":
                    fa_forward_kwargs["block_table"] = None
                if fa_utils.v2_6_0_plus:
                    fa_forward_kwargs["softcap"] = 0.0

        assert qkv_format != "thd", f"{qkv_format} format is not supported!"
        qkv_layout = qkv_format + "_" + qkv_format + "_" + qkv_format

        seq_dim = qkv_format.index("s")
        assert (
            q.shape[seq_dim] % 2 == 0 and k.shape[seq_dim] % 2 == 0
        ), "Sequence length per GPU needs to be divisible by 2!"

        max_seqlen_q = max_seqlen_q // (2 * cp_size)
        max_seqlen_kv = max_seqlen_kv // (2 * cp_size)
        if use_fused_attention or qkv_format == "thd":
            cu_seqlens_q = cu_seqlens_q // (2 * cp_size)
        if cu_seqlens_q_padded is not None and qkv_format == "thd":
            cu_seqlens_q_padded = cu_seqlens_q_padded // (2 * cp_size)
        else:
            cu_seqlens_q_padded = None

        # [b, s, h, d] -> [b, 2, s//2, h, d] or [s, b, h, d] -> [2, s//2, b, h, d]
        q = q.view(*q.shape[:seq_dim], 2, q.shape[seq_dim] // 2, *q.shape[(seq_dim + 1) :])
        # [b, s, h, d] or [s, b, h, d] -> [s, b, h, d]
        k, v = [x.movedim(seq_dim, 0).contiguous() for x in [k, v]]

        # [s, b, h, d] -> [cp, s, b, h, d]
        k_ag, _ = gather_along_first_dim(k, cp_group)
        v_ag, _ = gather_along_first_dim(v, cp_group)

        # [cp, s, b, h, d] -> [cp*2, s//2, b, h, d]
        k_ag = k_ag.view(2 * cp_size, k.shape[0] // 2, *k.shape[1:])
        v_ag = v_ag.view(2 * cp_size, v.shape[0] // 2, *v.shape[1:])
        chunk_ids_for_kv_ag = get_seq_chunk_ids_for_reordering_before_attn(cp_size, k.device)
        k_ag = torch.index_select(k_ag, dim=0, index=chunk_ids_for_kv_ag)
        v_ag = torch.index_select(v_ag, dim=0, index=chunk_ids_for_kv_ag)
        # [cp*2, s//2, b, h, d] -> [cp*s, b, h, d]
        k_ag = k_ag.view(-1, *k.shape[1:])
        v_ag = v_ag.view(-1, *v.shape[1:])
        cp_stream.wait_stream(torch.cuda.current_stream())

        # create two streams to resolve wave quantization issue of Flash Attn in each step
        flash_attn_streams = [torch.cuda.current_stream(), cp_stream]

        local_seq_chunk_ids = [rank, 2 * cp_size - rank - 1]
        kv_seq_range_per_step = [None, None]
        window_size_per_step = [None, None]
        cu_seqlens_kv_per_step = [None, None]
        out_per_step = [None, None]
        softmax_lse_per_step = [None, None]
        rng_states = [None, None]
        out = torch.empty_like(q)
        max_logit_per_step = [None, None]
        max_logit = None

        for i in range(len(local_seq_chunk_ids) + 1):
            if i < len(local_seq_chunk_ids):
                with torch.cuda.stream(flash_attn_streams[i]):
                    # [b, 2, sq//2, h, d] -> [b, sq//2, h, d]
                    # or [2, sq//2, b, h, d] -> [sq//2, b, h, d]
                    q_ = q.select(seq_dim, i).contiguous()
                    kv_seq_range_per_step[i], window_size_per_step[i] = (
                        get_kv_seq_info_after_all_gather(
                            local_seq_chunk_ids[i],
                            cp_size,
                            max_seqlen_q,
                            max_seqlen_kv,
                            window_size,
                            causal,
                        )
                    )
                    seq_start_idx, seq_end_idx = (
                        kv_seq_range_per_step[i][0],
                        kv_seq_range_per_step[i][1],
                    )
                    max_seqlen_kv_ = seq_end_idx - seq_start_idx
                    if use_fused_attention or qkv_format == "thd":
                        cu_seqlens_kv_per_step[i] = dpa_utils.get_full_cu_seqlens(
                            k.shape[1], max_seqlen_kv_, k.device
                        )
                    k_, v_ = [x[seq_start_idx:seq_end_idx] for x in [k_ag, v_ag]]
                    # [s_range, b, h, d] -> [b, s_range, h, d] or [s_range, b, h, d]
                    k_, v_ = [x.movedim(0, seq_dim).contiguous() for x in [k_, v_]]
                    if use_fused_attention:
                        (
                            out_per_step[i],
                            [softmax_lse_per_step[i], rng_states[i]],
                            *max_logit_,
                        ) = fused_attn_fwd(
                            is_training,
                            max_seqlen_q,
                            max_seqlen_kv_,
                            cu_seqlens_q,
                            cu_seqlens_kv_per_step[i],
                            q_,
                            k_,
                            v_,
                            qkv_dtype,
                            tex.NVTE_Fused_Attn_Backend.NVTE_F16_arbitrary_seqlen,
                            attn_scale=softmax_scale,
                            dropout=dropout_p,
                            qkv_layout=qkv_layout,
                            attn_mask_type=attn_mask_type,
                            attn_bias_type=attn_bias_type,
                            attn_bias=attn_bias,
                            cu_seqlens_q_padded=cu_seqlens_q_padded,
                            cu_seqlens_kv_padded=cu_seqlens_kv_per_step[i],
                            window_size=window_size_per_step[i],
                            return_max_logit=return_max_logit,
                            cuda_graph=is_graph_capturing(),
                        )
                        if return_max_logit:
                            max_logit_per_step[i] = max_logit_[0]
                    else:
                        fa_forward_args_thd = get_fa_args(
                            True,
                            use_flash_attn_3,
                            qkv_format,
                            cu_seqlens_q=cu_seqlens_q,
                            cu_seqlens_kv=cu_seqlens_kv_per_step[i],
                            max_seqlen_q=max_seqlen_q,
                            max_seqlen_kv=max_seqlen_kv_,
                        )
                        if use_flash_attn_3 or (fa_utils.v2_3_plus and not fa_utils.v2_7_0_plus):
                            fa_forward_kwargs["window_size"] = window_size_per_step[i]
                        elif fa_utils.v2_7_0_plus:
                            fa_forward_kwargs["window_size_left"] = window_size_per_step[i][0]
                            fa_forward_kwargs["window_size_right"] = window_size_per_step[i][1]
                        fa_outputs = flash_attn_fwd(
                            q_,
                            k_,
                            v_,
                            *fa_forward_args_thd,
                            causal=causal,
                            **fa_forward_kwargs,
                        )
                        if not fa_utils.v2_7_0_plus:
                            out_per_step[i] = fa_outputs[4]
                            softmax_lse_per_step[i] = fa_outputs[5]
                            if not use_flash_attn_3:
                                rng_states[i] = fa_outputs[7]
                        else:
                            out_per_step[i] = fa_outputs[0]
                            softmax_lse_per_step[i] = fa_outputs[1]
                            if not use_flash_attn_3:
                                rng_states[i] = fa_outputs[3]

            if return_max_logit and i == 0:
                max_logit = torch.clone(max_logit_per_step[0])
            if i > 0:
                with torch.cuda.stream(flash_attn_streams[i - 1]):
                    if qkv_format == "bshd":
                        out[:, i - 1].copy_(out_per_step[i - 1])
                    elif qkv_format == "sbhd":
                        out[i - 1].copy_(out_per_step[i - 1])
                if return_max_logit:
                    max_logit = torch.maximum(max_logit, max_logit_per_step[i - 1])

        torch.cuda.current_stream().wait_stream(cp_stream)
        if return_max_logit:
            torch.distributed.all_reduce(
                max_logit, op=torch.distributed.ReduceOp.MAX, group=cp_group
            )

        if use_fused_attention:
            if qkv_format == "bshd":
                out = out.view(out.shape[0], -1, *out.shape[-2:])
            elif qkv_format == "sbhd":
                out = out.view(-1, *out.shape[-3:])
        else:
            out = out.view(-1, *out.shape[-2:])

        ctx.save_for_backward(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_q_padded,
            *cu_seqlens_kv_per_step,
            *out_per_step,
            *softmax_lse_per_step,
            *rng_states,
        )

        ctx.qkv_dtype = qkv_dtype
        ctx.kv_seq_range_per_step = kv_seq_range_per_step
        ctx.window_size_per_step = window_size_per_step
        ctx.cp_group = cp_group
        ctx.cp_stream = cp_stream
        ctx.dropout_p = dropout_p
        ctx.max_seqlen_q = max_seqlen_q
        ctx.softmax_scale = softmax_scale
        ctx.qkv_format = qkv_format
        ctx.attn_bias_type = attn_bias_type
        ctx.attn_mask_type = attn_mask_type
        ctx.deterministic = deterministic
        ctx.use_fused_attention = use_fused_attention
        ctx.use_flash_attn_3 = use_flash_attn_3
        nvtx_range_pop("transformer_engine.AttnFuncWithCPAndKVAllGather.forward")
        if return_max_logit:
            return out, max_logit
        return out

    @staticmethod
    def backward(ctx, dout, *_args):
        # pylint: disable=missing-function-docstring
        nvtx_range_push("transformer_engine.AttnFuncWithCPAndKVAllGather.backward")
        cp_size = get_distributed_world_size(ctx.cp_group)
        rank = get_distributed_rank(ctx.cp_group)

        (*saved_tensors,) = ctx.saved_tensors
        (q, k, v, cu_seqlens_q, cu_seqlens_q_padded) = saved_tensors[:5]
        cu_seqlens_kv_per_step = saved_tensors[5:7]
        out_per_step = saved_tensors[7:9]
        softmax_lse_per_step = saved_tensors[9:11]
        rng_states = saved_tensors[11:13]
        kv_seq_range_per_step = ctx.kv_seq_range_per_step
        window_size_per_step = ctx.window_size_per_step

        seq_dim = ctx.qkv_format.index("s")
        qkv_layout = ctx.qkv_format + "_" + ctx.qkv_format + "_" + ctx.qkv_format

        dout = dout.view(q.shape)
        dq = torch.empty_like(q)
        dk = torch.zeros((k.shape[0] * cp_size, *k.shape[1:]), dtype=k.dtype, device=k.device)
        dv = torch.zeros_like(dk)
        dq_per_step = [None, None]
        dk_per_step = [None, None]
        dv_per_step = [None, None]

        # create two streams to resolve wave quantization issue of Flash Attn in each step
        flash_attn_streams = [torch.cuda.current_stream(), ctx.cp_stream]
        # synchronize dkv update across steps
        dkv_update_done = torch.cuda.Event()

        # [s, b, h, d] -> [cp, s, b, h, d]
        k_ag, _ = gather_along_first_dim(k, ctx.cp_group)
        v_ag, _ = gather_along_first_dim(v, ctx.cp_group)

        # [cp, s, b, h, d] -> [cp*2, s//2, b, h, d]
        k_ag = k_ag.view(2 * cp_size, k.shape[0] // 2, *k.shape[1:])
        v_ag = v_ag.view(2 * cp_size, v.shape[0] // 2, *v.shape[1:])
        chunk_ids_for_kv_ag = get_seq_chunk_ids_for_reordering_before_attn(cp_size, k.device)
        k_ag = torch.index_select(k_ag, dim=0, index=chunk_ids_for_kv_ag)
        v_ag = torch.index_select(v_ag, dim=0, index=chunk_ids_for_kv_ag)
        # [cp*2, s//2, b, h, d] -> [cp*s, b, h, d]
        k_ag = k_ag.view(-1, *k.shape[1:])
        v_ag = v_ag.view(-1, *v.shape[1:])
        ctx.cp_stream.wait_stream(torch.cuda.current_stream())

        local_seq_chunk_ids = [rank, 2 * cp_size - rank - 1]

        flash_attn_bwd = None
        if not ctx.use_fused_attention:
            fa_backward_kwargs = {"softmax_scale": ctx.softmax_scale}
            if ctx.use_flash_attn_3:
                from transformer_engine.pytorch.attention.dot_product_attention.backends import (
                    _flash_attn_bwd_v3,
                )

                flash_attn_bwd = _flash_attn_bwd_v3
                fa_backward_kwargs["deterministic"] = ctx.deterministic
            else:
                if ctx.qkv_format == "thd":
                    from transformer_engine.pytorch.attention.dot_product_attention.backends import (
                        _flash_attn_varlen_bwd,
                    )

                    flash_attn_bwd = _flash_attn_varlen_bwd
                else:
                    from transformer_engine.pytorch.attention.dot_product_attention.backends import (
                        _flash_attn_bwd,
                    )

                    flash_attn_bwd = _flash_attn_bwd
                fa_backward_kwargs["dropout_p"] = ctx.dropout_p
                if fa_utils.v2_4_plus:
                    fa_backward_kwargs["alibi_slopes"] = None
                if fa_utils.v2_4_1_plus:
                    fa_backward_kwargs["deterministic"] = ctx.deterministic
                if fa_utils.v2_6_0_plus:
                    fa_backward_kwargs["softcap"] = 0.0

        for i in range(len(local_seq_chunk_ids) + 1):
            if i < len(local_seq_chunk_ids):
                with torch.cuda.stream(flash_attn_streams[i]):
                    # [b, 2, sq//2, h, d] -> [b, sq//2, h, d]
                    # or [2, sq//2, b, h, d] -> [sq//2, b, h, d]
                    q_ = q.select(seq_dim, i).contiguous()
                    seq_start_idx, seq_end_idx = (
                        kv_seq_range_per_step[i][0],
                        kv_seq_range_per_step[i][1],
                    )
                    max_seqlen_kv = seq_end_idx - seq_start_idx
                    k_, v_ = [x[seq_start_idx:seq_end_idx] for x in [k_ag, v_ag]]
                    # [cp*s, b, h, d] -> [b, s_range, h, d] or [s_range, b, h, d]
                    k_, v_ = [x.movedim(0, seq_dim).contiguous() for x in [k_, v_]]
                    out_ = out_per_step[i]
                    dout_ = dout.select(seq_dim, i).contiguous().view(out_.shape)
                    if ctx.use_fused_attention:
                        aux_ctx_tensors = [softmax_lse_per_step[i], rng_states[i]]
                        dq_per_step[i], dk_per_step[i], dv_per_step[i], *_ = fused_attn_bwd(
                            ctx.max_seqlen_q,
                            max_seqlen_kv,
                            cu_seqlens_q,
                            cu_seqlens_kv_per_step[i],
                            q_,
                            k_,
                            v_,
                            out_,
                            dout_,
                            ctx.qkv_dtype,
                            TE_DType[dout.dtype],
                            aux_ctx_tensors,
                            tex.NVTE_Fused_Attn_Backend.NVTE_F16_arbitrary_seqlen,
                            cu_seqlens_q_padded=cu_seqlens_q_padded,
                            cu_seqlens_kv_padded=cu_seqlens_kv_per_step[i],
                            attn_scale=ctx.softmax_scale,
                            dropout=ctx.dropout_p,
                            qkv_layout=qkv_layout,
                            attn_mask_type=ctx.attn_mask_type,
                            attn_bias_type=ctx.attn_bias_type,
                            window_size=window_size_per_step[i],
                            deterministic=ctx.deterministic,
                            cuda_graph=is_graph_capturing(),
                        )
                    else:
                        dq_per_step[i], dk_per_step[i], dv_per_step[i] = [
                            torch.empty_like(x) for x in [q_, k_, v_]
                        ]
                        fa_backward_args_thd = get_fa_args(
                            False,
                            ctx.use_flash_attn_3,
                            ctx.qkv_format,
                            cu_seqlens_q=cu_seqlens_q,
                            cu_seqlens_kv=cu_seqlens_kv_per_step[i],
                            max_seqlen_q=ctx.max_seqlen_q,
                            max_seqlen_kv=max_seqlen_kv,
                            dq=dq_per_step[i],
                            dk=dk_per_step[i],
                            dv=dv_per_step[i],
                        )
                        if not ctx.use_flash_attn_3:
                            fa_backward_kwargs["rng_state"] = rng_states[i]
                        if ctx.use_flash_attn_3 or (
                            fa_utils.v2_3_plus and not fa_utils.v2_7_0_plus
                        ):
                            fa_backward_kwargs["window_size"] = window_size_per_step[i]
                        elif fa_utils.v2_7_0_plus:
                            fa_backward_kwargs["window_size_left"] = window_size_per_step[i][0]
                            fa_backward_kwargs["window_size_right"] = window_size_per_step[i][1]
                        flash_attn_bwd(
                            dout_,
                            q_,
                            k_,
                            v_,
                            out_,
                            softmax_lse_per_step[i],
                            *fa_backward_args_thd,
                            causal="causal" in ctx.attn_mask_type,
                            **fa_backward_kwargs,
                        )

            if i > 0:
                with torch.cuda.stream(flash_attn_streams[i - 1]):
                    if ctx.qkv_format == "bshd":
                        dq[:, i - 1].copy_(dq_per_step[i - 1])
                    elif ctx.qkv_format == "sbhd":
                        dq[i - 1].copy_(dq_per_step[i - 1])
                    # [b, s_range, h, d] or [s_range, b, h, d] -> [s_range, b, h, d]
                    dk_per_step[i - 1], dv_per_step[i - 1] = [
                        x.movedim(seq_dim, 0).contiguous()
                        for x in [dk_per_step[i - 1], dv_per_step[i - 1]]
                    ]
                    # wait until dkv update of last step is done
                    if i > 1:
                        flash_attn_streams[i - 1].wait_event(dkv_update_done)
                    seq_start_idx, seq_end_idx = (
                        kv_seq_range_per_step[i - 1][0],
                        kv_seq_range_per_step[i - 1][1],
                    )
                    dk[seq_start_idx:seq_end_idx].add_(dk_per_step[i - 1])
                    dv[seq_start_idx:seq_end_idx].add_(dv_per_step[i - 1])
                    if i < len(local_seq_chunk_ids):
                        flash_attn_streams[i - 1].record_event(dkv_update_done)

        torch.cuda.current_stream().wait_stream(ctx.cp_stream)

        # [cp*s, b, h, d] -> [cp*2, s//2, b, h, d]
        dk = dk.view(2 * cp_size, -1, *dk.shape[-3:])
        dv = dv.view(2 * cp_size, -1, *dv.shape[-3:])
        chunk_ids_for_kv_ag = get_seq_chunk_ids_for_reordering_after_attn(cp_size, dk.device)
        dk = torch.index_select(dk, dim=0, index=chunk_ids_for_kv_ag)
        dv = torch.index_select(dv, dim=0, index=chunk_ids_for_kv_ag)
        # [cp*2, s//2, b, h, d] -> [cp*s, b, h, d]
        dk = dk.view(-1, *dk.shape[-3:])
        dv = dv.view(-1, *dv.shape[-3:])
        dk, _ = reduce_scatter_along_first_dim(dk, ctx.cp_group)
        dv, _ = reduce_scatter_along_first_dim(dv, ctx.cp_group)

        dq = dq.view(*dq.shape[:seq_dim], -1, *dq.shape[(seq_dim + 2) :])
        dk = dk.movedim(0, seq_dim).contiguous()
        dv = dv.movedim(0, seq_dim).contiguous()
        nvtx_range_pop("transformer_engine.AttnFuncWithCPAndKVAllGather.backward")

        return (
            None,
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class AttnFuncWithCPAndQKVOA2A(torch.autograd.Function):
    """
    Attention implementation with context parallelism. Like Ulysses, applying A2A to QKVO.
    Refer the paper `DeepSpeed Ulysses <https://arxiv.org/abs/2309.14509>`_.
    """

    @staticmethod
    def forward(
        ctx,
        is_training,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
        dropout_p,
        softmax_scale,
        qkv_format,
        attn_mask_type,
        attn_bias_type,
        attn_bias,
        deterministic,
        use_fused_attention,
        return_max_logit,
        window_size,
        fp8,
        fp8_meta,
        cp_group,
        cp_stream,
        quantizers,
        use_flash_attn_3,
        softmax_type,
        softmax_offset,
        fp8_output,
    ):
        # pylint: disable=missing-function-docstring
        nvtx_range_push("transformer_engine.AttnFuncWithCPAndQKVOA2A.forward")
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        cp_size = get_distributed_world_size(cp_group)

        causal = "causal" in attn_mask_type
        padding = "padding" in attn_mask_type
        assert (
            not padding or qkv_format == "thd"
        ), f"{attn_mask_type} mask type is not supported for BSHD and SBHD!"
        assert attn_bias_type == "no_bias", f"{attn_bias_type} bias type is not supported!"
        assert q.shape[-1] % 8 == 0, "Hidden size per attention head should be multiple of 8!"
        assert (
            window_size == (-1, 0)
            or window_size == (-1, -1)
            or use_fused_attention
            or fa_utils.v2_3_plus
        ), "Sliding window attention only can work with FusedAttention or FlashAttention >= 2.3!"

        flash_attn_fwd = None
        if not use_fused_attention:
            fa_forward_kwargs = {"softmax_scale": softmax_scale}
            if use_flash_attn_3:
                from transformer_engine.pytorch.attention.dot_product_attention.backends import (
                    _flash_attn_fwd_v3,
                )

                flash_attn_fwd = _flash_attn_fwd_v3
                fa_forward_kwargs["window_size"] = window_size
            else:
                if qkv_format == "thd":
                    from transformer_engine.pytorch.attention.dot_product_attention.backends import (
                        _flash_attn_varlen_fwd,
                    )

                    flash_attn_fwd = _flash_attn_varlen_fwd
                else:
                    from transformer_engine.pytorch.attention.dot_product_attention.backends import (
                        _flash_attn_fwd,
                    )

                    flash_attn_fwd = _flash_attn_fwd
                fa_forward_kwargs["dropout_p"] = dropout_p
                fa_forward_kwargs["return_softmax"] = False
                if fa_utils.v2_3_plus and not fa_utils.v2_7_0_plus:
                    fa_forward_kwargs["window_size"] = window_size
                elif fa_utils.v2_7_0_plus:
                    fa_forward_kwargs["window_size_left"] = window_size[0]
                    fa_forward_kwargs["window_size_right"] = window_size[1]
                if fa_utils.v2_4_plus:
                    fa_forward_kwargs["alibi_slopes"] = None
                if fa_utils.v2_5_7_plus and qkv_format == "thd":
                    fa_forward_kwargs["block_table"] = None
                if fa_utils.v2_6_0_plus:
                    fa_forward_kwargs["softcap"] = 0.0

        assert (
            q.shape[-2] % cp_size == 0 and k.shape[-2] % cp_size == 0
        ), "The number of attention heads needs to be divisible by CP size!"

        qkv_layout = qkv_format + "_" + qkv_format + "_" + qkv_format

        if qkv_format in ["bshd", "sbhd"]:
            batch_dim = qkv_format.index("b")
            seq_dim = qkv_format.index("s")
        else:  # qkv_format == "thd"
            batch_dim = seq_dim = qkv_format.index("t")

        assert (
            q.shape[seq_dim] % 2 == 0 and k.shape[seq_dim] % 2 == 0
        ), "Sequence length per GPU needs to be divisible by 2!"

        assert isinstance(k, q.__class__) and isinstance(
            v, q.__class__
        ), "q, k, v must be of the same class, e.g. torch.Tensor or Float8Tensor."
        is_input_fp8 = isinstance(q, Float8Tensor)
        is_output_fp8 = fp8_output
        is_bwd_fp8 = int(os.getenv("NVTE_FP8_DPA_BWD", "1"))
        # recipe passed in through autocast or set by NVTE_DPA_FP8_RECIPE;
        # may be different from fp8_meta["recipe"]
        fp8_recipe = FP8GlobalStateManager.get_fp8_recipe()
        if fp8_meta is not None and fp8_meta.get("local_recipes", None) is not None:
            fp8_recipe = fp8_meta["local_recipes"][0]
        fwd_nominal_dtype = q.dtype
        fused_attn_backend = None
        max_logit = None

        QKV_quantizer, O_quantizer, S_quantizer, dQKV_quantizer, dO_quantizer, dP_quantizer = (
            dpa_utils.get_attention_quantizers(fp8, quantizers)
        )

        q_fp8, k_fp8, v_fp8 = (None, None, None)
        if fp8:
            if use_fused_attention:
                fused_attn_backend = FusedAttnBackend["FP8"]
                if is_input_fp8:
                    q_fp8, k_fp8, v_fp8 = q, k, v
                    q, k, v = q_fp8._data, k_fp8._data, v_fp8._data
                else:
                    q_fp8, k_fp8, v_fp8 = combine_and_quantize(qkv_layout, q, k, v, QKV_quantizer)
                    q, k, v = [q_fp8._data, k_fp8._data, v_fp8._data]
                fp8_meta_kwargs = {}
                fp8_meta_kwargs["s_quantizer"] = S_quantizer
                fp8_meta_kwargs["o_quantizer"] = O_quantizer
            else:
                assert False, "FP8 is only supported with Fused Attention!"
        else:
            if use_fused_attention:
                fp8_meta_kwargs = {}
                fused_attn_backend = FusedAttnBackend["F16_arbitrary_seqlen"]

        chunk_ids_for_a2a = get_seq_chunk_ids_for_reordering_before_attn(cp_size, q.device)
        q, k, v = flash_attn_a2a_communicate(
            [q, k, v],
            chunk_ids_for_a2a,
            seq_dim,
            cp_size,
            cp_group,
            cp_stream,
            before_attn=True,
            qkv_format=qkv_format,
            cu_seqlens_padded=cu_seqlens_q_padded,
        )
        if softmax_type != "vanilla":
            softmax_offset = flash_attn_a2a_communicate_softmax_offset(
                softmax_offset, 1, cp_size, cp_group, cp_stream, True
            )

        out_fp8 = None
        out_f16 = None
        batch_size = q.shape[batch_dim]
        q_part, k_part, v_part = q, k, v
        out_part = None
        if use_fused_attention:
            if fp8:
                q_part, k_part, v_part = [
                    Float8Tensor.make_like(x, data=y, dtype=fwd_nominal_dtype)
                    for x, y in zip([q_fp8, k_fp8, v_fp8], [q_part, k_part, v_part])
                ]
            out_, aux_ctx_tensors, *max_logit = fused_attn_fwd(
                is_training,
                max_seqlen_q,
                max_seqlen_kv,
                cu_seqlens_q,
                cu_seqlens_kv,
                q_part,
                k_part,
                v_part,
                fwd_nominal_dtype,
                fused_attn_backend,
                attn_scale=softmax_scale,
                dropout=dropout_p,
                qkv_layout=qkv_layout,
                attn_mask_type=attn_mask_type,
                attn_bias_type=attn_bias_type,
                attn_bias=attn_bias,
                cu_seqlens_q_padded=cu_seqlens_q_padded,
                cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                window_size=window_size,
                **fp8_meta_kwargs,
                softmax_type=softmax_type,
                softmax_offset=softmax_offset,
                return_max_logit=return_max_logit,
                cuda_graph=is_graph_capturing(),
            )
            if isinstance(out_, Float8Tensor):
                out_fp8 = out_
                out_ = out_._data
                if is_bwd_fp8 and not (
                    fp8_recipe.float8_current_scaling() and _dpa_fp8_cs_o_in_f16
                ):
                    out_part = out_fp8
                else:
                    out_part = out_fp8.dequantize(dtype=fwd_nominal_dtype)
            else:
                out_f16 = out_
                out_part = out_
                if (
                    fp8
                    and is_bwd_fp8
                    and not (fp8_recipe.float8_current_scaling() and _dpa_fp8_cs_o_in_f16)
                ):
                    out_part = O_quantizer(out_)
        else:
            fa_forward_args_thd = get_fa_args(
                True,
                use_flash_attn_3,
                qkv_format,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
            )
            fa_outputs = flash_attn_fwd(
                q_part,
                k_part,
                v_part,
                *fa_forward_args_thd,
                causal=causal,
                **fa_forward_kwargs,
            )
            if not fa_utils.v2_7_0_plus:
                out_, softmax_lse = fa_outputs[4], fa_outputs[5]
                rng_state = fa_outputs[7] if not use_flash_attn_3 else None
            else:
                out_, softmax_lse = fa_outputs[0], fa_outputs[1]
                rng_state = fa_outputs[3] if not use_flash_attn_3 else None
            aux_ctx_tensors = [softmax_lse, rng_state]
            out_part = out_

        chunk_ids_for_a2a = get_seq_chunk_ids_for_reordering_after_attn(cp_size, out_.device)
        out_ = flash_attn_a2a_communicate(
            out_,
            chunk_ids_for_a2a,
            seq_dim,
            cp_size,
            cp_group,
            cp_stream,
            before_attn=False,
            qkv_format=qkv_format,
            cu_seqlens_padded=cu_seqlens_q_padded,
        )
        if return_max_logit:
            max_logit = flash_attn_a2a_communicate_softmax_offset(
                *max_logit, 0, cp_size, cp_group, cp_stream, False
            )

        if use_fused_attention:
            if qkv_format == "bshd":
                # [b*s, h, d] -> [b, s, h, d]
                out_ = out_.view(batch_size, -1, *out_.shape[-2:])
            elif qkv_format == "sbhd":
                # [s*b, h, d] -> [s, b, h, d]
                out_ = out_.view(-1, batch_size, *out_.shape[-2:])

        if fp8 and use_fused_attention:
            if fp8_recipe.float8_current_scaling():
                out_f16 = out_
                if is_output_fp8:
                    out_fp8 = O_quantizer(out_)
            if fp8_recipe.delayed():
                out_fp8 = Float8Tensor.make_like(out_fp8, data=out_, dtype=fwd_nominal_dtype)
                if not is_output_fp8:
                    out_f16 = out_fp8.dequantize(dtype=fwd_nominal_dtype)
        else:
            out_f16 = out_

        out_ret = out_fp8 if is_output_fp8 else out_f16

        ctx.fp8 = fp8 and is_bwd_fp8
        fp8_tensors = (None, None, None, None)
        f16_tensors = (None, None, None, None)
        if ctx.fp8:
            if fp8_recipe.float8_current_scaling() and _dpa_fp8_cs_o_in_f16:
                fp8_tensors = (q_part, k_part, v_part, None)
                f16_tensors = (None, None, None, out_part)
            else:
                fp8_tensors = (q_part, k_part, v_part, out_part)
        elif fp8:
            q_part, k_part, v_part = combine_and_dequantize(qkv_layout, q_part, k_part, v_part)
            f16_tensors = (q_part, k_part, v_part, out_part)
        else:
            f16_tensors = (q_part, k_part, v_part, out_part)

        tensors_to_save, tensor_objects = prepare_for_saving(
            *fp8_tensors,
            *f16_tensors,
            cu_seqlens_q,
            cu_seqlens_kv,
            cu_seqlens_q_padded,
            cu_seqlens_kv_padded,
            *aux_ctx_tensors,
        )
        ctx.save_for_backward(*tensors_to_save)
        ctx.tensor_objects = tensor_objects
        ctx.out_shape = out_ret.shape

        ctx.batch_size = batch_size
        ctx.cp_group = cp_group
        ctx.cp_stream = cp_stream
        ctx.dropout_p = dropout_p
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_kv = max_seqlen_kv
        ctx.softmax_scale = softmax_scale
        ctx.qkv_format = qkv_format
        ctx.attn_mask_type = attn_mask_type
        ctx.attn_bias_type = attn_bias_type
        ctx.deterministic = deterministic
        ctx.window_size = window_size
        ctx.use_fused_attention = use_fused_attention
        ctx.fp8_meta = fp8_meta
        ctx.is_input_fp8 = is_input_fp8
        ctx.is_output_fp8 = is_output_fp8
        ctx.fwd_nominal_dtype = fwd_nominal_dtype
        ctx.fp8_recipe = fp8_recipe
        ctx.use_flash_attn_3 = use_flash_attn_3
        ctx.softmax_type = softmax_type

        ctx.dQKV_quantizer = dQKV_quantizer
        ctx.dO_quantizer = dO_quantizer
        ctx.dP_quantizer = dP_quantizer
        ctx.QKV_quantizer = QKV_quantizer
        ctx.O_quantizer = O_quantizer
        ctx.S_quantizer = S_quantizer
        if ctx.fp8:
            ctx.QKV_quantizer = QKV_quantizer.copy()
            ctx.QKV_quantizer.scale = QKV_quantizer.scale.clone()
            ctx.O_quantizer = O_quantizer.copy()
            ctx.O_quantizer.scale = O_quantizer.scale.clone()
            ctx.S_quantizer = S_quantizer.copy()
            ctx.S_quantizer.scale = S_quantizer.scale.clone()
        nvtx_range_pop("transformer_engine.AttnFuncWithCPAndQKVOA2A.forward")
        if return_max_logit:
            return out_ret, max_logit
        return out_ret

    @staticmethod
    def backward(ctx, dout, *_args):
        # pylint: disable=missing-function-docstring
        nvtx_range_push("transformer_engine.AttnFuncWithCPAndQKVOA2A.backward")
        cp_size = get_distributed_world_size(ctx.cp_group)

        (
            q_fp8,
            k_fp8,
            v_fp8,
            out_fp8,
            q,
            k,
            v,
            out,
            cu_seqlens_q,
            cu_seqlens_kv,
            cu_seqlens_q_padded,
            cu_seqlens_kv_padded,
            *aux_ctx_tensors,
        ) = restore_from_saved(ctx.tensor_objects, ctx.saved_tensors)

        qkv_format = ctx.qkv_format
        qkv_layout = qkv_format + "_" + qkv_format + "_" + qkv_format
        causal = "causal" in ctx.attn_mask_type

        if qkv_format in ["bshd", "sbhd"]:
            seq_dim = qkv_format.index("s")
        else:  # qkv_format == "thd"
            seq_dim = qkv_format.index("t")

        bwd_nominal_dtype = ctx.fwd_nominal_dtype
        dqkv_te_dtype = None
        fused_attn_backend = None
        dout_fp8 = dout
        if ctx.fp8:
            if ctx.use_fused_attention:
                fused_attn_backend = FusedAttnBackend["FP8"]
                if not isinstance(dout, QuantizedTensorStorage):
                    dout = ctx.dO_quantizer(dout)
                    dout_fp8 = dout
                dqkv_te_dtype = dout._fp8_dtype
                dout = dout._data
                fp8_meta_kwargs = {}
                fp8_meta_kwargs["s_quantizer"] = ctx.S_quantizer
                fp8_meta_kwargs["dp_quantizer"] = ctx.dP_quantizer
                fp8_meta_kwargs["dqkv_quantizer"] = ctx.dQKV_quantizer

            else:
                assert False, "FP8 is only supported with Fused Attention!"
        else:
            if isinstance(dout, QuantizedTensorStorage):
                dout = dout.dequantize(dtype=bwd_nominal_dtype)
            if ctx.use_fused_attention:
                fp8_meta_kwargs = {}
                dqkv_te_dtype = TE_DType[dout.dtype]
                fused_attn_backend = FusedAttnBackend["F16_arbitrary_seqlen"]

        if not ctx.use_fused_attention:
            if qkv_format in ["bshd", "sbhd"]:
                out = out.view(ctx.batch_size, -1, *out.shape[-2:])
                dout = dout.view(ctx.batch_size, -1, *dout.shape[-2:])
        else:
            dout = dout.view(*ctx.out_shape)

        chunk_ids_for_a2a = get_seq_chunk_ids_for_reordering_before_attn(cp_size, dout.device)
        dout = flash_attn_a2a_communicate(
            dout,
            chunk_ids_for_a2a,
            seq_dim,
            cp_size,
            ctx.cp_group,
            ctx.cp_stream,
            before_attn=True,
            qkv_format=qkv_format,
            cu_seqlens_padded=cu_seqlens_q_padded,
        )

        flash_attn_bwd = None
        if not ctx.use_fused_attention:
            fa_backward_kwargs = {"softmax_scale": ctx.softmax_scale}
            if ctx.use_flash_attn_3:
                from transformer_engine.pytorch.attention.dot_product_attention.backends import (
                    _flash_attn_bwd_v3,
                )

                flash_attn_bwd = (
                    _flash_attn_bwd_v3  # pylint: disable=possibly-used-before-assignment
                )
                fa_backward_kwargs["window_size"] = ctx.window_size
                fa_backward_kwargs["deterministic"] = ctx.deterministic
            else:
                if qkv_format == "thd":
                    from transformer_engine.pytorch.attention.dot_product_attention.backends import (
                        _flash_attn_varlen_bwd,
                    )

                    flash_attn_bwd = _flash_attn_varlen_bwd
                else:
                    from transformer_engine.pytorch.attention.dot_product_attention.backends import (
                        _flash_attn_bwd,
                    )

                    flash_attn_bwd = _flash_attn_bwd
                fa_backward_kwargs["dropout_p"] = ctx.dropout_p
                if fa_utils.v2_3_plus and not fa_utils.v2_7_0_plus:
                    fa_backward_kwargs["window_size"] = ctx.window_size
                elif fa_utils.v2_7_0_plus:
                    fa_backward_kwargs["window_size_left"] = ctx.window_size[0]
                    fa_backward_kwargs["window_size_right"] = ctx.window_size[1]
                if fa_utils.v2_4_plus:
                    fa_backward_kwargs["alibi_slopes"] = None
                if fa_utils.v2_4_1_plus:
                    fa_backward_kwargs["deterministic"] = ctx.deterministic
                if fa_utils.v2_6_0_plus:
                    fa_backward_kwargs["softcap"] = 0.0

        dq_fp8, dk_fp8, dv_fp8 = None, None, None
        if ctx.use_fused_attention:
            q_part, k_part, v_part, out_part, dout_part = q, k, v, out, dout
            if ctx.fp8:
                q_part, k_part, v_part, out_part = q_fp8, k_fp8, v_fp8, out_fp8
                if ctx.fp8_recipe.float8_current_scaling() and _dpa_fp8_cs_o_in_f16:
                    out_part = out
                dout_part = Float8Tensor.make_like(dout_fp8, data=dout, dtype=bwd_nominal_dtype)
            dq, dk, dv, *rest = fused_attn_bwd(
                ctx.max_seqlen_q,
                ctx.max_seqlen_kv,
                cu_seqlens_q,
                cu_seqlens_kv,
                q_part,
                k_part,
                v_part,
                out_part,
                dout_part,
                bwd_nominal_dtype,
                dqkv_te_dtype,
                aux_ctx_tensors,
                fused_attn_backend,
                cu_seqlens_q_padded=cu_seqlens_q_padded,
                cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                attn_scale=ctx.softmax_scale,
                dropout=ctx.dropout_p,
                qkv_layout=qkv_layout,
                attn_mask_type=ctx.attn_mask_type,
                attn_bias_type=ctx.attn_bias_type,
                window_size=ctx.window_size,
                deterministic=ctx.deterministic,
                cuda_graph=is_graph_capturing(),
                **fp8_meta_kwargs,
                softmax_type=ctx.softmax_type,
            )
            if isinstance(dq, Float8Tensor):
                dq_fp8, dk_fp8, dv_fp8 = dq, dk, dv
                dq, dk, dv = [x._data for x in [dq, dk, dv]]
        else:
            softmax_lse, rng_state = aux_ctx_tensors
            dq, dk, dv = [torch.empty_like(x) for x in [q, k, v]]
            fa_backward_args_thd = get_fa_args(
                False,
                ctx.use_flash_attn_3,
                qkv_format,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=ctx.max_seqlen_q,
                max_seqlen_kv=ctx.max_seqlen_kv,
                dq=dq,
                dk=dk,
                dv=dv,
            )
            if not ctx.use_flash_attn_3:
                fa_backward_kwargs["rng_state"] = rng_state
            flash_attn_bwd(
                dout,
                q,
                k,
                v,
                out,
                softmax_lse,
                *fa_backward_args_thd,
                causal=causal,
                **fa_backward_kwargs,
            )

        chunk_ids_for_a2a = get_seq_chunk_ids_for_reordering_after_attn(cp_size, dq.device)
        dq, dk, dv = flash_attn_a2a_communicate(
            [dq, dk, dv],
            chunk_ids_for_a2a,
            seq_dim,
            cp_size,
            ctx.cp_group,
            ctx.cp_stream,
            before_attn=False,
            qkv_format=qkv_format,
            cu_seqlens_padded=cu_seqlens_q_padded,
        )

        if qkv_format == "bshd":
            dq, dk, dv = [x.view(ctx.batch_size, -1, *x.shape[-2:]) for x in [dq, dk, dv]]
        elif qkv_format == "sbhd":
            dq, dk, dv = [x.view(-1, ctx.batch_size, *x.shape[-2:]) for x in [dq, dk, dv]]

        d_bias = None
        d_softmax_offset = None
        if ctx.use_fused_attention:
            if ctx.attn_bias_type not in ["no_bias", "alibi"]:
                d_bias = rest[0]
            if ctx.softmax_type != "vanilla":
                d_softmax_offset = rest[1]
                d_softmax_offset = flash_attn_a2a_communicate_softmax_offset(
                    d_softmax_offset, 1, cp_size, ctx.cp_group, ctx.cp_stream, False
                )

        if ctx.fp8:
            if ctx.fp8_recipe.float8_current_scaling() and ctx.is_input_fp8:
                dq, dk, dv = combine_and_quantize(qkv_layout, dq, dk, dv, ctx.dQKV_quantizer)
            if ctx.fp8_recipe.delayed():
                dq, dk, dv = [
                    Float8Tensor.make_like(x, data=y, dtype=bwd_nominal_dtype)
                    for x, y in zip([dq_fp8, dk_fp8, dv_fp8], [dq, dk, dv])
                ]
                if not ctx.is_input_fp8:
                    dq, dk, dv = combine_and_dequantize(
                        qkv_layout,
                        dq,
                        dk,
                        dv,
                        src_nominal_dtype=bwd_nominal_dtype,
                    )

        nvtx_range_pop("transformer_engine.AttnFuncWithCPAndQKVOA2A.backward")

        return (
            None,
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            d_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            d_softmax_offset,
            None,
        )


def attn_forward_func_with_cp(
    is_training,
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_kv,
    max_seqlen_q,
    max_seqlen_kv,
    cu_seqlens_q_padded,
    cu_seqlens_kv_padded,
    dropout_p,
    cp_group,
    cp_global_ranks,
    cp_stream,
    cp_comm_type,
    softmax_scale=None,
    qkv_format="bshd",
    attn_mask_type="causal",
    attn_bias_type="no_bias",
    attn_bias=None,
    deterministic=False,
    use_fused_attention=False,
    window_size=None,
    fp8=False,
    fp8_meta=None,
    quantizers=None,
    pad_between_seqs=False,
    use_flash_attn_3=False,
    softmax_type="vanilla",
    softmax_offset=None,
    fp8_output=False,
    layer_number=1,
    return_max_logit=False,
) -> torch.Tensor:
    """
    Attention implementation with context parallelism (CP). CP partitions tensors along the sequence
    dimension, and by reducing the memory and computational pressure on each GPU, it enables long-context
    LLMs in a distributed fashion. Transformer Engine's PyTorch CP implementation currently utilizes
    the DualChunkSwap strategy to ensure load balancing across CP ranks. It is applied to all `attn_mask_type`s
    and all `qkv_format`s, and it requires sequence lengths to be, or are padded to be, divisible by
    (cp_size * 2). It also requires tokens to be re-ordered before entering this function.

    For qkv_format = {'bshd', 'sbhd'}, the token re-ordering is illustrated as below, for an example
    use case of s = 12, attn_mask_type = 'causal', and cp_size = 2. seq_pos indicates each token's position
    in their corresponding sequence.

                   GPU0        |      GPU1                            GPU0        |      GPU1
    seq_pos | 0  1  2  3  4  5 | 6  7  8  9 10 11      seq_pos | 0  1  2  9 10 11 | 3  4  5  6  7  8
    ---------------------------|-----------------      ---------------------------|------------------
          0 | 1, 0, 0, 0, 0, 0,| 0, 0, 0, 0, 0, 0            0 | 1, 0, 0, 0, 0, 0,| 0, 0, 0, 0, 0, 0,
    G     1 | 1, 1, 0, 0, 0, 0,| 0, 0, 0, 0, 0, 0      G     1 | 1, 1, 0, 0, 0, 0,| 0, 0, 0, 0, 0, 0,
    P     2 | 1, 1, 1, 0, 0, 0,| 0, 0, 0, 0, 0, 0      P     2 | 1, 1, 1, 0, 0, 0,| 0, 0, 0, 0, 0, 0,
    U     3 | 1, 1, 1, 1, 0, 0,| 0, 0, 0, 0, 0, 0      U     9 | 1, 1, 1, 1, 0, 0,| 1, 1, 1, 1, 1, 1,
    0     4 | 1, 1, 1, 1, 1, 0,| 0, 0, 0, 0, 0, 0  ->  0    10 | 1, 1, 1, 1, 1, 0,| 1, 1, 1, 1, 1, 1,
          5 | 1, 1, 1, 1, 1, 1,| 0, 0, 0, 0, 0, 0           11 | 1, 1, 1, 1, 1, 1,| 1, 1, 1, 1, 1, 1,
    ---------------------------|-----------------      ---------------------------|------------------
          6 | 1, 1, 1, 1, 1, 1,| 1, 0, 0, 0, 0, 0            3 | 1, 1, 1, 0, 0, 0,| 1, 0, 0, 0, 0, 0,
    G     7 | 1, 1, 1, 1, 1, 1,| 1, 1, 0, 0, 0, 0      G     4 | 1, 1, 1, 0, 0, 0,| 1, 1, 0, 0, 0, 0,
    P     8 | 1, 1, 1, 1, 1, 1,| 1, 1, 1, 0, 0, 0,     P     5 | 1, 1, 1, 0, 0, 0,| 1, 1, 1, 0, 0, 0,
    U     9 | 1, 1, 1, 1, 1, 1,| 1, 1, 1, 1, 0, 0,     U     6 | 1, 1, 1, 0, 0, 0,| 1, 1, 1, 1, 0, 0,
    1    10 | 1, 1, 1, 1, 1, 1,| 1, 1, 1, 1, 1, 0,     1     7 | 1, 1, 1, 0, 0, 0,| 1, 1, 1, 1, 1, 0,
         11 | 1, 1, 1, 1, 1, 1,| 1, 1, 1, 1, 1, 1,           8 | 1, 1, 1, 0, 0, 0,| 1, 1, 1, 1, 1, 1,

    For qkv_format = 'thd', multiple sequences may be packed into the batch, and they may be of different
    lengths. DualChunkSwap divides each sequence into (cp_size * 2) chunks and distributes 2 chunks of
    every sequence onto a CP rank. The token matrix transformation is shown as follows, for an example of
    batch_size = 2, seq_ids = [0, 1], seq_lens = [8, 4], t = 12, attn_mask_type = 'padding_causal', and
    cp_size = 2.

                   GPU0        |      GPU1                            GPU0        |      GPU1
    seq_id  | 0  0  0  0  0  0 | 0  0  1  1  1  1      seq_id  | 0  0  0  0  1  1 | 0  0  0  0  1  1
    seq_pos | 0  1  2  3  4  5 | 6  7  0  1  2  3      seq_pos | 0  1  6  7  0  3 | 2  3  4  5  1  2
    ---------------------------|-----------------      ---------------------------|------------------
        0 0 | 1, 0, 0, 0, 0, 0,| 0, 0, 0, 0, 0, 0          0 0 | 1, 0, 0, 0, 0, 0,| 0, 0, 0, 0, 0, 0,
    G   0 1 | 1, 1, 0, 0, 0, 0,| 0, 0, 0, 0, 0, 0      G   0 1 | 1, 1, 0, 0, 0, 0,| 0, 0, 0, 0, 0, 0,
    P   0 2 | 1, 1, 1, 0, 0, 0,| 0, 0, 0, 0, 0, 0      P   0 6 | 1, 1, 1, 0, 0, 0,| 1, 1, 1, 1, 0, 0,
    U   0 3 | 1, 1, 1, 1, 0, 0,| 0, 0, 0, 0, 0, 0      U   0 7 | 1, 1, 1, 1, 0, 0,| 1, 1, 1, 1, 0, 0,
    0   0 4 | 1, 1, 1, 1, 1, 0,| 0, 0, 0, 0, 0, 0  ->  0   1 0 | 0, 0, 0, 0, 2, 0,| 0, 0, 0, 0, 0, 0,
        0 5 | 1, 1, 1, 1, 1, 1,| 0, 0, 0, 0, 0, 0          1 3 | 0, 0, 0, 0, 2, 2,| 0, 0, 0, 0, 2, 2,
    ---------------------------|-----------------      ---------------------------|------------------
        0 6 | 1, 1, 1, 1, 1, 1,| 1, 0, 0, 0, 0, 0          0 2 | 1, 1, 0, 0, 0, 0,| 1, 0, 0, 0, 0, 0,
    G   0 7 | 1, 1, 1, 1, 1, 1,| 1, 1, 0, 0, 0, 0      G   0 3 | 1, 1, 0, 0, 0, 0,| 1, 1, 0, 0, 0, 0,
    P   1 0 | 0, 0, 0, 0, 0, 0,| 0, 0, 2, 0, 0, 0      P   0 4 | 1, 1, 0, 0, 0, 0,| 1, 1, 1, 0, 0, 0,
    U   1 1 | 0, 0, 0, 0, 0, 0,| 0, 0, 2, 2, 0, 0      U   0 5 | 1, 1, 0, 0, 0, 0,| 1, 1, 1, 1, 0, 0,
    1   1 2 | 0, 0, 0, 0, 0, 0,| 0, 0, 2, 2, 2, 0      1   1 1 | 0, 0, 0, 0, 2, 0,| 0, 0, 0, 0, 2, 0,
        1 3 | 0, 0, 0, 0, 0, 0,| 0, 0, 2, 2, 2, 2          1 2 | 0, 0, 0, 0, 2, 0,| 0, 0, 0, 0, 2, 2,

    When all transformer layers in a model share the same CP configuration, i.e. cp_group, cp_global_ranks,
    cp_comm_type and cp_stream, token re-ordering can take place in the dataloader, i.e. only once for
    all the layers. An example of the re-ordering code is `get_batch_on_this_cp_rank
    <https://github.com/NVIDIA/Megatron-LM/blob/d6eb60b5ea1efca47401c0be97f456fbe3a55bcd/megatron/core/utils.py#L1725>`_
    in Megatron-LM.

    """

    if cp_comm_type == "a2a+p2p":
        assert (
            isinstance(cp_group, list) and len(cp_group) == 2
        ), "CP implementation a2a+p2p requires cp_group = [a2a_cp_group, p2p_cp_group]!"
        assert (
            qkv_format != "thd"
        ), f"{qkv_format} format is not supported with hierarchical CP implementation yet!"
        assert (
            attn_bias_type == "no_bias"
        ), f"{attn_bias_type} bias type is not supported with hierarchical CP implementation yet!"
        if get_distributed_world_size(cp_group[0]) == 1:
            cp_group = cp_group[1]
            cp_comm_type = "p2p"
        elif get_distributed_world_size(cp_group[1]) == 1:
            cp_group = cp_group[0]
            cp_comm_type = "a2a"
    else:
        assert isinstance(
            cp_group, dist_group_type
        ), f"cp_group must be {dist_group_type} type for {cp_comm_type=}!"

    assert qkv_format in [
        "bshd",
        "sbhd",
        "thd",
    ], f"Context parallelism does not support {qkv_format=}!"
    assert (
        qkv_format != "sbhd" or use_fused_attention
    ), "Context parallelism does not support FlashAttention backend with qkv_format = 'sbhd'!"
    assert attn_bias is None or (use_fused_attention and "padding" not in attn_mask_type), (
        "Context parallelism only supports attention bias with FusedAttention backend and"
        " non-padding mask types!"
    )
    assert qkv_format != "thd" or (
        cu_seqlens_q_padded is not None and cu_seqlens_kv_padded is not None
    ), "cu_seqlens_padded can not be None for context parallelism and qkv_format = 'thd'!"

    sliding_window_attn = (
        window_size is not None and window_size != (-1, 0) and window_size != (-1, -1)
    )
    assert not sliding_window_attn or cp_comm_type in [
        "a2a",
        "all_gather",
    ], "Context parallelism does not support sliding window attention with {cp_comm_type=}!"

    enable_mla = k.shape[-1] != v.shape[-1]
    assert not enable_mla or cp_comm_type in [
        "p2p",
        "a2a+p2p",
    ], "Context parallelism does not support MLA with {cp_comm_type=}!"

    if fp8 and fp8_meta is not None:
        if fp8_meta["recipe"].fp8_dpa:
            assert (
                softmax_type == "vanilla"
            ), "Context parallelism does not support {softmax_type=} with FP8 attention!"
    assert (
        softmax_type == "vanilla" or use_fused_attention
    ), "Context parallelism only supports {softmax_type=} with FusedAttention backend!"
    assert (
        softmax_type == "vanilla" or cp_comm_type == "a2a"
    ), "Context parallelism only supports {softmax_type=} with cp_comm_type = 'a2a'!"
    assert (
        softmax_type == "vanilla" or qkv_format != "thd"
    ), "Context parallelism does not support {softmax_type=} with qkv_format = 'thd'!"

    args = [
        is_training,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
        dropout_p,
        softmax_scale,
        qkv_format,
        attn_mask_type,
        attn_bias_type,
        attn_bias,
        deterministic,
        use_fused_attention,
        return_max_logit,
    ]

    if cp_comm_type in ["p2p", "a2a+p2p"]:
        args += [
            fp8,
            fp8_meta,
            cp_group,
            cp_global_ranks,
            cp_stream,
            quantizers,
            pad_between_seqs,
            use_flash_attn_3,
            fp8_output,
            layer_number,
        ]
        out = AttnFuncWithCPAndKVP2P.apply(*args)
    elif cp_comm_type == "all_gather":
        args.pop(5)
        args.pop(8)
        args += [window_size, cp_group, cp_stream, use_flash_attn_3]
        out = AttnFuncWithCPAndKVAllGather.apply(*args)
    elif cp_comm_type == "a2a":
        args += [
            window_size,
            fp8,
            fp8_meta,
            cp_group,
            cp_stream,
            quantizers,
            use_flash_attn_3,
            softmax_type,
            softmax_offset,
            fp8_output,
        ]
        out = AttnFuncWithCPAndQKVOA2A.apply(*args)
    else:
        raise ValueError(f"Unsupported communication type: {cp_comm_type}!")

    return out


def pad_thd_sequences_for_cp(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    cu_seqlens: torch.Tensor,
    divisibility_factor: int,
    padding_token_id: int = 0,
    padding_label_id: int = -100,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pads sequences to be divisible by the divisibility factor.

    Args:
        input_ids: Tensor of shape (1, N) or (N,) containing concatenated sequences
        labels: Tensor of shape (1, N) or (N,) containing labels for each token
        cu_seqlens: Tensor of shape (M,) containing cumulative sequence lengths
        divisibility_factor: Each sequence length must be divisible by this factor
        padding_token_id: Token ID to use for padding (default: 0)
        padding_label_id: Label ID to use for padding (default: -100)

    Returns:
        Tuple of:
        - input_ids_padded: Padded input_ids tensor
        - labels_padded: Padded labels tensor
        - cu_seqlens_padded: Cumulative sequence lengths accounting for padding
    """
    # Flatten input_ids and labels if needed
    if input_ids.dim() == 2:
        input_ids = input_ids.squeeze(0)
    if labels.dim() == 2:
        labels = labels.squeeze(0)

    # Compute the sequence lengths from cu_seqlens
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]

    # List: amount of padding needed for each sequence (make length a multiple of divisibility_factor)
    padding_amounts = [
        ((l.item() + divisibility_factor - 1) // divisibility_factor) * divisibility_factor
        - l.item()
        for l in seqlens
    ]

    # Extract sequences and labels for each batch item
    batch_sequences = [
        input_ids[start.item() : end.item()] for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:])
    ]
    batch_labels = [
        labels[start.item() : end.item()] for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:])
    ]

    # Pad sequences and labels to required length
    input_ids_padded = torch.cat(
        [
            (
                torch.cat([seq, torch.full((pad,), padding_token_id, dtype=seq.dtype)])
                if pad > 0
                else seq
            )
            for seq, pad in zip(batch_sequences, padding_amounts)
        ]
    )
    labels_padded = torch.cat(
        [
            (
                torch.cat([seq, torch.full((pad,), padding_label_id, dtype=seq.dtype)])
                if pad > 0
                else seq
            )
            for seq, pad in zip(batch_labels, padding_amounts)
        ]
    )

    # Compute cumulative padded sequence lengths, starting from 0
    padded_lengths = seqlens + torch.tensor(padding_amounts, dtype=seqlens.dtype)
    cu_seqlens_padded = torch.cumsum(
        torch.cat([torch.tensor([0], dtype=cu_seqlens.dtype), padded_lengths]), dim=0
    )

    return input_ids_padded, labels_padded, cu_seqlens_padded


def generate_positional_ids_for_cp(
    cu_seqlens: torch.Tensor,
    divisibility_factor: int,
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    """Generate positional IDs for sequences padded to be divisible by divisibility_factor.

    Args:
        cu_seqlens: Tensor of shape (M,) containing cumulative sequence lengths
        divisibility_factor: Each sequence length must be divisible by this factor
        dtype: Data type for the generated positional IDs (default: torch.long)

    Returns:
        Generated positional_ids tensor where each sequence starts from 0 and continues through padding
    """
    # Compute the sequence lengths from cu_seqlens
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]

    # List: amount of padding needed for each sequence
    padding_amounts = [
        ((l.item() + divisibility_factor - 1) // divisibility_factor) * divisibility_factor
        - l.item()
        for l in seqlens
    ]

    # Generate positional IDs for each padded sequence (each starts from 0)
    padded_lengths = seqlens + torch.tensor(padding_amounts, dtype=seqlens.dtype)
    positional_ids = torch.cat(
        [torch.arange(0, int(length), dtype=dtype) for length in padded_lengths]
    )

    return positional_ids


def get_batch_on_this_cp_rank(
    cu_seqlens_padded: torch.Tensor,
    input_ids_padded: torch.Tensor,
    labels_padded: torch.Tensor,
    position_ids_padded: torch.Tensor,
    cp_group: torch.distributed.ProcessGroup = None,
    qvk_format: str = "thd",
):
    """Slice batch input along sequence dimension into multiple chunks for THD format.

    This function is inteded for use in self attention. It will not work for cross attention because
    it does not handle the case where the sequence length of the query and key are different.

    Which are parallelized across GPUs in a context parallel group.
    This version works with variable-length sequences using cumulative sequence lengths.
    """
    if qvk_format not in ["thd", "bshd", "sbhd"]:
        raise ValueError(f"Unsupported qvk_format: {qvk_format}!")
    if qvk_format == "thd":
        # Get context parallel size and rank
        cp_size = torch.distributed.get_world_size(group=cp_group)
        if cp_size > 1:
            cp_rank = torch.distributed.get_rank(group=cp_group)

            # Calculate the chunk sizes for each sequence
            total_slices_of_any_sequence = 2 * cp_size
            slice_sizes = (
                cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]
            ) // total_slices_of_any_sequence

            # Process each tensor directly instead of using keys_to_change loop
            def process_tensor(val):
                if val is None:
                    return val
                # Determine which dimension is the sequence dimension
                # Ensure cu_seqlens_padded[-1] is a Python int, not a 0-dim tensor
                if isinstance(cu_seqlens_padded[-1], torch.Tensor):
                    seq_len_val = cu_seqlens_padded[-1].item()
                else:
                    seq_len_val = cu_seqlens_padded[-1]

                # Handle 1D tensors (like position_ids that don't have batch dimension)
                if val.ndim == 1:
                    if val.shape[0] == seq_len_val:
                        current_seq_dim = 0
                    else:
                        raise ValueError(
                            "1D tensor shape doesn't match expected sequence length. Make sure the"
                            " inputs are in THD format and padded correctly."
                        )
                elif val.ndim >= 2:
                    if val.shape[1] == seq_len_val:
                        current_seq_dim = 1
                    elif val.shape[0] == seq_len_val:
                        current_seq_dim = 0
                    else:
                        raise ValueError(
                            "Make sure the inputs are in THD format and padded correctly."
                        )
                else:
                    raise ValueError("Tensor must be at least 1D")

                # On this particular rank, for each sequence, get two slices, one from the beginning
                # and one from the end.
                cp_rank_slices = []
                for slice_size, seq_start in zip(slice_sizes, cu_seqlens_padded[:-1]):
                    # 1st segment
                    cp_rank_slices.append(
                        torch.arange(
                            seq_start + (cp_rank * slice_size),
                            seq_start + ((cp_rank + 1) * slice_size),
                            device=val.device,
                        )
                    )

                    # 2nd segment
                    cp_rank_slices.append(
                        torch.arange(
                            seq_start + ((total_slices_of_any_sequence - cp_rank - 1) * slice_size),
                            seq_start + ((total_slices_of_any_sequence - cp_rank) * slice_size),
                            device=val.device,
                        )
                    )

                return val.index_select(current_seq_dim, torch.cat(cp_rank_slices))

            # Process each tensor directly
            input_ids_padded = process_tensor(input_ids_padded)
            labels_padded = process_tensor(labels_padded)
            position_ids_padded = process_tensor(position_ids_padded)
    else:
        raise ValueError(f"Support not implemented yet for qvk_format: {qvk_format}!")

    return input_ids_padded, labels_padded, position_ids_padded
