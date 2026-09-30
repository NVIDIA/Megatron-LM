# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Dispatch for cuDNN's aligned HCA backward."""

from functools import lru_cache
from typing import Optional

import torch
from torch import Tensor

from megatron.core.packed_seq_params import PackedSeqParams


@lru_cache(maxsize=1)
def _get_aligned_hca_backward():
    """Load the optional API only for eligible calls."""
    try:
        from cudnn import AlignedHCABackward, aligned_hca_backward_wrapper
    except ImportError as exc:
        raise ImportError(
            "hca_aligned_backward requires cuDNN Frontend with "
            "aligned_hca_backward_wrapper (NVIDIA/cudnn-frontend#1198)."
        ) from exc
    return AlignedHCABackward, aligned_hca_backward_wrapper


def aligned_hca_cp_rank(
    query: Tensor,
    kv: Tensor,
    packed_seq_params: PackedSeqParams,
    cp_group: torch.distributed.ProcessGroup,
    *,
    window_size: int,
    compress_ratio: int,
    boundary_rows: int,
) -> Optional[int]:
    """Select a single-sequence aligned layout supported by cuDNN."""
    if query.ndim != 3:
        return None
    local_tokens, cp_size = query.shape[0], cp_group.size()
    sequence_length = local_tokens * cp_size
    compressed_rows = cp_size * (local_tokens // 128 + 1)
    if (
        cp_size not in (4, 8, 16)
        or not 8192 <= sequence_length <= 131072
        or local_tokens % 128 != 0
        or packed_seq_params.qkv_format != "thd"
        or packed_seq_params.cp_partition_mode != "contiguous"
        or packed_seq_params.max_seqlen_q != sequence_length
        or window_size != 128
        or compress_ratio != 128
        or boundary_rows != 128
        or tuple(query.shape[1:]) != (128, 512)
        or tuple(kv.shape) != (local_tokens + 128 + compressed_rows, 512)
        or query.dtype != torch.bfloat16
        or kv.dtype != torch.bfloat16
        or not query.is_cuda
        or kv.device != query.device
    ):
        return None
    cu = packed_seq_params.cu_seqlens_q_padded
    if cu is None:
        cu = packed_seq_params.cu_seqlens_q
    if cu is None or cu.ndim != 1 or cu.numel() < 2 or cu.device != query.device:
        return None
    single_sequence = (cu[0] == 0) & (cu[1:] == sequence_length).all()
    if torch.cuda.is_current_stream_capturing():
        # Recheck device metadata on every replay, including reused input buffers.
        torch._assert_async(
            single_sequence,
            "hca_aligned_backward CUDA graphs require one aligned padded sequence; "
            "disable the flag for changing pack boundaries.",
        )
    elif not single_sequence.item():
        return None
    api, _ = _get_aligned_hca_backward()
    if not api.supports_configuration(local_tokens, cp_size, query.device):
        return None
    return cp_group.rank()


def aligned_hca_backward(
    q: Tensor,
    kv: Tensor,
    out: Tensor,
    dout: Tensor,
    lse: Tensor,
    attn_sink: Tensor,
    *,
    cp_rank: int,
    softmax_scale: float,
    cp_size: int = 16,
):
    """Normalize strides after output RoPE and invoke the optional cuDNN API."""
    _, backward = _get_aligned_hca_backward()
    return backward(
        q.contiguous(),
        kv.contiguous(),
        out.contiguous(),
        dout.contiguous(),
        lse.contiguous(),
        attn_sink.contiguous(),
        cp_rank=cp_rank,
        softmax_scale=softmax_scale,
        cp_size=cp_size,
    )
