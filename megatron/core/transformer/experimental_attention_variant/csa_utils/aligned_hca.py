# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Dispatch for cuDNN's fixed-layout HCA backward."""

from functools import lru_cache
from typing import Optional

import torch
from torch import Tensor

from megatron.core.packed_seq_params import PackedSeqParams


@lru_cache(maxsize=1)
def _get_aligned_hca_backward():
    """Load the optional API only for eligible calls."""
    try:
        from cudnn import aligned_hca_backward_wrapper
    except ImportError as exc:
        raise ImportError(
            "hca_aligned_backward requires cuDNN Frontend with "
            "aligned_hca_backward_wrapper (NVIDIA/cudnn-frontend#1198)."
        ) from exc
    return aligned_hca_backward_wrapper


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
    """Select the aligned layout; captured calls require one padded 64K sequence."""
    if (
        cp_group.size() != 16
        or packed_seq_params.qkv_format != "thd"
        or packed_seq_params.cp_partition_mode != "contiguous"
        or packed_seq_params.max_seqlen_q != 65536
        or window_size != 128
        or compress_ratio != 128
        or boundary_rows != 128
        or tuple(query.shape) != (4096, 128, 512)
        or tuple(kv.shape) != (4752, 512)
        or query.dtype != torch.bfloat16
        or kv.dtype != torch.bfloat16
        or not query.is_cuda
        or kv.device != query.device
        or torch.cuda.get_device_capability(query.device) != (10, 3)
    ):
        return None
    cu = packed_seq_params.cu_seqlens_q_padded
    if cu is None:
        cu = packed_seq_params.cu_seqlens_q
    if cu is None or cu.ndim != 1 or cu.numel() < 2 or cu.device != query.device:
        return None
    single_sequence = (cu[0] == 0) & (cu[1:] == 65536).all()
    if torch.cuda.is_current_stream_capturing():
        # Recheck device metadata on every replay, including reused input buffers.
        torch._assert_async(
            single_sequence,
            "hca_aligned_backward CUDA graphs require one padded 64K sequence; "
            "disable the flag for changing pack boundaries.",
        )
    elif not single_sequence.item():
        return None
    _get_aligned_hca_backward()
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
):
    """Normalize strides after output RoPE and invoke the optional cuDNN API."""
    return _get_aligned_hca_backward()(
        q.contiguous(),
        kv.contiguous(),
        out.contiguous(),
        dout.contiguous(),
        lse.contiguous(),
        attn_sink.contiguous(),
        cp_rank=cp_rank,
        softmax_scale=softmax_scale,
    )
