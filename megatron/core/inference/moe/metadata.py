# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fused NVLS metadata update kernel for MoE expert parallelism.

Replaces the multi-kernel sequence:
    dist.all_gather_into_tensor(...)   # NCCL
    local_tokens_per_rank.sum()        # kernel
    local_tokens_per_rank[:rank].sum() # kernel
    local_tokens_per_rank.max()        # kernel
    _step_metadata.copy_(...)          # kernel

with a single Triton kernel that:
    1. Multicast-stores this rank's local_tokens to the symmetric memory buffer.
    2. Barrier (all ranks have written).
    3. Reads all ranks' counts, computes sum / prefix-sum / max.
    4. Writes the 3-element step_metadata tensor in-place.
"""

from unittest.mock import MagicMock

import torch

from megatron.core.utils import null_decorator

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()
    HAVE_TRITON = False

try:
    from torch._C._distributed_c10d import _SymmetricMemory
except ImportError:
    _SymmetricMemory = MagicMock()

from megatron.core.inference.communication.torch_symm_triton.barrier import symm_mem_sync
from megatron.core.inference.communication.torch_symm_triton.multimem_asm import st_32
from megatron.core.inference.communication.torch_symm_triton.utils import sync_threads


@triton.jit(do_not_specialize=["local_tokens"])
def _fused_metadata_kernel(
    local_tokens,
    local_buf_ptr,
    multicast_ptr,
    signal_pad_ptrs,
    step_metadata_ptr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    """Fused allgather + reduce kernel for MoE step metadata.

    Single CTA. Writes this rank's local_tokens to the symmetric buffer
    via multicast store, barriers, then reads all ranks' values from the
    local buffer and computes [valid_tokens, rank_token_offset, ep_max_tokens].

    Args:
        local_tokens: scalar int32, this rank's token count.
        local_buf_ptr: pointer to the local symmetric memory buffer (for reads).
        multicast_ptr: multicast pointer to the symmetric memory buffer (for writes).
        signal_pad_ptrs: signal pads for barrier synchronization.
        step_metadata_ptr: pointer to the 3-element int32 output tensor.
        RANK: this rank's index (constexpr).
        WORLD_SIZE: total number of ranks (constexpr).
    """

    tid = tl.program_id(0)
    if tid > 0:
        return


    # 1. Multicast-store local_tokens to buffer[RANK].
    mc_ptr = multicast_ptr.to(tl.pointer_type(tl.uint32)) + RANK
    mask = tl.full([], 1, dtype=tl.int1)
    val = tl.full([], local_tokens, dtype=tl.uint32)
    st_32(mc_ptr, val, mask, multicast_op=True)

    # 2. Barrier — wait for all ranks to have written.
    sync_threads()
    symm_mem_sync(
        signal_pad_ptrs,
        None,
        RANK,
        WORLD_SIZE,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True,
    )

    # 3. Load all ranks' values, reduce, and write metadata.
    offsets = tl.arange(0, WORLD_SIZE)
    vals = tl.load(local_buf_ptr + offsets)

    total = tl.sum(vals)
    prefix = tl.sum(tl.where(offsets < RANK, vals, tl.zeros_like(vals)))
    max_val = tl.max(vals)

    tl.store(step_metadata_ptr, total)
    tl.store(step_metadata_ptr + 1, prefix)
    tl.store(step_metadata_ptr + 2, max_val)


def fused_metadata_update(
    local_tokens: int,
    local_buf: torch.Tensor,
    symm_mem_hdl: _SymmetricMemory,
    step_metadata: torch.Tensor,
) -> None:
    """Fused NVLS allgather + reduce for MoE step metadata.

    Args:
        local_tokens: number of tokens on this rank this step.
        local_buf: the local symmetric memory buffer tensor ([WORLD_SIZE] int32).
            Used for reads after the barrier.
        symm_mem_hdl: symmetric memory handle for the metadata buffer.
            Provides the multicast pointer for writes and signal pads for barrier.
        step_metadata: [3] int32 CUDA tensor to write
            [valid_tokens, rank_token_offset, ep_max_tokens] into.
    """
    assert HAVE_TRITON, "Triton is required for fused_metadata_update."

    _fused_metadata_kernel[(1, 1, 1)](
        local_tokens,
        local_buf,
        symm_mem_hdl.multicast_ptr,
        symm_mem_hdl.signal_pad_ptrs_dev,
        step_metadata,
        RANK=symm_mem_hdl.rank,
        WORLD_SIZE=symm_mem_hdl.world_size,
        num_warps=max(1, (symm_mem_hdl.world_size + 31) // 32),
    )
    torch.cuda.synchronize() 
