# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Variable-count NVLS collectives (AllGatherV / ReduceScatterV).

Unlike the uniform collectives in collectives.py, each rank may contribute
a different number of tokens. The caller provides:
  - rank_token_offset: prefix sum of token counts for all lower-ranked ranks.
  - local_tokens: this rank's token count.

One CTA processes one token; the outer loop is persistent over local_tokens.
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

from .barrier import symm_mem_sync
from .multimem_asm import ld_128, st_128
from .utils import are_tensors_nvls_eligible, sync_threads


@triton.jit
def _multimem_all_gather_v_kernel(
    local_ptr,
    multicast_ptr,
    signal_pad_ptrs,
    local_tokens,
    rank_token_offset_ptr,
    byte_offset,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUMEL_PER_THREAD: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    """Variable-count multicast all-gather kernel. One CTA processes one token.

    Each rank contributes local_tokens tokens starting at rank_token_offset in
    the global output. Ranks may have different local_tokens values.

    Args:
        local_ptr: pointer to this rank's local input, shape [local_tokens, hidden_size].
        multicast_ptr: multicast pointer to the symmetric memory buffer.
        signal_pad_ptrs: signal pads for barrier synchronization.
        local_tokens: number of tokens this rank contributes.
        rank_token_offset_ptr: pointer to a scalar int32 CUDA tensor holding the index
            of the first token this rank writes in the global output (prefix sum of
            local_tokens for all lower-ranked ranks). The tensor is fixed at CUDA graph
            capture time; its value is written by the engine before each graph replay.
        byte_offset: byte offset of this tensor within the symmetric memory buffer.
        HIDDEN_SIZE: hidden dimension (constexpr).
        BLOCK_SIZE: threads per block (constexpr, >= numel_per_token).
        NUMEL_PER_THREAD: 128-bit elements per thread, i.e. 128 / (element_bits) (constexpr).
        RANK: this rank's index (constexpr).
        WORLD_SIZE: total number of ranks (constexpr).
    """
    pid = tl.program_id(axis=0)
    tid = tl.arange(0, BLOCK_SIZE)

    rank_token_offset = tl.load(rank_token_offset_ptr)

    numel_per_token = tl.cdiv(HIDDEN_SIZE, NUMEL_PER_THREAD)
    local_numel = local_tokens * numel_per_token
    thread_mask = tid < numel_per_token

    for token_offset in range(pid, local_tokens, tl.num_programs(axis=0)):
        program_offset = token_offset * numel_per_token

        for thread_offset in range(0, numel_per_token, BLOCK_SIZE):
            local_offsets = program_offset + thread_offset + tid
            mask = (local_offsets < local_numel) & thread_mask

            # This rank's tokens start at rank_token_offset in the global output.
            global_offsets = rank_token_offset * numel_per_token + local_offsets

            multicast_ptrs = (
                multicast_ptr.to(tl.pointer_type(tl.uint64))
                + byte_offset // 8
                + global_offsets * 2
            )
            local_ptrs = local_ptr.to(tl.pointer_type(tl.uint64)) + local_offsets * 2

            (x, y, z, w) = ld_128(local_ptrs, mask=mask, multicast_op=False)
            st_128(multicast_ptrs, x, y, z, w, mask=mask, multicast_op=True)

    sync_threads()
    symm_mem_sync(
        signal_pad_ptrs,
        None,
        RANK,
        WORLD_SIZE,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True,
    )


def multimem_all_gather_v(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    symm_mem_hdl: _SymmetricMemory,
    rank_token_offset: torch.Tensor,
    hidden_size: int,
    max_tokens: int,
    byte_offset: int = 0,
    **kwargs,
) -> torch.Tensor:
    """Variable-count multicast all-gather for a single tensor.

    Each EP rank may contribute a different number of tokens. rank_token_offset
    is a scalar int32 CUDA tensor whose address is fixed at CUDA graph capture
    time. The engine computes and writes its value once per step before graph
    replay — the kernel loads it at runtime.

    output_tensor must be a symmetric memory buffer sized for the total global
    tokens. input_tensor is a regular torch tensor of shape [local_tokens, hidden_size].

    Args:
        output_tensor: symmetric memory buffer, shape [global_tokens, hidden_size].
        input_tensor: this rank's local input, shape [local_tokens, hidden_size].
        symm_mem_hdl: symmetric memory handle.
        rank_token_offset: scalar int32 CUDA tensor. Holds the index of the first
            token this rank writes in the global output (prefix sum of local_tokens
            for all lower-ranked ranks). Fixed address; value set before each replay.
        hidden_size: hidden dimension of each token.
        max_tokens: maximum number of tokens any rank can contribute. Used to
            determine the grid size so that all ranks launch the same number of
            CTAs — required for symm_mem_sync to complete on all ranks.
        byte_offset: byte offset of this tensor within the symmetric memory buffer
            (for packing multiple tensors; 0 if only one tensor).

    Returns:
        output_tensor with all ranks' data written.
    """
    assert HAVE_TRITON, "Triton is required for multimem all-gather-v."
    assert are_tensors_nvls_eligible(
        input_tensor
    ), "Input tensor must be 16-byte divisible on Hopper+ for NVLS."
    assert (
        rank_token_offset.numel() == 1
        and rank_token_offset.dtype == torch.int32
        and rank_token_offset.is_cuda
    ), "rank_token_offset must be a scalar int32 CUDA tensor."

    MAX_NUM_BLOCKS = kwargs.get("max_num_blocks", 128)
    MAX_BLOCK_SIZE = 1024
    WARP_SIZE = 32

    local_tokens = input_tensor.shape[0]
    numel_per_thread = 128 // (input_tensor.element_size() * 8)
    numel_per_token = (hidden_size + numel_per_thread - 1) // numel_per_thread

    # BLOCK_SIZE must be a constexpr and >= numel_per_token; round up to next power of 2.
    block_size = 1
    while block_size < numel_per_token:
        block_size *= 2
    block_size = min(block_size, MAX_BLOCK_SIZE)
    num_warps = max(1, block_size // WARP_SIZE)

    # Grid is sized from max_tokens (same across all ranks) so that every rank
    # launches the same number of CTAs. CTAs with pid >= local_tokens skip the
    # data movement loop but still participate in symm_mem_sync.
    num_blocks = min(max_tokens, MAX_NUM_BLOCKS)

    _multimem_all_gather_v_kernel[(num_blocks, 1, 1)](
        input_tensor.data_ptr(),
        symm_mem_hdl.multicast_ptr,
        symm_mem_hdl.signal_pad_ptrs_dev,
        local_tokens=local_tokens,
        rank_token_offset_ptr=rank_token_offset,
        byte_offset=byte_offset,
        HIDDEN_SIZE=hidden_size,
        BLOCK_SIZE=block_size,
        NUMEL_PER_THREAD=numel_per_thread,
        RANK=symm_mem_hdl.rank,
        WORLD_SIZE=symm_mem_hdl.world_size,
        num_warps=num_warps,
    )

    return output_tensor
