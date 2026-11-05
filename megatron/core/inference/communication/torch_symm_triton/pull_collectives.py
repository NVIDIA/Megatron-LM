# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Reduce-scatter over symmetric memory that pulls peer shards with plain NVLink loads.

Each element is summed in FP32 in rank order and rounded to BF16 once. Batch-invariant
inference uses it at TP=2, where the result is bitwise NCCL's reduce-scatter.
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
from .kernel_config import SYMM_MEM_KERNEL_CONFIG
from .utils import sync_threads


@triton.jit
def _pull_reduce_scatter_kernel(
    out_ptr,
    buffer_ptrs_dev,
    signal_pad_ptrs,
    input_byte_offset,
    numel_per_rank,
    BLOCK_SIZE: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    """Reduce-scatter that pulls this rank's shard from every peer over NVLink.

    Each element is the FP32 sum of its partials in rank order, starting from rank 0's
    value (so signed zeros survive), rounded to BF16 once.
    """
    symm_mem_sync(
        signal_pad_ptrs,
        None,
        RANK,
        WORLD_SIZE,
        hasPreviousMemAccess=False,
        hasSubsequentMemAccess=True,
    )
    sync_threads()

    buffer_ptrs = buffer_ptrs_dev.to(tl.pointer_type(tl.uint64))
    shard_start = RANK * numel_per_rank
    for block_start in range(
        tl.program_id(axis=0) * BLOCK_SIZE, numel_per_rank, tl.num_programs(axis=0) * BLOCK_SIZE
    ):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel_per_rank
        base = tl.load(buffer_ptrs).to(tl.pointer_type(tl.uint8))
        src = (base + input_byte_offset).to(tl.pointer_type(tl.bfloat16))
        acc = tl.load(src + shard_start + offsets, mask=mask, other=0.0).to(tl.float32)
        for src_rank in tl.static_range(1, WORLD_SIZE):
            base = tl.load(buffer_ptrs + src_rank).to(tl.pointer_type(tl.uint8))
            src = (base + input_byte_offset).to(tl.pointer_type(tl.bfloat16))
            acc += tl.load(src + shard_start + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offsets, acc.to(tl.bfloat16), mask=mask)


def pull_reduce_scatter(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    symm_mem_hdl: _SymmetricMemory,
    block_size: int = 2048,
    num_warps: int = 8,
) -> torch.Tensor:
    """Reduce-scatter along the first dimension with a fixed FP32 rank-order sum per element.

    ``input_tensor`` must live in the symmetric buffer of ``symm_mem_hdl``. Each rank reads
    its shard of every peer's input with plain NVLink loads and adds the partials in FP32
    in rank order, rounding to BF16 once. Unlike ``multimem_reduce_scatter``, whose
    in-switch BF16 addition does not round to nearest even, the result is the correctly
    rounded sum at two ranks, bitwise equal to NCCL's reduce-scatter.
    """
    assert HAVE_TRITON, "Triton is required for pull reduce-scatter."
    assert input_tensor.dtype == output_tensor.dtype == torch.bfloat16
    world_size = symm_mem_hdl.world_size
    assert input_tensor.numel() == world_size * output_tensor.numel()
    assert output_tensor.is_contiguous() and input_tensor.is_contiguous()
    input_byte_offset = input_tensor.data_ptr() - symm_mem_hdl.buffer_ptrs[symm_mem_hdl.rank]
    assert 0 <= input_byte_offset and input_byte_offset % 16 == 0
    numel_per_rank = output_tensor.numel()
    num_blocks = min(triton.cdiv(numel_per_rank, block_size), SYMM_MEM_KERNEL_CONFIG.max_num_blocks)
    SYMM_MEM_KERNEL_CONFIG.check_num_blocks(num_blocks)
    _pull_reduce_scatter_kernel[(num_blocks, 1, 1)](
        output_tensor,
        symm_mem_hdl.buffer_ptrs_dev,
        symm_mem_hdl.signal_pad_ptrs_dev,
        input_byte_offset,
        numel_per_rank,
        BLOCK_SIZE=block_size,
        RANK=symm_mem_hdl.rank,
        WORLD_SIZE=world_size,
        num_warps=num_warps,
    )
    return output_tensor
