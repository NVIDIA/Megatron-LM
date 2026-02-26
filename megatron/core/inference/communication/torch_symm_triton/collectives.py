# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

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
from .utils import get_flat_tid, are_tensors_nvls_eligible, sync_threads

# ── Triton kernels ─────────────────────────────────────────────────────────

@triton.jit
def _ag_phase(local_ptr, multicast_ptr, byte_offset, numel, BLOCK_SIZE, NUMEL_PER_THREAD, RANK, WORLD_SIZE):
    """
    Core all-gather phase: load from local memory, multicast-store to symmetric buffer.
    This is the building block for both single-tensor and fused multi-tensor all-gathers.

    Each thread handles 128-bit (NUMEL_PER_THREAD elements) at a time.
    byte_offset locates the tensor within the multicast buffer.

    NOTE: When numel is not divisible by (NUMEL_PER_THREAD * WORLD_SIZE), the kernel
    rounds up via cdiv and may read/write up to 15 bytes past the logical tensor end.
    This is safe because PyTorch's CUDA caching allocator guarantees a minimum block
    size of 512 bytes (kMinBlockSize in CUDACachingAllocator.cpp), so small tensors
    always have sufficient backing memory.
    """
    pid = tl.program_id(axis=0)
    tid = get_flat_tid()

    numel_128 = numel // NUMEL_PER_THREAD
    numel_per_rank = tl.cdiv(numel_128, WORLD_SIZE)
    block_start = pid * BLOCK_SIZE

    while block_start < numel_per_rank:
        offsets = block_start + tid
        mask = offsets < numel_per_rank

        # byte_offset // 8 -> converts byte offset to uint64 offset
        # RANK * numel_per_rank -> start of our rank's segment
        # * 2 -> each 128-bit pack is 2 uint64s
        multicast_ptrs = (
            multicast_ptr.to(tl.pointer_type(tl.uint64))
            + byte_offset // 8
            + (RANK * numel_per_rank + offsets) * 2
        )
        local_ptrs = local_ptr.to(tl.pointer_type(tl.uint64)) + offsets * 2
        (x, y, z, w) = ld_128(local_ptrs, mask=mask, multicast_op=False)
        st_128(multicast_ptrs, x, y, z, w, mask=mask, multicast_op=True)

        block_start += tl.num_programs(axis=0) * BLOCK_SIZE


@triton.jit
def _multimem_all_gather_kernel(
    local_ptr,
    multicast_ptr,
    signal_pad_ptrs,
    numel,
    byte_offset,
    BLOCK_SIZE: tl.constexpr,
    NUMEL_PER_THREAD: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    """Single-tensor multicast all-gather kernel."""
    _ag_phase(local_ptr, multicast_ptr, byte_offset, numel,
              BLOCK_SIZE, NUMEL_PER_THREAD, RANK, WORLD_SIZE)
    sync_threads()
    symm_mem_sync(signal_pad_ptrs, None, RANK, WORLD_SIZE,
                  hasPreviousMemAccess=True, hasSubsequentMemAccess=True)


@triton.jit
def _multimem_all_gather_3_kernel(
    local_ptr_0, local_ptr_1, local_ptr_2,
    multicast_ptr,
    signal_pad_ptrs,
    numel_0, byte_offset_0,
    numel_1, byte_offset_1,
    numel_2, byte_offset_2,
    BLOCK_SIZE: tl.constexpr,
    NUMEL_PER_THREAD: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    """
    Fused 3-tensor multicast all-gather. Processes three tensors in sequence
    then synchronizes once, eliminating 2 kernel launches and 2 barriers
    compared to three separate multimem_all_gather calls.
    """
    _ag_phase(local_ptr_0, multicast_ptr, byte_offset_0, numel_0,
              BLOCK_SIZE, NUMEL_PER_THREAD, RANK, WORLD_SIZE)
    _ag_phase(local_ptr_1, multicast_ptr, byte_offset_1, numel_1,
              BLOCK_SIZE, NUMEL_PER_THREAD, RANK, WORLD_SIZE)
    _ag_phase(local_ptr_2, multicast_ptr, byte_offset_2, numel_2,
              BLOCK_SIZE, NUMEL_PER_THREAD, RANK, WORLD_SIZE)
    sync_threads()
    symm_mem_sync(signal_pad_ptrs, None, RANK, WORLD_SIZE,
                  hasPreviousMemAccess=True, hasSubsequentMemAccess=True)

@triton.jit
def _multimem_reduce_scatter_kernel(
    local_ptr,
    multicast_ptr,
    signal_pad_ptrs,
    numel,
    BLOCK_SIZE: tl.constexpr,
    NUMEL_PER_THREAD: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    """
    Triton kernel to perform multicast reduce-scatter over nvlink using multimem instructions.
    """
    symm_mem_sync(
        signal_pad_ptrs,
        None,
        RANK,
        WORLD_SIZE,
        hasPreviousMemAccess=False,
        hasSubsequentMemAccess=False,
    )
    sync_threads()

    pid = tl.program_id(axis=0)
    tid = get_flat_tid()

    # From this point on, we pretend each element is 128-bit
    numel = numel // NUMEL_PER_THREAD
    numel_per_rank = tl.cdiv(numel, WORLD_SIZE)
    block_start = pid * BLOCK_SIZE

    while block_start < numel_per_rank:
        offsets = block_start + tid
        mask = offsets < numel_per_rank

        # Each pointer points to a 128-bit bit pack
        multicast_ptrs = (
            multicast_ptr.to(tl.pointer_type(tl.uint64)) + (RANK * numel_per_rank + offsets) * 2
        )
        local_ptrs = local_ptr.to(tl.pointer_type(tl.uint64)) + offsets * 2
        (x, y, z, w) = ld_128(multicast_ptrs, mask=mask, multicast_op=True)
        st_128(local_ptrs, x, y, z, w, mask=mask, multicast_op=False)

        block_start += tl.num_programs(axis=0) * BLOCK_SIZE

# ── Python wrappers ─────────────────────────────────────────────────────────

_DEFAULT_KERNEL_CONFIG = {
    "max_num_blocks": 128,
    "num_warps": 32,
    "BLOCK_SIZE": 1024,
}


def _kernel_launch_config(element_size: int, max_numel: int, world_size: int, **kwargs):
    """Compute kernel launch config shared by all collective wrappers.

    Args:
        element_size: bytes per element (e.g. 2 for bf16).
        max_numel: largest tensor numel (determines grid size).
        world_size: number of ranks.

    Returns:
        (numel_per_thread, num_blocks, config) tuple.
    """
    config = {k: kwargs.get(k, v) for k, v in _DEFAULT_KERNEL_CONFIG.items()}
    numel_per_thread = 128 // (element_size * 8)
    num_threads = triton.cdiv(max_numel // numel_per_thread, world_size)
    num_blocks = min(triton.cdiv(num_threads, config["BLOCK_SIZE"]), config["max_num_blocks"])
    return numel_per_thread, num_blocks, config


def multimem_all_gather(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    symm_mem_hdl: _SymmetricMemory,
    byte_offset: int = 0,
    **kwargs,
) -> torch.Tensor:
    """
    Multicast all-gather for a single tensor.
    Output tensor must be a symmetric memory buffer.
    Input tensor can be a regular torch tensor.
    """
    assert HAVE_TRITON, "Triton is required for multimem all-gather."
    assert are_tensors_nvls_eligible(input_tensor), "Input tensor must be 16-byte divisible on Hopper+ for NVLS."
    assert output_tensor.numel() % input_tensor.numel() == 0 and \
        output_tensor.numel() // input_tensor.numel() == symm_mem_hdl.world_size, \
        "Output numel must be exactly world_size * input numel for all-gather."

    numel_per_thread, num_blocks, config = _kernel_launch_config(
        input_tensor.element_size(), output_tensor.numel(), symm_mem_hdl.world_size, **kwargs,
    )
    _multimem_all_gather_kernel[(num_blocks, 1, 1)](
        input_tensor.data_ptr(),
        symm_mem_hdl.multicast_ptr,
        symm_mem_hdl.signal_pad_ptrs_dev,
        numel=output_tensor.numel(),
        byte_offset=byte_offset,
        BLOCK_SIZE=config["BLOCK_SIZE"],
        NUMEL_PER_THREAD=numel_per_thread,
        RANK=symm_mem_hdl.rank,
        WORLD_SIZE=symm_mem_hdl.world_size,
        num_warps=config["num_warps"],
    )

    return output_tensor


def multimem_all_gather_fused(
    output_0: torch.Tensor, input_0: torch.Tensor, byte_offset_0: int,
    output_1: torch.Tensor, input_1: torch.Tensor, byte_offset_1: int,
    output_2: torch.Tensor, input_2: torch.Tensor, byte_offset_2: int,
    symm_mem_hdl: _SymmetricMemory,
    **kwargs,
) -> None:
    """
    Fused 3-tensor multicast all-gather. Equivalent to calling multimem_all_gather
    three times but with a single kernel launch and a single barrier.

    All tensors must share the same symmetric memory handle.
    """
    assert HAVE_TRITON, "Triton is required for multimem all-gather."
    assert are_tensors_nvls_eligible(input_0, input_1, input_2), \
        "All input tensors must be 16-byte divisible on Hopper+ for NVLS."
    for inp, out in [(input_0, output_0), (input_1, output_1), (input_2, output_2)]:
        assert out.numel() % inp.numel() == 0 and \
            out.numel() // inp.numel() == symm_mem_hdl.world_size, \
            "Output numel must be exactly world_size * input numel for all-gather."

    max_numel = max(output_0.numel(), output_1.numel(), output_2.numel())

    numel_per_thread, num_blocks, config = _kernel_launch_config(
        input_0.element_size(), max_numel, symm_mem_hdl.world_size, **kwargs,
    )
    _multimem_all_gather_3_kernel[(num_blocks, 1, 1)](
        input_0.data_ptr(), input_1.data_ptr(), input_2.data_ptr(),
        symm_mem_hdl.multicast_ptr,
        symm_mem_hdl.signal_pad_ptrs_dev,
        numel_0=output_0.numel(), byte_offset_0=byte_offset_0,
        numel_1=output_1.numel(), byte_offset_1=byte_offset_1,
        numel_2=output_2.numel(), byte_offset_2=byte_offset_2,
        BLOCK_SIZE=config["BLOCK_SIZE"],
        NUMEL_PER_THREAD=numel_per_thread,
        RANK=symm_mem_hdl.rank,
        WORLD_SIZE=symm_mem_hdl.world_size,
        num_warps=config["num_warps"],
    )


def multimem_reduce_scatter(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    symm_mem_hdl: _SymmetricMemory,
    **kwargs,
) -> torch.Tensor:
    """
    Multicast reduce-scatter for a single tensor.
    Input tensor must be a symmetric memory buffer.
    Output tensor can be a regular torch tensor.
    """
    assert HAVE_TRITON, "Triton is required for multimem reduce-scatter."
    assert input_tensor.dtype == torch.bfloat16, "Only bfloat16 is supported for now."
    assert output_tensor.dtype == torch.bfloat16, "Only bfloat16 is supported for now."
    assert are_tensors_nvls_eligible(output_tensor), "Output tensor must be 16-byte divisible on Hopper+ for NVLS."
    assert input_tensor.numel() % output_tensor.numel() == 0 and \
        input_tensor.numel() // output_tensor.numel() == symm_mem_hdl.world_size, \
        "Input numel must be exactly world_size * output numel for reduce-scatter."

    numel_per_thread, num_blocks, config = _kernel_launch_config(
        output_tensor.element_size(), input_tensor.numel(), symm_mem_hdl.world_size, **kwargs,
    )
    _multimem_reduce_scatter_kernel[(num_blocks, 1, 1)](
        output_tensor.data_ptr(),
        symm_mem_hdl.multicast_ptr,
        symm_mem_hdl.signal_pad_ptrs_dev,
        numel=input_tensor.numel(),
        BLOCK_SIZE=config["BLOCK_SIZE"],
        NUMEL_PER_THREAD=numel_per_thread,
        RANK=symm_mem_hdl.rank,
        WORLD_SIZE=symm_mem_hdl.world_size,
        num_warps=config["num_warps"],
    )

    return output_tensor
