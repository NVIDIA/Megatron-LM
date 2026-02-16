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
from .utils import get_flat_tid, sync_threads


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
    """
    Triton kernel to perform multicast all-gather over nvlink using multimem instructions.

    Args:
        byte_offset: Byte offset into the multicast buffer where this tensor starts.
    """
    # an all-gather is simply a multicast store operation
    # we only need a barrier at the end to ensure visibility of writes

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
        # byte_offset // 8 -> converts byte offset to uint64 offset
        # RANK * numel_per_rank -> brings us to the start of our rank's segment
        # offsets -> brings us to the right offset within our rank's segment
        # * 2 -> each 128-bit pack is 2 uint64s
        multicast_ptrs = (
            multicast_ptr.to(tl.pointer_type(tl.uint64)) + byte_offset // 8 + (RANK * numel_per_rank + offsets) * 2
        )
        local_ptrs = local_ptr.to(tl.pointer_type(tl.uint64)) + offsets * 2
        (x, y, z, w) = ld_128(local_ptrs, mask=mask, multicast_op=False)
        st_128(multicast_ptrs, x, y, z, w, mask=mask, multicast_op=True)

        block_start += tl.num_programs(axis=0) * BLOCK_SIZE

    sync_threads()
    symm_mem_sync(
        signal_pad_ptrs,
        None,
        RANK,
        WORLD_SIZE,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True,
    )


def multimem_all_gather(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    symm_mem_hdl: _SymmetricMemory,
    byte_offset: int = 0,
    **kwargs,
) -> torch.Tensor:
    """
    Calls a multicast all-gather triton kernel on the given tensor.
    Output tensor must be a symmetric memory buffer.
    Input tensor can be a regular torch tensor
    Arguments:
        output_tensor: torch.Tensor - output tensor to be all-gathered into
        input_tensor: torch.Tensor - input tensor to be all-gathered from
        symm_mem_hdl: _SymmetricMemory - handle to the symmetric memory buffer for output_tensor
        byte_offset: int - byte offset into the multicast buffer where output_tensor starts
    Returns:
        torch.Tensor - all-gathered tensor, which is output_tensor
    """
    assert HAVE_TRITON, "Triton is required for multimem all-gather."

    config = {
        "max_num_blocks": kwargs.get("max_num_blocks", 128),
        "num_warps": kwargs.get("num_warps", 32),
        "BLOCK_SIZE": kwargs.get("BLOCK_SIZE", 1024),
    }
    # assert input_tensor.dtype == torch.bfloat16, "Only bfloat16 is supported for now."
    # assert output_tensor.dtype == torch.bfloat16, "Only bfloat16 is supported for now."
    numel_per_thread = 128 // (input_tensor.element_size() * 8)

    assert (
        output_tensor.numel() % numel_per_thread == 0
    ), "The number of elements must be 128-bit aligned."

    num_threads = triton.cdiv(output_tensor.numel() // numel_per_thread, symm_mem_hdl.world_size)
    num_blocks = min(triton.cdiv(num_threads, config["BLOCK_SIZE"]), config["max_num_blocks"])

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


# ── Fused 3-tensor all-gather ───────────────────────────────────────────────
# Processes routing_map, probs, and hidden_states in a single kernel launch
# with a single barrier, eliminating 2 kernel launches + 2 barriers.


@triton.jit
def _ag_phase(local_ptr, multicast_ptr, byte_offset, numel, BLOCK_SIZE, NUMEL_PER_THREAD, RANK, WORLD_SIZE):
    """One all-gather phase: load from local memory, multicast-store to symmetric buffer."""
    pid = tl.program_id(axis=0)
    tid = get_flat_tid()

    numel_128 = numel // NUMEL_PER_THREAD
    numel_per_rank = tl.cdiv(numel_128, WORLD_SIZE)
    block_start = pid * BLOCK_SIZE

    while block_start < numel_per_rank:
        offsets = block_start + tid
        mask = offsets < numel_per_rank

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
    # Phase 1: routing_map
    _ag_phase(local_ptr_0, multicast_ptr, byte_offset_0, numel_0,
              BLOCK_SIZE, NUMEL_PER_THREAD, RANK, WORLD_SIZE)

    # Phase 2: probs
    _ag_phase(local_ptr_1, multicast_ptr, byte_offset_1, numel_1,
              BLOCK_SIZE, NUMEL_PER_THREAD, RANK, WORLD_SIZE)

    # Phase 3: hidden_states
    _ag_phase(local_ptr_2, multicast_ptr, byte_offset_2, numel_2,
              BLOCK_SIZE, NUMEL_PER_THREAD, RANK, WORLD_SIZE)

    # Single barrier for all three tensors
    sync_threads()
    symm_mem_sync(
        signal_pad_ptrs,
        None,
        RANK,
        WORLD_SIZE,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True,
    )


def multimem_all_gather_3(
    output_0: torch.Tensor, input_0: torch.Tensor, byte_offset_0: int,
    output_1: torch.Tensor, input_1: torch.Tensor, byte_offset_1: int,
    output_2: torch.Tensor, input_2: torch.Tensor, byte_offset_2: int,
    symm_mem_hdl: _SymmetricMemory,
    **kwargs,
) -> None:
    """
    Fused 3-tensor multicast all-gather. Equivalent to calling multimem_all_gather
    three times but with a single kernel launch and a single barrier.

    All tensors must share the same symmetric memory handle and be BF16.
    """
    assert HAVE_TRITON, "Triton is required for multimem all-gather."

    config = {
        "max_num_blocks": kwargs.get("max_num_blocks", 128),
        "num_warps": kwargs.get("num_warps", 32),
        "BLOCK_SIZE": kwargs.get("BLOCK_SIZE", 1024),
    }

    numel_per_thread = 128 // (input_0.element_size() * 8)

    assert output_0.numel() % numel_per_thread == 0, "Tensor 0 must be 128-bit aligned."
    assert output_1.numel() % numel_per_thread == 0, "Tensor 1 must be 128-bit aligned."
    assert output_2.numel() % numel_per_thread == 0, "Tensor 2 must be 128-bit aligned."

    # Size grid to the largest tensor
    max_numel = max(output_0.numel(), output_1.numel(), output_2.numel())
    num_threads = triton.cdiv(max_numel // numel_per_thread, symm_mem_hdl.world_size)
    num_blocks = min(triton.cdiv(num_threads, config["BLOCK_SIZE"]), config["max_num_blocks"])

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


def multimem_reduce_scatter(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    symm_mem_hdl: _SymmetricMemory,
    **kwargs,
) -> torch.Tensor:
    """
    Calls a multicast reduce-scatter triton kernel on the given tensor.
    Input tensor must be a symmetric memory buffer.
    Output tensor can be a regular torch tensor
    Arguments:
        output_tensor: torch.Tensor - output tensor to be reduce-scattered into
        input_tensor: torch.Tensor - input tensor to be reduce-scattered from
        symm_mem_hdl: _SymmetricMemory - handle to the symmetric memory buffer for input_tensor
        **kwargs: Additional keyword arguments for kernel configuration:
            max_num_blocks (int, optional): The maximum number of blocks to launch.
            num_warps (int, optional): The number of warps per block.
            BLOCK_SIZE (int, optional): The BLOCK_SIZE parameter for the kernel.
    Returns:
        torch.Tensor - reduce-scattered tensor, which is output_tensor
    """

    assert HAVE_TRITON, "Triton is required for multimem reduce-scatter."

    config = {
        "max_num_blocks": kwargs.get("max_num_blocks", 128),
        "num_warps": kwargs.get("num_warps", 32),
        "BLOCK_SIZE": kwargs.get("BLOCK_SIZE", 1024),
    }

    assert input_tensor.dtype == torch.bfloat16, "Only bfloat16 is supported for now."
    assert output_tensor.dtype == torch.bfloat16, "Only bfloat16 is supported for now."
    numel_per_thread = 128 // (output_tensor.element_size() * 8)

    assert (
        input_tensor.numel() % numel_per_thread == 0
    ), "The number of elements must be 128-bit aligned."

    num_threads = triton.cdiv(input_tensor.numel() // numel_per_thread, symm_mem_hdl.world_size)
    num_blocks = min(triton.cdiv(num_threads, config["BLOCK_SIZE"]), config["max_num_blocks"])

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
