import torch
import triton 
import triton.language as tl
from .asm_utils import multimem_ld_reduce_128, multimem_st_128, ld_128, st_128

from .triton_barrier import blockwise_barrier
from .triton_utils import get_flat_tid, sync_threads




@triton.jit
def multimem_all_reduce_kernel(
    multicast_ptr,
    signal_pad_ptrs,
    numel,
    BLOCK_SIZE: tl.constexpr,
    NUMEL_PER_THREAD: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    blockwise_barrier(signal_pad_ptrs, None, RANK, WORLD_SIZE, sem="relaxed")
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
        ptrs = (
            multicast_ptr.to(tl.pointer_type(tl.uint64))
            + (RANK * numel_per_rank + offsets) * 2
        )
        (x, y, z, w) = multimem_ld_reduce_128(ptrs, mask=mask)
        multimem_st_128(ptrs, x, y, z, w, mask=mask)

        block_start += tl.num_programs(axis=0) * BLOCK_SIZE

    sync_threads()
    blockwise_barrier(signal_pad_ptrs, None, RANK, WORLD_SIZE, sem="acq_rel")


def multimem_all_reduce(tensor: torch.Tensor, symm_mem_hdl) -> torch.Tensor:
    WARP_SIZE = 32
    MAX_NUM_BLOCKS = 4
    MAX_BLOCK_SIZE = 1024
    BYTES_PER_THREAD = 16
    

    assert tensor.dtype == torch.bfloat16, "Only bfloat16 is supported for now."
    numel_per_thread = BYTES_PER_THREAD // tensor.element_size()
    
    

    assert (
        tensor.numel() % numel_per_thread == 0
    ), "The number of elements must be 128-bit aligned."

    num_threads = triton.cdiv(
        tensor.numel() // numel_per_thread, symm_mem_hdl.world_size
    )
    if num_threads < MAX_BLOCK_SIZE:
        block_size = 1
        while block_size < num_threads:
            block_size *= 2
        num_warps = block_size // WARP_SIZE
        num_blocks = 1
    else:
        block_size = MAX_BLOCK_SIZE
        num_warps = MAX_BLOCK_SIZE // WARP_SIZE
        num_blocks = min(
            triton.cdiv(num_threads, MAX_BLOCK_SIZE),
            MAX_NUM_BLOCKS,
        )

    kernel = multimem_all_reduce_kernel[(num_blocks, 1, 1)](
        symm_mem_hdl.multicast_ptr,
        symm_mem_hdl.signal_pad_ptrs_dev,
        numel=tensor.numel(),
        BLOCK_SIZE=block_size,
        NUMEL_PER_THREAD=numel_per_thread,
        RANK=symm_mem_hdl.rank,
        WORLD_SIZE=symm_mem_hdl.world_size,
        num_warps=num_warps,
    )
    
    return tensor

@triton.jit
def multimem_all_gather_kernel(
    local_ptr,
    multicast_ptr,
    signal_pad_ptrs,
    numel,
    BLOCK_SIZE: tl.constexpr,
    NUMEL_PER_THREAD: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
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
        # RANK * numel_per_rank -> brings us to the start of our rank's segment
        # offsets -> brings us to the right offset within our rank's segment
        multicast_ptrs = (
            multicast_ptr.to(tl.pointer_type(tl.uint64))
            + (RANK * numel_per_rank + offsets) * 2
        )
        local_ptrs = (
            local_ptr.to(tl.pointer_type(tl.uint64))
            + offsets * 2
        )
        (x, y, z, w) = ld_128(local_ptrs, mask=mask)
        multimem_st_128(multicast_ptrs, x, y, z, w, mask=mask)

        block_start += tl.num_programs(axis=0) * BLOCK_SIZE

    sync_threads()
    blockwise_barrier(signal_pad_ptrs, None, RANK, WORLD_SIZE, sem="acq_rel")

def multimem_all_gather(output_tensor: torch.Tensor, input_tensor: torch.Tensor, symm_mem_hdl) -> torch.Tensor:
    """
    Output tensor must be a symmetric memory buffer.
    """
    WARP_SIZE = 32
    MAX_NUM_BLOCKS = 4
    MAX_BLOCK_SIZE = 1024
    BYTES_PER_THREAD = 16

    assert input_tensor.dtype == torch.bfloat16, "Only bfloat16 is supported for now."
    assert output_tensor.dtype == torch.bfloat16, "Only bfloat16 is supported for now."
    numel_per_thread = BYTES_PER_THREAD // input_tensor.element_size()

    assert (
        output_tensor.numel() % numel_per_thread == 0
    ), "The number of elements must be 128-bit aligned."

    num_threads = triton.cdiv(
        output_tensor.numel() // numel_per_thread, symm_mem_hdl.world_size
    )
    
    if num_threads < MAX_BLOCK_SIZE:
        block_size = 1
        while block_size < num_threads:
            block_size *= 2
        num_warps = block_size // WARP_SIZE
        num_blocks = 1
    else:
        block_size = MAX_BLOCK_SIZE
        num_warps = MAX_BLOCK_SIZE // WARP_SIZE
        num_blocks = min(
            triton.cdiv(num_threads, MAX_BLOCK_SIZE),
            MAX_NUM_BLOCKS,
        )

    kernel = multimem_all_gather_kernel[(num_blocks, 1, 1)](
        input_tensor.data_ptr(),
        symm_mem_hdl.multicast_ptr,
        symm_mem_hdl.signal_pad_ptrs_dev,
        numel=output_tensor.numel(),
        BLOCK_SIZE=block_size,
        NUMEL_PER_THREAD=numel_per_thread,
        RANK=symm_mem_hdl.rank,
        WORLD_SIZE=symm_mem_hdl.world_size,
        num_warps=num_warps,
    )
    
    return output_tensor

@triton.jit
def multimem_reduce_scatter_kernel(
    local_ptr,
    multicast_ptr,
    signal_pad_ptrs,
    numel,
    BLOCK_SIZE: tl.constexpr,
    NUMEL_PER_THREAD: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    # a reduce-scatter is simply a multicast load + local store operation
    # we only need a barrier at the beginning to ensure that all GPUs 
    # have arrived. 
    blockwise_barrier(signal_pad_ptrs, None, RANK, WORLD_SIZE, sem="relaxed")
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
            multicast_ptr.to(tl.pointer_type(tl.uint64))
            + (RANK * numel_per_rank + offsets) * 2
        )
        local_ptrs = (
            local_ptr.to(tl.pointer_type(tl.uint64))
            + offsets * 2
        )
        (x, y, z, w) = multimem_ld_reduce_128(multicast_ptrs, mask=mask)
        st_128(local_ptrs, x, y, z, w, mask=mask)

        block_start += tl.num_programs(axis=0) * BLOCK_SIZE


def multimem_reduce_scatter(output_tensor: torch.Tensor, input_tensor: torch.Tensor, symm_mem_hdl) -> torch.Tensor:
    """
    Input tensor must be a symmetric memory buffer.
    """
    WARP_SIZE = 32
    MAX_NUM_BLOCKS = 4
    MAX_BLOCK_SIZE = 1024
    BYTES_PER_THREAD = 16

    assert input_tensor.dtype == torch.bfloat16, "Only bfloat16 is supported for now."
    assert output_tensor.dtype == torch.bfloat16, "Only bfloat16 is supported for now."
    numel_per_thread = BYTES_PER_THREAD // output_tensor.element_size()

    assert (
        input_tensor.numel() % numel_per_thread == 0
    ), "The number of elements must be 128-bit aligned."

    num_threads = triton.cdiv(
        input_tensor.numel() // numel_per_thread, symm_mem_hdl.world_size
    )
    
    if num_threads < MAX_BLOCK_SIZE:
        block_size = 1
        while block_size < num_threads:
            block_size *= 2
        num_warps = block_size // WARP_SIZE
        num_blocks = 1
    else:
        block_size = MAX_BLOCK_SIZE
        num_warps = MAX_BLOCK_SIZE // WARP_SIZE
        num_blocks = min(
            triton.cdiv(num_threads, MAX_BLOCK_SIZE),
            MAX_NUM_BLOCKS,
        )

    kernel = multimem_reduce_scatter_kernel[(num_blocks, 1, 1)](
        output_tensor.data_ptr(),
        symm_mem_hdl.multicast_ptr,
        symm_mem_hdl.signal_pad_ptrs_dev,
        numel=input_tensor.numel(),
        BLOCK_SIZE=block_size,
        NUMEL_PER_THREAD=numel_per_thread,
        RANK=symm_mem_hdl.rank,
        WORLD_SIZE=symm_mem_hdl.world_size,
        num_warps=num_warps,
    )
    
    return output_tensor