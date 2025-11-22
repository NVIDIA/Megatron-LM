# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import warnings
from collections import deque
from contextlib import nullcontext
from typing import Any
import os
import torch
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor
try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

# Packed Moe Expert Offload implementation for pipeline parallelism
DEBUG = False
DEBUG_RANK = [0]
def debug_print(message):
    """Print debug message for a specific rank when DEBUG is enabled."""
    # pylint: disable=bad-builtin
    if not DEBUG:
        return
    assert torch.distributed.is_initialized()
    if torch.distributed.get_rank() in DEBUG_RANK:
        print(f'{torch.distributed.get_rank()}: {message}')

def set_ideal_affinity_for_current_gpu():
    """Set CPU affinity for the current GPU to optimize host-device transfers."""
    import uuid

    try:
        import cuda.bindings.driver as cuda_driver
        import cuda.bindings.runtime as cuda_runtime
    except ImportError:
        try:
            import cuda.cuda as cuda_driver
            import cuda.cudart as cuda_runtime
        except ImportError:
            # print("cuda-python may not be installed, skipping GPU affinity setting")
            warnings.warn("cuda-python may not be installed, skipping GPU affinity setting")
            return
    try:
        import pynvml
    except ImportError:
        warnings.warn("pynvml is not installed, skipping GPU affinity setting")
        return

    # Get current CUDA device ID
    err, device_id = cuda_runtime.cudaGetDevice()
    assert err == cuda_runtime.cudaError_t.cudaSuccess
    # Get device UUID
    err, device_uuid = cuda_driver.cuDeviceGetUuid(device_id)
    assert err == cuda_driver.CUresult.CUDA_SUCCESS
    # Set CPU affinity based on GPU's NUMA node
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByUUID("GPU-" + str(uuid.UUID(bytes=device_uuid.bytes)))
    pynvml.nvmlDeviceSetCpuAffinity(handle)

GLOBAL_BLOCK_SIZE = 1024
@triton.jit
def _stash_copy_kernel(
    src_ptr,
    dst_ptr,
    size_ptr,
    alloc_offset_ptr,
    free_offset_ptr,
    capacity_ptr,
    overflow_ptr,
    BLOCK_SIZE: tl.constexpr,
    num_iterations: tl.constexpr,
    max_tokens: tl.constexpr,
):
    """Triton kernel to copy tensor data to stash buffer.
    
    Each block can handle multiple chunks of data (num_iterations) to limit total blocks.
    Ignores out-of-bound writes if offset + size exceeds capacity.
    
    Args:
        src_ptr: Pointer to source tensor (flattened)
        dst_ptr: Pointer to destination buffer (stash_buffer)
        size_ptr: Pointer to scalar tensor containing the size to copy
        offset_original_ptr: Pointer to GPU tensor containing original offset (read-only)
        over_capacity_ptr: Pointer to counter tensor (incremented when over capacity)
        capacity: Total capacity of the buffer
        BLOCK_SIZE: Block size for Triton kernel
        num_iterations: Number of iterations each block should handle
    """
    # Get the program ID
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    
    # Load the size value from GPU tensor
    size = tl.load(size_ptr)
    
    # Load original offset from GPU tensor (for position calculations)
    alloc_offset = tl.load(alloc_offset_ptr)
    free_offset = tl.load(free_offset_ptr)
    capacity = tl.load(capacity_ptr)
    # Only the first thread checks capacity
    # Do this BEFORE the loop so it always happens
    overflow = False
    # Check if over capacity and increment counter
    avail_space = free_offset - alloc_offset
    if avail_space < 0:
        avail_space = -avail_space
    else:
        avail_space = capacity - avail_space
    if avail_space < size or max_tokens < size:
        overflow = True
    if pid == 0 and overflow:
        tl.store(overflow_ptr, 1)
    
    #if pid == 1:
    #    tl.device_print("free_offset: ", free_offset)
    if overflow:
        return

    # Each block handles num_iterations chunks of BLOCK_SIZE elements
    # Use while loop with early exit condition in the loop test
    iteration = 0
    block_start = (pid * num_iterations + iteration) * BLOCK_SIZE
    while iteration < num_iterations and block_start < size:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        
        # Create mask for valid elements within source size
        src_mask = offsets < size
        
        # Create mask for valid destination indices (within buffer capacity)
        dst_indices = free_offset + offsets
        dst_mask = dst_indices >= capacity
        dst_indices = tl.where(dst_mask, dst_indices - capacity, dst_indices)
        
        # Load from source
        src_data = tl.load(src_ptr + offsets, mask=src_mask, other=0.0)
        
        # Store to destination (ignores out-of-bound writes)
        tl.store(dst_ptr + dst_indices, src_data, mask=src_mask)
        
        # Move to next iteration
        iteration += 1
        block_start = (pid * num_iterations + iteration) * BLOCK_SIZE

    # Check if over capacity and increment counter
    size_page_aligned = tl.cdiv(size, BLOCK_SIZE) * BLOCK_SIZE

    free_offset = free_offset + size_page_aligned 
    if free_offset > capacity:
        free_offset -= capacity
    if pid == 0:
        tl.store(free_offset_ptr, free_offset)
        
@triton.jit
def _stash_pop_kernel(
    src_ptr,
    dst_ptr,
    size_ptr,
    tensor_offset_ptr,
    alloc_offset_ptr,
    free_offset_ptr,
    capacity_ptr,
    BLOCK_SIZE: tl.constexpr,
    num_iterations: tl.constexpr,
):
    """Triton kernel to copy tensor data from stash buffer.
    
    Each block can handle multiple chunks of data (num_iterations) to limit total blocks.
    Ignores out-of-bound writes if offset + size exceeds capacity.
    
    Args:
        src_ptr: Pointer to source tensor (flattened)
        dst_ptr: Pointer to destination buffer (stash_buffer)
        size_ptr: Pointer to scalar tensor containing the size to copy
        offset_original_ptr: Pointer to GPU tensor containing original offset (read-only)
        over_capacity_ptr: Pointer to counter tensor (incremented when over capacity)
        capacity: Total capacity of the buffer
        BLOCK_SIZE: Block size for Triton kernel
        num_iterations: Number of iterations each block should handle
    """
    # Get the program ID
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    
    # Load the size value from GPU tensor
    size = tl.load(size_ptr)
    
    # Load original offset from GPU tensor (for position calculations)
    tensor_offset = tl.load(tensor_offset_ptr)
    alloc_offset = tl.load(alloc_offset_ptr)
    free_offset = tl.load(free_offset_ptr)
    capacity = tl.load(capacity_ptr)
    
    # Each block handles num_iterations chunks of BLOCK_SIZE elements
    # Use while loop with early exit condition in the loop test
    iteration = 0
    block_start = (pid * num_iterations + iteration) * BLOCK_SIZE
    while iteration < num_iterations and block_start < size:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        
        # Create mask for valid elements within source size
        dst_mask = offsets < size
        
        # Create mask for valid destination indices (within buffer capacity)
        src_indices = tensor_offset + offsets
        src_mask = src_indices >= capacity
        src_indices = tl.where(src_mask, src_indices - capacity, src_indices)
        
        # Load from source
        src_data = tl.load(src_ptr + src_indices, mask=dst_mask, other=0.0)
        
        # Store to destination (ignores out-of-bound writes)
        tl.store(dst_ptr + offsets, src_data, mask=dst_mask)
        
        # Move to next iteration
        iteration += 1
        block_start = (pid * num_iterations + iteration) * BLOCK_SIZE

    # Check if over capacity and increment counter
    size_page_aligned = tl.cdiv(size, BLOCK_SIZE) * BLOCK_SIZE
    tensor_offset = tensor_offset + size_page_aligned 
    if tensor_offset > capacity:
        tensor_offset -= capacity
    if pid == 0:
        mask = tensor_offset > alloc_offset
        tl.store(alloc_offset_ptr, tensor_offset, mask=mask)

class StashBuffer:
    """
    A class to represent a stash buffer.
    """

    def __init__(self, size, device, overflow, dtype):

        self.buffer = None
        if os.getenv('PACKED_OFFLOAD_CPU', '0') == '1':
            self.buffer = torch.empty(size, dtype=dtype, device='cpu', pin_memory=True)
        else:
            self.buffer = torch.empty(size, dtype=dtype, device=device)
        self.overflow = overflow # GPU flag
        self.device = device
        self.free_offset = torch.zeros(1, dtype=torch.int64, device=device) # start offset of free space
        self.alloc_offset = torch.zeros(1, dtype=torch.int64, device=device) # start offset of allocations
        self.capacity = torch.zeros(1, dtype=torch.int64, device=device)
        self.capacity.fill_(size)
        self.dtype = dtype
    def reset(self):
        """Reset the stash buffer."""
        #assert self.alloc_offset.item() == self.free_offset.item(), f"alloc_offset {self.alloc_offset.item()} != free_offset {self.free_offset.item()}"
        #print 
        self.free_offset.zero_()
        self.alloc_offset.zero_()

    def __repr__(self):
        return f"StashBuffer(capacity={self.capacity}, device={self.device})"


class PackedTensor:
    """
    A class to represent a packed tensor.
    """
    def __init__(self, tensor, num_tokens_tensor=None, vp_stage=None, layer_name=None, max_tokens=None):
        self._tensor = tensor
        self._original_tensor = None
        assert num_tokens_tensor is not None and isinstance(num_tokens_tensor, torch.Tensor) and num_tokens_tensor.numel() == 1, f"num_tokens_tensor {num_tokens_tensor} is not a scalar tensor"
        self.num_tokens_tensor = num_tokens_tensor.clone()
        self.vp_stage = vp_stage
        self.layer_name = layer_name
        self.max_tokens = max_tokens
        # Original tensor information
        self.original_shape = list(tensor.shape)
        self.num_elements = tensor.numel()
        self.element_size = tensor.element_size()
        self.hidden_size = self.original_shape[1]
        self.dtype = tensor.dtype if not isinstance(tensor, MXFP8Tensor) else tensor._columnwise_data.dtype
        self.device = tensor.device

        self.stash_buffer_offset = None
        
    def offload_to_stash(self, stash_buffer: StashBuffer, max_blocks=2048):
        """Offload the packed tensor."""
        #self._tensor.record_stream(torch.cuda.current_stream())
        # TODO: Call offload function to offload the tensor
        # After offload stream joins main stream, the tensor is no longer needed and can be freed
        
        #pass
    
        """Copy tensor content into stash_buffer starting at current offset using Triton kernel.
        
        Out-of-bound writes are silently ignored by the kernel.
        Increments self.over_capacity counter if capacity was exceeded.
        
        Args:
            tensor (torch.Tensor): The tensor to stash. Will be flattened before copying.
            size (torch.Tensor): GPU tensor containing the number of bytes to copy.
            max_blocks (int): Maximum number of blocks to launch. Defaults to 2048.
            
        Returns:
            offset: GPU tensor indicating the offset where the tensor was stashed.
            
        Raises:
            RuntimeError: If Triton is not available.
        """
        if not HAVE_TRITON:
            raise RuntimeError("Triton is required for PackedTensor.offload_to_stash(). Please install triton.")
        
        self._tensor = self._tensor.contiguous()
        if self.num_tokens_tensor.dim() == 0:
            self.num_tokens_tensor = self.num_tokens_tensor.reshape(1)
        num_elements_tensor = self.num_tokens_tensor.mul(self.hidden_size)
        # Flatten the tensor to get total number of elements
        flat_tensor = self._tensor.flatten() if not isinstance(self._tensor, MXFP8Tensor) else self._tensor._columnwise_data.flatten()
        
        # Determine grid size with cap on max blocks
        BLOCK_SIZE = GLOBAL_BLOCK_SIZE
        max_size = flat_tensor.numel()
        total_blocks_needed = triton.cdiv(max_size, BLOCK_SIZE)
        
        # Cap the number of blocks and calculate iterations per block
        num_blocks = min(total_blocks_needed, max_blocks)
        num_iterations = triton.cdiv(total_blocks_needed, num_blocks)

        if DEBUG:
            debug_print (f"offload_to_stash ({self.layer_name}) {self._tensor.shape}-{self.dtype} stash_buffer {stash_buffer.buffer.dtype} num_tokens {self.num_tokens_tensor.item()} num_elements {num_elements_tensor.item()} max_blocks {max_blocks} total_blocks_needed {total_blocks_needed} num_blocks {num_blocks} num_iterations {num_iterations} oveflow {stash_buffer.overflow.item()}")
        #
        grid = (num_blocks,)
        self.stash_buffer_offset = stash_buffer.free_offset.clone()
        
        # Launch Triton kernel to copy data
        # self.offload_stream.wait_stream(torch.cuda.current_stream())
        # with torch.cuda.stream(self.offload_stream):
        # TODO: make this async. Something unexpected with TE on deallocate the tensor
        _stash_copy_kernel[grid](
            flat_tensor,
            stash_buffer.buffer,
            num_elements_tensor,
            stash_buffer.alloc_offset,  # Read-only: Write boundary
            stash_buffer.free_offset,  # Read+Write: Start offset for next offload
            stash_buffer.capacity,  # Read-only: Capacity of the buffer
            stash_buffer.overflow,  # Read+Write: Over capacity flag updated by kernel
            BLOCK_SIZE=BLOCK_SIZE,
            num_iterations=num_iterations,
            max_tokens=self.max_tokens*self.hidden_size,
        )
        # save reference to original tensor to avoid deallocation before offload is complete
        self._original_tensor = self._tensor
        # set tensor to None. This will be replaced by reload_from_stash.
        self._tensor = None
        if DEBUG:
            debug_print (f"After offload_to_stash offset {self.stash_buffer_offset.item()} free_offset {stash_buffer.free_offset.item()} overflow {stash_buffer.overflow.item()} capacity {stash_buffer.capacity.item()} max_tokens {self.max_tokens}")


    def reload_from_stash(self, stash_buffer: StashBuffer, max_blocks=2048):
        """Reload the packed tensor from the stash."""
        if not HAVE_TRITON:
            raise RuntimeError("Triton is required for PackedTensor.reload_from_stash(). Please install triton.")
        if isinstance(self._original_tensor, MXFP8Tensor):
            columnwise_data = torch.empty(self.original_shape, dtype=self.dtype, device=self.device)
            self._tensor = MXFP8Tensor(
                shape=self._original_tensor.shape,
                dtype=self._original_tensor.dtype,
                fp8_dtype=self._original_tensor._fp8_dtype,
                rowwise_data=self._original_tensor._rowwise_data,
                rowwise_scale_inv=self._original_tensor._rowwise_scale_inv,
                columnwise_data=columnwise_data,
                columnwise_scale_inv=self._original_tensor._columnwise_scale_inv,
                quantizer=self._original_tensor._quantizer,
            )
            flat_tensor = self._tensor._columnwise_data.flatten()
        else:
            self._tensor = torch.empty(self.original_shape, dtype=self.dtype, device=self.device)
            flat_tensor = self._tensor.flatten()

        num_elements_tensor = self.num_tokens_tensor.mul(self.hidden_size)
                
        # Determine grid size with cap on max blocks
        BLOCK_SIZE = GLOBAL_BLOCK_SIZE
        max_size = self.num_elements
        total_blocks_needed = triton.cdiv(max_size, BLOCK_SIZE)
        
        # Cap the number of blocks and calculate iterations per block
        num_blocks = min(total_blocks_needed, max_blocks)
        num_iterations = triton.cdiv(total_blocks_needed, num_blocks)
        
        if DEBUG:
            debug_print (f"reload_from_stash {self._tensor.shape}-{self.dtype} stash_buffer {stash_buffer.buffer.dtype} num_tokens {self.num_tokens_tensor.item()} num_elements {num_elements_tensor.item()} max_blocks {max_blocks} total_blocks_needed {total_blocks_needed} num_blocks {num_blocks} num_iterations {num_iterations}")
        #
        grid = (num_blocks,)
        
        
        # Launch Triton kernel to copy data
        # self.offload_stream.wait_stream(torch.cuda.current_stream())
        # with torch.cuda.stream(self.offload_stream):

        # TODO: make this async. Something unexpected with TE on deallocate the tensor
        _stash_pop_kernel[grid](
            stash_buffer.buffer,
            flat_tensor,
            num_elements_tensor,
            self.stash_buffer_offset,  # Read-only: Start offset for reload
            stash_buffer.alloc_offset,  # Read+write: Free stash buffer for model chunk
            stash_buffer.free_offset,  # Read: Start offset for offload
            stash_buffer.capacity,  # Read-only: Capacity of the buffer
            BLOCK_SIZE=BLOCK_SIZE,
            num_iterations=num_iterations,
        )
        #torch.cuda.synchronize()
        if DEBUG:
            debug_print (f"After reload_from_stash reload_offset {self.stash_buffer_offset.item()} alloc_offset {stash_buffer.alloc_offset.item()} free_offset {stash_buffer.free_offset.item()} capacity {stash_buffer.capacity.item()}")
    def __repr__(self):
        return f"PackedTensor(original_shape={self.original_shape}, num_tokens={self.num_tokens_tensor.item()}, vp_stage={self.vp_stage})"

class PP_ScheduleFunction(torch.autograd.Function):
    """
    This function is used to update the pp schedule.
    """

    @staticmethod
    def forward(ctx, tensor, offload_manager): # after forward
        # pylint: disable=missing-function-docstring

        ctx.offload_manager = offload_manager
        ctx.vp_stage = offload_manager.current_vp_stage
        if ctx.vp_stage is None:
            ctx.vp_stage = 0
        ctx.layer_no, ctx.microbatch_no = offload_manager.update_pp_schedule(ctx.vp_stage+1)
        current_stream = torch.cuda.current_stream()
        if offload_manager._pack_stream_status == 'offloading':
            current_stream.wait_stream(offload_manager.pack_stream)
            offload_manager._pack_stream_status = 'idle'

            # Deallocate original tensor after offload is complete
            while len(offload_manager.packed_tensors_offload_in_progress) > 0:
                packed_tensor = offload_manager.packed_tensors_offload_in_progress.pop(0)
                if not DEBUG:
                    if isinstance(packed_tensor._original_tensor, MXFP8Tensor):
                        packed_tensor._original_tensor._columnwise_data = None
                    else:
                        packed_tensor._original_tensor = None

        if offload_manager.status == 'captured':
            current_schedule_layer = (ctx.vp_stage+1)*100 + ctx.layer_no*10 + ctx.microbatch_no
            next_schedule_layer = ctx.offload_manager._pp_schedule[ctx.offload_manager.current_schedule_index+1]
            if current_schedule_layer != -next_schedule_layer:
                # Start offload for current layer
                ctx.offload_manager.offload_packed_tensors(current_schedule_layer)
                if next_schedule_layer < 0:
                    # reload for next backward layer
                    ctx.offload_manager.reload_packed_tensors(-next_schedule_layer)
            else:
                ctx.offload_manager.remove_packed_tensor_from_offload()

        ctx.offload_manager.current_schedule_index += 1
        # return the identical tensor
        return tensor

    @staticmethod
    def backward(ctx, *grad_output): # before backward
        # pylint: disable=missing-function-docstring
        #debug_print(f"PP_ScheduleFunction vp_stage {ctx.vp_stage} before backward")
        if ctx.vp_stage is not None:
            ctx.offload_manager.update_pp_schedule(-(ctx.vp_stage+1), -ctx.layer_no, -ctx.microbatch_no)
        ctx.offload_manager.current_schedule_index += 1
        current_stream = torch.cuda.current_stream()
        if ctx.offload_manager._unpack_stream_status == 'reloading':
            current_stream.wait_stream(ctx.offload_manager.unpack_stream)
            ctx.offload_manager._unpack_stream_status = 'idle'

        if ctx.offload_manager.status == 'captured' and ctx.offload_manager.current_schedule_index < len(ctx.offload_manager._pp_schedule):
            next_schedule_layer = ctx.offload_manager._pp_schedule[ctx.offload_manager.current_schedule_index]
            if next_schedule_layer < 0:
                ctx.offload_manager.reload_packed_tensors(-next_schedule_layer)
        
        return grad_output + (None, None)

class PackedOffloadManager:
    """
    Singleton manager for coordinating activation offloading across pipeline stages.
    Manages chunk handlers, synchronizes GPU-GPU transfers,
    and handles virtual pipeline parallelism.
    """

    OFFLOAD_MGR = None

    @classmethod
    def get_instance(cls):
        """Get the singleton instance of PipelineOffloadManager."""
        if cls.OFFLOAD_MGR is None:
            cls.OFFLOAD_MGR = PackedOffloadManager()
        return cls.OFFLOAD_MGR

    def __init__(self):
        """Initialize the manager with queues and dedicated CUDA streams."""
        # allocate streams and events for synchronization
        self._pack_stream = torch.cuda.Stream()
        self._unpack_stream = torch.cuda.Stream()
        self._pack_stream_status = 'idle' # idle, offloading
        self._unpack_stream_status = 'idle' # idle, reloading
        self.packed_tensors_to_offload = []
        self.packed_tensors_offload_in_progress = []
        self.packed_tensors_to_reload = {}

        self.iteration = 0
        self._current_layer_name = None
        self.vp_size = None
        self.current_vp_stage = None
        self._last_layer = False
        self.status = 'begin' # begin, capture, captured
        self._pp_schedule = None # If element is +ve, it denotes forward pass of vp stage, if -ve, it denotes backward pass of vp stage
        self.current_layer = None
        self.current_microbatch = None
        self.current_schedule_index = None
               
        self.page_size = GLOBAL_BLOCK_SIZE
        self.max_pages_per_vp_stage = None
        self.temp_pages_per_vp_stage = None
        self.num_tokens_tensor = None
        self.max_num_tokens = None
        self.stash_buffers = None
        self.overflow = None
        self.device = None

    @property
    def pack_stream(self):
        """Get the pack stream."""
        return self._pack_stream

    @property
    def unpack_stream(self):
        """Get the unpack stream."""
        return self._unpack_stream

    def set_current_layer_name(self, name):
        """Set the current layer name."""
        self._current_layer_name = name

    def add_packed_tensor_to_offload(self, packed_tensor):
        """Add a packed tensor to the offload list."""
        if self.status == 'captured':
            self.packed_tensors_to_offload.append(packed_tensor)
        else:
            pass

    def remove_packed_tensor_from_offload(self):
        """Remove all packed tensors from the offload list."""
        if self.status == 'captured':
            while len(self.packed_tensors_to_offload) > 0:
                packed_tensor = self.packed_tensors_to_offload.pop(0)
            assert len(self.packed_tensors_to_offload) == 0, f"packed_tensors_to_offload is not empty {self.packed_tensors_to_offload}"
        else:
            pass

    def offload_packed_tensors(self, pp_schedule_layer):
        """Offload the packed tensors."""
        current_stream = torch.cuda.current_stream()
        self.pack_stream.wait_stream(current_stream)

        with torch.cuda.stream(self.pack_stream):
            if self.status == 'captured':
                self._pack_stream_status = 'offloading'
                #assert self.packed_tensors_to_reload
                #for packed_tensor in self.packed_tensors_to_offload:
                #    packed_tensor.offload_to_stash(self.stash_buffers[packed_tensor.vp_stage])
                debug_print(f"offload_packed_tensors {len(self.packed_tensors_to_offload)}")
                if pp_schedule_layer not in self.packed_tensors_to_reload:
                    self.packed_tensors_to_reload[pp_schedule_layer] = []
                assert len(self.packed_tensors_to_reload[pp_schedule_layer]) == 0, f"packed_tensors_to_reload {pp_schedule_layer} is not empty {self.packed_tensors_to_reload[pp_schedule_layer]}"
                
                while len(self.packed_tensors_to_offload) > 0:
                    packed_tensor = self.packed_tensors_to_offload.pop(0)
                    packed_tensor.offload_to_stash(self.stash_buffers[packed_tensor.vp_stage][packed_tensor.dtype])
                    self.packed_tensors_to_reload[pp_schedule_layer].append(packed_tensor)
                    self.packed_tensors_offload_in_progress.append(packed_tensor)
            else:
                pass
        assert len(self.packed_tensors_to_offload) == 0, f"packed_tensors_to_offload is not empty {self.packed_tensors_to_offload}"
        
    def reload_packed_tensors(self, pp_schedule_layer):
        """Reload the packed tensors."""
        current_stream = torch.cuda.current_stream()
        self.unpack_stream.wait_stream(current_stream)

        with torch.cuda.stream(self.unpack_stream):
            if self.status == 'captured':
                self._unpack_stream_status = 'reloading'
                count = 0
                for item in self.packed_tensors_to_reload:
                    if len(self.packed_tensors_to_reload[item]) > 0:
                        count += 1
                
                debug_print(f"reload_packed_tensors {count}")
                while len(self.packed_tensors_to_reload[pp_schedule_layer]) > 0:
                    packed_tensor = self.packed_tensors_to_reload[pp_schedule_layer].pop(0)
                    packed_tensor.reload_from_stash(self.stash_buffers[packed_tensor.vp_stage][packed_tensor.dtype])
            else:
                pass
            assert len(self.packed_tensors_to_reload[pp_schedule_layer]) == 0, f"packed_tensors_to_reload {pp_schedule_layer} is not empty {self.packed_tensors_to_reload[pp_schedule_layer]}"

    
    def allocate_offload_pages(self, stash_buffer_size_factor=1.10):
        """Allocate offload pages for each vp stage."""
        self.stash_buffers = []
        self.overflow = torch.zeros(1, dtype=torch.int64, device=self.device)
        for vp_stage in range(self.vp_size):
            self.stash_buffers.append({})
            for dtype in self.max_pages_per_vp_stage[vp_stage]:
                self.max_pages_per_vp_stage[vp_stage][dtype] = int(self.max_pages_per_vp_stage[vp_stage][dtype] * stash_buffer_size_factor)
                self.stash_buffers[vp_stage][dtype] = StashBuffer(self.max_pages_per_vp_stage[vp_stage][dtype]*GLOBAL_BLOCK_SIZE, self.device, self.overflow, dtype)
                if torch.distributed.get_rank() == 0:
                    print(f'allocated stash buffer {vp_stage} {dtype} {self.stash_buffers[vp_stage][dtype]}')

    def update_pp_schedule(self, vp_stage, layer_no=None, microbatch_no=None):
        """Update the pp schedule."""
        if self._pp_schedule is None:
            self._pp_schedule = []
            # current layer and microbatch for each vp stage for forward pass
            self.current_layer = [1 for _ in range(self.vp_size)]
            self.current_microbatch = [1 for _ in range(self.vp_size)]

        assert self.vp_size is not None
        if layer_no is None:
            # forward pass
            layer_no = self.current_layer[vp_stage-1]
            self.current_layer[vp_stage-1] += 1
            microbatch_no = self.current_microbatch[vp_stage-1]
            if self._last_layer:
                self.current_layer[vp_stage-1] = 1
                self.current_microbatch[vp_stage-1] += 1

        if self.status == 'capture':
            self._pp_schedule.append(vp_stage*100 + layer_no*10 + microbatch_no)
            num_tokens = self.num_tokens_tensor.item()

        #debug_print(f"------{self.current_schedule_index} len PP_Schedule {len(self._pp_schedule)}")
        #debug_print(f"      {self.status} {self.current_schedule_index} {self._pp_schedule[self.current_schedule_index]} {vp_stage*100 + layer_no*10 + microbatch_no}")
        assert self._pp_schedule[self.current_schedule_index] == vp_stage*100 + layer_no*10 + microbatch_no, f"schedule {self._pp_schedule[self.current_schedule_index]} != {vp_stage*100 + layer_no*10 + microbatch_no}"
        
        
        return layer_no, microbatch_no
        #self._pp_schedule.append(vp_size)
        #self._pp_schedule.append(vp_stage)

    def on_save_for_backward(self, tensor: torch.Tensor) -> Any:
        """
        Hook called when autograd saves a tensor for backward pass.
        Returns a tag to identify the tensor later.
        """


        if self.max_num_tokens is None or tensor.size(0) != self.max_num_tokens:
            return tensor.detach()
        if isinstance(tensor, MXFP8Tensor):
            debug_print(f'on_save_for_backward MXFP8Tensor ({self._current_layer_name}) ndim {tensor.ndim} shape {tensor.shape} {tensor.dtype} rowwise {tensor._rowwise_data is not None} columnwise {(tensor._columnwise_data.shape, tensor._columnwise_data.dtype) if tensor._columnwise_data is not None else None}-scale_inv {tensor._columnwise_scale_inv.shape} {tensor._columnwise_scale_inv.dtype}')
            assert tensor._rowwise_data is None, f"rowwise_data is not None; Only columnwise data is supported for packed offloading"

        #if tensor.size(1) in [7168, 4096, 1] and DEBUG:
        #    return tensor.detach()
        if self.status == 'capture':

            self.num_tokens = self.num_tokens_tensor.item()
            num_elements = tensor.numel() * self.num_tokens // self.max_num_tokens
            num_pages = (num_elements + self.page_size - 1) // self.page_size

            dtype = tensor.dtype if not isinstance(tensor, MXFP8Tensor) else tensor._columnwise_data.dtype
            if dtype not in self.temp_pages_per_vp_stage[self.current_vp_stage]:
                self.temp_pages_per_vp_stage[self.current_vp_stage][dtype] = 0
                self.max_pages_per_vp_stage[self.current_vp_stage][dtype] = 0
            self.temp_pages_per_vp_stage[self.current_vp_stage][dtype] += num_pages
            self.max_pages_per_vp_stage[self.current_vp_stage][dtype] = max(self.max_pages_per_vp_stage[self.current_vp_stage][dtype], self.temp_pages_per_vp_stage[self.current_vp_stage][dtype])

            # Since capture stage does not use CUDA graph, we can truncate the saved tensor to actual num_tokens
            # Truncate the tensor to the actual number of tokens
            new_size = (self.num_tokens, *tensor.shape[1:])

            if isinstance(tensor, MXFP8Tensor):
                tensor_truncated = torch.empty(new_size, dtype=tensor._columnwise_data.dtype, device=tensor.device)
                tensor_truncated.copy_(tensor._columnwise_data[:self.num_tokens, ...])
                tensor._columnwise_data = tensor_truncated
            else:
                tensor_truncated = torch.empty(new_size, dtype=tensor.dtype, device=tensor.device)
                tensor_truncated.copy_(tensor[:self.num_tokens, ...])
                tensor = tensor_truncated

        packed_tensor = PackedTensor(tensor, num_tokens_tensor=self.num_tokens_tensor, vp_stage=self.current_vp_stage, layer_name=self._current_layer_name, max_tokens=self.max_num_tokens)
        if self.status == 'captured':
            self.add_packed_tensor_to_offload(packed_tensor)
        return packed_tensor
        
    def on_get_saved_tensor(self, saved_state: Any) -> torch.Tensor:
        """
        Hook called when autograd retrieves a saved tensor during backward pass.
        Returns the actual tensor (potentially reloading from CPU).
        """
        if isinstance(saved_state, PackedTensor):
            if self.status == 'capture':
                num_tokens = saved_state.num_tokens_tensor.item()
                num_elements = saved_state.num_elements * num_tokens // self.max_num_tokens
                num_pages = (num_elements + self.page_size - 1) // self.page_size
                self.temp_pages_per_vp_stage[saved_state.vp_stage][saved_state.dtype] -= num_pages

                # Pad the tensor to the max number of tokens
                npad = self.max_num_tokens - num_tokens
                pad = ()
                for _ in range(saved_state._tensor.ndim-1):
                    pad = pad + (0, 0)
                pad = pad + (0, npad)
                if isinstance(saved_state._tensor, MXFP8Tensor):
                    saved_state._tensor._columnwise_data = torch.nn.functional.pad(saved_state._tensor._columnwise_data, pad)
                else:
                    saved_state._tensor = torch.nn.functional.pad(saved_state._tensor, pad)

            if not DEBUG:
                assert saved_state._tensor is not None, f"saved_state._tensor is None {saved_state._tensor}"
            if saved_state._tensor is not None:
                if self.status == 'captured' and DEBUG:
                    #debug_print(f"on_get_saved_tensor {saved_state._original_tensor.shape} {saved_state.num_tokens_tensor.item()}")
                    original_tensor = saved_state._original_tensor if not isinstance(saved_state._original_tensor, MXFP8Tensor) else saved_state._original_tensor._columnwise_data
                    if original_tensor is not None:
                        original_flat = original_tensor.flatten() if not isinstance(original_tensor, MXFP8Tensor) else original_tensor._columnwise_data.flatten()
                        tensor_flat = saved_state._tensor.flatten() if not isinstance(saved_state._tensor, MXFP8Tensor) else saved_state._tensor._columnwise_data.flatten()
                        num_elements = saved_state.num_tokens_tensor.item() * saved_state.hidden_size
                        original_flat_sub = original_flat[:num_elements]
                        tensor_flat_sub = tensor_flat[:num_elements]
                        equal = torch.equal(original_flat_sub, tensor_flat_sub)
                        num_not_equal = (original_flat_sub != tensor_flat_sub).sum()
                        idx_not_equal = (original_flat_sub != tensor_flat_sub).nonzero()
                        debug_print(f"on_get_saved_tensor original: {saved_state._original_tensor.shape} tensor: {saved_state._tensor.shape} equal tensors {equal} num_not_equal {num_not_equal}/{num_elements} idx_not_equal {idx_not_equal} original_tensor {original_flat_sub[idx_not_equal]} tensor {tensor_flat_sub[idx_not_equal]}")
                        #debug_print(f"on_get_saved_tensor equal tensors {torch.equal(saved_state._original_tensor, saved_state._tensor)} original_tensor {original_flat[-100:]} tensor {tensor_flat[-100:]}")
                return saved_state._tensor
            else:
                return saved_state._original_tensor
        
        return saved_state
    
def packed_moe_expert_offloading_group_start(tensor, name=None):
    """Mark the start of a layer group and prepare for offload/reload."""
    rank = torch.distributed.get_rank()
    return tensor

def get_packed_moe_expert_offloading_context(name=None, max_num_tokens=None, num_tokens_tensor=None):
    """Get the fine-grained offload context"""
    #debug_print(f'get_packed_moe_expert_offloading_context name {name}')
    offload_manager = PackedOffloadManager.get_instance()
    offload_manager.max_num_tokens = max_num_tokens
    assert num_tokens_tensor is not None and isinstance(num_tokens_tensor, torch.Tensor)

    offload_manager.num_tokens_tensor = num_tokens_tensor
    offload_manager.set_current_layer_name(name) if name is not None else None
    pack_unpack_context = torch.autograd.graph.saved_tensors_hooks(offload_manager.on_save_for_backward, offload_manager.on_get_saved_tensor)
    return pack_unpack_context

def packed_moe_expert_offloading_group_commit(tensor, name=None):
    """Mark the end of a layer group and prepare for offload/reload."""
    rank = torch.distributed.get_rank()
    #debug_print(f'{rank}: packed_moe_expert_offloading_group_commit tensor {tensor.shape}-{tensor.dtype} name {name}')
    offload_manager = PackedOffloadManager.get_instance()
    offload_manager.device = tensor.device
    
    return PP_ScheduleFunction.apply(tensor, offload_manager)

def packed_moe_expert_offloading_init_chunk_handler(vp_size, vp_stage):
    """Initialize the chunk handler, called at the start of a microbatch forward pass."""
    #debug_print(f'packed_moe_expert_offloading_init_chunk_handler vp_size {vp_size} vp_stage {vp_stage}')
    offload_manager = PackedOffloadManager.get_instance()
    offload_manager.current_vp_stage = vp_stage if vp_stage is not None else 0
    if vp_size is not None:
        offload_manager.vp_size = vp_size
    else:
        offload_manager.vp_size = 1
    if offload_manager.max_pages_per_vp_stage is None:
        offload_manager.max_pages_per_vp_stage = [{} for _ in range(offload_manager.vp_size)]
        offload_manager.temp_pages_per_vp_stage = [{} for _ in range(offload_manager.vp_size)]

def packed_moe_expert_offloading_set_last_layer(is_last_layer=False):
    """Set the last layer flag."""
    #PipelineOffloadManager.get_instance().set_last_layer(is_last_layer)
    #debug_print(f'packed_moe_expert_offloading_set_last_layer is_last_layer {is_last_layer}')
    offload_manager = PackedOffloadManager.get_instance()
    offload_manager._last_layer = is_last_layer

def packed_moe_expert_offloading_reset(enabled=True):
    """Reset the chunk handler, called at the start of a training iteration."""
    offload_manager = PackedOffloadManager.get_instance()
    offload_manager.iteration += 1
    # current layer and microbatch for each vp stage for forward pass
    offload_manager.current_schedule_index = 0
    if os.getenv('MEM_PROFILE', '0') == '1':
        if offload_manager.iteration == 1 and torch.distributed.get_rank() == 0:
            torch.cuda.memory._record_memory_history()
            print(f'packed_moe_expert_offloading_reset record_memory_history')
        if offload_manager.iteration == 10 and torch.distributed.get_rank() == 0:
            torch.cuda.memory._dump_snapshot("packed_offloading_cg.pkl")
            torch.cuda.memory._record_memory_history(enabled=None)
            print(f'packed_moe_expert_offloading_reset dump_snapshot')

    if not enabled:
        return

    set_ideal_affinity_for_current_gpu() # Set the ideal affinity for the current GPU
    if offload_manager.status == 'begin':
        offload_manager.status = 'capture'
    elif offload_manager.status == 'capture':
        offload_manager.status = 'captured'
        stash_buffer_size_factor = float(os.getenv('STASH_BUFFER_SIZE_FACTOR', '1.10'))
        offload_manager.allocate_offload_pages(stash_buffer_size_factor=stash_buffer_size_factor)
        debug_print(f'packed_moe_expert_offloading_reset captured schedule: {offload_manager._pp_schedule}')
        debug_print(f'packed_moe_expert_offloading_reset max_pages_per_vp_stage: {offload_manager.max_pages_per_vp_stage}')
    elif offload_manager.status == 'captured':
        pass
    else:
        debug_print(f'packed_moe_expert_offloading_reset unknown status: {offload_manager.status}')

    if offload_manager.status == 'captured':
        if not torch.cuda.is_current_stream_capturing():
            overflow = offload_manager.overflow.item()
            assert overflow == 0, f"PackedOffloadManager overflow!!!"

        for vp_buffers in offload_manager.stash_buffers:
            for dtype in vp_buffers.keys():
                vp_buffers[dtype].reset()
        offload_manager.overflow.zero_()
        offload_manager.current_layer = [1 for _ in range(offload_manager.vp_size)]
        offload_manager.current_microbatch = [1 for _ in range(offload_manager.vp_size)]
        assert len(offload_manager.packed_tensors_to_offload) == 0, f"packed_tensors_to_offload is not empty {offload_manager.packed_tensors_to_offload}"
        assert len(offload_manager.packed_tensors_offload_in_progress) == 0, f"packed_tensors_offload_in_progress is not empty {offload_manager.packed_tensors_offload_in_progress}"

