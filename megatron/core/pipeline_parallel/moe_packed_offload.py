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
def _stash_copy_kernel_2d(
    src_ptr,
    dst_ptr,
    num_tokens_ptr,  # Number of tokens to copy
    alloc_offset_ptr,  # In tokens (read-only)
    free_offset_ptr,   # In tokens (read-only)
    capacity_ptr,      # In tokens (read-only)
    overflow_ptr,
    new_free_offset_ptr,  # Output: new free_offset value (written by kernel)
    HIDDEN_SIZE: tl.constexpr,  # Hidden dimension (compile-time constant)
    BLOCK_SIZE: tl.constexpr,   # Threads per block (for hidden dimension)
):
    """2D Triton kernel to copy tensor data to stash buffer.
    
    Grid: (num_blocks,) - fixed number of blocks
    Uses strided access pattern: block i handles tokens [i, i+num_blocks, i+2*num_blocks, ...].
    Works directly with contiguous 2D tensors [tokens, hidden_size].
    Offsets are tracked in tokens, not elements.
    """
    pid = tl.program_id(axis=0)
    num_blocks = tl.num_programs(axis=0)
    
    # Load parameters (in tokens, not elements)
    num_tokens = tl.load(num_tokens_ptr)
    alloc_offset = tl.load(alloc_offset_ptr)
    free_offset = tl.load(free_offset_ptr)
    capacity = tl.load(capacity_ptr)
    
    # All blocks check for overflow (same computation, avoids race condition)
    if free_offset >= alloc_offset:
        # No wraparound: available space is from free_offset to capacity, then 0 to alloc_offset
        avail_space = capacity - (free_offset - alloc_offset)
    else:
        # Wraparound: available space is from free_offset to alloc_offset
        avail_space = alloc_offset - free_offset
    overflow_detected = avail_space < num_tokens
    
    # Only block 0 writes the overflow flag
    if pid == 0 and overflow_detected:
        tl.store(overflow_ptr, 1)
    
    # All blocks return early if overflow detected
    if overflow_detected:
        return
    
    # Strided access: block pid handles tokens [pid, pid+num_blocks, pid+2*num_blocks, ...]
    token_idx = pid
    while token_idx < num_tokens:
        # Calculate destination token index with wraparound
        dst_token_idx = (free_offset + token_idx) % capacity
        
        # Each thread handles elements of the hidden dimension
        elements_per_thread = HIDDEN_SIZE // BLOCK_SIZE
        
        # Check if we need masking (only if HIDDEN_SIZE not divisible by BLOCK_SIZE)
        need_mask = (HIDDEN_SIZE % BLOCK_SIZE) != 0
        num_iters = elements_per_thread + (1 if need_mask else 0)
        
        # 2D indexing: base + token_idx * HIDDEN_SIZE + hidden_offsets
        src_base = src_ptr + token_idx * HIDDEN_SIZE
        dst_base = dst_ptr + dst_token_idx * HIDDEN_SIZE
        
        if need_mask:
            # Use mask for all iterations when HIDDEN_SIZE not divisible by BLOCK_SIZE
            for iter in range(num_iters):
                hidden_offsets = tl.arange(0, BLOCK_SIZE) + iter * BLOCK_SIZE
                hidden_mask = hidden_offsets < HIDDEN_SIZE
                data = tl.load(src_base + hidden_offsets, mask=hidden_mask, other=0)
                tl.store(dst_base + hidden_offsets, data, mask=hidden_mask)
        else:
            # No mask needed - HIDDEN_SIZE is multiple of BLOCK_SIZE
            for iter in range(elements_per_thread):
                hidden_offsets = tl.arange(0, BLOCK_SIZE) + iter * BLOCK_SIZE
                data = tl.load(src_base + hidden_offsets)
                tl.store(dst_base + hidden_offsets, data)
        
        # Stride to next token for this block
        token_idx += num_blocks
    
    # Update new_free_offset (only first block writes it)
    if pid == 0:
        new_free_offset = (free_offset + num_tokens) % capacity
        tl.store(new_free_offset_ptr, new_free_offset)

@triton.jit
def _stash_pop_kernel_2d(
    src_ptr,
    dst_ptr,
    num_tokens_ptr,    # Number of tokens to reload
    tensor_offset_ptr,  # In tokens - where data was stashed (read-only)
    alloc_offset_ptr,   # In tokens (read-only, not used in pop)
    free_offset_ptr,    # In tokens (write: updated directly by kernel)
    capacity_ptr,       # In tokens (read-only)
    HIDDEN_SIZE: tl.constexpr,  # Hidden dimension (compile-time constant)
    BLOCK_SIZE: tl.constexpr,   # Threads per block (for hidden dimension)
):
    """2D Triton kernel to reload tensor data from stash buffer.
    
    Grid: (num_blocks,) - fixed number of blocks
    Uses strided access pattern: block i handles tokens [i, i+num_blocks, i+2*num_blocks, ...].
    Works directly with contiguous 2D tensors [tokens, hidden_size].
    Offsets are tracked in tokens, not elements.
    Uses LIFO (stack) semantics - moves free_offset backward after popping.
    """
    pid = tl.program_id(axis=0)
    num_blocks = tl.num_programs(axis=0)
    
    # Load parameters (in tokens, not elements)
    num_tokens = tl.load(num_tokens_ptr)
    tensor_offset = tl.load(tensor_offset_ptr)  # Where data was stashed
    capacity = tl.load(capacity_ptr)
    
    # Strided access: block pid handles tokens [pid, pid+num_blocks, pid+2*num_blocks, ...]
    token_idx = pid
    while token_idx < num_tokens:
        # Calculate source token index with wraparound
        src_token_idx = (tensor_offset + token_idx) % capacity
        
        # Each thread handles elements of the hidden dimension
        elements_per_thread = HIDDEN_SIZE // BLOCK_SIZE
        
        # Check if we need masking
        need_mask = (HIDDEN_SIZE % BLOCK_SIZE) != 0
        num_iters = elements_per_thread + (1 if need_mask else 0)
        
        # 2D indexing
        src_base = src_ptr + src_token_idx * HIDDEN_SIZE
        dst_base = dst_ptr + token_idx * HIDDEN_SIZE
        
        if need_mask:
            # Use mask for all iterations when HIDDEN_SIZE not divisible by BLOCK_SIZE
            for iter in range(num_iters):
                hidden_offsets = tl.arange(0, BLOCK_SIZE) + iter * BLOCK_SIZE
                hidden_mask = hidden_offsets < HIDDEN_SIZE
                data = tl.load(src_base + hidden_offsets, mask=hidden_mask, other=0)
                tl.store(dst_base + hidden_offsets, data, mask=hidden_mask)
        else:
            # No mask needed
            for iter in range(elements_per_thread):
                hidden_offsets = tl.arange(0, BLOCK_SIZE) + iter * BLOCK_SIZE
                data = tl.load(src_base + hidden_offsets)
                tl.store(dst_base + hidden_offsets, data)
        
        # Stride to next token for this block
        token_idx += num_blocks
    
    # For LIFO (stack) behavior: move free_offset backward
    # After popping, free_offset should be at tensor_offset (freeing the space we just read)
    if pid == 0:
        # The data was stashed at tensor_offset, so after popping, free_offset moves back to tensor_offset
        tl.store(free_offset_ptr, tensor_offset)


class StashBuffer:
    """
    A class to represent a 2D stash buffer.
    
    The buffer is organized as [num_tokens, hidden_size].
    Offsets (free_offset, alloc_offset) are tracked in tokens, not elements.
    """

    def __init__(self, num_tokens, hidden_size, device, overflow, dtype):
        """
        Args:
            num_tokens: Maximum number of tokens the buffer can hold
            hidden_size: Hidden dimension size
            device: Device for the buffer
            overflow: Overflow flag tensor (shared across all buffers)
            dtype: Data type
        """
        self.buffer = None
        self.hidden_size = hidden_size
        self.num_tokens_capacity = num_tokens
        
        # Create 2D buffer [num_tokens, hidden_size]
        if os.getenv('PACKED_OFFLOAD_CPU', '0') == '1':
            self.buffer = torch.empty((num_tokens, hidden_size), dtype=dtype, device='cpu', pin_memory=True)
        else:
            self.buffer = torch.empty((num_tokens, hidden_size), dtype=dtype, device=device)
            
        self.overflow = overflow # GPU flag (shared)
        self.device = device
        
        # Offsets are in TOKENS
        self.free_offset = torch.zeros(1, dtype=torch.int64, device=device) # tail (write pointer)
        self.alloc_offset = torch.zeros(1, dtype=torch.int64, device=device) # head (read pointer)
        self.capacity = torch.zeros(1, dtype=torch.int64, device=device)
        self.capacity.fill_(num_tokens)  # Capacity in tokens
        self.dtype = dtype
        
    def reset(self):
        """Reset the stash buffer offsets."""
        self.free_offset.zero_()
        self.alloc_offset.zero_()

    def __repr__(self):
        return f"StashBuffer(capacity={self.num_tokens_capacity} tokens, hidden_size={self.hidden_size}, device={self.device}, dtype={self.dtype})"


class PagedStashBuffer:
    """
    A paged stash buffer with page-level memory management.
    
    The buffer is organized as [num_pages, page_size, hidden_size].
    Uses a free list (circular buffer) to track available pages.
    """
    
    def __init__(self, num_tokens, hidden_size, page_size, device, overflow, dtype):
        """
        Args:
            num_tokens: Maximum number of tokens the buffer can hold
            hidden_size: Hidden dimension size
            page_size: Number of tokens per page
            device: Device for the buffer
            overflow: Overflow flag tensor (shared across all buffers)
            dtype: Data type
        """
        self.hidden_size = hidden_size
        self.page_size = page_size
        self.num_pages = (num_tokens + page_size - 1) // page_size  # Ceiling division
        self.total_tokens = self.num_pages * page_size
        
        # Create 2D buffer [total_tokens, hidden_size]
        # Organized as pages: [page_0_tokens, page_1_tokens, ...]
        if os.getenv('PACKED_OFFLOAD_CPU', '0') == '1':
            self.buffer = torch.empty((self.total_tokens, hidden_size), dtype=dtype, device='cpu', pin_memory=True)
        else:
            self.buffer = torch.empty((self.total_tokens, hidden_size), dtype=dtype, device=device)
        
        self.overflow = overflow  # GPU flag (shared)
        self.device = device
        self.dtype = dtype
        
        # Free list as circular buffer: stores available page IDs
        self.free_list = torch.arange(self.num_pages, dtype=torch.int64, device=device)
        
        # Head and tail pointers for free_list circular buffer
        self.free_list_head = torch.zeros(1, dtype=torch.int64, device=device)  # Read pointer (allocation)
        self.free_list_tail = self.num_pages*torch.ones(1, dtype=torch.int64, device=device)  # Write pointer (deallocation)
        
        # Capacity of free list
        self.free_list_capacity = self.num_pages*torch.ones(1, dtype=torch.int64, device=device)
    
    def reset(self):
        """Reset the paged buffer - reinitialize free list."""
        self.free_list.copy_(torch.arange(self.num_pages, dtype=torch.int64, device=self.device))
        self.free_list_head.zero_()
        self.free_list_tail.fill_(self.num_pages)
    
    def __repr__(self):
        return f"PagedStashBuffer(num_pages={self.num_pages}, page_size={self.page_size}, hidden_size={self.hidden_size}, device={self.device}, dtype={self.dtype})"


@triton.jit
def _paged_stash_copy_kernel(
    src_ptr,
    dst_ptr,
    num_tokens_ptr,
    free_list_ptr,
    free_list_head_ptr,  # Read-only: current head position
    free_list_tail_ptr,  # Read-only: current tail position (for overflow check)
    free_list_capacity_ptr,
    page_record_ptr,  # Output: records which pages were used
    overflow_ptr,
    new_free_list_head_ptr,  # Output: new head position (written by kernel)
    PAGE_SIZE: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel to copy tokens to paged stash buffer.
    
    Allocates pages from free list (reads from head, advances head).
    Uses strided access pattern: block i handles tokens [i, i+num_blocks, i+2*num_blocks, ...].
    Grid: (num_blocks,) where blocks process tokens in a strided pattern.
    Writes new head to temporary tensor to avoid race conditions.
    """
    pid = tl.program_id(axis=0)
    num_blocks = tl.num_programs(axis=0)
    
    # Load parameters
    num_tokens = tl.load(num_tokens_ptr)
    free_list_head = tl.load(free_list_head_ptr)
    free_list_tail = tl.load(free_list_tail_ptr)
    free_list_capacity = tl.load(free_list_capacity_ptr)
    
    # Check available pages (unwrapped indices: simple subtraction, no modulo needed)
    avail_pages = free_list_tail - free_list_head
    
    # Calculate required pages
    required_pages = tl.cdiv(num_tokens, PAGE_SIZE)
    overflow_detected = avail_pages < required_pages
    
    # Only block 0 writes overflow flag
    if pid == 0 and overflow_detected:
        tl.store(overflow_ptr, 1)
    
    # All blocks return early if overflow
    if overflow_detected:
        return
    
    # Strided access: block pid handles tokens [pid, pid+num_blocks, pid+2*num_blocks, ...]
    token_idx = pid
    while token_idx < num_tokens:
        # Determine which page this token belongs to
        page_slot = token_idx // PAGE_SIZE
        token_in_page = token_idx % PAGE_SIZE
        
        # Read page ID from free list (with wraparound)
        free_list_idx = (free_list_head + page_slot) % free_list_capacity
        page_id = tl.load(free_list_ptr + free_list_idx)
        
        # First token in page: record the page ID (only if this block handles token 0 of the page)
        if token_in_page == 0:
            tl.store(page_record_ptr + page_slot, page_id)
        
        # Calculate destination address in paged buffer
        dst_token_idx = page_id * PAGE_SIZE + token_in_page
        
        # Copy token data (2D: hidden dimension)
        elements_per_thread = HIDDEN_SIZE // BLOCK_SIZE
        need_mask = (HIDDEN_SIZE % BLOCK_SIZE) != 0
        num_iters = elements_per_thread + (1 if need_mask else 0)
        
        src_base = src_ptr + token_idx * HIDDEN_SIZE
        dst_base = dst_ptr + dst_token_idx * HIDDEN_SIZE
        
        if need_mask:
            for iter in range(num_iters):
                hidden_offsets = tl.arange(0, BLOCK_SIZE) + iter * BLOCK_SIZE
                hidden_mask = hidden_offsets < HIDDEN_SIZE
                data = tl.load(src_base + hidden_offsets, mask=hidden_mask, other=0)
                tl.store(dst_base + hidden_offsets, data, mask=hidden_mask)
        else:
            for iter in range(elements_per_thread):
                hidden_offsets = tl.arange(0, BLOCK_SIZE) + iter * BLOCK_SIZE
                data = tl.load(src_base + hidden_offsets)
                tl.store(dst_base + hidden_offsets, data)
        
        # Stride to next token for this block
        token_idx += num_blocks
    
    # Calculate and store new free list head (only block 0)
    # We consumed pages, so advance head forward (unwrapped: no modulo)
    # Write to temporary tensor to avoid race conditions
    if pid == 0:
        new_head = free_list_head + required_pages
        tl.store(new_free_list_head_ptr, new_head)


@triton.jit
def _paged_stash_pop_kernel(
    src_ptr,
    dst_ptr,
    num_tokens_ptr,
    page_record_ptr,  # Input: which pages to read
    free_list_ptr,
    free_list_head_ptr,  # Read-only: current head position (not used)
    free_list_tail_ptr,  # Read-only: current tail position
    free_list_capacity_ptr,
    new_free_list_tail_ptr,  # Output: new tail position (written by kernel)
    PAGE_SIZE: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel to reload tokens from paged stash buffer.
    
    Returns pages to free list (writes to tail, advances tail).
    Uses strided access pattern: block i handles tokens [i, i+num_blocks, i+2*num_blocks, ...].
    Grid: (num_blocks,) where blocks process tokens in a strided pattern.
    Writes new tail to temporary tensor to avoid race conditions.
    """
    pid = tl.program_id(axis=0)
    num_blocks = tl.num_programs(axis=0)
    
    # Load parameters
    num_tokens = tl.load(num_tokens_ptr)
    free_list_tail = tl.load(free_list_tail_ptr)
    free_list_capacity = tl.load(free_list_capacity_ptr)
    
    # Strided access: block pid handles tokens [pid, pid+num_blocks, pid+2*num_blocks, ...]
    token_idx = pid
    while token_idx < num_tokens:
        # Determine which page this token belongs to
        page_slot = token_idx // PAGE_SIZE
        token_in_page = token_idx % PAGE_SIZE
        
        # Read page ID from page record
        page_id = tl.load(page_record_ptr + page_slot)
        
        # Calculate source address in paged buffer
        src_token_idx = page_id * PAGE_SIZE + token_in_page
        
        # Copy token data (2D: hidden dimension)
        elements_per_thread = HIDDEN_SIZE // BLOCK_SIZE
        need_mask = (HIDDEN_SIZE % BLOCK_SIZE) != 0
        num_iters = elements_per_thread + (1 if need_mask else 0)
        
        src_base = src_ptr + src_token_idx * HIDDEN_SIZE
        dst_base = dst_ptr + token_idx * HIDDEN_SIZE
        
        if need_mask:
            for iter in range(num_iters):
                hidden_offsets = tl.arange(0, BLOCK_SIZE) + iter * BLOCK_SIZE
                hidden_mask = hidden_offsets < HIDDEN_SIZE
                data = tl.load(src_base + hidden_offsets, mask=hidden_mask, other=0)
                tl.store(dst_base + hidden_offsets, data, mask=hidden_mask)
        else:
            for iter in range(elements_per_thread):
                hidden_offsets = tl.arange(0, BLOCK_SIZE) + iter * BLOCK_SIZE
                data = tl.load(src_base + hidden_offsets)
                tl.store(dst_base + hidden_offsets, data)
        
        # First token in page: release page back to free list
        if token_in_page == 0:
            # Write page ID back to free list at tail position (with wraparound)
            write_idx = (free_list_tail + page_slot) % free_list_capacity
            tl.store(free_list_ptr + write_idx, page_id)
        
        # Stride to next token for this block
        token_idx += num_blocks
    
    # Calculate and store new free list tail (only block 0)
    # We returned pages, so advance tail forward (unwrapped: no modulo)
    # Write to temporary tensor to avoid race conditions
    if pid == 0:
        required_pages = tl.cdiv(num_tokens, PAGE_SIZE)
        new_tail = free_list_tail + required_pages
        tl.store(new_free_list_tail_ptr, new_tail)


class PagedTensor:
    """
    A paged tensor that stores data in pages within a paged stash buffer.
    """
    
    def __init__(self, tensor, num_tokens_tensor=None, vp_stage=None, schedule_layer_no=None, layer_name=None, max_tokens=None, page_size=64, num_d2d_pages=0):
        """
        Args:
            tensor: The tensor to store
            num_tokens_tensor: Scalar tensor containing actual number of tokens
            vp_stage: Virtual pipeline stage
            layer_name: Name of the layer
            max_tokens: Maximum number of tokens
            page_size: Number of tokens per page
            num_d2d_pages: Number of pages to copy using native PyTorch (rest uses Triton)
        """
        self._tensor = tensor
        self._original_tensor = None
        assert num_tokens_tensor is not None and isinstance(num_tokens_tensor, torch.Tensor) and num_tokens_tensor.numel() == 1
        self.num_tokens_tensor = num_tokens_tensor.clone()
        self.vp_stage = vp_stage
        self.schedule_layer_no = schedule_layer_no
        self.layer_name = layer_name
        self.max_tokens = max_tokens
        self.page_size = page_size
        self.num_d2d_pages = num_d2d_pages
        
        # Original tensor information
        self.original_shape = list(tensor.shape)
        self.max_num_tokens = self.original_shape[0]
        self.element_size = tensor.element_size()
        self.hidden_size = self.original_shape[1]
        self.dtype = tensor.dtype if not isinstance(tensor, MXFP8Tensor) else tensor._columnwise_data.dtype
        self.device = tensor.device
        
        # Calculate number of pages needed
        self.max_num_pages = (self.max_num_tokens + page_size - 1) // page_size  # Ceiling division
        
        # Page record: stores which pages are being used for this tensor
        self.page_record = torch.zeros(self.max_num_pages, dtype=torch.int64, device=self.device)
        
        # Static tensor for D2D pages (allocate upfront if needed)
        d2d_tokens = min(self.num_d2d_pages * self.page_size, self.max_num_tokens)
        if d2d_tokens > 0:
            self.static_tensor = torch.empty((d2d_tokens, self.hidden_size), dtype=self.dtype, device=self.device)
        else:
            self.static_tensor = None

    @property
    def schedule_layer(self):
        """Get the schedule layer."""
        return self.schedule_layer_no

    def offload_to_stash(self, paged_stash_buffer: PagedStashBuffer, max_blocks=2048):
        """Offload the paged tensor to paged stash buffer.
        
        Args:
            paged_stash_buffer: The paged stash buffer to offload to
            max_blocks: Maximum number of blocks for Triton kernel
        """
        if not HAVE_TRITON:
            raise RuntimeError("Triton is required for PagedTensor.offload_to_stash(). Please install triton.")
        
        self._tensor = self._tensor.contiguous()
        if self.num_tokens_tensor.dim() == 0:
            self.num_tokens_tensor = self.num_tokens_tensor.reshape(1)
        
        # Get 2D tensor
        if isinstance(self._tensor, MXFP8Tensor):
            tensor_to_copy = self._tensor._columnwise_data
        else:
            tensor_to_copy = self._tensor
        
        # Split tensor into two parts: D2D portion and Triton portion
        # Use max_num_tokens for consistent size across iterations
        d2d_tokens = min(self.num_d2d_pages * self.page_size, self.max_num_tokens)
        triton_tokens = self.max_num_tokens - d2d_tokens
        
        # Perform both D2D copy and Triton kernel together
        # Part 1: Copy first d2d_tokens to static_tensor using native PyTorch
        if d2d_tokens > 0:
            self.static_tensor[:d2d_tokens] = tensor_to_copy[:d2d_tokens]
        # Part 2: Copy remaining tokens using Triton kernel
        if triton_tokens > 0:
            triton_tensor = tensor_to_copy[d2d_tokens:self.max_num_tokens]
            # Use actual num_tokens for the kernel (how many tokens to actually copy)
            triton_num_tokens = self.num_tokens_tensor - d2d_tokens
            
            # Determine grid size
            BLOCK_SIZE = GLOBAL_BLOCK_SIZE
            num_blocks = min(triton_tokens, max_blocks)
            grid = (num_blocks,)
            
            # Create temporary tensor for new head
            new_free_list_head = torch.empty(1, dtype=torch.int64, device=self.device)
            
            # Launch paged stash copy kernel
            _paged_stash_copy_kernel[grid](
                triton_tensor,
                paged_stash_buffer.buffer,
                triton_num_tokens,
                paged_stash_buffer.free_list,
                paged_stash_buffer.free_list_head,
                paged_stash_buffer.free_list_tail,
                paged_stash_buffer.free_list_capacity,
                self.page_record,  # Triton kernel will populate page_record
                paged_stash_buffer.overflow,
                new_free_list_head,
                PAGE_SIZE=self.page_size,
                HIDDEN_SIZE=self.hidden_size,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            
            # Update free list head
            paged_stash_buffer.free_list_head.copy_(new_free_list_head)
            
        # Save reference to original tensor
        self._original_tensor = self._tensor
        self._tensor = None
    
    def reload_from_stash(self, paged_stash_buffer: PagedStashBuffer, max_blocks=2048):
        """Reload the paged tensor from paged stash buffer.
        
        Args:
            paged_stash_buffer: The paged stash buffer to reload from
            max_blocks: Maximum number of blocks for Triton kernel
        """
        if not HAVE_TRITON:
            raise RuntimeError("Triton is required for PagedTensor.reload_from_stash(). Please install triton.")
        
        # Allocate output tensor
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
            tensor_to_reload = self._tensor._columnwise_data
        else:
            self._tensor = torch.empty(self.original_shape, dtype=self.dtype, device=self.device)
            tensor_to_reload = self._tensor
        
        # Split tensor into two parts: D2D portion and Triton portion
        # Use max_num_tokens for consistency with offload
        d2d_tokens = min(self.num_d2d_pages * self.page_size, self.max_num_tokens)
        triton_tokens = self.max_num_tokens - d2d_tokens
        
        # Perform both D2D copy and Triton kernel together
        # Part 1: Copy first d2d_tokens from static_tensor using native PyTorch
        if d2d_tokens > 0 and self.static_tensor is not None:
            tensor_to_reload[:d2d_tokens] = self.static_tensor[:d2d_tokens]
        
        # Part 2: Copy remaining tokens using Triton kernel
        if triton_tokens > 0:
            triton_tensor = tensor_to_reload[d2d_tokens:self.max_num_tokens]
            # Use actual num_tokens for the kernel (how many tokens to actually copy)
            triton_num_tokens = self.num_tokens_tensor - d2d_tokens
            
            # Determine grid size
            BLOCK_SIZE = GLOBAL_BLOCK_SIZE
            num_blocks = min(triton_tokens, max_blocks)
            grid = (num_blocks,)
            
            # Create temporary tensor for new tail
            new_free_list_tail = torch.empty(1, dtype=torch.int64, device=self.device)
            
            # Launch paged stash pop kernel
            _paged_stash_pop_kernel[grid](
                paged_stash_buffer.buffer,
                triton_tensor,
                triton_num_tokens,
                self.page_record,  # Triton kernel will read from page_record
                paged_stash_buffer.free_list,
                paged_stash_buffer.free_list_head,
                paged_stash_buffer.free_list_tail,
                paged_stash_buffer.free_list_capacity,
                new_free_list_tail,
                PAGE_SIZE=self.page_size,
                HIDDEN_SIZE=self.hidden_size,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            
            # Update free list tail
            paged_stash_buffer.free_list_tail.copy_(new_free_list_tail)


class PP_PreScheduleFunction(torch.autograd.Function):
    """
    This function is used to update the pp schedule.
    """

    @staticmethod
    def forward(ctx, tensor, offload_manager): # after forward
        # pylint: disable=missing-function-docstring
        ctx.offload_manager = offload_manager
        # Wait for offload to complete before starting the next layer
        offload_manager.wait_for_offload_to_complete()
        return tensor

    @staticmethod
    def backward(ctx, *grad_output): # before backward
        # pylint: disable=missing-function-docstring
        # Initiate reload for next layer
        if ctx.offload_manager.status == 'captured' and ctx.offload_manager.current_schedule_index < len(ctx.offload_manager._pp_schedule):
            next_schedule_layer = ctx.offload_manager._pp_schedule[ctx.offload_manager.current_schedule_index]
            if next_schedule_layer < 0:
                ctx.offload_manager.reload_packed_tensors(-next_schedule_layer)

        return grad_output + (None, None)

class PP_PostScheduleFunction(torch.autograd.Function):
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

        # Initiate offload for current layer and reload for next layer
        if offload_manager.status == 'captured':
            current_schedule_layer = offload_manager.get_schedule_layer(ctx.vp_stage+1, ctx.layer_no, ctx.microbatch_no)
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
        if ctx.vp_stage is not None:
            ctx.offload_manager.update_pp_schedule(-(ctx.vp_stage+1), -ctx.layer_no, -ctx.microbatch_no)
        ctx.offload_manager.current_schedule_index += 1
        current_stream = torch.cuda.current_stream()

        ctx.offload_manager.wait_for_offload_to_complete()
        if ctx.offload_manager._unpack_stream_status == 'reloading':
            current_stream.wait_stream(ctx.offload_manager.unpack_stream)
            ctx.offload_manager._unpack_stream_status = 'idle'
        
        return grad_output + (None, None)

class PackedOffloadManager:
    """
    Singleton manager for coordinating activation offloading across pipeline stages.
    Manages chunk handlers, synchronizes GPU-GPU transfers,
    and handles virtual pipeline parallelism
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
        self.enabled = False
        self._pack_stream = torch.cuda.Stream()
        # Currently paged/packed stashing is not stream-safe, so use the same stream for packing 
        # and unpacking
        self._unpack_stream = self._pack_stream 
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
               
        # Track max tokens needed per vp_stage, dtype, and hidden_size
        self.max_tokens_per_vp_stage = None
        self.temp_tokens_per_vp_stage = None
        # Track max tokens needed across all vp_stages grouped by dtype and hidden_size
        self.max_tokens_across_vp_stages = None
        self.temp_tokens_across_vp_stages = None

        self.num_tokens_tensor = None
        self.max_num_tokens = None
        self.stash_buffers = None
        self.overflow = None
        self.device = None
        
        # Page size for paged memory management
        self.page_size = int(os.getenv('PAGED_STASH_PAGE_SIZE', '64'))  # Default 64 tokens per page
        self.use_paged_stash = os.getenv('USE_PAGED_STASH', '0') == '1'  # Enable via env var
        
        # Number of pages to copy using native PyTorch (D2D)
        self.num_d2d_pages = int(os.getenv('NUM_D2D_PAGES', '0'))  # Default 0 (all Triton)

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

    def get_schedule_layer(self, vp_stage, layer_no, microbatch_no):
        """Get the schedule layer."""
        return vp_stage*1000000 + layer_no*1000 + microbatch_no

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
                if pp_schedule_layer not in self.packed_tensors_to_reload:
                    self.packed_tensors_to_reload[pp_schedule_layer] = []
                assert len(self.packed_tensors_to_reload[pp_schedule_layer]) == 0, f"packed_tensors_to_reload {pp_schedule_layer} is not empty {self.packed_tensors_to_reload[pp_schedule_layer]}"
                
                while len(self.packed_tensors_to_offload) > 0:
                    packed_tensor = self.packed_tensors_to_offload.pop(0)
                    stash_buffers_vp_stage = self.stash_buffers[packed_tensor.vp_stage] if not self.use_paged_stash else self.stash_buffers[0]
                    stash_buffer = stash_buffers_vp_stage[packed_tensor.dtype][packed_tensor.hidden_size]
                    packed_tensor.offload_to_stash(stash_buffer)
                    self.packed_tensors_to_reload[pp_schedule_layer].append(packed_tensor)
                    self.packed_tensors_offload_in_progress.append(packed_tensor)
            else:
                pass
        assert len(self.packed_tensors_to_offload) == 0, f"packed_tensors_to_offload is not empty {self.packed_tensors_to_offload}"

    def wait_for_offload_to_complete(self):
        """Wait for offload to complete."""
        current_stream = torch.cuda.current_stream()
        if self._pack_stream_status == 'offloading':
            current_stream.wait_stream(self.pack_stream)
            self._pack_stream_status = 'idle'

            # Deallocate original tensor after offload is complete
            while len(self.packed_tensors_offload_in_progress) > 0:
                packed_tensor = self.packed_tensors_offload_in_progress.pop(0)
                if isinstance(packed_tensor._original_tensor, MXFP8Tensor):
                    packed_tensor._original_tensor._columnwise_data = None
                else:
                    packed_tensor._original_tensor = None

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
                
                while len(self.packed_tensors_to_reload[pp_schedule_layer]) > 0:
                    packed_tensor = self.packed_tensors_to_reload[pp_schedule_layer].pop(0)
                    stash_buffers_vp_stage = self.stash_buffers[packed_tensor.vp_stage] if not self.use_paged_stash else self.stash_buffers[0]
                    stash_buffer = stash_buffers_vp_stage[packed_tensor.dtype][packed_tensor.hidden_size]
                    packed_tensor.reload_from_stash(stash_buffer)
            else:
                pass
            assert len(self.packed_tensors_to_reload[pp_schedule_layer]) == 0, f"packed_tensors_to_reload {pp_schedule_layer} is not empty {self.packed_tensors_to_reload[pp_schedule_layer]}"

    
    def allocate_offload_buffers(self, stash_buffer_size_factor=1.10):
        """Allocate offload buffers for each vp stage, organized by [vp_stage][dtype][hidden_size]."""
        self.stash_buffers = []
        self.overflow = torch.zeros(1, dtype=torch.int64, device=self.device)

        if self.use_paged_stash:
            self.stash_buffers.append({})
            for dtype, hidden_size in self.max_tokens_across_vp_stages:
                if dtype not in self.stash_buffers[0]:
                    self.stash_buffers[0][dtype] = {}
                assert hidden_size not in self.stash_buffers[0][dtype]
                num_tokens = int(self.max_tokens_across_vp_stages[dtype, hidden_size] * stash_buffer_size_factor)
                self.stash_buffers[0][dtype][hidden_size] = PagedStashBuffer(
                    num_tokens, hidden_size, self.page_size, self.device, self.overflow, dtype
                )
                if torch.distributed.get_rank() == 0:
                    print(f'allocated paged stash buffer dtype={dtype} hidden_size={hidden_size}: {self.stash_buffers[0][dtype][hidden_size]}')
            return
        # Regular stash buffers
        for vp_stage in range(self.vp_size):
            self.stash_buffers.append({})
            for dtype in self.max_tokens_per_vp_stage[vp_stage]:
                self.stash_buffers[vp_stage][dtype] = {}
                for hidden_size in self.max_tokens_per_vp_stage[vp_stage][dtype]:
                    # Calculate number of tokens we can store (with safety factor)
                    num_tokens = int(self.max_tokens_per_vp_stage[vp_stage][dtype][hidden_size] * stash_buffer_size_factor)
                    
                    # Create buffer (regular)
                    self.stash_buffers[vp_stage][dtype][hidden_size] = StashBuffer(
                        num_tokens, hidden_size, self.device, self.overflow, dtype
                    )
                    
                    if torch.distributed.get_rank() == 0:
                        buffer_type = "paged" if self.use_paged_stash else "regular"
                        print(f'allocated {buffer_type} stash buffer vp_stage={vp_stage} dtype={dtype} hidden_size={hidden_size}: {self.stash_buffers[vp_stage][dtype][hidden_size]}')

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
            self._pp_schedule.append(self.get_schedule_layer(vp_stage, layer_no, microbatch_no))
            num_tokens = self.num_tokens_tensor.item()

        assert self._pp_schedule[self.current_schedule_index] == self.get_schedule_layer(vp_stage, layer_no, microbatch_no), f"schedule {self._pp_schedule[self.current_schedule_index]} != {self.get_schedule_layer(vp_stage, layer_no, microbatch_no)}"
        
        
        return layer_no, microbatch_no
        #self._pp_schedule.append(vp_size)
        #self._pp_schedule.append(vp_stage)

    def on_save_for_backward(self, tensor: torch.Tensor) -> Any:
        """
        Hook called when autograd saves a tensor for backward pass.
        Returns a tag to identify the tensor later.
        """

        # Handle 0-dim tensors (torch.Size([])) - they have no size(0)
        if self.max_num_tokens is None or tensor.dim() == 0 or tensor.size(0) != self.max_num_tokens:
            return tensor.detach()
        if isinstance(tensor, MXFP8Tensor):
            assert tensor._rowwise_data is None, f"rowwise_data is not None; Only columnwise data is supported for packed offloading"

        if self.status == 'capture':

            self.num_tokens = self.num_tokens_tensor.item()

            dtype = tensor.dtype if not isinstance(tensor, MXFP8Tensor) else tensor._columnwise_data.dtype
            # Get hidden_size from tensor shape
            if isinstance(tensor, MXFP8Tensor):
                hidden_size = tensor._columnwise_data.shape[1] if tensor._columnwise_data.ndim > 1 else tensor._columnwise_data.numel()
            else:
                hidden_size = tensor.shape[1] if tensor.ndim > 1 else tensor.numel()
                
            if dtype not in self.temp_tokens_per_vp_stage[self.current_vp_stage]:
                self.temp_tokens_per_vp_stage[self.current_vp_stage][dtype] = {}
                self.max_tokens_per_vp_stage[self.current_vp_stage][dtype] = {}
            if hidden_size not in self.temp_tokens_per_vp_stage[self.current_vp_stage][dtype]:
                self.temp_tokens_per_vp_stage[self.current_vp_stage][dtype][hidden_size] = 0
                self.max_tokens_per_vp_stage[self.current_vp_stage][dtype][hidden_size] = 0

            self.temp_tokens_per_vp_stage[self.current_vp_stage][dtype][hidden_size] += self.num_tokens
            self.max_tokens_per_vp_stage[self.current_vp_stage][dtype][hidden_size] = max(
                self.max_tokens_per_vp_stage[self.current_vp_stage][dtype][hidden_size],
                self.temp_tokens_per_vp_stage[self.current_vp_stage][dtype][hidden_size]
            )
            if (dtype, hidden_size) not in self.temp_tokens_across_vp_stages:
                self.temp_tokens_across_vp_stages[dtype, hidden_size] = 0
                self.max_tokens_across_vp_stages[dtype, hidden_size] = 0

            self.temp_tokens_across_vp_stages[dtype, hidden_size] += self.num_tokens
            self.max_tokens_across_vp_stages[dtype, hidden_size] = max(
                self.max_tokens_across_vp_stages[dtype, hidden_size],
                self.temp_tokens_across_vp_stages[dtype, hidden_size]
            )
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

        # Create tensor (paged or regular based on configuration)
        assert self.use_paged_stash, "Paged stashing must be used."
        if self.use_paged_stash:
            packed_tensor = PagedTensor(
                tensor, 
                num_tokens_tensor=self.num_tokens_tensor, 
                vp_stage=self.current_vp_stage,
                schedule_layer_no=self._pp_schedule[self.current_schedule_index] if self._pp_schedule is not None and self.current_schedule_index < len(self._pp_schedule) else None,
                layer_name=self._current_layer_name, 
                max_tokens=self.max_num_tokens,
                page_size=self.page_size,
                num_d2d_pages=self.num_d2d_pages
            )

        if self.status == 'captured':
            self.add_packed_tensor_to_offload(packed_tensor)
        return packed_tensor
        
    def on_get_saved_tensor(self, saved_state: Any) -> torch.Tensor:
        """
        Hook called when autograd retrieves a saved tensor during backward pass.
        Returns the actual tensor (potentially reloading from CPU).
        """
        if isinstance(saved_state, (PagedTensor)):
            if self.status == 'capture':
                num_tokens = saved_state.num_tokens_tensor.item()
                self.temp_tokens_per_vp_stage[saved_state.vp_stage][saved_state.dtype][saved_state.hidden_size] -= num_tokens
                self.temp_tokens_across_vp_stages[saved_state.dtype, saved_state.hidden_size] -= num_tokens
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

            assert saved_state._tensor is not None, f"saved_state._tensor is None {saved_state._tensor}"
            return saved_state._tensor

        return saved_state
    
class PackedOffloadContext:
    """Wrapper context manager that adds custom enter/exit behavior around saved_tensors_hooks."""
    
    def __init__(self, offload_manager):
        self.offload_manager = offload_manager
        self.saved_tensors_context = torch.autograd.graph.saved_tensors_hooks(
            offload_manager.on_save_for_backward, 
            offload_manager.on_get_saved_tensor
        )
    
    def __enter__(self):
        from megatron.core.extensions.transformer_engine import cpu_offload

        if cpu_offload is not None:
            cpu_offload.CPUOffloadEnabled = True
        # Call the underlying context manager's __enter__
        result = self.saved_tensors_context.__enter__()
        
        # Add more custom logic after entering if needed
        return result
    
    def __exit__(self, *args: Any):
        # Call the underlying context manager's __exit__
        result = self.saved_tensors_context.__exit__(*args)
        from megatron.core.extensions.transformer_engine import cpu_offload

        if cpu_offload is not None:
            cpu_offload.CPUOffloadEnabled = False
        return result

def packed_moe_expert_offloading_group_start(tensor, name=None):
    """Mark the start of a layer group and prepare for offload/reload."""
    rank = torch.distributed.get_rank()
    offload_manager = PackedOffloadManager.get_instance()
    if not offload_manager.enabled:
        return tensor
    return PP_PreScheduleFunction.apply(tensor, offload_manager)

def get_packed_moe_expert_offloading_context(name=None, max_num_tokens=None, num_tokens_tensor=None):
    """Get the fine-grained offload context"""
    offload_manager = PackedOffloadManager.get_instance()
    if not offload_manager.enabled:
        return nullcontext()
    offload_manager.max_num_tokens = max_num_tokens
    assert num_tokens_tensor is not None and isinstance(num_tokens_tensor, torch.Tensor)
    offload_manager.num_tokens_tensor = num_tokens_tensor
    offload_manager.set_current_layer_name(name) if name is not None else None
    pack_unpack_context = PackedOffloadContext(offload_manager)
    return pack_unpack_context

def packed_moe_expert_offloading_group_commit(tensor, name=None):
    """Mark the end of a layer group and prepare for offload/reload."""
    rank = torch.distributed.get_rank()
    offload_manager = PackedOffloadManager.get_instance()
    offload_manager.device = tensor.device
    if not offload_manager.enabled:
        return tensor
    return PP_PostScheduleFunction.apply(tensor, offload_manager)

def packed_moe_expert_offloading_init_chunk_handler(vp_size, vp_stage):
    """Initialize the chunk handler, called at the start of a microbatch forward pass."""
    offload_manager = PackedOffloadManager.get_instance()
    if not offload_manager.enabled:
        return
    offload_manager.current_vp_stage = vp_stage if vp_stage is not None else 0
    if vp_size is not None:
        offload_manager.vp_size = vp_size
    else:
        offload_manager.vp_size = 1
    if offload_manager.max_tokens_per_vp_stage is None:
        offload_manager.max_tokens_per_vp_stage = [{} for _ in range(offload_manager.vp_size)]
        offload_manager.temp_tokens_per_vp_stage = [{} for _ in range(offload_manager.vp_size)]
    if offload_manager.max_tokens_across_vp_stages is None:
        offload_manager.max_tokens_across_vp_stages = {}
        offload_manager.temp_tokens_across_vp_stages = {}

def packed_moe_expert_offloading_set_last_layer(is_last_layer=False):
    """Set the last layer flag."""
    offload_manager = PackedOffloadManager.get_instance()
    if not offload_manager.enabled:
        return
    offload_manager._last_layer = is_last_layer

def packed_moe_expert_offloading_reset(enabled=True):
    """Reset the chunk handler, called at the start of a training iteration."""
    offload_manager = PackedOffloadManager.get_instance()
    offload_manager.enabled = enabled
    offload_manager.iteration += 1
    # current layer and microbatch for each vp stage for forward pass
    offload_manager.current_schedule_index = 0

    if not enabled:
        return

    set_ideal_affinity_for_current_gpu() # Set the ideal affinity for the current GPU
    if offload_manager.status == 'begin':
        offload_manager.status = 'capture'
    elif offload_manager.status == 'capture':
        offload_manager.status = 'captured'
        stash_buffer_size_factor = float(os.getenv('STASH_BUFFER_SIZE_FACTOR', '1.10'))
        offload_manager.allocate_offload_buffers(stash_buffer_size_factor=stash_buffer_size_factor)
    elif offload_manager.status == 'captured':
        pass

    if offload_manager.status == 'captured':
        if not torch.cuda.is_current_stream_capturing():
            overflow = offload_manager.overflow.item()
            assert overflow == 0, f"PackedOffloadManager overflow!!!"

        for vp_buffers in offload_manager.stash_buffers:
            for dtype in vp_buffers.keys():
                for hidden_size in vp_buffers[dtype].keys():
                    vp_buffers[dtype][hidden_size].reset()
        offload_manager.overflow.zero_()
        offload_manager.current_layer = [1 for _ in range(offload_manager.vp_size)]
        offload_manager.current_microbatch = [1 for _ in range(offload_manager.vp_size)]
        assert len(offload_manager.packed_tensors_to_offload) == 0, f"packed_tensors_to_offload is not empty {offload_manager.packed_tensors_to_offload}"
        assert len(offload_manager.packed_tensors_offload_in_progress) == 0, f"packed_tensors_offload_in_progress is not empty {offload_manager.packed_tensors_offload_in_progress}"

