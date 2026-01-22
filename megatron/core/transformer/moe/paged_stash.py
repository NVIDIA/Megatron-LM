# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
from contextlib import nullcontext
from typing import Any

import torch
import triton
import triton.language as tl
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor

GLOBAL_BLOCK_SIZE = 1024


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
        if os.getenv('PAGED_STASH_TO_CPU', '0') == '1':
            self.buffer = torch.empty(
                (self.total_tokens, hidden_size), dtype=dtype, device='cpu', pin_memory=True
            )
        else:
            self.buffer = torch.empty((self.total_tokens, hidden_size), dtype=dtype, device=device)

        self.overflow = overflow  # GPU flag (shared)
        self.device = device
        self.dtype = dtype

        # Free list as circular buffer: stores available page IDs
        self.free_list = torch.arange(self.num_pages, dtype=torch.int64, device=device)

        # Head and tail pointers for free_list circular buffer
        self.free_list_head = torch.zeros(
            1, dtype=torch.int64, device=device
        )  # Read pointer (allocation)
        self.free_list_tail = self.num_pages * torch.ones(
            1, dtype=torch.int64, device=device
        )  # Write pointer (deallocation)

        # Capacity of free list
        self.free_list_capacity = self.num_pages * torch.ones(1, dtype=torch.int64, device=device)

    def reset(self):
        """Reset the paged buffer - reinitialize free list."""
        self.free_list.copy_(torch.arange(self.num_pages, dtype=torch.int64, device=self.device))
        self.free_list_head.zero_()
        self.free_list_tail.fill_(self.num_pages)

    def __repr__(self):
        return (
            f"PagedStashBuffer(num_pages={self.num_pages}, page_size={self.page_size}, "
            f"hidden_size={self.hidden_size}, device={self.device}, dtype={self.dtype})"
        )


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

        # Use int64 for address math to avoid int32 overflow when indices get large.
        token_idx_i64 = token_idx.to(tl.int64)
        dst_token_idx_i64 = dst_token_idx.to(tl.int64)
        src_base = src_ptr + token_idx_i64 * HIDDEN_SIZE
        dst_base = dst_ptr + dst_token_idx_i64 * HIDDEN_SIZE

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

        # Use int64 for address math to avoid int32 overflow when indices get large.
        src_token_idx_i64 = src_token_idx.to(tl.int64)
        token_idx_i64 = token_idx.to(tl.int64)
        src_base = src_ptr + src_token_idx_i64 * HIDDEN_SIZE
        dst_base = dst_ptr + token_idx_i64 * HIDDEN_SIZE

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

    def __init__(
        self,
        tensor,
        num_tokens_tensor=None,
        avg_num_tokens: int = None,
        vp_stage=None,
        schedule_layer_no=None,
        layer_name=None,
        max_tokens=None,
        page_size=64,
    ):
        """
        Args:
            tensor: The tensor to store
            num_tokens_tensor: Scalar tensor containing actual number of tokens
            vp_stage: Virtual pipeline stage
            layer_name: Name of the layer
            max_tokens: Maximum number of tokens
            page_size: Number of tokens per page
        """
        self._tensor = tensor
        self._original_tensor = None
        assert (
            num_tokens_tensor is not None
            and isinstance(num_tokens_tensor, torch.Tensor)
            and num_tokens_tensor.numel() == 1
        )
        self.num_tokens_tensor = num_tokens_tensor.clone()
        self.avg_num_tokens = avg_num_tokens
        self.vp_stage = vp_stage
        self.schedule_layer_no = schedule_layer_no
        self.layer_name = layer_name
        self.max_tokens = max_tokens
        self.page_size = page_size

        # Original tensor information
        self.original_shape = list(tensor.shape)
        self.max_num_tokens = self.original_shape[0]
        self.element_size = tensor.element_size()
        self.hidden_size = self.original_shape[1]
        self.dtype = (
            tensor.dtype if not isinstance(tensor, MXFP8Tensor) else tensor._columnwise_data.dtype
        )
        self.device = tensor.device

        # Calculate number of pages needed
        self.max_num_pages = (self.max_num_tokens + page_size - 1) // page_size  # Ceiling division

        # Page record: stores which pages are being used for this tensor
        self.page_record = torch.zeros(self.max_num_pages, dtype=torch.int64, device=self.device)

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
        self._tensor = self._tensor.contiguous()
        if self.num_tokens_tensor.dim() == 0:
            self.num_tokens_tensor = self.num_tokens_tensor.reshape(1)

        # Get 2D tensor
        if isinstance(self._tensor, MXFP8Tensor):
            tensor_to_copy = self._tensor._columnwise_data
        else:
            tensor_to_copy = self._tensor

        # Determine grid size
        BLOCK_SIZE = GLOBAL_BLOCK_SIZE
        num_blocks = min(self.max_num_tokens, max_blocks)
        grid = (num_blocks,)

        # Create temporary tensor for new head
        new_free_list_head = torch.empty(1, dtype=torch.int64, device=self.device)

        # Launch paged stash copy kernel
        _paged_stash_copy_kernel[grid](
            tensor_to_copy,
            paged_stash_buffer.buffer,
            self.num_tokens_tensor,
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

        # Determine grid size
        BLOCK_SIZE = GLOBAL_BLOCK_SIZE
        num_blocks = min(self.max_num_tokens, max_blocks)
        grid = (num_blocks,)

        # Create temporary tensor for new tail
        new_free_list_tail = torch.empty(1, dtype=torch.int64, device=self.device)

        # Launch paged stash pop kernel
        _paged_stash_pop_kernel[grid](
            paged_stash_buffer.buffer,
            tensor_to_reload,
            self.num_tokens_tensor,
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
    def forward(ctx, tensor, stash_manager):  # after forward
        # pylint: disable=missing-function-docstring
        ctx.stash_manager = stash_manager
        # Wait for stash to complete before starting the next layer
        stash_manager.wait_for_stash_to_complete()
        return tensor

    @staticmethod
    def backward(ctx, *grad_output):  # before backward
        # pylint: disable=missing-function-docstring
        # Initiate reload for next layer
        if (
            ctx.stash_manager.status == 'captured'
            and ctx.stash_manager.current_schedule_index < len(ctx.stash_manager._pp_schedule)
        ):
            next_schedule_layer = ctx.stash_manager._pp_schedule[
                ctx.stash_manager.current_schedule_index
            ]
            if next_schedule_layer < 0:
                ctx.stash_manager.reload_paged_tensors(-next_schedule_layer)

        return grad_output + (None, None)


class PP_PostScheduleFunction(torch.autograd.Function):
    """
    This function is used to update the pp schedule.
    """

    @staticmethod
    def forward(ctx, tensor, stash_manager):  # after forward
        # pylint: disable=missing-function-docstring

        ctx.stash_manager = stash_manager
        ctx.vp_stage = stash_manager.current_vp_stage
        if ctx.vp_stage is None:
            ctx.vp_stage = 0
        ctx.layer_no, ctx.microbatch_no = stash_manager.update_pp_schedule(ctx.vp_stage + 1)

        # Initiate stash for current layer and reload for next layer
        if stash_manager.status == 'captured':
            current_schedule_layer = stash_manager.get_schedule_layer(
                ctx.vp_stage + 1, ctx.layer_no, ctx.microbatch_no
            )
            next_schedule_layer = ctx.stash_manager._pp_schedule[
                ctx.stash_manager.current_schedule_index + 1
            ]
            if current_schedule_layer != -next_schedule_layer:
                # Start stash for current layer
                ctx.stash_manager.stash_paged_tensors(current_schedule_layer)
                if next_schedule_layer < 0:
                    # reload for next backward layer
                    ctx.stash_manager.reload_paged_tensors(-next_schedule_layer, no_wait=True)
            else:
                ctx.stash_manager.remove_paged_tensor_from_stash()

        ctx.stash_manager.current_schedule_index += 1
        # return the identical tensor
        return tensor

    @staticmethod
    def backward(ctx, *grad_output):  # before backward
        # pylint: disable=missing-function-docstring
        if ctx.vp_stage is not None:
            ctx.stash_manager.update_pp_schedule(
                -(ctx.vp_stage + 1), -ctx.layer_no, -ctx.microbatch_no
            )
        ctx.stash_manager.current_schedule_index += 1
        current_stream = torch.cuda.current_stream()

        ctx.stash_manager.wait_for_stash_to_complete()
        if ctx.stash_manager._unpack_stream_status == 'reloading':
            current_stream.wait_stream(ctx.stash_manager.unpack_stream)
            ctx.stash_manager._unpack_stream_status = 'idle'

        return grad_output + (None, None)


class PagedStashManager:
    """
    Singleton manager for coordinating paged stashing across pipeline stages.
    Manages chunk handlers, synchronizes GPU-GPU transfers,
    and handles virtual pipeline parallelism
    """

    STASH_MGR = None

    @classmethod
    def get_instance(cls):
        """Get the singleton instance of PagedStashManager."""
        if cls.STASH_MGR is None:
            cls.STASH_MGR = PagedStashManager()
        return cls.STASH_MGR

    def __init__(self):
        """Initialize the manager with queues and dedicated CUDA streams."""
        # allocate streams and events for synchronization
        self.enabled = False
        self._pack_stream = torch.cuda.current_stream()#torch.cuda.Stream()
        # Currently paged stashing is not stream-safe, so use the same stream for packing
        # and unpacking
        self._unpack_stream = self._pack_stream
        self._pack_stream_status = 'idle'  # idle, stashing
        self._unpack_stream_status = 'idle'  # idle, reloading
        self.paged_tensors_to_stash = []
        self.paged_tensors_stash_in_progress = []
        self.paged_tensors_to_reload = {}

        self.iteration = 0
        self._current_layer_name = None
        self.vp_size = None
        self.current_vp_stage = None
        self._last_layer = False
        self.status = 'begin'  # begin, capture, captured
        # If element is +ve, it denotes forward pass of vp stage,
        # if -ve, it denotes backward pass of vp stage
        self._pp_schedule = None
        self.current_layer = None
        self.current_microbatch = None
        self.current_schedule_index = None

        # Track max tokens needed across all vp_stages grouped by dtype and hidden_size
        self.max_tokens_across_vp_stages = None
        self.temp_tokens_across_vp_stages = None
        # Track max tokens computed from avg_num_tokens (heuristic) across all vp_stages
        self.max_avg_tokens_across_vp_stages = None
        self.temp_avg_tokens_across_vp_stages = None

        self.num_tokens_tensor = None
        self.max_num_tokens = None
        # Optional hint: expected/average number of tokens (e.g., pre-padding estimate)
        self.avg_num_tokens = None
        self.stash_buffers = None
        self.overflow = None
        self.device = None

        # Page size for paged memory management
        self.page_size = int(os.getenv('PAGED_STASH_PAGE_SIZE', '64'))  # Default 64 tokens per page

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
        return vp_stage * 1000000 + layer_no * 1000 + microbatch_no

    def add_paged_tensor_to_stash(self, paged_tensor):
        """Add a paged tensor to the stash list."""
        if self.status == 'captured':
            self.paged_tensors_to_stash.append(paged_tensor)
        else:
            pass

    def remove_paged_tensor_from_stash(self):
        """Remove all paged tensors from the stash list."""
        if self.status == 'captured':
            while len(self.paged_tensors_to_stash) > 0:
                paged_tensor = self.paged_tensors_to_stash.pop(0)
            assert (
                len(self.paged_tensors_to_stash) == 0
            ), f"paged_tensors_to_stash is not empty {self.paged_tensors_to_stash}"
        else:
            pass

    def stash_paged_tensors(self, pp_schedule_layer):
        """Stash the paged tensors."""
        current_stream = torch.cuda.current_stream()
        self.pack_stream.wait_stream(current_stream)

        with torch.cuda.stream(self.pack_stream):
            if self.status == 'captured':
                self._pack_stream_status = 'stashing'
                if pp_schedule_layer not in self.paged_tensors_to_reload:
                    self.paged_tensors_to_reload[pp_schedule_layer] = []
                assert len(self.paged_tensors_to_reload[pp_schedule_layer]) == 0, (
                    f"paged_tensors_to_reload {pp_schedule_layer} is not empty "
                    f"{self.paged_tensors_to_reload[pp_schedule_layer]}"
                )

                while len(self.paged_tensors_to_stash) > 0:
                    paged_tensor = self.paged_tensors_to_stash.pop(0)
                    stash_buffer = self.stash_buffers[paged_tensor.dtype][paged_tensor.hidden_size]
                    paged_tensor.offload_to_stash(stash_buffer)
                    self.paged_tensors_to_reload[pp_schedule_layer].append(paged_tensor)
                    self.paged_tensors_stash_in_progress.append(paged_tensor)
            else:
                pass
        assert (
            len(self.paged_tensors_to_stash) == 0
        ), f"paged_tensors_to_stash is not empty {self.paged_tensors_to_stash}"

    def wait_for_stash_to_complete(self):
        """Wait for stash to complete."""
        current_stream = torch.cuda.current_stream()
        if self._pack_stream_status == 'stashing':
            current_stream.wait_stream(self.pack_stream)
            self._pack_stream_status = 'idle'

            # Deallocate original tensor after stash is complete
            while len(self.paged_tensors_stash_in_progress) > 0:
                paged_tensor = self.paged_tensors_stash_in_progress.pop(0)
                if isinstance(paged_tensor._original_tensor, MXFP8Tensor):
                    paged_tensor._original_tensor._columnwise_data = None
                else:
                    paged_tensor._original_tensor = None

    def reload_paged_tensors(self, pp_schedule_layer, no_wait=False):
        """Reload the paged tensors."""
        # Avoid waiting for main stream if reload is immediately after stash
        # since stash is already waiting for main stream
        if not no_wait or self.unpack_stream != self.pack_stream:
            current_stream = torch.cuda.current_stream()
            self.unpack_stream.wait_stream(current_stream)

        with torch.cuda.stream(self.unpack_stream):
            if self.status == 'captured':
                self._unpack_stream_status = 'reloading'
                count = 0
                for item in self.paged_tensors_to_reload:
                    if len(self.paged_tensors_to_reload[item]) > 0:
                        count += 1

                while len(self.paged_tensors_to_reload[pp_schedule_layer]) > 0:
                    paged_tensor = self.paged_tensors_to_reload[pp_schedule_layer].pop(0)
                    stash_buffer = self.stash_buffers[paged_tensor.dtype][paged_tensor.hidden_size]
                    paged_tensor.reload_from_stash(stash_buffer)
            else:
                pass
            assert len(self.paged_tensors_to_reload[pp_schedule_layer]) == 0, (
                f"paged_tensors_to_reload {pp_schedule_layer} is not empty "
                f"{self.paged_tensors_to_reload[pp_schedule_layer]}"
            )

    def allocate_stash_buffers(self, stash_buffer_size_factor=1.10):
        """Allocate stash buffers organized by [dtype][hidden_size]."""
        self.stash_buffers = {}
        self.overflow = torch.zeros(1, dtype=torch.int64, device=self.device)

        # stash_buffer_size_factor controls both which sizing signal to use and how much headroom
        # to allocate:
        # - positive: size based on avg_num_tokens-derived maxima
        # - negative: size based on actual num_tokens-derived maxima (legacy behavior)
        # In both cases we scale by abs(stash_buffer_size_factor).
        if stash_buffer_size_factor >= 0:
            max_tokens_dict = self.max_avg_tokens_across_vp_stages
            scale = stash_buffer_size_factor
        else:
            max_tokens_dict = self.max_tokens_across_vp_stages
            scale = -stash_buffer_size_factor

        # Fallback safety: if avg-based dict is not available/populated yet, use actual-max dict.
        if not max_tokens_dict:
            max_tokens_dict = self.max_tokens_across_vp_stages

        for dtype, hidden_size in max_tokens_dict:
            if dtype not in self.stash_buffers:
                self.stash_buffers[dtype] = {}
            assert hidden_size not in self.stash_buffers[dtype]
            num_tokens = int(
                max_tokens_dict[dtype, hidden_size] * scale
            )
            self.stash_buffers[dtype][hidden_size] = PagedStashBuffer(
                num_tokens, hidden_size, self.page_size, self.device, self.overflow, dtype
            )

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
            vp_stage_index = vp_stage - 1
            layer_no = self.current_layer[vp_stage_index]
            self.current_layer[vp_stage_index] += 1
            microbatch_no = self.current_microbatch[vp_stage_index]
            if self._last_layer:
                self.current_layer[vp_stage_index] = 1
                self.current_microbatch[vp_stage_index] += 1

        if self.status == 'capture':
            self._pp_schedule.append(self.get_schedule_layer(vp_stage, layer_no, microbatch_no))
            num_tokens = self.num_tokens_tensor.item()

        expected = self.get_schedule_layer(vp_stage, layer_no, microbatch_no)
        actual = self._pp_schedule[self.current_schedule_index]
        assert actual == expected, f"schedule {actual} != {expected}"

        return layer_no, microbatch_no

    def on_save_for_backward(self, tensor: torch.Tensor) -> Any:
        """
        Hook called when autograd saves a tensor for backward pass.
        Returns a tag to identify the tensor later.
        """

        # Handle 0-dim tensors (torch.Size([])) - they have no size(0)
        if (
            self.max_num_tokens is None
            or tensor.dim() == 0
            or tensor.size(0) != self.max_num_tokens
        ):
            return tensor.detach()
        if isinstance(tensor, MXFP8Tensor):
            assert (
                tensor._rowwise_data is None
            ), f"rowwise_data is not None; Only columnwise data is supported for paged stashing"

        avg_num_tokens = None
        if self.status == 'capture':

            self.num_tokens = self.num_tokens_tensor.item()
            avg_num_tokens = (
                int(self.avg_num_tokens) if self.avg_num_tokens is not None else None
            )

            dtype = (
                tensor.dtype
                if not isinstance(tensor, MXFP8Tensor)
                else tensor._columnwise_data.dtype
            )
            # Get hidden_size from tensor shape
            if isinstance(tensor, MXFP8Tensor):
                hidden_size = (
                    tensor._columnwise_data.shape[1]
                    if tensor._columnwise_data.ndim > 1
                    else tensor._columnwise_data.numel()
                )
            else:
                hidden_size = tensor.shape[1] if tensor.ndim > 1 else tensor.numel()

            if (dtype, hidden_size) not in self.temp_tokens_across_vp_stages:
                self.temp_tokens_across_vp_stages[dtype, hidden_size] = 0
                self.max_tokens_across_vp_stages[dtype, hidden_size] = 0
                self.temp_avg_tokens_across_vp_stages[dtype, hidden_size] = 0
                self.max_avg_tokens_across_vp_stages[dtype, hidden_size] = 0

            self.temp_tokens_across_vp_stages[dtype, hidden_size] += self.num_tokens
            self.max_tokens_across_vp_stages[dtype, hidden_size] = max(
                self.max_tokens_across_vp_stages[dtype, hidden_size],
                self.temp_tokens_across_vp_stages[dtype, hidden_size],
            )

            # Track avg tokens across vp stages (if provided) using the same accumulation model.
            if avg_num_tokens is not None:
                self.temp_avg_tokens_across_vp_stages[dtype, hidden_size] += avg_num_tokens
                self.max_avg_tokens_across_vp_stages[dtype, hidden_size] = max(
                    self.max_avg_tokens_across_vp_stages[dtype, hidden_size],
                    self.temp_avg_tokens_across_vp_stages[dtype, hidden_size],
                )
            # Since capture stage does not use CUDA graph, we can truncate
            # the saved tensor to actual num_tokens
            new_size = (self.num_tokens, *tensor.shape[1:])

            if isinstance(tensor, MXFP8Tensor):
                tensor_truncated = torch.empty(
                    new_size, dtype=tensor._columnwise_data.dtype, device=tensor.device
                )
                tensor_truncated.copy_(tensor._columnwise_data[: self.num_tokens, ...])
                tensor._columnwise_data = tensor_truncated
            else:
                tensor_truncated = torch.empty(new_size, dtype=tensor.dtype, device=tensor.device)
                tensor_truncated.copy_(tensor[: self.num_tokens, ...])
                tensor = tensor_truncated

        paged_tensor = PagedTensor(
            tensor,
            num_tokens_tensor=self.num_tokens_tensor,
            avg_num_tokens=avg_num_tokens,
            vp_stage=self.current_vp_stage,
            schedule_layer_no=(
                self._pp_schedule[self.current_schedule_index]
                if self._pp_schedule is not None
                and self.current_schedule_index < len(self._pp_schedule)
                else None
            ),
            layer_name=self._current_layer_name,
            max_tokens=self.max_num_tokens,
            page_size=self.page_size,
        )

        if self.status == 'captured':
            self.add_paged_tensor_to_stash(paged_tensor)
        return paged_tensor

    def on_get_saved_tensor(self, saved_state: Any) -> torch.Tensor:
        """
        Hook called when autograd retrieves a saved tensor during backward pass.
        Returns the actual tensor (potentially reloading from CPU).
        """
        if isinstance(saved_state, (PagedTensor)):
            if self.status == 'capture':
                num_tokens = saved_state.num_tokens_tensor.item()
                key = (saved_state.dtype, saved_state.hidden_size)
                if key in self.temp_tokens_across_vp_stages:
                    self.temp_tokens_across_vp_stages[key] -= num_tokens
                if (
                    saved_state.avg_num_tokens is not None
                    and key in self.temp_avg_tokens_across_vp_stages
                ):
                    self.temp_avg_tokens_across_vp_stages[key] -= int(saved_state.avg_num_tokens)
                # Pad the tensor to the max number of tokens
                npad = self.max_num_tokens - num_tokens
                pad = ()
                # check if the tensor is 2D
                assert (
                    saved_state._tensor.ndim == 2
                ), f"saved_state._tensor.ndim is not 2 {saved_state._tensor.ndim}"
                for _ in range(saved_state._tensor.ndim - 1):
                    pad = pad + (0, 0)
                pad = pad + (0, npad)
                if isinstance(saved_state._tensor, MXFP8Tensor):
                    saved_state._tensor._columnwise_data = torch.nn.functional.pad(
                        saved_state._tensor._columnwise_data, pad
                    )
                else:
                    saved_state._tensor = torch.nn.functional.pad(saved_state._tensor, pad)

            assert (
                saved_state._tensor is not None
            ), f"saved_state._tensor is None {saved_state._tensor}"

            # Record cross-stream usage (important when tensor was produced on another stream).
            if isinstance(saved_state._tensor, MXFP8Tensor):
                saved_state._tensor._columnwise_data.record_stream(torch.cuda.current_stream())
            elif isinstance(saved_state._tensor, torch.Tensor) and saved_state._tensor.is_cuda:
                saved_state._tensor.record_stream(torch.cuda.current_stream())

            return saved_state._tensor

        return saved_state


class PagedStashContext:
    """Wrapper context manager that adds custom enter/exit behavior around saved_tensors_hooks."""

    def __init__(self, stash_manager):
        self.stash_manager = stash_manager
        self.saved_tensors_context = torch.autograd.graph.saved_tensors_hooks(
            stash_manager.on_save_for_backward, stash_manager.on_get_saved_tensor
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


def paged_stash_group_start(tensor):
    """Mark the start of a layer group and prepare for stash/reload."""
    rank = torch.distributed.get_rank()
    stash_manager = PagedStashManager.get_instance()
    if not stash_manager.enabled:
        return tensor
    return PP_PreScheduleFunction.apply(tensor, stash_manager)


def get_paged_stash_context(
    name=None,
    max_num_tokens=None,
    num_tokens_tensor=None,
    avg_num_tokens=None,
):
    """Get the paged stash context"""
    stash_manager = PagedStashManager.get_instance()
    if not stash_manager.enabled:
        return nullcontext()
    stash_manager.max_num_tokens = max_num_tokens
    stash_manager.avg_num_tokens = avg_num_tokens
    assert num_tokens_tensor is not None and isinstance(num_tokens_tensor, torch.Tensor)
    stash_manager.num_tokens_tensor = num_tokens_tensor
    stash_manager.set_current_layer_name(name) if name is not None else None
    pack_unpack_context = PagedStashContext(stash_manager)
    return pack_unpack_context


def paged_stash_group_commit(tensor, name=None):
    """Mark the end of a layer group and prepare for stash/reload."""
    rank = torch.distributed.get_rank()
    stash_manager = PagedStashManager.get_instance()
    stash_manager.device = tensor.device
    if not stash_manager.enabled:
        return tensor
    return PP_PostScheduleFunction.apply(tensor, stash_manager)


def paged_stash_init_chunk_handler(vp_size, vp_stage):
    """Initialize the chunk handler, called at the start of a microbatch forward pass."""
    stash_manager = PagedStashManager.get_instance()
    if not stash_manager.enabled:
        return
    stash_manager.current_vp_stage = vp_stage if vp_stage is not None else 0
    if vp_size is not None:
        stash_manager.vp_size = vp_size
    else:
        stash_manager.vp_size = 1
    if stash_manager.max_tokens_across_vp_stages is None:
        stash_manager.max_tokens_across_vp_stages = {}
        stash_manager.temp_tokens_across_vp_stages = {}
        stash_manager.max_avg_tokens_across_vp_stages = {}
        stash_manager.temp_avg_tokens_across_vp_stages = {}


def paged_stash_set_last_layer(is_last_layer=False):
    """Set the last layer flag."""
    stash_manager = PagedStashManager.get_instance()
    if not stash_manager.enabled:
        return
    stash_manager._last_layer = is_last_layer


def paged_stash_reset(enabled=True):
    """Reset the chunk handler, called at the start of a training iteration."""
    stash_manager = PagedStashManager.get_instance()
    stash_manager.enabled = enabled
    stash_manager.iteration += 1
    # current layer and microbatch for each vp stage for forward pass
    stash_manager.current_schedule_index = 0

    if not enabled:
        return

    if stash_manager.status == 'begin':
        stash_manager.status = 'capture'
    elif stash_manager.status == 'capture':
        stash_manager.status = 'captured'
        stash_buffer_size_factor = float(os.getenv('STASH_BUFFER_SIZE_FACTOR', '1.10'))
        stash_manager.allocate_stash_buffers(stash_buffer_size_factor=stash_buffer_size_factor)
    elif stash_manager.status == 'captured':
        pass

    if stash_manager.status == 'captured':
        if not torch.cuda.is_current_stream_capturing():
            overflow = stash_manager.overflow.item()
            assert overflow == 0, f"PagedStashManager overflow!!!"

        for dtype in stash_manager.stash_buffers.keys():
            for hidden_size in stash_manager.stash_buffers[dtype].keys():
                stash_manager.stash_buffers[dtype][hidden_size].reset()
        stash_manager.overflow.zero_()
        stash_manager.current_layer = [1 for _ in range(stash_manager.vp_size)]
        stash_manager.current_microbatch = [1 for _ in range(stash_manager.vp_size)]
        assert (
            len(stash_manager.paged_tensors_to_stash) == 0
        ), f"paged_tensors_to_stash is not empty {stash_manager.paged_tensors_to_stash}"
        assert len(stash_manager.paged_tensors_stash_in_progress) == 0, (
            f"paged_tensors_stash_in_progress is not empty "
            f"{stash_manager.paged_tensors_stash_in_progress}"
        )


def check_paged_stash_overflow():
    """Check if paged stash overflow"""
    stash_manager = PagedStashManager.get_instance()
    if not stash_manager.enabled or stash_manager.overflow is None:
        return
    overflow = stash_manager.overflow.item()
    if overflow != 0:
        raise RuntimeError("PagedStashManager overflow!!!")
