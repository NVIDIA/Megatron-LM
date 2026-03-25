# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging
from contextlib import nullcontext
from typing import Any

import torch
import triton
import triton.language as tl

from megatron.core._rank_utils import log_single_rank
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.full_cuda_graph import FullCudaGraphWrapper
from megatron.core.utils import get_attr_wrapped_model

logger = logging.getLogger(__name__)

GLOBAL_BLOCK_SIZE = 1024
SCALE_INV_BLOCK_SIZE = 32


class PagedStashBuffer:
    """
    A paged stash buffer with page-level memory management.
    Supports both CUDA and optional pinned host buffer for overflow fallback.

    Buffers are organized as [num_pages, page_size, hidden_size].
    Uses per-buffer free lists (circular buffer) tracked as two-element state: [0]=CUDA, [1]=host.
    """

    def __init__(
        self, num_tokens, hidden_size, page_size, device, overflow, host_spill, dtype, num_tokens_host=0
    ):
        """
        Args:
            num_tokens: Maximum number of tokens the CUDA buffer can hold
            hidden_size: Hidden dimension size
            page_size: Number of tokens per page
            device: Device for the buffer
            overflow: Overflow flag tensor (shared across all buffers)
            host_spill: Global flag set to 1 if any stash used pinned host (shared)
            dtype: Data type
            num_tokens_host: If > 0, allocate pinned host buffer with this many tokens for spillover.
        """
        self.hidden_size = hidden_size
        self.page_size = page_size
        self.device = device
        self.dtype = dtype
        self.overflow = overflow  # GPU flag (shared)
        self.host_spill = host_spill

        # CUDA buffer
        self.num_cuda_pages = (num_tokens + page_size - 1) // page_size
        self.total_cuda_tokens = self.num_cuda_pages * page_size
        self.cuda_buffer = torch.empty(
            (self.total_cuda_tokens, hidden_size), dtype=dtype, device=device
        )

        # Host buffer (pinned), optional
        self.num_host_pages = (num_tokens_host + page_size - 1) // page_size if num_tokens_host > 0 else 0
        self.total_host_tokens = self.num_host_pages * page_size if self.num_host_pages > 0 else 0
        if self.num_host_pages > 0:
            self.host_buffer = torch.empty(
                (self.total_host_tokens, hidden_size), dtype=dtype, device='cpu', pin_memory=True
            )
        else:
            self.host_buffer = None

        # Free list state: shape (2,) index 0 = CUDA, 1 = host (all in device memory for kernel)
        self.free_list_head = torch.zeros(2, dtype=torch.int64, device=device)
        self.free_list_tail = torch.tensor(
            [self.num_cuda_pages, self.num_host_pages], dtype=torch.int64, device=device
        )
        self.free_list_capacity = torch.tensor(
            [self.num_cuda_pages, self.num_host_pages], dtype=torch.int64, device=device
        )

        # Free list arrays (device memory): page IDs for each buffer
        self.free_list_cuda = torch.arange(self.num_cuda_pages, dtype=torch.int64, device=device)
        if self.num_host_pages > 0:
            self.free_list_host = torch.arange(self.num_host_pages, dtype=torch.int64, device=device)
        else:
            self.free_list_host = torch.empty(0, dtype=torch.int64, device=device)

        # Pre-allocated reset values (CUDA graph safe: no allocation in reset())
        self._reset_tail = torch.tensor(
            [self.num_cuda_pages, self.num_host_pages],
            dtype=torch.int64,
            device=device,
        )
        self._reset_free_list_cuda = torch.arange(
            self.num_cuda_pages, dtype=torch.int64, device=device
        )
        if self.num_host_pages > 0:
            self._reset_free_list_host = torch.arange(
                self.num_host_pages, dtype=torch.int64, device=device
            )
        else:
            self._reset_free_list_host = None

    def reset(self):
        """Reset both CUDA and host free lists (CUDA graph safe: no new allocations)."""
        self.free_list_cuda.copy_(self._reset_free_list_cuda)
        self.free_list_head.zero_()
        self.free_list_tail.copy_(self._reset_tail)
        if self._reset_free_list_host is not None:
            self.free_list_host.copy_(self._reset_free_list_host)

    def __repr__(self):
        return (
            f"PagedStashBuffer(num_cuda_pages={self.num_cuda_pages}, num_host_pages={self.num_host_pages}, "
            f"page_size={self.page_size}, hidden_size={self.hidden_size}, device={self.device}, dtype={self.dtype})"
        )


@triton.jit
def _paged_stash_copy_kernel(
    src_ptr,
    cuda_dst_ptr,
    host_dst_ptr,
    num_tokens_ptr,
    free_list_cuda_ptr,
    free_list_host_ptr,
    free_list_head_ptr,   # shape (2,): [cuda_head, host_head]
    free_list_tail_ptr,  # shape (2,)
    free_list_capacity_ptr,
    page_record_ptr,
    overflow_ptr,
    host_spill_global_ptr,  # 1 if any successful host spill (not set on overflow path)
    spilled_to_host_ptr,   # Output: 0 = stored in CUDA, 1 = stored in host or overflow
    new_free_list_head_ptr,  # Output: shape (2,) updated heads
    PAGE_SIZE: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_HOST_BUFFER: tl.constexpr,
):
    """Copy tokens to paged stash: try CUDA first (fast path), then host if CUDA full."""
    pid = tl.program_id(axis=0)
    num_blocks = tl.num_programs(axis=0)

    # Load overflow first (get in flight early); branch on it only before any write
    overflow = tl.load(overflow_ptr)

    num_tokens = tl.load(num_tokens_ptr)
    required_pages = tl.cdiv(num_tokens, PAGE_SIZE)

    # Common case: load only CUDA state (and head_host for output when use_cuda)
    head_cuda = tl.load(free_list_head_ptr)
    head_host = tl.load(free_list_head_ptr + 1)
    tail_cuda = tl.load(free_list_tail_ptr)
    cap_cuda = tl.load(free_list_capacity_ptr)

    avail_cuda = tail_cuda - head_cuda
    use_cuda = avail_cuda >= required_pages

    # Assume CUDA path: set everything for GPU stash
    spill = 0
    dst_ptr = cuda_dst_ptr
    free_list_ptr = free_list_cuda_ptr
    head = head_cuda
    cap = cap_cuda
    new_head_cuda = head_cuda + required_pages
    new_head_host = head_host
    
    if overflow == 1:
        return

    # Only when CUDA is full: load host state and maybe switch to host
    if not use_cuda:
        tail_host = tl.load(free_list_tail_ptr + 1)
        cap_host = tl.load(free_list_capacity_ptr + 1)
        use_host = HAS_HOST_BUFFER == 1 and (tail_host - head_host) >= required_pages
        if use_host:
            spill = 1
            dst_ptr = host_dst_ptr
            free_list_ptr = free_list_host_ptr
            head = head_host
            cap = cap_host
            new_head_cuda = head_cuda
            new_head_host = head_host + required_pages
        else:
            if pid == 0:
                tl.store(overflow_ptr, 1)
                tl.store(spilled_to_host_ptr, 1)
                tl.store(new_free_list_head_ptr, head_cuda)
                tl.store(new_free_list_head_ptr + 1, head_host)
            return

    if pid == 0:
        tl.store(spilled_to_host_ptr, spill)
        if spill == 1:
            tl.store(host_spill_global_ptr, 1)

    # Copy loop: strided over tokens
    token_idx = pid
    while token_idx < num_tokens:
        page_slot = token_idx // PAGE_SIZE
        token_in_page = token_idx % PAGE_SIZE
        free_list_idx = (head + page_slot) % cap
        page_id = tl.load(free_list_ptr + free_list_idx)
        if token_in_page == 0:
            tl.store(page_record_ptr + page_slot, page_id)
        dst_token_idx = page_id * PAGE_SIZE + token_in_page

        elements_per_thread = HIDDEN_SIZE // BLOCK_SIZE
        need_mask = (HIDDEN_SIZE % BLOCK_SIZE) != 0
        num_iters = elements_per_thread + (1 if need_mask else 0)
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
        token_idx += num_blocks

    if pid == 0:
        tl.store(new_free_list_head_ptr, new_head_cuda)
        tl.store(new_free_list_head_ptr + 1, new_head_host)


@triton.jit
def _paged_stash_pop_kernel(
    cuda_src_ptr,
    host_src_ptr,
    dst_ptr,
    num_tokens_ptr,
    page_record_ptr,
    spilled_to_host_ptr,  # 0 = read from CUDA, 1 = read from host
    overflow_ptr,
    free_list_cuda_ptr,
    free_list_host_ptr,
    free_list_tail_ptr,   # shape (2,)
    free_list_capacity_ptr,
    new_free_list_tail_ptr,  # Output: shape (2,) updated tails
    PAGE_SIZE: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Reload tokens from paged stash; CUDA path fast, host path when spilled_to_host."""
    pid = tl.program_id(axis=0)
    num_blocks = tl.num_programs(axis=0)

    # Load overflow first (get in flight early); branch on it only before any write
    overflow = tl.load(overflow_ptr)

    num_tokens = tl.load(num_tokens_ptr)
    spill = tl.load(spilled_to_host_ptr)
    required_pages = tl.cdiv(num_tokens, PAGE_SIZE)

    # Common case: load only CUDA state (and tail_host for output when spill=0)
    tail_cuda = tl.load(free_list_tail_ptr)
    tail_host = tl.load(free_list_tail_ptr + 1)
    cap_cuda = tl.load(free_list_capacity_ptr)

    # Assume CUDA path
    src_ptr = cuda_src_ptr
    free_list_ptr = free_list_cuda_ptr
    tail = tail_cuda
    cap = cap_cuda
    new_tail_cuda = tail_cuda + required_pages
    new_tail_host = tail_host

    # Only when spilled to host: load host state and switch
    if spill == 1:
        cap_host = tl.load(free_list_capacity_ptr + 1)
        if cap_host == 0:
            return
        src_ptr = host_src_ptr
        free_list_ptr = free_list_host_ptr
        tail = tail_host
        cap = cap_host
        new_tail_cuda = tail_cuda
        new_tail_host = tail_host + required_pages

    if overflow == 1:
        return

    token_idx = pid
    while token_idx < num_tokens:
        page_slot = token_idx // PAGE_SIZE
        token_in_page = token_idx % PAGE_SIZE
        page_id = tl.load(page_record_ptr + page_slot)
        src_token_idx = page_id * PAGE_SIZE + token_in_page

        elements_per_thread = HIDDEN_SIZE // BLOCK_SIZE
        need_mask = (HIDDEN_SIZE % BLOCK_SIZE) != 0
        num_iters = elements_per_thread + (1 if need_mask else 0)
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

        if token_in_page == 0:
            write_idx = (tail + page_slot) % cap
            tl.store(free_list_ptr + write_idx, page_id)
        token_idx += num_blocks

    if pid == 0:
        tl.store(new_free_list_tail_ptr, new_tail_cuda)
        tl.store(new_free_list_tail_ptr + 1, new_tail_host)


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
        original_shape=None,
        schedule_layer_no=None,
        layer_name=None,
        max_num_tokens=None,
        hidden_size=None,
        page_size=64,
    ):
        """
        Args:
            tensor: The tensor to store
            num_tokens_tensor: Scalar tensor containing actual number of tokens
            vp_stage: Virtual pipeline stage
            layer_name: Name of the layer
            max_num_tokens: Maximum number of tokens
            hidden_size: Hidden size
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
        self.max_num_tokens = max_num_tokens
        self.hidden_size = hidden_size
        self.page_size = page_size

        # Original tensor information
        self.original_shape = list(tensor.shape) if original_shape is None else original_shape
        self.element_size = tensor.element_size()
        self.dtype = tensor.dtype
        self.device = tensor.device

        # Calculate number of pages needed
        self.max_num_pages = (self.max_num_tokens + page_size - 1) // page_size  # Ceiling division

        # Page record: stores which pages are being used for this tensor
        self.page_record = torch.zeros(self.max_num_pages, dtype=torch.int64, device=self.device)
        # Set by copy kernel: 0 = data in CUDA stash, 1 = data in host (pinned) stash
        self.spilled_to_host = torch.zeros(1, dtype=torch.int64, device=self.device)

    @property
    def schedule_layer(self):
        """Get the schedule layer."""
        return self.schedule_layer_no

    def offload_to_stash(self, paged_stash_buffer: PagedStashBuffer, max_blocks=2048):
        """Offload the paged tensor to paged stash buffer (CUDA or host if CUDA full)."""
        self._tensor = self._tensor.contiguous()
        if self.num_tokens_tensor.dim() == 0:
            self.num_tokens_tensor = self.num_tokens_tensor.reshape(1)
        if 'columnwise_scale_inv' in self.layer_name:
            num_tokens_tensor = self.num_tokens_tensor // SCALE_INV_BLOCK_SIZE
            max_num_tokens = self.max_num_tokens // SCALE_INV_BLOCK_SIZE
        else:
            num_tokens_tensor = self.num_tokens_tensor
            max_num_tokens = self.max_num_tokens

        tensor_to_copy = self._tensor
        BLOCK_SIZE = GLOBAL_BLOCK_SIZE
        num_blocks = min(max_num_tokens, max_blocks)
        grid = (num_blocks,)

        new_free_list_head = torch.empty(2, dtype=torch.int64, device=self.device)
        has_host = 1 if paged_stash_buffer.host_buffer is not None else 0
        host_dst = (
            paged_stash_buffer.host_buffer
            if paged_stash_buffer.host_buffer is not None
            else paged_stash_buffer.cuda_buffer
        )

        _paged_stash_copy_kernel[grid](
            tensor_to_copy.view(paged_stash_buffer.cuda_buffer.dtype),
            paged_stash_buffer.cuda_buffer,
            host_dst,
            num_tokens_tensor,
            paged_stash_buffer.free_list_cuda,
            paged_stash_buffer.free_list_host,
            paged_stash_buffer.free_list_head,
            paged_stash_buffer.free_list_tail,
            paged_stash_buffer.free_list_capacity,
            self.page_record,
            paged_stash_buffer.overflow,
            paged_stash_buffer.host_spill,
            self.spilled_to_host,
            new_free_list_head,
            PAGE_SIZE=self.page_size,
            HIDDEN_SIZE=self.hidden_size,
            BLOCK_SIZE=BLOCK_SIZE,
            HAS_HOST_BUFFER=has_host,
        )
        paged_stash_buffer.free_list_head.copy_(new_free_list_head)
        self._original_tensor = self._tensor
        self._tensor = None

    def reload_from_stash(self, paged_stash_buffer: PagedStashBuffer, max_blocks=2048):
        """Reload the paged tensor from paged stash buffer (CUDA or host from spilled_to_host)."""
        self._tensor = torch.empty(self.original_shape, dtype=self.dtype, device=self.device)
        tensor_to_reload = self._tensor

        if 'columnwise_scale_inv' in self.layer_name:
            num_tokens_tensor = self.num_tokens_tensor // SCALE_INV_BLOCK_SIZE
            max_num_tokens = self.max_num_tokens // SCALE_INV_BLOCK_SIZE
        else:
            num_tokens_tensor = self.num_tokens_tensor
            max_num_tokens = self.max_num_tokens
        BLOCK_SIZE = GLOBAL_BLOCK_SIZE
        num_blocks = min(max_num_tokens, max_blocks)
        grid = (num_blocks,)

        new_free_list_tail = torch.empty(2, dtype=torch.int64, device=self.device)
        host_src = (
            paged_stash_buffer.host_buffer
            if paged_stash_buffer.host_buffer is not None
            else paged_stash_buffer.cuda_buffer
        )
        _paged_stash_pop_kernel[grid](
            paged_stash_buffer.cuda_buffer,
            host_src,
            tensor_to_reload.view(paged_stash_buffer.cuda_buffer.dtype),
            num_tokens_tensor,
            self.page_record,
            self.spilled_to_host,
            paged_stash_buffer.overflow,
            paged_stash_buffer.free_list_cuda,
            paged_stash_buffer.free_list_host,
            paged_stash_buffer.free_list_tail,
            paged_stash_buffer.free_list_capacity,
            new_free_list_tail,
            PAGE_SIZE=self.page_size,
            HIDDEN_SIZE=self.hidden_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )

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
        self._pack_stream = torch.cuda.Stream()
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
        self.host_spill = None
        self.device = None

        # Page size for paged memory management (default; overwritten from config in paged_stash_reset)
        self.page_size = 64

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

    def allocate_stash_buffers(
        self,
        stash_buffer_size_factor_cuda: float = 1.10,
        stash_buffer_size_factor_cpu: float = 0.0,
    ):
        """Allocate stash buffers organized by [dtype][hidden_size]."""
        self.stash_buffers = {}
        self.overflow = torch.zeros(1, dtype=torch.int64, device=self.device)
        self.host_spill = torch.zeros(1, dtype=torch.int64, device=self.device)

        cuda_factor = stash_buffer_size_factor_cuda
        cpu_factor = stash_buffer_size_factor_cpu

        # Both factors use the same sign convention:
        # - positive: size based on avg_num_tokens-derived maxima
        # - negative: size based on actual num_tokens-derived maxima (legacy behavior)
        # Scale is always abs(factor). For CPU, 0 means no host buffer.
        if cuda_factor >= 0:
            max_tokens_dict = self.max_avg_tokens_across_vp_stages
            cuda_scale = cuda_factor
        else:
            max_tokens_dict = self.max_tokens_across_vp_stages
            cuda_scale = -cuda_factor

        # Fallback safety: if avg-based dict is not available/populated yet, use actual-max dict.
        if not max_tokens_dict:
            max_tokens_dict = self.max_tokens_across_vp_stages

        if cpu_factor > 0:
            host_tokens_dict = self.max_avg_tokens_across_vp_stages or self.max_tokens_across_vp_stages
            cpu_scale = cpu_factor
        elif cpu_factor < 0:
            host_tokens_dict = self.max_tokens_across_vp_stages
            cpu_scale = -cpu_factor
        else:
            host_tokens_dict = None
            cpu_scale = 0.0

        for dtype, hidden_size in max_tokens_dict:
            if dtype not in self.stash_buffers:
                self.stash_buffers[dtype] = {}
            assert hidden_size not in self.stash_buffers[dtype]
            num_tokens = int(max_tokens_dict[dtype, hidden_size] * cuda_scale)
            num_tokens_host = (
                int(host_tokens_dict[dtype, hidden_size] * cpu_scale)
                if host_tokens_dict is not None and (dtype, hidden_size) in host_tokens_dict
                else 0
            )
            buf_dtype = torch.uint8 if dtype in [torch.float8_e4m3fn, torch.float8_e8m0fnu] else dtype
            self.stash_buffers[dtype][hidden_size] = PagedStashBuffer(
                num_tokens,
                hidden_size,
                self.page_size,
                self.device,
                self.overflow,
                self.host_spill,
                buf_dtype,
                num_tokens_host=num_tokens_host,
            )
            sb = self.stash_buffers[dtype][hidden_size]
            msg = f'allocate_stash_buffers cuda: {sb.cuda_buffer.shape}'
            if sb.host_buffer is not None:
                msg += f' host: {sb.host_buffer.shape}'
            msg += f' dtype={sb.dtype} ({dtype})'
            log_single_rank(logger, logging.INFO, msg)

    def update_pp_schedule(self, vp_stage, layer_no=None, microbatch_no=None):
        """Update the pp schedule."""
        if self._pp_schedule is None:
            self._pp_schedule = []

        assert self.vp_size is not None
        if layer_no is None:
            # forward pass
            vp_stage_index = vp_stage - 1
            layer_no = self.current_layer[vp_stage_index]
            self.current_layer[vp_stage_index] += 1
            microbatch_no = self.current_microbatch[vp_stage_index]

        if self.status == 'capture':
            self._pp_schedule.append(self.get_schedule_layer(vp_stage, layer_no, microbatch_no))
            num_tokens = self.num_tokens_tensor.item()

        expected = self.get_schedule_layer(vp_stage, layer_no, microbatch_no)
        actual = self._pp_schedule[self.current_schedule_index]
        assert actual == expected, f"schedule {actual} != {expected}"

        return layer_no, microbatch_no


    def update_model_chunk(self, vp_stage_index):
        """Update layer=1, increment microbatch of new vp vp_stage."""
        if self.current_layer is None:
            # current layer and microbatch for each vp stage for forward pass
            self.current_layer = [1 for _ in range(self.vp_size)]
            self.current_microbatch = [0 for _ in range(self.vp_size)]
        self.current_layer[vp_stage_index] = 1
        self.current_microbatch[vp_stage_index] += 1

    def on_save_for_backward(self, tensor: torch.Tensor) -> Any:
        """
        Hook called when autograd saves a tensor for backward pass.
        Returns a tag to identify the tensor later.
        """
        # Handle 0-dim tensors (torch.Size([])) - they have no size(0)
        if (
            self.max_num_tokens is None
            or tensor.dim() == 0
            or not hasattr(tensor, 'grouped_name')
            or (tensor.size(0) != self.max_num_tokens and (tensor.logical_shape is None or tensor.logical_shape[0] != self.max_num_tokens))
        ):
            return tensor.detach()

        assert isinstance(tensor, torch.Tensor), f"tensor is not a torch.Tensor {type(tensor)}"

        original_shape = tensor.shape
        grouped_name = tensor.grouped_name
        tensor = tensor.flatten()
        dtype = tensor.dtype
        columnwise_scale_inv = 'columnwise_scale_inv' in grouped_name
        hidden_size = tensor.numel() // (self.max_num_tokens if not columnwise_scale_inv else self.max_num_tokens // SCALE_INV_BLOCK_SIZE)

        if self.max_tokens_across_vp_stages is None:
            self.max_tokens_across_vp_stages = {}
            self.temp_tokens_across_vp_stages = {}
            self.max_avg_tokens_across_vp_stages = {}
            self.temp_avg_tokens_across_vp_stages = {}

        avg_num_tokens = None
        if self.status == 'capture':

            self.num_tokens = self.num_tokens_tensor.item()
            actual_num_tokens = self.num_tokens // SCALE_INV_BLOCK_SIZE if columnwise_scale_inv else self.num_tokens

            avg_num_tokens = (
                int(self.avg_num_tokens) if self.avg_num_tokens is not None else None
            )

            if (dtype, hidden_size) not in self.temp_tokens_across_vp_stages:
                self.temp_tokens_across_vp_stages[dtype, hidden_size] = 0
                self.max_tokens_across_vp_stages[dtype, hidden_size] = 0
                self.temp_avg_tokens_across_vp_stages[dtype, hidden_size] = 0
                self.max_avg_tokens_across_vp_stages[dtype, hidden_size] = 0

            self.temp_tokens_across_vp_stages[dtype, hidden_size] += actual_num_tokens
            self.max_tokens_across_vp_stages[dtype, hidden_size] = max(
                self.max_tokens_across_vp_stages[dtype, hidden_size],
                self.temp_tokens_across_vp_stages[dtype, hidden_size],
            )

            # Track avg tokens across vp stages (if provided) using the same accumulation model.
            if avg_num_tokens is not None:
                self.temp_avg_tokens_across_vp_stages[dtype, hidden_size] += (avg_num_tokens if not columnwise_scale_inv else avg_num_tokens // SCALE_INV_BLOCK_SIZE)
                self.max_avg_tokens_across_vp_stages[dtype, hidden_size] = max(
                    self.max_avg_tokens_across_vp_stages[dtype, hidden_size],
                    self.temp_avg_tokens_across_vp_stages[dtype, hidden_size],
                )

            # Since capture stage does not use CUDA graph, we can truncate
            # the saved tensor to actual num_tokens
            new_size = (actual_num_tokens * hidden_size,)

            tensor_truncated = torch.empty(new_size, dtype=dtype, device=tensor.device)
            tensor_truncated.copy_(tensor[: actual_num_tokens * hidden_size])
            tensor = tensor_truncated

        tensor.grouped_name = grouped_name
        paged_tensor = PagedTensor(
            tensor,
            num_tokens_tensor=self.num_tokens_tensor,
            avg_num_tokens=avg_num_tokens,
            vp_stage=self.current_vp_stage,
            original_shape=original_shape,
            schedule_layer_no=(
                self._pp_schedule[self.current_schedule_index]
                if self._pp_schedule is not None
                and self.current_schedule_index < len(self._pp_schedule)
                else None
            ),
            layer_name=tensor.grouped_name,
            max_num_tokens=self.max_num_tokens,
            hidden_size=hidden_size,
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
            columnwise_scale_inv = 'columnwise_scale_inv' in saved_state.layer_name
            if self.status == 'capture':
                num_tokens = saved_state.num_tokens_tensor.item()
                key = (saved_state.dtype, saved_state.hidden_size)
                if key in self.temp_tokens_across_vp_stages:
                    self.temp_tokens_across_vp_stages[key] -= (num_tokens if not columnwise_scale_inv else num_tokens // SCALE_INV_BLOCK_SIZE)
                if (
                    saved_state.avg_num_tokens is not None
                    and key in self.temp_avg_tokens_across_vp_stages
                ):
                    self.temp_avg_tokens_across_vp_stages[key] -= (int(saved_state.avg_num_tokens) if not columnwise_scale_inv else int(saved_state.avg_num_tokens) // SCALE_INV_BLOCK_SIZE)

                # Handle 1-byte tensors (torch.uint8)
                dtype = saved_state._tensor.dtype
                if saved_state._tensor.element_size() == 1:
                    saved_state._tensor = saved_state._tensor.view(torch.uint8)

                # Pad the tensor to the max number of tokens
                # check if the tensor is 1D
                assert saved_state._tensor.ndim == 1, f"saved_state._tensor.ndim is not 1 {saved_state._tensor.ndim}"
                npad = (self.max_num_tokens - num_tokens) * saved_state.hidden_size
                if columnwise_scale_inv:
                    npad = npad // SCALE_INV_BLOCK_SIZE
                pad = (0, npad)
                saved_state._tensor = torch.nn.functional.pad(saved_state._tensor, pad).view(dtype)

            assert (
                saved_state._tensor is not None
            ), f"saved_state._tensor is None {saved_state._tensor}"

            # Record cross-stream usage (important when tensor was produced on another stream).
            if isinstance(saved_state._tensor, torch.Tensor) and saved_state._tensor.is_cuda:
                saved_state._tensor.record_stream(torch.cuda.current_stream())

            return saved_state._tensor.view(saved_state.original_shape)

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
    stash_manager.vp_size = vp_size if vp_size is not None else 1
    stash_manager.current_vp_stage = vp_stage if vp_stage is not None else 0
    stash_manager.update_model_chunk(stash_manager.current_vp_stage)

def paged_stash_set_last_layer(is_last_layer=False):
    """Set the last layer flag."""
    stash_manager = PagedStashManager.get_instance()
    if not stash_manager.enabled:
        return
    stash_manager._last_layer = is_last_layer

def paged_stash_reset(enabled=True, config=None):
    """Reset the chunk handler, called at the start of a training iteration.

    config: optional TransformerConfig; if provided, stash_buffer_size_factor_cuda/cpu and
    moe_paged_stash_page_size are read from it. Otherwise defaults to 1.10 (CUDA), 0.0 (CPU).
    """
    stash_manager = PagedStashManager.get_instance()
    stash_manager.enabled = enabled
    stash_manager.iteration += 1
    if config is not None:
        stash_manager.page_size = config.moe_paged_stash_page_size
    # current layer and microbatch for each vp stage for forward pass
    stash_manager.current_schedule_index = 0

    if not enabled:
        return

    if stash_manager.status == 'begin':
        stash_manager.status = 'capture'
    elif stash_manager.status == 'capture':
        stash_manager.status = 'captured'
        cuda_factor = config.stash_buffer_size_factor_cuda if config is not None else 1.10
        cpu_factor = config.stash_buffer_size_factor_cpu if config is not None else 0.0
        stash_manager.allocate_stash_buffers(
            stash_buffer_size_factor_cuda=cuda_factor,
            stash_buffer_size_factor_cpu=cpu_factor,
        )
    elif stash_manager.status == 'captured':
        pass

    if stash_manager.status == 'captured':
        for dtype in stash_manager.stash_buffers.keys():
            for hidden_size in stash_manager.stash_buffers[dtype].keys():
                stash_manager.stash_buffers[dtype][hidden_size].reset()
        stash_manager.overflow.zero_()
        stash_manager.host_spill.zero_()
        stash_manager.current_layer = [1 for _ in range(stash_manager.vp_size)]
        stash_manager.current_microbatch = [0 for _ in range(stash_manager.vp_size)]
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
        return torch.zeros(1, dtype=torch.bool, device='cuda')
    overflow = stash_manager.overflow.ne(0)
    return overflow


def check_paged_stash_host_spill():
    """True if any activation was stashed to pinned host (successful spill, not overflow path)."""
    stash_manager = PagedStashManager.get_instance()
    if not stash_manager.enabled or stash_manager.host_spill is None:
        return torch.zeros(1, dtype=torch.bool, device='cuda')
    return stash_manager.host_spill.ne(0)


class PagedStashRunner:
    """Runner for paged stash"""

    def __init__(self, config, copy_main_params, model, optimizer, forward_backward_func):
        self.stash_manager = PagedStashManager.get_instance()
        self.config = config
        self.copy_main_params = copy_main_params
        self.model = model
        self.optimizer = optimizer
        self.forward_backward_func = forward_backward_func
        self.moe_layers = []
        for model_chunk in self.model:
            model_with_decoder = get_attr_wrapped_model(
                model_chunk, "decoder", allow_none=False, return_model_obj=True
            )
            for layer in model_with_decoder.decoder.layers:
                mlp = layer.mlp
                if hasattr(mlp, 'token_dispatcher') and hasattr(
                    mlp.token_dispatcher, 'check_over_budget'
                ):
                    self.moe_layers.append(mlp)
            if model_with_decoder.mtp_process:
                for layer in model_with_decoder.mtp.layers:
                    mlp = layer.mtp_model_layer.mlp
                    if hasattr(mlp, 'token_dispatcher') and hasattr(
                        mlp.token_dispatcher, 'check_over_budget'
                    ):
                        self.moe_layers.append(mlp)

    def data_read(self, data_iterator, model, training, num_microbatches):
        """Read all microbatch inputs from Dataloader and copy to static buffers."""
        data_iterator_saved = []
        if not isinstance(model, list) or len(model) == 1:
            assert not isinstance(data_iterator, list) or len(data_iterator) == 1
            iterator0 = data_iterator if not isinstance(data_iterator, list) else data_iterator[0]
            data_list = []
            if iterator0 is not None:
                for b in range(num_microbatches):
                    data_list.append(next(iterator0))
                data_iterator_saved.append(data_list)
                data_list = [iter(data_list)]
            else:
                data_list.append(None)
        else:
            assert isinstance(data_iterator, list) and len(data_iterator) == len(model)
            data_list = []
            for i in range(len(model)):
                if data_iterator[i] is not None:
                    data_list_i = []
                    for b in range(num_microbatches):
                        data_list_i.append(next(data_iterator[i]))
                    data_iterator_saved.append(iter(data_list_i))
                    data_list.append(iter(data_list_i))
                else:
                    data_list.append(None)
        return data_iterator_saved, data_list

    def check_moe_overflow(self):
        """(stash_overflow_rank_sum, overbudget_rank_sum, host_spill_rank_sum); one all_reduce."""
        stash_overflow = check_paged_stash_overflow().view(-1)[0]
        host_spill = check_paged_stash_host_spill().view(-1)[0]
        overbudget = torch.zeros(1, dtype=torch.bool, device=stash_overflow.device).view(-1)[0]
        for mlp in self.moe_layers:
            ob = mlp.token_dispatcher.check_over_budget()
            if ob is not None:
                overbudget |= ob.view(-1)[0]

        flags = torch.stack(
            [
                stash_overflow.to(torch.int32),
                overbudget.to(torch.int32),
                host_spill.to(torch.int32),
            ],
            dim=0,
        )
        torch.distributed.all_reduce(flags, op=torch.distributed.ReduceOp.SUM)
        return flags[0].item(), flags[1].item(), flags[2].item()

    def prepare_for_rerun(self, is_training=True):
        """Prepare for rerun"""
        log_single_rank(
            logger,
            logging.WARNING,
            "Paged stash: rerunning forward-backward without moe_expert_rank_capacity_factor padding "
            "and with moe_paged_stash disabled.",
        )
        # check for token dispatcher overflow
        for mlp in self.moe_layers:
            if hasattr(mlp, 'token_dispatcher') and hasattr(
                mlp.token_dispatcher._comm_manager, 'moe_expert_rank_capacity_factor'
            ):
                mlp.token_dispatcher._comm_manager.moe_expert_rank_capacity_factor = None
                mlp.token_dispatcher._comm_manager.over_budget.fill_(0)
        self.stash_manager.overflow.zero_()
        if self.stash_manager.host_spill is not None:
            self.stash_manager.host_spill.zero_()
        self.config.moe_paged_stash = False

        # Set grad to zero.
        for model_chunk in self.model:
            model_chunk.zero_grad_buffer()
        if self.optimizer is not None:
            self.optimizer.zero_grad()

        #_handle_mxfp8_param_buffer_copy
        if self.copy_main_params:
            def _try_copy_main_params(opt):
                if isinstance(opt, DistributedOptimizer) and hasattr(opt, 'shard_fp32_from_float16_groups'):
                    opt._copy_main_params_to_param_buffer()
            # Handle both ChainedOptimizer and direct DistributedOptimizer cases
            # Note: FSDP's DistributedOptimizer doesn't have shard_fp32_from_float16_groups,
            # so we check for this attribute before calling _copy_main_params_to_param_buffer
            if self.optimizer is not None:
                if hasattr(self.optimizer, 'chained_optimizers'):
                    for optim_instance in self.optimizer.chained_optimizers:
                        _try_copy_main_params(optim_instance)
                else:
                    _try_copy_main_params(self.optimizer)

        # Delete the CUDA graph
        if isinstance(self.forward_backward_func, FullCudaGraphWrapper):
            self.forward_backward_func.reset_cuda_graph(stage='training' if is_training else 'validation')

    def __call__(self, *args, **kwargs):
        """Run the paged stash"""
        assert len(args) == 0, 'forward_backward_func does not accept positional args'
        assert all(
            [
                kwarg in kwargs
                for kwarg in [
                    'model',
                    'data_iterator',
                    'num_microbatches',
                    'seq_length',
                    'forward_only',
                ]
            ]
        )
        model = kwargs['model']
        num_microbatches = kwargs['num_microbatches']

        training = not kwargs['forward_only']
        data_iterator = kwargs['data_iterator']
        saved_moe_paged_stash = self.config.moe_paged_stash
        num_tries = 0
        while True:
            assert num_tries < 2, f"PagedStashRunner: num_tries {num_tries} exceeded max attempts!!!"
            num_tries += 1
            data_iterator, data_list = self.data_read(data_iterator, model, training, num_microbatches)
            kwargs['data_iterator'] = data_list
            result = self.forward_backward_func(*args, **kwargs)

            stash_overflow_ranks, overbudget_ranks, host_spill_ranks = self.check_moe_overflow()
            # if no overflow, set the expert_rank_capacity_factor to the original value
            if stash_overflow_ranks == 0 and overbudget_ranks == 0:
                if host_spill_ranks > 0:
                    log_single_rank(
                        logger,
                        logging.INFO,
                        "Paged stash: spilled activations to pinned host "
                        f"on {host_spill_ranks} rank(s) (CUDA stash full). "
                        "Consider increasing stash_buffer_size_factor_cuda for potentially better performance.",
                    )
                for mlp in self.moe_layers:
                    if hasattr(mlp, 'token_dispatcher') and hasattr(
                        mlp.token_dispatcher._comm_manager, 'moe_expert_rank_capacity_factor'
                    ):
                        mlp.token_dispatcher._comm_manager.moe_expert_rank_capacity_factor = mlp.token_dispatcher.config.moe_expert_rank_capacity_factor
                self.config.moe_paged_stash = saved_moe_paged_stash
                break

            # if overflow or overbudget, set the expert_rank_capacity_factor to None
            if overbudget_ranks > 0:
                log_single_rank(
                    logger,
                    logging.INFO,
                    "Paged stash: token drop during MoE token dispatch (over budget) "
                    f"on {overbudget_ranks} rank(s). "
                    "Consider increasing moe_expert_rank_capacity_factor.",
                )
            if stash_overflow_ranks > 0:
                log_single_rank(
                    logger,
                    logging.INFO,
                    "Paged stash: stashing buffer overflow "
                    f"on {stash_overflow_ranks} rank(s). "
                    "Consider increasing stash_buffer_size_factor_cuda or stash_buffer_size_factor_cpu.",
                )
            self.prepare_for_rerun(is_training=training)
        return result
