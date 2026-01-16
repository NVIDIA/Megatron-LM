# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from collections import deque
from contextlib import nullcontext
from typing import Any, Dict, Tuple

import torch

# CPU offload implementation for pipeline parallelism
DEBUG = False
DEBUG_RANK = 0

from megatron.core.transformer.cuda_graphs import is_graph_capturing


def debug_rank(message):
    """Print debug message for a specific rank when DEBUG is enabled."""
    # pylint: disable=bad-builtin
    if not DEBUG:
        return
    assert torch.distributed.is_initialized()
    if torch.distributed.get_rank() == DEBUG_RANK:
        print(message)


def print_offload_summary_table(total_offload_bytes: Dict[str, int]):
    """
    Print an ASCII table summarizing offload bytes across all ranks.

    Gathers offload data from all ranks and prints a formatted table on rank 0,
    with rows representing ranks and columns representing groups.

    Args:
        total_offload_bytes: Dict mapping group names to offload bytes for this rank.
    """
    # pylint: disable=bad-builtin
    assert torch.distributed.is_initialized()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    # Gather all group names across ranks
    local_names = list(total_offload_bytes.keys())
    all_names_list = [None] * world_size
    torch.distributed.all_gather_object(all_names_list, local_names)
    all_group_names = sorted(set(name for names in all_names_list for name in names))

    # Gather offload bytes from all ranks: each rank sends a list of bytes per group
    local_bytes = [total_offload_bytes.get(name, 0) for name in all_group_names]
    all_bytes_list = [None] * world_size
    torch.distributed.all_gather_object(all_bytes_list, local_bytes)

    # Print ASCII table on rank 0
    if rank == 0:
        # Calculate column widths
        col_width = max(12, max((len(name) for name in all_group_names), default=8) + 2)
        rank_col_width = max(6, len(f"Rank {world_size - 1}") + 2)

        # Build header
        header = "Rank".ljust(rank_col_width)
        header += "".join(name.rjust(col_width) for name in all_group_names)
        header += "Total".rjust(col_width)
        separator = "-" * len(header)

        print("\n" + "=" * len(header))
        print("Activation Offload Summary (MB)".center(len(header)))
        print("=" * len(header))
        print(header)
        print(separator)

        # Build rows for each rank
        grand_total = 0
        col_totals = [0] * len(all_group_names)
        for r in range(world_size):
            row_bytes = all_bytes_list[r]
            row_total = sum(row_bytes)
            grand_total += row_total
            for i, b in enumerate(row_bytes):
                col_totals[i] += b
            row_str = f"Rank {r}".ljust(rank_col_width)
            for b in row_bytes:
                row_str += f"{b / (1024 * 1024):.2f}".rjust(col_width)
            row_str += f"{row_total / (1024 * 1024):.2f}".rjust(col_width)
            print(row_str)

        # Print totals row
        print(separator)
        totals_row = "Total".ljust(rank_col_width)
        for ct in col_totals:
            totals_row += f"{ct / (1024 * 1024):.2f}".rjust(col_width)
        totals_row += f"{grand_total / (1024 * 1024):.2f}".rjust(col_width)
        print(totals_row)
        print("=" * len(header) + "\n")

    torch.distributed.barrier()


class GPUTensorPool:
    """
    GPU memory pool for efficient allocation and deallocation of tensors.

    Features:
    - Supports multiple tensor shapes and dtypes, each with its own pool
    - Dynamic allocation: tensors are created on-demand during allocation
    - Efficient reuse: freed tensors are returned to the pool for reuse
    - Uses queue-based management for O(1) allocation and deallocation

    Example:
        pool = GPUTensorPool(device='cuda:0')
        tensor = pool.allocate((128, 512), dtype=torch.float32)
        # ... use tensor ...
        pool.free(tensor, (128, 512), dtype=torch.float32)
    """

    def __init__(self, device: str = 'cuda', pin_memory: bool = False):
        """
        Initialize GPU tensor pool.

        Args:
            device: GPU device, default 'cuda'
            pin_memory: Whether to use pinned memory (mainly for CPU tensors)
        """
        self.device = torch.device(device)
        self.pin_memory = pin_memory

        # Maintain a separate pool for each (shape, dtype) combination
        # Structure: {(shape, dtype): {'free': deque, 'all': list, 'allocated_count': int}}
        self._pools: Dict[Tuple, Dict[str, Any]] = {}

        # Statistics
        self._stats = {
            'total_allocated': 0,  # Total number of tensors ever allocated
            'current_in_use': 0,  # Number of tensors currently in use
            'allocation_requests': 0,  # Number of allocation requests
            'free_requests': 0,  # Number of free requests
            'pool_hits': 0,  # Number of times a tensor was reused from pool
            'pool_misses': 0,  # Number of times a new tensor was created
        }

        debug_rank("GPUTensorPool: Initialized with dynamic allocation")

    def _get_pool_key(self, shape: Tuple, dtype: torch.dtype) -> Tuple:
        """Generate a unique key for the pool based on shape and dtype."""
        return (shape, dtype)

    @staticmethod
    def _calculate_memory_size(shape: Tuple, dtype: torch.dtype) -> int:
        """Calculate memory size in bytes."""
        element_size = torch.tensor([], dtype=dtype).element_size()
        numel = 1
        for dim in shape:
            numel *= dim
        return numel * element_size

    def allocate(self, shape: Tuple, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Allocate a tensor with the specified shape and dtype.

        Args:
            shape: Shape of the tensor
            dtype: Data type of the tensor, default torch.float32

        Returns:
            Allocated tensor
        """
        self._stats['allocation_requests'] += 1

        pool_key = self._get_pool_key(shape, dtype)

        # Create pool for this (shape, dtype) if it doesn't exist
        if pool_key not in self._pools:
            self._pools[pool_key] = {
                'free': deque(),  # Queue of available tensors
                'all': [],  # List of all tensors (for tracking)
                'allocated_count': 0,  # Number of allocated tensors
            }

        pool = self._pools[pool_key]

        # Try to reuse a tensor from the pool
        if len(pool['free']) > 0:
            tensor = pool['free'].popleft()
            self._stats['pool_hits'] += 1
            debug_rank(
                f"GPUTensorPool.allocate: Reused tensor from pool, "
                f"shape={shape}, dtype={dtype}, "
                f"remaining in pool={len(pool['free'])}"
            )
        else:
            # Allocate a new tensor
            tensor = torch.empty(shape, dtype=dtype, device=self.device, pin_memory=self.pin_memory)
            pool['all'].append(tensor)
            self._stats['total_allocated'] += 1
            self._stats['pool_misses'] += 1

            memory_mb = self._calculate_memory_size(shape, dtype) / (1024**2)
            debug_rank(
                f"GPUTensorPool.allocate: Created new tensor, "
                f"shape={shape}, dtype={dtype}, "
                f"memory={memory_mb:.2f} MB, "
                f"total_created={len(pool['all'])}"
            )

        pool['allocated_count'] += 1
        self._stats['current_in_use'] += 1

        return tensor

    def free(self, tensor: torch.Tensor):
        """
        Return a tensor to the pool for reuse.

        Args:
            tensor: Tensor to free

        Raises:
            ValueError: If tensor doesn't belong to this pool
        """
        self._stats['free_requests'] += 1

        shape = tensor.shape
        dtype = tensor.dtype

        pool_key = self._get_pool_key(shape, dtype)

        if pool_key not in self._pools:
            raise ValueError(
                f"No pool exists for shape={shape}, dtype={dtype}. "
                f"Available pools: {list(self._pools.keys())}"
            )

        pool = self._pools[pool_key]

        # Verify tensor belongs to this pool (use identity check, not value comparison)
        tensor_found = any(tensor is t for t in pool['all'])
        if not tensor_found:
            raise ValueError(
                f"Attempting to free a tensor that doesn't belong to this pool "
                f"(shape={shape}, dtype={dtype})"
            )

        # Return tensor to the free queue
        pool['free'].append(tensor)
        pool['allocated_count'] -= 1
        self._stats['current_in_use'] -= 1

        debug_rank(
            f"GPUTensorPool.free: shape={shape}, dtype={dtype}, "
            f"available in pool={len(pool['free'])}"
        )

    def get_pool_status(self, shape: Tuple = None, dtype: torch.dtype = None) -> Dict[str, Any]:
        """
        Get the status of the memory pool.

        Args:
            shape: If specified along with dtype, return status for that specific pool
            dtype: Data type (required if shape is specified)

        Returns:
            Dictionary containing status information
        """
        if shape is not None:
            if dtype is None:
                raise ValueError("dtype must be specified when shape is provided")

            pool_key = self._get_pool_key(shape, dtype)

            if pool_key not in self._pools:
                raise ValueError(f"No pool exists for shape={shape}, dtype={dtype}")

            pool = self._pools[pool_key]
            total_count = len(pool['all'])

            return {
                'shape': shape,
                'dtype': dtype,
                'total_count': total_count,
                'allocated_count': pool['allocated_count'],
                'free_count': len(pool['free']),
                'utilization': (
                    pool['allocated_count'] / total_count * 100 if total_count > 0 else 0
                ),
            }
        else:
            # Return status for all pools
            status = {'global_stats': self._stats.copy(), 'pools': {}}

            for pool_key in self._pools:
                shape, dtype = pool_key
                status['pools'][pool_key] = self.get_pool_status(shape, dtype)

            return status

    def reset(self):
        """Reset the pool, marking all tensors as available."""
        debug_rank("GPUTensorPool: Resetting pool...")

        for pool_key, pool in self._pools.items():
            # Clear and refill the free queue
            pool['free'].clear()
            for tensor in pool['all']:
                pool['free'].append(tensor)
            pool['allocated_count'] = 0

        self._stats['current_in_use'] = 0
        debug_rank("GPUTensorPool: Reset complete")

    def clear(self):
        """Clear the pool and release all GPU memory."""
        debug_rank("GPUTensorPool: Clearing pool...")

        for pool_key, pool in self._pools.items():
            # Clear all references, allowing PyTorch GC to reclaim memory
            pool['free'].clear()
            pool['all'].clear()

        self._pools.clear()
        self._stats['current_in_use'] = 0

        # Trigger GPU cache cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        debug_rank("GPUTensorPool: Clear complete")

    def __del__(self):
        """Destructor to ensure resources are released."""
        self.clear()


class OffloadTensorGroup:
    """
    A group of tensors to be offloaded together.
    """

    def __init__(self, name):
        self._name = name
        self._tensors = {}
        self._offload_event = torch.cuda.Event()
        self._reload_event = torch.cuda.Event()
        self.offload = True
        self.total_offload_bytes = 0
        self.total_tensor_count = 0
        # Using memory pool is for the compatibility with cuda graph.
        # Shapes of tensors for expert_fc1 and moe_act are not known in advance,
        # so we do not use CPU pool for them.
        if name == "expert_fc1" or name == "moe_act":
            self.use_cpu_pool = False
        else:
            self.use_cpu_pool = True

    def push_tensor(self, tag, tensor):
        """Push a tensor to the group."""
        self._tensors[tag] = tensor

    def pop_tensor(self, tag):
        """Pop a tensor from the group."""
        return self._tensors.pop(tag)

    def record_offload_event(self, stream):
        """Record the offload event."""
        self._offload_event.record(stream)

    def wait_offload_event(self, stream):
        """Wait for the offload event."""
        stream.wait_event(self._offload_event)

    def record_reload_event(self, stream):
        """Record the reload event."""
        self._reload_event.record(stream)

    def wait_reload_event(self, stream):
        """Wait for the reload event."""
        stream.wait_event(self._reload_event)

    def update_offload_info(self, tensor):
        """Update the offload information."""
        self.total_offload_bytes += tensor.numel() * tensor.element_size()
        self.total_tensor_count += 1


class PipelineOffloadManager:
    """
    Singleton manager for coordinating activation offloading across pipeline stages.
    Manages chunk handlers, synchronizes GPU-CPU transfers,
    and handles virtual pipeline parallelism.
    """

    OFFLOAD_MGR = None

    @classmethod
    def get_instance(cls):
        """Get the singleton instance of PipelineOffloadManager."""
        if cls.OFFLOAD_MGR is None:
            cls.OFFLOAD_MGR = PipelineOffloadManager()
        return cls.OFFLOAD_MGR

    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance of PipelineOffloadManager."""
        cls.OFFLOAD_MGR = None
        cls.OFFLOAD_MGR = PipelineOffloadManager()

    def __init__(self):
        """Initialize the manager with queues and dedicated CUDA streams."""
        # Queue to store chunk handlers for backward pass
        self._queue = deque()
        # Cache chunk handlers for each virtual pipeline stage
        self._stages = None
        # allocate streams and events for synchronization
        self._d2h_stream = torch.cuda.Stream()
        self._h2d_stream = torch.cuda.Stream()
        # Shared CPU tensor pool for all chunks to improve reuse efficiency
        self._cpu_tensor_pool = GPUTensorPool(device="cpu", pin_memory=True)

        # Whether the manager is in warmup phase.
        self._is_warmup = True
        # Cache OffloadChunkHandler objects for each virtual pipeline stage and each forward pass.
        self._cached_chunks_forward = []
        # Cache OffloadChunkHandler objects for each virtual pipeline stage and each backward pass.
        self._cached_chunks_backward = []
        # Index of the current backward chunk in the cached chunks backward.
        self._cached_chunks_index_backward = 0
        # Index of the current forward chunk in the cached chunks forward.
        self._cached_chunks_index_forward = 0

        self.do_offload = True

        # Do not offload the last X groups so that the reloading won't block the computing stream.
        self._offload_margin = 0
        # Sometimes we need to delay the offloading and launch it later.
        # The delayed offload groups are stored in a queue.
        self._delayed_offload_groups = []
        self.reset()

    @property
    def d2h_stream(self):
        """Get the device-to-host (GPU to CPU) transfer stream."""
        return self._d2h_stream

    @property
    def h2d_stream(self):
        """Get the host-to-device (CPU to GPU) transfer stream."""
        return self._h2d_stream

    @property
    def cpu_tensor_pool(self):
        """Get the shared CPU tensor pool."""
        return self._cpu_tensor_pool

    def push_offload_groups(self, group_hook, forced_released_tensors):
        """Push the offload groups to the delayed queue."""
        debug_rank(f"pushing offload groups to the delayed queue")
        self._delayed_offload_groups.append((group_hook, forced_released_tensors))

    def flush_delayed_groups(self):
        """Flush the delayed groups."""
        debug_rank("flushing delayed groups")
        # Flush the delayed groups in reverse order to maintain the order of the groups.
        for group_hook, forced_released_tensors in reversed(self._delayed_offload_groups):
            group_hook(forced_released_tensors)
        self._delayed_offload_groups = []

    def reset(self):
        """Reset manager state for a new training iteration."""
        self._inside_context = False
        self._cur_forward_chunk = None
        self._cur_backward_chunk = None
        # Reset CPU tensor pool to reuse all CPU tensors for next iteration
        if hasattr(self, '_cpu_tensor_pool'):
            self._cpu_tensor_pool.reset()

        # Call post_warmup_callback after warmup to collect the offload information.
        if self._is_warmup and len(self._cached_chunks_forward) > 0:
            self.post_warmup_callback()
        self._cached_chunks_index_backward = 0
        self._cached_chunks_index_forward = 0

        for chunk in self._cached_chunks_forward:
            chunk.reset()
        self._delayed_offload_groups = []

    @property
    def offload_summary_bytes(self) -> Dict[str, int]:
        """Offload summary bytes per group collected after warmup."""
        return self._offload_summary_bytes

    @property
    def offload_summary_total_bytes(self) -> int:
        """Total offloaded bytes collected after warmup."""
        return self._offload_summary_total_bytes

    def flush(self):
        """Flush all staged chunks to the backward queue in reverse order."""
        # Ensure all virtual pipeline stages have the same number of chunks
        if len(self._stages[0]) == len(self._stages[-1]):
            lens = [len(e) for e in self._stages]
            assert min(lens) == max(lens), "All stages must have same chunk count"
            # Clear the last stage and push all chunks in reverse order for backward
            self._stages[-1] = []
            for chunks in reversed(self._stages):
                for chunk in chunks:
                    self.push(chunk)
            # Clear all stages after flushing
            for i in range(self._vpp):
                self._stages[i] = []

    def disable_offload(self):
        """Disable the offload."""
        debug_rank("disable_offload")
        self.do_offload = False
        for chunk in self._cached_chunks_forward:
            chunk.do_offload = False

    def enable_offload(self):
        """Enable the offload."""
        debug_rank("enable_offload")
        self.do_offload = True
        for chunk in self._cached_chunks_forward:
            chunk.do_offload = True

    def post_warmup_callback(self):
        """Callback after warmup."""
        # pylint: disable=bad-builtin
        debug_rank("post_warmup_callback")
        self._is_warmup = False
        assert len(self._cached_chunks_forward) == len(
            self._cached_chunks_backward
        ), "Cached chunks forward and backward must have the same length"
        for chunk in self._cached_chunks_forward:
            chunk.is_warmup = False
            assert (
                chunk in self._cached_chunks_backward
            ), "Chunk not found in cached chunks backward"
            # Update the offload margin to the maximum number of deduplicated groups
            self._offload_margin = max(self._offload_margin, chunk.get_max_deduplicated_groups())
            debug_rank(f"offload margin {self._offload_margin}")
        # Find the last group with the same name in the cached chunks backward
        last_group_with_same_name = {}
        for chunk_idx, chunk in enumerate(reversed(self._cached_chunks_backward)):
            for group in chunk.offload_groups:
                last_group_with_same_name[group._name] = group
        # Mark the last group with the same name as not offloadable to make sure
        # the reloading won't block the main stream.
        for name, group in last_group_with_same_name.items():
            if self._offload_margin > 0:
                group.offload = False
                self._offload_margin -= 1
                debug_rank(f"setting offload to false for group {name} at chunk index {chunk_idx}")
            else:
                break
        debug_rank(f"offload margin {self._offload_margin}")
        assert self._offload_margin == 0, "Offload margin is not 0"
        # Dump the offload information
        total_tensor_count = {}
        total_offload_bytes = {}
        for chunk in self._cached_chunks_forward:
            for group in chunk.offload_groups:
                if group.offload:
                    if group._name not in total_tensor_count:
                        total_tensor_count[group._name] = 0
                    total_tensor_count[group._name] += group.total_tensor_count
                    if group._name not in total_offload_bytes:
                        total_offload_bytes[group._name] = 0
                    total_offload_bytes[group._name] += group.total_offload_bytes
            # Stop statistics at the first backward chunk after which 1F1B is running,
            # where the memory cost will not increase anymore.
            if chunk is self._cached_chunks_backward[0]:
                break
        # Cache summary for downstream consumers (e.g., unit tests).
        self._offload_summary_bytes = dict(total_offload_bytes)
        self._offload_summary_total_bytes = int(sum(total_offload_bytes.values()))
        print_offload_summary_table(total_offload_bytes)

    def push(self, handler):
        """Add a chunk handler to the backward queue."""
        debug_rank(f"pushing handler {handler}")
        self._queue.append(handler)
        if self._is_warmup:
            self._cached_chunks_backward.append(handler)

    def pop_backward_chunk(self, name=None):
        """Get the next non-empty backward chunk containing the group with the given name."""
        self._cur_backward_chunk = None
        debug_rank(f"popping backward chunk {self._cached_chunks_index_backward}")
        debug_rank(f"cached chunks backward {self._cached_chunks_backward}")
        for idx, handler in enumerate(
            self._cached_chunks_backward[self._cached_chunks_index_backward :]
        ):
            self._cached_chunks_index_backward += 1
            if not handler.is_empty_chunk(name):
                self._cur_backward_chunk = (
                    handler  # set the first non-empty chunk as the current backward chunk
                )
                debug_rank(f"handler {handler} at index {idx} is not empty")
                break
        assert self._cur_backward_chunk is not None, "No non-empty chunk found"

    def front_backward_chunk(self, name=None):
        """Get the first non-empty backward chunk containing the group with the given name."""
        for idx, handler in enumerate(
            self._cached_chunks_backward[self._cached_chunks_index_backward :]
        ):
            if not handler.is_empty_chunk(name):
                debug_rank(f"front handler {handler} at index {idx}")
                return handler
        return None

    def init_model_chunk_offload_handler(
        self, vp_size, vp_stage, min_offloaded_tensor_size=1024 * 1024
    ):
        """
        Initialize a chunk offload handler for a model chunk (microbatch).

        Args:
            vp_size: Virtual pipeline size
            vp_stage: Virtual pipeline stage index (None means stage 0)
            min_offloaded_tensor_size: Minimum tensor size (in elements) to offload
        """
        if not self._is_warmup:
            return

        vp_size = 1 if vp_size is None else vp_size
        if self._stages is None:
            self._vpp = vp_size
            self._stages = [[] for _ in range(vp_size)]

        if vp_stage is None:
            cur_vpp_rank = 0
        else:
            cur_vpp_rank = vp_stage

        # Flush staged chunks when reaching the last virtual pipeline stage
        if cur_vpp_rank == self._vpp - 1:
            self.flush()

        # Use shared CPU tensor pool for better reuse across chunks
        cur_chunk = ChunkOffloadHandler(min_offloaded_tensor_size, self._cpu_tensor_pool)
        debug_rank(f"init_model_chunk_offload_handler {cur_chunk}")
        self._stages[cur_vpp_rank].append(cur_chunk)
        # For the last stage, push immediately and flush
        if cur_vpp_rank == self._vpp - 1:
            self.push(cur_chunk)
            self.flush()
        self._cur_forward_chunk = cur_chunk
        cur_chunk.vpp_rank = cur_vpp_rank
        self._cached_chunks_forward.append(cur_chunk)

    def pop_forward_chunk(self, name=None):
        """Get the next forward pass chunk handler."""
        debug_rank(f"pop_forward_chunk {self._cur_forward_chunk}")
        if not self.do_offload:
            return self._cur_forward_chunk
        while not self._is_warmup and (
            self._cur_forward_chunk is None or self._cur_forward_chunk.finish_all_groups(name)
        ):
            self._cur_forward_chunk = self._cached_chunks_forward[self._cached_chunks_index_forward]
            self._cached_chunks_index_forward += 1
            debug_rank(f"new cur_forward_chunk {self._cur_forward_chunk}")
        return self._cur_forward_chunk

    def cur_forward_chunk(self):
        """Get the current forward pass chunk handler."""
        return self._cur_forward_chunk

    def cur_backward_chunk(self):
        """Get the current backward pass chunk handler."""
        return self._cur_backward_chunk

    def mark_not_offloadable(self, tensor: torch.Tensor):
        """Mark the current forward chunk as not offloadable."""
        if tensor is not None:
            tensor.offloading_activation = False

    def __enter__(self):
        """Enter context manager to enable activation offloading hooks."""
        debug_rank("----__enter__")
        if self._cur_forward_chunk is None or not self.cur_forward_chunk().do_offload:
            return
        from megatron.core.extensions.transformer_engine import cpu_offload

        if cpu_offload is not None:
            cpu_offload.CPUOffloadEnabled = True
        else:
            raise RuntimeError("TE CPU offload is not available")
        self.inside_context = True

        torch._C._autograd._push_saved_tensors_default_hooks(
            self.on_save_for_backward, self.on_get_saved_tensor
        )

    def __exit__(self, *args: Any):
        """Exit context manager and restore original tensor saving behavior."""
        debug_rank("----__exit__")
        if self._cur_forward_chunk is None or not self.cur_forward_chunk().do_offload:
            return
        from megatron.core.extensions.transformer_engine import cpu_offload

        if cpu_offload is not None:
            cpu_offload.CPUOffloadEnabled = False
        else:
            raise RuntimeError("TE CPU offload is not available")
        self.inside_context = False
        torch._C._autograd._pop_saved_tensors_default_hooks()

    def on_save_for_backward(self, tensor: torch.Tensor) -> Any:
        """
        Hook called when autograd saves a tensor for backward pass.
        Returns a tag to identify the tensor later.
        """
        debug_rank(f"------on_save_for_backward {tensor.shape}")
        assert self.inside_context, "Must be inside offload context"
        return self.cur_forward_chunk().tensor_push(tensor)

    def on_get_saved_tensor(self, saved_state: Any) -> torch.Tensor:
        """
        Hook called when autograd retrieves a saved tensor during backward pass.
        Returns the actual tensor (potentially reloading from CPU).
        """
        debug_rank(f"----on_get_saved_tensor {saved_state}")
        return self.cur_backward_chunk().tensor_pop(saved_state)


class ChunkOffloadHandler:
    """
    Handles activation offloading and reloading for a single pipeline chunk (microbatch).
    Manages tensor groups, coordinates asynchronous GPU-CPU transfers, and handles synchronization.
    """

    def offload(self, src_tensor, pin_memory=True, use_cpu_pool=True):
        """Offload."""
        debug_rank("--------offload")

        if not src_tensor.is_contiguous():
            src_tensor = src_tensor.contiguous()

        if use_cpu_pool:
            cpu_backup = self.cpu_tensor_pool.allocate(src_tensor.shape, dtype=src_tensor.dtype)
        else:
            cpu_backup = torch.empty(
                src_tensor.shape, dtype=src_tensor.dtype, device="cpu", pin_memory=pin_memory
            )

        cpu_backup.copy_(src_tensor, non_blocking=pin_memory)
        state = (src_tensor.device, cpu_backup, use_cpu_pool)
        return state

    def reload(self, state, non_blocking=None):
        """Reload."""
        debug_rank("------reload")
        dev, cpu_backup, use_cpu_pool = state
        if non_blocking is None:
            non_blocking = cpu_backup.is_pinned()
        gpu_tensor = torch.empty(
            cpu_backup.size(), dtype=cpu_backup.dtype, layout=cpu_backup.layout, device=dev
        )
        gpu_tensor.copy_(cpu_backup, non_blocking=non_blocking)
        if use_cpu_pool:
            self.cpu_tensor_pool.free(cpu_backup)
        return gpu_tensor

    def __init__(self, min_offloaded_tensor_size, cpu_tensor_pool):
        self.do_offload = True

        # Group management for batching offload/reload operations
        self.offload_groups = []
        self._offloaded_group_index = 0
        # Groups to be offloaded.
        self._groups_to_offload = []
        # Groups to be reloaded.
        self._groups_to_reload = []
        # Tensor count for the current group.
        self._tensor_count_current_group = 0
        # Maximum number of groups to offload or reload.
        self._max_group_size = 0
        # Groups being reloaded.
        self._reloading_group = []
        # Counter for special torch tensor types (FakeTensor, FunctionalTensor)
        self.torch_tensor_count = 0
        self.d2h_stream = PipelineOffloadManager.get_instance().d2h_stream
        self.h2d_stream = PipelineOffloadManager.get_instance().h2d_stream
        self.min_offloaded_tensor_size = min_offloaded_tensor_size
        self.cpu_tensor_pool = cpu_tensor_pool
        self.is_warmup = True

    def reset(self):
        """Reset the chunk offload handler."""
        self._offloaded_group_index = 0
        self._groups_to_offload = []
        self._groups_to_reload = []
        self._tensor_count_current_group = 0
        self._reloading_group = []

    def find_group_with_name(self, name: str, start_index: int = 0):
        """Find the group with the given name starting from the given index."""
        return next(
            (group for group in self.offload_groups[start_index:] if group._name == name), None
        )

    def is_empty_chunk(self, name=None):
        """Check if this chunk has no tensors to manage."""
        debug_rank(f"------is_empty_chunk {self._max_group_size}")
        if name is not None:
            return self.find_group_with_name(name) is None
        return self._max_group_size == 0

    def finish_all_groups(self, name=None) -> bool:
        """Finish all groups."""
        debug_rank(
            f"------finish_all_groups {self} {self._max_group_size} {self._offloaded_group_index}"
        )
        # TODO: check if this is correct
        # Mark it as finished when there are no groups to offload or reload
        if (
            len(self._groups_to_reload) == 0
            and len(self._groups_to_offload) == 0
            and self._offloaded_group_index > 0
        ):
            return True
        assert name is not None, "Name is required"
        return self.find_group_with_name(name, self._offloaded_group_index) is None

    def find_next_group(self, name=None):
        """Find the next group with the given name."""
        assert name is not None, "Name is required"
        return self.find_group_with_name(name, self._offloaded_group_index)

    def tensor_push(self, tensor):
        """Push tensor to the offload handler."""
        torch_stray_tensor = isinstance(
            tensor,
            (
                torch._subclasses.fake_tensor.FakeTensor,
                torch._subclasses.functional_tensor.FunctionalTensor,
            ),
        )
        assert not torch_stray_tensor, "Stray tensor should not be offloaded"

        # Assign unique tag based on group index and position within group
        tensor_tag = (self._offloaded_group_index, self._tensor_count_current_group)
        self._tensor_count_current_group += 1
        self.offload_groups[self._offloaded_group_index - 1].push_tensor(tensor_tag, tensor)
        debug_rank(f"--------tensor_push {tensor_tag}")
        return tensor_tag

    def tensor_pop(self, tensor_tag):
        """Pop tensor from the offload handler."""
        debug_rank(f"--------tensor_pop {tensor_tag}")
        group_id, idx = tensor_tag
        tensor = self.offload_groups[group_id - 1].pop_tensor(tensor_tag)
        # If tensor is offloaded (stored as tuple), reload it
        if isinstance(tensor, tuple):
            tensor = self.reload(tensor)
        debug_rank(f"--------tensor_pop {tensor.shape}")
        return tensor

    def tensor_need_offloading_checker(self, tensor):
        """Check if the tensor needs to be offloaded."""
        debug_rank(
            f"tensor_need_offloading_checker {getattr(tensor, 'offloading_activation', None)}"
        )
        if tensor.numel() < self.min_offloaded_tensor_size:
            return False
        # Respect tensor's offload preference if specified
        if hasattr(tensor, "offloading_activation") and not tensor.offloading_activation:
            return False
        return True

    def bulk_offload_group(self):
        """offload a group of tensors recorded in tensor_push()."""
        debug_rank("------bulk_offload_group")
        group_to_offload = self._groups_to_offload[-1]
        torch.cuda.nvtx.range_push("activation offloading " + group_to_offload._name)
        with torch.cuda.stream(self.d2h_stream):
            for tensor_tag, tensor_on_device in group_to_offload._tensors.items():
                if self.tensor_need_offloading_checker(tensor_on_device):
                    state = self.offload(
                        tensor_on_device, use_cpu_pool=group_to_offload.use_cpu_pool
                    )
                    if self.is_warmup:
                        group_to_offload.update_offload_info(tensor_on_device)
                    tensor_on_device.record_stream(self.d2h_stream)
                    group_to_offload.push_tensor(tensor_tag, state)
            group_to_offload.record_offload_event(self.d2h_stream)
        self._groups_to_offload.pop()
        torch.cuda.nvtx.range_pop()

    def get_max_deduplicated_groups(self):
        """Get the maximum number of deduplicated groups."""
        count_modules = []
        for group in self.offload_groups:
            if group._name not in count_modules:
                count_modules.append(group._name)
        return len(count_modules)

    def bulk_reload_group(self):
        """Bulk reload group."""
        debug_rank("----bulk_reload_group")
        group_to_reload = self._groups_to_reload[-1]
        torch.cuda.nvtx.range_push("activation reloading " + group_to_reload._name)
        with torch.cuda.stream(self.h2d_stream):
            # Wait for offload to complete before reloading
            if not is_graph_capturing():
                group_to_reload.wait_offload_event(self.h2d_stream)
            for tensor_tag, state in group_to_reload._tensors.items():
                # Only reload if tensor was offloaded (stored as tuple)
                if isinstance(state, tuple):
                    recovered_tensor = self.reload(state)
                    debug_rank(f"----recovered_tensor {recovered_tensor.shape}")
                    group_to_reload.push_tensor(tensor_tag, recovered_tensor)
            group_to_reload.record_reload_event(self.h2d_stream)
        self._groups_to_reload.pop()
        # Add the group to the reloading group to wait for the reload event.
        self._reloading_group.append(group_to_reload)
        torch.cuda.nvtx.range_pop()

    def pre_reload_last_layer(self):
        """Pre-reload the last layer of this chunk to hide reload latency."""
        debug_rank("pre_reload_last_layer")
        debug_rank(f"len(self._groups_to_reload) {len(self._groups_to_reload)}")
        if len(self._groups_to_reload) > 0:
            # Reload the last group (last layer) early
            self.bulk_reload_group()

    def should_bulk_offload(self):
        """Determine if the current group should be offloaded."""
        assert len(self._groups_to_offload) > 0, "No groups to offload"
        group = self._groups_to_offload[-1]
        debug_rank(f"should_bulk_offload {self.is_warmup} {group.offload}")
        # Don't offload if the chunk is not in warmup stage
        if self.is_warmup:
            return True
        # Don't offload if the group is marked as not offloadable
        if not group.offload:
            return False

        # Check if next backward chunk is this chunk (for last pipeline stage)
        next_backward_chunk = PipelineOffloadManager.get_instance().front_backward_chunk(
            group._name
        )
        if next_backward_chunk is not None and next_backward_chunk is self:
            # Don't offload the last group with the same name if it's about to be used immediately
            if self.find_next_group(group._name) is None:
                debug_rank(f"next group {group._name} is not found")
                return False

        return True

    def bulk_offload(self, forced_released_tensors):
        """Offload a group of tensors and optionally release their GPU memory."""
        debug_rank("----bulk_offload")
        if self.should_bulk_offload():
            self._groups_to_reload.append(self._groups_to_offload[-1])
            self.bulk_offload_group()
            # Manually release tensors not auto-freed by torch GC
            if len(forced_released_tensors) > 0:
                cur_stream = torch.cuda.current_stream()
                for release_tensor in forced_released_tensors:
                    if self.tensor_need_offloading_checker(release_tensor):
                        # Ensure tensor is not in use before freeing
                        release_tensor.record_stream(cur_stream)
                        release_tensor.untyped_storage().resize_(0)

    def on_group_commit_forward(self, forced_released_tensors):
        """Called at the end of a layer group's forward pass to trigger offloading."""
        if not self.do_offload:
            return
        debug_rank("--on_group_commit_forward")
        # Wait for compute to finish before starting offload
        self.d2h_stream.wait_stream(torch.cuda.current_stream())
        self.bulk_offload(forced_released_tensors)

    def bulk_reload(self):
        """Reload the next group of tensors from CPU to GPU."""
        debug_rank("--bulk_reload")
        if len(self._groups_to_reload) > 0:
            # Reload the next layer group
            self.bulk_reload_group()
        else:
            # Pre-load the last layer of the next backward chunk to hide latency
            next_backward_chunk = PipelineOffloadManager.get_instance().front_backward_chunk()
            # Don't pre-reload the last layer if the next backward chunk hasn't finished fprop yet.
            if (
                next_backward_chunk is not None
                and next_backward_chunk._offloaded_group_index
                == next_backward_chunk._max_group_size
            ):
                next_backward_chunk.pre_reload_last_layer()

    def on_group_commit_backward(self, name):
        """
        Called at the end of a layer group's backward pass.
        Ensures correct chunk is active and synchronizes reloads.
        """
        if not self.do_offload:
            return
        debug_rank("--on_group_commit_backward")
        cur_backward_chunk = PipelineOffloadManager.get_instance().cur_backward_chunk()
        # Switch to this chunk if it's not already current
        if cur_backward_chunk is not self:
            PipelineOffloadManager.get_instance().pop_backward_chunk(name)
        cur_backward_chunk = PipelineOffloadManager.get_instance().cur_backward_chunk()
        assert cur_backward_chunk is self, f"Chunk mismatch {cur_backward_chunk} {self}"
        # Wait for reload to complete before using tensors
        if not is_graph_capturing() and len(self._reloading_group) > 0:
            for reloading_group in self._reloading_group:
                if reloading_group._name == name:
                    reloading_group.wait_reload_event(torch.cuda.current_stream())
                    self._reloading_group.remove(reloading_group)
                    break

    def on_group_start_forward(self, name):
        """
        Called at the start of a layer group's forward pass.
        Increments group index and prepares for offloading.
        """
        if not self.do_offload:
            return
        debug_rank(f"--on_group_start_forward {name}")
        self._offloaded_group_index = self._offloaded_group_index + 1
        if self.is_warmup:
            self.offload_groups.append(OffloadTensorGroup(name))
            self._max_group_size = max(self._max_group_size, self._offloaded_group_index)
            debug_rank(f"max group size {self._max_group_size}")
        else:
            for group in self.offload_groups[self._offloaded_group_index - 1 :]:
                if group._name == name:
                    break
                self._offloaded_group_index = self._offloaded_group_index + 1
        self._tensor_count_current_group = 0
        self._groups_to_offload.append(self.offload_groups[self._offloaded_group_index - 1])
        debug_rank(f"groups to offload {self._groups_to_offload}")

    def on_group_start_backward(self):
        """
        Called at the start of a layer group's backward pass.
        Triggers reloading of tensors from CPU.
        """
        if not self.do_offload:
            return
        debug_rank(f"--on_group_start_backward {self}")
        # Wait for compute to finish before starting reload
        self.h2d_stream.wait_stream(torch.cuda.current_stream())
        self.bulk_reload()


def fine_grained_offloading_disable_offload():
    """Disable the offload."""
    debug_rank("fine_grained_offloading_disable_offload")
    PipelineOffloadManager.get_instance().disable_offload()


def fine_grained_offloading_enable_offload():
    """Enable the offload."""
    debug_rank("fine_grained_offloading_enable_offload")
    PipelineOffloadManager.get_instance().enable_offload()


class FineGrainedOffloadingGroupCommitFunction(torch.autograd.Function):
    """
    Identity operation that marks the end of a layer group for offload synchronization.
    Triggers offload during forward and synchronizes reload during backward.
    """

    @staticmethod
    def forward(ctx, tensor, cur_forward_chunk, name, forced_released_tensors, delay_offload):
        # pylint: disable=missing-function-docstring
        debug_rank("FineGrainedOffloadingGroupCommitFunction forward")

        if delay_offload:
            PipelineOffloadManager.get_instance().push_offload_groups(
                cur_forward_chunk.on_group_commit_forward, forced_released_tensors
            )
        else:
            cur_forward_chunk.on_group_commit_forward(forced_released_tensors)
        ctx.cpu_offload_handler = cur_forward_chunk
        ctx.name = name
        return tensor

    @staticmethod
    def backward(ctx, *grad_output):
        # pylint: disable=missing-function-docstring
        debug_rank("FineGrainedOffloadingGroupCommitFunction backward")

        cpu_offload_handler = ctx.cpu_offload_handler
        cpu_offload_handler.on_group_commit_backward(ctx.name)
        return grad_output + (None, None, None, None)


def fine_grained_offloading_group_commit(
    tensor, name, forced_released_tensors=None, delay_offload=False
):
    """
    Specify the tensors to be released after offloading.
    forced_released_tensors is a list of tensors to be released after offloading.
    The tensors will be untyped_storage().resize_(0) after offloading.
    Note: specify the tensors only when they are not automatically released by torch gc.
    """
    # Be permissive: callers may pass a tuple/list of outputs (e.g., (q, k, v)).
    # We only need to insert a single identity op into the autograd graph; applying
    # it to the first tensor output is sufficient and keeps callers' code minimal.
    if forced_released_tensors is None:
        forced_released_tensors = []
    if isinstance(tensor, tuple):
        if len(tensor) == 0:
            return tensor
        committed0 = fine_grained_offloading_group_commit(
            tensor[0],
            name=name,
            forced_released_tensors=forced_released_tensors,
            delay_offload=delay_offload,
        )
        return (committed0,) + tensor[1:]
    if isinstance(tensor, list):
        if len(tensor) == 0:
            return tensor
        committed0 = fine_grained_offloading_group_commit(
            tensor[0],
            name=name,
            forced_released_tensors=forced_released_tensors,
            delay_offload=delay_offload,
        )
        return [committed0] + tensor[1:]

    cur_forward_chunk = PipelineOffloadManager.get_instance().cur_forward_chunk()
    if cur_forward_chunk is None:
        return tensor
    return FineGrainedOffloadingGroupCommitFunction.apply(
        tensor, cur_forward_chunk, name, forced_released_tensors, delay_offload
    )


def fine_grained_offloading_group_flush_delayed_groups():
    """Flush the delayed groups."""
    debug_rank("fine_grained_offloading_group_flush_delayed_groups")
    PipelineOffloadManager.get_instance().flush_delayed_groups()


class FineGrainedOffloadingGroupStartFunction(torch.autograd.Function):
    """
    Identity operation that marks the start of a layer group for offload/reload.
    Prepares for offload during forward and triggers reload during backward.
    """

    @staticmethod
    def forward(ctx, tensor, cpu_offload_handler, name):
        # pylint: disable=missing-function-docstring
        ctx.cpu_offload_handler = cpu_offload_handler
        debug_rank("FineGrainedOffloadingGroupStartFunction forward")

        cpu_offload_handler.on_group_start_forward(name)
        # return the identical tensor
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        # pylint: disable=missing-function-docstring
        debug_rank("FineGrainedOffloadingGroupStartFunction backward")
        cpu_offload_handler = ctx.cpu_offload_handler
        cpu_offload_handler.on_group_start_backward()
        return grad_output, None, None, None


def fine_grained_offloading_group_start(tensor, name=None):
    """Mark the start of a layer group and prepare for offload/reload."""
    cur_forward_chunk = PipelineOffloadManager.get_instance().pop_forward_chunk(name=name)
    if cur_forward_chunk is None:
        return tensor
    return FineGrainedOffloadingGroupStartFunction.apply(tensor, cur_forward_chunk, name)


def fine_grained_offloading_forward_record(event: torch.cuda.Event) -> None:
    """Record the forward event for cuda graph capture."""
    d2h_stream = PipelineOffloadManager.get_instance().d2h_stream
    torch.cuda.current_stream().record_event(event)
    torch.cuda.current_stream().wait_stream(d2h_stream)


class FineGrainedOffloadingBackwardRecordFunction(torch.autograd.Function):
    """
    Identity operation that marks the end of a layer group for offload synchronization.
    Triggers offload during forward and synchronizes reload during backward.
    """

    @staticmethod
    def forward(ctx, tensor, event: torch.cuda.Event) -> torch.Tensor:
        """Forward pass for cuda graph capture."""
        ctx.event = event
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        """Record the backward event and wait for the h2d stream on cuda graph stream."""
        h2d_stream = PipelineOffloadManager.get_instance().h2d_stream
        torch.cuda.current_stream().record_event(ctx.event)
        torch.cuda.current_stream().wait_stream(h2d_stream)
        return grad_output, None


def fine_grained_offloading_backward_record(tensor, event: torch.cuda.Event) -> torch.Tensor:
    """Record the backward event for cuda graph capture."""
    return FineGrainedOffloadingBackwardRecordFunction.apply(tensor, event)


class FineGrainedActivationOffloadingInterface:
    """Interface for fine-grained activation offloading."""

    def __init__(self, offload: bool, tensor: torch.Tensor, name: str):
        self.offload = offload
        self.tensor = tensor
        self.name = name

    def __enter__(self):
        """Enter context manager to enable activation offloading hooks."""
        if self.offload:
            self.tensor = fine_grained_offloading_group_start(self.tensor, self.name)
            PipelineOffloadManager.get_instance().__enter__()
        return self.tensor

    def __exit__(self, *args: Any):
        """Exit context manager to disable activation offloading hooks."""
        if self.offload:
            PipelineOffloadManager.get_instance().__exit__()

    @staticmethod
    def init_chunk_handler(vp_size, vp_stage, min_offloaded_tensor_size):
        """Initialize the chunk handler, called at the start of a microbatch forward pass."""
        PipelineOffloadManager.get_instance().init_model_chunk_offload_handler(
            vp_size, vp_stage, min_offloaded_tensor_size
        )

    @staticmethod
    def get_context(flag):
        """Get the fine-grained offload context"""
        return PipelineOffloadManager.get_instance() if flag else nullcontext()

    @staticmethod
    def group_commit(tensor, name, forced_released_tensors=None, delay_offload=False):
        """Group commit the tensors."""
        return fine_grained_offloading_group_commit(
            tensor, name, forced_released_tensors, delay_offload
        )

    @staticmethod
    def mark_not_offloadable(tensor: torch.Tensor):
        """Mark the tensor as not offloadable."""
        PipelineOffloadManager.get_instance().mark_not_offloadable(tensor)

    @staticmethod
    def forward_record(event: torch.cuda.Event) -> None:
        """Record the forward event for cuda graph capture."""
        d2h_stream = PipelineOffloadManager.get_instance().d2h_stream
        torch.cuda.current_stream().record_event(event)
        torch.cuda.current_stream().wait_stream(d2h_stream)

    @staticmethod
    def reset():
        """Reset the chunk handler."""
        PipelineOffloadManager.get_instance().reset()

    @staticmethod
    def reset_instance():
        """Reset the singleton instance."""
        PipelineOffloadManager.reset_instance()
