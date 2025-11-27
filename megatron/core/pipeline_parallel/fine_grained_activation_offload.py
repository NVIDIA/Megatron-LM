# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import warnings
from collections import deque
from contextlib import nullcontext
from typing import Any, Dict, List, Tuple

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
    
    def __init__(
        self, 
        device: str = 'cuda',
        pin_memory: bool = False
    ):
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
            'total_allocated': 0,       # Total number of tensors ever allocated
            'current_in_use': 0,        # Number of tensors currently in use
            'allocation_requests': 0,   # Number of allocation requests
            'free_requests': 0,         # Number of free requests
            'pool_hits': 0,             # Number of times a tensor was reused from pool
            'pool_misses': 0,           # Number of times a new tensor was created
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
                'free': deque(),      # Queue of available tensors
                'all': [],            # List of all tensors (for tracking)
                'allocated_count': 0, # Number of allocated tensors
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
            tensor = torch.empty(
                shape,
                dtype=dtype,
                device=self.device,
                pin_memory=self.pin_memory
            )
            pool['all'].append(tensor)
            self._stats['total_allocated'] += 1
            self._stats['pool_misses'] += 1
            
            memory_mb = self._calculate_memory_size(shape, dtype) / (1024 ** 2)
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
                'utilization': pool['allocated_count'] / total_count * 100 if total_count > 0 else 0,
            }
        else:
            # Return status for all pools
            status = {
                'global_stats': self._stats.copy(),
                'pools': {}
            }
            
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

    def reset(self):
        """Reset manager state for a new training iteration."""
        set_ideal_affinity_for_current_gpu()
        self._inside_context = False
        self._cur_forward_chunk = None
        self._cur_backward_chunk = None
        # Track the first microbatch of the last virtual pipeline stage
        self._is_first_last_vpp_chunk = True
        # Reset CPU tensor pool to reuse all CPU tensors for next iteration
        if hasattr(self, '_cpu_tensor_pool'):
            self._cpu_tensor_pool.reset()

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

    def push(self, handler):
        """Add a chunk handler to the backward queue."""
        debug_rank(f"pushing handler {handler}")
        self._queue.append(handler)

    def pop(self):
        """Remove and set the next non-empty chunk as the current backward chunk."""
        assert self.size(), "Cannot pop from empty queue"
        while self._queue:
            self._cur_backward_chunk = self._queue.popleft()
            if not self._cur_backward_chunk.is_empty_chunk():
                break
        debug_rank(f"popping handler {self._cur_backward_chunk}")

    def front(self):
        """Get the first non-empty chunk handler without removing it from the queue."""
        if not self.size():
            return None
        for chunk_handler in self._queue:
            if not chunk_handler.is_empty_chunk():
                return chunk_handler
        return None

    def size(self):
        """Return the number of chunk handlers in the queue."""
        return len(self._queue)

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
        vp_size = 1 if vp_size is None else vp_size
        if self._stages is None:
            self._vpp = vp_size
            self._stages = [[] for _ in range(vp_size)]

        if vp_stage is None:
            cur_vpp_rank = 0
        else:
            cur_vpp_rank = vp_stage

        is_first_last_vpp_chunk = self._is_first_last_vpp_chunk
        # Flush staged chunks when reaching the last virtual pipeline stage
        if cur_vpp_rank == self._vpp - 1:
            self.flush()
        # Determine if this is the first microbatch of the last virtual pipeline stage
        is_first_last_vpp_chunk = is_first_last_vpp_chunk and (cur_vpp_rank == self._vpp - 1)

        # Use shared CPU tensor pool for better reuse across chunks
        cur_chunk = ChunkOffloadHandler(
            is_first_last_vpp_chunk, min_offloaded_tensor_size, self._cpu_tensor_pool
        )
        self._stages[cur_vpp_rank].append(cur_chunk)
        # For the last stage, push immediately and flush
        if cur_vpp_rank == self._vpp - 1:
            if vp_size > 1:
                self._is_first_last_vpp_chunk = False
            self.push(cur_chunk)
            self.flush()
        self._cur_forward_chunk = cur_chunk
        cur_chunk.vpp_rank = cur_vpp_rank

    def set_last_layer(self, is_last_layer):
        """Mark whether the current forward chunk is processing the last layer."""
        self._cur_forward_chunk.is_last_layer = is_last_layer

    def cur_forward_chunk(self):
        """Get the current forward pass chunk handler."""
        return self._cur_forward_chunk

    def cur_backward_chunk(self):
        """Get the current backward pass chunk handler."""
        return self._cur_backward_chunk

    def __enter__(self):
        """Enter context manager to enable activation offloading hooks."""
        debug_rank("----__enter__")
        from megatron.core.extensions.transformer_engine import cpu_offload

        if cpu_offload is not None:
            cpu_offload.CPUOffloadEnabled = True
        self.inside_context = True

        torch._C._autograd._push_saved_tensors_default_hooks(
            self.on_save_for_backward, self.on_get_saved_tensor
        )

    def __exit__(self, *args: Any):
        """Exit context manager and restore original tensor saving behavior."""
        debug_rank("----__exit__")
        from megatron.core.extensions.transformer_engine import cpu_offload

        if cpu_offload is not None:
            cpu_offload.CPUOffloadEnabled = False
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

    def offload(self, src_tensor, pin_memory=True):
        """Offload."""
        debug_rank("--------offload")

        if not src_tensor.is_contiguous():
            src_tensor = src_tensor.contiguous()

        cpu_backup = self.cpu_tensor_pool.allocate(src_tensor.shape, dtype=src_tensor.dtype)

        cpu_backup.copy_(src_tensor, non_blocking=pin_memory)
        state = (src_tensor.device, cpu_backup)
        return state

    def reload(self, state, non_blocking=None):
        """Reload."""
        debug_rank("------reload")
        dev, cpu_backup = state
        if non_blocking is None:
            non_blocking = cpu_backup.is_pinned()
        gpu_tensor = torch.empty(
            cpu_backup.size(),
            dtype=cpu_backup.dtype,
            layout=cpu_backup.layout,
            device=dev,
        )
        gpu_tensor.copy_(cpu_backup, non_blocking=non_blocking)
        self.cpu_tensor_pool.free(cpu_backup)
        return gpu_tensor

    def __init__(self, is_first_last_vpp_chunk, min_offloaded_tensor_size, cpu_tensor_pool):
        self.do_offload = True
        # Data Structure to maintain reference to activation tensors
        self._tensor_tag_to_state = {}
        # Mark the first microbatch of the last virtual pipeline stage
        self._is_first_last_vpp_chunk = is_first_last_vpp_chunk

        # Group management for batching offload/reload operations
        self._offloaded_group_index = 0
        self._groups_to_offload = []
        self._groups_to_reload = []
        self._tensor_count_current_group = 0

        # Counter for special torch tensor types (FakeTensor, FunctionalTensor)
        self.torch_tensor_count = 0
        self.d2h_stream = PipelineOffloadManager.get_instance().d2h_stream
        self.h2d_stream = PipelineOffloadManager.get_instance().h2d_stream
        self._offload_events = {}
        self._reload_events = {}
        self.min_offloaded_tensor_size = min_offloaded_tensor_size
        self.is_last_layer = False
        self.cpu_tensor_pool = cpu_tensor_pool


    def is_empty_chunk(self):
        """Check if this chunk has no tensors to manage."""
        return len(self._tensor_tag_to_state) == 0

    def is_first_last_layer(self):
        """
        Check if this is the last layer of the first microbatch of the last vp stage.
        These tensors should not be offloaded to avoid unnecessary overhead.
        """
        debug_rank(
            f"------is_first_last_layer {self._is_first_last_vpp_chunk} {self.is_last_layer}"
        )
        return self._is_first_last_vpp_chunk and self.is_last_layer

    def tensor_push(self, tensor):
        """Push tensor to the offload handler."""
        torch_stray_tensor = isinstance(
            tensor,
            (
                torch._subclasses.fake_tensor.FakeTensor,
                torch._subclasses.functional_tensor.FunctionalTensor,
            ),
        )

        if not torch_stray_tensor:
            # Assign unique tag based on group index and position within group
            tensor_tag = (self._offloaded_group_index, self._tensor_count_current_group)
            self._tensor_count_current_group += 1
            assert tensor_tag not in self._tensor_tag_to_state, "Duplicate tensor tag"
            self._tensor_tag_to_state[tensor_tag] = tensor
        else:
            # Use negative group ID for special tensor types
            tensor_tag = (-1, self.torch_tensor_count)
            self.torch_tensor_count += 1
            self._tensor_tag_to_state[tensor_tag] = tensor
        debug_rank(f"--------tensor_push {tensor_tag}")
        return tensor_tag

    def tensor_pop(self, tensor_tag):
        """Pop tensor from the offload handler."""
        debug_rank(f"--------tensor_pop {tensor_tag}")
        assert tensor_tag in self._tensor_tag_to_state, f"Tag {tensor_tag} not found"
        tensor = self._tensor_tag_to_state.pop(tensor_tag)
        # If tensor is offloaded (stored as tuple), reload it
        if isinstance(tensor, tuple):
            tensor = self.reload(tensor)
        debug_rank(f"--------tensor_pop {tensor.shape}")
        return tensor

    def tensor_need_offloading_checker(self, tensor):
        """Check if the tensor needs to be offloaded."""
        if tensor.numel() < self.min_offloaded_tensor_size:
            return False
        # Respect tensor's offload preference if specified
        if hasattr(tensor, "offloading_activation") and not tensor.offloading_activation:
            return False
        return True

    def bulk_offload_group(self, group_to_offload):
        """offload a group of tensors recorded in tensor_push()."""
        debug_rank("------bulk_offload_group")
        assert not self.is_first_last_layer(), "Should not offload first-last layer"
        group_id_to_offload, name = group_to_offload
        torch.cuda.nvtx.range_push("activation offloading " + name)
        with torch.cuda.stream(self.d2h_stream):
            for tensor_tag, state in self._tensor_tag_to_state.items():
                group_id, _ = tensor_tag
                if group_id == group_id_to_offload:
                    debug_rank(f"------tensor_tag {tensor_tag}")
                    debug_rank(f"------group_to_offload {group_to_offload}")
                    assert not isinstance(state, tuple), "Tensor already offloaded"
                    tensor_on_device = state
                    if self.tensor_need_offloading_checker(tensor_on_device):
                        state = self.offload(tensor_on_device)
                        event = torch.cuda.Event()
                        event.record(self.d2h_stream)
                        self._offload_events[name] = event
                        tensor_on_device.record_stream(self.d2h_stream)
                        self._tensor_tag_to_state[tensor_tag] = state
        torch.cuda.nvtx.range_pop()

    def get_offload_event(self, name):
        """Get the CUDA event for a named offload operation."""
        return self._offload_events.get(name, None)

    def get_reload_event(self, name):
        """Get the CUDA event for a named reload operation."""
        return self._reload_events.get(name, None)

    def bulk_reload_group(self, group_to_reload):
        """Bulk reload group."""
        debug_rank("----bulk_reload_group")
        found_reload_group = False
        group_id_to_reload, name = group_to_reload
        torch.cuda.nvtx.range_push("activation reloading " + name)
        with torch.cuda.stream(self.h2d_stream):
            for tensor_label, state in self._tensor_tag_to_state.items():
                group_id, _ = tensor_label
                if group_id == group_id_to_reload:
                    debug_rank(f"----tensor_label {tensor_label}")
                    found_reload_group = True
                    event = self.get_offload_event(name)
                    # Only reload if tensor was offloaded (stored as tuple)
                    if isinstance(state, tuple):
                        # Wait for offload to complete before reloading
                        if not is_graph_capturing():
                            torch.cuda.current_stream().wait_event(event)
                        recovered_tensor = self.reload(state)
                        event.record(self.h2d_stream)
                        self._reload_events[name] = event
                        debug_rank(f"----recovered_tensor {recovered_tensor.shape}")
                        self._tensor_tag_to_state[tensor_label] = recovered_tensor
        torch.cuda.nvtx.range_pop()
        return found_reload_group

    def pre_reload_last_layer(self):
        """Pre-reload the last layer of this chunk to hide reload latency."""
        debug_rank("pre_reload_last_layer")
        assert not self._is_first_last_vpp_chunk, "Should not pre-reload first chunk"
        debug_rank(f"len(self._groups_to_reload) {len(self._groups_to_reload)}")
        if len(self._groups_to_reload) > 0:
            # Reload the last group (last layer) early
            if self.bulk_reload_group(self._groups_to_reload[-1]):
                self._groups_to_reload.pop()

    def should_bulk_offload(self):
        """Determine if the current group should be offloaded."""
        if not self.do_offload:
            return False
        # Don't offload the first backward chunk's last layer
        if self.is_first_last_layer():
            return False

        # Check if next backward chunk is this chunk (for last pipeline stage)
        next_backward_chunk = PipelineOffloadManager.get_instance().front()
        if next_backward_chunk is not None and next_backward_chunk is self:
            # Don't offload last layer if it's about to be used immediately
            if self.is_last_layer:
                return False

        return True

    def bulk_offload(self, forced_released_tensors):
        """Offload a group of tensors and optionally release their GPU memory."""
        debug_rank("----bulk_offload")
        if self.should_bulk_offload():
            group_to_offload = self._groups_to_offload.pop()
            # if group_to_offload[0] == 8:
            #     # print("rank", torch.distributed.get_rank(), "group_to_offload", group_to_offload)
            #     return
            self._groups_to_reload.append(group_to_offload)
            self.bulk_offload_group(group_to_offload)
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
        debug_rank("--on_group_commit_forward")
        # Wait for compute to finish before starting offload
        self.d2h_stream.wait_stream(torch.cuda.current_stream())
        self.bulk_offload(forced_released_tensors)

    def bulk_reload(self):
        """Reload the next group of tensors from CPU to GPU."""
        debug_rank("--bulk_reload")
        if len(self._groups_to_reload) > 0:
            # Reload the next layer group
            if self.bulk_reload_group(self._groups_to_reload[-1]):
                debug_rank(f"--bulk_reload_group {self._groups_to_reload}")
                self._groups_to_reload.pop()
        else:
            # Pre-load the last layer of the next backward chunk to hide latency
            next_backward_chunk = PipelineOffloadManager.get_instance().front()
            if next_backward_chunk is not None:
                next_backward_chunk.pre_reload_last_layer()

    def on_group_commit_backward(self, name):
        """
        Called at the end of a layer group's backward pass.
        Ensures correct chunk is active and synchronizes reloads.
        """
        debug_rank("--on_group_commit_backward")
        cur_backward_chunk = PipelineOffloadManager.get_instance().cur_backward_chunk()
        # Switch to this chunk if it's not already current
        if cur_backward_chunk is not self:
            PipelineOffloadManager.get_instance().pop()
        cur_backward_chunk = PipelineOffloadManager.get_instance().cur_backward_chunk()
        assert cur_backward_chunk is self, "Chunk mismatch"
        # Wait for reload to complete before using tensors
        event = self.get_reload_event(name)
        if event is not None and not is_graph_capturing():
            torch.cuda.current_stream().wait_event(event)
        self._offloaded_group_index = self._offloaded_group_index - 1

    def on_group_start_forward(self, name):
        """
        Called at the start of a layer group's forward pass.
        Increments group index and prepares for offloading.
        """
        debug_rank(f"--on_group_start_forward")
        self._offloaded_group_index = self._offloaded_group_index + 1
        self._tensor_count_current_group = 0
        self._groups_to_offload.append((self._offloaded_group_index, name))

    def on_group_start_backward(self):
        """
        Called at the start of a layer group's backward pass.
        Triggers reloading of tensors from CPU.
        """
        debug_rank("--on_group_start_backward")
        # Wait for compute to finish before starting reload
        self.h2d_stream.wait_stream(torch.cuda.current_stream())
        self.bulk_reload()

def fine_grained_offloading_disable_offload():
    """Disable the offload."""
    debug_rank("fine_grained_offloading_disable_offload")
    PipelineOffloadManager.get_instance().cur_forward_chunk().do_offload = False

def fine_grained_offloading_enable_offload():
    """Enable the offload."""
    debug_rank("fine_grained_offloading_enable_offload")
    PipelineOffloadManager.get_instance().cur_forward_chunk().do_offload = True

class FineGrainedOffloadingGroupCommitFunction(torch.autograd.Function):
    """
    Identity operation that marks the end of a layer group for offload synchronization.
    Triggers offload during forward and synchronizes reload during backward.
    """

    @staticmethod
    def forward(ctx, *args):
        # pylint: disable=missing-function-docstring
        debug_rank("FineGrainedOffloadingGroupCommitFunction forward")

        forced_released_tensors = args[-1]
        name = args[-2]
        cpu_offload_handler = args[-3]
        tensor = args[:-3]
        cpu_offload_handler.on_group_commit_forward(forced_released_tensors)
        ctx.cpu_offload_handler = cpu_offload_handler
        ctx.name = name

        # return the identical tensor
        return tensor

    @staticmethod
    def backward(ctx, *grad_output):
        # pylint: disable=missing-function-docstring
        debug_rank("FineGrainedOffloadingGroupCommitFunction backward")

        cpu_offload_handler = ctx.cpu_offload_handler
        cpu_offload_handler.on_group_commit_backward(ctx.name)
        return grad_output + (None, None, None)


def fine_grained_offloading_group_commit(*tensor, name, forced_released_tensors=[]):
    """
    Specify the tensors to be released after offloading.
    forced_released_tensors is a list of tensors to be released after offloading.
    The tensors will be untyped_storage().resize_(0) after offloading.
    Note: specify the tensors only when they are not automatically released by torch gc.
    """
    cur_forward_chunk = PipelineOffloadManager.get_instance().cur_forward_chunk()
    return FineGrainedOffloadingGroupCommitFunction.apply(
        *tensor, cur_forward_chunk, name, forced_released_tensors
    )


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
        return grad_output, None, None


def fine_grained_offloading_group_start(tensor, name=None):
    """Mark the start of a layer group and prepare for offload/reload."""
    cur_forward_chunk = PipelineOffloadManager.get_instance().cur_forward_chunk()
    return FineGrainedOffloadingGroupStartFunction.apply(tensor, cur_forward_chunk, name)


def get_fine_grained_offloading_context(flag):
    """Get the fine-grained offload context"""
    return PipelineOffloadManager.get_instance() if flag else nullcontext()


def fine_grained_offloading_set_last_layer(is_last_layer):
    """Set the last layer flag."""
    PipelineOffloadManager.get_instance().set_last_layer(is_last_layer)


def fine_grained_offloading_init_chunk_handler(vp_size, vp_stage, min_offloaded_tensor_size):
    """Initialize the chunk handler, called at the start of a microbatch forward pass."""
    PipelineOffloadManager.get_instance().init_model_chunk_offload_handler(
        vp_size, vp_stage, min_offloaded_tensor_size
    )

def fine_grained_offloading_reset():
    """Reset the chunk handler, called at the start of a training iteration."""
    PipelineOffloadManager.get_instance().reset()

def fine_grained_offloading_forward_record(event: torch.cuda.Event) -> None:
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
        ctx.event = event
        return tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        h2d_stream = PipelineOffloadManager.get_instance().h2d_stream
        torch.cuda.current_stream().record_event(ctx.event)
        torch.cuda.current_stream().wait_stream(h2d_stream)
        return grad_output, None

def fine_grained_offloading_backward_record(tensor, event: torch.cuda.Event) -> torch.Tensor:
    return FineGrainedOffloadingBackwardRecordFunction.apply(tensor, event)