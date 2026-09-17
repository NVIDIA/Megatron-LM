# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
import time
from collections import defaultdict, deque
from contextlib import nullcontext
from typing import Any, Dict, Optional, Tuple

import torch
from torch.autograd.graph import saved_tensors_hooks

# CPU offload implementation for pipeline parallelism
DEBUG = False
DEBUG_RANK = 0

from megatron.core.transformer.cuda_graphs import is_graph_capturing
from megatron.core.utils import nvtx_range_pop, nvtx_range_push

try:
    from transformer_engine.pytorch.vmm_activation import (
        launch_remap_slot_h2d,
        release_hooks_after,
        remap_and_copy_slot_after,
        remap_only_slot_after,
        wait_remap_slot_on_stream,
        wait_until_remap_slot_submitted,
    )
except ImportError:
    release_hooks_after = None  # eager-only fallback; local graph paths require TE
    remap_and_copy_slot_after = None
    remap_only_slot_after = None
    launch_remap_slot_h2d = None
    wait_remap_slot_on_stream = None
    wait_until_remap_slot_submitted = None


def debug_rank(message):
    """Print debug message for a specific rank when DEBUG is enabled."""
    # pylint: disable=bad-builtin
    if not DEBUG:
        return
    assert torch.distributed.is_initialized()
    if torch.distributed.get_rank() == DEBUG_RANK:
        print(message)


def _te_do_not_offload(tensor):
    """Return whether TE marked a tensor-like object as non-offloadable."""
    if getattr(tensor, "_TE_do_not_offload", False):
        return True
    if not hasattr(tensor, "get_data_tensors"):
        return False
    try:
        data_tensors = tensor.get_data_tensors()
    except Exception:  # pragma: no cover - best effort for third-party tensor wrappers
        return False
    return any(
        data_tensor is not None and getattr(data_tensor, "_TE_do_not_offload", False)
        for data_tensor in data_tensors
    )


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


class OffloadTensorPool:
    """
    Memory pool for efficient allocation and deallocation of tensors.

    Features:
    - Supports multiple tensor shapes and dtypes, each with its own pool
    - Dynamic allocation: tensors are created on-demand during allocation
    - Efficient reuse: freed tensors are returned to the pool for reuse
    - Uses queue-based management for O(1) allocation and deallocation

    Example:
        pool = OffloadTensorPool(device='cuda:0')
        tensor = pool.allocate((128, 512), dtype=torch.float32)
        # ... use tensor ...
        pool.free(tensor, (128, 512), dtype=torch.float32)
    """

    def __init__(self, device: str = 'cuda', pin_memory: bool = False):
        """
        Initialize offload tensor pool.

        Args:
            device: Device, default 'cuda'
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

        debug_rank("OffloadTensorPool: Initialized with dynamic allocation")

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
                f"OffloadTensorPool.allocate: Reused tensor from pool, "
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
                f"OffloadTensorPool.allocate: Created new tensor, "
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
            f"OffloadTensorPool.free: shape={shape}, dtype={dtype}, "
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
        debug_rank("OffloadTensorPool: Resetting pool...")

        for pool_key, pool in self._pools.items():
            # Clear and refill the free queue
            pool['free'].clear()
            for tensor in pool['all']:
                pool['free'].append(tensor)
            pool['allocated_count'] = 0

        self._stats['current_in_use'] = 0
        debug_rank("OffloadTensorPool: Reset complete")

    def clear(self):
        """Clear the pool and release all GPU memory."""
        debug_rank("OffloadTensorPool: Clearing pool...")

        for pool_key, pool in self._pools.items():
            # Clear all references, allowing PyTorch GC to reclaim memory
            pool['free'].clear()
            pool['all'].clear()

        self._pools.clear()
        self._stats['current_in_use'] = 0

        # Trigger GPU cache cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        debug_rank("OffloadTensorPool: Clear complete")

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
        # Shapes of tensors for MoE activation offload groups are not known in advance,
        # so we do not use CPU pool for them.
        if name in ("expert_fc1", "moe_act", "fused_group_mlp"):
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


class LocalCudaGraphOffloadGroup:
    """Fixed-address VMM slots for one local-graph activation group."""

    def __init__(self, name: str, min_tensor_size: int):
        self.name = name
        self.min_tensor_size = min_tensor_size
        self.descriptors = []
        self.allocations = []
        self.device_tensors = []
        self.host_tensors = []
        self.capture_index = 0
        self.capture_slots = []
        self.consumer_slots = []
        self.capture_epoch = 0
        self.capture_group_index = None
        self.state = "discovering"

        self.d2h_event = torch.cuda.Event()
        self.h2d_event = torch.cuda.Event()
        # Async release/remap contexts returned by the per-slot host functions.
        self.release_context = None
        self.release_contexts = []
        self.remap_context = None
        self.slot_bank = None
        self.resident = False
        # Build-time ownership of the resident bank.  The backward graph must
        # restore the values captured by this exact forward runner, not merely
        # whichever runner most recently used the same ping-pong bank.
        self.capture_forward_runner = None
        self.capture_snapshot_event = None

    def _eligible(self, tensor):
        return (
            torch.is_tensor(tensor)
            and not isinstance(tensor, torch.nn.Parameter)
            and tensor.device.type == "musa"
            and tensor.numel() >= self.min_tensor_size
            and not _te_do_not_offload(tensor)
            and not getattr(tensor, "_do_not_offload", False)
        )

    @staticmethod
    def _descriptor(tensor):
        return (
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
            tensor.device,
        )

    def discover_tensor(self, tensor):
        """Record an eligible saved-tensor descriptor without changing eager autograd."""
        if self._eligible(tensor):
            self.descriptors.append(self._descriptor(tensor))
        return tensor

    def load_tensor(self, tensor):
        """Record the activation slot consumed by backward replay."""
        if self._eligible(tensor):
            slot = next(
                (item for item in self.device_tensors if item.data_ptr() == tensor.data_ptr()),
                None,
            )
            if slot is not None:
                slot_index = self.device_tensors.index(slot)
                self.consumer_slots.append(slot_index)
                context = PipelineOffloadManager.get_instance()._local_graph_consumer_context
                if context is not None:
                    context["slots"].append({
                        "group_index": getattr(self, "capture_group_index", None),
                        "slot_index": slot_index,
                        "vmm_address": tensor.data_ptr(),
                    })
                runner = getattr(self, "capture_runner", None)
                PipelineOffloadManager._local_graph_debug(
                    f"consume rank={torch.distributed.get_rank() if torch.distributed.is_initialized() else -1} "
                    f"runner_id={id(runner) if runner is not None else None} "
                    f"group_index={getattr(self, 'capture_group_index', None)} "
                    f"group={self.name} slot={slot_index} "
                    f"vmm_va={hex(tensor.data_ptr())} epoch={self.capture_epoch}"
                )
        return tensor

    def allocate(self, manager=None, slot_bank=None, group_index=None):
        """Allocate fixed-address slots, optionally from a resident shared bank."""
        from transformer_engine.pytorch.vmm_activation import MUSAActivationVMMAllocation

        assert self.state == "discovering"
        self.slot_bank = slot_bank
        self.resident = slot_bank is not None
        for tensor_index, (shape, stride, dtype, device) in enumerate(self.descriptors):
            if self.resident:
                allocation = manager.local_graph_resident_allocation(
                    slot_bank,
                    self.name,
                    group_index,
                    tensor_index,
                    shape,
                    stride,
                    dtype,
                    device,
                )
            else:
                allocation = MUSAActivationVMMAllocation(shape, stride, dtype, device)
            self.allocations.append(allocation)
            allocation.set_slot_id(
                f"rank={torch.distributed.get_rank() if torch.distributed.is_initialized() else -1}/"
                f"group={group_index if group_index is not None else 'na'}/slot={tensor_index}/"
                f"va=0x{allocation.address:x}"
            )
            self.device_tensors.append(allocation.tensor)
            # Host storage remains runner-local even when the GPU slot is shared.
            self.host_tensors.append(
                torch.empty_strided(shape, stride, dtype=dtype, device="cpu", pin_memory=True)
            )
        self.state = "mapped"

    def begin_capture(self):
        """Reset ordered descriptor consumption before forward graph capture."""
        assert self.state == "mapped"
        self.capture_index = 0
        self.capture_slots = []
        self.consumer_slots = []
        self.capture_epoch += 1

    def save_tensor(self, tensor):
        """Capture a D2D write and save the matching VMM tensor for backward."""
        if not self._eligible(tensor):
            return tensor
        if self.capture_index >= len(self.descriptors):
            raise RuntimeError(f"{self.name}: capture saved more tensors than discovery")
        descriptor = self._descriptor(tensor)
        expected = self.descriptors[self.capture_index]
        if descriptor != expected:
            raise RuntimeError(
                f"{self.name}: activation descriptor drift at slot {self.capture_index}: "
                f"expected={expected}, actual={descriptor}"
            )
        slot_tensor = self.device_tensors[self.capture_index]
        slot_tensor.copy_(tensor)
        self.capture_slots.append(
            {
                "slot_index": self.capture_index,
                "descriptor": descriptor,
                "vmm_address": int(slot_tensor.data_ptr()),
                "aligned_bytes": self.allocations[self.capture_index].aligned_bytes,
            }
        )
        PipelineOffloadManager._local_graph_debug(
            f"capture group={self.name} slot={self.capture_index} "
            f"source={hex(tensor.data_ptr())} vmm_va={hex(slot_tensor.data_ptr())} "
            f"shape={tuple(tensor.shape)} stride={tuple(tensor.stride())} dtype={tensor.dtype} "
            f"aligned_bytes={self.allocations[self.capture_index].aligned_bytes}"
        )
        self.capture_index += 1
        return slot_tensor

    def finish_capture(self):
        """Ensure capture consumed exactly the descriptors discovered during warmup."""
        if self.capture_index != len(self.descriptors):
            raise RuntimeError(
                f"{self.name}: capture consumed {self.capture_index}/{len(self.descriptors)} slots"
            )

    @property
    def logical_bytes(self):
        return sum(tensor.numel() * tensor.element_size() for tensor in self.device_tensors)

    @property
    def physical_bytes(self):
        return sum(allocation.aligned_bytes for allocation in self.allocations)

    def _debug_slot_state(self, stage):
        """Log VMM and tensor storage state at resident-slot checkpoints."""
        if os.getenv("MEGATRON_LOCAL_GRAPH_OFFLOAD_DEBUG") != "1":
            return
        slots = []
        for index, (allocation, tensor, host_tensor) in enumerate(
            zip(self.allocations, self.device_tensors, self.host_tensors)
        ):
            try:
                device_storage_bytes = tensor.untyped_storage().nbytes()
            except Exception:  # pragma: no cover - backend diagnostic only
                device_storage_bytes = "unavailable"
            try:
                allocation_info = allocation.info()
            except Exception as exc:  # pragma: no cover - backend diagnostic only
                allocation_info = f"error={type(exc).__name__}: {exc}"
            slots.append(
                f"slot={index} device_shape={tuple(tensor.shape)} "
                f"device_numel={tensor.numel()} device_ptr={hex(tensor.data_ptr())} "
                f"device_storage_bytes={device_storage_bytes} "
                f"host_shape={tuple(host_tensor.shape)} host_numel={host_tensor.numel()} "
                f"allocation={allocation_info}"
            )
        PipelineOffloadManager._local_graph_debug(
            f"checkpoint={stage} group={self.name} bank={self.slot_bank} state={self.state} "
            + " | ".join(slots)
        )

    def rebind_resident(self, manager, slot_bank, group_index):
        """Replace provisional capture allocations with a shared resident bank."""
        if self.resident:
            return
        from transformer_engine.pytorch.vmm_activation import MUSAActivationVMMAllocation

        old_allocations = self.allocations
        self.allocations = []
        self.device_tensors = []
        self.resident = True
        self.slot_bank = slot_bank
        for tensor_index, (shape, stride, dtype, device) in enumerate(self.descriptors):
            allocation = manager.local_graph_resident_allocation(
                slot_bank, self.name, group_index, tensor_index, shape, stride, dtype, device
            )
            self.allocations.append(allocation)
            allocation.set_slot_id(
                f"rank={torch.distributed.get_rank() if torch.distributed.is_initialized() else -1}/"
                f"group={group_index if group_index is not None else 'na'}/slot={tensor_index}/"
                f"va=0x{allocation.address:x}"
            )
            self.device_tensors.append(allocation.tensor)
        self._debug_slot_state("after_resident_rebind")
        for allocation in old_allocations:
            allocation.close()

    def capture_snapshot(self, runner=None):
        """Preserve values and ownership for this runner's later bwd capture."""
        if not self.resident:
            return
        self._debug_slot_state("before_capture_snapshot")
        # This copy is intentionally synchronous: it establishes the host
        # snapshot before another forward is allowed to reuse the bank.
        for host_tensor, device_tensor in zip(self.host_tensors, self.device_tensors):
            host_tensor.copy_(device_tensor, non_blocking=False)
        self.capture_forward_runner = runner
        # Keep an explicit stream dependency even on backends where the
        # synchronous copy currently happens to drain the stream.  MUSA graph
        # capture and the caller's stream are not assumed to be identical.
        self.capture_snapshot_event = torch.cuda.Event()
        self.capture_snapshot_event.record(torch.cuda.current_stream())
        self._debug_slot_state("after_capture_snapshot")

    def capture_restore(self, forward_runner=None):
        """Restore the exact forward snapshot before capturing its bwd graph."""
        if not self.resident:
            return
        if self.capture_forward_runner is not forward_runner:
            raise RuntimeError(
                f"{self.name}: backward capture is bound to the wrong forward runner "
                f"(expected={id(self.capture_forward_runner)}, actual={id(forward_runner)})"
            )
        if self.capture_snapshot_event is not None:
            torch.cuda.current_stream().wait_event(self.capture_snapshot_event)
        self._debug_slot_state("before_capture_restore")
        for device_tensor, host_tensor in zip(self.device_tensors, self.host_tensors):
            device_tensor.copy_(host_tensor, non_blocking=False)
        self._debug_slot_state("after_capture_restore")

    def enqueue_resident_d2h(self, d2h_stream, compute_stream):
        """Save a resident slot without releasing its fixed mapping."""
        if self.state != "mapped":
            raise RuntimeError(f"{self.name}: cannot save resident slots in state {self.state}")
        d2h_stream.wait_stream(compute_stream)
        with torch.cuda.stream(d2h_stream):
            for host_tensor, device_tensor in zip(self.host_tensors, self.device_tensors):
                host_tensor.copy_(device_tensor, non_blocking=True)
        self.state = "resident_d2h_pending"

    def enqueue_resident_h2d(self, h2d_stream):
        """Restore host values into an always-mapped resident slot."""
        if self.state not in ("resident_d2h_pending", "mapped"):
            raise RuntimeError(f"{self.name}: cannot reload resident slots in state {self.state}")
        with torch.cuda.stream(h2d_stream):
            for device_tensor, host_tensor in zip(self.device_tensors, self.host_tensors):
                device_tensor.copy_(host_tensor, non_blocking=True)
        self.state = "resident_reload_pending"

    def adopt_resident_reload(self):
        """Mark a resident reload consumable after its stream wait is installed."""
        if self.state != "resident_reload_pending":
            raise RuntimeError(f"{self.name}: resident reload is in state {self.state}")
        self.state = "mapped"

    def enqueue_d2h(
        self, d2h_stream, compute_stream, release_stream=None, graph_done_event=None
    ):
        """Queue this group's D2H copies and hand off release to a stream host func.

        The copies run on ``d2h_stream`` (ordered behind the producing compute
        via ``wait_stream``) and an event records the burst's completion. The
        release host func is gated ONLY on that event: it is enqueued on
        ``release_stream`` after a ``wait_event(d2h_event)``, so the worker
        unmaps this group as soon as ITS OWN copies finish - not when every
        later group's D2H on the shared stream also completes. Driver calls
        still run on the resident worker thread, not the callback thread.
        """
        if not self.device_tensors:
            return
        if self.state != "mapped":
            raise RuntimeError(f"{self.name}: cannot offload slots in state {self.state}")
        if os.getenv("MEGATRON_LOCAL_GRAPH_OFFLOAD_DEBUG") == "1":
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            print(
                f"[vmm-offload][rank={rank}] submit_d2h group={self.name} "
                f"state={self.state} allocations={len(self.allocations)} "
                f"bytes={self.logical_bytes} compute_stream={getattr(compute_stream, 'musa_stream', None)} "
                f"d2h_stream={getattr(d2h_stream, 'musa_stream', None)} "
                f"release_stream={getattr(release_stream, 'musa_stream', None)}",
                flush=True,
            )
        # Event-gated per-group release: fall back to the old shared-stream
        # placement only when no dedicated release stream was provided.
        if release_stream is not None:
            with torch.cuda.stream(d2h_stream):
                if graph_done_event is not None:
                    d2h_stream.wait_event(graph_done_event)
                else:
                    d2h_stream.wait_stream(compute_stream)
                previous_slot_event = None
                for slot_index, (host_tensor, device_tensor) in enumerate(
                    zip(self.host_tensors, self.device_tensors)
                ):
                    if previous_slot_event is not None:
                        d2h_stream.wait_event(previous_slot_event)
                    host_tensor.copy_(device_tensor, non_blocking=True)
                    slot_done = torch.cuda.Event()
                    slot_done.record(d2h_stream)
                    previous_slot_event = slot_done
                    PipelineOffloadManager._local_graph_debug(
                        f"d2h_slot_submit ts_ns={time.time_ns()} "
                        f"group={self.name} slot={slot_index}"
                    )
                self.d2h_event.record(d2h_stream)
            with torch.cuda.stream(release_stream):
                release_stream.wait_event(self.d2h_event)
                self.release_contexts = []
                for allocation in self.allocations:
                    self.release_contexts.append(allocation.release_hook_after(release_stream))
                self.release_context = self.release_contexts[-1] if self.release_contexts else None
        else:
            d2h_stream.wait_stream(compute_stream)
            with torch.cuda.stream(d2h_stream):
                for host_tensor, device_tensor in zip(self.host_tensors, self.device_tensors):
                    host_tensor.copy_(device_tensor, non_blocking=True)
                self.d2h_event.record(d2h_stream)
                self.release_contexts = []
                for allocation in self.allocations:
                    self.release_contexts.append(allocation.release_hook_after(d2h_stream))
                self.release_context = self.release_contexts[-1] if self.release_contexts else None
        self.state = "d2h_pending"

    def try_release(self):
        """Adopt the worker's release if it already finished; never blocks."""
        if self.state != "d2h_pending":
            return False
        if not self._release_context_done():
            return False
        return self._adopt_released()

    def wait_and_release(self):
        """Wait for the async release then adopt the unmapped state; backward time."""
        # Already released by a forward-time drain: nothing to wait for.
        if self.state == "unmapped":
            return False
        if self.state != "d2h_pending":
            raise RuntimeError(f"{self.name}: cannot release slots in state {self.state}")
        self._wait_release_context()
        return self._adopt_released()

    def _release_context_done(self):
        """Whether every per-slot release request finished on the worker."""
        contexts = self.release_contexts
        if contexts:
            return all(dict(context.status())["done"] == 1 for context in contexts)
        if self.release_context is None:
            return all(allocation.async_release_done() for allocation in self.allocations)
        return dict(self.release_context.status())["done"] == 1

    def _wait_release_context(self):
        """Block until every per-slot release request finishes."""
        if self.release_contexts:
            for context in self.release_contexts:
                status = dict(context.status())
                if status["done"] != 1:
                    # The allocation owns the context and provides the blocking
                    # compatibility API without synchronizing a MUSA stream.
                    pass
        for allocation in self.allocations:
            allocation.wait_for_async_release()

    def _adopt_released(self):
        """Apply the worker-completed release to this group's bookkeeping."""
        contexts = self.release_contexts
        self.release_contexts = []
        if contexts:
            for context in contexts:
                status = dict(context.status())
                if status["error"]:
                    raise RuntimeError(
                        f"{self.name}: async release failed on worker: MUresult {status['error_code']}"
                    )
        elif self.release_context is not None:
            status = dict(self.release_context.status())
            if status["error"]:
                raise RuntimeError(
                    f"{self.name}: async release failed on worker: MUresult {status['error_code']}"
                )
        self.release_context = None
        for allocation in self.allocations:
            if not allocation.async_release_done():
                allocation.wait_for_async_release()
        self.state = "unmapped"
        PipelineOffloadManager._local_graph_debug(
            f"release group={self.name} logical_bytes={self.logical_bytes} "
            f"physical_bytes={self.physical_bytes}"
        )
        return True
    def prepare_remap(self, remap_context):
        """Bind a slot request context to this group's bookkeeping."""
        if not self.device_tensors:
            return False
        if self.state not in ("d2h_pending", "unmapped", "remap_pending"):
            raise RuntimeError(f"{self.name}: cannot prepare reload in state {self.state}")
        self.remap_context = remap_context
        self.state = "remap_pending"
        return True

    def adopt_reload_submission(self):
        """Compatibility adoption for callers submitting a complete context."""
        if self.state != "remap_pending":
            raise RuntimeError(f"{self.name}: cannot adopt reload in state {self.state}")
        if self.remap_context is not None:
            for allocation in self.allocations:
                allocation.adopt_async_remap()
            status = dict(self.remap_context.status())
            if status["error"]:
                raise RuntimeError(
                    f"{self.name}: async reload failed on worker: error {status['error_code']}"
                )
        self.remap_context = None
        self.release_context = None
        self.state = "mapped"
        return True


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
        # TE CUDA graph offload paths need a stream/event pair that lives outside
        # individual layer objects so capture, replay, and backward hooks order
        # the same D2H/H2D work with the same synchronization primitives.
        self._cuda_graph_stream = torch.cuda.Stream()
        self._cuda_graph_event = torch.cuda.Event(external=True)
        # Shared CPU tensor pool for all chunks to improve reuse efficiency
        self._cpu_tensor_pool = OffloadTensorPool(device="cpu", pin_memory=True)

        # Whether the manager is in warmup phase.
        self._is_warmup = True
        # Whether the manager is in CUDA graph replay phase.
        self._in_replay = False
        # Cache OffloadChunkHandler objects for each virtual pipeline stage and each forward pass.
        self._cached_chunks_forward = []
        # Cache OffloadChunkHandler objects for each virtual pipeline stage and each backward pass.
        self._cached_chunks_backward = []
        # Index of the current backward chunk in the cached chunks backward.
        self._cached_chunks_index_backward = 0
        # Index of the current forward chunk in the cached chunks forward.
        self._cached_chunks_index_forward = 0

        # Whole-layer local CUDA graph capture records fixed activation slots here. These are
        # separate from eager chunk state because schedule reset runs before graph creation.
        self._local_graph_capture_runner = None
        self._local_graph_mode = None
        self._local_graph_capture_groups = []
        self._local_graph_capture_group = None
        self._local_graph_group_index = 0
        self._local_graph_saved_tensors_hooks = None
        self._local_graph_d2h_bytes = 0
        self._local_graph_h2d_bytes = 0
        self._local_graph_resident_slots = [{}, {}]
        self._local_graph_bank_d2h_events = [torch.cuda.Event(), torch.cuda.Event()]
        self._local_graph_bank_d2h_recorded = [False, False]
        self._local_graph_bank_consumed_events = [None, None]
        # Dedicated stream for the graph-level release host func. Putting the
        # host func on the shared d2h stream would serialize the next graph's
        # D2H copies behind this graph's unmap callback.
        self._local_graph_release_stream = torch.cuda.Stream()
        self._local_graph_consumer_context = None

        self.do_offload = True

        # Do not offload the last X groups so that the reloading won't block the computing stream.
        self._offload_margin = 0
        # Sometimes we need to delay the offloading and launch it later.
        # The delayed offload groups are stored in a queue.
        self._delayed_offload_groups = []
        self.reset()

        # Keep the hook context object around so each offload scope can enter/exit
        # the same autograd saved-tensor hooks without touching private torch APIs.
        self._saved_tensors_hooks = saved_tensors_hooks(
            self.on_save_for_backward, self.on_get_saved_tensor
        )

    @property
    def d2h_stream(self):
        """Get the device-to-host (GPU to CPU) transfer stream."""
        return self._d2h_stream

    @property
    def h2d_stream(self):
        """Get the host-to-device (CPU to GPU) transfer stream."""
        return self._h2d_stream

    @property
    def cuda_graph_stream(self):
        """Get the CUDA graph stream."""
        return self._cuda_graph_stream

    @property
    def cuda_graph_event(self):
        """Get the CUDA graph event."""
        return self._cuda_graph_event

    @property
    def cpu_tensor_pool(self):
        """Get the shared CPU tensor pool."""
        return self._cpu_tensor_pool

    @property
    def in_local_graph_capture(self):
        """Whether a whole-layer local graph is currently recording activation slots."""
        return self._local_graph_capture_runner is not None

    @staticmethod
    def _local_graph_debug(message):
        if os.getenv("MEGATRON_LOCAL_GRAPH_OFFLOAD_DEBUG") != "1":
            return
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
        print(f"[local-full-graph-offload][rank={rank}] {message}", flush=True)

    @staticmethod
    def _local_graph_group_bytes(groups):
        return sum(group.logical_bytes for group in groups)

    @staticmethod
    def _local_graph_ping_pong_enabled(runner=None):
        """Whether this runner uses resident slots for local Graph CPU offload."""
        if runner is None:
            return False
        enabled = getattr(runner, "local_graph_slot_bank", None) is not None
        if enabled and not getattr(runner, "local_graph_offload_groups", []):
            # During discovery, groups are not attached until allocation completes.
            enabled = True
        return enabled

    def prepare_local_graph_ping_pong(self, runner, slot_bank):
        """Rebind provisional runner slots to the selected resident bank."""
        runner.local_graph_slot_bank = slot_bank
        for group_index, group in enumerate(runner.local_graph_offload_groups):
            group.rebind_resident(self, slot_bank, group_index)

    def local_graph_resident_allocation(
        self, bank, name, group_index, tensor_index, shape, stride, dtype, device
    ):
        """Get or create one descriptor-compatible allocation in a resident bank."""
        from transformer_engine.pytorch.vmm_activation import MUSAActivationVMMAllocation

        key = (name, group_index, tensor_index, tuple(shape), tuple(stride), dtype, device)
        allocations = self._local_graph_resident_slots[bank]
        if key not in allocations:
            allocations[key] = MUSAActivationVMMAllocation(shape, stride, dtype, device)
        return allocations[key]

    def local_graph_capture_snapshot(self, runner):
        """Preserve the values belonging to this forward runner."""
        if not self._local_graph_ping_pong_enabled(runner):
            return
        for group in runner.local_graph_offload_groups:
            group.capture_snapshot(runner)
        # The next runner may reuse this bank immediately during its warmup.
        # Drain every device stream here, not just the current stream, so a
        # side-stream producer cannot still touch the old bank while the next
        # graph is being prepared.
        torch.cuda.synchronize()

    def local_graph_capture_restore(self, runner, forward_runner=None):
        """Restore the forward runner explicitly bound to this bwd runner."""
        if not self._local_graph_ping_pong_enabled(runner):
            return
        if forward_runner is None:
            forward_runner = runner
        if forward_runner is not runner:
            raise RuntimeError("local graph bwd runner must use its own forward snapshot")
        for group in runner.local_graph_offload_groups:
            group.capture_restore(forward_runner)

    def begin_local_graph_discovery(self, runner):
        """Discover saved activation descriptors during the final eager warmup."""
        assert not self.in_local_graph_capture, "Nested local graph offload discovery"
        self._local_graph_capture_runner = runner
        self._local_graph_mode = "discovery"
        self._local_graph_capture_groups = []
        self._local_graph_capture_group = None
        self._local_graph_group_index = 0

    def end_local_graph_discovery(self, success=True):
        """Allocate fixed-address slots after descriptor discovery."""
        runner = self._local_graph_capture_runner
        if runner is None or self._local_graph_mode != "discovery":
            return
        assert self._local_graph_capture_group is None, "Unclosed discovery group"
        groups = list(self._local_graph_capture_groups)
        if success:
            slot_bank = (
                runner.local_graph_slot_bank
                if self._local_graph_ping_pong_enabled(runner)
                else None
            )
            for group_index, group in enumerate(groups):
                group.allocate(self, slot_bank, group_index)
            runner.local_graph_offload_groups = groups
            self._local_graph_debug(
                f"discovered groups={len(groups)} logical_bytes={self._local_graph_group_bytes(groups)} "
                f"physical_bytes={sum(group.physical_bytes for group in groups)}"
            )
        else:
            runner.local_graph_offload_groups = []
        self._local_graph_capture_runner = None
        self._local_graph_mode = None
        self._local_graph_capture_groups = []
        self._local_graph_group_index = 0

    def begin_local_graph_capture(self, runner):
        """Bind graph capture to slots allocated during descriptor discovery."""
        assert not self.in_local_graph_capture, "Nested local CUDA graph offload capture"
        groups = getattr(runner, "local_graph_offload_groups", [])
        self._local_graph_capture_runner = runner
        self._local_graph_mode = "capture"
        self._local_graph_capture_groups = groups
        self._local_graph_group_index = 0
        runner.local_graph_capture_epoch = getattr(runner, "local_graph_capture_epoch", 0) + 1
        runner.local_graph_capture_metadata = {
            "epoch": runner.local_graph_capture_epoch,
            "rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else -1,
            "runner_id": id(runner),
            "groups": [],
            "consumer_slots": [],
        }
        for group_index, group in enumerate(groups):
            group.capture_runner = runner
            group.capture_group_index = group_index
        self._local_graph_capture_group = None

    def end_local_graph_capture(self, success=True):
        """Validate capture-time descriptor consumption and retain runner slots."""
        runner = self._local_graph_capture_runner
        if runner is None or self._local_graph_mode != "capture":
            return
        assert self._local_graph_capture_group is None, "Unclosed local graph offload group"
        groups = self._local_graph_capture_groups
        if success:
            if self._local_graph_group_index != len(groups):
                raise RuntimeError(
                    f"local graph capture consumed {self._local_graph_group_index}/{len(groups)} groups"
                )
            for group_index, group in enumerate(groups):
                group.finish_capture()
                metadata = getattr(runner, "local_graph_capture_metadata", None)
                if metadata is not None:
                    metadata["groups"].append(
                        {
                            "group_index": group_index,
                            "name": group.name,
                            "slots": list(group.capture_slots),
                        }
                    )
            self._local_graph_debug(
                f"captured groups={len(groups)} bytes={self._local_graph_group_bytes(groups)} "
                f"consumer_slots={runner.local_graph_capture_metadata['consumer_slots']}"
            )
        else:
            runner.local_graph_offload_groups = []
        self._local_graph_capture_runner = None
        self._local_graph_mode = None
        self._local_graph_capture_groups = []
        self._local_graph_group_index = 0

    def local_graph_group_start(self, name, min_tensor_size):
        """Open saved-tensor hooks for one discovery or capture group."""
        assert self.in_local_graph_capture
        assert self._local_graph_capture_group is None, "Overlapping local graph offload groups"
        if self._local_graph_mode == "discovery":
            group = LocalCudaGraphOffloadGroup(name, min_tensor_size)
            pack_hook = group.discover_tensor
        else:
            if self._local_graph_group_index >= len(self._local_graph_capture_groups):
                raise RuntimeError(f"capture produced unexpected activation group {name}")
            group = self._local_graph_capture_groups[self._local_graph_group_index]
            if group.name != name or group.min_tensor_size != min_tensor_size:
                raise RuntimeError(
                    f"local graph activation group drift: expected={group.name}, actual={name}"
                )
            pack_hook = group.save_tensor
        self._local_graph_capture_group = group
        self._local_graph_saved_tensors_hooks = saved_tensors_hooks(pack_hook, group.load_tensor)
        self._local_graph_saved_tensors_hooks.__enter__()

    def local_graph_group_context_exit(self):
        """Close the current group's saved-tensor hooks."""
        if self._local_graph_saved_tensors_hooks is not None:
            self._local_graph_saved_tensors_hooks.__exit__(None, None, None)
            self._local_graph_saved_tensors_hooks = None

    def local_graph_group_commit(self, name):
        """Commit one ordered discovery or capture group."""
        group = self._local_graph_capture_group
        assert group is not None and group.name == name
        assert self._local_graph_saved_tensors_hooks is None, "Offload context must exit before commit"
        if self._local_graph_mode == "discovery":
            self._local_graph_capture_groups.append(group)
        else:
            group.finish_capture()
        self._local_graph_group_index += 1
        self._local_graph_capture_group = None

    def local_graph_forward_wait_ready(self, runner, replay_stream):
        """Fence reuse of a resident bank until its preceding D2H finishes."""
        if not self._local_graph_ping_pong_enabled(runner):
            return False
        bank = runner.local_graph_slot_bank
        if self._local_graph_bank_d2h_recorded[bank]:
            replay_stream.wait_event(self._local_graph_bank_d2h_events[bank])
        consumed_event = self._local_graph_bank_consumed_events[bank]
        if consumed_event is not None:
            replay_stream.wait_event(consumed_event)
        return True

    def local_graph_forward_replay(self, runner, graph_done_event=None):
        """Enqueue D2H only after the replay completion fence."""
        compute_stream = torch.cuda.current_stream()
        groups = getattr(runner, "local_graph_offload_groups", [])
        active_groups = [group for group in groups if group.device_tensors]
        step_bytes = self._local_graph_group_bytes(groups)
        self._local_graph_d2h_bytes += step_bytes
        if not active_groups:
            self._local_graph_debug(f"d2h bytes={step_bytes} total_d2h={self._local_graph_d2h_bytes}")
            return
        if self._local_graph_ping_pong_enabled(runner):
            bank = runner.local_graph_slot_bank
            for group in active_groups:
                group.enqueue_resident_d2h(self.d2h_stream, compute_stream)
            with torch.cuda.stream(self.d2h_stream):
                self._local_graph_bank_d2h_events[bank].record(self.d2h_stream)
            self._local_graph_bank_d2h_recorded[bank] = True
        else:
            # After this graph's forward replay, offload every slot in every
            # group that belongs to the graph as one D2H burst and one batched
            # release host func. Per-slot host funcs previously unmapped each
            # allocation independently as soon as its own copy finished.
            self._enqueue_graph_groups_offload(
                runner, active_groups, compute_stream, graph_done_event
            )
            self.drain_pending_d2h(runner)

    def _enqueue_graph_groups_offload(self, runner, active_groups, compute_stream, graph_done_event):
        """D2H then batch-release every slot owned by this graph's offload groups."""
        if release_hooks_after is None:
            raise RuntimeError(
                "TE release_hooks_after is required to batch-offload local graph activation slots"
            )
        allocations = []
        for group in active_groups:
            if group.state != "mapped":
                raise RuntimeError(f"{group.name}: cannot offload slots in state {group.state}")
            allocations.extend(group.allocations)
        if not allocations:
            return

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
        self._local_graph_debug(
            f"d2h_submit_begin ts_ns={time.time_ns()} rank={rank} "
            f"runner_id={id(runner)} groups={[group.name for group in active_groups]} "
            f"slots={len(allocations)}"
        )
        if os.getenv("MEGATRON_LOCAL_GRAPH_OFFLOAD_DEBUG") == "1":
            print(
                f"[vmm-offload][rank={rank}] submit_graph_d2h "
                f"runner_id={id(runner)} groups={[group.name for group in active_groups]} "
                f"allocations={len(allocations)} "
                f"compute_stream={getattr(compute_stream, 'musa_stream', None)} "
                f"d2h_stream={getattr(self.d2h_stream, 'musa_stream', None)} "
                f"release_stream={getattr(self._local_graph_release_stream, 'musa_stream', None)}",
                flush=True,
            )

        with torch.cuda.stream(self.d2h_stream):
            if graph_done_event is not None:
                self.d2h_stream.wait_event(graph_done_event)
            else:
                self.d2h_stream.wait_stream(compute_stream)
            for group in active_groups:
                for slot_index, (host_tensor, device_tensor) in enumerate(
                    zip(group.host_tensors, group.device_tensors)
                ):
                    host_tensor.copy_(device_tensor, non_blocking=True)
                    self._local_graph_debug(
                        f"d2h_slot_submit ts_ns={time.time_ns()} "
                        f"group={group.name} slot={slot_index}"
                    )
                group.d2h_event.record(self.d2h_stream)

        with torch.cuda.stream(self._local_graph_release_stream):
            self._local_graph_release_stream.wait_event(active_groups[-1].d2h_event)
            batch_context = release_hooks_after(allocations, self._local_graph_release_stream)

        for group in active_groups:
            group.release_contexts = [batch_context]
            group.release_context = batch_context
            group.state = "d2h_pending"

        self._local_graph_debug(
            f"d2h_submit_return ts_ns={time.time_ns()} rank={rank} "
            f"runner_id={id(runner)} slots={len(allocations)}"
        )


    def drain_pending_d2h(self, runner):
        """Release VMM backing for groups whose D2H already completed; never blocks."""
        groups = getattr(runner, "local_graph_offload_groups", [])
        released = sum(1 for group in groups if group.try_release())
        if released:
            self._local_graph_debug(f"lazy released groups={released}")

    @staticmethod
    def _local_graph_active_groups(runner):
        groups = getattr(runner, "local_graph_offload_groups", [])
        return [group for group in reversed(groups) if group.device_tensors]

    def local_graph_backward_prepare(self, runner, block_until_submitted=True):
        """Submit one runner's resident reload or VMM remap/H2D batch.

        With ``block_until_submitted=True`` (per-runner slot path default) the
        call additionally blocks until the RemapWorker finished the VMM
        transition and submitted the H2D — see the call site in
        ``Graphed.backward`` for why this runs before the predecessor's
        replay. The forward-time priming of the backward-chain head passes
        ``False`` because its release dependency (the forward D2H burst) is
        still in flight there; blocking would serialize the offload copies.
        """
        active_groups = self._local_graph_active_groups(runner)
        if not active_groups:
            return False
        state = getattr(runner, "local_graph_reload_state", None)
        if state == "reload_pending":
            return True
        if state is not None:
            raise RuntimeError(f"cannot prepare local graph runner in reload state {state}")
        if self._local_graph_ping_pong_enabled(runner):
            bank = runner.local_graph_slot_bank
            # Before the first backward use, do not overwrite this bank until
            # its final forward D2H has saved the activation currently in it.
            if self._local_graph_bank_d2h_recorded[bank]:
                self.h2d_stream.wait_event(self._local_graph_bank_d2h_events[bank])
            # Later uses wait until the prior backward graph reading this bank
            # has completed. Adjacent runners use the opposite bank and overlap.
            consumed_event = self._local_graph_bank_consumed_events[bank]
            if consumed_event is not None:
                self.h2d_stream.wait_event(consumed_event)
            for group in active_groups:
                group.enqueue_resident_h2d(self.h2d_stream)
            reload_event = torch.cuda.Event()
            with torch.cuda.stream(self.h2d_stream):
                reload_event.record(self.h2d_stream)
            runner.local_graph_reload_event = reload_event
            runner.local_graph_reload_context = None
        else:
            # Submit in backward consumption order. The last forward-offloaded
            # group has the latest release dependency, so waiting on it first
            # makes every earlier dependency a fast path.
            # Non-ping-pong reload is deliberately deferred until this runner is
            # consumed.  Creating one batch context here used to enqueue every
            # slot early, allowing fast ranks to reload the whole runner long
            # before graph replay.  Keep the slot/group description on the
            # runner; submit each group from local_graph_backward_wait_ready().
            runner.local_graph_reload_context = None
            runner.local_graph_reload_groups = tuple(active_groups)
            runner.local_graph_reload_slot_count = sum(
                len(group.allocations) for group in active_groups
            )
            runner.local_graph_reload_state = "reload_deferred"
        return True

    def local_graph_backward_submit_reload(self, runner):
        """Submit the next runner's H2D requests without gating its compute stream."""
        if self._local_graph_ping_pong_enabled(runner):
            return self.local_graph_backward_prepare(runner)
        if getattr(runner, "local_graph_reload_state", None) == "reload_submitted":
            return True
        if getattr(runner, "local_graph_reload_state", None) is None:
            self.local_graph_backward_prepare(runner)
        if getattr(runner, "local_graph_reload_state", None) != "reload_deferred":
            return False
        active_groups = self._local_graph_active_groups(runner)
        observed = getattr(runner, "local_graph_consumer_slots", ())
        slot_plan = {}
        for group_index, slot_index, vmm_address in observed:
            if group_index >= len(runner.local_graph_offload_groups):
                slot_plan = {}
                break
            group = runner.local_graph_offload_groups[group_index]
            if slot_index >= len(group.allocations) or group.allocations[slot_index].tensor.data_ptr() != vmm_address:
                slot_plan = {}
                break
            slot_plan.setdefault(id(group), set()).add(slot_index)
        valid = bool(slot_plan) and all(id(group) in slot_plan for group in active_groups)
        if not valid:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
            raise RuntimeError(
                f"invalid local graph consumer slots: rank={rank} "
                f"runner_id={id(runner)} observed={observed} "
                f"active_groups={[group.name for group in active_groups]}"
            )
        runner.local_graph_reload_slot_plan = {
            id(group): sorted(slot_plan[id(group)]) for group in active_groups
        }
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
        self._local_graph_debug(
            f"reload_submit_plan rank={rank} runner_id={id(runner)} "
            f"consumer_slots={observed} active_groups="
            f"{[(group.name, id(group)) for group in active_groups]} "
            f"slot_plan={runner.local_graph_reload_slot_plan}"
        )
        runner.local_graph_reload_slots = []
        runner.local_graph_reload_slot_map = {}
        for group_index, slot_index, vmm_address in observed:
            group = runner.local_graph_offload_groups[group_index]
            key = (group_index, slot_index)
            if key in runner.local_graph_reload_slot_map:
                raise RuntimeError(
                    f"duplicate local graph consumer slot: runner_id={id(runner)} "
                    f"group_index={group_index} slot_index={slot_index}"
                )
            runner.local_graph_reload_slot_map[key] = len(runner.local_graph_reload_slots)
            runner.local_graph_reload_slots.append((group_index, slot_index, vmm_address))
        runner.local_graph_reload_contexts = {}
        for group_index, slot_index, _ in runner.local_graph_reload_slots:
            group = runner.local_graph_offload_groups[group_index]
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
            self._local_graph_debug(
                f"reload_submit_slot rank={rank} runner_id={id(runner)} "
                f"position={runner.local_graph_reload_slot_map[(group_index, slot_index)]} "
                f"group_index={group_index} group={group.name} slot={slot_index}"
            )
            allocation = group.allocations[slot_index]
            # Remap-only submission: the worker performs the VMM transition but
            # does NOT submit H2D yet; that happens at replay time so the copy
            # and the graph launch are issued together.
            reload_context = remap_only_slot_after(
                allocation, group.host_tensors[slot_index], self.h2d_stream
            )
            group.prepare_remap(reload_context)
            runner.local_graph_reload_contexts[(group_index, slot_index)] = reload_context
        runner.local_graph_reload_state = "reload_submitted"
        return True


        if not getattr(runner, "local_graph_reload_groups", None):
            return False
        if self._local_graph_ping_pong_enabled(runner):
            return False
        # Each group is adopted immediately after its own request is consumed;
        # there is no batch remap context left to drain here.
        runner.local_graph_reload_context = None
        runner.local_graph_reload_groups = None
        runner.local_graph_reload_event = None
        runner.local_graph_reload_state = None
        return True

    def local_graph_backward_wait_ready(self, runner, compute_stream=None):
        """Wait only for this runner's consumer slots."""
        active_groups = self._local_graph_active_groups(runner)
        if not active_groups:
            return False
        if compute_stream is None:
            compute_stream = torch.cuda.current_stream()
        if self._local_graph_ping_pong_enabled(runner):
            if runner.local_graph_reload_event is None:
                raise RuntimeError("local graph resident reload has no completion event")
            compute_stream.wait_event(runner.local_graph_reload_event)
            for group in active_groups:
                group.adopt_resident_reload()
            runner.local_graph_reload_event = None
            runner.local_graph_reload_state = None
            return True
        if wait_remap_slot_on_stream is None or launch_remap_slot_h2d is None:
            raise RuntimeError("TE per-slot remap wait API is unavailable")
        if getattr(runner, "local_graph_reload_state", None) != "reload_submitted":
            self.local_graph_backward_submit_reload(runner)
        contexts = getattr(runner, "local_graph_reload_contexts", {})
        reload_slots = getattr(runner, "local_graph_reload_slots", ())
        # Two-position H2D pipeline: runner K's copies gate on the completion
        # event of the graph two execution-positions earlier (K+2), never on
        # the immediate predecessor.  By Backward(N)'s replay moment the gate
        # (Backward(N+2)) has long completed, so slot(N)'s H2D overlaps the
        # still-running Backward(N+1) instead of serializing behind it.
        previous_runner = getattr(runner, "prev_bwd_runner", None)
        pre_previous_runner = getattr(previous_runner, "prev_bwd_runner", None) if previous_runner is not None else None
        gate_event = (
            getattr(pre_previous_runner, "bwd_graph_replay_complete_event", None)
            if pre_previous_runner is not None and not self._local_graph_ping_pong_enabled(runner)
            else None
        )
        if gate_event is not None:
            self.h2d_stream.wait_event(gate_event)
            self._local_graph_debug(
                f"h2d_gate rank={torch.distributed.get_rank() if torch.distributed.is_initialized() else -1} "
                f"runner_id={id(runner)} gate_runner_id={id(pre_previous_runner)}"
            )
        for group_index, slot_index, _ in reload_slots:
            group = runner.local_graph_offload_groups[group_index]
            context = contexts.get((group_index, slot_index))
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
            h2d_begin = time.time_ns()
            # The remap worker only performs the VMM transition.  Submit the
            # H2D from this training thread so the memcpy is ordered with the
            # replay that consumes this slot.  launch_remap_slot_h2d waits for
            # the worker's remap completion before issuing musaMemcpyAsync on
            # h2d_stream; the returned slot event is then propagated to both
            # streams before the allocation is adopted.
            launch_remap_slot_h2d(context, 0, self.h2d_stream)
            wait_remap_slot_on_stream(context, 0, self.h2d_stream)
            wait_remap_slot_on_stream(context, 0, compute_stream)
            self._local_graph_debug(
                f"h2d_launch rank={rank} runner_id={id(runner)} "
                f"position={runner.local_graph_reload_slot_map.get((group_index, slot_index), -1)} "
                f"group_index={group_index} group={group.name} slot={slot_index} "
                f"h2d_submit_ns={time.time_ns() - h2d_begin}"
            )
            group.allocations[slot_index].adopt_async_remap()
        for group in active_groups:
            group.state = "mapped"
        runner.local_graph_reload_state = None
        runner.local_graph_reload_contexts = None
        runner.local_graph_reload_slots = None
        runner.local_graph_reload_slot_map = None
        runner.local_graph_reload_groups = None
        return True

    def begin_local_graph_consumer_capture(self, runner):
        """Collect saved-tensor slots while capturing one backward graph."""
        if self._local_graph_consumer_context is not None:
            raise RuntimeError("Nested local graph consumer capture")
        self._local_graph_consumer_context = {"runner": runner, "slots": []}

    def end_local_graph_consumer_capture(self, runner, success=True):
        """Publish the consumer batch captured by one backward graph."""
        context = self._local_graph_consumer_context
        self._local_graph_consumer_context = None
        if context is None:
            return
        if success:
            runner.local_graph_consumer_slots = tuple(
                (item["group_index"], item["slot_index"], item["vmm_address"])
                for item in context["slots"]
            )
            self._local_graph_debug(
                f"consumer_batch rank={torch.distributed.get_rank() if torch.distributed.is_initialized() else -1} "
                f"runner_id={id(runner)} slots={runner.local_graph_consumer_slots}"
            )

    def local_graph_backward_mark_consumed(self, runner, consumed_event):
        """Publish when a resident bank may be overwritten by a later reload."""
        if not self._local_graph_ping_pong_enabled(runner):
            return False
        self._local_graph_bank_consumed_events[runner.local_graph_slot_bank] = consumed_event
        return True

    def local_graph_backward_replay(self, runner):
        """Compatibility fallback for callers without look-ahead scheduling."""
        self.local_graph_backward_wait_ready(runner)

    def push_offload_groups(self, group_hook, name, forced_released_tensors):
        """Push the offload groups to the delayed queue."""
        debug_rank(f"pushing offload groups to the delayed queue")
        # Store the group name because delayed CUDA graph replay flushes later,
        # after the original group-start site has already moved on.
        self._delayed_offload_groups.append((group_hook, name, forced_released_tensors))

    def flush_delayed_groups(self):
        """Flush the delayed groups."""
        debug_rank("flushing delayed groups")
        # Preserve the original forward commit order; reload scheduling still
        # relies on the same group order discovered during warmup.
        for group_hook, name, forced_released_tensors in self._delayed_offload_groups:
            group_hook(name, forced_released_tensors)
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
        assert self._offload_margin == 0, "Offload margin is not 0"
        # Disable the groups to meet the delta offload bytes across PP ranks.
        keep_on_gpu_bytes = self._pp_rank * self._delta_offload_bytes_across_pp_ranks
        for chunk in self._cached_chunks_backward:
            for group in chunk.offload_groups:
                if group.offload and keep_on_gpu_bytes > 0:
                    debug_rank(
                        f"group {group._name} offload {group.offload} \
                        keep_on_gpu_bytes {keep_on_gpu_bytes}"
                    )
                    keep_on_gpu_bytes -= group.total_offload_bytes
                    group.offload = False
        # Disable the later groups to meet the activation offload fraction.
        for chunk in self._cached_chunks_backward:
            eligible_offload_groups = [
                group
                for group in chunk.offload_groups
                if group.offload and group.total_offload_bytes > 0
            ]
            offloaded_groups_count = len(eligible_offload_groups)
            disabled_groups_count = int(
                offloaded_groups_count * (1 - self._activation_offload_fraction)
            )
            debug_rank(f"Disabled {disabled_groups_count}/{offloaded_groups_count} groups")
            # Prefer keeping earlier forward groups offloaded because releasing
            # those activations sooner gives the longest memory-pressure relief.
            for group in reversed(eligible_offload_groups):
                if disabled_groups_count > 0:
                    disabled_groups_count -= 1
                    group.offload = False
                else:
                    break
        # Dump the offload information
        total_tensor_count = {}
        total_offload_bytes = {}
        for chunk in self._cached_chunks_forward:
            for group in chunk.offload_groups:
                debug_rank(f"chunk {chunk} group {group} offload {group.offload}")
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
        debug_rank(f"total_tensor_count {total_tensor_count}")
        debug_rank(f"total_offload_bytes {total_offload_bytes}")
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
        self,
        pp_rank,
        vp_size,
        vp_stage,
        min_offloaded_tensor_size=1024 * 1024,
        delta_offload_bytes_across_pp_ranks=0,
        activation_offload_fraction: float = 1.0,
        max_inflight_offloads: Optional[int] = None,
    ):
        """
        Initialize a chunk offload handler for a model chunk (microbatch).

        Args:
            pp_rank: Pipeline parallel rank
            vp_size: Virtual pipeline size
            vp_stage: Virtual pipeline stage index (None means stage 0)
            min_offloaded_tensor_size: Minimum tensor size (in elements) to offload
            delta_offload_bytes_across_pp_ranks:
                Difference of offload bytes across PP ranks to balance the offload load.
            activation_offload_fraction: Fraction of eligible groups to offload, in range [0, 1].
            max_inflight_offloads: If set, cap pending offloads per group name before main
                wait_event; see ``fine_grained_offloading_max_inflight_offloads`` on
                ``TransformerConfig``.
        """
        if not self._is_warmup:
            return

        vp_size = 1 if vp_size is None else vp_size
        if self._stages is None:
            self._vpp = vp_size
            self._stages = [[] for _ in range(vp_size)]

        self._delta_offload_bytes_across_pp_ranks = delta_offload_bytes_across_pp_ranks
        self._pp_rank = pp_rank
        self._activation_offload_fraction = activation_offload_fraction

        if vp_stage is None:
            cur_vpp_rank = 0
        else:
            cur_vpp_rank = vp_stage

        # Flush staged chunks when reaching the last virtual pipeline stage
        if cur_vpp_rank == self._vpp - 1:
            self.flush()

        # Use shared CPU tensor pool for better reuse across chunks
        cur_chunk = ChunkOffloadHandler(
            min_offloaded_tensor_size,
            self._cpu_tensor_pool,
            max_inflight_offloads=max_inflight_offloads,
        )
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
            if self._cached_chunks_index_forward >= len(self._cached_chunks_forward):
                self._cur_forward_chunk = None
                break
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

    def mark_not_offload(self, tensor: torch.Tensor):
        """Mark the current forward chunk as not offloadable."""
        if tensor is not None:
            # TE marks some tensors with _TE_do_not_offload; this local flag
            # gives Megatron-owned tensors the same opt-out path.
            tensor._do_not_offload = True

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
        self._saved_tensors_hooks.__enter__()

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
        self._saved_tensors_hooks.__exit__()

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
        debug_rank("----on_get_saved_tensor")
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

    def __init__(
        self,
        min_offloaded_tensor_size,
        cpu_tensor_pool,
        max_inflight_offloads: Optional[int] = None,
    ):
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
        # Max per-group-name inflight offloads not yet joined on the main stream (None = off).
        self._max_inflight_offloads = max_inflight_offloads
        # group_name -> FIFO of offload events for that name (same cap for every name).
        self._offload_pending_by_name: Dict[str, deque] = defaultdict(deque)

    def reset(self):
        """Reset the chunk offload handler."""
        self._offloaded_group_index = 0
        self._groups_to_offload = []
        self._groups_to_reload = []
        self._tensor_count_current_group = 0
        self._reloading_group = []
        # Clear the pending-event FIFO at iter boundary so we never wait on
        # an event recorded in a previous (non-captured) iteration.
        self._offload_pending_by_name.clear()

    def find_group_with_name(
        self, groups: list[OffloadTensorGroup], name: str, start_index: int = 0
    ):
        """Find the group with the given name starting from the given index."""
        return next((group for group in groups[start_index:] if group._name == name), None)

    def is_empty_chunk(self, name=None):
        """Check if this chunk has no tensors to manage."""
        debug_rank(f"------is_empty_chunk {self._max_group_size}")
        if name is not None:
            return self.find_group_with_name(self.offload_groups, name) is None
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
        return (
            self.find_group_with_name(self.offload_groups, name, self._offloaded_group_index)
            is None
        )

    def find_next_group(self, name=None):
        """Find the next group with the given name."""
        assert name is not None, "Name is required"
        return self.find_group_with_name(self.offload_groups, name, self._offloaded_group_index)

    @staticmethod
    def _can_manage_tensor_for_offload(tensor):
        """Return whether the tensor can be managed by activation offload hooks."""
        torch_stray_tensor = isinstance(
            tensor,
            (
                torch._subclasses.fake_tensor.FakeTensor,
                torch._subclasses.functional_tensor.FunctionalTensor,
            ),
        )
        return (
            not isinstance(tensor, torch.nn.Parameter)
            and not torch_stray_tensor
            and tensor.device.type in ("cuda", "musa")
        )

    def tensor_push(self, tensor):
        """Push tensor to the offload handler."""
        if not self._can_manage_tensor_for_offload(tensor):
            return tensor

        # Assign unique tag based on group index and position within group
        tensor_tag = (self._offloaded_group_index, self._tensor_count_current_group)
        self._tensor_count_current_group += 1
        self.offload_groups[self._offloaded_group_index - 1].push_tensor(tensor_tag, tensor)
        debug_rank(f"--------tensor_push {tensor_tag}")
        return tensor_tag

    def tensor_pop(self, tensor_tag):
        """Pop tensor from the offload handler."""
        if isinstance(tensor_tag, torch.Tensor):
            debug_rank(f"--------tensor_pop passthrough tensor {tensor_tag.shape}")
            return tensor_tag
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
        debug_rank("tensor_need_offloading_checker")
        if not self._can_manage_tensor_for_offload(tensor):
            return False
        if _te_do_not_offload(tensor):
            return False
        if tensor.numel() < self.min_offloaded_tensor_size:
            return False
        # Respect tensor's offload preference if specified
        if getattr(tensor, "_TE_do_not_offload", False) or getattr(
            tensor, "_do_not_offload", False
        ):
            return False
        return True

    def bulk_offload_group(self, group_to_offload):
        """offload a group of tensors recorded in tensor_push()."""
        debug_rank("------bulk_offload_group")
        nvtx_msg = "activation offloading " + group_to_offload._name
        nvtx_range_push(nvtx_msg)
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
        nvtx_range_pop(nvtx_msg)
        # Under full-iteration CG capture, the main stream may not wait on d2h
        # events; optional max-inflight enqueues each group's offload event and
        # has main wait on older events for this group name when its pending
        # count exceeds the cap (each name is tracked separately).
        if self._max_inflight_offloads is not None:
            gname = group_to_offload._name
            self._offload_pending_by_name[gname].append(group_to_offload._offload_event)
            self._drain_offload_pending(gname)

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
        nvtx_msg = "activation reloading " + group_to_reload._name
        nvtx_range_push(nvtx_msg)
        with torch.cuda.stream(self.h2d_stream):
            # Wait for offload to complete before reloading
            if not is_graph_capturing():
                group_to_reload.wait_offload_event(self.h2d_stream)
            for tensor_tag, state in group_to_reload._tensors.items():
                # Only reload if tensor was offloaded (stored as tuple)
                if isinstance(state, tuple):
                    recovered_tensor = self.reload(state)
                    group_to_reload.push_tensor(tensor_tag, recovered_tensor)
            group_to_reload.record_reload_event(self.h2d_stream)
        self._groups_to_reload.pop()
        # Add the group to the reloading group to wait for the reload event.
        self._reloading_group.append(group_to_reload)
        nvtx_range_pop(nvtx_msg)

    def pre_reload_last_layer(self):
        """Pre-reload the last layer of this chunk to hide reload latency."""
        debug_rank("pre_reload_last_layer")
        debug_rank(f"len(self._groups_to_reload) {len(self._groups_to_reload)}")
        if len(self._groups_to_reload) > 0:
            # Reload the last group (last layer) early
            self.bulk_reload_group()

    def should_bulk_offload(self, group):
        """Determine if the current group should be offloaded."""
        assert group in self._groups_to_offload, f"Group {group} is not pending offload"
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

    def bulk_offload(self, name, forced_released_tensors):
        """Offload a group of tensors and optionally release their GPU memory."""
        debug_rank("----bulk_offload")
        # CUDA graph scoped modules can create several pending groups before a
        # commit runs, so match by name instead of assuming LIFO order.
        group_to_offload = self.find_group_with_name(self._groups_to_offload, name)
        assert group_to_offload is not None, f"Group {name} not found in {self._groups_to_offload}"
        if self.should_bulk_offload(group_to_offload):
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
        # A group commit is consumed even when policy keeps its tensors on GPU.
        self._groups_to_offload.remove(group_to_offload)

    def _drain_offload_pending(self, group_name: str) -> None:
        """For ``group_name``, have the main stream wait on older D2H events
        when that name's pending count exceeds ``_max_inflight_offloads``
        (same cap for every name; 0 = wait on each commit for that name)."""
        if self._max_inflight_offloads is None:
            return
        cur = torch.cuda.current_stream()
        q = self._offload_pending_by_name[group_name]
        while len(q) > self._max_inflight_offloads:
            old_evt = q.popleft()
            cur.wait_event(old_evt)

    def on_group_commit_forward(self, name, forced_released_tensors):
        """Called at the end of a layer group's forward pass to trigger offloading."""
        if not self.do_offload:
            return
        debug_rank(f"--on_group_commit_forward {name}")
        # Wait for compute to finish before starting offload
        self.d2h_stream.wait_stream(torch.cuda.current_stream())
        self.bulk_offload(name, forced_released_tensors)

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
        # Fence H2D submission with a point event from the current compute
        # stream, rather than waiting on the stream as a whole.  This keeps
        # later graph work from becoming an implicit dependency of this reload.
        compute_ready = torch.cuda.Event()
        compute_ready.record(torch.cuda.current_stream())
        self.h2d_stream.wait_event(compute_ready)
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

        if delay_offload and PipelineOffloadManager.get_instance()._in_replay:
            # During TE CUDA graph replay, queue D2H work and launch it after
            # replay returns, where CPU scheduling can overlap with graph/comm gaps.
            PipelineOffloadManager.get_instance().push_offload_groups(
                cur_forward_chunk.on_group_commit_forward, name, forced_released_tensors
            )
        else:
            cur_forward_chunk.on_group_commit_forward(name, forced_released_tensors)
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


def fine_grained_offloading_group_offload(
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
        offloaded0 = fine_grained_offloading_group_offload(
            tensor[0],
            name=name,
            forced_released_tensors=forced_released_tensors,
            delay_offload=delay_offload,
        )
        return (offloaded0,) + tensor[1:]
    if isinstance(tensor, list):
        if len(tensor) == 0:
            return tensor
        offloaded0 = fine_grained_offloading_group_offload(
            tensor[0],
            name=name,
            forced_released_tensors=forced_released_tensors,
            delay_offload=delay_offload,
        )
        return [offloaded0] + tensor[1:]

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


class FineGrainedOffloadingBackwardRecordFunction(torch.autograd.Function):
    """
    Identity operation that marks the end of a layer group for offload synchronization.
    Triggers offload during forward and synchronizes reload during backward.
    """

    @staticmethod
    def forward(ctx, tensor) -> torch.Tensor:
        """Forward pass for cuda graph capture."""
        debug_rank("FineGrainedOffloadingBackwardRecordFunction forward")
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        """Record the backward event and wait for the h2d stream on cuda graph stream."""
        debug_rank("FineGrainedOffloadingBackwardRecordFunction backward")
        mgr = PipelineOffloadManager.get_instance()
        # Record the graph-stream position for offload bookkeeping, but do not
        # wait on the whole H2D stream here. Each consumed slot installs its
        # own completion-event dependency in local_graph_backward_wait_ready();
        # waiting on h2d_stream globally would also wait for later layers.
        torch.cuda.current_stream().record_event(mgr.cuda_graph_event)
        return (grad_output,)


class FineGrainedActivationOffloadingInterface:
    """Interface for fine-grained activation offloading."""

    def __init__(self, offload: bool, tensor: torch.Tensor, name: str):
        self.offload = offload
        self.tensor = tensor
        self.name = name
        self._local_graph_capture = False

    def __enter__(self):
        """Enter context manager to enable activation offloading hooks."""
        if self.offload:
            manager = PipelineOffloadManager.get_instance()
            if manager.in_local_graph_capture:
                self._local_graph_capture = True
                min_tensor_size = manager._local_graph_capture_runner.base_module.config.min_offloaded_tensor_size
                manager.local_graph_group_start(self.name, min_tensor_size)
            else:
                self.tensor = fine_grained_offloading_group_start(self.tensor, self.name)
                manager.__enter__()
        return self.tensor

    def __exit__(self, *args: Any):
        """Exit context manager to disable activation offloading hooks."""
        if self.offload:
            manager = PipelineOffloadManager.get_instance()
            if self._local_graph_capture:
                manager.local_graph_group_context_exit()
            else:
                manager.__exit__()

    @staticmethod
    def cuda_graph_stream():
        """Get the CUDA graph stream."""
        return PipelineOffloadManager.get_instance().cuda_graph_stream

    @staticmethod
    def cuda_graph_event():
        """Get the CUDA graph event."""
        return PipelineOffloadManager.get_instance().cuda_graph_event

    @staticmethod
    def init_chunk_handler(
        pp_rank,
        vp_size,
        vp_stage,
        min_offloaded_tensor_size,
        delta_offload_bytes_across_pp_ranks,
        activation_offload_fraction,
        max_inflight_offloads: Optional[int] = None,
    ):
        """Initialize the chunk handler, called at the start of a microbatch forward pass."""
        PipelineOffloadManager.get_instance().init_model_chunk_offload_handler(
            pp_rank,
            vp_size,
            vp_stage,
            min_offloaded_tensor_size,
            delta_offload_bytes_across_pp_ranks,
            activation_offload_fraction,
            max_inflight_offloads=max_inflight_offloads,
        )

    @staticmethod
    def get_context(flag):
        """Get the fine-grained offload context"""
        return PipelineOffloadManager.get_instance() if flag else nullcontext()

    def group_offload(self, tensor, forced_released_tensors=None, delay_offload=False):
        """Group offload the tensors."""
        if self.offload:
            manager = PipelineOffloadManager.get_instance()
            if self._local_graph_capture:
                manager.local_graph_group_commit(self.name)
                return tensor
            return fine_grained_offloading_group_offload(
                tensor, self.name, forced_released_tensors, delay_offload
            )
        return tensor

    @staticmethod
    def mark_not_offload(tensor: torch.Tensor):
        """Mark the tensor as not offloadable."""
        PipelineOffloadManager.get_instance().mark_not_offload(tensor)

    @staticmethod
    def forward_record() -> None:
        """Record the forward event for cuda graph capture."""
        mgr = PipelineOffloadManager.get_instance()
        torch.cuda.current_stream().record_event(mgr.cuda_graph_event)
        torch.cuda.current_stream().wait_stream(mgr.d2h_stream)

    @staticmethod
    def backward_record(tensor) -> torch.Tensor:
        """Record the backward event for cuda graph capture."""
        return FineGrainedOffloadingBackwardRecordFunction.apply(tensor)

    @staticmethod
    def reset():
        """Reset the chunk handler."""
        PipelineOffloadManager.get_instance().reset()

    @staticmethod
    def reset_instance():
        """Reset the singleton instance."""
        PipelineOffloadManager.reset_instance()

    @staticmethod
    def flush_delayed_groups():
        """Flush the delayed groups."""
        PipelineOffloadManager.get_instance().flush_delayed_groups()

    @staticmethod
    def disable_offload():
        """Disable the offload."""
        PipelineOffloadManager.get_instance().disable_offload()

    @staticmethod
    def enable_offload():
        """Enable the offload."""
        PipelineOffloadManager.get_instance().enable_offload()

    @staticmethod
    def enter_replay():
        """Enter CUDA graph replay mode to enable delayed offloading."""
        PipelineOffloadManager.get_instance()._in_replay = True

    @staticmethod
    def exit_replay():
        """Exit CUDA graph replay mode."""
        PipelineOffloadManager.get_instance()._in_replay = False
