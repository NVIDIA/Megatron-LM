# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Trace-planned storage allocator for experimental Megatron-FSDP.

The allocator observes a deterministic trace phase of temporary-buffer
lifetimes, then assigns non-overlapping keys to persistent tensor slots. Trace
allocations keep one Storage object per logical key so autograd-saved parameter
views remain valid between the forward release and backward re-gather. The
transition to shared slots happens only after that global batch has completed.
"""

import dataclasses
import logging
from collections import defaultdict
from collections.abc import Hashable
from contextlib import nullcontext

import torch
import torch.distributed._symmetric_memory as symm_mem

logger = logging.getLogger(__name__)

AllocatorKey = Hashable


@dataclasses.dataclass(frozen=True)
class _TraceEvent:
    """One allocation-lifetime boundary."""

    sequence: int
    operation: str
    key: AllocatorKey


@dataclasses.dataclass
class _Slot:
    """One physical tensor allocation reusable by compatible keys."""

    tensor: torch.Tensor
    capacity: int
    dtype: torch.dtype
    device: torch.device
    arena: AllocatorKey | None
    in_use: bool = False


class TracePoolAllocator:
    """Trace temporary tensor lifetimes and reuse persistent physical slots.

    ``allocate`` and ``free`` are called with a stable logical key. During the
    trace phase, each key owns a stable tensor Storage that is resized to
    zero on free and restored on its next allocation. This matches the storage
    aliasing required by autograd-saved weight views. ``plan`` then colors the
    resulting conflict graph and assigns every key a fixed steady-state slot.
    When ``use_symmetric_memory`` is enabled, slot creation and every storage
    restoration run inside PyTorch's NCCL symmetric-memory pool.

    Slots are also partitioned by ``arena``. Callers use one arena per ordered
    CUDA stream so host-side non-overlap never aliases asynchronous operations
    executing on different streams.
    """

    def __init__(self, *, use_symmetric_memory: bool = False) -> None:
        if use_symmetric_memory and not hasattr(symm_mem, "is_symm_mem_tensor"):
            raise RuntimeError("Symmetric-memory MFSDP requires PyTorch 2.12 or later.")
        if use_symmetric_memory:
            # PyTorch caches this in C++ and returns early when the backend is already NCCL.
            symm_mem.set_backend("NCCL")

        self._use_symmetric_memory = use_symmetric_memory
        self._phase = "trace"
        self._sequence = 0
        self._trace: list[_TraceEvent] = []
        self._metadata: dict[
            AllocatorKey, tuple[int, torch.dtype, torch.device, AllocatorKey | None]
        ] = {}
        self._active_keys: set[AllocatorKey] = set()

        # Trace slots are one-to-one with logical keys. Their Storage identity
        # must not change within the trace phase.
        self._trace_slots: list[_Slot] = []
        self._trace_key_to_slot: dict[AllocatorKey, int] = {}

        # Steady-state slots are populated by plan().
        self._slots: list[_Slot] = []
        self._key_to_slot: dict[AllocatorKey, int] = {}
        self._key_to_view: dict[AllocatorKey, torch.Tensor] = {}

    @property
    def phase(self) -> str:
        """Return ``trace`` or ``optimized``."""
        return self._phase

    @property
    def use_symmetric_memory(self) -> bool:
        """Whether physical slots come from PyTorch's symmetric-memory pool."""
        return self._use_symmetric_memory

    @property
    def total_pool_bytes(self) -> int:
        """Return physical slot capacity in bytes for the current phase."""
        slots = self._slots if self._phase == "optimized" else self._trace_slots
        return sum(slot.capacity * slot.tensor.element_size() for slot in slots)

    @property
    def active_keys(self) -> frozenset[AllocatorKey]:
        """Return logical allocations that have not been freed."""
        return frozenset(self._active_keys)

    def allocate(
        self,
        key: AllocatorKey,
        size: int,
        dtype: torch.dtype,
        device: torch.device | str,
        *,
        arena: AllocatorKey | None = None,
    ) -> torch.Tensor:
        """Return a flat tensor for ``key`` with at least ``size`` elements."""
        if size < 0:
            raise ValueError(f"Allocation size must be non-negative, got {size}.")
        device = torch.device(device)
        if self._phase == "trace":
            return self._trace_allocate(key, size, dtype, device, arena)
        return self._optimized_allocate(key, size, dtype, device, arena)

    def free(self, key: AllocatorKey) -> None:
        """End the current lifetime for ``key``; duplicate frees are ignored."""
        if key not in self._active_keys:
            return
        if self._phase == "trace":
            self._trace.append(_TraceEvent(self._sequence, "free", key))
            self._sequence += 1
            slot = self._trace_slots[self._trace_key_to_slot[key]]
            slot.tensor.untyped_storage().resize_(0)
            slot.in_use = False
        else:
            self._slots[self._key_to_slot[key]].in_use = False
        self._active_keys.remove(key)

    def _trace_allocate(
        self,
        key: AllocatorKey,
        size: int,
        dtype: torch.dtype,
        device: torch.device,
        arena: AllocatorKey | None,
    ) -> torch.Tensor:
        metadata = self._metadata.get(key)
        if metadata is not None:
            max_size, expected_dtype, expected_device, expected_arena = metadata
            if (dtype, device, arena) != (expected_dtype, expected_device, expected_arena):
                raise ValueError(
                    f"Allocation metadata changed for {key!r}: "
                    f"{(expected_dtype, expected_device, expected_arena)} -> "
                    f"{(dtype, device, arena)}."
                )
            self._metadata[key] = (max(max_size, size), dtype, device, arena)

        if key in self._active_keys:
            slot = self._trace_slots[self._trace_key_to_slot[key]]
            if size > slot.capacity:
                raise RuntimeError(
                    f"Active allocation {key!r} grew from {slot.capacity} to {size} elements."
                )
            return slot.tensor.narrow(0, 0, size)

        self._trace.append(_TraceEvent(self._sequence, "alloc", key))
        self._sequence += 1
        if metadata is None:
            self._metadata[key] = (size, dtype, device, arena)

        slot_index = self._trace_key_to_slot.get(key)
        if slot_index is None:
            slot_index = len(self._trace_slots)
            slot = _Slot(
                tensor=self._empty(size, dtype, device),
                capacity=size,
                dtype=dtype,
                device=device,
                arena=arena,
            )
            self._trace_slots.append(slot)
            self._trace_key_to_slot[key] = slot_index
        else:
            slot = self._trace_slots[slot_index]
            if size > slot.capacity:
                self._resize_slot(slot, size)
                slot.capacity = size
            else:
                self._resize_slot(slot, slot.capacity)

        slot.in_use = True
        self._active_keys.add(key)
        return slot.tensor.narrow(0, 0, size)

    def plan(self) -> int:
        """Compile recorded lifetimes and enter the optimized phase.

        CUDA work is synchronized once at this trace-to-optimized boundary.
        After final slots have claimed their allocations, unused caching-
        allocator blocks are released. Later batches rebuild any non-FSDP
        cache around the fixed pool without further trimming.

        Returns:
            Number of bytes owned by the persistent pool.
        """
        if self._phase != "trace":
            return self.total_pool_bytes
        if self._active_keys:
            raise RuntimeError(
                "Cannot plan the trace pool with live allocations: "
                f"{tuple(self._active_keys)!r}."
            )

        cuda_devices = {
            device for _, _, device, _ in self._metadata.values() if device.type == "cuda"
        }
        for device in cuda_devices:
            torch.cuda.synchronize(device)

        intervals = self._build_intervals()
        groups: dict[tuple[torch.dtype, torch.device, AllocatorKey | None], list[AllocatorKey]] = (
            defaultdict(list)
        )
        for key in intervals:
            _, dtype, device, arena = self._metadata[key]
            groups[(dtype, device, arena)].append(key)

        required_slots: list[tuple[int, torch.dtype, torch.device, AllocatorKey | None]] = []
        key_to_required_slot: dict[AllocatorKey, int] = {}
        for (dtype, device, arena), keys in groups.items():
            capacities, colors = self._color_group(keys, intervals)
            offset = len(required_slots)
            required_slots.extend((capacity, dtype, device, arena) for capacity in capacities)
            for key, color in colors.items():
                key_to_required_slot[key] = offset + color

        self._materialize_plan(required_slots)
        self._key_to_slot = key_to_required_slot
        self._key_to_view = {
            key: self._slots[slot_index].tensor.narrow(0, 0, self._metadata[key][0])
            for key, slot_index in self._key_to_slot.items()
        }
        self._trace_key_to_slot.clear()
        self._phase = "optimized"

        # Trace allocations were released before planning. Keep the selected
        # pool slots live and return only surplus cached blocks to CUDA.
        for device in cuda_devices:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()

        logger.info(
            "MFSDP trace pool planned %d keys into %d slots (%.1f MiB).",
            len(self._key_to_slot),
            len(self._slots),
            self.total_pool_bytes / 1024**2,
        )
        return self.total_pool_bytes

    def _build_intervals(self) -> dict[AllocatorKey, list[tuple[int, int]]]:
        starts: dict[AllocatorKey, list[int]] = defaultdict(list)
        intervals: dict[AllocatorKey, list[tuple[int, int]]] = defaultdict(list)
        for event in self._trace:
            if event.operation == "alloc":
                starts[event.key].append(event.sequence)
                continue
            if starts[event.key]:
                intervals[event.key].append((starts[event.key].pop(0), event.sequence))

        sentinel = 1 << 60
        for key, pending in starts.items():
            for start in pending:
                intervals[key].append((start, sentinel))
                sentinel += 1
        return dict(intervals)

    def _color_group(
        self, keys: list[AllocatorKey], intervals: dict[AllocatorKey, list[tuple[int, int]]]
    ) -> tuple[list[int], dict[AllocatorKey, int]]:
        conflicts: dict[AllocatorKey, set[AllocatorKey]] = defaultdict(set)
        for index, left in enumerate(keys):
            for right in keys[index + 1 :]:
                if _intervals_overlap(intervals[left], intervals[right]):
                    conflicts[left].add(right)
                    conflicts[right].add(left)

        colors: dict[AllocatorKey, int] = {}
        capacities: list[int] = []
        for key in sorted(keys, key=lambda item: self._metadata[item][0], reverse=True):
            size = self._metadata[key][0]
            unavailable = {colors[neighbor] for neighbor in conflicts[key] if neighbor in colors}
            candidates = [
                (max(capacity, size) - size, color)
                for color, capacity in enumerate(capacities)
                if color not in unavailable
            ]
            if candidates:
                _, color = min(candidates)
                capacities[color] = max(capacities[color], size)
            else:
                color = len(capacities)
                capacities.append(size)
            colors[key] = color
        return capacities, colors

    def _materialize_plan(
        self, required_slots: list[tuple[int, torch.dtype, torch.device, AllocatorKey | None]]
    ) -> None:
        """Reuse trace Storage objects for final slots and retire the surplus."""
        available = set(range(len(self._trace_slots)))
        final_slots: list[_Slot | None] = [None] * len(required_slots)
        required_indices = sorted(
            range(len(required_slots)), key=lambda index: required_slots[index][0], reverse=True
        )
        for required_index in required_indices:
            size, dtype, device, arena = required_slots[required_index]
            candidates = [
                (self._trace_slots[index].capacity, index)
                for index in available
                if self._trace_slots[index].dtype == dtype
                and self._trace_slots[index].device == device
                and self._trace_slots[index].arena == arena
                and self._trace_slots[index].capacity >= size
            ]
            if not candidates:
                raise RuntimeError(
                    "Trace-pool plan could not reuse a compatible trace allocation for "
                    f"{(size, dtype, device, arena)!r}."
                )
            _, index = min(candidates)
            available.remove(index)
            slot = self._trace_slots[index]
            self._resize_slot(slot, size)
            slot.capacity = size
            slot.in_use = False
            final_slots[required_index] = slot

        for index in available:
            self._trace_slots[index].tensor.untyped_storage().resize_(0)
        self._trace_slots.clear()
        assert all(slot is not None for slot in final_slots)
        self._slots = [slot for slot in final_slots if slot is not None]

    def _optimized_allocate(
        self,
        key: AllocatorKey,
        size: int,
        dtype: torch.dtype,
        device: torch.device,
        arena: AllocatorKey | None,
    ) -> torch.Tensor:
        if key not in self._key_to_slot:
            self._add_late_key(key, size, dtype, device, arena)
        expected_size, expected_dtype, expected_device, expected_arena = self._metadata[key]
        if (
            dtype != expected_dtype
            or device != expected_device
            or arena != expected_arena
            or size > expected_size
        ):
            raise ValueError(
                f"Optimized allocation for {key!r} does not match its plan: "
                f"requested {(size, dtype, device, arena)}, planned "
                f"{(expected_size, expected_dtype, expected_device, expected_arena)}."
            )

        slot_index = self._key_to_slot[key]
        slot = self._slots[slot_index]
        if slot.in_use and key not in self._active_keys:
            owners = tuple(
                active_key
                for active_key in self._active_keys
                if self._key_to_slot.get(active_key) == slot_index
            )
            raise RuntimeError(
                f"Trace-pool slot {slot_index} is still used by {owners!r} "
                f"while allocating {key!r}."
            )
        slot.in_use = True
        self._active_keys.add(key)
        return self._key_to_view[key].narrow(0, 0, size)

    def _add_late_key(
        self,
        key: AllocatorKey,
        size: int,
        dtype: torch.dtype,
        device: torch.device,
        arena: AllocatorKey | None,
    ) -> None:
        slot_index = len(self._slots)
        tensor = self._empty(size, dtype, device)
        self._slots.append(_Slot(tensor, size, dtype, device, arena))
        self._metadata[key] = (size, dtype, device, arena)
        self._key_to_slot[key] = slot_index
        self._key_to_view[key] = tensor
        logger.warning("MFSDP trace pool added late allocation key %r.", key)

    def reset(self) -> None:
        """Release trace/pool storage and return to tracing."""
        for slot in (*self._trace_slots, *self._slots):
            slot.tensor.untyped_storage().resize_(0)
        self._phase = "trace"
        self._sequence = 0
        self._trace.clear()
        self._metadata.clear()
        self._active_keys.clear()
        self._trace_slots.clear()
        self._trace_key_to_slot.clear()
        self._slots.clear()
        self._key_to_slot.clear()
        self._key_to_view.clear()

    def _empty(self, size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Allocate a flat slot from the configured CUDA memory pool."""
        with self._allocation_context(device):
            return torch.empty(size, dtype=dtype, device=device)

    def _resize_slot(self, slot: _Slot, size: int) -> None:
        """Resize a slot while preserving its Storage object and allocation pool."""
        with self._allocation_context(slot.device):
            slot.tensor.untyped_storage().resize_(size * slot.tensor.element_size())
            slot.tensor.resize_(size)

    def _allocation_context(self, device: torch.device):
        """Select PyTorch's symmetric-memory pool when requested."""
        if not self._use_symmetric_memory:
            return nullcontext()
        if device.type != "cuda":
            raise ValueError(f"Symmetric-memory trace-pool allocations require CUDA, got {device}.")
        return torch.cuda.use_mem_pool(symm_mem.get_mem_pool(device))


def _intervals_overlap(left: list[tuple[int, int]], right: list[tuple[int, int]]) -> bool:
    return any(
        left_start < right_end and right_start < left_end
        for left_start, left_end in left
        for right_start, right_end in right
    )
