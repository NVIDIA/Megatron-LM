# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import logging
from collections import defaultdict
from typing import Dict, Hashable, List, Optional, Set, Tuple

import torch

logger = logging.getLogger(__name__)

AllocatorKey = Hashable


def _resolve_key(key: Optional[AllocatorKey], param_group_id: Optional[AllocatorKey]):
    if key is not None:
        return key
    assert param_group_id is not None, "allocator key is required"
    return param_group_id


@dataclasses.dataclass
class Bucket:
    """Lightweight container for a temporary allocated tensor buffer."""

    data: torch.Tensor


class BucketAllocator:
    """Interface for allocating and freeing temporary buckets."""

    def allocate(
        self,
        key: Optional[AllocatorKey] = None,
        size: int = 0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        *,
        param_group_id: Optional[AllocatorKey] = None,
    ) -> Bucket:
        """Allocate a bucket for the given key."""
        raise NotImplementedError

    def free(
        self,
        key: Optional[AllocatorKey] = None,
        *,
        param_group_id: Optional[AllocatorKey] = None,
    ) -> None:
        """Free the bucket associated with the given key."""
        raise NotImplementedError


class TemporaryBucketAllocator(BucketAllocator):
    """Manages temporary flat buffers keyed by a caller-provided key.

    Used by DataParallelBuffer for unshard (all-gather) and gradient
    reduction (reduce-scatter) operations.
    """

    def __init__(self):
        super().__init__()
        self.buckets = {}

    def allocate(
        self,
        key: Optional[AllocatorKey] = None,
        size: int = 0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        *,
        param_group_id: Optional[AllocatorKey] = None,
    ) -> Bucket:
        key = _resolve_key(key, param_group_id)
        assert dtype is not None and device is not None
        if key not in self.buckets:
            self.buckets[key] = Bucket(
                data=torch.empty(size, dtype=dtype, device=device)
            )
        return self.buckets[key]

    def free(
        self,
        key: Optional[AllocatorKey] = None,
        *,
        param_group_id: Optional[AllocatorKey] = None,
    ) -> None:
        key = _resolve_key(key, param_group_id)
        if key in self.buckets:
            _free_storage(self.buckets[key].data)
            del self.buckets[key]


class StorageFreeingBucketAllocator(BucketAllocator):
    """Manages temporary flat buffers keyed by caller-provided allocation key."""

    def __init__(self):
        super().__init__()
        self.buckets = {}

    def allocate(
        self,
        key: Optional[AllocatorKey] = None,
        size: int = 0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        *,
        param_group_id: Optional[AllocatorKey] = None,
    ) -> Bucket:
        key = _resolve_key(key, param_group_id)
        assert dtype is not None and device is not None
        if key not in self.buckets:
            self.buckets[key] = Bucket(
                data=torch.empty(size, dtype=dtype, device=device)
            )
            return self.buckets[key]
        _alloc_storage(self.buckets[key].data, torch.Size([size]))
        return self.buckets[key]

    def free(
        self,
        key: Optional[AllocatorKey] = None,
        *,
        param_group_id: Optional[AllocatorKey] = None,
    ) -> None:
        key = _resolve_key(key, param_group_id)
        if key in self.buckets:
            _free_storage(self.buckets[key].data)


class TracePoolAllocator(BucketAllocator):
    """Two-phase bucket allocator for CUDA graph-compatible training.

    Profiles one micro-batch to record allocation patterns, then builds a
    static key-to-address plan that is identical across all subsequent
    micro-batches — essential for CUDA graph capture.

    **Phase 1 — Trace** (first micro-batch)

    Records alloc/free calls with monotonic sequence numbers.  Buckets are
    created with ``torch.empty`` and freed via ``_free_storage`` so the same
    tensor object can be resurrected on re-alloc (keeping outstanding views
    alive, e.g.  NVFP4 ``_rowwise_data`` references).

    **Phase 2 — Plan** (``plan()``)

    Replays the trace to build per-key live intervals, then uses
    **conflict-graph coloring** to assign slots:

    * An interval-overlap graph is built: edges connect keys whose live
      intervals overlap.
    * Nodes are colored greedily (largest-size-first, best-fit bin packing)
      so two keys share a slot iff they never overlap.
    * Yields the **theoretical minimum** number of slots.

    Each slot is a **separate** ``torch.empty()`` tensor (per-slot allocation),
    not a slice of a monolithic pool.  The CUDA caching allocator can place
    them independently, reducing fragmentation pressure from giant contiguous
    blocks, while each key still resolves to a fixed memory address.

    **Phase 3 — Optimized** (after ``plan()``)

    ``allocate`` / ``free`` are O(1) dict lookups that return pre-computed
    tensor views.  No allocations or storage resizes occur in this phase —
    memory addresses are stable across all micro-batches.
    """

    # -- Inner types ---------------------------------------------------- #

    @dataclasses.dataclass
    class _SlotInfo:
        """Metadata for a physical slot (backed by its own tensor)."""

        tensor: torch.Tensor  # The actual backing tensor for this slot
        size: int  # Capacity in elements
        dtype: torch.dtype
        device: torch.device
        in_use: bool = False

    @dataclasses.dataclass
    class _TraceEvent:
        """A single alloc or free recorded during the trace phase."""

        seq: int
        op: str  # "alloc" | "free"
        key: AllocatorKey

    # -- Init ----------------------------------------------------------- #

    def __init__(self) -> None:
        super().__init__()
        self._phase: str = "trace"  # "trace" | "optimized"

        # Trace state
        self._seq: int = 0
        self._trace: List["TracePoolAllocator._TraceEvent"] = []
        self._trace_meta: Dict[AllocatorKey, Tuple[int, torch.dtype, torch.device]] = {}
        self._buckets: Dict[AllocatorKey, Bucket] = {}
        self._active_keys: Set[AllocatorKey] = set()

        # Pool state — populated by plan(), used in optimized phase
        self._slots: List["TracePoolAllocator._SlotInfo"] = []
        self._key_to_slot: Dict[AllocatorKey, int] = {}
        # For each key, the view into its slot (pre-computed for O(1) access)
        self._key_to_view: Dict[AllocatorKey, torch.Tensor] = {}

    # -- Public interface ------------------------------------------------ #

    def allocate(
        self,
        key: Optional[AllocatorKey] = None,
        size: int = 0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        *,
        param_group_id: Optional[AllocatorKey] = None,
    ) -> Bucket:
        key = _resolve_key(key, param_group_id)
        assert dtype is not None and device is not None
        if self._phase == "released":
            self._auto_resume()
        if self._phase != "optimized":
            return self._trace_allocate(key, size, dtype, device)
        else:
            return self._optimized_allocate(key, size, dtype, device)

    def free(
        self,
        key: Optional[AllocatorKey] = None,
        *,
        param_group_id: Optional[AllocatorKey] = None,
    ) -> None:
        key = _resolve_key(key, param_group_id)
        if self._phase == "released":
            self._auto_resume()
        if self._phase != "optimized":
            self._trace_free(key)
        else:
            self._optimized_free(key)

    # -- Phase 1: trace -------------------------------------------------- #

    def _trace_allocate(
        self, key: AllocatorKey, size: int, dtype: torch.dtype, device: torch.device
    ) -> Bucket:
        if key in self._active_keys:
            return self._buckets[key]

        if key not in self._buckets:
            self._trace.append(self._TraceEvent(seq=self._seq, op="alloc", key=key))
            self._seq += 1
            self._trace_meta[key] = (size, dtype, device)
            self._buckets[key] = Bucket(
                data=torch.empty(size, dtype=dtype, device=device)
            )
        else:
            self._trace.append(self._TraceEvent(seq=self._seq, op="alloc", key=key))
            self._seq += 1
            _alloc_storage(self._buckets[key].data, torch.Size([size]))

        self._active_keys.add(key)
        return self._buckets[key]

    def _trace_free(self, key: AllocatorKey) -> None:
        if key not in self._active_keys:
            return
        self._trace.append(self._TraceEvent(seq=self._seq, op="free", key=key))
        self._seq += 1
        if key in self._buckets:
            _free_storage(self._buckets[key].data)
        self._active_keys.discard(key)

    # -- Phase 2: plan --------------------------------------------------- #

    def plan(self) -> int:
        """Build the static key→slot plan from the recorded trace.

        Uses conflict-graph coloring to achieve optimal memory usage, and
        allocates each slot as a separate tensor (per-slot allocation) to
        minimize fragmentation pressure on the CUDA caching allocator.

        Returns:
            Total pool size in elements (sum across all dtype/device groups).
        """
        assert self._phase == "trace", "plan() can only be called in trace phase"
        if len(self._trace) == 0:
            self._phase = "optimized"
            return 0

        # Step 1: Build per-key intervals from alloc/free pairs
        alloc_stack: Dict[AllocatorKey, List[int]] = {}
        intervals_per_key: Dict[AllocatorKey, List[Tuple[int, int]]] = defaultdict(list)

        for ev in self._trace:
            if ev.op == "alloc":
                alloc_stack.setdefault(ev.key, []).append(ev.seq)
            else:
                if ev.key in alloc_stack and alloc_stack[ev.key]:
                    alloc_seq = alloc_stack[ev.key].pop(0)
                    intervals_per_key[ev.key].append((alloc_seq, ev.seq))

        # Keys allocated but never freed get a sentinel free_seq
        _SENTINEL_FREE_SEQ = 1 << 60
        sentinel_seq = _SENTINEL_FREE_SEQ
        for key, pending_allocs in alloc_stack.items():
            for alloc_seq in pending_allocs:
                intervals_per_key[key].append((alloc_seq, sentinel_seq))
                sentinel_seq += 1

        if not intervals_per_key:
            self._phase = "optimized"
            return 0

        # Step 2: Compute per-key max size
        key_max_size: Dict[AllocatorKey, int] = {}
        for key in intervals_per_key:
            meta = self._trace_meta.get(key)
            if meta is not None:
                key_max_size[key] = meta[0]

        # Step 3: Group keys by (dtype, device)
        groups: Dict[
            Tuple[torch.dtype, torch.device], List[AllocatorKey]
        ] = defaultdict(list)
        for key in intervals_per_key:
            meta = self._trace_meta.get(key)
            if meta is not None:
                groups[(meta[1], meta[2])].append(key)

        # Step 4: Color each group and allocate per-slot tensors
        self._slots.clear()
        self._key_to_slot.clear()
        self._key_to_view.clear()

        total_elems = 0
        for (dtype, device), keys in groups.items():
            total_elems += self._color_and_allocate_slots(
                keys, intervals_per_key, key_max_size, dtype, device
            )

        # Free trace-phase resources
        self._buckets.clear()
        self._active_keys.clear()

        self._phase = "optimized"

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                logger.debug(
                    f"TracePoolAllocator plan complete: {len(self._slots)} slots, "
                    f"{total_elems} total elements, "
                    f"{self.total_pool_bytes / 1024 / 1024:.1f} MB"
                )
        return total_elems

    def _color_and_allocate_slots(
        self,
        keys: List[AllocatorKey],
        intervals_per_key: Dict[AllocatorKey, List[Tuple[int, int]]],
        key_max_size: Dict[AllocatorKey, int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> int:
        """Conflict-graph coloring + per-slot tensor allocation.

        Each color/slot gets its own ``torch.empty()`` tensor rather than
        being a slice of a monolithic pool. This reduces CUDA caching
        allocator fragmentation: slot tensors can be placed in gaps between
        other allocations rather than requiring one massive contiguous block.

        Returns total elements allocated across all slots in this group.
        """
        n = len(keys)
        if n == 0:
            return 0

        # Build conflict graph
        conflicts: Dict[AllocatorKey, Set[AllocatorKey]] = defaultdict(set)
        for i in range(n):
            key_a = keys[i]
            ivs_a = intervals_per_key[key_a]
            for j in range(i + 1, n):
                key_b = keys[j]
                ivs_b = intervals_per_key[key_b]
                if _intervals_overlap(ivs_a, ivs_b):
                    conflicts[key_a].add(key_b)
                    conflicts[key_b].add(key_a)

        # Greedy graph coloring: largest-first, best-fit
        keys_sorted = sorted(keys, key=lambda k: key_max_size.get(k, 0), reverse=True)
        color_of: Dict[AllocatorKey, int] = {}
        slot_sizes: List[int] = []  # color_idx -> capacity in elements

        for k in keys_sorted:
            size_k = key_max_size.get(k, 0)
            neighbor_colors: Set[int] = set()
            for neighbor in conflicts[k]:
                if neighbor in color_of:
                    neighbor_colors.add(color_of[neighbor])

            # Best-fit: find smallest existing slot that fits and doesn't conflict
            best_slot: Optional[int] = None
            best_waste = -1

            for slot_idx in range(len(slot_sizes)):
                if slot_idx in neighbor_colors:
                    continue
                new_capacity = max(slot_sizes[slot_idx], size_k)
                waste = new_capacity - size_k
                if best_slot is None or waste < best_waste:
                    best_waste = waste
                    best_slot = slot_idx

            if best_slot is not None:
                color_of[k] = best_slot
                slot_sizes[best_slot] = max(slot_sizes[best_slot], size_k)
            else:
                color_of[k] = len(slot_sizes)
                slot_sizes.append(size_k)

        # Allocate each slot as a SEPARATE tensor
        global_slot_offset = len(self._slots)
        slot_tensors: List[torch.Tensor] = []

        for slot_size in slot_sizes:
            t = torch.empty(slot_size, dtype=dtype, device=device)
            slot_tensors.append(t)
            self._slots.append(
                self._SlotInfo(
                    tensor=t, size=slot_size, dtype=dtype, device=device
                )
            )

        # Map each key to its slot and pre-compute the view
        for k in keys:
            local_idx = color_of[k]
            global_idx = global_slot_offset + local_idx
            self._key_to_slot[k] = global_idx
            size_k = key_max_size.get(k, 0)
            # View into the slot tensor (first size_k elements)
            self._key_to_view[k] = slot_tensors[local_idx][:size_k]

        return sum(slot_sizes)

    # -- Phase 3: optimized runtime ------------------------------------- #

    def _optimized_allocate(
        self, key: AllocatorKey, size: int, dtype: torch.dtype, device: torch.device
    ) -> Bucket:
        slot_idx = self._key_to_slot[key]
        slot = self._slots[slot_idx]
        assert size <= slot.size, (
            f"requested {size} > slot capacity {slot.size} (key={key!r})"
        )
        slot.in_use = True
        self._active_keys.add(key)
        # Return a view of the pre-allocated slot tensor
        view = self._key_to_view[key]
        return Bucket(data=view)

    def _optimized_free(self, key: AllocatorKey) -> None:
        if key not in self._active_keys:
            return
        self._slots[self._key_to_slot[key]].in_use = False
        self._active_keys.discard(key)

    # -- Lifecycle ------------------------------------------------------- #

    def reset(self) -> None:
        """Full teardown: discard pool, plan, and trace; return to trace phase."""
        self._phase = "trace"
        self._seq = 0
        self._trace.clear()
        self._trace_meta.clear()
        self._buckets.clear()
        self._active_keys.clear()
        self._slots.clear()
        self._key_to_slot.clear()
        self._key_to_view.clear()

    def release(self) -> None:
        """Release all slot tensor memory while preserving the slot plan.

        In ``"trace"`` phase this is equivalent to ``reset()`` — discards all
        trace data and returns to a clean trace state.

        In ``"optimized"`` phase this drops every ``_SlotInfo.tensor`` reference
        (freeing GPU memory) but retains the plan metadata: ``_slots``
        (size/dtype/device), ``_key_to_slot``, ``_key_to_view``, and the trace
        history.  The allocator transitions to ``"released"``.

        On the next ``allocate()`` or ``free()`` call the allocator
        **automatically re-allocates** all slots and returns to ``"optimized"``
        — no explicit ``resume()`` call is needed.
        """
        if self._phase == "released":
            return

        if self._phase == "trace":
            self.reset()
            return

        assert self._phase == "optimized", (
            f"release() requires 'optimized' or 'trace' phase, got '{self._phase}'"
        )
        for slot in self._slots:
            _free_storage(slot.tensor)
            slot.tensor = torch.empty(0, dtype=slot.dtype, device=slot.device)
            slot.in_use = False
        self._active_keys.clear()
        self._phase = "released"

    def _auto_resume(self) -> None:
        """Lazily re-allocate all slot tensors when the first ``allocate``/``free``
        arrives in the ``"released"`` phase.

        Internal — called automatically from ``allocate`` and ``free``.
        """
        if self._phase != "released":
            return

        for slot_idx, slot in enumerate(self._slots):
            new_tensor = torch.empty(slot.size, dtype=slot.dtype, device=slot.device)
            slot.tensor = new_tensor

        for key, slot_idx in self._key_to_slot.items():
            slot = self._slots[slot_idx]
            meta = self._trace_meta.get(key)
            if meta is not None:
                size_k, _, _ = meta
            else:
                size_k = slot.size
            self._key_to_view[key] = slot.tensor[: min(size_k, slot.size)]

        self._active_keys.clear()
        self._phase = "optimized"

    def resume(self) -> None:
        """Explicitly re-allocate slots and return to ``"optimized"`` phase.

        Normally you do not need to call this — the first ``allocate`` or
        ``free`` after ``release()`` will auto-resume.  Use this only when
        you need to restore the pool before any alloc/free call.
        """
        self._auto_resume()

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def total_pool_bytes(self) -> int:
        total = 0
        for slot in self._slots:
            total += slot.size * slot.tensor.element_size()
        return total

    # -- Debug ---------------------------------------------------------- #

    def dump_trace(self) -> str:
        """Return a human-readable dump of the trace and pool plan."""
        lines = []
        lines.append(f"=== TracePoolAllocator (phase={self._phase}) ===")
        lines.append(f"trace events: {len(self._trace)}")
        for ev in self._trace:
            meta = self._trace_meta.get(ev.key)
            size_str = f"size={meta[0]}" if meta else "size=?"
            dtype_str = f"dtype={meta[1]}" if meta else "dtype=?"
            device_str = f"device={meta[2]}" if meta else "device=?"
            lines.append(
                f"  seq={ev.seq:>4}  {ev.op:>5}  key={ev.key}  "
                f"{size_str}  {dtype_str}  {device_str}"
            )

        if self._phase in ("optimized", "released"):
            lines.append(f"\nslots: {len(self._slots)} ({self._phase})")
            for i, slot in enumerate(self._slots):
                keys_in_slot = [
                    k for k, idx in self._key_to_slot.items() if idx == i
                ]
                if self._phase == "optimized":
                    addr_str = f"addr=0x{slot.tensor.data_ptr():x}"
                else:
                    addr_str = "addr=<released>"
                lines.append(
                    f"  slot[{i}]: size={slot.size} "
                    f"dtype={slot.dtype} device={slot.device} "
                    f"{addr_str} "
                    f"{'IN_USE' if slot.in_use else 'free'} "
                    f"keys={keys_in_slot}"
                )
            if self._phase == "optimized":
                lines.append(
                    f"\ntotal pool: {len(self._slots)} slots, "
                    f"{self.total_pool_bytes} bytes "
                    f"({self.total_pool_bytes / 1024 / 1024:.1f} MB)"
                )
            else:
                lines.append(
                    f"\ntotal pool: {len(self._slots)} slots, "
                    f"memory released (call resume() to restore)"
                )

        return "\n".join(lines)


def _intervals_overlap(
    ivs_a: List[Tuple[int, int]], ivs_b: List[Tuple[int, int]]
) -> bool:
    """Check if any interval in ivs_a overlaps with any interval in ivs_b.

    Two intervals (a_start, a_end) and (b_start, b_end) overlap iff
    a_start < b_end AND b_start < a_end.
    """
    # For small lists (common case: 1-3 intervals per key), brute force
    if len(ivs_a) * len(ivs_b) <= 16:
        for a_start, a_end in ivs_a:
            for b_start, b_end in ivs_b:
                if a_start < b_end and b_start < a_end:
                    return True
        return False

    # Sweep-line for larger sets
    events: List[Tuple[int, int, int]] = []
    for start, end in ivs_a:
        events.append((start, 0, 0))
        events.append((end, 1, 0))
    for start, end in ivs_b:
        events.append((start, 0, 1))
        events.append((end, 1, 1))
    events.sort(key=lambda e: (e[0], -e[1]))

    active_a = 0
    active_b = 0
    for time, typ, group in events:
        if typ == 0:
            if group == 0:
                active_a += 1
                if active_b > 0:
                    return True
            else:
                active_b += 1
                if active_a > 0:
                    return True
        else:
            if group == 0:
                active_a -= 1
            else:
                active_b -= 1
    return False


def _is_torchdynamo_compiling() -> bool:
    """Check whether torchdynamo is compiling — safe across PyTorch versions."""
    try:
        return torch.distributed._functional_collectives.is_torchdynamo_compiling()
    except (AttributeError, RuntimeError):
        return False


def _free_storage(tensor: torch.Tensor) -> None:
    """Free the underlying storage of ``tensor`` by resizing it to 0."""
    with torch.no_grad():
        if not _is_torchdynamo_compiling():
            already_freed = tensor._typed_storage()._size() == 0
            if not already_freed:
                assert tensor.storage_offset() == 0, (
                    "Freeing a tensor's storage is unsafe when it is not the sole occupant\n"
                    f"storage offset: {tensor.storage_offset()}\n"
                    f"storage size: {tensor._typed_storage()._size()}\n"
                    f"tensor shape: {tensor.shape}"
                )
                tensor._typed_storage()._resize_(0)


def _alloc_storage(tensor: torch.Tensor, size: torch.Size) -> None:
    """Re-allocate storage for ``tensor`` to the given ``size``.

    Requires that the tensor's storage has been freed (resized to 0)
    before calling.  The caller must ensure ``size`` matches the tensor's
    existing shape.
    """
    with torch.no_grad():
        if not _is_torchdynamo_compiling():
            already_allocated = tensor._typed_storage()._size() == size.numel()
            if not already_allocated:
                tensor_storage_size = tensor._typed_storage()._size()
                assert tensor_storage_size == 0, (
                    "Tensor storage should have been resized to 0 but got "
                    f"{tensor_storage_size} (shape={tensor.shape})"
                )
                tensor._typed_storage()._resize_(size.numel())
