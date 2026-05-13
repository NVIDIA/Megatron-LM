import dataclasses
from typing import Dict, List, Optional, Tuple

import torch

from .utils import ParamGroupIdx


@dataclasses.dataclass
class Bucket:
    data: torch.Tensor


class BucketAllocator:
    """Interface for allocating and freeing temporary buckets."""

    def allocate(
        self, param_group_id: ParamGroupIdx, size: int, dtype: torch.dtype, device: torch.device
    ) -> Bucket:
        """Allocate a bucket for the given param group."""
        raise NotImplementedError

    def free(self, param_group_id: ParamGroupIdx) -> None:
        """Free the bucket associated with the given param group."""
        raise NotImplementedError


class TemporaryBucketAllocator(BucketAllocator):
    """Manages temporary flat buffers keyed by param_group_id.

    Used by DataParallelBuffer for unshard (all-gather) and gradient
    reduction (reduce-scatter) operations.
    """

    def __init__(self):
        super().__init__()
        self.buckets = {}

    def allocate(
        self, param_group_id: ParamGroupIdx, size: int, dtype: torch.dtype, device: torch.device
    ) -> Bucket:
        if param_group_id not in self.buckets:
            self.buckets[param_group_id] = Bucket(
                data=torch.empty(size, dtype=dtype, device=device)
            )
        return self.buckets[param_group_id]

    def free(self, param_group_id: ParamGroupIdx) -> None:
        if param_group_id in self.buckets:
            _free_storage(self.buckets[param_group_id].data)
            del self.buckets[param_group_id]


class StorageFreeingBucketAllocator(BucketAllocator):
    """Manages temporary flat buffers keyed by param_group_id, and frees the
    underlying storage after use without deleting the bucket entry, so the
    same tensor object can be reused on the next allocation.
    """

    def __init__(self):
        super().__init__()
        self.buckets = {}

    def allocate(
        self, param_group_id: ParamGroupIdx, size: int, dtype: torch.dtype, device: torch.device
    ) -> Bucket:
        if param_group_id not in self.buckets:
            self.buckets[param_group_id] = Bucket(
                data=torch.empty(size, dtype=dtype, device=device)
            )
            return self.buckets[param_group_id]
        _alloc_storage(self.buckets[param_group_id].data, torch.Size([size]))
        return self.buckets[param_group_id]

    def free(self, param_group_id: ParamGroupIdx) -> None:
        if param_group_id in self.buckets:
            _free_storage(self.buckets[param_group_id].data)


class TracePoolAllocator(BucketAllocator):
    """Two-phase bucket allocator that eliminates per-call ``torch.empty`` overhead.

    **Design**

    The FSDP framework allocates and frees temporary flat buffers (for
    all-gather input/output and gradient accumulation) in a deterministic,
    repeatable order across micro-batches.  ``TracePoolAllocator`` exploits
    this by profiling one pass and then serving all subsequent passes from
    a pre-allocated pool.

    **Phase 1 — Trace** (``plan()`` not yet called)

    Behaves like ``TemporaryBucketAllocator``: ``allocate`` creates a
    ``torch.empty`` bucket on first use, ``free`` releases its storage.
    Additionally, every alloc/free call is recorded as a ``_TraceEvent``
    with a monotonic ``seq`` number, and metadata ``(size, dtype, device)``
    is stored per ``param_group_id`` for later planning.

    **Phase 2 — Plan** (``plan()``)

    The trace is replayed to extract *intervals*: for each alloc/free pair
    an ``_Interval(alloc_seq, free_seq, size)`` is built.  Intervals are
    grouped by ``(dtype, device)`` and then colored with a greedy
    left-edge algorithm:

    1. Sort intervals by ``alloc_seq``.
    2. For each interval, try to reuse a *slot* whose previous occupant
       freed before this interval starts (``slot_free_seq < alloc_seq``).
    3. If no slot is free, allocate a new one.
    4. Grow the slot's capacity to ``max(size, current)``.
    5. Record the assignment: append the slot index to the per-pg_id list
       in ``_slot_map``.

    After coloring, slots are laid out contiguously and a single
    ``torch.empty`` per ``(dtype, device)`` group is allocated.

    **Phase 3 — Optimized** (after ``plan()``)

    ``allocate`` returns a ``Bucket`` with a slice-view into the pool;
    ``free`` marks the slot as unused but never releases storage.  Because
    the same ``param_group_id`` can appear in multiple intervals (e.g.,
    forward unshard → free → backward unshard → free), ``_slot_map`` maps
    each pg_id to a **list** of slot indices in alloc order.  A per-pg_id
    ``_slot_cursors`` counter tracks which index to consume next.  Call
    ``reset_cursor()`` at the start of each micro-batch to rewind all
    cursors to 0.

    The trace pattern must be **repeatable** — the same alloc/free call
    sequence is expected every micro-batch.
    """

    # -- Inner types ---------------------------------------------------- #

    class _Slot:
        """A contiguous slice of the pool tensor assigned to one or more
        non-overlapping intervals."""

        __slots__ = ("offset", "size", "dtype", "device", "in_use")

        def __init__(self, offset: int, size: int, dtype: torch.dtype, device: torch.device):
            self.offset = offset
            self.size = size
            self.dtype = dtype
            self.device = device
            self.in_use = False

    @dataclasses.dataclass
    class _TraceEvent:
        """A single alloc or free recorded during the trace phase."""
        seq: int
        op: str  # "alloc" | "free"
        param_group_id: ParamGroupIdx

    @dataclasses.dataclass
    class _Interval:
        """An allocation's lifetime: from alloc_seq to free_seq with a given size."""
        param_group_id: ParamGroupIdx
        size: int
        alloc_seq: int
        free_seq: int

    # -- Init ----------------------------------------------------------- #

    def __init__(self) -> None:
        super().__init__()
        # Phase bookkeeping
        self._phase: str = "trace"                       # "trace" | "optimized"
        self._seq: int = 0                               # monotonic alloc/free counter

        # Trace state
        self._trace: List["TracePoolAllocator._TraceEvent"] = []
        self._trace_meta: Dict[ParamGroupIdx, Tuple[int, torch.dtype, torch.device]] = {}
        self._buckets: Dict[ParamGroupIdx, Bucket] = {}   # only used in trace phase

        # Pool state — populated by plan(), used in optimized phase
        self._pools: Dict[Tuple[torch.dtype, torch.device], torch.Tensor] = {}
        self._slot_map: Dict[ParamGroupIdx, List[int]] = {}  # pg_id -> [slot indices]
        self._slot_cursors: Dict[ParamGroupIdx, int] = {}    # pg_id -> next index to use
        self._slots: List["TracePoolAllocator._Slot"] = []

    # -- Phase 1: trace -------------------------------------------------- #

    def allocate(
        self, param_group_id: ParamGroupIdx, size: int, dtype: torch.dtype, device: torch.device
    ) -> Bucket:
        """Dispatch to trace or pool path depending on phase."""
        if self._phase != "optimized":
            return self._trace_allocate(param_group_id, size, dtype, device)
        return self._pool_allocate(param_group_id, size, dtype, device)

    def free(self, param_group_id: ParamGroupIdx) -> None:
        """Dispatch to trace or pool path depending on phase."""
        if self._phase != "optimized":
            self._trace_free(param_group_id)
        else:
            self._pool_free(param_group_id)

    def _trace_allocate(
        self, param_group_id: ParamGroupIdx, size: int, dtype: torch.dtype, device: torch.device
    ) -> Bucket:
        """Trace-phase allocate: record the event and create a bucket on first use.

        Duplicate allocs (without an intervening free) do NOT generate
        new trace events — they are no-ops that return the existing bucket.
        """
        if param_group_id not in self._buckets:
            self._trace.append(
                self._TraceEvent(seq=self._seq, op="alloc", param_group_id=param_group_id)
            )
            self._seq += 1
            self._trace_meta[param_group_id] = (size, dtype, device)
            self._buckets[param_group_id] = Bucket(
                data=torch.empty(size, dtype=dtype, device=device)
            )
        return self._buckets[param_group_id]

    def _trace_free(self, param_group_id: ParamGroupIdx) -> None:
        """Trace-phase free: record the event and release the bucket storage."""
        self._trace.append(
            self._TraceEvent(seq=self._seq, op="free", param_group_id=param_group_id)
        )
        self._seq += 1
        if param_group_id in self._buckets:
            _free_storage(self._buckets[param_group_id].data)
            del self._buckets[param_group_id]

    # -- Phase 2: plan --------------------------------------------------- #

    def plan(self) -> int:
        """Build the static pool from the recorded trace.

        1. Replay the trace to pair alloc/free events into ``_Interval``\ s.
        2. Group intervals by ``(dtype, device)``.
        3. Color each group with the greedy left-edge algorithm.
        4. Allocate one flat pool tensor per group.

        Returns:
            Total pool size in **elements** (sum across all groups).
            Multiply by ``element_size(dtype)`` for bytes.
        """
        assert self._phase == "trace", "plan() can only be called in trace phase"
        assert len(self._trace) > 0, "empty trace — nothing to plan"

        # ---- step 1: build intervals from alloc/free pairs ----
        alloc_stack: Dict[ParamGroupIdx, List[int]] = {}  # pg_id -> [alloc_seq, ...]
        intervals: List["TracePoolAllocator._Interval"] = []

        for ev in self._trace:
            pg_id = ev.param_group_id
            if ev.op == "alloc":
                alloc_stack.setdefault(pg_id, []).append(ev.seq)
            else:  # "free"
                if pg_id in alloc_stack and alloc_stack[pg_id]:
                    alloc_seq = alloc_stack[pg_id].pop(0)
                    meta = self._trace_meta.get(pg_id)
                    if meta is not None:
                        size, dtype, device = meta
                        intervals.append(
                            self._Interval(
                                param_group_id=pg_id,
                                size=size,
                                alloc_seq=alloc_seq,
                                free_seq=ev.seq,
                            )
                        )

        assert len(intervals) > 0, "no paired alloc/free intervals found in trace"

        # ---- step 2 & 3: color and allocate ----
        return self._assign_pool(intervals)

    def _assign_pool(self, intervals: List["TracePoolAllocator._Interval"]) -> int:
        """Group intervals by (dtype, device), color each group, sum sizes."""
        groups: Dict[Tuple[torch.dtype, torch.device], List["TracePoolAllocator._Interval"]] = {}
        for iv in intervals:
            meta = self._trace_meta[iv.param_group_id]
            key = (meta[1], meta[2])  # (dtype, device)
            groups.setdefault(key, []).append(iv)

        # Clear any previous plan state before rebuilding
        self._slot_map.clear()
        self._slot_cursors.clear()
        self._slots.clear()
        self._pools.clear()

        total_elems = 0
        for (dtype, device), group in groups.items():
            total_elems += self._color_group(group, dtype, device)

        self._phase = "optimized"
        return total_elems

    def _color_group(
        self,
        intervals: List["TracePoolAllocator._Interval"],
        dtype: torch.dtype,
        device: torch.device,
    ) -> int:
        """Greedy left-edge interval coloring for one (dtype, device) group.

        Algorithm:

        1. Sort intervals by ``alloc_seq``.
        2. Maintain a *free-list* of ``(slot_index, free_seq)`` — slots that
           become free at ``free_seq``.
        3. For each interval:
           a. Scan the free-list for a slot whose ``free_seq < alloc_seq``.
              If found, reuse it (grow its size if needed) and update its
              free time to this interval's ``free_seq``.
           b. If no slot is free, create a new one.
           c. Append the assigned slot index to ``_slot_map[pg_id]``.

        After all intervals are colored, slots are laid out contiguously
        and a single ``torch.empty`` is issued for the group.

        Complexity: O(n²) worst-case (linear scan per interval), negligible
        for the typical tens-to-hundreds of param groups per dtype/device.
        """
        intervals = sorted(intervals, key=lambda iv: iv.alloc_seq)

        # free_slots: list of (local_slot_index, free_seq)
        free_slots: List[Tuple[int, int]] = []
        group_slots: List["TracePoolAllocator._Slot"] = []    # slots for this group
        local_to_global: Dict[int, int] = {}                   # local index -> global index

        for iv in intervals:
            assigned = False
            # Try to reuse an existing slot whose previous occupant has already freed
            for i, (slot_idx, slot_free_seq) in enumerate(free_slots):
                if slot_free_seq < iv.alloc_seq:
                    slot = group_slots[slot_idx]
                    if iv.size > slot.size:
                        slot.size = iv.size                   # grow if needed
                    free_slots[i] = (slot_idx, iv.free_seq)   # update free time
                    self._slot_map.setdefault(iv.param_group_id, []).append(
                        local_to_global[slot_idx]
                    )
                    assigned = True
                    break

            if not assigned:
                # No reusable slot — allocate a new one
                local_idx = len(group_slots)
                global_idx = len(self._slots)
                local_to_global[local_idx] = global_idx
                slot = self._Slot(offset=0, size=iv.size, dtype=dtype, device=device)
                group_slots.append(slot)
                self._slots.append(slot)
                free_slots.append((local_idx, iv.free_seq))
                self._slot_map.setdefault(iv.param_group_id, []).append(global_idx)

        # Lay out slots contiguously within the group pool
        offset = 0
        for slot in group_slots:
            slot.offset = offset
            offset += slot.size

        if offset > 0:
            self._pools[(dtype, device)] = torch.empty(offset, dtype=dtype, device=device)
        return offset

    # -- Phase 3: optimized runtime ------------------------------------- #
    #
    # Each micro-batch replays the same alloc/free sequence.  Because a
    # single param_group_id may appear multiple times (e.g., forward
    # unshard → free → backward unshard → free), ``_slot_map[pg_id]`` is
    # a **list** of slot indices in alloc order.  A per-pg_id cursor in
    # ``_slot_cursors`` tracks which index to consume next.  Between
    # micro-batches, ``reset_cursor()`` rewinds all cursors to 0.

    def _pool_allocate(
        self, param_group_id: ParamGroupIdx, size: int, dtype: torch.dtype, device: torch.device
    ) -> Bucket:
        """Return a ``Bucket`` whose data is a slice of the pre-allocated pool.

        Advances the per-pg_id slot cursor after consuming the slot.
        """
        slot_list = self._slot_map[param_group_id]
        cursor = self._slot_cursors.get(param_group_id, 0)
        assert cursor < len(slot_list), (
            f"no slot available for pg={param_group_id} "
            f"(cursor={cursor}, slots={slot_list})"
        )
        slot_idx = slot_list[cursor]
        self._slot_cursors[param_group_id] = cursor + 1

        slot = self._slots[slot_idx]
        assert not slot.in_use, (
            f"slot {slot_idx} already in use (pg={param_group_id}, "
            f"cursor={cursor}, slot_list={slot_list})"
        )
        assert size <= slot.size, (
            f"requested {size} > slot capacity {slot.size} (pg={param_group_id})"
        )
        pool = self._pools[(slot.dtype, slot.device)]
        slot.in_use = True
        self._seq += 1
        return Bucket(data=pool[slot.offset : slot.offset + size])

    def _pool_free(self, param_group_id: ParamGroupIdx) -> None:
        """Mark the most recently allocated slot for this pg_id as free.

        Double-frees are silently ignored (idempotent).
        """
        # The last allocated slot for this pg_id is at cursor - 1
        slot_idx = self._slot_map[param_group_id][
            self._slot_cursors.get(param_group_id, 1) - 1
        ]
        slot = self._slots[slot_idx]
        self._seq += 1
        if not slot.in_use:
            return
        slot.in_use = False

    # -- Debug ---------------------------------------------------------- #

    def dump_trace(self) -> str:
        """Return a human-readable dump of the trace and pool plan.

        Useful for debugging slot-conflict errors (e.g., "slot already in use").
        """
        lines = []
        lines.append(f"=== TracePoolAllocator (phase={self._phase}, seq={self._seq}) ===")
        lines.append(f"trace events: {len(self._trace)}")
        for ev in self._trace:
            meta = self._trace_meta.get(ev.param_group_id)
            size_str = f"size={meta[0]}" if meta else "size=?"
            dtype_str = f"dtype={meta[1]}" if meta else "dtype=?"
            device_str = f"device={meta[2]}" if meta else "device=?"
            lines.append(
                f"  seq={ev.seq:>4}  {ev.op:>5}  pg={ev.param_group_id}  "
                f"{size_str}  {dtype_str}  {device_str}"
            )

        if self._phase == "optimized":
            lines.append(f"\nslots: {len(self._slots)}")
            for i, slot in enumerate(self._slots):
                lines.append(
                    f"  slot[{i}]: offset={slot.offset} size={slot.size} "
                    f"dtype={slot.dtype} device={slot.device}"
                )
            lines.append("\nslot_map (pg_id -> [slot indices]):")
            for pg_id, slot_list in self._slot_map.items():
                cursor = self._slot_cursors.get(pg_id, 0)
                lines.append(f"  {pg_id} -> {slot_list}  cursor={cursor}")

        return "\n".join(lines)

    # -- Lifecycle ------------------------------------------------------- #

    def reset_cursor(self) -> None:
        """Reset all slot cursors to 0 for the next iteration / micro-batch.

        Must be called between micro-batches in the optimized phase so that
        the alloc/free sequence replays from the start of the slot lists.
        """
        self._slot_cursors.clear()
        self._seq = 0

    def reset(self) -> None:
        """Reset to trace phase, discarding the pool and all recorded state."""
        self._phase = "trace"
        self._seq = 0
        self._trace.clear()
        self._trace_meta.clear()
        self._buckets.clear()
        self._pools.clear()
        self._slot_map.clear()
        self._slot_cursors.clear()
        self._slots.clear()

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def total_pool_bytes(self) -> int:
        """Total pool size in bytes across all dtype/device groups."""
        total = 0
        for (dtype, _), pool in self._pools.items():
            total += pool.numel() * pool.element_size()
        return total


def _free_storage(tensor: torch.Tensor) -> None:
    """Free the underlying storage of ``tensor`` by resizing it to 0."""
    with torch.no_grad():
        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
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
        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
            already_allocated = tensor._typed_storage()._size() == size.numel()
            if not already_allocated:
                tensor_storage_size = tensor._typed_storage()._size()
                assert tensor_storage_size == 0, (
                    "Tensor storage should have been resized to 0 but got "
                    f"{tensor_storage_size} (shape={tensor.shape})"
                )
                tensor._typed_storage()._resize_(size.numel())
