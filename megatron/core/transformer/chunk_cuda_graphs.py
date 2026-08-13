# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Schedule-derived slot planning for Megatron-owned chunk CUDA graphs."""

from collections import deque
from dataclasses import dataclass
from math import ceil
from typing import Optional, Sequence, Tuple


class ChunkCudaGraphRuntimeSlots:
    """Runtime FIFO slot allocator for Megatron-owned chunk CUDA graphs."""

    def __init__(self, num_slots: int):
        assert num_slots >= 1, "num_slots must be >= 1"
        self.num_slots = num_slots
        self.reset()

    def reset(self):
        """Reset live slot state at the start of a forward-backward iteration."""
        self.available_slots = deque(range(self.num_slots))
        self.live_slots_by_microbatch = {}
        self.current_slot = None
        self.current_op = None

    def forward(self, microbatch_id: int) -> int:
        """Reserve a graph slot for a forward microbatch."""
        microbatch_id = int(microbatch_id)
        assert microbatch_id not in self.live_slots_by_microbatch, (
            "Invalid chunk CUDA graph runtime slot state: forward called twice for "
            f"microbatch {microbatch_id} before backward."
        )
        assert self.available_slots, (
            "No free chunk CUDA graph runtime slot for forward microbatch "
            f"{microbatch_id}. Increase cuda_graph_num_microbatch_slots."
        )
        slot = self.available_slots.popleft()
        self.live_slots_by_microbatch[microbatch_id] = slot
        self.current_slot = slot
        self.current_op = "forward"
        return slot

    def backward(self, microbatch_id: int) -> int:
        """Release and return the graph slot owned by a backward microbatch."""
        microbatch_id = int(microbatch_id)
        assert microbatch_id in self.live_slots_by_microbatch, (
            "Invalid chunk CUDA graph runtime slot state: backward called for "
            f"microbatch {microbatch_id} before its forward."
        )
        slot = self.live_slots_by_microbatch.pop(microbatch_id)
        self.available_slots.append(slot)
        self.current_slot = slot
        self.current_op = "backward"
        return slot


@dataclass(frozen=True)
class ChunkCudaGraphScheduleSignature:
    """Paged-stash actions captured by one chunk forward/backward graph pair.

    Targets are ``(chunk_id, physical_slot_id)``. Absolute microbatch ids are deliberately
    excluded. The signature records which variants are semantically equivalent at the schedule
    boundary; paged-stash's Python queues may still require distinct logical graph runners.
    """

    forward_wait_for_stash: bool
    forward_action: str
    forward_target: Optional[Tuple[int, int]]
    backward_wait_for_reload: bool
    backward_action: str
    backward_target: Optional[Tuple[int, int]]


@dataclass(frozen=True)
class ChunkCudaGraphSlotPlan:
    """Per-order-entry slot assignment for a chunk CUDA graph schedule.

    ``slot_ids`` is parallel to ``order``. Integer forward/backward entries get a slot id local
    to their chunk. Non-integer entries, such as delayed wgrad sub-steps, are marked as ``aux`` and
    intentionally left without a slot in this initial plan.
    """

    order: Tuple[float, ...]
    stage_order: Tuple[str, ...]
    num_slots_per_chunk: Tuple[int, ...]
    slot_ids: Tuple[Optional[int], ...]
    chunk_ids: Tuple[Optional[int], ...]
    op_types: Tuple[str, ...]
    virtual_microbatch_ids: Tuple[Optional[int], ...]
    microbatch_ids: Tuple[Optional[int], ...]
    forward_slot_by_virtual_microbatch: Tuple[Optional[int], ...]
    backward_slot_by_virtual_microbatch: Tuple[Optional[int], ...]
    forward_slot_by_chunk_microbatch: Tuple[Tuple[Optional[int], ...], ...]
    backward_slot_by_chunk_microbatch: Tuple[Tuple[Optional[int], ...], ...]
    runner_signature_by_chunk_microbatch: Tuple[
        Tuple[Optional[ChunkCudaGraphScheduleSignature], ...], ...
    ]

    def get_forward_slot(self, chunk_id: int, microbatch_id: int) -> Optional[int]:
        """Return the forward slot for ``chunk_id`` and local ``microbatch_id``."""
        if chunk_id >= len(self.forward_slot_by_chunk_microbatch):
            return None
        slots = self.forward_slot_by_chunk_microbatch[chunk_id]
        if microbatch_id >= len(slots):
            return None
        return slots[microbatch_id]

    def get_backward_slot(self, chunk_id: int, microbatch_id: int) -> Optional[int]:
        """Return the backward slot for ``chunk_id`` and local ``microbatch_id``."""
        if chunk_id >= len(self.backward_slot_by_chunk_microbatch):
            return None
        slots = self.backward_slot_by_chunk_microbatch[chunk_id]
        if microbatch_id >= len(slots):
            return None
        return slots[microbatch_id]

    def get_runner_signature(
        self, chunk_id: int, microbatch_id: int
    ) -> Optional[ChunkCudaGraphScheduleSignature]:
        """Return the schedule-dependent runner signature for one chunk microbatch."""
        if chunk_id >= len(self.runner_signature_by_chunk_microbatch):
            return None
        signatures = self.runner_signature_by_chunk_microbatch[chunk_id]
        if microbatch_id >= len(signatures):
            return None
        return signatures[microbatch_id]


def _is_integer_schedule_entry(schedule_entry):
    """Return whether a schedule entry represents a normal forward/backward graph."""
    return ceil(schedule_entry) == schedule_entry


def get_cuda_graph_schedule_stage_order_from_counts(
    num_warmup_microbatches: int, num_scheduled_microbatches: int
):
    """Return exact warmup/steady/cooldown stages for Megatron's F,B schedule order."""
    num_warmup_microbatches = int(num_warmup_microbatches)
    num_scheduled_microbatches = int(num_scheduled_microbatches)
    assert num_warmup_microbatches >= 0, "num_warmup_microbatches must be >= 0"
    assert num_scheduled_microbatches >= num_warmup_microbatches, (
        "num_scheduled_microbatches must be >= num_warmup_microbatches: "
        f"{num_scheduled_microbatches} < {num_warmup_microbatches}"
    )
    num_steady_microbatches = num_scheduled_microbatches - num_warmup_microbatches
    return (
        ("warmup",) * num_warmup_microbatches
        + ("steady",) * (num_steady_microbatches * 2)
        + ("cooldown",) * num_warmup_microbatches
    )


def _get_chunk_index(schedule_entry, num_model_chunks):
    """Convert a signed 1-indexed schedule entry to a 0-indexed chunk id."""
    chunk_idx = abs(int(schedule_entry)) - 1
    assert 0 <= chunk_idx < num_model_chunks, (
        f"Invalid chunk id in CUDA graph schedule: entry={schedule_entry}, "
        f"num_model_chunks={num_model_chunks}"
    )
    return chunk_idx


def get_required_num_microbatch_slots_per_chunk(order, num_model_chunks):
    """Infer the minimum live slot count required for each chunk in ``order``."""
    assert num_model_chunks >= 1, "num_model_chunks must be >= 1"
    outstanding = [0] * num_model_chunks
    max_outstanding = [0] * num_model_chunks

    for schedule_entry in order:
        if not _is_integer_schedule_entry(schedule_entry):
            continue

        chunk_idx = _get_chunk_index(schedule_entry, num_model_chunks)
        if schedule_entry > 0:
            outstanding[chunk_idx] += 1
            max_outstanding[chunk_idx] = max(max_outstanding[chunk_idx], outstanding[chunk_idx])
        else:
            outstanding[chunk_idx] -= 1
            assert outstanding[chunk_idx] >= 0, (
                "Invalid PP/VPP schedule: negative outstanding microbatches while "
                f"inferring chunk CUDA graph slots for chunk {chunk_idx}."
            )

    assert all(count == 0 for count in outstanding), (
        "Invalid PP/VPP schedule: outstanding microbatches did not drain to zero when "
        f"inferring chunk CUDA graph slots. outstanding={outstanding}"
    )
    return tuple(max(1, count) for count in max_outstanding)


def get_chunk_cuda_graph_topology_probe_microbatch_counts(
    pipeline_parallel_size,
    num_model_chunks=1,
    microbatch_group_size_per_vp_stage=None,
):
    """Return microbatch counts that expose every legal VPP tail-group shape.

    A single large, full-group schedule is insufficient: a legal partial final group can move the
    peak live-slot count between model chunks. Probe one saturated schedule for each legal
    ``num_microbatches % group_size`` value and take the per-chunk maximum across the resulting
    plans.
    """
    pipeline_parallel_size = int(pipeline_parallel_size)
    num_model_chunks = max(1, int(num_model_chunks))
    assert pipeline_parallel_size >= 1, "pipeline_parallel_size must be >= 1"

    if pipeline_parallel_size == 1:
        return (1,)

    group_size = microbatch_group_size_per_vp_stage
    if group_size is None:
        group_size = pipeline_parallel_size
    group_size = int(group_size)
    assert group_size >= 1, "microbatch_group_size_per_vp_stage must be >= 1"

    raw_probe_count = max(
        pipeline_parallel_size * num_model_chunks * 4,
        group_size * num_model_chunks * 2,
    )
    full_group_probe_count = (
        (raw_probe_count + group_size - 1) // group_size
    ) * group_size
    if num_model_chunks == 1:
        return (full_group_probe_count,)

    assert group_size >= pipeline_parallel_size, (
        "VPP microbatch_group_size_per_vp_stage must be at least the pipeline parallel size: "
        f"group_size={group_size}, pipeline_parallel_size={pipeline_parallel_size}"
    )
    return (full_group_probe_count,) + tuple(
        full_group_probe_count + remainder
        for remainder in range(pipeline_parallel_size, group_size)
    )


def _build_chunk_cuda_graph_slot_plan_from_entries(
    schedule_entries,
    num_model_chunks,
    stage_order=None,
    num_virtual_microbatches=0,
    num_slots_per_chunk=None,
):
    """Build a deterministic slot assignment from indexed schedule entries."""
    order = tuple(entry[0] for entry in schedule_entries)
    if stage_order is None:
        stage_order = tuple("unknown" for _ in order)
    else:
        stage_order = tuple(stage_order)
    assert len(stage_order) == len(
        order
    ), f"stage_order length must match order length: {len(stage_order)} != {len(order)}"

    required_num_slots_per_chunk = get_required_num_microbatch_slots_per_chunk(
        order, num_model_chunks
    )
    if num_slots_per_chunk is None:
        num_slots_per_chunk = required_num_slots_per_chunk
    elif isinstance(num_slots_per_chunk, int):
        num_slots_per_chunk = (num_slots_per_chunk,) * num_model_chunks
    else:
        num_slots_per_chunk = tuple(num_slots_per_chunk)
    assert len(num_slots_per_chunk) == num_model_chunks, (
        "num_slots_per_chunk must have one entry per model chunk: "
        f"{num_slots_per_chunk} vs {num_model_chunks}"
    )
    assert all(
        actual >= required
        for actual, required in zip(num_slots_per_chunk, required_num_slots_per_chunk)
    ), (
        "num_slots_per_chunk is smaller than the schedule requirement: "
        f"requested={num_slots_per_chunk}, required={required_num_slots_per_chunk}"
    )
    available_slots = [deque(range(num_slots)) for num_slots in num_slots_per_chunk]
    live_slots = [deque() for _ in num_slots_per_chunk]

    slot_ids = []
    chunk_ids = []
    op_types = []
    virtual_microbatch_ids = []
    microbatch_ids = []
    forward_slot_by_virtual_microbatch = [None] * num_virtual_microbatches
    backward_slot_by_virtual_microbatch = [None] * num_virtual_microbatches
    max_microbatch_id = max(
        (entry[2] for entry in schedule_entries if entry[2] is not None), default=-1
    )
    forward_slot_by_chunk_microbatch = [
        [None] * (max_microbatch_id + 1) for _ in range(num_model_chunks)
    ]
    backward_slot_by_chunk_microbatch = [
        [None] * (max_microbatch_id + 1) for _ in range(num_model_chunks)
    ]

    for schedule_entry, virtual_microbatch_id, microbatch_id in schedule_entries:
        virtual_microbatch_ids.append(virtual_microbatch_id)
        microbatch_ids.append(microbatch_id)
        if not _is_integer_schedule_entry(schedule_entry):
            slot_ids.append(None)
            chunk_ids.append(None)
            op_types.append("aux")
            continue

        chunk_idx = _get_chunk_index(schedule_entry, num_model_chunks)
        chunk_ids.append(chunk_idx)
        if schedule_entry > 0:
            assert available_slots[chunk_idx], (
                "Invalid chunk CUDA graph slot plan: no free slot for forward "
                f"entry={schedule_entry}, chunk={chunk_idx}"
            )
            slot_id = available_slots[chunk_idx].popleft()
            live_slots[chunk_idx].append(slot_id)
            slot_ids.append(slot_id)
            op_types.append("forward")
            if virtual_microbatch_id is not None:
                forward_slot_by_virtual_microbatch[virtual_microbatch_id] = slot_id
            if microbatch_id is not None:
                forward_slot_by_chunk_microbatch[chunk_idx][microbatch_id] = slot_id
        else:
            assert live_slots[chunk_idx], (
                "Invalid chunk CUDA graph slot plan: no live slot for backward "
                f"entry={schedule_entry}, chunk={chunk_idx}"
            )
            slot_id = live_slots[chunk_idx].popleft()
            available_slots[chunk_idx].append(slot_id)
            slot_ids.append(slot_id)
            op_types.append("backward")
            if virtual_microbatch_id is not None:
                backward_slot_by_virtual_microbatch[virtual_microbatch_id] = slot_id
            if microbatch_id is not None:
                backward_slot_by_chunk_microbatch[chunk_idx][microbatch_id] = slot_id

    assert all(len(slots) == 0 for slots in live_slots), (
        "Invalid chunk CUDA graph slot plan: live slots did not drain. "
        f"live_slots={[list(slots) for slots in live_slots]}"
    )

    runner_signature_by_chunk_microbatch = [
        [None] * (max_microbatch_id + 1) for _ in range(num_model_chunks)
    ]
    forward_index_by_chunk_microbatch = {}
    backward_index_by_chunk_microbatch = {}
    for idx, op_type in enumerate(op_types):
        if microbatch_ids[idx] is None or op_type == "aux":
            continue
        chunk_microbatch = (chunk_ids[idx], microbatch_ids[idx])
        index_table = (
            forward_index_by_chunk_microbatch
            if op_type == "forward"
            else backward_index_by_chunk_microbatch
        )
        assert chunk_microbatch not in index_table
        index_table[chunk_microbatch] = idx

    def adjacent_normal_index(index, step):
        index += step
        while 0 <= index < len(order):
            if op_types[index] != "aux":
                return index
            index += step
        return None

    def transition_target(index):
        return (chunk_ids[index], slot_ids[index])

    def chunk_microbatch(index):
        return (chunk_ids[index], microbatch_ids[index])

    def forward_action(index):
        next_index = adjacent_normal_index(index, 1)
        if next_index is None or op_types[next_index] == "forward":
            return "stash", None
        if chunk_microbatch(next_index) == chunk_microbatch(index):
            return "discard", None
        return "stash_prefetch", transition_target(next_index)

    def backward_action(index):
        next_index = adjacent_normal_index(index, 1)
        if next_index is not None and op_types[next_index] == "backward":
            return "reload", transition_target(next_index)
        return "none", None

    for chunk_microbatch_id, forward_index in forward_index_by_chunk_microbatch.items():
        backward_index = backward_index_by_chunk_microbatch[chunk_microbatch_id]
        microbatch_id = microbatch_ids[forward_index]
        chunk_id = chunk_ids[forward_index]
        assert microbatch_id is not None and chunk_id is not None
        assert chunk_ids[backward_index] == chunk_id
        assert microbatch_ids[backward_index] == microbatch_id

        previous_forward_index = adjacent_normal_index(forward_index, -1)
        forward_wait_for_stash = False
        if previous_forward_index is not None and op_types[previous_forward_index] == "forward":
            previous_action, _ = forward_action(previous_forward_index)
            forward_wait_for_stash = previous_action != "discard"

        previous_backward_index = adjacent_normal_index(backward_index, -1)
        backward_wait_for_reload = False
        if previous_backward_index is not None:
            if op_types[previous_backward_index] == "forward":
                previous_action, previous_target = forward_action(previous_backward_index)
                backward_wait_for_reload = (
                    previous_action == "stash_prefetch"
                    and previous_target == transition_target(backward_index)
                )
            else:
                previous_action, previous_target = backward_action(previous_backward_index)
                backward_wait_for_reload = (
                    previous_action == "reload"
                    and previous_target == transition_target(backward_index)
                )

        current_forward_action, forward_target = forward_action(forward_index)
        current_backward_action, backward_target = backward_action(backward_index)
        runner_signature_by_chunk_microbatch[chunk_id][microbatch_id] = (
            ChunkCudaGraphScheduleSignature(
                forward_wait_for_stash=forward_wait_for_stash,
                forward_action=current_forward_action,
                forward_target=forward_target,
                backward_wait_for_reload=backward_wait_for_reload,
                backward_action=current_backward_action,
                backward_target=backward_target,
            )
        )

    return ChunkCudaGraphSlotPlan(
        order=order,
        stage_order=stage_order,
        num_slots_per_chunk=num_slots_per_chunk,
        slot_ids=tuple(slot_ids),
        chunk_ids=tuple(chunk_ids),
        op_types=tuple(op_types),
        virtual_microbatch_ids=tuple(virtual_microbatch_ids),
        microbatch_ids=tuple(microbatch_ids),
        forward_slot_by_virtual_microbatch=tuple(forward_slot_by_virtual_microbatch),
        backward_slot_by_virtual_microbatch=tuple(backward_slot_by_virtual_microbatch),
        forward_slot_by_chunk_microbatch=tuple(
            tuple(slots) for slots in forward_slot_by_chunk_microbatch
        ),
        backward_slot_by_chunk_microbatch=tuple(
            tuple(slots) for slots in backward_slot_by_chunk_microbatch
        ),
        runner_signature_by_chunk_microbatch=tuple(
            tuple(signatures) for signatures in runner_signature_by_chunk_microbatch
        ),
    )


def build_chunk_cuda_graph_slot_plan(
    order: Sequence[float],
    num_model_chunks: int,
    stage_order: Optional[Sequence[str]] = None,
    num_slots_per_chunk=None,
):
    """Build a deterministic slot assignment for a chunk CUDA graph schedule.

    Forward entries reserve the first available slot for their chunk. Matching backward entries
    release the oldest live slot for that chunk, matching Megatron's per-chunk FIFO activation
    lifetime in 1F1B schedules.
    """
    schedule_entries = tuple((entry, None, None) for entry in order)
    return _build_chunk_cuda_graph_slot_plan_from_entries(
        schedule_entries, num_model_chunks, stage_order, num_slots_per_chunk=num_slots_per_chunk
    )


def build_chunk_cuda_graph_slot_plan_from_schedule(
    num_warmup_microbatches: int,
    num_model_chunks: int,
    schedule_table: Sequence[Tuple[int, int]],
    stage_order: Optional[Sequence[str]] = None,
    num_slots_per_chunk=None,
):
    """Build a slot plan while preserving virtual microbatch ids from Megatron's schedule table."""
    if stage_order is None:
        stage_order = get_cuda_graph_schedule_stage_order_from_counts(
            num_warmup_microbatches, len(schedule_table)
        )
    forward_entries = tuple(
        (model_chunk_id + 1, virtual_microbatch_id, microbatch_id)
        for virtual_microbatch_id, (microbatch_id, model_chunk_id) in enumerate(schedule_table)
    )
    backward_entries = tuple(
        (model_chunk_id - num_model_chunks, virtual_microbatch_id, microbatch_id)
        for virtual_microbatch_id, (microbatch_id, model_chunk_id) in enumerate(schedule_table)
    )

    schedule_entries = list(forward_entries[:num_warmup_microbatches])
    for idx in range(num_warmup_microbatches, len(forward_entries)):
        schedule_entries.append(forward_entries[idx])
        schedule_entries.append(backward_entries[idx - num_warmup_microbatches])
    if num_warmup_microbatches > 0:
        schedule_entries.extend(backward_entries[-num_warmup_microbatches:])

    return _build_chunk_cuda_graph_slot_plan_from_entries(
        tuple(schedule_entries),
        num_model_chunks,
        stage_order,
        num_virtual_microbatches=len(schedule_table),
        num_slots_per_chunk=num_slots_per_chunk,
    )
