# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Runtime slot allocation for Megatron-owned chunk CUDA graphs."""

from collections import deque
from typing import Sequence, Tuple


class ChunkCudaGraphRuntimeSlots:
    """FIFO slot allocator for in-flight chunk CUDA graph microbatches."""

    def __init__(self, num_slots: int):
        assert num_slots >= 1, "num_slots must be >= 1"
        self.num_slots = num_slots
        self.reset()

    def reset(self):
        """Reset slot state at the start of a forward-backward iteration."""
        self.available_slots = deque(range(self.num_slots))
        self.live_slots_by_microbatch = {}

    def forward(self, microbatch_id: int) -> int:
        """Reserve a slot for a forward microbatch."""
        microbatch_id = int(microbatch_id)
        assert (
            microbatch_id not in self.live_slots_by_microbatch
        ), f"Forward called twice for microbatch {microbatch_id} before backward."
        assert (
            self.available_slots
        ), f"No free chunk CUDA graph slot for microbatch {microbatch_id}."
        slot = self.available_slots.popleft()
        self.live_slots_by_microbatch[microbatch_id] = slot
        return slot

    def backward(self, microbatch_id: int) -> int:
        """Release the slot owned by a backward microbatch."""
        microbatch_id = int(microbatch_id)
        assert (
            microbatch_id in self.live_slots_by_microbatch
        ), f"Backward called for microbatch {microbatch_id} before its forward."
        slot = self.live_slots_by_microbatch.pop(microbatch_id)
        self.available_slots.append(slot)
        return slot


def get_chunk_cuda_graph_topology_probe_microbatch_counts(
    pipeline_parallel_size, num_model_chunks=1, microbatch_group_size_per_vp_stage=None
):
    """Return microbatch counts that expose all legal VPP tail-group shapes."""
    pipeline_parallel_size = int(pipeline_parallel_size)
    num_model_chunks = max(1, int(num_model_chunks))
    assert pipeline_parallel_size >= 1

    if pipeline_parallel_size == 1:
        return (1,)

    group_size = microbatch_group_size_per_vp_stage
    if group_size is None:
        group_size = pipeline_parallel_size
    group_size = int(group_size)
    assert group_size >= pipeline_parallel_size

    probe_count = max(
        pipeline_parallel_size * num_model_chunks * 4, group_size * num_model_chunks * 2
    )
    full_group_count = ((probe_count + group_size - 1) // group_size) * group_size
    if num_model_chunks == 1:
        return (full_group_count,)
    return (full_group_count,) + tuple(
        full_group_count + remainder for remainder in range(pipeline_parallel_size, group_size)
    )


def get_chunk_cuda_graph_slot_counts_from_schedule(
    num_warmup_microbatches: int, num_model_chunks: int, schedule_table: Sequence[Tuple[int, int]]
):
    """Return each chunk's maximum in-flight count for a rank-local PP/VPP schedule."""
    num_warmup_microbatches = int(num_warmup_microbatches)
    num_model_chunks = int(num_model_chunks)
    assert num_model_chunks >= 1
    assert 0 <= num_warmup_microbatches <= len(schedule_table)

    forwards = tuple(
        ("forward", model_chunk_id, microbatch_id)
        for microbatch_id, model_chunk_id in schedule_table
    )
    # Megatron traverses virtual chunks in reverse during backward.
    backwards = tuple(
        ("backward", num_model_chunks - model_chunk_id - 1, microbatch_id)
        for microbatch_id, model_chunk_id in schedule_table
    )
    operations = list(forwards[:num_warmup_microbatches])
    for index in range(num_warmup_microbatches, len(forwards)):
        operations.extend((forwards[index], backwards[index - num_warmup_microbatches]))
    if num_warmup_microbatches:
        operations.extend(backwards[-num_warmup_microbatches:])

    outstanding = [0] * num_model_chunks
    required_slots = [0] * num_model_chunks
    for op, chunk_id, _ in operations:
        assert 0 <= chunk_id < num_model_chunks
        outstanding[chunk_id] += 1 if op == "forward" else -1
        assert outstanding[chunk_id] >= 0, "Invalid PP/VPP schedule: backward before forward."
        required_slots[chunk_id] = max(required_slots[chunk_id], outstanding[chunk_id])
    assert not any(outstanding), "Invalid PP/VPP schedule: live microbatches did not drain."
    return tuple(max(1, count) for count in required_slots)
