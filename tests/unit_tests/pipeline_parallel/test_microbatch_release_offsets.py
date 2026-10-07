# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU checks for interleaved pipeline release counts and buffered activation indexing."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.pipeline_parallel.schedules import (
    _get_microbatch_release_offsets,
    get_pp_rank_microbatches,
    get_schedule_table,
)


def _reference_offsets(schedule_table, chunks, warmup, forward_only):
    """Preserve the original exclusive-prefix counting rule as a reference."""
    chunk_ids = tuple(chunk for _, chunk in schedule_table)
    result = []
    for step, (_, chunk) in enumerate(schedule_table):
        if forward_only:
            result.append(chunk_ids[:step].count(chunk))
        elif step < warmup:
            result.append(0)
        else:
            result.append(chunk_ids[: step - warmup].count(chunks - chunk - 1))
    return result


@pytest.mark.parametrize("microbatches", [1, 2, 5, 8, 17, 128])
@pytest.mark.parametrize("chunks", [1, 2, 4, 8])
@pytest.mark.parametrize("group_size", [1, 2, 3, 8, 16])
def test_release_offsets_match_prefix_counts(microbatches, chunks, group_size):
    """Match every forward position, including partial groups and warmup boundaries."""
    table = get_schedule_table(microbatches, chunks, group_size)
    for warmup in sorted({0, 1, len(table) // 2, len(table), len(table) + 1}):
        for forward_only in (False, True):
            assert _get_microbatch_release_offsets(table, chunks, warmup, forward_only) == (
                _reference_offsets(table, chunks, warmup, forward_only)
            )


@pytest.mark.parametrize(
    "pp_size,microbatches,chunks,group_size",
    [(2, 5, 2, 3), (2, 8, 4, 2), (4, 12, 3, 8), (4, 32, 4, 4), (8, 8, 2, 8), (8, 32, 8, 8)],
)
@pytest.mark.parametrize("forward_only", [False, True])
@pytest.mark.parametrize("overlap_moe", [False, True])
def test_buffered_activation_indexing(
    pp_size, microbatches, chunks, group_size, forward_only, overlap_moe
):
    """Select the exact activation after forward/backward pops on every pipeline rank."""
    table = get_schedule_table(microbatches, chunks, group_size)
    for rank in range(pp_size):
        communicator = SimpleNamespace(
            pp_group=SimpleNamespace(size=lambda: pp_size, rank=lambda: rank),
            virtual_pipeline_model_parallel_size=chunks,
        )
        total, _, warmup, _ = get_pp_rank_microbatches(
            microbatches,
            chunks,
            group_size,
            forward_only=forward_only,
            overlap_moe_expert_parallel_comm=overlap_moe,
            p2p_communicator=communicator,
        )
        offsets = _get_microbatch_release_offsets(table, chunks, warmup, forward_only)
        # Prebuffer identifiable CPU activations. Received inputs keep this order;
        # actual scheduling only limits how many future entries have arrived.
        inputs = [
            [torch.tensor([chunk, microbatch]) for microbatch in range(microbatches)]
            for chunk in range(chunks)
        ]
        for step, (microbatch, chunk) in enumerate(table):
            index = microbatch - offsets[step]
            assert index >= 0
            assert torch.equal(inputs[chunk][index], torch.tensor([chunk, microbatch]))
            if forward_only:
                inputs[chunk].pop(0)
            elif step >= warmup:
                backward_step = step - warmup
                backward_microbatch, forward_chunk = table[backward_step]
                backward_chunk = chunks - forward_chunk - 1
                assert torch.equal(
                    inputs[backward_chunk].pop(0),
                    torch.tensor([backward_chunk, backward_microbatch]),
                )
        if not forward_only:
            for backward_step in range(total - warmup, total):
                backward_microbatch, forward_chunk = table[backward_step]
                backward_chunk = chunks - forward_chunk - 1
                assert torch.equal(
                    inputs[backward_chunk].pop(0),
                    torch.tensor([backward_chunk, backward_microbatch]),
                )
        assert all(not buffered for buffered in inputs)


def test_empty_schedule():
    """Return no offsets for an empty schedule."""
    assert _get_microbatch_release_offsets([], 2, 0, False) == []
