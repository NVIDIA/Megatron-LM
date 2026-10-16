# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the exact early exit in the dynamic-CP planner.

The early exit must not change any plan, so every test compares
next_hdp_group_packing_aware with a frozen copy of the planner from before the
early exit was added. That copy pins the whole planner, not just the early exit:
an intentional change to the planner algorithm must update the copy in the same
commit (or drop the equivalence tests) instead of loosening the comparisons.
"""

import math
import random
from typing import Dict, List, Optional, Tuple

import pytest

from megatron.core.datasets import data_schedule_utils
from megatron.core.datasets.data_schedule import DefaultDynamicCPScheduler
from megatron.core.datasets.data_schedule_utils import (
    _DYNAMIC_CP_WORKLOAD_CAP_DELTA,
    dcp_gpus_needed,
    next_hdp_group_packing_aware,
)


def _reference_next_hdp_group_packing_aware(
    sample_seqlens: List[Tuple[int, int]],
    total_gpus: int,
    max_seq_len_per_rank: int,
    min_cp_size: int = 1,
) -> Tuple[List[List[int]], List[Tuple[int, int]], List[float], List[List[int]]]:
    """Frozen copy of next_hdp_group_packing_aware from before the exact early exit.

    It is that planner verbatim, minus the early exit. If the planner
    algorithm changes on purpose, update this copy in the same commit.
    """
    if not sample_seqlens:
        return (
            [[] for _ in range(total_gpus)],
            [],
            [0.0 for _ in range(total_gpus)],
            [[] for _ in range(total_gpus)],
        )

    def cp_min_fn(seq_len: int) -> int:
        return dcp_gpus_needed(seq_len, max_seq_len_per_rank, min_cp_size)

    def workload(seq_len: int, cp_size: int) -> float:
        return (seq_len * seq_len) / cp_size

    sample_seqlens = sorted(sample_seqlens, key=lambda x: x[1], reverse=True)
    local_tall = sample_seqlens[0][1]
    cap = float(local_tall) * float(max_seq_len_per_rank) * (1.0 + _DYNAMIC_CP_WORKLOAD_CAP_DELTA)

    micro_batches: List[List[int]] = [[] for _ in range(total_gpus)]
    exec_times: List[float] = [0.0 for _ in range(total_gpus)]
    sample_ids_per_gpu: List[List[int]] = [[] for _ in range(total_gpus)]
    packing_sequence_len: Dict[int, float] = {}

    gpu_group_id: List[Optional[int]] = [None] * total_gpus
    group_members: Dict[int, List[int]] = {}
    group_size: Dict[int, int] = {}
    next_gid = 0

    sample_id, seq_len = sample_seqlens[0]
    cp_size = cp_min_fn(seq_len)
    assert cp_size <= total_gpus, (
        f"Sequence length {seq_len} requires CP size {cp_size}, "
        f"but only {total_gpus} DPxCP ranks are available."
    )
    group_id = next_gid
    next_gid += 1
    members = list(range(cp_size))
    group_members[group_id] = members
    group_size[group_id] = cp_size
    packing_sequence_len[group_id] = seq_len / cp_size
    per_gpu_cost = workload(seq_len, cp_size)
    for rank in members:
        gpu_group_id[rank] = group_id
        micro_batches[rank].append(seq_len)
        exec_times[rank] += per_gpu_cost
        sample_ids_per_gpu[rank].append(sample_id)

    leftovers: List[Tuple[int, int]] = []
    for sample_id, seq_len in sample_seqlens[1:]:
        min_needed = cp_min_fn(seq_len)
        best = None

        cp_size = min_needed
        while cp_size <= total_gpus:
            per_gpu_cost = workload(seq_len, cp_size)

            for group_id, size in list(group_size.items()):
                if size != cp_size:
                    continue
                if packing_sequence_len.get(group_id, 0) + seq_len / cp_size > max_seq_len_per_rank:
                    continue
                members = group_members[group_id]
                member_set = set(members)
                projected_max = max(
                    time + per_gpu_cost if rank in member_set else time
                    for rank, time in enumerate(exec_times)
                )
                if projected_max <= cap and (best is None or projected_max < best[0]):
                    best = (projected_max, cp_size, "add", group_id, None)

            free_ranks = [
                rank
                for rank, assigned_group_id in enumerate(gpu_group_id)
                if assigned_group_id is None
            ]
            if len(free_ranks) >= cp_size:
                chosen_members = sorted(free_ranks, key=lambda rank: exec_times[rank])[:cp_size]
                chosen_set = set(chosen_members)
                projected_max = max(
                    time + per_gpu_cost if rank in chosen_set else time
                    for rank, time in enumerate(exec_times)
                )
                if projected_max <= cap and (best is None or projected_max < best[0]):
                    best = (projected_max, cp_size, "new", None, chosen_members)

            cp_size *= 2

        if best is None:
            leftovers.append((sample_id, seq_len))
            continue

        _, selected_cp_size, action, group_id, chosen_members = best
        per_gpu_cost = workload(seq_len, selected_cp_size)
        if action == "add":
            members = group_members[group_id]
            packing_sequence_len[group_id] += seq_len / selected_cp_size
            for rank in members:
                micro_batches[rank].append(seq_len)
                exec_times[rank] += per_gpu_cost
                sample_ids_per_gpu[rank].append(sample_id)
        else:
            group_id = next_gid
            next_gid += 1
            group_members[group_id] = chosen_members
            group_size[group_id] = selected_cp_size
            packing_sequence_len[group_id] = seq_len / selected_cp_size
            for rank in chosen_members:
                gpu_group_id[rank] = group_id
                micro_batches[rank].append(seq_len)
                exec_times[rank] += per_gpu_cost
                sample_ids_per_gpu[rank].append(sample_id)

    def fill_empty_gpus_once() -> bool:
        nonlocal micro_batches, exec_times, sample_ids_per_gpu

        empty_ranks = [rank for rank, micro_batch in enumerate(micro_batches) if not micro_batch]
        if not empty_ranks:
            return False
        assert all(
            not micro_batches[rank] for rank in range(empty_ranks[0], total_gpus)
        ), "fill_empty_gpus_once assumes empty ranks are contiguous at the tail"

        existing_group_sizes = set(group_size.values())
        if not existing_group_sizes:
            return False
        min_group_size = min(existing_group_sizes)
        next_power = min(min_group_size * 2, total_gpus)

        for group_id, size in list(group_size.items()):
            if size != min_group_size:
                continue

            members = group_members[group_id]
            needed_count = next_power - min_group_size
            group_start_rank = members[0]
            group_end_rank = members[-1]
            empty_rank = empty_ranks[0]
            if group_end_rank + 1 > empty_rank or group_end_rank + needed_count >= total_gpus:
                continue

            work_to_push = micro_batches[group_end_rank + 1 : empty_rank]
            exec_times_to_push = exec_times[group_end_rank + 1 : empty_rank]
            sample_ids_to_push = sample_ids_per_gpu[group_end_rank + 1 : empty_rank]

            new_micro_batches: List[List[int]] = [[] for _ in range(total_gpus)]
            new_exec_times: List[float] = [0.0 for _ in range(total_gpus)]
            new_sample_ids_per_gpu: List[List[int]] = [[] for _ in range(total_gpus)]

            for rank in range(group_start_rank):
                new_micro_batches[rank] = micro_batches[rank]
                new_exec_times[rank] = exec_times[rank]
                new_sample_ids_per_gpu[rank] = sample_ids_per_gpu[rank]

            for rank in range(group_start_rank, group_end_rank + needed_count + 1):
                new_micro_batches[rank] = list(micro_batches[group_end_rank])
                new_exec_times[rank] = sum(
                    workload(length, next_power) for length in micro_batches[group_end_rank]
                )
                new_sample_ids_per_gpu[rank] = list(sample_ids_per_gpu[group_end_rank])

            for idx, work in enumerate(work_to_push):
                target_rank = group_end_rank + needed_count + 1 + idx
                new_micro_batches[target_rank] = work
                new_exec_times[target_rank] = exec_times_to_push[idx]
                new_sample_ids_per_gpu[target_rank] = sample_ids_to_push[idx]

            group_size[group_id] = next_power
            group_members[group_id] = list(
                range(group_start_rank, group_end_rank + needed_count + 1)
            )
            for other_group_id in list(group_size.keys()):
                if other_group_id == group_id:
                    continue
                if min(group_members[other_group_id]) > group_end_rank:
                    group_members[other_group_id] = [
                        rank + needed_count for rank in group_members[other_group_id]
                    ]

            micro_batches = new_micro_batches
            exec_times = new_exec_times
            sample_ids_per_gpu = new_sample_ids_per_gpu
            return True

        return False

    def fill_with_full_dpxcp_group() -> None:
        nonlocal micro_batches, exec_times, sample_ids_per_gpu, leftovers

        selected: List[Tuple[int, int]] = []
        next_leftovers: List[Tuple[int, int]] = []
        packed_sequence_len = 0.0

        for sample_id, seq_len in sample_seqlens:
            per_rank_len = seq_len / total_gpus
            if packed_sequence_len + per_rank_len <= max_seq_len_per_rank:
                selected.append((sample_id, seq_len))
                packed_sequence_len += per_rank_len
            else:
                next_leftovers.append((sample_id, seq_len))

        assert selected, (
            "At least one sequence should fit in the full DPxCP group; "
            "try to increase 'max-seqlen-per-dp-cp-rank'."
        )

        selected_ids = [sample_id for sample_id, _ in selected]
        selected_lens = [seq_len for _, seq_len in selected]
        per_rank_work = sum(workload(seq_len, total_gpus) for _, seq_len in selected)

        micro_batches = [list(selected_lens) for _ in range(total_gpus)]
        exec_times = [per_rank_work for _ in range(total_gpus)]
        sample_ids_per_gpu = [list(selected_ids) for _ in range(total_gpus)]
        leftovers = next_leftovers

    while any(not micro_batch for micro_batch in micro_batches):
        if not fill_empty_gpus_once():
            fill_with_full_dpxcp_group()
            break

    return micro_batches, leftovers, exec_times, sample_ids_per_gpu


def _schedule(planner, sample_seqlens, total_gpus, max_seq_len_per_rank, min_cp_size=1):
    """Call ``planner`` on its own leftovers until none remain, as the DCP scheduler does.

    Returns the full output of every call, ending with the exception type and message
    if a call raised.
    """
    outputs = []
    try:
        while True:
            output = planner(sample_seqlens, total_gpus, max_seq_len_per_rank, min_cp_size)
            outputs.append(output)
            sample_seqlens = output[1]
            if not sample_seqlens:
                break
    except Exception as error:  # Both planners must fail in the same way.
        # pytest rewrites the asserts in this file, which appends the failed expression
        # to the reference's assertion messages, so only compare the first line.
        outputs.append((type(error), str(error).split("\n")[0]))
    return outputs


def _random_case(rng):
    """Draw one planner input with lognormal, fixed, tie-heavy, exact-fit or uniform lengths."""
    total_gpus = rng.choice([1, 2, 3, 4, 5, 6, 8, 12, 14, 16, 32, 64])
    min_cp_size = rng.choice([cp for cp in (1, 1, 2, 4) if cp <= total_gpus])
    max_seq_len_per_rank = rng.choice([7, 100, 1000, 4000, 4096])
    # The longest sample the largest power-of-two CP group can hold. Longer samples
    # make both planners raise, which test_matches_reference_on_edge_cases covers.
    max_len = max_seq_len_per_rank * 2 ** int(math.log2(total_gpus))
    num_samples = rng.randint(1, 96)

    kind = rng.choice(["lognormal", "fixed", "ties", "exact_fit", "uniform"])
    if kind == "lognormal":
        mu = math.log(max_seq_len_per_rank) + rng.uniform(-2.0, 1.0)
        sigma = rng.uniform(0.2, 1.5)
        lengths = [int(rng.lognormvariate(mu, sigma)) for _ in range(num_samples)]
    elif kind == "fixed":
        length = rng.choice(
            [
                1,
                max_seq_len_per_rank // 3,
                max_seq_len_per_rank // 2,
                max_seq_len_per_rank // 2 + 1,
                max_seq_len_per_rank * 31 // 32,
                max_seq_len_per_rank - 1,
                max_seq_len_per_rank,
                max_seq_len_per_rank * 2,
                rng.randint(1, max_len),
            ]
        )
        lengths = [length] * num_samples
    elif kind == "ties":
        palette = [rng.randint(1, max_len) for _ in range(rng.randint(1, 3))]
        lengths = [rng.choice(palette) for _ in range(num_samples)]
    elif kind == "exact_fit":
        # Lengths that fill a rank exactly, or miss by one token, at several CP sizes.
        lengths = [
            max_seq_len_per_rank * rng.choice([1, 2, 4]) // rng.choice([1, 2, 3, 4, 8])
            + rng.choice([-1, 0, 0, 1])
            for _ in range(num_samples)
        ]
    else:
        lengths = [rng.randint(1, max_len) for _ in range(num_samples)]
    lengths = [min(max(length, 1), max_len) for length in lengths]

    sample_ids = rng.sample(range(4 * num_samples), num_samples)
    return list(zip(sample_ids, lengths)), total_gpus, max_seq_len_per_rank, min_cp_size


@pytest.fixture
def scanned_samples(monkeypatch):
    """Count the samples next_hdp_group_packing_aware scans.

    The planner calls dcp_gpus_needed once per scanned sample, so a count below the
    number of input samples means that the early exit skipped the rest.
    """
    counter = [0]

    def counting_dcp_gpus_needed(*args):
        counter[0] += 1
        return dcp_gpus_needed(*args)

    monkeypatch.setattr(data_schedule_utils, "dcp_gpus_needed", counting_dcp_gpus_needed)
    return counter


def test_matches_reference_on_random_batches(scanned_samples):
    rng = random.Random(20261016)
    num_calls = 0
    num_early_exits = 0

    def planner(sample_seqlens, *args):
        nonlocal num_calls, num_early_exits
        scanned_before = scanned_samples[0]
        output = next_hdp_group_packing_aware(sample_seqlens, *args)
        num_calls += 1
        if scanned_samples[0] - scanned_before < len(sample_seqlens):
            num_early_exits += 1
        return output

    for _ in range(400):
        case = _random_case(rng)
        expected = _schedule(_reference_next_hdp_group_packing_aware, *case)
        assert _schedule(planner, *case) == expected, case

    # The early exit must fire on a good share of the calls, or the comparison above
    # says little about it.
    assert num_early_exits * 4 > num_calls, (num_early_exits, num_calls)


@pytest.mark.parametrize(
    ("sample_seqlens", "total_gpus", "max_seq_len_per_rank", "min_cp_size"),
    [
        pytest.param([], 4, 4096, 1, id="empty"),
        pytest.param([(0, 4096)], 4, 4096, 1, id="single_sample"),
        pytest.param([(i, 1000 + 10 * i) for i in range(10)], 1, 4096, 1, id="single_rank"),
        # Lists instead of tuples: the leftovers must still come back as tuples.
        pytest.param([[i, 3968] for i in range(16)], 4, 4096, 1, id="list_samples"),
        pytest.param([(i, 3968) for i in range(64)], 8, 4096, 1, id="fixed_4k"),
        pytest.param([(i, 3968) for i in range(256)], 64, 4096, 1, id="fixed_4k_64_ranks"),
        pytest.param([(i, 4096) for i in range(20)], 8, 4096, 1, id="fixed_full_rank"),
        pytest.param([(i, 2048) for i in range(40)], 8, 4096, 1, id="fixed_exact_half"),
        pytest.param([(i, 2049) for i in range(40)], 8, 4096, 1, id="fixed_just_over_half"),
        pytest.param([(i, 8192) for i in range(12)], 8, 4096, 1, id="fixed_cp2"),
        pytest.param([(i, 1) for i in range(50)], 4, 7, 1, id="fixed_tiny"),
        pytest.param(
            [(0, 4096), (1, 2048), (2, 2048), (3, 1024), (4, 1024), (5, 1024), (6, 1024), (7, 1)],
            2,
            4096,
            1,
            id="exact_fit_mixed",
        ),
        pytest.param([(i, 50) for i in range(30)], 14, 100, 1, id="non_power_of_two_ranks"),
        pytest.param([(i, 3000 + 7 * i) for i in range(30)], 6, 4000, 2, id="min_cp_2"),
        pytest.param([(0, 4096 * 8 + 1), (1, 10)], 8, 4096, 1, id="too_long_sample"),
        pytest.param([(0, 100)], 2, 4096, 4, id="min_cp_above_ranks"),
        pytest.param([(0, 3968), (1, 3968), (2, 0)], 2, 4096, 1, id="zero_length_sample"),
    ],
)
def test_matches_reference_on_edge_cases(
    sample_seqlens, total_gpus, max_seq_len_per_rank, min_cp_size
):
    args = (sample_seqlens, total_gpus, max_seq_len_per_rank, min_cp_size)
    assert _schedule(next_hdp_group_packing_aware, *args) == _schedule(
        _reference_next_hdp_group_packing_aware, *args
    )


@pytest.mark.parametrize(
    ("sample_seqlens", "total_gpus", "expected_scanned", "expected_leftovers"),
    [
        # Fixed 4K-style lengths: two samples never share a rank, so once the first 8
        # samples hold the 8 ranks, the other 56 are leftovers without being scanned.
        pytest.param(
            [(i, 3968) for i in range(64)],
            8,
            8,
            [(i, 3968) for i in range(8, 64)],
            id="exit_once_ranks_are_full",
        ),
        # The first sample spans the whole DPxCP group, so the check before the second
        # sample already exits.
        pytest.param(
            [(0, 8 * 4096)] + [(i, 4000) for i in range(1, 20)],
            8,
            1,
            [(i, 4000) for i in range(1, 20)],
            id="exit_when_first_sample_spans_all_ranks",
        ),
        pytest.param(
            [(i, 3968) for i in range(10)],
            1,
            1,
            [(i, 3968) for i in range(1, 10)],
            id="exit_on_single_rank",
        ),
        # Both ranks are taken after two samples, but the shortest sample still fits
        # rank 0 exactly (4000 + 96 == 4096), so every sample is scanned.
        pytest.param(
            [(0, 4000), (1, 4000), (2, 3000), (3, 96)],
            2,
            4,
            [(2, 3000)],
            id="no_exit_while_shortest_fits",
        ),
        # Samples 3 and 4 fill both ranks exactly, after sample 2 already became a
        # leftover; sample 5 is then skipped and keeps its place after sample 2.
        pytest.param(
            [(0, 4000), (1, 4000), (2, 3000), (3, 96), (4, 96), (5, 96)],
            2,
            5,
            [(2, 3000), (5, 96)],
            id="exit_after_leftovers",
        ),
    ],
)
def test_early_exit_is_taken_only_when_nothing_fits(
    scanned_samples, sample_seqlens, total_gpus, expected_scanned, expected_leftovers
):
    output = next_hdp_group_packing_aware(
        sample_seqlens, total_gpus=total_gpus, max_seq_len_per_rank=4096
    )

    assert scanned_samples[0] == expected_scanned
    assert output[1] == expected_leftovers
    assert output == _reference_next_hdp_group_packing_aware(
        sample_seqlens, total_gpus=total_gpus, max_seq_len_per_rank=4096
    )


def test_default_dynamic_cp_scheduler_matches_reference(scanned_samples):
    rng = random.Random(4096)
    sample_seqlens = [(i, 3968) for i in range(64)] + [
        (64 + i, min(max(int(rng.lognormvariate(7.0, 1.0)), 1), 4 * 4096)) for i in range(64)
    ]
    scheduler = DefaultDynamicCPScheduler(
        max_seqlen_per_dp_cp_rank=4096,
        cp_size=4,
        dp_size=2,
        microbatch_group_size_per_vp_stage=None,
    )

    sample_id_groups = scheduler.get_groups_and_subsamples(sample_seqlens)

    expected = _schedule(_reference_next_hdp_group_packing_aware, sample_seqlens, 8, 4096)
    assert sample_id_groups == [sample_ids for _, _, _, sample_ids in expected]
    # Without the early exit, every planner call scans its whole input: the batch first,
    # then the leftovers of each call.
    full_scan = len(sample_seqlens) + sum(len(leftovers) for _, leftovers, _, _ in expected)
    assert scanned_samples[0] < full_scan, (scanned_samples[0], full_scan)
