# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import random
from itertools import groupby
from math import ceil

import pytest

from megatron.core.datasets.data_schedule import DefaultDynamicCPScheduler
from megatron.core.datasets.data_schedule_utils import next_hdp_group_packing_aware


def _groups(layout):
    return [(list(samples), len(list(ranks))) for samples, ranks in groupby(layout)]


def _local_length(length, cp_size):
    """Rows of one rank once the sequence is padded to the zigzag chunks of its CP group."""
    alignment = 2 * cp_size if cp_size > 1 else 1
    return ceil(length / alignment) * alignment // cp_size


def _layout_workload(layout, lengths):
    workloads = []
    for samples, cp_size in _groups(layout):
        local_lengths = [_local_length(lengths[sid], cp_size) for sid in samples]
        workloads.extend([sum(length**2 * cp_size for length in local_lengths)] * cp_size)
    return workloads


@pytest.mark.parametrize('cp_size', [1, 2, 4, 8, 16])
def test_sequence_at_capacity_fills_its_cp_group(cp_size):
    length = cp_size * 128
    microbatches, leftovers, times, samples = next_hdp_group_packing_aware(
        [(0, length)], total_gpus=cp_size, max_seq_len_per_rank=128
    )
    assert not leftovers
    assert samples == [[0]] * cp_size
    assert microbatches == [[length]] * cp_size
    assert times == _layout_workload(samples, {0: length})


@pytest.mark.parametrize('pad_sequences', [False, True])
def test_scheduler_respects_padded_token_capacity(pad_sequences):
    # 2036 tokens fill four ranks of 509. Padded to the zigzag chunks of a CP4 group they
    # are 2040 tokens, 510 on each rank, so with padding the sequence needs eight ranks.
    lengths = {0: 2036, 1: 128}
    _, leftovers, _, layout = next_hdp_group_packing_aware(
        list(lengths.items()), total_gpus=8, max_seq_len_per_rank=509, pad_sequences=pad_sequences
    )

    assert not leftovers
    assert sorted(sid for samples, _ in _groups(layout) for sid in samples) == [0, 1]
    assert [cp_size for samples, cp_size in _groups(layout) if 0 in samples] == [
        8 if pad_sequences else 4
    ]
    for samples, cp_size in _groups(layout):
        if pad_sequences:
            local_lengths = [_local_length(lengths[sid], cp_size) for sid in samples]
        else:
            local_lengths = [lengths[sid] / cp_size for sid in samples]
        assert sum(local_lengths) <= 509


@pytest.mark.parametrize('parent', [4, 8, 16, 32])
def test_scheduler_forms_size_aligned_power_of_two_groups(parent):
    # The process groups of a CP size are the consecutive blocks of that many DPxCP ranks.
    rng = random.Random(20261026)
    scheduler = DefaultDynamicCPScheduler(256, parent, 1, None)
    for _ in range(20):
        samples = [(sid, rng.randint(1, parent * 8) * 32) for sid in range(16)]
        rows = scheduler.get_groups_and_subsamples(samples)
        for row in rows:
            start = 0
            for _, cp in _groups(row):
                assert cp & (cp - 1) == 0
                assert start % cp == 0
                start += cp
            assert start == parent


def test_vpp_alignment_keeps_all_samples_within_capacity():
    lengths = {0: 2304, 1: 2304, 2: 1408, 3: 1408}
    scheduler = DefaultDynamicCPScheduler(
        max_seqlen_per_dp_cp_rank=512, cp_size=16, dp_size=1, microbatch_group_size_per_vp_stage=2
    )
    layouts = scheduler.get_groups_and_subsamples(list(lengths.items()))
    assert len(layouts) % 2 == 0
    seen = []
    for layout in layouts:
        assert len(layout) == 16 and all(layout)
        for samples, cp_size in _groups(layout):
            seen.extend(samples)
            assert sum(_local_length(lengths[sid], cp_size) for sid in samples) <= 512
    assert sorted(seen) == sorted(lengths)


def test_vpp_split_fills_spare_ranks_by_expanding_cp_groups():
    scheduler = DefaultDynamicCPScheduler(
        max_seqlen_per_dp_cp_rank=1024, cp_size=16, dp_size=1, microbatch_group_size_per_vp_stage=2
    )
    result = scheduler.get_groups_and_subsamples([(sid, 4096) for sid in range(4)])
    # One microbatch of four CP4 groups is split in two; each half doubles its groups to CP8.
    assert len(result) == 2
    for layout in result:
        assert [(len(samples), cp_size) for samples, cp_size in _groups(layout)] == [(1, 8)] * 2
    assert sorted(
        sid for layout in result for samples, _ in _groups(layout) for sid in samples
    ) == list(range(4))
