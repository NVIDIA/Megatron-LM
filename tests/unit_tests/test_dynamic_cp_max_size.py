# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU scheduling decisions and explicit CP limits; no attention kernels are run."""

import sys
from types import SimpleNamespace

import pytest

from megatron.core.context_parallel.groups import get_valid_dynamic_context_parallel_group_sizes
from megatron.core.datasets import data_schedule
from megatron.core.datasets.data_schedule import (
    DefaultDynamicCPScheduler,
    PackingSchedulerEnum,
    wrap_data_iterator,
)
from megatron.core.datasets.data_schedule_utils import next_hdp_group_packing_aware


def _scheduler(total_gpus, max_cp_size, vpp_multiple=None, min_cp_size=1, **kwargs):
    return DefaultDynamicCPScheduler(
        max_seqlen_per_dp_cp_rank=kwargs.pop("max_seqlen_per_dp_cp_rank", 1),
        cp_size=1,
        dp_size=total_gpus,
        microbatch_group_size_per_vp_stage=vpp_multiple,
        min_cp_size=min_cp_size,
        max_cp_size=max_cp_size,
        **kwargs,
    )


def _groups(total_gpus, dp_size, dynamic):
    return SimpleNamespace(
        dp_cp_gtp_remat=None,
        dp_gtp_remat=None,
        dp_cp=SimpleNamespace(size=lambda: total_gpus),
        dp=SimpleNamespace(size=lambda: dp_size),
        tp=SimpleNamespace(size=lambda: 1),
        pp=SimpleNamespace(size=lambda: 1),
        dynamic_cp=(
            dict.fromkeys(get_valid_dynamic_context_parallel_group_sizes(total_gpus), object())
            if dynamic
            else None
        ),
    )


@pytest.mark.parametrize(
    "total_gpus,max_cp_size,min_cp_size",
    [
        (8, 0, 1),
        (8, 3, 1),
        (8, 16, 1),
        (12, 8, 1),
        (8, 2, 4),
        (8, True, 1),
        (8, 2.0, 1),
        (12, 4, 0),
        (12, 4, 3),
    ],
)
def test_explicit_cap_must_be_supported_by_actual_dpxcp(
    monkeypatch, total_gpus, max_cp_size, min_cp_size
):
    with pytest.raises(ValueError, match="max_cp_size") as init_error:
        _scheduler(total_gpus, max_cp_size, min_cp_size=min_cp_size)

    # The training entry point must reject the same configuration before scheduling.
    monkeypatch.setattr(data_schedule.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        DefaultDynamicCPScheduler, "run", lambda *args: pytest.fail("Scheduling must not start")
    )
    config = SimpleNamespace(
        dynamic_context_parallel=True,
        sequence_packing_scheduler="default_dynamic_cp",
        min_dynamic_context_parallel_size=min_cp_size,
        max_cp_size=max_cp_size,
        max_seqlen_per_dp_cp_rank=1,
        virtual_pipeline_model_parallel_size=None,
    )
    groups = _groups(total_gpus, 2, True)
    with pytest.raises(ValueError, match="max_cp_size") as wrapper_error:
        wrap_data_iterator(iter(()), config, 1, pg_collection=groups)
    assert str(wrapper_error.value) == str(init_error.value)


@pytest.mark.parametrize("total_gpus,max_cp_size", [(8, 1), (8, 2), (8, 4), (12, 4)])
def test_capped_packing_keeps_all_samples_and_fills_all_ranks(total_gpus, max_cp_size):
    samples = [(i, max_cp_size) for i in range(total_gpus // max_cp_size)]
    micro_batches, leftovers, _, sample_ids = next_hdp_group_packing_aware(
        samples, total_gpus, 1, max_cp_size=max_cp_size
    )
    expected_ids = [[rank // max_cp_size] for rank in range(total_gpus)]
    assert leftovers == []
    assert sample_ids == expected_ids
    assert micro_batches == [[max_cp_size] for _ in range(total_gpus)]
    assert _scheduler(total_gpus, max_cp_size).get_groups_and_subsamples(samples) == [expected_ids]


def test_sequence_that_cannot_fit_explicit_cap_fails_before_placement():
    with pytest.raises(ValueError) as error:
        next_hdp_group_packing_aware([(0, 5)], total_gpus=8, max_seq_len_per_rank=1, max_cp_size=4)
    for detail in (
        "seq_len=5",
        "required_cp_size=8",
        "effective_cp_limit=4",
        "max_cp_size=4",
        "DPxCP size=8",
        "max_seq_len_per_rank=1",
    ):
        assert detail in str(error.value)


@pytest.mark.parametrize("max_cp_size", [None, 8])
def test_sequence_beyond_available_groups_preserves_dev_capacity_error(max_cp_size):
    with pytest.raises(ValueError) as error:
        next_hdp_group_packing_aware([(0, 9)], 8, 1, max_cp_size=max_cp_size)
    assert str(error.value) == (
        "Sequence length 9, including CP padding, exceeds the capacity of "
        "configured CP groups (1, 2, 4, 8) at 1 tokens per rank"
    )


@pytest.mark.parametrize("max_cp_size", [8, None])
def test_sequence_at_effective_limit_succeeds(max_cp_size):
    micro_batches, leftovers, _, sample_ids = next_hdp_group_packing_aware(
        [(0, 8)], total_gpus=8, max_seq_len_per_rank=1, max_cp_size=max_cp_size
    )
    assert leftovers == []
    assert micro_batches == [[8] for _ in range(8)]
    assert sample_ids == [[0] for _ in range(8)]


def test_no_cap_preserves_non_power_of_two_full_group_and_vpp_expansion():
    micro_batches, leftovers, _, sample_ids = next_hdp_group_packing_aware(
        [(0, 50), (1, 50)], total_gpus=14, max_seq_len_per_rank=100
    )
    assert leftovers == []
    assert micro_batches == [[50, 50] for _ in range(14)]
    assert sample_ids == [[0, 1] for _ in range(14)]
    assert _scheduler(8, None, vpp_multiple=2).get_groups_and_subsamples([(0, 4), (1, 4)]) == [
        [[0]] * 8,
        [[1]] * 8,
    ]


@pytest.mark.parametrize("with_cap", [False, True])
@pytest.mark.parametrize("dynamic_cp,cap,total_gpus,dp_size", [(False, 3, 12, 4), (True, 4, 8, 2)])
def test_wrapper_passes_cap_only_with_dcp(
    monkeypatch, with_cap, dynamic_cp, cap, total_gpus, dp_size
):
    instances = []
    expected = ("iterator", 2, 8, 32)

    class RecordingScheduler(DefaultDynamicCPScheduler):
        def run(self, *args, **kwargs):
            instances.append(self)
            return expected

    monkeypatch.setitem(
        data_schedule.scheduler_map, PackingSchedulerEnum.DEFAULT_DYNAMIC_CP, RecordingScheduler
    )
    monkeypatch.setattr(data_schedule.torch.cuda, "current_device", lambda: 0)
    config = SimpleNamespace(
        dynamic_context_parallel=dynamic_cp,
        context_parallel_size=total_gpus // dp_size,
        sequence_packing_scheduler="default_dynamic_cp",
        min_dynamic_context_parallel_size=1,
        max_seqlen_per_dp_cp_rank=1,
        virtual_pipeline_model_parallel_size=None,
    )
    if with_cap:
        config.max_cp_size = cap
    groups = _groups(total_gpus, dp_size, dynamic_cp)
    assert wrap_data_iterator(iter(()), config, 1, pg_collection=groups) == expected
    assert len(instances) == 1
    assert instances[0].max_cp_size == (cap if with_cap and dynamic_cp else None)
    assert instances[0].total_dpxcp_ranks == total_gpus


def test_cap_validation_runs_only_once_at_initialization(monkeypatch):
    calls = []
    original = data_schedule._validate_dynamic_cp_max_size

    def record(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setitem(
        DefaultDynamicCPScheduler.__init__.__globals__, "_validate_dynamic_cp_max_size", record
    )
    scheduler = _scheduler(8, 4)
    for _ in range(2):
        scheduler.get_groups_and_subsamples([(0, 4), (1, 4)])
    assert calls == [(4, 1, 8)]


@pytest.mark.parametrize("total_gpus,sizes,cap", [(6, (1, 2, 6), 2), (12, (1, 4, 12), 4)])
def test_sparse_group_family_is_preserved_even_when_fill_exceeds_cap(total_gpus, sizes, cap):
    scheduler = _scheduler(total_gpus, cap, cp_group_sizes=sizes, max_seqlen_per_dp_cp_rank=100)
    schedule = scheduler.get_groups_and_subsamples([(0, 50), (1, 50)])
    assert scheduler.cp_group_sizes == sizes
    assert schedule == [[[0, 1] for _ in range(total_gpus)]]
    with pytest.raises(
        ValueError, match=f"required_cp_size={total_gpus} exceeds effective_cp_limit={cap}"
    ):
        scheduler.get_groups_and_subsamples([(0, cap * 100 + 1)])


def test_padding_increases_minimum_cp_before_applying_cap():
    with pytest.raises(ValueError, match="required_cp_size=4 exceeds effective_cp_limit=2"):
        next_hdp_group_packing_aware([(0, 5)], 4, 3, pad_sequences=True, max_cp_size=2)
    # With padding disabled, CP2 fits the same length under the same token limit.
    assert next_hdp_group_packing_aware([(0, 5)], 4, 3, max_cp_size=2)[1] == []
    micro_batches, leftovers, _, ids = next_hdp_group_packing_aware(
        [(0, 5)], 4, 3, pad_sequences=True, max_cp_size=4
    )
    assert leftovers == []
    assert micro_batches == [[5]] * 4
    assert ids == [[0]] * 4


def test_candidate_workload_evaluation_stops_at_cap_but_fill_remains_unbounded():
    def evaluated_sizes(cap):
        sizes = []

        def record(frame, event, arg):
            if (
                event == "call"
                and frame.f_code.co_name == "workload"
                and frame.f_back.f_code is next_hdp_group_packing_aware.__code__
            ):
                sizes.append(frame.f_locals["cp_size"])

        previous = sys.getprofile()
        try:
            sys.setprofile(record)
            result = next_hdp_group_packing_aware([(0, 4), (1, 1)], 8, 8, max_cp_size=cap)
        finally:
            sys.setprofile(previous)
        assert {sid for rank_ids in result[3] for sid in rank_ids} == {0, 1}
        return set(sizes)

    assert evaluated_sizes(None) == {1, 2, 4, 8}
    assert evaluated_sizes(1) == {1}


def test_vpp_keeps_full_group_family_and_can_expand_beyond_cap():
    scheduler = _scheduler(8, 4, vpp_multiple=2)
    schedule = scheduler.get_groups_and_subsamples([(0, 4), (1, 4)])
    assert scheduler.cp_group_sizes == (1, 2, 4, 8)
    assert schedule == [[[0]] * 8, [[1]] * 8]
