# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU scheduling decisions and explicit CP limits; no attention kernels are run."""

from types import SimpleNamespace

import pytest

from megatron.core.datasets import data_schedule
from megatron.core.datasets.data_schedule import DefaultDynamicCPScheduler, wrap_data_iterator
from megatron.core.datasets.data_schedule_utils import next_hdp_group_packing_aware


def _scheduler(total_gpus, max_cp_size, vpp_multiple=None, min_cp_size=1):
    return DefaultDynamicCPScheduler(
        max_seqlen_per_dp_cp_rank=1,
        cp_size=1,
        dp_size=total_gpus,
        microbatch_group_size_per_vp_stage=vpp_multiple,
        min_cp_size=min_cp_size,
        max_cp_size=max_cp_size,
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
def test_explicit_cap_must_be_supported_by_actual_dpxcp(total_gpus, max_cp_size, min_cp_size):
    with pytest.raises(ValueError, match="max_cp_size"):
        _scheduler(total_gpus, max_cp_size, min_cp_size=min_cp_size)
    with pytest.raises(ValueError, match="max_cp_size"):
        next_hdp_group_packing_aware(
            [], total_gpus, 1, min_cp_size=min_cp_size, max_cp_size=max_cp_size
        )


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


@pytest.mark.parametrize("seq_len,max_cp_size,required_cp", [(5, 4, 8), (9, 8, 16)])
def test_sequence_that_cannot_fit_cap_fails_before_placement(seq_len, max_cp_size, required_cp):
    with pytest.raises(ValueError) as error:
        next_hdp_group_packing_aware(
            [(0, seq_len)], total_gpus=8, max_seq_len_per_rank=1, max_cp_size=max_cp_size
        )
    for detail in (
        f"seq_len={seq_len}",
        f"required_cp_size={required_cp}",
        f"max_cp_size={max_cp_size}",
        "DPxCP size=8",
        "max_seq_len_per_rank=1",
        "token limit",
    ):
        assert detail in str(error.value)


@pytest.mark.parametrize(
    "samples,total_gpus,token_limit,max_cp_size,effective_cp",
    [([(0, 1)], 8, 1, 4, 8), ([(0, 50), (1, 50)], 14, 100, 2, 14)],
)
def test_fill_or_full_group_fallback_cannot_escape_cap(
    samples, total_gpus, token_limit, max_cp_size, effective_cp
):
    with pytest.raises(ValueError, match="packing/fill") as error:
        next_hdp_group_packing_aware(samples, total_gpus, token_limit, max_cp_size=max_cp_size)
    message = str(error.value)
    assert f"effective_cp_size={effective_cp}" in message
    assert "known CP expansion issue" in message
    assert "Raise or remove the cap only if the model supports it" in message


def test_vpp_alignment_is_checked_after_it_expands_valid_packing():
    samples = [(0, 4), (1, 4)]
    assert _scheduler(8, 4).get_groups_and_subsamples(samples) == [[[0]] * 4 + [[1]] * 4]
    with pytest.raises(ValueError, match="final/VPP alignment") as error:
        _scheduler(8, 4, vpp_multiple=2).get_groups_and_subsamples(samples)
    assert "effective_cp_size=8" in str(error.value)


def test_no_cap_preserves_non_power_of_two_full_group_and_vpp_expansion():
    micro_batches, leftovers, _, sample_ids = next_hdp_group_packing_aware(
        [(0, 50), (1, 50)], total_gpus=14, max_seq_len_per_rank=100
    )
    assert leftovers == []
    assert micro_batches == [[50, 50] for _ in range(14)]
    assert sample_ids == [[0, 1] for _ in range(14)]
    assert _scheduler(8, None, vpp_multiple=2).get_groups_and_subsamples([(0, 4), (1, 4)]) == [
        [[1]] * 8,
        [[0]] * 8,
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

    monkeypatch.setitem(data_schedule.scheduler_map, "default_dynamic_cp", RecordingScheduler)
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
    groups = SimpleNamespace(
        dp_cp=SimpleNamespace(size=lambda: total_gpus),
        dp=SimpleNamespace(size=lambda: dp_size),
        tp=SimpleNamespace(size=lambda: 1),
        pp=SimpleNamespace(size=lambda: 1),
    )
    assert wrap_data_iterator(iter(()), config, 1, pg_collection=groups) == expected
    assert len(instances) == 1
    assert instances[0].max_cp_size == (cap if with_cap and dynamic_cp else None)
    assert instances[0].total_hdp_gpus == total_gpus
