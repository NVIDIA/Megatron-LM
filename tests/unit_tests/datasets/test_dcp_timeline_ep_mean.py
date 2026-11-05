# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Rank-weighted EP means for the explicitly forced-balanced timing recipe."""

import pytest

from megatron.core.datasets.dcp_runtime_scheduler.runtime.timeline import TimelineModel, _freeze_op


def test_named_ep_mean_is_explicit_and_legacy_operations_freeze_identically():
    assert _freeze_op({"sync": "ep", "cost": "rest", "aggregation": "mean"}) == ("ep_mean", "rest")
    assert _freeze_op({"sync": "ep", "duration_ms": 0.75}) == ("ep", 0.75)
    assert _freeze_op({"sync": "cp", "cost": "attention"}) == ("cp", "attention")
    assert _freeze_op({"cost": "attention"}) == ("cp", "attention")
    assert _freeze_op({"sync": "local", "duration_ms": 1.25}) == ("local", 1.25)
    with pytest.raises(ValueError, match="explicit mean"):
        _freeze_op({"sync": "ep", "cost": "rest"})


def test_ep_mean_preserves_valid_tokens_across_cp1_cp2_cp4_layout():
    # Packs contain 512, 1024, 4096 and 16384 valid tokens. Their CP widths
    # are 1, 1, 2 and 4. A rank contributes 2 + 4 * valid_tokens_per_rank/4096.
    groups = [
        (0, 1, 0.0, 0.0, {"rest": 2.5}),
        (1, 1, 0.0, 0.0, {"rest": 3.0}),
        (2, 2, 0.0, 0.0, {"rest": 4.0}),
        (4, 4, 0.0, 0.0, {"rest": 6.0}),
    ]
    timeline = TimelineModel(8, 8, 1, program=((("ep_mean", "rest"),), ()))
    state = [0.0, 1.0, 5.0, 2.0, 4.0, 0.0, 3.0, 0.0]
    # Average valid tokens/rank is 22016/8=2752, so cost is 4.6875 ms.
    # The EP operation waits for rank 2 at 5 ms before starting.
    expected = [9.6875] * 8
    assert timeline._program_phase(state, groups, timeline.program[0]) == expected
    assert timeline._program_phase(state, list(reversed(groups)), timeline.program[0]) == expected


def test_ep_blocks_have_independent_start_times_and_rank_weighted_costs():
    groups = [
        (0, 2, 0.0, 0.0, {"rest": 2.0}),
        (2, 2, 0.0, 0.0, {"rest": 4.0}),
        (4, 4, 0.0, 0.0, {"rest": 8.0}),
        (8, 8, 0.0, 0.0, {"rest": 12.0}),
    ]
    timeline = TimelineModel(16, 8, 1, program=((("ep_mean", "rest"),), ()))
    state = [7.0] + [0.0] * 7 + [3.0] + [0.0] * 7
    assert timeline._program_phase(state, groups, timeline.program[0]) == [12.5] * 8 + [15.0] * 8


def test_forward_and_backward_mean_costs_follow_the_mixed_cp_critical_path():
    groups = []
    for start, cp, forward, backward, rest in (
        (0, 1, 1.0, 2.0, 2.5),
        (1, 1, 2.0, 4.0, 3.0),
        (2, 2, 3.0, 6.0, 4.0),
        (4, 4, 4.0, 8.0, 6.0),
    ):
        groups.append(
            (
                start,
                cp,
                forward,
                backward,
                {"att_f": forward, "att_b": backward, "rest_f": rest, "rest_b": rest / 2},
            )
        )
    timeline = TimelineModel(
        8,
        8,
        1,
        iteration_tail_ms=0.25,
        program=(
            (("cp", "att_f"), ("ep_mean", "rest_f")),
            (("ep_mean", "rest_b"), ("cp", "att_b")),
        ),
    )
    # First forward: max attention 4 + EP mean 4.6875 = 8.6875.
    # Backward: EP mean 2.34375, then max attention 8 => 19.03125.
    assert timeline.evaluate([groups]) == 19.28125
    # Second forward retains unequal completion times from the first backward.
    assert timeline.evaluate([groups, groups]) == 38.3125


def test_legacy_fixed_ep_and_named_cp_timeline_keeps_frozen_step_times():
    groups = [
        (0, 1, 1.0, 2.0, {"att_f": 1.0, "att_b": 2.0}),
        (1, 1, 2.0, 4.0, {"att_f": 2.0, "att_b": 4.0}),
        (2, 2, 3.0, 6.0, {"att_f": 3.0, "att_b": 6.0}),
        (4, 4, 4.0, 8.0, {"att_f": 4.0, "att_b": 8.0}),
    ]
    timeline = TimelineModel(
        8,
        8,
        2,
        iteration_tail_ms=2.0,
        program=((("cp", "att_f"), ("ep", 0.5)) * 2, (("cp", "att_b"), ("ep", 0.75)) * 2),
    )
    assert timeline.evaluate([groups]) == 28.5
    assert timeline.evaluate([groups, groups]) == 55.0
