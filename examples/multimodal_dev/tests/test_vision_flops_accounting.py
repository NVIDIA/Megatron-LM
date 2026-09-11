# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the ``forward_step`` side of vision FLOPs accounting.

The FLOPs formula and the accumulator math live in
``tests/unit_tests/test_num_floating_point_operations.py``. What is pinned
here is the gate that decides WHETHER a microbatch is reported: only the
accepted training execution may contribute, so a rerun-state-machine replay
of the same microbatches must not double-count, and ``evaluate()`` must not
contribute at all.

CPU-only: no model, no process group.
"""

from types import SimpleNamespace

import pytest
import torch

import megatron.training.training as training_module
from examples.multimodal_dev.forward_step import record_vision_model_flops
from megatron.core.rerun_state_machine import RerunState
from megatron.training.training import consume_vision_model_flops_stats

# One image: T=1, H=4, W=4 with spatial_merge_size=2.
_GRID = torch.tensor([[1, 4, 4]], dtype=torch.int64)
_ONE_PASS = (16.0, 256.0, 4.0)


@pytest.fixture
def args():
    return SimpleNamespace(count_vision_model_flops=True, vision_spatial_merge_size=2)


@pytest.fixture(autouse=True)
def reset_accumulator():
    training_module._vision_flops_stats_in_iteration = None
    yield
    training_module._vision_flops_stats_in_iteration = None


@pytest.fixture
def rerun_state(monkeypatch):
    """Drive ``get_rerun_state_machine().state`` without a real state machine."""
    stub = SimpleNamespace(state=RerunState.INITIAL_RUN)
    monkeypatch.setattr(
        "examples.multimodal_dev.forward_step.get_rerun_state_machine", lambda: stub
    )
    return stub


def test_initial_run_records_the_microbatch(args, rerun_state):
    record_vision_model_flops(args, SimpleNamespace(training=True), {"image_grid_thw": _GRID})
    assert consume_vision_model_flops_stats(True) == _ONE_PASS


def test_in_place_rerun_does_not_double_count(args, rerun_state):
    """Regression: one forced rerun must consume as a single logical pass.

    ``--check-for-nan-in-loss`` / ``--check-for-spiky-loss`` re-execute
    ``train_step`` on the SAME microbatches with the model still in train
    mode, while ``consume_*`` still runs once per training iteration.
    """
    model = SimpleNamespace(training=True)
    batch = {"image_grid_thw": _GRID}

    # Initial run.
    rerun_state.state = RerunState.INITIAL_RUN
    record_vision_model_flops(args, model, batch)
    # The state machine requests a rerun; the same microbatch is replayed.
    rerun_state.state = RerunState.RERUNNING_IN_PLACE
    record_vision_model_flops(args, model, batch)

    # train() drains once for the whole iteration.
    assert consume_vision_model_flops_stats(True) == _ONE_PASS


def test_evaluation_does_not_contribute(args, rerun_state):
    record_vision_model_flops(args, SimpleNamespace(training=False), {"image_grid_thw": _GRID})
    # Nothing reported at all -> the caller omits the vision terms.
    assert consume_vision_model_flops_stats(True) == (None, None, None)


def test_exhausted_iterator_reports_an_explicit_zero(args, rerun_state):
    """``batch is None`` must still report, so the rank is not silently dropped."""
    record_vision_model_flops(args, SimpleNamespace(training=True), None)
    assert consume_vision_model_flops_stats(True) == (0.0, 0.0, 0.0)


def test_text_only_microbatch_reports_an_explicit_zero(args, rerun_state):
    record_vision_model_flops(args, SimpleNamespace(training=True), {})
    assert consume_vision_model_flops_stats(True) == (0.0, 0.0, 0.0)


def test_disabled_feature_never_touches_the_accumulator(rerun_state):
    args = SimpleNamespace(count_vision_model_flops=False)
    record_vision_model_flops(args, SimpleNamespace(training=True), {"image_grid_thw": _GRID})
    assert training_module._vision_flops_stats_in_iteration is None


def test_missing_spatial_merge_size_fails_with_an_actionable_error(rerun_state):
    """A registry that enables the flag without the merge size must not
    ``AttributeError`` deep in the forward step."""
    args = SimpleNamespace(count_vision_model_flops=True)
    with pytest.raises(ValueError, match="spatial_merge_size must be positive"):
        record_vision_model_flops(args, SimpleNamespace(training=True), {"image_grid_thw": _GRID})
