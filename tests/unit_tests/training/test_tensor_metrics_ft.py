# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tensor-metric iterations use their own fault-tolerance timeout section."""

from unittest import mock

import pytest

from megatron.training import ft_integration
from megatron.training.tensor_metrics.observer import build_tensor_metric_observer


@pytest.fixture
def rank_monitor(monkeypatch):
    client = mock.Mock()
    monkeypatch.setattr(ft_integration, "_GLOBAL_RANK_MONITOR_CLIENT", client)
    monkeypatch.setattr(ft_integration, "_NUM_WARMUP_ITERS", 1)
    monkeypatch.setattr(ft_integration, "_seen_tr_iters_cnt", 0)
    monkeypatch.setattr(ft_integration, "_curr_eval_iter_idx", 0)
    monkeypatch.setattr(ft_integration, "_is_setup_section_open", True)
    monkeypatch.setattr(ft_integration, "_is_step_section_open", False)
    monkeypatch.setattr(ft_integration, "_step_section_name", "step")
    return client


def test_metric_schedule_selects_timeout_without_observing_tensors(rank_monitor):
    observer = build_tensor_metric_observer(["layer-param-l2:2", "layer-wgrad-l2:3"])
    assert not observer.has_due_metrics(None)

    for iteration in range(7):
        ft_integration.on_training_step_start(observer.has_due_metrics(iteration))
        ft_integration.on_training_step_end()

    # Iteration zero warms up; due-ness uses completed steps (iteration + 1).
    sections = ["tensor_metrics"] * 3 + ["step", "tensor_metrics", "step"]
    expected = [mock.call.end_section("setup")]
    for section in sections:
        expected.extend([mock.call.start_section(section), mock.call.end_section(section)])
    assert rank_monitor.mock_calls == expected


def test_metric_iteration_during_warmup_does_not_open_a_section(rank_monitor):
    ft_integration.on_training_step_start(is_tensor_metrics_iteration=True)
    ft_integration.on_training_step_end()

    assert rank_monitor.mock_calls == [mock.call.end_section("setup")]


def test_evaluation_uses_step_timeout_after_metric_iteration(rank_monitor):
    for is_metric_iteration in (False, True):
        ft_integration.on_training_step_start(is_metric_iteration)
        ft_integration.on_training_step_end()
    rank_monitor.reset_mock()

    for _ in range(2):
        ft_integration.on_eval_step_start()
        ft_integration.on_eval_step_end()

    assert rank_monitor.mock_calls == [
        mock.call.start_section("step"),
        mock.call.end_section("step"),
    ]


def test_metric_iteration_with_fault_tolerance_disabled(monkeypatch):
    monkeypatch.setattr(ft_integration, "_GLOBAL_RANK_MONITOR_CLIENT", None)
    ft_integration.on_training_step_start(is_tensor_metrics_iteration=True)
    ft_integration.on_training_step_end()
