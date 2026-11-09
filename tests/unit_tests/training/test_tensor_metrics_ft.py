# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tensor-metric iterations use their own fault-tolerance timeout section."""

import copy
import json
import sys
from argparse import ArgumentParser, Namespace
from types import SimpleNamespace
from unittest import mock

import pytest

from megatron.training import arguments, ft_integration, global_vars, training
from megatron.training.argument_utils import fault_tolerance_config_from_args
from megatron.training.config import (
    FaultInjectorConfig,
    FaultToleranceConfig,
    PretrainConfigContainer,
)
from megatron.training.tensor_metrics.observer import build_tensor_metric_observer


@pytest.mark.parametrize("enabled", [False, True])
def test_fault_tolerance_cli_matches_native_config(enabled):
    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    argv = (
        [
            "--adlr-autoresume",
            "--adlr-autoresume-interval",
            "7",
            "--enable-ft-package",
            "--calc-ft-timeouts",
            "--ft-num-warmup-iters",
            "3",
            "--fault-injector-ranks",
            "0",
            "--fault-injector-fault-types",
            "hang",
            "--fault-injector-fault-delay",
            "10",
            "--fault-injector-delay-start-iteration",
            "12",
        ]
        if enabled
        else []
    )
    cfg = fault_tolerance_config_from_args(parser.parse_args(argv))
    expected = (
        FaultToleranceConfig(
            adlr_autoresume=True,
            adlr_autoresume_interval=7,
            enable_ft_package=True,
            calc_ft_timeouts=True,
            ft_num_warmup_iters=3,
            fault_injector=FaultInjectorConfig(
                fault_injector_ranks="0",
                fault_injector_fault_types="hang",
                fault_injector_fault_delay=10,
                fault_injector_delay_start_iteration=12,
            ),
        )
        if enabled
        else FaultToleranceConfig()
    )
    assert cfg == expected


def test_fault_tolerance_serialization_and_read_only_validation(run_config, tmp_path):
    run_config.fault_tolerance = FaultToleranceConfig(
        enable_ft_package=True,
        calc_ft_timeouts=True,
        fault_injector=FaultInjectorConfig(fault_injector_ranks="0", fault_injector_seed=42),
    )
    before = copy.deepcopy(run_config.to_dict())
    for _ in range(2):
        with pytest.raises(ValueError, match="checkpointing save dir must be set"):
            run_config.validate()
        assert run_config.to_dict() == before
    run_config.checkpoint.save = str(tmp_path)
    before = copy.deepcopy(run_config.to_dict())
    run_config.validate()
    run_config.validate()
    assert run_config.to_dict() == before
    path = str(tmp_path / "run_config.yaml")
    run_config.to_yaml(path)
    assert PretrainConfigContainer.from_yaml(path).fault_tolerance == run_config.fault_tolerance
    # Old saved containers have no section; loading them leaves monitoring/injection disabled.
    before.pop("fault_tolerance")
    assert PretrainConfigContainer.from_dict(before).fault_tolerance == FaultToleranceConfig()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("calc_timeouts", [False, True])
@pytest.mark.parametrize("async_save", [False, True])
def test_fault_tolerance_setup_uses_config(
    monkeypatch, run_config, rank_monitor, tmp_path, enabled, calc_timeouts, async_save
):
    monkeypatch.setattr(ft_integration, "_GLOBAL_RANK_MONITOR_CLIENT", None)
    monkeypatch.setattr(ft_integration, "_ft_state_path", None)
    monkeypatch.setattr(ft_integration, "_is_async_chkpt_enabled", False)
    monkeypatch.setattr(ft_integration, "_is_calculating_timeouts", False)
    monkeypatch.setattr(arguments, "parse_args", mock.Mock(side_effect=AssertionError("No CLI")))
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", None)
    client_factory = mock.Mock(return_value=rank_monitor)
    monkeypatch.setitem(
        sys.modules,
        "nvidia_resiliency_ext.fault_tolerance",
        SimpleNamespace(RankMonitorClient=client_factory),
    )
    run_config.fault_tolerance = FaultToleranceConfig(
        enable_ft_package=enabled, calc_ft_timeouts=calc_timeouts, ft_num_warmup_iters=2
    )
    run_config.checkpoint.save = str(tmp_path)
    run_config.checkpoint.async_save = async_save
    run_config.checkpoint.use_persistent_ckpt_worker = async_save
    run_config.checkpoint.validate()
    saved_timeouts = {"section_timeouts": {"setup": 123, "step": 45}}
    (tmp_path / "ft_state.json").write_text(json.dumps(saved_timeouts))

    monkeypatch.setattr(global_vars, "_GLOBAL_RUN_CONFIG", None)
    ft_integration.setup(
        run_config.fault_tolerance,
        checkpoint_dir=run_config.checkpoint.save,
        async_save=run_config.checkpoint.async_save,
    )

    if not enabled:
        client_factory.assert_not_called()
        assert ft_integration.get_rank_monitor_client() is None
        return
    rank_monitor.init_workload_monitoring.assert_called_once_with(num_warmup_iters=2)
    rank_monitor.load_state_dict.assert_called_once_with(saved_timeouts)
    rank_monitor.start_section.assert_called_once_with("setup")
    assert ft_integration._is_calculating_timeouts is calc_timeouts
    assert ft_integration._is_async_chkpt_enabled is async_save
    rank_monitor.reset_mock()
    for _ in range(3):
        ft_integration.on_training_step_start()
        ft_integration.on_training_step_end()
    assert rank_monitor.mock_calls == [
        mock.call.end_section("setup"),
        mock.call.start_section("step"),
        mock.call.end_section("step"),
    ]


@pytest.mark.parametrize("enabled", [False, True])
def test_autoresume_service_and_cadence_use_config(monkeypatch, run_config, enabled):
    autoresume = mock.Mock()
    monkeypatch.setitem(sys.modules, "userlib.auto_resume", SimpleNamespace(AutoResume=autoresume))
    monkeypatch.setattr(global_vars, "_GLOBAL_ADLR_AUTORESUME", None)
    # Autoresume setup must not need a namespace at all.
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", None)
    monkeypatch.setattr(sys, "path", list(sys.path))
    run_config.fault_tolerance = FaultToleranceConfig(
        adlr_autoresume=enabled, adlr_autoresume_interval=2
    )
    global_vars._set_adlr_autoresume()
    assert global_vars.get_adlr_autoresume() is (autoresume if enabled else None)
    monkeypatch.setattr(training, "get_args", lambda: Namespace(log_straggler=False))
    check = mock.Mock()
    monkeypatch.setattr(training, "check_adlr_autoresume_termination", check)
    for iteration in (1, 2, 3):
        training.post_training_step_callbacks([], None, None, iteration, None, 0)
    assert check.call_args_list == ([mock.call(2, [], None, None)] if enabled else [])


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
