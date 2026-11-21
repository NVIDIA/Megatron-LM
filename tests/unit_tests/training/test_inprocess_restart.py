# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests that the in-process restart arguments reach nvidia-resiliency-ext."""

from argparse import ArgumentParser
from datetime import timedelta
from unittest.mock import MagicMock

from megatron.training import arguments, inprocess_restart


def test_inprocess_restart_forwards_max_iterations_and_watchdog_interval(monkeypatch):
    inprocess = MagicMock()
    monkeypatch.setattr(inprocess_restart, "_get_inprocess_module", lambda: inprocess)
    monkeypatch.setenv("MASTER_PORT", "29500")
    args = arguments._add_inprocess_restart_args(ArgumentParser()).parse_args(
        [
            "--inprocess-granularity",
            "rank",
            "--inprocess-max-iterations",
            "3",
            "--inprocess-progress-watchdog-interval",
            "5",
        ]
    )

    inprocess_restart.inprocess_restart(MagicMock(), args)

    retry_controller_kwargs = inprocess.initialize.RetryController.call_args.kwargs
    assert retry_controller_kwargs["max_iterations"] == 3
    wrapper_kwargs = inprocess.Wrapper.call_args.kwargs
    assert wrapper_kwargs["progress_watchdog_interval"] == timedelta(seconds=5)
