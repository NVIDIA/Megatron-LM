# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""``--strict-process-group-fallbacks`` makes global process-group fallbacks raise."""

import logging
import warnings
from argparse import ArgumentParser
from unittest.mock import Mock

import pytest

from megatron.core import process_groups_config
from megatron.core.process_groups_config import (
    ProcessGroupFallbackWarning,
    warn_global_process_group_fallback,
)
from megatron.rank_log_setup import suppress_duplicate_logs_off_rank0
from megatron.training import arguments, global_vars, initialize


def _parse_args(strict):
    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    return parser.parse_args(["--strict-process-group-fallbacks"] if strict else [])


def _fallbacks(record):
    return [w for w in record if issubclass(w.category, ProcessGroupFallbackWarning)]


@pytest.fixture(autouse=True)
def isolated_fallback_state(monkeypatch):
    """Start with no recorded fallbacks, and undo what the rank-0 log suppression changes."""
    monkeypatch.setattr(process_groups_config, "_warned_global_process_group_fallbacks", set())
    pytree_logger = logging.getLogger("torch.utils._pytree")
    original_level = pytree_logger.level
    yield
    pytree_logger.setLevel(original_level)


@pytest.fixture
def initialize_without_services(monkeypatch, run_config):
    """Run initialize_megatron for real, but without the services it starts."""
    monkeypatch.setattr(run_config, "finalize", Mock())
    for name in (
        "initialize_runtime_services",
        "print_rank_0",
        "set_default_log_ranks",
        "_set_random_seed",
    ):
        monkeypatch.setattr(initialize, name, Mock())
    monkeypatch.setattr(initialize.tuning, "install", Mock())


@pytest.mark.parametrize("strict", [False, True])
def test_flag_is_parsed_and_off_by_default(strict):
    assert _parse_args(strict).strict_process_group_fallbacks is strict


@pytest.mark.parametrize("rank", ["0", "1"], ids=["rank0", "rank1"])
@pytest.mark.parametrize("strict", [False, True], ids=["default", "strict"])
def test_initialize_megatron_applies_the_flag_after_logging_setup(
    monkeypatch, initialize_without_services, rank, strict
):
    """The entry point quiets duplicate logs off rank 0 on import; strict mode raises anyway."""
    args = _parse_args(strict)
    args.rank = int(rank)
    monkeypatch.setenv("RANK", rank)
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", args)
    events = []
    monkeypatch.setattr(initialize, "setup_logging", lambda: events.append("setup_logging"))
    enable_strict = initialize.enable_strict_process_group_fallbacks

    def record_enable_strict():
        events.append("strict")
        enable_strict()

    monkeypatch.setattr(initialize, "enable_strict_process_group_fallbacks", record_enable_strict)

    with warnings.catch_warnings(record=True) as record:
        # Whatever filters the test run installs, only the flag may make the fallback raise.
        warnings.simplefilter("default", ProcessGroupFallbackWarning)
        suppress_duplicate_logs_off_rank0()
        initialize.initialize_megatron(allow_no_cuda=True, skip_mpu_initialization=True)
        if strict:
            with pytest.raises(ProcessGroupFallbackWarning, match="Owner was called without"):
                warn_global_process_group_fallback("Owner")
        else:
            warn_global_process_group_fallback("Owner")

    expected_events = ["setup_logging", "strict"] if strict else ["setup_logging"]
    assert events == expected_events
    # Without the flag, the fallback only warns.
    assert len(_fallbacks(record)) == (0 if strict else 1)


@pytest.mark.parametrize("strict", [False, True], ids=["default", "strict"])
def test_fallback_during_distributed_initialization_raises_under_the_flag(
    monkeypatch, initialize_without_services, strict
):
    args = _parse_args(strict)
    args.rank = 0
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", args)
    monkeypatch.setattr(initialize, "setup_logging", Mock())

    def initialize_distributed(*unused_args, **unused_kwargs):
        # A component that omits its process groups while model parallelism is set up.
        warn_global_process_group_fallback("Owner")

    monkeypatch.setattr(initialize, "_initialize_distributed", initialize_distributed)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("default", ProcessGroupFallbackWarning)
        if strict:
            with pytest.raises(ProcessGroupFallbackWarning, match="Owner was called without"):
                initialize.initialize_megatron(allow_no_cuda=True)
        else:
            initialize.initialize_megatron(allow_no_cuda=True)

    assert len(_fallbacks(record)) == (0 if strict else 1)
    # Only without the flag does initialization go on past the fallback.
    assert initialize._set_random_seed.call_count == (0 if strict else 1)
