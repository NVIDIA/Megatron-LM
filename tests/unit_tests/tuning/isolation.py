# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Resetting the process-wide autotune adapter between tests, shared by the CPU and GPU suites."""

import inspect
from weakref import WeakKeyDictionary, WeakSet

from megatron.core.tuning import cutile, interception, selection


def reset_tuning_state(monkeypatch) -> None:
    """Restore the prior patch, policy and diagnostics of ``megatron.core.tuning`` after the test."""
    try:
        from triton.runtime.autotuner import Autotuner
    except ImportError:
        Autotuner = None
    if Autotuner is not None:
        monkeypatch.setattr(Autotuner, "run", inspect.unwrap(Autotuner.run))
    for name, value in {
        "_installed": False,
        "_policy": None,
        "_tables": {},
        "_selected_configs": WeakKeyDictionary(),
        "_scopes": WeakKeyDictionary(),
        "_logged_singletons": WeakSet(),
        "_choice_log": {},
        "_unverified": {},
        "_verified": {},
        "_tune_records": {},
        "_record_rank": None,
        "_enumerated": set(),
    }.items():
        monkeypatch.setattr(interception, name, value)
    monkeypatch.setattr(selection, "_untuned_kernels_warned", set())
    monkeypatch.setattr(selection, "_mamba_env_warned", False)
    monkeypatch.setattr(cutile, "_previous", {name: None for name in cutile.HOOKS})
