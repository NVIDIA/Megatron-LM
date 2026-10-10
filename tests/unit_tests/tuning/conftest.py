# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared isolation for tests of the process-wide Triton autotune adapter."""

import inspect
from weakref import WeakKeyDictionary, WeakSet

import pytest

from megatron.core.tuning import interception, selection


@pytest.fixture
def isolated_policy(monkeypatch):
    """Keep the process-wide patch, policy inputs and diagnostics local to each test."""
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
    monkeypatch.setattr(selection, "arch_tag", lambda: "test_arch")
    monkeypatch.setattr(interception.atexit, "register", lambda *_: None)
