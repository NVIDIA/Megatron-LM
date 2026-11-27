# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared isolation for tests of the process-wide Triton autotune adapter."""

import pytest

from megatron.core.tuning import selection
from tests.unit_tests.tuning.isolation import reset_tuning_state


@pytest.fixture
def isolated_policy(monkeypatch):
    """Keep the process-wide patch, policy inputs and diagnostics local to each test."""
    reset_tuning_state(monkeypatch)
    monkeypatch.setattr(selection, "arch_tag", lambda: "test_arch")
