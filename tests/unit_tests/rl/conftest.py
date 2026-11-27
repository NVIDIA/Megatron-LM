# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from megatron.training import global_vars


@pytest.fixture(autouse=True)
def reset_rl_run_config(monkeypatch):
    """The run's root container is a process global; never leak it."""
    monkeypatch.setattr(global_vars, "_GLOBAL_ROOT_CONFIG_CONTAINER", None)
