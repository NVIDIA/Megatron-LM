# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from megatron.rl.config import unset_rl_run_config


@pytest.fixture(autouse=True)
def reset_rl_run_config():
    """The RL container is a process global like the run config; never leak it across tests."""
    yield
    unset_rl_run_config()
