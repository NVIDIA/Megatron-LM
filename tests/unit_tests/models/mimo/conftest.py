# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest


@pytest.fixture(autouse=True)
def mimo_run_config(run_config):
    """Match the seed used by the shared MIMO model helper."""
    run_config.rng.seed = 123
    return run_config
