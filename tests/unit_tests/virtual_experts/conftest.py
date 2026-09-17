# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def ensure_test_data():
    """These synthetic regressions need no downloaded unit-test datasets."""


@pytest.fixture(scope="module", autouse=True)
def supported_world():
    """Support four-GPU Blackwell and eight-GPU H100 nodes."""
    if int(os.environ.get("WORLD_SIZE", "1")) not in (4, 8):
        pytest.skip("virtual-expert regressions require a four- or eight-rank torchrun launch")
