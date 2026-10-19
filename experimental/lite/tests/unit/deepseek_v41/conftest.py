# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import sys

import pytest


@pytest.fixture(autouse=True)
def reject_rollout_modules():
    assert not any(n == "vllm" or n.startswith("vllm.") for n in sys.modules)
    yield
    assert not any(n == "vllm" or n.startswith("vllm.") for n in sys.modules)
