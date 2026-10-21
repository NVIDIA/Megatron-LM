# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import sys

import pytest


def _rollout_loaded():
    return any(n == "vllm" or n.startswith("vllm.") for n in sys.modules)


@pytest.fixture(autouse=True)
def reject_rollout_modules(request):
    """Native tests never import vLLM; deployment-operator tests (``test_vllm_*``) may."""
    if request.module.__name__.rpartition(".")[2].startswith("test_vllm_"):
        yield
        return
    before = _rollout_loaded()
    yield
    assert before or not _rollout_loaded()
