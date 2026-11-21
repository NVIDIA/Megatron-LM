# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os

import pytest
import torch

from megatron.core.tuning import AutotunePolicy, install
from megatron.core.tuning.policy import DEFAULT_MODULES


@pytest.fixture(scope="session", autouse=True)
def select_local_cuda_device():
    """Bind each torchrun process before an SSM test initializes CUDA or Triton."""
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank % torch.cuda.device_count())


@pytest.fixture(scope="package", autouse=True)
def pinned_triton_autotune():
    """Pin each autotuned Triton kernel to one config, as deterministic runs do.

    In the default mode every fresh test process compiles and times every candidate config of
    each autotuned kernel (the in-tree Mamba and GDP kernels, FLA's gated delta rule) at its first
    shape, which was most of these tests' time. The timed winner depends on the machine, so no test
    can rely on it.
    """
    install(AutotunePolicy(mode="pinned", modules=DEFAULT_MODULES + ("fla",)))
    yield
    install()
