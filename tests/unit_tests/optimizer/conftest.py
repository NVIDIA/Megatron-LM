# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared fixtures for the torchrun-launched layer-sharded Muon test modules."""

import os

import pytest
import torch

from tests.unit_tests.test_utilities import Utils

_SEED = 42


@pytest.fixture(scope="module")
def layer_sharded_dist_init():
    """Initialize the torchrun-managed process group for a layer-sharded test module.

    Binds this rank to its GPU and makes it the default device (``Utils`` uses the NCCL
    backend, so exchanged tensors must live there), seeds, and pins the fp32 matmul
    precision to the optimizer's so reference Newton-Schulz calls compare exactly.
    Teardown resets the default device with ``None`` (``set_default_device`` installs a
    global mode that only ``None`` removes; ``"cpu"`` would leave it active for later
    modules of the same torchrun session) and destroys the process groups.
    """
    Utils.initialize_model_parallel()
    cuda = os.environ.get("TEST_DEVICE", "cuda" if torch.cuda.is_available() else "cpu") == "cuda"
    if cuda:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        torch.set_default_device(f"cuda:{local_rank}")
    torch.manual_seed(_SEED)
    prev_prec = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(prev_prec)
    if cuda:
        torch.set_default_device(None)
    Utils.destroy_model_parallel()
