# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared fixtures and helpers for all GTP unit tests.
"""

import pytest
import torch
import transformer_engine.pytorch as te
from transformer_engine.pytorch import is_mxfp8_available, is_nvfp4_available
from transformer_engine.pytorch.quantization import FP8GlobalStateManager

from megatron.experimental.gtp import GTPShardedParam
from tests.unit_tests.test_utilities import Utils

# ---------------------------------------------------------------------------
# Fixtures (import into each test module so pytest discovers them)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _torchrun_dist_init():
    """Initialize the torchrun-managed dist group once per module."""
    Utils.initialize_model_parallel()
    yield
    Utils.destroy_model_parallel()


@pytest.fixture(autouse=True)
def reset_fp8_state():
    yield
    FP8GlobalStateManager.reset()


@pytest.fixture(autouse=True)
def reset_gtp_globals():
    """Reset GTP mutable class-level state between tests."""
    yield
    GTPShardedParam._chain_state = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_distributed(fn, required_world_size: int, *args) -> None:
    """Run ``fn(rank, world_size, port, *args)`` on every torchrun rank.

    ``port`` is unused (dist already initialized by torchrun) but kept so
    worker signatures don't need editing.
    """
    actual_world_size = torch.distributed.get_world_size()
    if actual_world_size != required_world_size:
        pytest.skip(
            f"Requires world_size={required_world_size}, "
            f"got {actual_world_size} (launch with torchrun --nproc-per-node={required_world_size})"
        )
    fn(torch.distributed.get_rank(), actual_world_size, None, *args)


def _requires_multi_gpu(n: int = 4):
    if torch.cuda.device_count() < n:
        pytest.skip(f"Requires at least {n} CUDA devices")


def _requires_mxfp8():
    available, reason = is_mxfp8_available(return_reason=True)
    if not available:
        pytest.skip(f"MXFP8 not available: {reason}")


def _requires_nvfp4():
    if not is_nvfp4_available():
        pytest.skip("NVFP4 not available (requires compute capability >= 10.0)")


def _make_gtp_linear(in_f, out_f, gtp_group, dtype=torch.bfloat16, **kwargs):
    """Construct a bias-free GTP-sharded te.Linear on CUDA."""
    return te.Linear(
        in_features=in_f,
        out_features=out_f,
        bias=False,
        params_dtype=dtype,
        device="cuda",
        gtp_group=gtp_group,
        **kwargs,
    )


def _make_gtp_grouped_linear(num_gemms, in_f, out_f, gtp_group, dtype=torch.bfloat16, **kwargs):
    """Construct a bias-free GTP-sharded te.GroupedLinear on CUDA."""
    return te.GroupedLinear(
        num_gemms=num_gemms,
        in_features=in_f,
        out_features=out_f,
        bias=False,
        params_dtype=dtype,
        device="cuda",
        gtp_group=gtp_group,
        **kwargs,
    )
