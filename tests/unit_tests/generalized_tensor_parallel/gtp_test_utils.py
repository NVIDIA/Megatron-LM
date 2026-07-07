# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared fixtures and helpers for all GTP unit tests.
"""

import pytest
import torch
import transformer_engine.pytorch as te
from transformer_engine.pytorch import is_mxfp8_available, is_nvfp4_available
from transformer_engine.pytorch.quantization import FP8GlobalStateManager

from megatron.core.tensor_parallel.generalized_tensor_parallelism import GTPShardedParam
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


def _make_gtp_linear(in_f, out_f, gtp_remat_group, dtype=torch.bfloat16, **kwargs):
    """Construct a bias-free GTP-sharded te.Linear on CUDA.

    Mirrors the production integration (extensions/transformer_engine.py): TE has no GTP
    construction hooks, so the module is built stock, ``module.gtp_remat_size`` (the forward-gather
    gate) is stamped post-init, and the BF16 weight is sliced by ``wrap_module_params_gtp``.
    """
    from megatron.core.tensor_parallel.gtp import wrap_module_params_gtp

    layer = te.Linear(
        in_features=in_f,
        out_features=out_f,
        bias=False,
        params_dtype=dtype,
        device="cuda",
        **kwargs,
    )
    layer.gtp_remat_size = gtp_remat_group.size()
    wrap_module_params_gtp(layer, layer.weight_names, gtp_remat_group)
    return layer


def _make_gtp_remat_grouped_linear(
    num_gemms, in_f, out_f, gtp_remat_group, dtype=torch.bfloat16, **kwargs
):
    """Construct a bias-free GTP-sharded te.GroupedLinear on CUDA (post-init slice, see
    _make_gtp_linear)."""
    from megatron.core.tensor_parallel.gtp import wrap_module_params_gtp

    layer = te.GroupedLinear(
        num_gemms=num_gemms,
        in_features=in_f,
        out_features=out_f,
        bias=False,
        params_dtype=dtype,
        device="cuda",
        **kwargs,
    )
    layer.gtp_remat_size = gtp_remat_group.size()
    wrap_module_params_gtp(layer, layer.weight_names, gtp_remat_group, is_grouped=True)
    return layer


def _restore_gtp_shards_and_init_main_grad(module, saved_weights, gtp_rank, dtype=torch.bfloat16):
    """Load saved full weights into a GTP_remat_size>1 module and prep it for backward.

    GTPShardedParams receive their ``gtp_rank`` axis-0 shard; replicated params get the full
    tensor. Then pre-allocate ``main_grad`` on every GTPShardedParam (required before the first
    backward). Used by the dense two-phase baseline-vs-GTP tests.
    """
    for name, p in module.named_parameters():
        full = saved_weights[name]
        if isinstance(p, GTPShardedParam):
            shard_size = p.shape[0]
            p.data.copy_(full[gtp_rank * shard_size : (gtp_rank + 1) * shard_size])
        else:
            p.data.copy_(full)
    for p in module.parameters():
        if isinstance(p, GTPShardedParam):
            p.main_grad = torch.zeros(p.shape, dtype=dtype, device='cuda')


def _assert_loss_trajectories_match(baseline_losses, test_losses, steps, label="gtp_remat"):
    """On rank 0: print and assert two per-step loss trajectories match.

    GTP (ZeRO-3-like) reduces grads with a reduce-scatter-sum while the no-GTP baseline
    all-reduces; in BF16 these differ only in reduction order, so the trajectories track to
    BF16 precision (observed max |diff| ~1e-2 over 10 steps) rather than bitwise. The tolerance
    is set for that BF16 noise floor and still trips on any real GTP sharding bug, which diverges
    by O(loss). (The sibling grad-correctness test likewise checks ~1e-2-scale grad error.)
    """
    assert (
        len(baseline_losses) == len(test_losses) == steps
    ), f"loss counts: baseline={len(baseline_losses)} {label}={len(test_losses)} want {steps}"
    for step, (lb, lt) in enumerate(zip(baseline_losses, test_losses)):
        print(f"Step {step:2d}: baseline={lb:.6f}  {label}={lt:.6f}", flush=True)
    torch.testing.assert_close(
        torch.tensor(test_losses), torch.tensor(baseline_losses), atol=5e-2, rtol=5e-2
    )
