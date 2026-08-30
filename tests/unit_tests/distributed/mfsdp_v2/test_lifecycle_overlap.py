# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the M-FSDP v2 1F1B EP-overlap lifecycle controls.

Covers PR #6949 ("Expose FsdpModule in M-FSDP v2 1F1B Direct"):
- ``FsdpContext.custom_forward_backward_hooks`` relaxes ``set_phase`` so the
  1F1B EP-overlap schedule can drive the lifecycle outside the strict
  RESTING/FORWARD/BACKWARD ordering.
- ``FsdpModule.post_backward`` re-arms the grad-readiness counter so every
  nested FSDP unit re-completes the counter on each backward (the frozen-experts
  fix).
- Delayed-wgrad (``skip_backward_post_hook``) params get an unconstrained
  ``grad_dtype`` so TE's FP32 fused grouped-GEMM wgrad assignment is accepted.

These are distributed tests: run under ``torchrun`` (see the suite's conftest).
"""

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Placements,
    fully_shard,
    fully_shard_context,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.distributed.fsdp.src.megatron_fsdp.mixed_precision import MixedPrecisionPolicy


class SingleLinear(nn.Module):
    """Tiny model with a root bias and one linear child, convenient for lifecycle tests."""

    def __init__(self, dim: int = 4) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.ones(dim))
        self.fc = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""
        return self.fc(x) + self.bias


@pytest.fixture(scope="function")
def dp_mesh(distributed_setup):
    """A single-dimension DP device mesh over the default process group."""
    if distributed_setup.world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return init_device_mesh(
        distributed_setup.device.type, (distributed_setup.world_size,), mesh_dim_names=("dp",)
    )


@pytest.fixture(scope="function")
def placements():
    """All-Shard(0) placements (plain DP / optim-grads-params)."""
    return Placements(dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)])


def _wrap(model, mesh, placements, device, custom_forward_backward_hooks=False):
    """fully_shard a single model under a context with the given flag."""
    policy = MixedPrecisionPolicy(main_params_dtype=torch.bfloat16, main_grads_dtype=torch.bfloat16)
    model = model.to(device=device)
    ctx = fully_shard_context(
        device=device, custom_forward_backward_hooks=custom_forward_backward_hooks
    )
    context = ctx.__enter__()
    try:
        fully_shard(model, mesh=mesh, placements=placements, mixed_precision_policy=policy)
    except BaseException:
        ctx.__exit__(None, None, None)
        raise
    assert isinstance(model, FsdpModule), "fully_shard should attach the FsdpModule mixin."
    return context


def test_custom_forward_backward_hooks_relaxes_phase_check(dp_mesh, placements, distributed_setup):
    """With the flag, ``set_phase`` allows transitions outside the strict table."""
    device = distributed_setup.device
    model = SingleLinear(dim=8)
    context = _wrap(model, dp_mesh, placements, device, custom_forward_backward_hooks=True)

    assert context.custom_forward_backward_hooks is True
    # FORWARD -> BACKWARD is rejected by the strict table, but the overlap path
    # may drive it; with the flag set the check must be relaxed (no raise).
    model.phase = FsdpModule.Phase.FORWARD
    model.phase = FsdpModule.Phase.BACKWARD  # no raise
    model.phase = FsdpModule.Phase.RESTING


def test_custom_forward_backward_hooks_default_enforces_phase_check(
    dp_mesh, placements, distributed_setup
):
    """Without the flag, an out-of-order transition raises RuntimeError."""
    device = distributed_setup.device
    model = SingleLinear(dim=8)
    context = _wrap(model, dp_mesh, placements, device, custom_forward_backward_hooks=False)

    assert context.custom_forward_backward_hooks is False
    model.phase = FsdpModule.Phase.FORWARD
    with pytest.raises(RuntimeError):
        model.phase = FsdpModule.Phase.BACKWARD  # FORWARD -> BACKWARD is invalid
    # Leave the module in a valid state for the rest of the test.
    model.phase = FsdpModule.Phase.RESTING


def test_post_backward_rearms_readiness_counter(dp_mesh, placements, distributed_setup):
    """``post_backward`` resets ``_num_ready_grad_parameters`` so the counter can
    re-complete on the next backward (the frozen-experts fix)."""
    device = distributed_setup.device
    model = SingleLinear(dim=8)
    _wrap(model, dp_mesh, placements, device, custom_forward_backward_hooks=False)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
    x = torch.randn(4, 8, device=device, dtype=torch.bfloat16, requires_grad=True)

    # One full forward+backward through the module's own hooks: the readiness
    # counter reaches total and post_backward fires.
    optimizer.zero_grad(set_to_none=True)
    model(x).sum().backward()

    # post_backward re-armed the counter to 0 (not a stale value that would
    # prevent the next backward from re-completing).
    assert model._num_ready_grad_parameters == 0
    assert model._num_ready_grad_parameters < model._num_trainable_parameters


def test_grad_dtype_stopgap_only_for_delayed_wgrad(dp_mesh, placements, distributed_setup):
    """``skip_backward_post_hook`` params get ``grad_dtype=None``; normal params keep
    the configured main-grad dtype (the PR #6949 stopgap)."""
    device = distributed_setup.device
    model = SingleLinear(dim=8)
    # Simulate a TE delayed-wgrad flag on the linear weight only.
    model.fc.weight.skip_backward_post_hook = True  # type: ignore[attr-defined]
    _wrap(model, dp_mesh, placements, device, custom_forward_backward_hooks=False)

    grad_dtype_by_fqn = {}
    for group in model._parameter_groups:
        for p in group.fsdp_parameters:
            grad_dtype_by_fqn[p.fqns[0]] = p.sharded.grad_dtype

    assert "fc.weight" in grad_dtype_by_fqn
    assert "bias" in grad_dtype_by_fqn
    # Delayed-wgrad param: unconstrained so the FP32 wgrad assignment is accepted.
    assert grad_dtype_by_fqn["fc.weight"] is None
    # Normal param: keeps the configured main-grad dtype.
    assert grad_dtype_by_fqn["bias"] is not None
