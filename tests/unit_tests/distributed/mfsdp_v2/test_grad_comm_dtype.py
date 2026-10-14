# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Gradient reduction dtype tests for MFSDP v2 (``MixedPrecisionPolicy.grad_comm_dtype``).

The reduce-scatter input (partial_grad) must be allocated in ``grad_comm_dtype``, or in the
main-gradient dtype when it is unset, independently of the compute-gradient dtype, and
training must still track an unsharded baseline in either reduction dtype.
"""

import pytest
import torch
import torch.distributed as dist
import transformer_engine.pytorch as te
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard
from transformer_engine.common.recipe import MXFP8BlockScaling

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Placements,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.parameter_group import (
    FsdpParameterGroup,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.quantized_dbuffer import (
    QuantizedDBuffer,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.mixed_precision import MixedPrecisionPolicy


class TinyModel(nn.Module):
    """Two separately shardable linear layers."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the tiny model."""
        return self.fc2(torch.relu(self.fc1(x)))


def _default_placements() -> Placements:
    return Placements(dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)])


@pytest.mark.parametrize(
    ("main_grads_dtype", "grad_comm_dtype", "expected_comm_dtype"),
    [
        # Default: reduce in the main-gradient dtype even though compute grads are bf16.
        (torch.float32, None, torch.float32),
        # Explicit override: reduce in bf16, then accumulate into fp32 main_grad.
        (torch.float32, torch.bfloat16, torch.bfloat16),
        # No main_grads_dtype: both main_grad and the reduction follow the parameter dtype.
        (None, None, torch.bfloat16),
    ],
)
def test_partial_grad_buffer_follows_grad_comm_dtype(
    distributed_setup, monkeypatch, main_grads_dtype, grad_comm_dtype, expected_comm_dtype
):
    """partial_grad is allocated in grad_comm_dtype, defaulting to the main_grad dtype."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = nn.Linear(8, 8, device=device, dtype=torch.bfloat16)
    policy = MixedPrecisionPolicy(
        main_params_dtype=torch.float32,
        main_grads_dtype=main_grads_dtype,
        grad_comm_dtype=grad_comm_dtype,
    )
    with fully_shard_context(device=device):
        fully_shard(
            model, mesh=mesh, placements=_default_placements(), mixed_precision_policy=policy
        )

    # Observe the dtype of the partial-grad buffer FSDP allocates during backward.
    allocated_dtypes: list[torch.dtype] = []
    original = FsdpParameterGroup.allocate_partial_grad_buffer

    def spy(self: FsdpParameterGroup):
        buffer = original(self)
        allocated_dtypes.append(buffer.dtype)
        return buffer

    monkeypatch.setattr(FsdpParameterGroup, "allocate_partial_grad_buffer", spy)

    x = torch.randn(4, 8, device=device, dtype=torch.bfloat16)
    model(x).float().sum().backward()

    assert allocated_dtypes == [expected_comm_dtype], allocated_dtypes
    for name, parameter in model.named_parameters():
        assert parameter.grad.dtype == (main_grads_dtype or torch.bfloat16), name


@pytest.mark.launch_on_gb200
@pytest.mark.skipif(
    torch.cuda.get_device_capability()[0] < 10,
    reason="MXFP8 requires Blackwell-or-newer CUDA hardware.",
)
@pytest.mark.parametrize(
    ("main_grads_dtype", "grad_comm_dtype", "expected_comm_dtype"),
    [
        (torch.float32, None, torch.float32),
        (torch.float32, torch.bfloat16, torch.bfloat16),
        # MXFP8 weights are stored as uint8 planes, but their gradients are bf16, so the
        # fallback must follow the parameter's compute dtype, not the storage dtype.
        (None, None, torch.bfloat16),
    ],
)
def test_mxfp8_partial_grad_buffer_follows_grad_comm_dtype(
    distributed_setup, monkeypatch, main_grads_dtype, grad_comm_dtype, expected_comm_dtype
):
    """The quantized (MXFP8) parameter group reduces its gradients in grad_comm_dtype too."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    recipe = MXFP8BlockScaling()
    with te.quantized_model_init(recipe=recipe, preserve_high_precision_init_val=True):
        torch.manual_seed(2026)
        model = te.Linear(64, 128, params_dtype=torch.bfloat16, device=device)
    policy = MixedPrecisionPolicy(
        main_params_dtype=torch.float32,
        main_grads_dtype=main_grads_dtype,
        grad_comm_dtype=grad_comm_dtype,
    )
    with fully_shard_context(device=device):
        fully_shard(
            model, mesh=mesh, placements=_default_placements(), mixed_precision_policy=policy
        )
    # The weight is an MXFP8 group (QuantizedDBuffer); the bias is a regular bf16 group.
    groups_by_kind = {
        isinstance(g.model_weight, QuantizedDBuffer): g for g in model.parameter_groups
    }
    assert set(groups_by_kind) == {True, False}
    for group in groups_by_kind.values():
        assert group.grad_comm_dtype == expected_comm_dtype
        assert group.main_grad.dtype == (main_grads_dtype or torch.bfloat16)

    allocated_dtypes: list[torch.dtype] = []
    original = FsdpParameterGroup.allocate_partial_grad_buffer

    def spy(self: FsdpParameterGroup):
        buffer = original(self)
        allocated_dtypes.append(buffer.dtype)
        return buffer

    monkeypatch.setattr(FsdpParameterGroup, "allocate_partial_grad_buffer", spy)

    # FP8 GEMMs need the token count divisible by 8 and the feature dim by 16.
    x = torch.randn(32, 64, device=device, dtype=torch.bfloat16)
    with te.autocast(recipe=recipe):
        output = model(x)
    output.float().sum().backward()

    # One partial buffer per group (weight group and bias group).
    assert allocated_dtypes == [expected_comm_dtype] * 2, allocated_dtypes


@pytest.mark.parametrize(
    ("grad_comm_dtype", "atol"),
    [
        # fp32 reduction: the only difference from the baseline is the reduction order of
        # fp32 sums of bf16-representable values; observed bit-exact on 4 to 256 ranks.
        pytest.param(None, 1e-6, id="fp32"),
        # bf16 reduction rounds the partial sums of distinct per-rank gradients to bf16.
        # Measured max |loss diff| over 5 seeds: 4.5e-4 on 4 ranks, 1.0e-3 on 16 ranks (it
        # keeps growing with the DP size: 1.6e-2 on 256 ranks with a 256-wide hidden layer).
        pytest.param(torch.bfloat16, 3e-3, id="bf16"),
    ],
)
def test_grad_comm_dtype_losses_match_baseline(distributed_setup, grad_comm_dtype, atol):
    """Mixed-precision FSDP Adam tracks an unsharded mixed-precision Adam baseline.

    The baseline keeps fp32 master weights and computes in bf16 autocast, i.e. the same
    scheme as FSDP with fp32 main params/grads, so the only remaining difference is the
    gradient reduction. Every rank feeds distinct data so the reduce-scatter really averages
    different gradients; the baseline all-reduces its gradients to match.
    """
    rank = distributed_setup.rank
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    torch.manual_seed(2026)
    model = TinyModel().to(device=device, dtype=torch.bfloat16)
    baseline = TinyModel().to(device=device, dtype=torch.float32)
    baseline.load_state_dict({name: value.float() for name, value in model.state_dict().items()})
    policy = MixedPrecisionPolicy(
        main_params_dtype=torch.float32,
        main_grads_dtype=torch.float32,
        grad_comm_dtype=grad_comm_dtype,
    )
    with fully_shard_context(device=device):
        for layer in (model.fc1, model.fc2):
            fully_shard(
                layer, mesh=mesh, placements=_default_placements(), mixed_precision_policy=policy
            )

    baseline_optimizer = torch.optim.Adam(baseline.parameters(), lr=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    fully_shard_optimizer(optimizer)

    # Keep initialization identical, but exercise reduction of distinct rank-local gradients.
    torch.manual_seed(5678 + rank)
    x = torch.randn(3, 8, device=device, dtype=torch.bfloat16)
    target = torch.randn(3, 4, device=device, dtype=torch.bfloat16)

    def train(module, module_optimizer, *, reduce_grads: bool) -> list[torch.Tensor]:
        losses = []
        for _ in range(3):
            module_optimizer.zero_grad()
            # The fp32 baseline computes in bf16 like the FSDP model's bf16 compute weights.
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=reduce_grads):
                output = module(x)
            loss = torch.nn.functional.mse_loss(output.float(), target.float())
            losses.append(loss.detach())
            loss.backward()
            if reduce_grads:
                # The unsharded baseline needs the same DP average as FSDP.
                for parameter in module.parameters():
                    dist.all_reduce(parameter.grad, op=dist.ReduceOp.AVG)
            module_optimizer.step()
        return losses

    baseline_losses = train(baseline, baseline_optimizer, reduce_grads=True)
    sharded_losses = train(model, optimizer, reduce_grads=False)

    torch.testing.assert_close(
        torch.stack(sharded_losses), torch.stack(baseline_losses), rtol=0, atol=atol
    )
