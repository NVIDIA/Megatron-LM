# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compare FSDP Muon training losses with unsharded Emerging Optimizers Muon."""

import copy

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Placements,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.placement import RowAtomic

pytest.importorskip("emerging_optimizers")
from emerging_optimizers.orthogonalized_optimizers.muon import Muon

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import muon


class TinyModel(nn.Module):
    """Hidden matrices optimized by Muon, with an output layer optimized by SGD."""

    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(8, 16, bias=False), nn.ReLU(), nn.Linear(16, 4, bias=False)
        )
        self.block2 = nn.Sequential(
            nn.Linear(4, 16, bias=False),
            nn.ReLU(),
            nn.Linear(16, 1, bias=False),
            nn.Linear(1, 1),  # Output projection and bias belong to another optimizer.
        )

    def forward(self, x):
        """Run both blocks."""
        return self.block2(self.block1(x))


def _train(
    model: nn.Module, optimizers: tuple[torch.optim.Optimizer, ...], inputs: torch.Tensor
) -> torch.Tensor:
    """Train on the given batches and return the loss curve."""
    losses = []
    for x in inputs:
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=inputs.dtype, enabled=inputs.dtype == torch.bfloat16):
            loss = model(x).float().sum()
        losses.append(loss.detach())
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
    return torch.stack(losses)


@pytest.mark.parametrize("lookahead", [0, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("weight_decay_method", ["decoupled", "independent", "l2", "palm"])
@pytest.mark.parametrize(
    "nesterov,kernel_options",
    [
        pytest.param(False, {}, id="momentum"),
        pytest.param(
            True,
            dict(
                coefficient_type="simple",
                num_ns_steps=3,
                scale_mode="shape_scaling",
                extra_scale_factor=0.2,
            ),
            id="nesterov",
        ),
    ],
)
def test_muon_matches_reference(
    distributed_setup, dtype, weight_decay_method, nesterov, kernel_options, lookahead
):
    """FSDP and unsharded Muon should produce identical loss curves.

    Each linear layer is its own FSDP group, so lookahead pipelines several groups' communication.
    """
    world_size = distributed_setup.world_size
    if world_size < 2 or world_size % 2:
        pytest.skip("Requires an even number of ranks for two-rank DP meshes.")
    device = distributed_setup.device
    # With more than two GPUs, also exercise meshes whose global ranks do not start at zero.
    mesh = init_device_mesh("cuda", (world_size // 2, 2), mesh_dim_names=("replica", "dp"))["dp"]
    torch.manual_seed(1234)
    model = TinyModel().to(device=device, dtype=dtype)
    # Match FSDP's FP32 optimizer weights; autocast handles BF16 computation.
    reference = copy.deepcopy(model).float()
    reference_params = dict(reference.named_parameters())
    with fully_shard_context(device=device):
        for layer in (m for m in model.modules() if isinstance(m, nn.Linear)):
            fully_shard(
                layer,
                mesh=mesh,
                placements=Placements(
                    dp_axes=[0],
                    parameter=[RowAtomic()],
                    gradient=[RowAtomic()],
                    optimizer=[RowAtomic()],
                ),
            )
    params = dict(model.named_parameters())
    kwargs = dict(
        lr=0.05,
        momentum=0.9,
        nesterov=nesterov,
        weight_decay_method=weight_decay_method,
        **kernel_options,
    )
    excluded_names = ["block2.3.weight", "block2.3.bias"]
    muon_names = [name for name in params if name not in excluded_names]
    optimizer = muon.FsdpMuon([params[name] for name in muon_names], lookahead=lookahead, **kwargs)
    fully_shard_optimizer(optimizer, precision_aware=True)
    reference_optimizer = Muon([reference_params[name] for name in muon_names], **kwargs)
    other_optimizer = torch.optim.SGD([params[n] for n in excluded_names], lr=0.01, foreach=False)
    fully_shard_optimizer(other_optimizer)
    reference_other = torch.optim.SGD([reference_params[n] for n in excluded_names], lr=0.01)

    # Every rank uses identical batches, so the reference needs no gradient reduction.
    inputs = torch.randn(5, 4, 8, device=device, dtype=dtype)
    reference_losses = _train(reference, (reference_optimizer, reference_other), inputs)
    losses = _train(model, (optimizer, other_optimizer), inputs)
    torch.testing.assert_close(losses, reference_losses, atol=0, rtol=0)
