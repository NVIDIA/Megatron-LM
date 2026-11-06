# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compare FSDP Muon training losses with unsharded Emerging Optimizers Muon."""

import copy

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Partial, Replicate

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    FsdpContext,
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


@pytest.mark.parametrize("dp_mode", ["fsdp", "hsdp", "hfsdp"])
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
    distributed_setup, dp_mode, dtype, weight_decay_method, nesterov, kernel_options
):
    """FSDP, HSDP, HFSDP, and unsharded Muon should produce identical loss curves."""
    world_size = distributed_setup.world_size
    if dp_mode == "fsdp":
        if world_size < 2 or world_size % 2:
            pytest.skip("Requires an even number of ranks for two-rank DP meshes.")
        # With more than two GPUs, also exercise meshes whose global ranks do not start at zero.
        mesh = init_device_mesh("cuda", (world_size // 2, 2), mesh_dim_names=("replica", "dp"))
        mesh = mesh["dp"]
        placements = Placements(
            dp_axes=[0], parameter=[RowAtomic()], gradient=[RowAtomic()], optimizer=[RowAtomic()]
        )
    else:
        if world_size < 4 or world_size % 4:
            pytest.skip("Requires a multiple of four ranks for (2, 2) DP meshes.")
        mesh = init_device_mesh(
            "cuda", (world_size // 4, 2, 2), mesh_dim_names=("replica", "dp_outer", "dp_shard")
        )
        mesh = mesh["dp_outer", "dp_shard"]
        # HSDP replicates the optimizer state across dp_outer; HFSDP also shards it there.
        outer_optimizer = Replicate() if dp_mode == "hsdp" else RowAtomic()
        placements = Placements(
            dp_axes=[0, 1],
            parameter=[Replicate(), RowAtomic()],
            gradient=[Partial("avg"), RowAtomic()],
            optimizer=[outer_optimizer, RowAtomic()],
        )
    device = distributed_setup.device
    torch.manual_seed(1234)
    model = TinyModel().to(device=device, dtype=dtype)
    # Match FSDP's FP32 optimizer weights; autocast handles BF16 computation.
    reference = copy.deepcopy(model).float()
    reference_params = dict(reference.named_parameters())
    with fully_shard_context(FsdpContext(device=device)):
        fully_shard(model, mesh=mesh, placements=placements)
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
    optimizer = muon.FsdpMuon([params[name] for name in muon_names], **kwargs)
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
    if dp_mode == "hsdp":
        # Each replica runs Newton-Schulz on its own, so nothing re-syncs the replicas:
        # their main weights and momentum must stay bitwise equal.
        local = torch.cat(
            [
                tensor.to_local().flatten()
                for name in muon_names
                for tensor in (params[name], optimizer.state[params[name]]["momentum_buffer"])
            ]
        ).detach()
        replicas = [torch.empty_like(local) for _ in range(mesh.size(0))]
        dist.all_gather(replicas, local, group=mesh.get_group(0))
        for replica in replicas:
            torch.testing.assert_close(replica, local, atol=0, rtol=0)
