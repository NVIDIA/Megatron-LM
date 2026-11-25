# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compare FSDP Muon training losses with unsharded Emerging Optimizers Muon."""

import copy

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
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


def _mlp(*widths: int) -> nn.Sequential:
    """Bias-free linear layers of the given widths with ReLU in between."""
    layers: list[nn.Module] = []
    for fan_in, fan_out in zip(widths, widths[1:]):
        layers += [nn.Linear(fan_in, fan_out, bias=False), nn.ReLU()]
    return nn.Sequential(*layers[:-1])


def _mesh_and_placements(dp_mode: str, world_size: int) -> tuple[DeviceMesh, Placements]:
    """Build the DP mesh and placements of an FSDP, HSDP, or HFSDP run."""
    if dp_mode == "fsdp":
        if world_size < 2 or world_size % 2:
            pytest.skip("Requires an even number of ranks for two-rank DP meshes.")
        # With more than two GPUs, also exercise meshes whose global ranks do not start at zero.
        mesh = init_device_mesh("cuda", (world_size // 2, 2), mesh_dim_names=("replica", "dp"))
        placements = Placements(
            dp_axes=[0], parameter=[RowAtomic()], gradient=[RowAtomic()], optimizer=[RowAtomic()]
        )
        return mesh["dp"], placements
    if world_size < 4 or world_size % 4:
        pytest.skip("Requires a multiple of four ranks for (2, 2) DP meshes.")
    mesh = init_device_mesh(
        "cuda", (world_size // 4, 2, 2), mesh_dim_names=("replica", "dp_outer", "dp_shard")
    )
    # HSDP replicates the optimizer state across dp_outer; HFSDP also shards it there.
    outer_optimizer = Replicate() if dp_mode == "hsdp" else RowAtomic()
    placements = Placements(
        dp_axes=[0, 1],
        parameter=[Replicate(), RowAtomic()],
        gradient=[Partial("avg"), RowAtomic()],
        optimizer=[outer_optimizer, RowAtomic()],
    )
    return mesh["dp_outer", "dp_shard"], placements


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


def _param_groups(groups: list[dict], module: nn.Module) -> list[dict]:
    """Replace the parameter names in ``groups`` with ``module``'s parameters."""
    params = dict(module.named_parameters())
    return [{**group, "params": [params[name] for name in group["params"]]} for group in groups]


def _assert_matches_reference(
    model: nn.Module,
    muon_groups: list[dict],
    kwargs: dict,
    *,
    mesh: DeviceMesh,
    placements: Placements,
    shard_units: list[nn.Module],
    sgd_names: list[str] | None = None,
    num_groups_in_flight: int = 2,
) -> None:
    """Train ``model`` sharded with FsdpMuon and unsharded with Muon; the losses must match.

    Each of ``shard_units`` becomes one FSDP group. ``muon_groups`` are Muon parameter groups
    whose "params" are parameter names; the parameters named in ``sgd_names`` use SGD instead.
    """
    weight = next(model.parameters())
    # Match FSDP's FP32 optimizer weights; autocast handles BF16 computation.
    reference = copy.deepcopy(model).float()
    with fully_shard_context(FsdpContext(device=weight.device)):
        for unit in shard_units:
            fully_shard(unit, mesh=mesh, placements=placements)

    optimizer = muon.FsdpMuon(
        _param_groups(muon_groups, model), num_groups_in_flight=num_groups_in_flight, **kwargs
    )
    fully_shard_optimizer(optimizer, precision_aware=True)
    optimizers = [optimizer]
    reference_optimizers = [Muon(_param_groups(muon_groups, reference), **kwargs)]
    if sgd_names:
        sgd_group = [{"params": sgd_names}]
        other_optimizer = torch.optim.SGD(_param_groups(sgd_group, model), lr=0.01, foreach=False)
        fully_shard_optimizer(other_optimizer)
        optimizers.append(other_optimizer)
        reference_optimizers.append(torch.optim.SGD(_param_groups(sgd_group, reference), lr=0.01))

    # Every rank uses identical batches, so the reference needs no gradient reduction.
    inputs = torch.randn(5, 4, 8, device=weight.device, dtype=weight.dtype)
    reference_losses = _train(reference, tuple(reference_optimizers), inputs)
    losses = _train(model, tuple(optimizers), inputs)
    torch.testing.assert_close(losses, reference_losses, atol=0, rtol=0)


@pytest.mark.parametrize("dp_mode", ["fsdp", "hsdp", "hfsdp"])
@pytest.mark.parametrize("num_groups_in_flight", [0, 2, 8])
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
    distributed_setup,
    dp_mode,
    dtype,
    weight_decay_method,
    nesterov,
    kernel_options,
    num_groups_in_flight,
):
    """FSDP, HSDP, HFSDP, and unsharded Muon should produce identical loss curves.

    Each linear layer is its own FSDP group, so several groups' communication is in flight at once;
    with 8, all four Muon groups are.
    """
    mesh, placements = _mesh_and_placements(dp_mode, distributed_setup.world_size)
    torch.manual_seed(1234)
    model = TinyModel().to(device=distributed_setup.device, dtype=dtype)
    sgd_names = ["block2.3.weight", "block2.3.bias"]
    muon_names = [name for name, _ in model.named_parameters() if name not in sgd_names]
    _assert_matches_reference(
        model,
        [{"params": muon_names}],
        dict(
            lr=0.05,
            momentum=0.9,
            nesterov=nesterov,
            weight_decay_method=weight_decay_method,
            **kernel_options,
        ),
        mesh=mesh,
        placements=placements,
        shard_units=[layer for layer in model.modules() if isinstance(layer, nn.Linear)],
        sgd_names=sgd_names,
        num_groups_in_flight=num_groups_in_flight,
    )


@pytest.mark.parametrize("weight_decay_method", ["decoupled", "independent", "l2", "palm"])
@pytest.mark.parametrize("nesterov", [False, True], ids=["momentum", "nesterov"])
def test_muon_batches_matrices_in_one_fsdp_group(distributed_setup, weight_decay_method, nesterov):
    """The Muon matrices of one FSDP group and optimizer group are updated as one batch."""
    mesh, placements = _mesh_and_placements("fsdp", distributed_setup.world_size)
    torch.manual_seed(1234)
    model = _mlp(8, 16, 16, 4).to(distributed_setup.device)
    _assert_matches_reference(
        model,
        [{"params": ["0.weight", "2.weight", "4.weight"]}],
        dict(lr=0.05, momentum=0.9, nesterov=nesterov, weight_decay_method=weight_decay_method),
        mesh=mesh,
        placements=placements,
        shard_units=[model],
    )


@pytest.mark.parametrize("weight_decay_method", ["decoupled", "independent", "l2", "palm"])
def test_muon_keeps_optimizer_groups_apart(distributed_setup, weight_decay_method):
    """In one FSDP group, each optimizer group's matrices use that group's hyperparameters."""
    mesh, placements = _mesh_and_placements("fsdp", distributed_setup.world_size)
    torch.manual_seed(1234)
    model = _mlp(8, 16, 16, 16, 4).to(distributed_setup.device)
    _assert_matches_reference(
        model,
        [
            {"params": ["0.weight", "4.weight"]},
            {"params": ["2.weight", "6.weight"], "lr": 0.02, "weight_decay": 0.3},
        ],
        dict(lr=0.05, momentum=0.9, weight_decay=0.1, weight_decay_method=weight_decay_method),
        mesh=mesh,
        placements=placements,
        shard_units=[model],
    )
