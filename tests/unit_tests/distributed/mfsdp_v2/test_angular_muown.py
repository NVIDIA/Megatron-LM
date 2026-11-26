# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compare FSDP AngularMuOwn training losses with an unsharded reference."""

import copy

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    FsdpContext,
    Placements,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.placement import RowAtomic

pytest.importorskip("emerging_optimizers")
from emerging_optimizers import utils as eo_utils
from emerging_optimizers.orthogonalized_optimizers.muon import Muon

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.angular_muown import (
    FsdpAngularMuOwn,
)
from tests.unit_tests.distributed.mfsdp_v2.test_muon import TinyModel, _train

# The angular decay starts after the second step, so kappa_t < 1 for the rest of the run.
KWARGS = dict(lr=0.05, momentum=0.9, angular_warmup_steps=1, angular_decay_scale=0.5)


class ReferenceAngularMuOwn(Muon):
    """Unsharded AngularMuOwn on whole matrices, written from the update rule.

    It reuses EO Muon's orthogonalize(), with FsdpAngularMuOwn's default hyperparameters.
    """

    def __init__(self, params, *, angular_warmup_steps, angular_decay_scale, **kwargs) -> None:
        super().__init__(params, scale_mode="shape_scaling", **kwargs)
        self.warmup_steps = angular_warmup_steps
        self.decay_scale = angular_decay_scale
        self.angular_step = 0

    @torch.no_grad()
    def step(self, closure=None):
        (beta1, beta2), eps = (0.9, 0.95), 1e-8
        # kappa_t = (1 + c * max(0, t - t_w))^(-1), with t counted from zero.
        kappa = 1 / (1 + self.decay_scale * max(0, self.angular_step - self.warmup_steps))
        self.angular_step += 1
        for group in self.param_groups:
            lr, beta = group["lr"], group["momentum"]
            for p in group["params"]:
                state = self.state[p]
                if not state:
                    state["M"] = torch.zeros_like(p)
                    state["g"] = p.norm(dim=1, keepdim=True)
                    state["m"] = torch.zeros_like(state["g"])
                    state["v"] = torch.zeros_like(state["g"])
                    state["t"] = 0
                state["t"] += 1
                g, M, m, v, t = state["g"], state["M"], state["m"], state["v"], state["t"]
                G, U = p.grad.float(), p / g
                # Split the gradient into the row-scale part d and the direction part dU.
                d = (G * U).sum(dim=1, keepdim=True)
                dU = g * (G - d * U)
                M.mul_(beta).add_(dU)
                M_tilde = dU + beta * M if self.nesterov else M
                with eo_utils.fp32_matmul_precision(self.fp32_matmul_prec):
                    hyperparameters = {k: val for k, val in group.items() if k != "params"}
                    O = self.orthogonalize(p, M_tilde, **hyperparameters)
                # Rotate the unit rows, then take the Adam step on the row scales.
                U_plus = U - lr * kappa * O
                m.mul_(beta1).add_((1 - beta1) * d)
                v.mul_(beta2).add_((1 - beta2) * d * d)
                adam = (m / (1 - beta1**t)) / ((v / (1 - beta2**t)).sqrt() + eps)
                g.sub_(lr * adam)
                g.copy_(torch.copysign(g.abs().clamp_min(1e-7), g))
                p.copy_(g * U_plus / U_plus.norm(dim=1, keepdim=True))


@pytest.mark.parametrize("nesterov", [False, True], ids=["momentum", "nesterov"])
def test_angular_muown_matches_reference(distributed_setup, nesterov):
    """FSDP and unsharded AngularMuOwn should produce the same loss curves."""
    world_size = distributed_setup.world_size
    if world_size < 2 or world_size % 2:
        pytest.skip("Requires an even number of ranks for two-rank DP meshes.")
    device = distributed_setup.device
    mesh = init_device_mesh("cuda", (world_size // 2, 2), mesh_dim_names=("replica", "dp"))["dp"]
    placements = Placements(
        dp_axes=[0], parameter=[RowAtomic()], gradient=[RowAtomic()], optimizer=[RowAtomic()]
    )
    torch.manual_seed(1234)
    model = TinyModel().to(device=device)
    reference = copy.deepcopy(model)
    with fully_shard_context(FsdpContext(device=device)):
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                fully_shard(layer, mesh=mesh, placements=placements)
    params, reference_params = dict(model.named_parameters()), dict(reference.named_parameters())
    other_names = ["block2.3.weight", "block2.3.bias"]
    muon_names = [name for name in params if name not in other_names]

    optimizer = FsdpAngularMuOwn([params[n] for n in muon_names], nesterov=nesterov, **KWARGS)
    fully_shard_optimizer(optimizer, precision_aware=True)
    other_optimizer = torch.optim.SGD([params[n] for n in other_names], lr=0.01, foreach=False)
    fully_shard_optimizer(other_optimizer)
    reference_optimizer = ReferenceAngularMuOwn(
        [reference_params[n] for n in muon_names], nesterov=nesterov, **KWARGS
    )
    reference_other = torch.optim.SGD([reference_params[n] for n in other_names], lr=0.01)

    # Every rank uses identical batches, so the reference needs no gradient reduction.
    inputs = torch.randn(6, 4, 8, device=device)
    reference_losses = _train(reference, (reference_optimizer, reference_other), inputs)
    losses = _train(model, (optimizer, other_optimizer), inputs)
    torch.testing.assert_close(losses, reference_losses, rtol=1e-5, atol=1e-6)
