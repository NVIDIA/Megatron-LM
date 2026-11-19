# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compare FSDP AngularMuOwn with an unsharded reference written from its update rule."""

import copy

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    FsdpContext,
    Placements,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
    init_optimizer_state,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.placement import RowAtomic
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.uneven_dtensor import (
    chunk_metadata_by_fqn,
)

pytest.importorskip("emerging_optimizers")
from emerging_optimizers import utils as eo_utils
from emerging_optimizers.orthogonalized_optimizers.muon import Muon

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.angular_muown import (
    FsdpAngularMuOwn,
)

ANGULAR_KWARGS = dict(
    angular_decay_scale=8.9,
    angular_decay_degree=1.0,
    angular_plateau_degree=1.0,
    angular_warmup_steps=1,
    angular_horizon_steps=4,
)
# A flatter start (r = 2) and a longer warmup, so that kappa_t stays 1 for two steps.
PLATEAU_ANGULAR_KWARGS = dict(ANGULAR_KWARGS, angular_plateau_degree=2.0, angular_warmup_steps=2)


def _reference_angular_multiplier(
    t: int, *, warmup_steps: int, horizon_steps: int, decay_scale, decay_degree, plateau_degree
) -> float:
    """kappa_t = (1 + C * u**r) ** (-p / r) with u = (t - t_w) / T, and 1 during warmup."""
    if t <= warmup_steps:
        return 1.0
    u = (t - warmup_steps) / horizon_steps
    return (1 + decay_scale * u**plateau_degree) ** (-decay_degree / plateau_degree)


class ReferenceAngularMuOwn(Muon):
    """Unsharded AngularMuOwn on whole matrices, written step by step from the update rule.

    It reuses EO Muon's orthogonalize() so that only the AngularMuOwn-specific steps differ in
    implementation from the sharded optimizer.
    """

    def __init__(
        self,
        params,
        *,
        angular_decay_scale: float,
        angular_decay_degree: float,
        angular_plateau_degree: float,
        angular_warmup_steps: int,
        angular_horizon_steps: int,
        row_scale_betas: tuple[float, float],
        row_scale_eps: float,
        weight_decay_mult: float,
        **kwargs,
    ) -> None:
        super().__init__(params, **kwargs)
        self.schedule = dict(
            warmup_steps=angular_warmup_steps,
            horizon_steps=angular_horizon_steps,
            decay_scale=angular_decay_scale,
            decay_degree=angular_decay_degree,
            plateau_degree=angular_plateau_degree,
        )
        self.row_scale_betas = row_scale_betas
        self.row_scale_eps = row_scale_eps
        self.weight_decay_mult = weight_decay_mult
        self.num_steps = 0

    @torch.no_grad()
    def step(self, closure=None):
        self.num_steps += 1
        t = self.num_steps
        beta1, beta2 = self.row_scale_betas
        for group in self.param_groups:
            lr, beta = group["lr"], group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["M"] = torch.zeros_like(p)
                    state["g"] = p.norm(dim=1, keepdim=True)
                    state["m"] = torch.zeros_like(state["g"])
                    state["v"] = torch.zeros_like(state["g"])
                g, M, m, v = state["g"], state["M"], state["m"], state["v"]
                G = p.grad.float()

                U = p / g
                d = (G * U).sum(dim=1, keepdim=True)
                dU = g * (G - d * U)
                M.mul_(beta).add_(dU)
                M_tilde = dU + beta * M if self.nesterov else M.clone()
                with eo_utils.fp32_matmul_precision(self.fp32_matmul_prec):
                    kwargs = {k: val for k, val in group.items() if k != "params"}
                    O = self.orthogonalize(p, M_tilde, **kwargs)

                kappa = _reference_angular_multiplier(t, **self.schedule)
                U_plus = U - lr * kappa * O
                U_new = U_plus / U_plus.norm(dim=1, keepdim=True)

                m.mul_(beta1).add_((1 - beta1) * d)
                v.mul_(beta2).add_((1 - beta2) * d * d)
                adam = (m / (1 - beta1**t)) / ((v / (1 - beta2**t)).sqrt() + self.row_scale_eps)
                g.mul_(1 - lr * group["weight_decay"] * self.weight_decay_mult).sub_(lr * adam)
                p.copy_(g * U_new)


class TinyModel(nn.Module):
    """Hidden matrices optimized by AngularMuOwn, with an output layer optimized by SGD."""

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


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(scale_mode="spectral"),
        dict(weight_decay_method="l2"),
        dict(angular_plateau_degree=0.5),
        dict(norm_eps=0.0),
    ],
)
def test_angular_muown_rejects_invalid_options(kwargs):
    """Invalid options fail before any FSDP state is touched."""
    with pytest.raises(ValueError):
        FsdpAngularMuOwn([nn.Parameter(torch.ones(2, 2))], **kwargs)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("nesterov", [False, True], ids=["momentum", "nesterov"])
@pytest.mark.parametrize("weight_decay", [0.0, 0.1], ids=["no_wd", "wd"])
@pytest.mark.parametrize(
    "angular_kwargs", [ANGULAR_KWARGS, PLATEAU_ANGULAR_KWARGS], ids=["decay", "plateau"]
)
def test_angular_muown_matches_reference(
    distributed_setup, dtype, nesterov, weight_decay, angular_kwargs
):
    """FSDP and unsharded AngularMuOwn should produce the same losses and weights."""
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
    with fully_shard_context(FsdpContext(device=device)):
        fully_shard(
            model,
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
        weight_decay=weight_decay,
        nesterov=nesterov,
        scale_mode="shape_scaling",
        row_scale_betas=(0.9, 0.95),
        row_scale_eps=1e-8,
        weight_decay_mult=1.0,
        **angular_kwargs,
    )
    excluded_names = ["block2.3.weight", "block2.3.bias"]
    muon_names = [name for name in params if name not in excluded_names]
    optimizer = FsdpAngularMuOwn([params[name] for name in muon_names], **kwargs)
    fully_shard_optimizer(optimizer, precision_aware=True)
    reference_optimizer = ReferenceAngularMuOwn(
        [reference_params[name] for name in muon_names], **kwargs
    )
    other_optimizer = torch.optim.SGD([params[n] for n in excluded_names], lr=0.01, foreach=False)
    fully_shard_optimizer(other_optimizer)
    reference_other = torch.optim.SGD([reference_params[n] for n in excluded_names], lr=0.01)

    # Every rank uses identical batches, so the reference needs no gradient reduction.
    inputs = torch.randn(6, 4, 8, device=device, dtype=dtype)
    reference_losses = _train(reference, (reference_optimizer, reference_other), inputs)
    losses = _train(model, (optimizer, other_optimizer), inputs)
    torch.testing.assert_close(losses, reference_losses, rtol=1e-5, atol=1e-6)

    # The decay is active by the last step, which uses kappa_t of step 6.
    assert optimizer.last_angular_multiplier == pytest.approx(
        _reference_angular_multiplier(len(inputs), **reference_optimizer.schedule)
    )
    assert all(group["muown_step"] == len(inputs) for group in optimizer.param_groups)
    # MFSDP shards packed buffers unevenly, so compare each rank's local rows with the
    # corresponding reference rows instead of calling DTensor.full_tensor().
    chunks = chunk_metadata_by_fqn(model)
    for name in muon_names:
        param = params[name]
        rows = slice(chunks[name].offsets[0], chunks[name].offsets[0] + chunks[name].sizes[0])
        row_scale = optimizer.state[param]["row_scale"]
        assert isinstance(row_scale, DTensor)
        assert row_scale.shape == (param.shape[0], 1)
        weight = param.to_local()
        # |g_i| = ||W_i|| on every local row.
        torch.testing.assert_close(
            row_scale.to_local().abs(), weight.norm(dim=1, keepdim=True), rtol=1e-5, atol=1e-7
        )
        # The reference orders its arithmetic differently, and Adam's m / sqrt(v) amplifies
        # last-bit differences in rows whose radial gradient is near zero; the absolute tolerance
        # is about 3e-4 of a typical row norm here. A sharding or slicing error would differ by
        # O(1), far beyond these tolerances.
        reference_state = reference_optimizer.state[reference_params[name]]
        torch.testing.assert_close(
            row_scale.to_local(), reference_state["g"][rows], rtol=1e-3, atol=1e-4
        )
        torch.testing.assert_close(weight, reference_params[name][rows], rtol=1e-3, atol=1e-4)


def test_angular_muown_zero_gradient_step(distributed_setup):
    """The state-initializing zero-gradient step leaves weights unchanged without decay.

    The step still renormalizes every row, so the weights match up to rounding.
    """
    world_size = distributed_setup.world_size
    if world_size < 2 or world_size % 2:
        pytest.skip("Requires an even number of ranks for two-rank DP meshes.")
    device = distributed_setup.device
    mesh = init_device_mesh("cuda", (world_size // 2, 2), mesh_dim_names=("replica", "dp"))["dp"]
    torch.manual_seed(1234)
    model = TinyModel().to(device=device)
    with fully_shard_context(FsdpContext(device=device)):
        fully_shard(
            model,
            mesh=mesh,
            placements=Placements(
                dp_axes=[0],
                parameter=[RowAtomic()],
                gradient=[RowAtomic()],
                optimizer=[RowAtomic()],
            ),
        )
    params = [p for name, p in model.named_parameters() if not name.startswith("block2.3")]
    before = [p.to_local().clone() for p in params]
    # T = 0 disables the angular decay, so kappa_t = 1 after warmup too.
    optimizer = FsdpAngularMuOwn(
        params,
        lr=0.05,
        weight_decay=0.0,
        **dict(ANGULAR_KWARGS, angular_warmup_steps=0, angular_horizon_steps=0),
    )
    init_optimizer_state(optimizer)
    assert optimizer.last_angular_multiplier == 1.0
    for param, expected in zip(params, before, strict=True):
        torch.testing.assert_close(param.to_local(), expected, rtol=1e-6, atol=1e-7)
        state = optimizer.state[param]
        assert set(state) == {
            "momentum_buffer",
            "row_scale",
            "row_scale_exp_avg",
            "row_scale_exp_avg_sq",
        }
        torch.testing.assert_close(
            state["row_scale"].to_local(), expected.norm(dim=1, keepdim=True)
        )
