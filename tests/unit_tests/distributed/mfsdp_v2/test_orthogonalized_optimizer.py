# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Distributed tests for the MFSDP v2 owner-compute orthogonalized (Muon) optimizer.

The numerics tests run under `torchrun` and compare the sharded FSDP optimizer against a
single-rank reference using the same Newton-Schulz kernel (bitwise) and against
`torch.optim.Muon` (tolerance, since the kernels normalize differently).
"""

import contextlib
from typing import Any, cast

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from emerging_optimizers.orthogonalized_optimizers.muon import Muon
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import Placements, fully_shard
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.orthogonalized_optimizer import (
    FsdpMuon,
    FsdpOrthogonalizedOptimizer,
    _require_emerging_optimizers,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.owner_planning import (
    GroupOwnerLayout,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.parameter_group import (
    get_containing_parameter_group,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.placement import Flat

try:
    from tests.unit_tests.distributed.mfsdp_v2.conftest import DistributedSetup
except Exception:  # pragma: no cover
    import dataclasses

    @dataclasses.dataclass(frozen=True)
    class DistributedSetup:
        rank: int
        world_size: int
        device: torch.device


_require_emerging_optimizers()


@pytest.fixture(scope="function")
def distributed_setup():
    """Same as the bucket conftest fixture but pins to `local_rank % device_count`."""
    import os

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("Not running under torchrun. Use torchrun to run this test file.")
    os.environ.pop("NCCL_MAXNCHANNELS", None)
    os.environ.pop("NCCL_NVLS_ENABLE", None)
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank % torch.cuda.device_count())
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device("cpu")
    yield DistributedSetup(rank=rank, world_size=world_size, device=device)
    if dist.is_initialized():
        if device.type == "cuda":
            dist.barrier(device_ids=[device.index])
        else:
            dist.barrier()


# ---------------------------------------------------------------------------
# Distributed tests: full optimizer step numerics
# ---------------------------------------------------------------------------


def _flat_placements() -> Placements:
    """All-`Flat` placements for the models in this file."""
    return Placements(dp_axes=[0], parameter=[Flat()], gradient=[Flat()], optimizer=[Flat()])


def _needs_p2p_ranks(distributed_setup):
    """Skip unless the config can run the owner-compute P2P step."""
    world_size = distributed_setup.world_size
    if world_size < 2 or torch.cuda.device_count() < world_size:
        pytest.skip(
            "Needs >=2 ranks and >=1 GPU per rank; a 1-GPU-multi-rank config cannot "
            "run the owner-compute P2P step."
        )
    return init_device_mesh(distributed_setup.device.type, (world_size,))


class TinyModel(nn.Module):
    """Two separately shardable 2D linears (no bias, so all params are matrix params)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


class BoundaryModel(nn.Module):
    """A single 2D linear whose weight is sized to straddle the DP rank boundary."""

    def __init__(self, rows: int, in_features: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, rows, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class Bias1d(nn.Module):
    """A single 1D parameter in its own module so it can be Flat-sharded separately."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bias


class MixedModel(nn.Module):
    """A 2D linear (routed to `FsdpMuon`) and a 1D bias (routed to a separate optimizer)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 4, bias=False)
        self.relu = nn.ReLU()
        self.bias_mod = Bias1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bias_mod(self.relu(self.fc(x)))


class NonMatrixModel(nn.Module):
    """A model whose only parameter is 1D; the base class accepts it, the kernel does not."""

    def __init__(self) -> None:
        super().__init__()
        self.bias_mod = Bias1d(8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bias_mod(x)


class ContractionLinear(nn.Module):
    """A `Linear`-like module whose weight may have any shape with a fixed `(dim-0,
    trailing-product)`.

    The forward contracts the weight back to a matrix, so weights of different
    dimensionalities compute the same function and receive the same flat gradients.
    """

    def __init__(self, weight_shape: tuple[int, ...]) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(weight_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.view(self.weight.shape[0], -1).T


class ContractionModel(nn.Module):
    """Two `ContractionLinear` layers; different weight shapes compute the same function."""

    def __init__(self, shapes: tuple[tuple[int, ...], ...]) -> None:
        super().__init__()
        self.fc1 = ContractionLinear(shapes[0])
        self.fc2 = ContractionLinear(shapes[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


SHAPES_2D = (torch.Size((8, 4)), torch.Size((4, 8)))
SHAPES_ND = (torch.Size((8, 2, 2)), torch.Size((4, 2, 2, 2)))


def _make_fsdp_model(device: torch.device, mesh, seed: int = 1234) -> TinyModel:
    """Build the sharded `TinyModel` with deterministic initial weights."""
    torch.manual_seed(seed)
    model = TinyModel().to(device)
    fully_shard(model.fc1, mesh=mesh, placements=_flat_placements())
    fully_shard(model.fc2, mesh=mesh, placements=_flat_placements())
    return model


def _make_muon(params, **kwargs):
    """Build a `Muon` inner optimizer with standard test defaults.

    `cast(Any, ...)`: the upstream `@registry.register_optimizer` decorator's annotations
    erase the subclass (`Callable[[type[Optimizer]], type[Optimizer]]`), so `ty` only sees
    `Muon` as its base class and cannot check the constructor. Drop the cast once upstream
    ships the generic identity decorator.
    """
    return cast(Any, Muon)(
        params,
        lr=kwargs.get("lr", 0.05),
        momentum=kwargs.get("momentum", 0.9),
        weight_decay=kwargs.get("weight_decay", 0.0),
        nesterov=kwargs.get("nesterov", True),
        coefficient_type=kwargs.get("coefficient_type", "quintic"),
        num_ns_steps=kwargs.get("num_ns_steps", 5),
        scale_mode=kwargs.get("scale_mode", "spectral"),
        fp32_matmul_prec=kwargs.get("fp32_matmul_prec", "medium"),
        use_syrk=kwargs.get("use_syrk", False),
    )


def test_compute_orthogonalization_inputs_matches_reference(distributed_setup):
    """Local pre-NS (weight decay + momentum + Nesterov) matches a plain reference."""
    mesh = _needs_p2p_ranks(distributed_setup)
    device = distributed_setup.device
    model = _make_fsdp_model(device, mesh)
    x = torch.randn(4, 8, device=device)

    inner_optimizer = _make_muon(model.parameters())
    optimizer = FsdpMuon(model.parameters(), inner_optimizer=inner_optimizer, dp_mesh=mesh)
    optimizer.zero_grad(set_to_none=True)
    model(x).sum().backward()

    # The sharded parameters are `nn.Parameter`s wrapping `DTensor`s — they carry
    # `DTensor` methods at runtime, which the static types cannot express.
    param = cast(DTensor, model.fc1.weight)
    optimizer._init_group(optimizer.param_groups[0], skip_non_grad_params=False)
    pre_ns = optimizer._compute_orthogonalization_inputs(
        param, cast(DTensor, param.grad), optimizer.param_groups[0], optimizer.param_groups[0]["lr"]
    )

    # Reference: same math on the local shard with a plain Muon optimizer state.
    ref_param = nn.Parameter(param.to_local().clone())
    ref_opt = _make_muon([ref_param])
    ref_param.grad = cast(DTensor, param.grad).to_local().clone()
    with torch.no_grad():
        ref_opt._init_group(ref_opt.param_groups[0])
        ref_mom = ref_opt.state[ref_param]["momentum_buffer"]
        ref_grad = ref_param.grad.to(ref_mom.dtype)
        ref_opt._apply_weight_decay_inplace(ref_param, ref_grad, 0.05, 0.0)
        ref_mom.lerp_(ref_grad, 1 - 0.9)
        ref_pre = ref_grad.lerp(ref_mom, 0.9)
    torch.testing.assert_close(pre_ns, ref_pre, atol=0, rtol=0)


def test_step_bitwise_matches_single_rank_reference(distributed_setup):
    """The sharded FSDP Muon step must match a single-rank Muon with the same kernel."""
    mesh = _needs_p2p_ranks(distributed_setup)
    device = distributed_setup.device

    sharded = _make_fsdp_model(device, mesh)
    torch.manual_seed(1234)
    baseline = TinyModel().to(device)
    for name, shard_param in (
        ("fc1.weight", sharded.fc1.weight),
        ("fc2.weight", sharded.fc2.weight),
    ):
        full = _gather_full_param(shard_param, mesh)
        with torch.no_grad():
            getattr(baseline, name.split(".")[0]).weight.copy_(full)

    sharded_opt = FsdpMuon(
        sharded.parameters(), inner_optimizer=_make_muon(sharded.parameters()), dp_mesh=mesh
    )
    base_opt = _make_muon(baseline.parameters())

    x = torch.randn(4, 8, device=device)
    for _ in range(3):
        sharded_opt.zero_grad(set_to_none=True)
        base_opt.zero_grad(set_to_none=True)
        sharded(x).sum().backward()
        baseline(x).sum().backward()
        sharded_opt.step()
        base_opt.step()

    for name, shard_param in (
        ("fc1.weight", sharded.fc1.weight),
        ("fc2.weight", sharded.fc2.weight),
    ):
        full = _gather_full_param(shard_param, mesh)
        expected = getattr(baseline, name.split(".")[0]).weight
        torch.testing.assert_close(full, expected, atol=0, rtol=0)


def test_step_explicit_boundary_param_bitwise_matches_reference(distributed_setup):
    """A parameter explicitly straddling the DP boundary is handled correctly."""
    mesh = _needs_p2p_ranks(distributed_setup)
    device = distributed_setup.device
    rows, in_features = 16, 8

    torch.manual_seed(1234)
    model = BoundaryModel(rows, in_features).to(device)
    fully_shard(model.fc, mesh=mesh, placements=_flat_placements())

    # Assert the single parameter is a boundary parameter on this rank.
    fsdp_group = get_containing_parameter_group(cast(nn.Parameter, model.fc.weight))
    assert fsdp_group is not None
    owner_layout = GroupOwnerLayout.from_group(fsdp_group)
    assert owner_layout.layouts[0].is_boundary(), "BoundaryModel weight must straddle the boundary."

    torch.manual_seed(1234)
    baseline = BoundaryModel(rows, in_features).to(device)
    full = _gather_full_param(model.fc.weight, mesh)
    with torch.no_grad():
        baseline.fc.weight.copy_(full)

    sharded_opt = FsdpMuon(
        model.parameters(), inner_optimizer=_make_muon(model.parameters()), dp_mesh=mesh
    )
    base_opt = _make_muon(baseline.parameters())

    x = torch.randn(4, in_features, device=device)
    for _ in range(3):
        sharded_opt.zero_grad(set_to_none=True)
        base_opt.zero_grad(set_to_none=True)
        model(x).sum().backward()
        baseline(x).sum().backward()
        sharded_opt.step()
        base_opt.step()

    full = _gather_full_param(model.fc.weight, mesh)
    torch.testing.assert_close(full, baseline.fc.weight, atol=0, rtol=0)


def test_step_mixed_params_with_separate_optimizer(distributed_setup):
    """1D params are routed to a separate optimizer; 2D params to `FsdpMuon`."""
    mesh = _needs_p2p_ranks(distributed_setup)
    device = distributed_setup.device

    torch.manual_seed(1234)
    model = MixedModel().to(device)
    fully_shard(model.fc, mesh=mesh, placements=_flat_placements())
    fully_shard(model.bias_mod, mesh=mesh, placements=_flat_placements())

    torch.manual_seed(1234)
    baseline = MixedModel().to(device)
    with torch.no_grad():
        baseline.fc.weight.copy_(_gather_full_param(model.fc.weight, mesh))
        baseline.bias_mod.bias.copy_(_gather_full_param(model.bias_mod.bias, mesh))

    # The 2D weight goes to `FsdpMuon`; the 1D bias goes to a separate optimizer.
    matrix_opt = FsdpMuon(
        [model.fc.weight], inner_optimizer=_make_muon([model.fc.weight]), dp_mesh=mesh
    )
    bias_opt = torch.optim.SGD([model.bias_mod.bias], lr=0.05, momentum=0.9)
    base_matrix_opt = _make_muon([baseline.fc.weight])
    base_bias_opt = torch.optim.SGD([baseline.bias_mod.bias], lr=0.05, momentum=0.9)

    x = torch.randn(4, 8, device=device)
    for _ in range(3):
        for opt in (matrix_opt, bias_opt, base_matrix_opt, base_bias_opt):
            opt.zero_grad(set_to_none=True)
        model(x).sum().backward()
        baseline(x).sum().backward()
        matrix_opt.step()
        bias_opt.step()
        base_matrix_opt.step()
        base_bias_opt.step()

    torch.testing.assert_close(
        _gather_full_param(model.fc.weight, mesh), baseline.fc.weight, atol=0, rtol=0
    )
    torch.testing.assert_close(
        _gather_full_param(model.bias_mod.bias, mesh), baseline.bias_mod.bias, atol=0, rtol=0
    )


def test_contraction_matches_2d(distributed_setup):
    """The same flat parameters in >2D shapes evolve identically to the 2D setup.

    `ContractionModel` computes the same function for either shape set, so both models
    receive identical flat gradients; with identical flat layouts (same dim-0 and
    trailing products), the contracted `(shape[0], rest)` matrices are the same
    matrices, so the steps must agree bitwise.
    """
    mesh = _needs_p2p_ranks(distributed_setup)
    device = distributed_setup.device

    torch.manual_seed(1234)
    model_2d = ContractionModel(SHAPES_2D).to(device)
    fully_shard(model_2d.fc1, mesh=mesh, placements=_flat_placements())
    fully_shard(model_2d.fc2, mesh=mesh, placements=_flat_placements())

    torch.manual_seed(1234)
    model_nd = ContractionModel(SHAPES_ND).to(device)
    fully_shard(model_nd.fc1, mesh=mesh, placements=_flat_placements())
    fully_shard(model_nd.fc2, mesh=mesh, placements=_flat_placements())

    opt_2d = FsdpMuon(
        model_2d.parameters(), inner_optimizer=_make_muon(model_2d.parameters()), dp_mesh=mesh
    )
    opt_nd = FsdpMuon(
        model_nd.parameters(), inner_optimizer=_make_muon(model_nd.parameters()), dp_mesh=mesh
    )

    torch.manual_seed(4321)
    x = torch.randn(4, 4, device=device)
    for _ in range(3):
        for opt in (opt_2d, opt_nd):
            opt.zero_grad(set_to_none=True)
        model_2d(x).sum().backward()
        model_nd(x).sum().backward()
        opt_2d.step()
        opt_nd.step()
        for shard_2d, shard_nd in (
            (model_2d.fc1.weight, model_nd.fc1.weight),
            (model_2d.fc2.weight, model_nd.fc2.weight),
        ):
            torch.testing.assert_close(
                _gather_full_param(shard_2d, mesh).flatten(),
                _gather_full_param(shard_nd, mesh).flatten(),
                atol=0,
                rtol=0,
            )


def test_base_class_1d_params_error_at_kernel(distributed_setup):
    """The base class accepts 1D params, but the kernel rejects them at step time.

    Like the upstream `OrthogonalizedOptimizer`, `<2D` parameters are not contracted —
    the error surfaces from `orthogonalize` when it is called. `FsdpMuon` rejects them
    at construction instead.
    """
    mesh = _needs_p2p_ranks(distributed_setup)
    device = distributed_setup.device

    torch.manual_seed(1234)
    model = NonMatrixModel().to(device)
    fully_shard(model.bias_mod, mesh=mesh, placements=_flat_placements())

    base_opt = FsdpOrthogonalizedOptimizer(
        model.parameters(), inner_optimizer=_make_muon(model.parameters()), dp_mesh=mesh
    )
    with pytest.raises(ValueError, match=">=2D"):
        FsdpMuon(model.parameters(), inner_optimizer=_make_muon(model.parameters()), dp_mesh=mesh)

    base_opt.zero_grad(set_to_none=True)
    model(torch.randn(4, 8, device=device)).sum().backward()
    with pytest.raises(ValueError, match="Only 2D"):
        base_opt.step()


def test_step_losses_track_torch_muon(distributed_setup):
    """The FSDP Muon step should track `torch.optim.Muon` over several steps."""
    mesh = _needs_p2p_ranks(distributed_setup)
    device = distributed_setup.device

    sharded = _make_fsdp_model(device, mesh)
    torch.manual_seed(1234)
    baseline = TinyModel().to(device)
    for name, shard_param in (
        ("fc1.weight", sharded.fc1.weight),
        ("fc2.weight", sharded.fc2.weight),
    ):
        full = _gather_full_param(shard_param, mesh)
        with torch.no_grad():
            getattr(baseline, name.split(".")[0]).weight.copy_(full)

    sharded_opt = FsdpMuon(
        sharded.parameters(),
        inner_optimizer=_make_muon(
            sharded.parameters(), coefficient_type="simple", scale_mode="shape_scaling"
        ),
        dp_mesh=mesh,
    )
    base_opt = torch.optim.Muon(
        baseline.parameters(), lr=0.05, momentum=0.9, weight_decay=0.0, nesterov=True, ns_steps=5
    )

    x = torch.randn(4, 8, device=device)
    sharded_losses, base_losses = [], []
    for _ in range(5):
        sharded_opt.zero_grad(set_to_none=True)
        base_opt.zero_grad(set_to_none=True)
        sharded_loss = sharded(x).sum()
        base_loss = baseline(x).sum()
        sharded_losses.append(sharded_loss.detach())
        base_losses.append(base_loss.detach())
        sharded_loss.backward()
        base_loss.backward()
        sharded_opt.step()
        base_opt.step()

    torch.testing.assert_close(
        torch.stack(sharded_losses),
        torch.stack(base_losses),
        atol=5e-2,
        rtol=2e-1,
        msg="FSDP Muon losses did not track torch.optim.Muon within the tolerance.",
    )


def _gather_full_param(param, mesh):
    """All-gather a Flat-sharded parameter into its full tensor."""
    local = param.to_local().contiguous()
    parts = [torch.empty_like(local) for _ in range(mesh.size())]
    dist.all_gather(parts, local, group=mesh.get_group())
    return torch.cat(parts)[: param.numel()].view(param.shape)


# ---------------------------------------------------------------------------
# Import guard tests
# ---------------------------------------------------------------------------

_EO_MOD = "megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.orthogonalized_optimizer"


def test_import_guard_emerging_optimizers_available():
    """When emerging_optimizers is installed, HAVE_EMERGING_OPTIMIZERS is True."""
    from emerging_optimizers.orthogonalized_optimizers import Muon as RealMuon  # noqa: I001
    from emerging_optimizers.orthogonalized_optimizers import OrthogonalizedOptimizer as RealEO

    from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
        orthogonalized_optimizer as mod,
    )

    assert mod.HAVE_EMERGING_OPTIMIZERS is True
    assert mod.OrthogonalizedOptimizer is RealEO
    assert mod.Muon is RealMuon


@contextlib.contextmanager
def _simulate_no_emerging_optimizers():
    """Reload `orthogonalized_optimizer` with `emerging_optimizers` blocked."""
    import importlib
    import sys

    class _BlockEO:
        @staticmethod
        def find_spec(name, _path, _target=None):
            if name == "emerging_optimizers" or name.startswith("emerging_optimizers."):
                raise ModuleNotFoundError(name)
            return None

    saved_eo = {
        k: sys.modules.pop(k)
        for k in list(sys.modules)
        if k == "emerging_optimizers" or k.startswith("emerging_optimizers.")
    }
    saved_mod = sys.modules.pop(_EO_MOD, None)

    sys.meta_path.insert(0, _BlockEO)
    try:
        yield importlib.import_module(_EO_MOD)
    finally:
        sys.meta_path.pop(0)
        sys.modules.update(saved_eo)
        if saved_mod is not None:
            sys.modules[_EO_MOD] = saved_mod
        else:
            sys.modules.pop(_EO_MOD, None)


def test_import_guard_without_emerging_optimizers():
    """When emerging_optimizers is not installed, the module falls back gracefully."""
    with _simulate_no_emerging_optimizers() as mod:
        assert mod.HAVE_EMERGING_OPTIMIZERS is False
        assert mod.OrthogonalizedOptimizer is object
        assert mod.Muon is object


def test_import_guard_construction_error_without_emerging_optimizers():
    """Constructing without emerging_optimizers raises ModuleNotFoundError."""
    with _simulate_no_emerging_optimizers() as mod:
        with pytest.raises(ModuleNotFoundError, match="emerging-optimizers"):
            mod.FsdpOrthogonalizedOptimizer([], object(), dp_mesh=None)
        with pytest.raises(ModuleNotFoundError, match="emerging-optimizers"):
            mod.FsdpMuon([], object(), dp_mesh=None)
