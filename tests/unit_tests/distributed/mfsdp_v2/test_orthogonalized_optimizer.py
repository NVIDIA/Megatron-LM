# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for the M-FSDPv2 owner-compute orthogonalized (Muon) optimizer.

The pure-function tests (shard-plan math, owner assignment, pack/unpack) run on
CPU without a process group. The optimizer-step numerics tests run under
`torchrun` and compare the sharded FSDP optimizer against a single-rank
reference using the same Newton-Schulz kernel (bitwise) and against
`torch.optim.Muon` (tolerance, since the kernels normalize differently).
"""

import contextlib
import types
import weakref

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Flat,
    Placements,
    fully_shard,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.orthogonalized_optimizer import (
    FsdpMuon,
    Muon,
    _require_emerging_optimizers,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.parameter_group import (
    _CONTAINING_PARAMETER_GROUP_ATTR,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.shard_plan import (
    ShardPlan,
    assign_owner_work,
    compute_shard_plan,
    pack_owner_work,
    pack_update_shards,
    reconstruct_full_tensor,
    unpack_update_shards,
)

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
    """Same as the bucket conftest fixture but pins to `local_rank % device_count`.

    The shared `distributed_setup` does `set_device(local_rank)`, which is
    correct when every rank has its own GPU (the CI configuration). On a
    single-GPU dev box running multiple ranks, `local_rank` exceeds the
    device count and `set_device` raises. Pinning modulo the device count is
    identical to the suite fixture when `nproc <= device_count` (the CI case)
    and lets these tests run on one GPU with `NCCL_MULTI_RANK_GPU_ENABLE=1`.
    """
    import os
    from collections.abc import Iterator

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("Not running under torchrun. Use torchrun to run this test file.")
    # Clear the suite-wide NCCL defaults (set in the top-level conftest.py) before
    # init_device_mesh initializes NCCL communicators, matching the bucket
    # conftest fixture. With NCCL_MAX_NCHANNELS=1, multiple concurrent
    # collectives (FSDP reduce-scatter + the owner all-to-all) on several ranks
    # sharing one GPU exhaust the single NCCL channel.
    os.environ.pop("NCCL_MAX_NCHANNELS", None)
    os.environ.pop("NCCL_NVLS_ENABLE", None)
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank % torch.cuda.device_count())
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        device = torch.device("cpu")
    yield DistributedSetup(rank=rank, world_size=world_size, device=device)
    if dist.is_initialized():
        if device.type == "cuda":
            dist.barrier(device_ids=[device.index])
        else:
            dist.barrier()


# ---------------------------------------------------------------------------
# Pure CPU tests: grouping
# ---------------------------------------------------------------------------


def test_group_updates_separates_mixed_dtypes():
    """`_group_updates` groups shards by (device, dtype), separating mixed dtypes."""
    optimizer = object.__new__(FsdpMuon)
    cpu = torch.device("cpu")
    params = [
        nn.Parameter(torch.zeros(2, 2, dtype=torch.float32, device=cpu)),  # 0: fp32
        nn.Parameter(torch.zeros(2, 2, dtype=torch.bfloat16, device=cpu)),  # 1: bf16
        nn.Parameter(torch.zeros(2, 2, dtype=torch.float32, device=cpu)),  # 2: fp32
        nn.Parameter(torch.zeros(2, 2, dtype=torch.bfloat16, device=cpu)),  # 3: bf16
    ]
    shards = [p.detach() for p in params]
    chunks = optimizer._group_updates(params, shards)

    # Two chunks: one for fp32, one for bf16.
    assert len(chunks) == 2
    # Every chunk is homogeneous in dtype.
    for chunk in chunks:
        assert len({shards[i].dtype for i in chunk}) == 1
    # All indices are covered exactly once.
    assert sorted(i for chunk in chunks for i in chunk) == [0, 1, 2, 3]
    # fp32 shards (0, 2) are in a different chunk than bf16 shards (1, 3).
    chunk_of_0 = next(c for c in chunks if 0 in c)
    chunk_of_1 = next(c for c in chunks if 1 in c)
    assert chunk_of_0 is not chunk_of_1
    assert 2 in chunk_of_0
    assert 3 in chunk_of_1


class _MockMesh:
    """Mock `DeviceMesh` whose `get_group()` returns a fixed `ProcessGroup`."""

    def __init__(self, pg: object) -> None:
        self._pg = pg

    def get_group(self) -> object:
        return self._pg


class _MockParamGroup:
    """Mock `FsdpParameterGroup` (supports `weakref`) with a `mesh` attribute."""

    def __init__(self, mesh: _MockMesh) -> None:
        self.mesh = mesh


def test_group_updates_separates_collective_groups():
    """`_group_updates` separates params from different collective groups.

    Even when all params share the same dtype and device, params whose
    `FsdpParameterGroup` resolves to a different `ProcessGroup` (via
    `group.mesh.get_group()`) must land in separate chunks so that each
    chunk's P2P communication uses a single collective group.
    """
    optimizer = object.__new__(FsdpMuon)
    cpu = torch.device("cpu")
    # Mock ProcessGroups (the actual collective groups) and FsdpParameterGroups
    # whose `mesh.get_group()` returns them.
    pg_a = object()  # mock ProcessGroup A
    pg_b = object()  # mock ProcessGroup B
    group_a = _MockParamGroup(_MockMesh(pg_a))
    group_b = _MockParamGroup(_MockMesh(pg_b))
    params = [
        nn.Parameter(torch.zeros(2, 2, dtype=torch.float32, device=cpu)),
        nn.Parameter(torch.zeros(2, 2, dtype=torch.float32, device=cpu)),
        nn.Parameter(torch.zeros(2, 2, dtype=torch.float32, device=cpu)),
        nn.Parameter(torch.zeros(2, 2, dtype=torch.float32, device=cpu)),
    ]
    # 0, 1 -> group_a;  2, 3 -> group_b  (all fp32, same device)
    for p, g in zip(params, [group_a, group_a, group_b, group_b]):
        setattr(p, _CONTAINING_PARAMETER_GROUP_ATTR, weakref.ref(g))
    shards = [p.detach() for p in params]
    chunks = optimizer._group_updates(params, shards)

    # Two chunks: one per collective group.
    assert len(chunks) == 2
    chunk_of_0 = next(c for c in chunks if 0 in c)
    chunk_of_2 = next(c for c in chunks if 2 in c)
    assert chunk_of_0 is not chunk_of_2
    assert 1 in chunk_of_0
    assert 3 in chunk_of_2


# ---------------------------------------------------------------------------
# Distributed tests: full optimizer step numerics
# ---------------------------------------------------------------------------


def _flat_placements() -> Placements:
    return Placements(dp_axes=[0], parameter=[Flat()], gradient=[Flat()], optimizer=[Flat()])


class TinyModel(nn.Module):
    """Two separately shardable 2D linears (no bias, so all params are matrix params)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


class MixedDtypeModel(nn.Module):
    """Two 2D linears with different dtypes (fp32 + bf16) to exercise mixed-dtype grouping."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16, bias=False)  # fp32
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 4, bias=False).to(torch.bfloat16)  # bf16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        return self.fc2(x.to(self.fc2.weight.dtype))


class BoundaryModel(nn.Module):
    """A single 2D linear whose weight is sized to straddle the DP rank boundary.

    With `world_size` ranks and the all-`Flat` layout, the weight
    `(rows, in_features)` (`rows` divisible by `world_size`) occupies the
    whole DBuffer, so every rank owns a contiguous row slice and the parameter
    is a boundary parameter for `world_size > 1`.
    """

    def __init__(self, rows: int, in_features: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, rows, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class Bias1d(nn.Module):
    """A single 1D parameter in its own module so it can be Flat-sharded separately.

    Sharding it alone (its own FSDP group) makes it span all DP ranks with equal
    shards, so it is a boundary parameter that is handled by the non-matrix
    momentum-SGD fallback (no orthogonalization, no owner-compute P2P).
    """

    def __init__(self, n: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bias


class MixedModel(nn.Module):
    """A 2D linear (matrix path) and a 1D bias (non-matrix path), separately sharded.

    `fc.weight` (4, 8) is a 2D boundary parameter updated by the owner-compute
    P2P + Newton-Schulz path; `bias` (4,) is a 1D boundary parameter updated by
    the plain momentum-SGD fallback. They are Flat-sharded in separate FSDP groups
    (one parameter each) so each spans all DP ranks with equal shards.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 4, bias=False)
        self.relu = nn.ReLU()
        self.bias_mod = Bias1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bias_mod(self.relu(self.fc(x)))


class NonMatrixModel(nn.Module):
    """A model whose only parameter is 1D (a bias), exercising the non-matrix path.

    `Bias1d` has a single 1D parameter, Flat-sharded in its own FSDP group so
    it spans all DP ranks with equal shards (a boundary parameter handled by
    the plain momentum-SGD fallback, no orthogonalization, no owner-compute P2P).
    """

    def __init__(self) -> None:
        super().__init__()
        self.bias_mod = Bias1d(8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bias_mod(x)


def _make_fsdp_model(device: torch.device, mesh, seed: int = 1234) -> TinyModel:
    torch.manual_seed(seed)
    model = TinyModel().to(device)
    fully_shard(model.fc1, mesh=mesh, placements=_flat_placements())
    fully_shard(model.fc2, mesh=mesh, placements=_flat_placements())
    return model


def test_compute_orthogonalization_inputs_matches_reference(distributed_setup):
    """Local pre-NS (weight decay + momentum + Nesterov) matches a plain reference."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2 or torch.cuda.device_count() < world_size:
        pytest.skip("Needs >=2 ranks.")
    mesh = init_device_mesh(device.type, (world_size,))
    model = _make_fsdp_model(device, mesh)
    x = torch.randn(4, 8, device=device)

    inner_optimizer = Muon(
        model.parameters(),
        lr=0.05,
        momentum=0.9,
        weight_decay=0.0,
        nesterov=True,
        coefficient_type="quintic",
        num_ns_steps=5,
        scale_mode="spectral",
        fp32_matmul_prec="medium",
        use_syrk=False,
    )
    optimizer = FsdpMuon(
        model.parameters(),
        inner_optimizer=inner_optimizer,
        dp_mesh=mesh,
        use_owner_comm_stream=False,
    )
    optimizer.zero_grad(set_to_none=True)
    model(x).sum().backward()

    param = model.fc1.weight
    optimizer._init_group(optimizer.param_groups[0], skip_non_grad_params=False)
    pre_ns = optimizer._compute_orthogonalization_inputs(
        param, param.grad, optimizer.param_groups[0], optimizer.param_groups[0]["lr"]
    )

    # Reference: same math on the local shard with a plain Muon optimizer state.
    ref_param = nn.Parameter(param.to_local().clone())
    ref_opt = Muon(
        [ref_param],
        lr=0.05,
        momentum=0.9,
        weight_decay=0.0,
        nesterov=True,
        coefficient_type="quintic",
        num_ns_steps=5,
        scale_mode="spectral",
        fp32_matmul_prec="medium",
    )
    ref_param.grad = param.grad.to_local().clone()
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
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2 or torch.cuda.device_count() < world_size:
        pytest.skip(
            "Needs >=2 ranks and >=1 GPU per rank; "
            "a 1-GPU-multi-rank config cannot run the owner-compute P2P step."
        )
    mesh = init_device_mesh(device.type, (world_size,))

    sharded = _make_fsdp_model(device, mesh)
    torch.manual_seed(1234)
    baseline = TinyModel().to(device)
    # Copy full weights from the sharded model's gathered (pre-shard) initial state.
    for name, shard_param in (
        ("fc1.weight", sharded.fc1.weight),
        ("fc2.weight", sharded.fc2.weight),
    ):
        full = _gather_full_param(shard_param, mesh, world_size)
        with torch.no_grad():
            getattr(baseline, name.split(".")[0]).weight.copy_(full)

    lr = 0.05
    momentum = 0.9
    nesterov = True
    weight_decay = 0.0
    inner_optimizer = Muon(
        sharded.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov,
        coefficient_type="quintic",
        num_ns_steps=5,
        scale_mode="spectral",
        fp32_matmul_prec="medium",
        use_syrk=False,
    )
    sharded_opt = FsdpMuon(
        sharded.parameters(),
        inner_optimizer=inner_optimizer,
        dp_mesh=mesh,
        use_owner_comm_stream=False,
    )
    base_opt = Muon(
        baseline.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov,
        coefficient_type="quintic",
        num_ns_steps=5,
        scale_mode="spectral",
        fp32_matmul_prec="medium",
    )

    x = torch.randn(4, 8, device=device)
    for step in range(3):
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
        full = _gather_full_param(shard_param, mesh, world_size)
        expected = getattr(baseline, name.split(".")[0]).weight
        torch.testing.assert_close(full, expected, atol=0, rtol=0)


def test_step_explicit_boundary_param_bitwise_matches_reference(distributed_setup):
    """A parameter explicitly straddling the DP boundary is handled correctly.

    Builds a single-linear model whose weight is a boundary parameter on every
    rank, asserts the shard plan classifies it as boundary, then verifies the
    FSDP Muon step matches a single-rank Muon reference bitwise. This guards the
    owner-gather -> fully-local -> finish restructure and the `pre_ns` argument
    to `_orthogonalize_with_precision` on the boundary path.
    """
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2 or torch.cuda.device_count() < world_size:
        pytest.skip(
            "Needs >=2 ranks and >=1 GPU per rank; "
            "a 1-GPU-multi-rank config cannot run the owner-compute P2P step."
        )
    rows = 16
    in_features = 8
    if rows % world_size != 0:
        pytest.skip(f"rows {rows} not divisible by world_size {world_size}.")
    mesh = init_device_mesh(device.type, (world_size,))

    torch.manual_seed(1234)
    model = BoundaryModel(rows, in_features).to(device)
    fully_shard(model.fc, mesh=mesh, placements=_flat_placements())

    # Assert the single parameter is a boundary parameter on this rank.
    plan = compute_shard_plan(
        torch.Size((rows, in_features)),
        tensor_flat_offset=0,
        rank_flat_shard_size=(rows * in_features) // world_size,
        world_size=world_size,
    )
    assert plan.is_boundary(), "BoundaryModel weight must straddle the DP boundary."

    # Baseline: single-rank model with the same initial full weights.
    torch.manual_seed(1234)
    baseline = BoundaryModel(rows, in_features).to(device)
    full = _gather_full_param(model.fc.weight, mesh, world_size)
    with torch.no_grad():
        baseline.fc.weight.copy_(full)

    lr, momentum, nesterov, weight_decay = 0.05, 0.9, True, 0.0
    inner = Muon(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov,
        coefficient_type="quintic",
        num_ns_steps=5,
        scale_mode="spectral",
        fp32_matmul_prec="medium",
        use_syrk=False,
    )
    sharded_opt = FsdpMuon(
        model.parameters(), inner_optimizer=inner, dp_mesh=mesh, use_owner_comm_stream=False
    )
    base_opt = Muon(
        baseline.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov,
        coefficient_type="quintic",
        num_ns_steps=5,
        scale_mode="spectral",
        fp32_matmul_prec="medium",
    )

    x = torch.randn(4, in_features, device=device)
    for _ in range(3):
        sharded_opt.zero_grad(set_to_none=True)
        base_opt.zero_grad(set_to_none=True)
        model(x).sum().backward()
        baseline(x).sum().backward()
        sharded_opt.step()
        base_opt.step()

    full = _gather_full_param(model.fc.weight, mesh, world_size)
    torch.testing.assert_close(full, baseline.fc.weight, atol=0, rtol=0)


def test_step_reconstruct_full_param_bitwise_matches_reference(distributed_setup):
    """With `reconstruct_full_param=True`, the owner gathers weight shards too.

    Same boundary-param setup as `test_step_explicit_boundary_param_bitwise_matches_reference`,
    but the optimizer is constructed with `reconstruct_full_param=True` so the owner
    performs a second P2P round to gather each rank's local weight shard,
    reconstructs the full parameter, and passes it as the `param` argument to
    `orthogonalize`. The result must still match the single-rank reference
    bitwise, guarding the optional full-parameter reconstruction path.
    """
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2 or torch.cuda.device_count() < world_size:
        pytest.skip(
            "Needs >=2 ranks and >=1 GPU per rank; "
            "a 1-GPU-multi-rank config cannot run the owner-compute P2P step."
        )
    rows = 16
    in_features = 8
    if rows % world_size != 0:
        pytest.skip(f"rows {rows} not divisible by world_size {world_size}.")
    mesh = init_device_mesh(device.type, (world_size,))

    torch.manual_seed(1234)
    model = BoundaryModel(rows, in_features).to(device)
    fully_shard(model.fc, mesh=mesh, placements=_flat_placements())

    plan = compute_shard_plan(
        torch.Size((rows, in_features)),
        tensor_flat_offset=0,
        rank_flat_shard_size=(rows * in_features) // world_size,
        world_size=world_size,
    )
    assert plan.is_boundary(), "BoundaryModel weight must straddle the DP boundary."

    torch.manual_seed(1234)
    baseline = BoundaryModel(rows, in_features).to(device)
    full = _gather_full_param(model.fc.weight, mesh, world_size)
    with torch.no_grad():
        baseline.fc.weight.copy_(full)

    lr, momentum, nesterov, weight_decay = 0.05, 0.9, True, 0.0
    inner = Muon(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov,
        coefficient_type="quintic",
        num_ns_steps=5,
        scale_mode="spectral",
        fp32_matmul_prec="medium",
        use_syrk=False,
    )
    sharded_opt = FsdpMuon(
        model.parameters(),
        inner_optimizer=inner,
        dp_mesh=mesh,
        use_owner_comm_stream=False,
        reconstruct_full_param=True,
    )
    base_opt = Muon(
        baseline.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov,
        coefficient_type="quintic",
        num_ns_steps=5,
        scale_mode="spectral",
        fp32_matmul_prec="medium",
    )

    x = torch.randn(4, in_features, device=device)
    for _ in range(3):
        sharded_opt.zero_grad(set_to_none=True)
        base_opt.zero_grad(set_to_none=True)
        model(x).sum().backward()
        baseline(x).sum().backward()
        sharded_opt.step()
        base_opt.step()

    full = _gather_full_param(model.fc.weight, mesh, world_size)
    torch.testing.assert_close(full, baseline.fc.weight, atol=0, rtol=0)


def test_step_non_matrix_param_matches_reference(distributed_setup):
    """A 1D (non-matrix) parameter is updated by the momentum-SGD fallback.

    `NonMatrixModel` has a single 1D `Bias1d` parameter, Flat-sharded in its own
    FSDP group so it spans all DP ranks with equal shards (a boundary
    parameter). It is classified `non_matrix` and updated by
    `_step_non_matrix` (plain momentum-SGD, no orthogonalization, no
    owner-compute P2P). This also exercises the `if not matrix_indices:
    continue` edge and the `_owner_comm_needed=False` path (no boundary
    2D params -> no `new_group`). The wrapper's momentum-SGD uses the Muon
    EMA (`mom.lerp_(grad, 1-momentum)`, matching the inner
    `OrthogonalizedOptimizer`), which differs from `torch.optim.SGD`, so the
    reference replicates `_step_non_matrix` by hand (nesterov disabled) and
    the result must match bitwise.
    """
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2 or torch.cuda.device_count() < world_size:
        pytest.skip(
            "Needs >=2 ranks and >=1 GPU per rank; "
            "a 1-GPU-multi-rank config cannot run the owner-compute P2P step."
        )
    mesh = init_device_mesh(device.type, (world_size,))

    torch.manual_seed(1234)
    model = NonMatrixModel().to(device)
    fully_shard(model.bias_mod, mesh=mesh, placements=_flat_placements())

    torch.manual_seed(1234)
    baseline = NonMatrixModel().to(device)
    full = _gather_full_1d_param(model.bias_mod.bias, mesh, world_size)
    with torch.no_grad():
        baseline.bias_mod.bias.data.copy_(full)

    lr, momentum, weight_decay = 0.05, 0.9, 0.0
    inner = Muon(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=False,
        coefficient_type="quintic",
        num_ns_steps=5,
        scale_mode="spectral",
        fp32_matmul_prec="medium",
        use_syrk=False,
    )
    sharded_opt = FsdpMuon(
        model.parameters(), inner_optimizer=inner, dp_mesh=mesh, use_owner_comm_stream=False
    )

    # Hand-coded reference replicating `_step_non_matrix` (Muon EMA, no
    # orthogonalization, no scale, weight_decay=0, nesterov=False, no pre/post hooks).
    ref_state: dict = {}

    def ref_step(m: NonMatrixModel, x: torch.Tensor) -> None:
        for p in m.parameters():
            p.grad = None
        m(x).sum().backward()
        for p in m.parameters():
            if p.grad is None:
                continue
            st = ref_state.setdefault(p, {})
            if "momentum_buffer" not in st:
                st["momentum_buffer"] = torch.zeros_like(p.data)
            mom = st["momentum_buffer"]
            grad = p.grad
            if grad.dtype != mom.dtype:
                grad = grad.to(dtype=mom.dtype)
            mom.lerp_(grad, 1 - momentum)
            update = mom  # non-nesterov
            with torch.no_grad():
                p.data.add_(update, alpha=-lr)

    x = torch.randn(4, 8, device=device)
    for _ in range(3):
        sharded_opt.zero_grad(set_to_none=True)
        model(x).sum().backward()
        sharded_opt.step()
        ref_step(baseline, x)
    full = _gather_full_1d_param(model.bias_mod.bias, mesh, world_size)
    torch.testing.assert_close(full, baseline.bias_mod.bias.data, atol=0, rtol=0)


def test_step_mixed_paths_matches_reference(distributed_setup):
    """A model with both 2D (matrix) and 1D (non-matrix) boundary parameters.

    `MixedModel` has a 2D `fc.weight` (boundary, updated by the owner-compute P2P + Newton-Schulz
    path) and a 1D `bias` (boundary, updated by the momentum-SGD fallback). Both are
    Flat-sharded in separate FSDP groups so each spans all DP ranks with equal
    shards. The 2D result matches a single-rank `Muon` (orthogonalize) bitwise;
    the 1D result matches a hand-coded Muon-EMA reference (nesterov disabled) bitwise.
    """
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2 or torch.cuda.device_count() < world_size:
        pytest.skip(
            "Needs >=2 ranks and >=1 GPU per rank; "
            "a 1-GPU-multi-rank config cannot run the owner-compute P2P step."
        )
    mesh = init_device_mesh(device.type, (world_size,))

    torch.manual_seed(1234)
    model = MixedModel().to(device)
    fully_shard(model.fc, mesh=mesh, placements=_flat_placements())
    fully_shard(model.bias_mod, mesh=mesh, placements=_flat_placements())

    torch.manual_seed(1234)
    baseline = MixedModel().to(device)
    full_w = _gather_full_param(model.fc.weight, mesh, world_size)
    with torch.no_grad():
        baseline.fc.weight.data.copy_(full_w)
    full_b = _gather_full_1d_param(model.bias_mod.bias, mesh, world_size)
    with torch.no_grad():
        baseline.bias_mod.bias.data.copy_(full_b)

    lr, momentum, weight_decay = 0.05, 0.9, 0.0
    inner = Muon(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
        coefficient_type="quintic",
        num_ns_steps=5,
        scale_mode="spectral",
        fp32_matmul_prec="medium",
        use_syrk=False,
    )
    sharded_opt = FsdpMuon(
        model.parameters(), inner_optimizer=inner, dp_mesh=mesh, use_owner_comm_stream=False
    )
    # Reference for the 2D `fc.weight`: a single-rank Muon (orthogonalize).
    base_opt = Muon(
        [baseline.fc.weight],
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
        coefficient_type="quintic",
        num_ns_steps=5,
        scale_mode="spectral",
        fp32_matmul_prec="medium",
    )

    # Hand-coded reference for the 1D `bias` (Muon EMA, no orthogonalization).
    ref_state: dict = {}

    def ref_bias_step(m: MixedModel, x: torch.Tensor) -> None:
        for p in m.parameters():
            p.grad = None
        m(x).sum().backward()
        base_opt.step()  # updates only `baseline.fc.weight` (the 2D param).
        p = baseline.bias_mod.bias
        st = ref_state.setdefault(p, {})
        if "momentum_buffer" not in st:
            st["momentum_buffer"] = torch.zeros_like(p.data)
        mom = st["momentum_buffer"]
        grad = p.grad
        if grad.dtype != mom.dtype:
            grad = grad.to(dtype=mom.dtype)
        mom.lerp_(grad, 1 - momentum)
        # Match `_step_non_matrix`'s Nesterov path (`self._inner.nesterov` is True here
        # so the fallback uses `grad.lerp(mom, momentum)`, not `mom`).
        update = grad.lerp(mom, momentum)
        with torch.no_grad():
            p.data.add_(update, alpha=-lr)

    x = torch.randn(4, 8, device=device)
    for _ in range(3):
        sharded_opt.zero_grad(set_to_none=True)
        model(x).sum().backward()
        sharded_opt.step()
        ref_bias_step(baseline, x)

    full_w = _gather_full_param(model.fc.weight, mesh, world_size)
    torch.testing.assert_close(full_w, baseline.fc.weight, atol=0, rtol=0)
    full_b = _gather_full_1d_param(model.bias_mod.bias, mesh, world_size)
    torch.testing.assert_close(full_b, baseline.bias_mod.bias.data, atol=0, rtol=0)


def test_step_losses_track_torch_muon(distributed_setup):
    """The FSDP Muon step should track torch.optim.Muon over several steps.

    The FSDP wrapper reuses the `emerging_optimizers` Newton-Schulz kernel,
    which normalizes the orthogonalization input in FP32 and then drops to BF16,
    whereas `torch.optim.Muon` casts to BF16 before normalizing. The two
    kernels therefore produce slightly different updates, so losses track
    within a tolerance that admits that kernel difference (empirically
    `max|Δloss|` ~3e-2 / `max rel` ~1e-1 over many seeds). Bitwise numerics
    against the same kernel are covered by `test_step_bitwise_matches_single_rank_reference`.
    """
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2 or torch.cuda.device_count() < world_size:
        pytest.skip(
            "Needs >=2 ranks and >=1 GPU per rank; "
            "a 1-GPU-multi-rank config cannot run the owner-compute P2P step."
        )
    mesh = init_device_mesh(device.type, (world_size,))

    sharded = _make_fsdp_model(device, mesh)
    torch.manual_seed(1234)
    baseline = TinyModel().to(device)
    for name, shard_param in (
        ("fc1.weight", sharded.fc1.weight),
        ("fc2.weight", sharded.fc2.weight),
    ):
        full = _gather_full_param(shard_param, mesh, world_size)
        with torch.no_grad():
            getattr(baseline, name.split(".")[0]).weight.copy_(full)

    inner_optimizer = Muon(
        sharded.parameters(),
        lr=0.05,
        momentum=0.9,
        weight_decay=0.0,
        nesterov=True,
        coefficient_type="simple",
        num_ns_steps=5,
        scale_mode="shape_scaling",
        fp32_matmul_prec="medium",
        use_syrk=False,
    )
    sharded_opt = FsdpMuon(
        sharded.parameters(),
        inner_optimizer=inner_optimizer,
        dp_mesh=mesh,
        use_owner_comm_stream=False,
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
        msg="FSDP Muon losses did not track torch.optim.Muon within the kernel-difference tolerance.",
    )


def _make_mixed_dtype_fsdp_model(device: torch.device, mesh, seed: int = 1234):
    """Sharded `MixedDtypeModel`: `fc1` in fp32, `fc2` in bf16, both Flat-sharded."""
    torch.manual_seed(seed)
    model = MixedDtypeModel().to(device)
    fully_shard(model.fc1, mesh=mesh, placements=_flat_placements())
    fully_shard(model.fc2, mesh=mesh, placements=_flat_placements())
    return model


def test_step_mixed_dtypes_bitwise_matches_reference(distributed_setup):
    """The FSDP Muon step must handle a param group with mixed dtypes (fp32 + bf16).

    Exercises the `_group_updates` chunking: boundary params of different dtypes
    must be processed in separate boundary chunks (one `_finish_boundary_step`
    call each) with matching P2P buffer metadata, and the result must match a
    single-rank Muon reference.
    """
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2 or torch.cuda.device_count() < world_size:
        pytest.skip(
            "Needs >=2 ranks and >=1 GPU per rank; "
            "a 1-GPU-multi-rank config cannot run the owner-compute P2P step."
        )
    mesh = init_device_mesh(device.type, (world_size,))

    sharded = _make_mixed_dtype_fsdp_model(device, mesh)
    torch.manual_seed(1234)
    baseline = MixedDtypeModel().to(device)
    # Copy full weights from the sharded model's gathered (pre-shard) initial state.
    for name, shard_param in (
        ("fc1.weight", sharded.fc1.weight),
        ("fc2.weight", sharded.fc2.weight),
    ):
        full = _gather_full_param(shard_param, mesh, world_size)
        with torch.no_grad():
            getattr(baseline, name.split(".")[0]).weight.copy_(full)

    lr = 0.05
    momentum = 0.9
    nesterov = True
    weight_decay = 0.0
    inner_optimizer = Muon(
        sharded.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov,
        coefficient_type="quintic",
        num_ns_steps=5,
        scale_mode="spectral",
        fp32_matmul_prec="medium",
        use_syrk=False,
    )
    sharded_opt = FsdpMuon(
        sharded.parameters(),
        inner_optimizer=inner_optimizer,
        dp_mesh=mesh,
        use_owner_comm_stream=False,
    )
    base_opt = Muon(
        baseline.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov,
        coefficient_type="quintic",
        num_ns_steps=5,
        scale_mode="spectral",
        fp32_matmul_prec="medium",
    )

    x = torch.randn(4, 8, device=device)
    for step in range(3):
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
        full = _gather_full_param(shard_param, mesh, world_size)
        expected = getattr(baseline, name.split(".")[0]).weight
        torch.testing.assert_close(full, expected, atol=0, rtol=0)


def _gather_full_param(param, mesh, world_size):
    """All-gather a Flat-sharded 2D DTensor parameter into its full matrix."""
    local = param.to_local().contiguous()
    full_shape = tuple(param.shape)
    row_size = full_shape[1]
    parts = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(parts, local, group=mesh.get_group())
    # Flat shards are contiguous rows in rank order; concatenate along dim 0.
    return torch.cat([p.view(-1, row_size) if p.numel() else p for p in parts], dim=0)[
        : full_shape[0]
    ]


def _gather_full_1d_param(param, mesh, world_size):
    """All-gather a Flat-sharded 1D DTensor parameter into its full vector.

    The parameter must be Flat-sharded in its own FSDP group so it spans all DP ranks
    with equal shards; `dist.all_gather` requires equal-sized parts on every
    rank and would deadlock on a fully-local 1D parameter (0-size shard on
    some ranks).
    """
    local = param.to_local().contiguous()
    parts = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(parts, local, group=mesh.get_group())
    # Flat shards are contiguous elements in rank order; concatenate along dim 0.
    return torch.cat(parts, dim=0)


def test_owner_p2p_round_trip_multi_owner(distributed_setup):
    """The owner gather + scatter P2P round-trips with multiple owners.

    Exercises the multi-owner path that a 2-rank FSDP run cannot (owners on
    different ranks, updates scattered to ranks that own neither parameter).
    Uses synthetic shards and an identity "orthogonalization" so the test
    isolates the communication (pack/send/reconstruct/pack/scatter/unpack) from
    the Newton-Schulz kernel and from FSDP's forward/backward collectives.
    """
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 4:
        pytest.skip("Needs >=4 ranks for a multi-owner P2P.")
    mesh = init_device_mesh(device.type, (world_size,))

    # Two boundary params, every rank owns a shard of each. Owners are balanced
    # to ranks 0 and 1 by assign_owner_work.
    plan0 = compute_shard_plan(torch.Size((8, 8)), 0, 16, world_size)  # 2 rows/rank
    plan1 = compute_shard_plan(torch.Size((4, 16)), 0, 16, world_size)  # 1 row/rank
    plans = [plan0, plan1]
    owners = assign_owner_work(plans, num_ns_steps=5)
    this_rank = mesh.get_local_rank()

    full_pre0 = torch.arange(64, dtype=torch.float32, device=device).view(8, 8)
    full_pre1 = torch.arange(64, dtype=torch.float32, device=device).view(4, 16) + 100.0
    full_pres = [full_pre0, full_pre1]

    local_shards = []
    for plan, full in zip(plans, full_pres):
        rs, rc = plan.rank_rows[this_rank]
        local_shards.append(full[rs : rs + rc].clone())

    optimizer = object.__new__(FsdpMuon)
    optimizer.dp_mesh = mesh
    optimizer.use_owner_comm_stream = False
    optimizer._owner_comm_stream_cache = {}

    gather_plan = pack_owner_work(
        plans, owners, local_shards, world_size, this_rank, device=device, dtype=torch.float32
    )
    recv_buffers, works = optimizer._send_to_owner(gather_plan, device, dtype=torch.float32)
    optimizer._wait_for_dist_buffer(works)

    # Identity orthogonalization: the full update equals the gathered input.
    full_updates: dict[int, torch.Tensor] = {}
    for local_index in range(len(plans)):
        if owners[local_index] != this_rank:
            continue
        plan = plans[local_index]
        if plan.rank_row_count(this_rank) == 0:
            continue
        full = reconstruct_full_tensor(
            local_index, plan, gather_plan, recv_buffers, owner_rank=this_rank
        )
        full_updates[local_index] = full

    scatter_plan = pack_update_shards(
        full_updates, plans, owners, world_size, this_rank, device=device, dtype=torch.float32
    )
    scatter_recv, scatter_works = optimizer._send_to_destination(
        scatter_plan, device, dtype=torch.float32
    )
    optimizer._wait_for_dist_buffer(scatter_works)
    received = unpack_update_shards(scatter_plan, scatter_recv)

    for local_index, plan in enumerate(plans):
        rs, rc = plan.rank_rows[this_rank]
        expected = full_pres[local_index][rs : rs + rc]
        if owners[local_index] == this_rank:
            # Owner applies its own shard directly from the full update.
            own = full_updates[local_index][rs : rs + rc]
            torch.testing.assert_close(own, expected, atol=0, rtol=0)
        else:
            torch.testing.assert_close(received[local_index], expected, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Import guard tests
# ---------------------------------------------------------------------------

_EO_MOD = "megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.orthogonalized_optimizer"


def test_import_guard_emerging_optimizers_available():
    """When emerging_optimizers is installed, HAVE_EMERGING_OPTIMIZERS is True and the real classes are bound."""
    from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
        orthogonalized_optimizer as mod,
    )

    assert mod.HAVE_EMERGING_OPTIMIZERS is True
    from emerging_optimizers.orthogonalized_optimizers import OrthogonalizedOptimizer as RealEO

    assert mod.OrthogonalizedOptimizer is RealEO

    from emerging_optimizers.orthogonalized_optimizers.muon import Muon as RealMuon

    assert mod.Muon is RealMuon


@contextlib.contextmanager
def _simulate_no_emerging_optimizers():
    """Reload `orthogonalized_optimizer` with `emerging_optimizers` blocked.

    Saves and restores `sys.modules` / `sys.meta_path` so the rest of the
    test session sees the real (installed) module.
    """
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
    """Constructing the optimizers without emerging_optimizers raises a helpful ModuleNotFoundError."""
    with _simulate_no_emerging_optimizers() as mod:
        with pytest.raises(ModuleNotFoundError, match="emerging-optimizers"):
            mod.FsdpOrthogonalizedOptimizer([], object(), dp_mesh=None)
        with pytest.raises(ModuleNotFoundError, match="emerging-optimizers"):
            mod.FsdpMuon([], object(), dp_mesh=None)
