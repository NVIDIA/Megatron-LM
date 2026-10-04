# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Fallback tests for MFSDP v2.

Use this file only as a last resort when no focused test file fits. Read the
other test files' module docstrings to choose a destination.
"""

import logging
from typing import NamedTuple

import pytest
import torch
import torch.distributed as dist
import transformer_engine.pytorch as te
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
from torch.utils.checkpoint import checkpoint

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Placements,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
    microbatch,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.placement import (
    RowAtomic,
    TensorAtomic,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.mixed_precision import MixedPrecisionPolicy
from tests.unit_tests.distributed.mfsdp_v2.profiler_utils import collect_linked_event_groups

logger = logging.getLogger(__name__)


class TinyModel(nn.Module):
    """Small model with two separately shardable modules."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the tiny model."""
        return self.fc2(self.relu(self.fc1(x)))


class CheckpointedTinyModel(TinyModel):
    """Tiny model that activation-checkpoints each shardable module."""

    def __init__(self, use_reentrant: bool) -> None:
        super().__init__()
        self.use_reentrant = use_reentrant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run each linear layer through activation checkpointing."""
        x = checkpoint(self.fc1, x, use_reentrant=self.use_reentrant)
        return checkpoint(self.fc2, self.relu(x), use_reentrant=self.use_reentrant)


class MultiChildModel(nn.Module):
    """Model with direct parameters and multiple child FsdpModules."""

    def __init__(self, dim: int, num_children: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.ones(dim))
        self.layers = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(num_children)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run through every child layer with a root-owned bias."""
        x = x + self.bias
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x


def _default_placements() -> Placements:
    return Placements(dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)])


def _no_shard_placements() -> Placements:
    return Placements(
        dp_axes=[0], parameter=[Replicate()], gradient=[Partial("avg")], optimizer=[Replicate()]
    )


def _zero1_placements() -> Placements:
    return Placements(
        dp_axes=[0], parameter=[Replicate()], gradient=[Partial("avg")], optimizer=[Shard(0)]
    )


def _zero2_placements() -> Placements:
    return Placements(
        dp_axes=[0], parameter=[Replicate()], gradient=[Shard(0)], optimizer=[Shard(0)]
    )


_TENSOR_ATOMIC_ZERO1_PLACEMENTS = Placements(
    dp_axes=[0], parameter=[Replicate()], gradient=[Partial("avg")], optimizer=[TensorAtomic()]
)

_TENSOR_ATOMIC_ZERO2_PLACEMENTS = Placements(
    dp_axes=[0], parameter=[Replicate()], gradient=[TensorAtomic()], optimizer=[TensorAtomic()]
)

_TENSOR_ATOMIC_ZERO3_PLACEMENTS = Placements(
    dp_axes=[0], parameter=[TensorAtomic()], gradient=[TensorAtomic()], optimizer=[TensorAtomic()]
)


def _hsdp_placements() -> Placements:
    """HSDP: params/optimizer replicated across DP-outer (axis 0), sharded within
    DP-inner (axis 1). main_grad rests [Partial, Shard(0)] between microbatches and is
    all-reduced to [Replicate, Shard(0)] on the last microbatch."""
    return Placements(
        dp_axes=[0, 1],
        parameter=[Replicate(), Shard(0)],
        gradient=[Partial("avg"), Shard(0)],
        optimizer=[Replicate(), Shard(0)],
    )


def _hfsdp_placements() -> Placements:
    """HFSDP: params replicated across DP-outer (axis 0) for compute but the
    optimizer sharded across it, all sharded within DP-inner (axis 1). main_grad
    rests [Partial, Shard(0)] between microbatches and is reduce-scattered to
    [Shard(0), Shard(0)] (the optimizer placement) on the last microbatch."""
    return Placements(
        dp_axes=[0, 1],
        parameter=[Replicate(), Shard(0)],
        gradient=[Partial("avg"), Shard(0)],
        optimizer=[Shard(0), Shard(0)],
    )


# CPU ops that a device event chains up to via cpu_parent, used to attribute the device
# work to its enclosing collective or matmul operation.
_REDUCE_SCATTER_OP_NAME_SUBSTRING = "reduce_scatter"
_ALLREDUCE_OP_NAME_SUBSTRING = "allreduce"


@pytest.mark.parametrize(
    "placements_factory",
    [_no_shard_placements, _zero1_placements, _zero2_placements, _default_placements],
    ids=["no_shard", "zero1", "zero2", "zero3"],
)
@pytest.mark.parametrize("num_microbatches", [1, 3])
def test_fully_shard_sgd_losses_match_baseline(
    distributed_setup, num_microbatches, placements_factory
):
    """Every supported sharding strategy should match gradient-averaged SGD."""
    rank = distributed_setup.rank
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    placements = placements_factory()
    torch.manual_seed(1234)
    baseline = TinyModel().to(device)
    model = TinyModel().to(device)
    model.load_state_dict(baseline.state_dict())

    with fully_shard_context(device=device) as context:
        fully_shard(model.fc1, mesh=mesh, placements=placements)
        fully_shard(model.fc2, mesh=mesh, placements=placements)
    baseline_optimizer = torch.optim.SGD(baseline.parameters(), lr=0.05)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    fully_shard_optimizer(optimizer)

    microbatch_size = 2
    # Keep initialization identical, but exercise reduction of distinct rank-local gradients.
    torch.manual_seed(5678 + rank)
    x = torch.randn(num_microbatches, microbatch_size, 8, device=device)
    target = torch.randn(num_microbatches, microbatch_size, 4, device=device)
    microbatches = tuple(zip(x.unbind(), target.unbind()))

    def train(model, optimizer, log_prefix, *, reduce_grads: bool) -> list[torch.Tensor]:
        losses = []
        for step in range(5):
            optimizer.zero_grad()

            for microbatch_index, (microbatch_x, microbatch_target) in enumerate(microbatches):
                with microbatch(context, is_last=microbatch_index == num_microbatches - 1):
                    loss = torch.nn.functional.mse_loss(model(microbatch_x), microbatch_target)
                    losses.append(loss.detach())
                    logger.debug(
                        "%s train parity: rank=%s, step=%s, microbatch=%s, loss=%s",
                        log_prefix,
                        rank,
                        step,
                        microbatch_index,
                        loss,
                    )
                    (loss / num_microbatches).backward()

            if reduce_grads:
                # The unsharded baseline needs the same DP average as FSDP, after accumulation.
                for parameter in model.parameters():
                    dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                    parameter.grad.div_(world_size)

            optimizer.step()
        return losses

    baseline_losses = train(baseline, baseline_optimizer, "Baseline", reduce_grads=True)
    sharded_losses = train(model, optimizer, "FSDP", reduce_grads=False)

    torch.testing.assert_close(
        torch.stack(sharded_losses),
        torch.stack(baseline_losses),
        msg="Sharded losses did not match baseline losses.",
    )


def test_fully_shard_waits_for_delayed_te_weight_gradient(distributed_setup):
    """TE's callback, not AccumulateGrad, completes MFSDP backward."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device

    mesh = init_device_mesh(device.type, (world_size,))
    model = te.Linear(
        16,
        16,
        bias=False,
        params_dtype=torch.bfloat16,
        device=device,
        delay_wgrad_compute=True,
        fuse_wgrad_accumulation=False,
    )
    with fully_shard_context(device=device):
        fully_shard(model, mesh=mesh, placements=_default_placements())

    x = torch.randn(4, 16, device=device, dtype=torch.bfloat16, requires_grad=True)
    model(x).float().square().mean().backward()
    assert model.weight.grad is None
    assert model.phase is FsdpModule.Phase.BACKWARD

    model.backward_dw()

    assert model.weight.grad is not None
    assert model.phase is FsdpModule.Phase.RESTING


def test_fully_shard_rejects_tied_delayed_weight_gradients(distributed_setup):
    """Tied delayed weights are unsupported until TE accumulates their gradients."""
    device = distributed_setup.device
    model = nn.Sequential(
        *(
            te.Linear(
                16,
                16,
                bias=False,
                params_dtype=torch.bfloat16,
                device=device,
                delay_wgrad_compute=True,
                fuse_wgrad_accumulation=False,
            )
            for _ in range(2)
        )
    )
    model[1].weight = model[0].weight

    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    with (
        fully_shard_context(device=device),
        pytest.raises(ValueError, match="Transformer Engine does not accumulate their gradients"),
    ):
        fully_shard(model, mesh=mesh, placements=_default_placements())


@pytest.mark.parametrize("use_reentrant", [False, True], ids=["non_reentrant", "reentrant"])
def test_fully_shard_activation_recompute_reshards_parameters(distributed_setup, use_reentrant):
    """Activation recomputation should leave every FSDP module resharded.

    Backward completes ``fc2`` before recomputing ``fc1``. Without suppressing
    forward prefetch during recomputation, ``fc1`` unshards ``fc2`` again after
    its backward hook has run, leaving ``fc2.weight`` as an unsharded Parameter
    instead of a sharded DTensor at the end of backward.
    """
    world_size = distributed_setup.world_size
    device = distributed_setup.device

    mesh = init_device_mesh(device.type, (world_size,))
    model = CheckpointedTinyModel(use_reentrant=use_reentrant).to(device)
    with fully_shard_context(device=device):
        fully_shard(model.fc1, mesh=mesh, placements=_default_placements())
        fully_shard(model.fc2, mesh=mesh, placements=_default_placements())
        fully_shard(model, mesh=mesh, placements=_default_placements())

    x = torch.randn(2, 8, device=device, requires_grad=True)
    model(x).sum().backward()

    # Without the forward-prefetch suppression, ``fc1``'s recomputed forward
    # would unshard ``fc2`` after ``fc2``'s backward already resharded it,
    # leaving an unsharded Parameter here.
    assert isinstance(model.fc1.weight, DTensor)
    assert isinstance(model.fc2.weight, DTensor)

    # Backward completes each module before recomputing the previous one, so
    # every module-local phase must be cleared after its matching backward.
    assert model.phase is FsdpModule.Phase.RESTING
    assert model.fc1.phase is FsdpModule.Phase.RESTING
    assert model.fc2.phase is FsdpModule.Phase.RESTING

    # A second forward after backward runs in the forward phase again, so
    # forward-order prefetch resumes and the module phases return to resting.
    model(x).sum().backward()
    assert model.phase is FsdpModule.Phase.RESTING
    assert model.fc1.phase is FsdpModule.Phase.RESTING
    assert model.fc2.phase is FsdpModule.Phase.RESTING


@pytest.mark.parametrize("set_to_none", [True, False])
@pytest.mark.parametrize("num_microbatches", [1, 3])
def test_hsdp_losses_match_baseline(distributed_setup, num_microbatches, set_to_none):
    """HSDP (DP-outer replicated, DP-inner sharded) training should match single-rank SGD.

    Gradients reduce-scatter within DP-inner every backward and accumulate into
    main_grad; the DP-outer all-reduce runs only on the last microbatch, scoped
    via ``microbatch(...)``. Every rank sees identical data, so the averaged
    gradient equals the single-rank gradient and losses must match. Both
    ``zero_grad`` modes are covered: ``set_to_none=True`` overwrites main_grad,
    ``set_to_none=False`` accumulates into a zeroed main_grad.
    """
    rank = distributed_setup.rank
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 4 or world_size % 2 != 0:
        pytest.skip("This test requires an even number of at least 4 ranks for a 2-D DP mesh.")

    outer_size = 2
    inner_size = world_size // outer_size
    mesh = init_device_mesh(
        device.type, (outer_size, inner_size), mesh_dim_names=("dp_outer", "dp_inner")
    )
    torch.manual_seed(1234)
    dim = 8
    baseline = MultiChildModel(dim=dim, num_children=2).to(device)
    model = MultiChildModel(dim=dim, num_children=2).to(device)
    model.load_state_dict(baseline.state_dict())

    with fully_shard_context(device=device) as context:
        for layer in model.layers:
            fully_shard(layer, mesh=mesh, placements=_hsdp_placements())
        fully_shard(model, mesh=mesh, placements=_hsdp_placements())
    baseline_optimizer = torch.optim.SGD(baseline.parameters(), lr=0.05)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    micro_batch_size = 2
    x = torch.randn(num_microbatches, micro_batch_size, dim, device=device)
    target = torch.randn(num_microbatches, micro_batch_size, dim, device=device)
    microbatches = tuple(zip(x.unbind(), target.unbind()))

    def train(model, optimizer, log_prefix) -> list[torch.Tensor]:
        losses = []
        for step in range(5):
            optimizer.zero_grad(set_to_none=set_to_none)

            for microbatch_index, (microbatch_x, microbatch_target) in enumerate(microbatches):
                is_last = microbatch_index == num_microbatches - 1
                with microbatch(context, is_last=is_last):
                    loss = torch.nn.functional.mse_loss(model(microbatch_x), microbatch_target)
                    (loss / num_microbatches).backward()
                losses.append(loss.detach())
                logger.debug(
                    "%s train parity: rank=%s, step=%s, microbatch=%s, loss=%s",
                    log_prefix,
                    rank,
                    step,
                    microbatch_index,
                    loss,
                )

            optimizer.step()
        return losses

    baseline_losses = train(baseline, baseline_optimizer, "Baseline")
    sharded_losses = train(model, optimizer, "HSDP")

    torch.testing.assert_close(
        torch.stack(sharded_losses),
        torch.stack(baseline_losses),
        msg="HSDP losses did not match baseline losses.",
    )


@pytest.mark.parametrize("set_to_none", [True, False])
@pytest.mark.parametrize("num_microbatches", [1, 3])
def test_hfsdp_losses_match_baseline(distributed_setup, num_microbatches, set_to_none):
    """HFSDP (optimizer sharded across DP-outer too) training should match single-rank SGD."""
    rank = distributed_setup.rank
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    # world_size=2 gives a 2x1 mesh: DP-inner is trivial but the DP-outer
    # reduce-scatter finalize and the fresh-buffer reset still run and converge.
    if world_size % 2 != 0:
        pytest.skip("This test requires an even number of ranks for a 2-D DP mesh.")

    outer_size = 2
    inner_size = world_size // outer_size
    mesh = init_device_mesh(
        device.type, (outer_size, inner_size), mesh_dim_names=("dp_outer", "dp_inner")
    )
    torch.manual_seed(1234)
    dim = 8
    baseline = MultiChildModel(dim=dim, num_children=2).to(device)
    model = MultiChildModel(dim=dim, num_children=2).to(device)
    model.load_state_dict(baseline.state_dict())

    # Shard the child layers, then the model, so the children share a root context
    # and reduce through the overlap path instead of as independent roots.
    with fully_shard_context(device=device) as context:
        for layer in model.layers:
            fully_shard(layer, mesh=mesh, placements=_hfsdp_placements())
        fully_shard(model, mesh=mesh, placements=_hfsdp_placements())
    baseline_optimizer = torch.optim.SGD(baseline.parameters(), lr=0.05)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    # HFSDP's optimizer placement [Shard(0), Shard(0)] differs from the parameter placement
    # [Replicate, Shard(0)], so main_weight and model_weight are distinct buffers and the
    # compute weight is stale until the step post-hook registered here refreshes it.
    # HSDP needs no wrapper: its two placements match, so the buffers alias.
    fully_shard_optimizer(optimizer)

    micro_batch_size = 2
    x = torch.randn(num_microbatches, micro_batch_size, dim, device=device)
    target = torch.randn(num_microbatches, micro_batch_size, dim, device=device)
    microbatches = tuple(zip(x.unbind(), target.unbind()))

    def train(model, optimizer, log_prefix) -> list[torch.Tensor]:
        losses = []
        for step in range(5):
            optimizer.zero_grad(set_to_none=set_to_none)

            for microbatch_index, (microbatch_x, microbatch_target) in enumerate(microbatches):
                is_last = microbatch_index == num_microbatches - 1
                with microbatch(context, is_last=is_last):
                    loss = torch.nn.functional.mse_loss(model(microbatch_x), microbatch_target)
                    (loss / num_microbatches).backward()
                losses.append(loss.detach())
                logger.debug(
                    "%s train parity: rank=%s, step=%s, microbatch=%s, loss=%s",
                    log_prefix,
                    rank,
                    step,
                    microbatch_index,
                    loss,
                )

            optimizer.step()
        return losses

    baseline_losses = train(baseline, baseline_optimizer, "Baseline")
    sharded_losses = train(model, optimizer, "HFSDP")

    torch.testing.assert_close(
        torch.stack(sharded_losses),
        torch.stack(baseline_losses),
        msg="HFSDP losses did not match baseline losses.",
    )


def test_hsdp_defers_dp_outer_allreduce_to_last_microbatch(distributed_setup):
    """HSDP reduce-scatters DP-inner every microbatch but all-reduces DP-outer once.

    Counting linked NCCL kernels over a multi-microbatch step, the DP-inner
    reduce-scatter fires once per microbatch per group while the DP-outer
    all-reduce that finalizes main_grad fires only on the last microbatch. This
    asserts on kernel counts only, not numerics.
    """
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 4 or world_size % 2 != 0:
        pytest.skip("This test requires an even number of at least 4 ranks for a 2-D DP mesh.")

    outer_size = 2
    inner_size = world_size // outer_size
    mesh = init_device_mesh(
        device.type, (outer_size, inner_size), mesh_dim_names=("dp_outer", "dp_inner")
    )
    torch.manual_seed(1234)
    dim = 8
    num_children = 2
    model = MultiChildModel(dim=dim, num_children=num_children).to(device)
    with fully_shard_context(device=device) as context:
        for layer in model.layers:
            fully_shard(layer, mesh=mesh, placements=_hsdp_placements())
        fully_shard(model, mesh=mesh, placements=_hsdp_placements())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    num_microbatches = 3
    micro_batch_size = 2
    x = torch.randn(num_microbatches, micro_batch_size, dim, device=device)
    target = torch.randn(num_microbatches, micro_batch_size, dim, device=device)
    microbatches = tuple(zip(x.unbind(), target.unbind()))

    def train_one_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        for microbatch_index, (microbatch_x, microbatch_target) in enumerate(microbatches):
            is_last = microbatch_index == num_microbatches - 1
            with microbatch(context, is_last=is_last):
                loss = torch.nn.functional.mse_loss(model(microbatch_x), microbatch_target)
                (loss / num_microbatches).backward()
        optimizer.step()

    train_one_step()
    torch.cuda.synchronize(device)

    with torch.profiler.profile() as prof:
        train_one_step()
        torch.cuda.synchronize(device)

    reduce_scatter_groups = collect_linked_event_groups(prof, _REDUCE_SCATTER_OP_NAME_SUBSTRING)
    allreduce_groups = collect_linked_event_groups(prof, _ALLREDUCE_OP_NAME_SUBSTRING)
    # One DP-outer all-reduce per parameter group -- each child layer plus the
    # root unit's bias -- fired only on the last microbatch. Plain DP fires none.
    assert len(allreduce_groups) == num_children + 1, [event.name for event in prof.events()]
    # DP-inner reduce-scatter runs every microbatch; the DP-outer all-reduce runs
    # only on the last, so the counts differ by exactly the microbatch factor.
    assert len(reduce_scatter_groups) == len(allreduce_groups) * num_microbatches, (
        f"Expected reduce-scatter ({len(reduce_scatter_groups)}) to be {num_microbatches}x "
        f"the DP-outer all-reduce count ({len(allreduce_groups)})."
    )


def test_hfsdp_reduce_scatters_dp_outer_on_last_microbatch(distributed_setup):
    """HFSDP finalizes with a reduce-scatter, not an all-reduce, on the last microbatch.

    Because the optimizer is sharded across DP-outer, the DP-outer finalize is a
    reduce-scatter like the per-microbatch DP-inner reduction -- so there are no
    all-reduces at all, and the reduce-scatter count is (num_microbatches + 1) per
    parameter group: one DP-inner reduce-scatter every microbatch plus one DP-outer
    reduce-scatter on the last. This asserts on kernel counts only, not numerics.
    """
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 4 or world_size % 2 != 0:
        pytest.skip("This test requires an even number of at least 4 ranks for a 2-D DP mesh.")

    outer_size = 2
    inner_size = world_size // outer_size
    mesh = init_device_mesh(
        device.type, (outer_size, inner_size), mesh_dim_names=("dp_outer", "dp_inner")
    )
    torch.manual_seed(1234)
    dim = 8
    num_children = 2
    model = MultiChildModel(dim=dim, num_children=num_children).to(device)
    with fully_shard_context(device=device) as context:
        for layer in model.layers:
            fully_shard(layer, mesh=mesh, placements=_hfsdp_placements())
        fully_shard(model, mesh=mesh, placements=_hfsdp_placements())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    num_microbatches = 3
    micro_batch_size = 2
    x = torch.randn(num_microbatches, micro_batch_size, dim, device=device)
    target = torch.randn(num_microbatches, micro_batch_size, dim, device=device)
    microbatches = tuple(zip(x.unbind(), target.unbind()))

    def train_one_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        for microbatch_index, (microbatch_x, microbatch_target) in enumerate(microbatches):
            is_last = microbatch_index == num_microbatches - 1
            with microbatch(context, is_last=is_last):
                loss = torch.nn.functional.mse_loss(model(microbatch_x), microbatch_target)
                (loss / num_microbatches).backward()
        optimizer.step()

    train_one_step()
    torch.cuda.synchronize(device)

    with torch.profiler.profile() as prof:
        train_one_step()
        torch.cuda.synchronize(device)

    reduce_scatter_groups = collect_linked_event_groups(prof, _REDUCE_SCATTER_OP_NAME_SUBSTRING)
    allreduce_groups = collect_linked_event_groups(prof, _ALLREDUCE_OP_NAME_SUBSTRING)
    # HFSDP reduce-scatters the DP-outer axis, so it never all-reduces.
    assert not allreduce_groups, [event.name for event in prof.events()]
    # Per group (each child layer plus the root bias): one DP-inner reduce-scatter
    # every microbatch plus one DP-outer reduce-scatter on the last microbatch.
    expected = (num_microbatches + 1) * (num_children + 1)
    assert len(reduce_scatter_groups) == expected, (
        f"Expected {expected} reduce-scatters ((num_microbatches + 1) x (num_children + 1)), "
        f"got {len(reduce_scatter_groups)}."
    )


def test_backward_averages_across_dp_and_accumulates_across_calls(distributed_setup):
    """Each backward averages over DP ranks; repeated backwards accumulate by summing."""
    rank = distributed_setup.rank
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    model = nn.Linear(1, world_size, bias=False).to(device)
    nn.init.constant_(model.weight, 1.0)

    with fully_shard_context(device=device) as context:
        fully_shard(model, mesh=mesh, placements=_default_placements())

    x = torch.full((1, 1), float(rank + 1), device=device)
    with microbatch(context, is_last=False):
        model(x).sum().backward()
        model(x).sum().backward()

    assert isinstance(model.weight.grad, DTensor)
    local_grad = model.weight.grad.to_local()
    expected = torch.full_like(local_grad, float(world_size + 1))
    torch.testing.assert_close(local_grad, expected, rtol=0, atol=0)


def test_rejects_optimizer_placements_larger_than_model_weight_placements(distributed_setup):
    """Optimizer placements must fit within the model-weight placements."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device

    mesh = init_device_mesh(device.type, (world_size,))
    model = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16).to(device)
    placements = Placements(
        dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Replicate()]
    )
    with pytest.raises(ValueError, match="DBuffer.view"):
        with fully_shard_context(device=device):
            fully_shard(
                model,
                mesh=mesh,
                placements=placements,
                mixed_precision_policy=MixedPrecisionPolicy(main_params_dtype=torch.float32),
            )


def test_fully_shard_adam_mixed_precision_losses_match_baseline(distributed_setup):
    """Mixed-precision FSDP Adam should track an unsharded Adam baseline."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")
    mesh = init_device_mesh(device.type, (world_size,))
    torch.manual_seed(2026)
    baseline = TinyModel().to(device=device, dtype=torch.bfloat16)
    model = TinyModel().to(device=device, dtype=torch.bfloat16)
    model.load_state_dict(baseline.state_dict())
    with fully_shard_context(device=device):
        fully_shard(model.fc1, mesh=mesh, placements=_default_placements())
        fully_shard(model.fc2, mesh=mesh, placements=_default_placements())

    baseline_optimizer = torch.optim.Adam(baseline.parameters(), lr=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    fully_shard_optimizer(optimizer)

    x = torch.randn(3, 8, device=device, dtype=torch.bfloat16)
    target = torch.randn(3, 4, device=device, dtype=torch.bfloat16)

    for _ in range(3):
        baseline_optimizer.zero_grad()
        optimizer.zero_grad()

        baseline_loss = torch.nn.functional.mse_loss(baseline(x).float(), target.float())
        loss = torch.nn.functional.mse_loss(model(x).float(), target.float())
        torch.testing.assert_close(loss, baseline_loss, rtol=0, atol=3e-3)

        baseline_loss.backward()
        loss.backward()
        baseline_optimizer.step()
        optimizer.step()


def test_fully_shard_shares_class_stream(distributed_setup):
    """Wrapping must not turn a lazy per-class stream into a per-instance stream."""

    class LinearWithStream(nn.Linear):
        _stream = None

        @classmethod
        def get_stream(cls) -> torch.cuda.Stream:
            if cls._stream is None:
                cls._stream = torch.cuda.Stream()
            return cls._stream

    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    layers = [LinearWithStream(4, 4, device=device) for _ in range(2)]

    with fully_shard_context(device=device):
        for layer in layers:
            fully_shard(layer, mesh=mesh, placements=_default_placements())

    assert layers[0].get_stream() is layers[1].get_stream()


def test_fully_shard_keeps_instance_state_separate(distributed_setup):
    """Reusing the class must not reuse the FSDP context or parameter groups."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    layers = [nn.Linear(4, 4, device=device) for _ in range(2)]
    for layer in layers:
        with fully_shard_context(device=device):
            fully_shard(layer, mesh=mesh, placements=_default_placements())

    assert type(layers[0]) is type(layers[1])
    assert layers[0].context is not layers[1].context
    assert layers[0].parameter_groups[0] is not layers[1].parameter_groups[0]
    assert layers[0].weight is not layers[1].weight


def test_fully_shard_preserves_parameter_attributes(distributed_setup):
    """Sharded parameters should retain the original model metadata."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = nn.Linear(8, 8, bias=False, device=device)
    attributes = {"use_muon": False}
    for name, value in attributes.items():
        setattr(model.weight, name, value)

    with fully_shard_context(device=device):
        fully_shard(model, mesh=mesh, placements=_default_placements())

    for name, value in attributes.items():
        assert getattr(model.weight, name) == value, name


def test_fully_shard_validates_tensor_atomic_owner_mapping(distributed_setup):
    """TensorAtomic sharding accepts a complete owner mapping and rejects incomplete ones."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))

    def shard(owner_mapping, placements=_TENSOR_ATOMIC_ZERO3_PLACEMENTS) -> nn.Linear:
        linear = nn.Linear(4, 4).to(device)
        parameter_to_owner = owner_mapping(linear) if owner_mapping is not None else None
        with fully_shard_context(device=device, parameter_to_owner=parameter_to_owner):
            fully_shard(linear, mesh=mesh, placements=placements)
        return linear

    # A complete mapping shards the module; entries for unrelated parameters are ignored.
    unrelated = nn.Parameter(torch.zeros(2, device=device))
    linear = shard(lambda linear: {linear.weight: 1, linear.bias: 0, unrelated: 0})
    (group,) = linear.parameter_groups
    assert group.main_weight.placements == (TensorAtomic(),)
    assert [tuple(p.fqns) for p in group.fsdp_parameters] == [("weight",), ("bias",)]

    mixed = Placements(
        dp_axes=[0], parameter=[TensorAtomic()], gradient=[TensorAtomic()], optimizer=[RowAtomic()]
    )
    with pytest.raises(ValueError, match="cannot be mixed"):
        shard(lambda linear: {linear.weight: 1, linear.bias: 0}, placements=mixed)
    with pytest.raises(ValueError, match="require parameter_to_owner"):
        shard(None)
    with pytest.raises(ValueError, match="missing entries.*'bias'"):
        shard(lambda linear: {linear.weight: 0})
    with pytest.raises(ValueError, match="integer within the range"):
        shard(lambda linear: {linear.weight: 0, linear.bias: world_size})


class _BufferShapes(NamedTuple):
    """Local view shapes of one parameter across a group's buffers."""

    main_weight: torch.Size
    model_weight: torch.Size
    main_grad: torch.Size


def _tensor_atomic_buffer_shapes(distributed_setup, placements):
    """Shard TinyModel with owners on ranks 0 to 3; yield per-parameter buffer shapes.

    Each item is ``(shapes, owned, full)``: ``owned`` is the full shape on the owner rank
    and a zero-row shape elsewhere, ``full`` is the unsharded shape.
    """
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 4:
        pytest.skip("This test spreads owners over 4 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    model = TinyModel().to(device)
    parameter_to_owner = {
        model.fc1.weight: 1,
        model.fc1.bias: 3,
        model.fc2.weight: 2,
        model.fc2.bias: 0,
    }
    with fully_shard_context(device=device, parameter_to_owner=parameter_to_owner):
        fully_shard(model.fc1, mesh=mesh, placements=placements)
        fully_shard(model.fc2, mesh=mesh, placements=placements)

    rank = mesh.get_local_rank(0)
    for group in (*model.fc1.parameter_groups, *model.fc2.parameter_groups):
        for index, fsdp_parameter in enumerate(group.fsdp_parameters):
            full = group.main_weight.layout.tensor_shapes[index]
            is_owner = parameter_to_owner[fsdp_parameter.unsharded] == rank
            owned = full if is_owner else torch.Size((0, *full[1:]))
            shapes = _BufferShapes(
                main_weight=group.main_weight.get_tensor_view(index).shape,
                model_weight=group.model_weight.get_tensor_view(index).shape,
                main_grad=group.main_grad.get_tensor_view(index).shape,
            )
            yield shapes, owned, full


def test_fully_shard_tensor_atomic_zero3_places_each_parameter_on_its_owner(distributed_setup):
    """ZeRO-3: every weight and gradient buffer holds a parameter only on its owner rank."""
    for shapes, owned, _ in _tensor_atomic_buffer_shapes(
        distributed_setup, _TENSOR_ATOMIC_ZERO3_PLACEMENTS
    ):
        assert shapes.main_weight == owned
        assert shapes.model_weight == owned
        assert shapes.main_grad == owned


def test_fully_shard_tensor_atomic_zero2_places_each_parameter_on_its_owner(distributed_setup):
    """ZeRO-2: gradient and optimizer buffers live only on the owner rank; weights stay full."""
    for shapes, owned, full in _tensor_atomic_buffer_shapes(
        distributed_setup, _TENSOR_ATOMIC_ZERO2_PLACEMENTS
    ):
        assert shapes.main_weight == owned
        assert shapes.main_grad == owned
        assert shapes.model_weight == full


def test_fully_shard_tensor_atomic_zero1_places_each_parameter_on_its_owner(distributed_setup):
    """ZeRO-1: optimizer buffers live only on the owner rank; compute buffers stay full."""
    for shapes, owned, full in _tensor_atomic_buffer_shapes(
        distributed_setup, _TENSOR_ATOMIC_ZERO1_PLACEMENTS
    ):
        assert shapes.main_weight == owned
        assert shapes.model_weight == full
        assert shapes.main_grad == full


@pytest.mark.parametrize("mixed_placements", [False, True], ids=["all_atomic", "mixed"])
@pytest.mark.parametrize(
    "placements",
    [
        _TENSOR_ATOMIC_ZERO1_PLACEMENTS,
        _TENSOR_ATOMIC_ZERO2_PLACEMENTS,
        _TENSOR_ATOMIC_ZERO3_PLACEMENTS,
    ],
    ids=["tensor_atomic_zero1", "tensor_atomic_zero2", "tensor_atomic_zero3"],
)
def test_fully_shard_tensor_atomic_losses_match_baseline(
    distributed_setup, placements, mixed_placements
):
    """TensorAtomic sharding driven through fully_shard should match single-rank SGD."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    torch.manual_seed(1234)
    baseline = TinyModel().to(device)
    model = TinyModel().to(device)
    model.load_state_dict(baseline.state_dict())
    # Every TensorAtomic group mixes ranks 0 and 1; ordinary groups need no owner entries
    # even when their context has a mapping.
    parameter_to_owner = {model.fc1.weight: 1, model.fc1.bias: 0}
    if not mixed_placements:
        parameter_to_owner |= {model.fc2.weight: 0, model.fc2.bias: 1}

    with fully_shard_context(device=device, parameter_to_owner=parameter_to_owner) as context:
        fully_shard(model.fc1, mesh=mesh, placements=placements)
        fully_shard(
            model.fc2,
            mesh=mesh,
            placements=_default_placements() if mixed_placements else placements,
        )
    baseline_optimizer = torch.optim.SGD(baseline.parameters(), lr=0.05)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    fully_shard_optimizer(optimizer)

    num_microbatches = 2
    x = torch.randn(num_microbatches, 2, 8, device=device)
    target = torch.randn(num_microbatches, 2, 4, device=device)
    microbatches = tuple(zip(x.unbind(), target.unbind()))

    def train(model, optimizer) -> list[torch.Tensor]:
        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            for microbatch_index, (microbatch_x, microbatch_target) in enumerate(microbatches):
                with microbatch(context, is_last=microbatch_index == num_microbatches - 1):
                    loss = torch.nn.functional.mse_loss(model(microbatch_x), microbatch_target)
                    losses.append(loss.detach())
                    (loss / num_microbatches).backward()
            optimizer.step()
        return losses

    baseline_losses = train(baseline, baseline_optimizer)
    sharded_losses = train(model, optimizer)
    torch.testing.assert_close(
        torch.stack(sharded_losses),
        torch.stack(baseline_losses),
        msg="TensorAtomic sharded losses did not match baseline losses.",
    )
