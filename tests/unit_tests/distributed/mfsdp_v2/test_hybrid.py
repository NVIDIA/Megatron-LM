# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for Megatron-FSDP hybrid sharding."""

import logging

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.profiler import ProfilerActivity, profile

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Flat,
    Partial,
    Placements,
    Replicate,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
    microbatch,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.mixed_precision import MixedPrecisionPolicy
from tests.unit_tests.distributed.mfsdp_v2.profiler_utils import collect_linked_kernels

logger = logging.getLogger(__name__)

# CPU ops that a device event chains up to via cpu_parent, used to attribute the device
# work to its enclosing collective or matmul operation.
_REDUCE_SCATTER_OP_NAME_SUBSTRING = "reduce_scatter"
_ALLREDUCE_OP_NAME_SUBSTRING = "allreduce"


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


def _hsdp_placements() -> Placements:
    """HSDP: params/optimizer replicated across DP-outer (axis 0), sharded within
    DP-inner (axis 1). main_grad rests [Partial, Flat] between microbatches and is
    all-reduced to [Replicate, Flat] on the last microbatch."""
    return Placements(
        dp_axes=[0, 1],
        parameter=[Replicate(), Flat()],
        gradient=[Partial(dist.ReduceOp.AVG), Flat()],
        optimizer=[Replicate(), Flat()],
    )


def _hfsdp_placements() -> Placements:
    """HFSDP: params replicated across DP-outer (axis 0) for compute but the
    optimizer sharded across it, all sharded within DP-inner (axis 1). main_grad
    rests [Partial, Flat] between microbatches and is reduce-scattered to
    [Flat, Flat] (the optimizer placement) on the last microbatch."""
    return Placements(
        dp_axes=[0, 1],
        parameter=[Replicate(), Flat()],
        gradient=[Partial(dist.ReduceOp.AVG), Flat()],
        optimizer=[Flat(), Flat()],
    )


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
    """HFSDP (optimizer sharded across DP-outer too) training should match single-rank SGD.

    Like HSDP, gradients reduce-scatter within DP-inner every backward and
    accumulate into main_grad. Unlike HSDP, the last-microbatch DP-outer reduction
    is a reduce-scatter (not an all-reduce) that finalizes main_grad to the
    optimizer's [Flat, Flat] placement, shrinking the buffer; the next step's reset
    therefore allocates a fresh [Partial, Flat] accumulation buffer. Every rank
    sees identical data, so the averaged gradient equals the single-rank gradient
    and losses must match. Both ``zero_grad`` modes are covered.
    """
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
    # HFSDP's optimizer placement [Flat, Flat] differs from the parameter placement
    # [Replicate, Flat], so main_weight and model_weight are distinct buffers and the
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

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        train_one_step()
        torch.cuda.synchronize(device)

    reduce_scatter_kernels = collect_linked_kernels(prof, _REDUCE_SCATTER_OP_NAME_SUBSTRING)
    allreduce_kernels = collect_linked_kernels(prof, _ALLREDUCE_OP_NAME_SUBSTRING)
    # One DP-outer all-reduce per parameter group -- each child layer plus the
    # root unit's bias -- fired only on the last microbatch. Plain DP fires none.
    assert len(allreduce_kernels) == num_children + 1, [event.name for event in prof.events()]
    # DP-inner reduce-scatter runs every microbatch; the DP-outer all-reduce runs
    # only on the last, so the counts differ by exactly the microbatch factor.
    assert len(reduce_scatter_kernels) == len(allreduce_kernels) * num_microbatches, (
        f"Expected reduce-scatter ({len(reduce_scatter_kernels)}) to be {num_microbatches}x "
        f"the DP-outer all-reduce count ({len(allreduce_kernels)})."
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

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        train_one_step()
        torch.cuda.synchronize(device)

    reduce_scatter_kernels = collect_linked_kernels(prof, _REDUCE_SCATTER_OP_NAME_SUBSTRING)
    allreduce_kernels = collect_linked_kernels(prof, _ALLREDUCE_OP_NAME_SUBSTRING)
    # HFSDP reduce-scatters the DP-outer axis, so it never all-reduces.
    assert not allreduce_kernels, [event.name for event in prof.events()]
    # Per group (each child layer plus the root bias): one DP-inner reduce-scatter
    # every microbatch plus one DP-outer reduce-scatter on the last microbatch.
    expected = (num_microbatches + 1) * (num_children + 1)
    assert len(reduce_scatter_kernels) == expected, (
        f"Expected {expected} reduce-scatters ((num_microbatches + 1) x (num_children + 1)), "
        f"got {len(reduce_scatter_kernels)}."
    )
