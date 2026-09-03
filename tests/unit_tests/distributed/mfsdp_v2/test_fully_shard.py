# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the minimal Megatron-FSDP path."""

import logging

import pytest
import torch
import torch.distributed as dist
import transformer_engine.pytorch as te
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
from torch.profiler import ProfilerActivity, profile
from torch.utils.checkpoint import checkpoint

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Placements,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
    microbatch,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
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


class NestedModel(nn.Module):
    """Model with direct and child-owned parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.ones(4))
        self.inner = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the nested model."""
        return self.inner(x) + self.bias


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


class TiedLM(nn.Module):
    """Tiny language model with shared input and output embedding weights."""

    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4, dtype=torch.bfloat16)
        self.lm_head = nn.Linear(4, 8, bias=False, dtype=torch.bfloat16)
        self.lm_head.weight = self.embed_tokens.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Compute a scalar loss using both aliases of the shared weight."""
        return self.lm_head(self.embed_tokens(token_ids)).float().sum()


class SaveNonLeafWeightView(torch.autograd.Function):
    """Autograd function that saves a non-leaf parameter view for backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight_view: torch.Tensor) -> torch.Tensor:
        """Save the non-leaf weight view and run a simple elementwise op."""
        ctx.save_for_backward(x, weight_view)
        return x * weight_view

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Use the saved non-leaf weight view during backward."""
        x, weight_view = ctx.saved_tensors
        return grad_output * weight_view, grad_output * x


class NonLeafViewModel(nn.Module):
    """Model that saves a non-leaf parameter view across forward and backward."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run using a non-leaf view of the parameter."""
        weight_view = self.weight.view_as(self.weight)
        assert self.weight.is_leaf
        assert not weight_view.is_leaf
        return SaveNonLeafWeightView.apply(x, weight_view)


def _flat_placements() -> Placements:
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
    [_no_shard_placements, _zero1_placements, _zero2_placements, _flat_placements],
    ids=["no_shard", "zero1", "zero2", "zero3"],
)
@pytest.mark.parametrize("num_microbatches", [1, 3])
def test_fully_shard_sgd_losses_match_baseline(
    distributed_setup, num_microbatches, placements_factory
):
    """Every supported sharding strategy should match single-rank SGD."""
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

    micro_batch_size = 2
    x = torch.randn(num_microbatches, micro_batch_size, 8, device=device)
    target = torch.randn(num_microbatches, micro_batch_size, 4, device=device)
    microbatches = tuple(zip(x.unbind(), target.unbind()))

    def train(model, optimizer, log_prefix) -> list[torch.Tensor]:
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

            optimizer.step()
        return losses

    baseline_losses = train(baseline, baseline_optimizer, "Baseline")
    sharded_losses = train(model, optimizer, "FSDP")

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
        fully_shard(model, mesh=mesh, placements=_flat_placements())

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
        fully_shard(model, mesh=mesh, placements=_flat_placements())


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
        fully_shard(model.fc1, mesh=mesh, placements=_flat_placements())
        fully_shard(model.fc2, mesh=mesh, placements=_flat_placements())
        fully_shard(model, mesh=mesh, placements=_flat_placements())

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


@pytest.mark.parametrize(
    "placements_factory", [_hsdp_placements, _hfsdp_placements], ids=["hsdp", "hfsdp"]
)
@pytest.mark.parametrize("overlap", [False, True], ids=["serial", "overlap"])
@pytest.mark.parametrize("set_to_none", [True, False])
@pytest.mark.parametrize("num_microbatches", [1, 3])
def test_losses_match_baseline(
    distributed_setup, num_microbatches, set_to_none, overlap, placements_factory
):
    """HSDP/HFSDP training should match single-rank SGD, with and without DP-outer overlap.

    Gradients reduce-scatter within DP-inner every backward and accumulate into main_grad;
    the DP-outer reduction (all-reduce for HSDP, reduce-scatter for HFSDP) runs only on the
    last microbatch. overlap_dp_outer_communication moves that reduction to its own stream,
    which changes where it executes, not what it computes -- so losses stay bit-identical to
    the serialized path. Both zero_grad modes are covered.
    """
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size % 2 != 0:
        pytest.skip("This test requires an even number of ranks for a 2-D DP mesh.")

    outer_size = 2
    inner_size = world_size // outer_size
    mesh = init_device_mesh(
        device.type, (outer_size, inner_size), mesh_dim_names=("dp_outer", "dp_inner")
    )
    placements = placements_factory()
    torch.manual_seed(1234)
    dim = 8
    baseline = MultiChildModel(dim=dim, num_children=2).to(device)
    model = MultiChildModel(dim=dim, num_children=2).to(device)
    model.load_state_dict(baseline.state_dict())

    with fully_shard_context(device=device, overlap_dp_outer_communication=overlap) as context:
        for layer in model.layers:
            fully_shard(layer, mesh=mesh, placements=placements)
        fully_shard(model, mesh=mesh, placements=placements)
    baseline_optimizer = torch.optim.SGD(baseline.parameters(), lr=0.05)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    # A no-op for HSDP (aliased weights) and required for HFSDP, so wrap unconditionally
    # to refresh the compute weight after each optimizer step.
    fully_shard_optimizer(optimizer)

    micro_batch_size = 2
    x = torch.randn(num_microbatches, micro_batch_size, dim, device=device)
    target = torch.randn(num_microbatches, micro_batch_size, dim, device=device)
    microbatches = tuple(zip(x.unbind(), target.unbind()))

    def train(model, optimizer) -> list[torch.Tensor]:
        losses = []
        for _ in range(5):
            optimizer.zero_grad(set_to_none=set_to_none)
            for microbatch_index, (microbatch_x, microbatch_target) in enumerate(microbatches):
                is_last = microbatch_index == num_microbatches - 1
                with microbatch(context, is_last=is_last):
                    loss = torch.nn.functional.mse_loss(model(microbatch_x), microbatch_target)
                    (loss / num_microbatches).backward()
                losses.append(loss.detach())
            optimizer.step()
        return losses

    baseline_losses = train(baseline, baseline_optimizer)
    sharded_losses = train(model, optimizer)

    torch.testing.assert_close(
        torch.stack(sharded_losses),
        torch.stack(baseline_losses),
        msg="Losses did not match baseline losses.",
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

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
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


def test_nested_fully_shard_excludes_child_owned_parameters(distributed_setup):
    """An outer FsdpModule owns direct parameters but not nested child FsdpModule parameters."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    model = NestedModel().to(device)

    with fully_shard_context(device=device):
        fully_shard(model.inner, mesh=mesh, placements=_flat_placements())
        fully_shard(model, mesh=mesh, placements=_flat_placements())

    (inner_group,) = model.inner.parameter_groups
    (outer_group,) = model.parameter_groups

    assert [parameter.fqns for parameter in inner_group.fsdp_parameters] == [("weight",)]
    assert [parameter.fqns for parameter in outer_group.fsdp_parameters] == [("bias",)]


def test_tied_child_parameters_allocate_one_physical_weight(distributed_setup):
    """Tied registrations should allocate one DBuffer entry and optimizer parameter."""
    model = TiedLM()
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    with fully_shard_context(device=distributed_setup.device):
        fully_shard(model, mesh=mesh, placements=_flat_placements())

    (parameter_group,) = model.parameter_groups
    (parameter,) = parameter_group.fsdp_parameters
    assert parameter.fqns == ("embed_tokens.weight", "lm_head.weight")
    assert parameter_group.main_weight.layout.size == 8 * 4
    # Both aliases must expose the same optimizer-visible sharded parameter.
    assert len(list(model.parameters())) == 1


def test_parameterless_parent_with_child_modules_trains(distributed_setup):
    """A parent with no unowned parameters should still root trainable child FsdpModules."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device

    mesh = init_device_mesh(device.type, (world_size,))
    torch.manual_seed(5678)
    model = nn.Sequential(nn.Linear(4, 4, bias=False), nn.Linear(4, 2, bias=False)).to(device)

    with fully_shard_context(device=device):
        fully_shard(model[0], mesh=mesh, placements=_flat_placements())
        fully_shard(model[1], mesh=mesh, placements=_flat_placements())
        fully_shard(model, mesh=mesh, placements=_flat_placements())

    assert model.parameter_groups == ()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    x = torch.randn(3, 4, device=device)

    optimizer.zero_grad(set_to_none=True)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()


def test_frozen_parameter_group_does_not_allocate_main_grad(distributed_setup):
    """A non-trainable parameter group should not allocate persistent main gradients."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    model = nn.Linear(4, 4, bias=False).to(device)
    model.weight.requires_grad_(False)

    with fully_shard_context(device=device):
        fully_shard(model, mesh=mesh, placements=_flat_placements())

    (group,) = model.parameter_groups
    assert not group.requires_grad
    assert group.main_grad is None


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
        fully_shard(model, mesh=mesh, placements=_flat_placements())

    x = torch.full((1, 1), float(rank + 1), device=device)
    with microbatch(context, is_last=False):
        model(x).sum().backward()
        model(x).sum().backward()

    assert isinstance(model.weight.grad, DTensor)
    local_grad = model.weight.grad.to_local()
    expected = torch.full_like(local_grad, float(world_size + 1))
    torch.testing.assert_close(local_grad, expected, rtol=0, atol=0)


def test_next_forward_uses_optimizer_updated_weights(distributed_setup):
    """The next forward should observe weights updated by the previous optimizer step."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    model = nn.Linear(1, world_size, bias=False, dtype=torch.bfloat16).to(device)
    nn.init.constant_(model.weight, 1.0)

    with fully_shard_context(device=device):
        fully_shard(
            model,
            mesh=mesh,
            placements=_flat_placements(),
            mixed_precision_policy=MixedPrecisionPolicy(main_params_dtype=torch.float32),
        )
    # SGD's foreach/fused CUDA paths require matching parameter and gradient dtypes.
    # Use the scalar path to exercise FP32 main weights with default BF16 main grads.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.25, foreach=False)
    fully_shard_optimizer(optimizer)
    x = torch.ones(1, 1, device=device, dtype=torch.bfloat16)

    def train_iteration() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        return loss.detach().float()

    first_loss = train_iteration()
    second_loss = train_iteration()

    with pytest.raises(AssertionError):
        torch.testing.assert_close(second_loss, first_loss)


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


def test_optimizer_post_step_syncs_once_per_parameter_group(distributed_setup, monkeypatch):
    """Optimizer synchronization should run once per group, not once per microbatch."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    model = TinyModel().to(device=device, dtype=torch.bfloat16)
    with fully_shard_context(device=device):
        fully_shard(model.fc1, mesh=mesh, placements=_flat_placements())
        fully_shard(model.fc2, mesh=mesh, placements=_flat_placements())
    parameter_groups = (*model.fc1.parameter_groups, *model.fc2.parameter_groups)
    sync_counts = {parameter_group: 0 for parameter_group in parameter_groups}

    def make_count_sync(parameter_group):
        sync_model_weight = parameter_group.sync_model_weight_from_main_weight

        def count_sync():
            sync_counts[parameter_group] += 1
            sync_model_weight()

        return count_sync

    for parameter_group in parameter_groups:
        monkeypatch.setattr(
            parameter_group, "sync_model_weight_from_main_weight", make_count_sync(parameter_group)
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    fully_shard_optimizer(optimizer)
    inputs = torch.randn(3, 2, 8, device=device, dtype=torch.bfloat16)

    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        for microbatch_input in inputs:
            (model(microbatch_input).sum() / len(inputs)).backward()

        assert all(sync_count == step for sync_count in sync_counts.values())
        optimizer.step()
        assert all(sync_count == step + 1 for sync_count in sync_counts.values())


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
        fully_shard(model.fc1, mesh=mesh, placements=_flat_placements())
        fully_shard(model.fc2, mesh=mesh, placements=_flat_placements())

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


def test_microbatch_scopes_context(distributed_setup):
    """microbatch() should scope state on the supplied FSDP context."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device

    mesh = init_device_mesh(device.type, (world_size,))
    model = nn.Sequential(nn.Linear(1, 1, bias=False), nn.Linear(1, 1, bias=False)).to(device)
    with fully_shard_context(device=device) as context:
        for layer in model:
            fully_shard(layer, mesh=mesh, placements=_flat_placements())

    with microbatch(context, is_last=False):
        assert not context.is_last_microbatch

    assert context.is_last_microbatch


def test_cpu_initialized_parameters_shard_to_mesh_device(distributed_setup):
    """A CPU model should support sharding a child before moving the full model to CUDA."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device

    mesh = init_device_mesh(device.type, (world_size,))
    model = nn.Sequential(nn.Linear(4, 4, bias=False), nn.Linear(4, 4, bias=False))
    nn.init.constant_(model[0].weight, 2.0)
    nn.init.constant_(model[1].weight, 3.0)
    x = torch.ones(1, 4)
    expected_output = model(x).to(device)

    # Shard the second layer's parameters onto the mesh device; the unwrapped
    # first layer's parameters remain on CPU until model.to(device) below.
    with fully_shard_context(device=device):
        fully_shard(model[1], mesh=mesh, placements=_flat_placements())

    assert model[0].weight.device.type == "cpu"
    assert isinstance(model[1].weight, DTensor)
    assert model[1].weight.device == device

    model.to(device)

    output = model(x.to(device))
    torch.testing.assert_close(output, expected_output)


def test_meta_parameters_shard_to_mesh_device(distributed_setup):
    """A sharded meta model should support initialization and forward."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device

    mesh = init_device_mesh(device.type, (world_size,))
    model = nn.Sequential(
        nn.Linear(4, 4, bias=False, device="meta", dtype=torch.bfloat16),
        nn.Linear(4, 4, bias=False, device="meta", dtype=torch.bfloat16),
    )

    with fully_shard_context(device=device):
        fully_shard(model, mesh=mesh, placements=_flat_placements())

    nn.init.constant_(model[0].weight, 2.0)
    nn.init.constant_(model[1].weight, 3.0)
    # The exposed parameters update FP32 main weights, while forward uses separate BF16
    # model weights. This simulates load_checkpoint() until
    # https://github.com/NVIDIA/Megatron-LM/pull/6024 lands and syncs after loading.
    for parameter_group in model.parameter_groups:
        parameter_group.sync_model_weight_from_main_weight()

    output = model(torch.ones(1, 4, device=device, dtype=torch.bfloat16))
    torch.testing.assert_close(output, torch.full_like(output, 96.0))


def test_non_leaf_parameter_view_survives_storage_resize(distributed_setup):
    """A non-leaf parameter view saved for backward should survive full-storage resize."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    model = NonLeafViewModel().to(device)
    with fully_shard_context(device=device):
        fully_shard(model, mesh=mesh, placements=_flat_placements())

    group = model.parameter_groups[0]
    x = torch.randn(8, device=device, requires_grad=True)
    loss = model(x).sum()

    assert group._unsharded_model_weight is not None
    assert group._unsharded_model_weight.local_buffer.untyped_storage().nbytes() == 0

    loss.backward()

    assert group.main_grad is not None
    assert group._unsharded_model_weight is not None
    assert group._unsharded_model_weight.local_buffer.untyped_storage().nbytes() == 0
