# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the minimal Megatron-FSDP path."""

import logging

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.profiler import ProfilerActivity, profile
from torch.utils.checkpoint import checkpoint

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
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.distributed.fsdp.src.megatron_fsdp.mixed_precision import MixedPrecisionPolicy
from tests.unit_tests.distributed.mfsdp_v2.profiler_utils import collect_linked_kernels

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


class ElementwiseModel(nn.Module):
    """Small activation path over a large FSDP-managed weight."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the first weight row to an activation tensor."""
        return torch.relu(x + self.weight[0])


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
    return Placements(dp_axes=[0], parameter=[Flat()], gradient=[Flat()], optimizer=[Flat()])


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


def _mb(num_bytes: int) -> str:
    return f"{num_bytes / 1024**2:.2f} MB"


# CPU ops that a device event chains up to via cpu_parent, used to attribute the device
# work to its enclosing collective or matmul operation.
_REDUCE_SCATTER_OP_NAME_SUBSTRING = "reduce_scatter"
_ALLREDUCE_OP_NAME_SUBSTRING = "allreduce"


@pytest.mark.parametrize("num_microbatches", [1, 3])
def test_fully_shard_sgd_losses_match_baseline(distributed_setup, num_microbatches):
    """Minimal per-module FSDP training should match single-rank SGD."""
    rank = distributed_setup.rank
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    torch.manual_seed(1234)
    baseline = TinyModel().to(device)
    model = TinyModel().to(device)
    model.load_state_dict(baseline.state_dict())

    with fully_shard_context(device=device):
        fully_shard(model.fc1, mesh=mesh, placements=_flat_placements())
        fully_shard(model.fc2, mesh=mesh, placements=_flat_placements())
    baseline_optimizer = torch.optim.SGD(baseline.parameters(), lr=0.05)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    micro_batch_size = 2
    x = torch.randn(num_microbatches, micro_batch_size, 8, device=device)
    target = torch.randn(num_microbatches, micro_batch_size, 4, device=device)
    microbatches = tuple(zip(x.unbind(), target.unbind()))

    def train(model, optimizer, log_prefix) -> list[torch.Tensor]:
        losses = []
        for step in range(5):
            optimizer.zero_grad()

            for microbatch, (microbatch_x, microbatch_target) in enumerate(microbatches):
                loss = torch.nn.functional.mse_loss(model(microbatch_x), microbatch_target)
                losses.append(loss.detach())
                logger.debug(
                    "%s train parity: rank=%s, step=%s, microbatch=%s, loss=%s",
                    log_prefix,
                    rank,
                    step,
                    microbatch,
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


@pytest.mark.parametrize("main_params_dtype", [torch.bfloat16, torch.float32])
def test_persistent_sharded_storage(distributed_setup, main_params_dtype):
    """FSDP should retain only its sharded weights and gradients at rest."""
    rank = distributed_setup.rank
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    dim = 4096
    dtype = torch.bfloat16
    model = MultiChildModel(dim=dim, num_children=8).to(dtype=dtype)
    placements = _flat_placements()
    policy = MixedPrecisionPolicy(main_params_dtype=main_params_dtype)
    allocated_before = torch.cuda.memory_allocated(device)
    with fully_shard_context(device=device):
        for layer in model.layers:
            fully_shard(layer, mesh=mesh, placements=placements, mixed_precision_policy=policy)
        fully_shard(model, mesh=mesh, placements=placements, mixed_precision_policy=policy)

    child_weight_nbytes = dim * dim * torch.empty((), dtype=dtype).element_size()
    persistent_allocated = torch.cuda.memory_allocated(device) - allocated_before
    if main_params_dtype == dtype:
        # Model and main weights alias, leaving only one BF16 weight buffer and one
        # BF16 main-gradient buffer per child.
        assert all(
            group.model_weight is group.main_weight
            for layer in model.layers
            for group in layer.parameter_groups
        )
        expected_per_child_nbytes = 2 * child_weight_nbytes
    else:
        # FP32 main weights require a distinct buffer in addition to the BF16 model
        # weight and BF16 main-gradient buffers.
        main_weight_nbytes = dim * dim * torch.empty((), dtype=main_params_dtype).element_size()
        expected_per_child_nbytes = 2 * child_weight_nbytes + main_weight_nbytes

    # All persistent buffers are sharded over the data-parallel group. Small bookkeeping
    # allocations stay below 1 MiB.
    expected_persistent_nbytes = len(model.layers) * expected_per_child_nbytes // world_size
    assert (
        expected_persistent_nbytes <= persistent_allocated < expected_persistent_nbytes + 1024**2
    ), (
        "FSDP persistent memory does not match its sharded weight and gradient storage: "
        f"rank={rank}, persistent_allocated={_mb(persistent_allocated)}, "
        f"expected={_mb(expected_persistent_nbytes)}"
    )


def test_training_step_peak_memory_bounds_full_size_buffers(distributed_setup):
    """A training step should stay below five full-size child buffers."""
    rank = distributed_setup.rank
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    dim = 4096
    dtype = torch.bfloat16
    model = MultiChildModel(dim=dim, num_children=8).to(dtype=dtype)
    placements = _flat_placements()
    policy = MixedPrecisionPolicy(main_params_dtype=dtype, main_grads_dtype=dtype)
    with fully_shard_context(device=device):
        for layer in model.layers:
            fully_shard(layer, mesh=mesh, placements=placements, mixed_precision_policy=policy)
        fully_shard(model, mesh=mesh, placements=placements, mixed_precision_policy=policy)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    fully_shard_optimizer(optimizer)
    x = torch.randn(2, dim, device=device, dtype=dtype)

    def train_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        model(x).float().sum().backward()
        optimizer.step()

    child_weight_nbytes = dim * dim * torch.empty((), dtype=dtype).element_size()
    resting_allocated = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    train_step()
    peak_delta = torch.cuda.max_memory_allocated(device) - resting_allocated

    # Backward keeps the current child and one prefetched child unsharded. The current
    # child also has a full wgrad until it is copied into a full reduce-scatter input,
    # for a four-full-child-buffer peak. Allow one additional buffer for cuBLAS
    # workspace, allocator granularity, and small temporaries.
    bound_nbytes = (4 + 1) * child_weight_nbytes

    assert peak_delta < bound_nbytes, (
        "FSDP training-step peak memory exceeded the full-size-buffer bound: "
        f"rank={rank}, peak_delta={_mb(peak_delta)}, "
        f"five_full_child_buffers={_mb(bound_nbytes)}"
    )


def test_deleted_model_releases_fsdp_storage(distributed_setup):
    """Deleting an FSDP model should release its persistent storage."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device

    mesh = init_device_mesh(device.type, (world_size,))
    # Earlier tests may retain process-global CUDA allocations such as the
    # CuBLAS workspace. Capture them before creating this model, so the test
    # only detects storage retained by the deleted FSDP model itself.
    allocated_before = torch.cuda.memory_allocated(device)
    model = ElementwiseModel(dim=8192).to(dtype=torch.bfloat16, device=device)
    with fully_shard_context(device=device):
        fully_shard(model, mesh=mesh, placements=_flat_placements())

    x = torch.ones(1, 8192, dtype=torch.bfloat16, device=device)
    output = model(x)
    del output, x, model

    assert torch.cuda.memory_allocated(device) - allocated_before < 1024**2


def test_fully_shard_returns_to_resting_memory(distributed_setup):
    """Fully-sharded temporary storage should be released after forward and backward."""
    rank = distributed_setup.rank
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    dim = 4096
    dtype = torch.bfloat16
    model = MultiChildModel(dim=dim, num_children=2).to(dtype=dtype, device=device)
    placements = _flat_placements()
    policy = MixedPrecisionPolicy(main_params_dtype=dtype, main_grads_dtype=dtype)
    with fully_shard_context(device=device):
        for layer in model.layers:
            fully_shard(layer, mesh=mesh, placements=placements, mixed_precision_policy=policy)
        fully_shard(model, mesh=mesh, placements=placements, mixed_precision_policy=policy)

    x = torch.randn(2, dim, device=device, dtype=dtype)

    def clear_cublas_workspaces_and_get_allocated_memory() -> int:
        # PyTorch retains a cuBLAS workspace for each handle/stream pair. Clear those
        # library caches so this measurement isolates FSDP-managed storage.
        torch._C._cuda_clearCublasWorkspaces()
        return torch.cuda.memory_allocated(device)

    resting_allocated = clear_cublas_workspaces_and_get_allocated_memory()

    def assert_returns_to_resting_memory(phase: str) -> None:
        extra_allocated = clear_cublas_workspaces_and_get_allocated_memory() - resting_allocated
        # The live output, activations, and root-owned bias gradient are small; unsharded
        # parameter storage must be released.
        assert extra_allocated < 1024**2, (
            f"Fully-sharded storage did not return to resting memory after {phase}: "
            f"rank={rank}, extra_allocated={_mb(extra_allocated)}, "
            "max_extra_allocated=1.00 MB"
        )

    output = model(x)
    assert_returns_to_resting_memory("forward")

    loss = output.float().square().mean()
    loss.backward()
    del loss, output
    assert_returns_to_resting_memory("backward")


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

    with fully_shard_context(device=device):
        fully_shard(model, mesh=mesh, placements=_flat_placements())

    x = torch.full((1, 1), float(rank + 1), device=device)
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


def test_fully_shard_reduces_peak_training_memory(distributed_setup):
    """Per-layer FSDP should reduce peak CUDA memory during training."""
    rank = distributed_setup.rank
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")
    mesh = init_device_mesh(device.type, (world_size,))
    dim = 1024
    layers = 16
    batch = 8
    steps = 2
    dtype = torch.bfloat16

    def train_steps(model: nn.Module, optimizer: torch.optim.Optimizer, x: torch.Tensor) -> None:
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            model(x).sum().backward()
            optimizer.step()

    torch.manual_seed(4321)
    baseline = nn.Sequential(*[nn.Linear(dim, dim, dtype=dtype) for _ in range(layers)]).to(device)
    baseline_optimizer = torch.optim.AdamW(baseline.parameters(), lr=0.01)
    x = torch.randn(batch, dim, device=device, dtype=dtype)
    torch.cuda.reset_peak_memory_stats(device)
    train_steps(baseline, baseline_optimizer, x)
    baseline_peak = torch.cuda.max_memory_allocated(device)

    del baseline_optimizer
    del baseline
    del x

    torch.manual_seed(4321)
    model = nn.Sequential(*[nn.Linear(dim, dim, dtype=dtype) for _ in range(layers)]).to(device)
    with fully_shard_context(device=device):
        for layer in model:
            fully_shard(
                layer,
                mesh=mesh,
                placements=_flat_placements(),
                mixed_precision_policy=MixedPrecisionPolicy(
                    main_params_dtype=dtype, main_grads_dtype=dtype
                ),
            )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    x = torch.randn(batch, dim, device=device, dtype=dtype)
    torch.cuda.reset_peak_memory_stats(device)
    train_steps(model, optimizer, x)
    sharded_peak = torch.cuda.max_memory_allocated(device)
    logger.info(
        "FSDP peak memory: rank=%s, baseline=%s, sharded=%s",
        rank,
        _mb(baseline_peak),
        _mb(sharded_peak),
    )

    assert sharded_peak < baseline_peak
