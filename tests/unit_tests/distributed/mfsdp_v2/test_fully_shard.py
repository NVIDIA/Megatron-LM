# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the minimal Megatron-FSDP path."""

import logging

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor
from torch.profiler import ProfilerActivity, profile

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Flat,
    Partial,
    Placements,
    Replicate,
    fully_shard,
    fully_shard_optimizer,
    microbatch,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.mixed_precision import MixedPrecisionPolicy
from tests.unit_tests.distributed.mfsdp_v2.profiler_utils import (
    collect_linked_kernels,
    events_overlap,
)

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


def _mb(num_bytes: int) -> str:
    return f"{num_bytes / 1024**2:.2f} MB"


# CPU ops that a device event chains up to via cpu_parent, used to attribute the device
# work to its enclosing collective or matmul operation.
_ALL_GATHER_OP_NAME_SUBSTRING = "allgather"
_REDUCE_SCATTER_OP_NAME_SUBSTRING = "reduce_scatter"
_GEMM_OP_NAME_SUBSTRING = "aten::mm"


def _nccl_events(cuda_events, *name_fragments):
    """CUDA NCCL events whose name contains any of ``name_fragments`` (case-insensitive)."""
    return [
        event
        for event in cuda_events
        if "nccl" in event.name.lower()
        and event.activity_type == "kernel"
        and any(fragment in event.name.lower() for fragment in name_fragments)
    ]


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

    # Shard the child layers, then the model, so the children share a root context
    # and reduce through the overlap path instead of as independent roots.
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
                with microbatch(model, is_last=is_last):
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


def test_hsdp_defers_dp_outer_allreduce_to_last_microbatch(distributed_setup):
    """HSDP reduce-scatters DP-inner every microbatch but all-reduces DP-outer once.

    ``fully_shard(model)`` makes the child units share a root context so their
    reductions run through the overlap path rather than as independent roots.
    Counting NCCL events over a multi-microbatch step, the DP-inner reduce-scatter
    fires once per microbatch per group while the DP-outer all-reduce that
    finalizes main_grad fires only on the last microbatch, so the reduce-scatter
    count is exactly ``num_microbatches`` times the all-reduce count. This asserts
    on event counts only, not numerics.
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
            with microbatch(model, is_last=is_last):
                loss = torch.nn.functional.mse_loss(model(microbatch_x), microbatch_target)
                (loss / num_microbatches).backward()
        optimizer.step()

    train_one_step()
    torch.cuda.synchronize(device)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        train_one_step()
        torch.cuda.synchronize(device)

    cuda_events = [event for event in prof.events() if event.device_type.name == "CUDA"]
    reduce_scatter_events = _nccl_events(cuda_events, "reducescatter", "reduce_scatter")
    all_reduce_events = _nccl_events(cuda_events, "allreduce")
    # One DP-outer all-reduce per parameter group -- each child layer plus the
    # root unit's bias -- fired only on the last microbatch. Plain DP fires none.
    assert len(all_reduce_events) == num_children + 1, [event.name for event in cuda_events]
    # DP-inner reduce-scatter runs every microbatch; the DP-outer all-reduce runs
    # only on the last, so the counts differ by exactly the microbatch factor.
    assert len(reduce_scatter_events) == len(all_reduce_events) * num_microbatches, (
        f"Expected reduce-scatter ({len(reduce_scatter_events)}) to be {num_microbatches}x "
        f"the DP-outer all-reduce count ({len(all_reduce_events)})."
    )


def test_nested_fully_shard_excludes_child_owned_parameters(distributed_setup):
    """An outer FsdpModule owns direct parameters but not nested child FsdpModule parameters."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    model = NestedModel().to(device)

    fully_shard(model.inner, mesh=mesh, placements=_flat_placements())
    fully_shard(model, mesh=mesh, placements=_flat_placements())

    inner_names = [name for group in model.inner.parameter_groups for name in group.parameter_names]
    outer_names = [name for group in model.parameter_groups for name in group.parameter_names]

    assert inner_names == ["weight"]
    assert outer_names == ["bias"]


def test_forward_peak_memory_bounds_in_flight_child_all_gathers(distributed_setup):
    """Forward peak memory should stay below three live child all-gathers."""
    rank = distributed_setup.rank
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    dim = 4096
    dtype = torch.bfloat16
    model = MultiChildModel(dim=dim, num_children=4).to(dtype=dtype, device=device)
    placements = _flat_placements()
    policy = MixedPrecisionPolicy(main_params_dtype=dtype, main_grads_dtype=dtype)
    for layer in model.layers:
        fully_shard(layer, mesh=mesh, placements=placements, mixed_precision_policy=policy)
    fully_shard(model, mesh=mesh, placements=placements, mixed_precision_policy=policy)

    x = torch.randn(2, dim, device=device, dtype=dtype)
    with torch.no_grad():
        model(x)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    resting_allocated = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        model(x)
    torch.cuda.synchronize(device)
    peak_delta = torch.cuda.max_memory_allocated(device) - resting_allocated

    child_weight_nbytes = dim * dim * torch.empty((), dtype=dtype).element_size()
    bound_nbytes = 3 * child_weight_nbytes

    # A parent forward should keep one previous child unsharded until its compute
    # stream consumer is safe, plus the current child being unsharded. The bound
    # is looser than two child weights to avoid coupling this test to CUDA
    # allocator granularity and small temporary buffers, while still catching
    # delayed releases piling up across the four child layers.
    assert peak_delta < bound_nbytes, (
        "FSDP forward peak memory exceeded the in-flight all-gather bound: "
        f"rank={rank}, peak_delta={_mb(peak_delta)}, "
        f"three_child_weights={_mb(bound_nbytes)}"
    )


def test_root_forward_returns_to_resting_memory(distributed_setup):
    """Root forward should release child all-gather storage before returning."""
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
    for layer in model.layers:
        fully_shard(layer, mesh=mesh, placements=placements, mixed_precision_policy=policy)
    fully_shard(model, mesh=mesh, placements=placements, mixed_precision_policy=policy)

    x = torch.randn(2, dim, device=device, dtype=dtype)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    resting_allocated = torch.cuda.memory_allocated(device)

    with torch.no_grad():
        output = model(x)
    del output
    torch.cuda.synchronize(device)
    allocated_after_forward = torch.cuda.memory_allocated(device)
    extra_allocated = allocated_after_forward - resting_allocated
    child_weight_nbytes = dim * dim * torch.empty((), dtype=dtype).element_size()

    assert extra_allocated < child_weight_nbytes, (
        "Root forward did not return to resting memory after draining child releases: "
        f"rank={rank}, extra_allocated={_mb(extra_allocated)}, "
        f"one_child_weight={_mb(child_weight_nbytes)}"
    )


def test_root_backward_returns_to_resting_memory(distributed_setup):
    """Root backward should release child all-gather storage before returning."""
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
    for layer in model.layers:
        fully_shard(layer, mesh=mesh, placements=placements, mixed_precision_policy=policy)
    fully_shard(model, mesh=mesh, placements=placements, mixed_precision_policy=policy)

    x = torch.randn(2, dim, device=device, dtype=dtype, requires_grad=True)
    output = model(x)
    loss = output.float().square().mean()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    allocated_before_backward = torch.cuda.memory_allocated(device)

    loss.backward()
    del loss, output
    torch.cuda.synchronize(device)
    allocated_after_backward = torch.cuda.memory_allocated(device)
    extra_allocated = allocated_after_backward - allocated_before_backward
    child_weight_nbytes = dim * dim * torch.empty((), dtype=dtype).element_size()

    assert extra_allocated < child_weight_nbytes, (
        "Root backward did not return to resting memory after draining child releases: "
        f"rank={rank}, extra_allocated={_mb(extra_allocated)}, "
        f"one_child_weight={_mb(child_weight_nbytes)}"
    )


@pytest.mark.parametrize("use_symm_mem", [False, True], ids=["default", "symmetric_memory"])
def test_overlaps_communication_and_compute(distributed_setup, use_symm_mem):
    """Forward and backward communication should overlap GEMM compute."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    # A large hidden size keeps the per-layer GEMMs long enough that the
    # collectives reliably overlap them. The overlap count is otherwise
    # launch-bound: the host issues kernels with gaps (amplified by CI's
    # `coverage run` wrapper), so with short GEMMs a collective can land in a
    # gap between GEMMs instead of running alongside one, making the count jitter
    # run to run. At dim=16384 the GEMMs dominate that launch jitter and the
    # overlap becomes deterministic. (dim=8192 was flaky under coverage.)
    dim = 16384
    num_children = 4
    dtype = torch.bfloat16

    # new_group requires a default process group. Initialize it here so this test works
    # in isolation. Do not eagerly initialize it with device_id in the shared fixture:
    # that can hang teardown after communicator splits; see
    # https://github.com/pytorch/pytorch/issues/190396.
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    if use_symm_mem:
        # Dedicated communicator with NCCL's zero-CTA policy. cta_policy is a
        # per-communicator property, so scoping it to this group leaves the rest of the
        # bucket on default-CTA symmetric-memory kernels (test_symmetric_memory.py asserts
        # ncclSymk all-gather kernel counts, which zero-CTA would turn into copy-engine
        # memcpys). This 1-D group models the DP (FSDP) sub-mesh that mfsdp is handed in
        # production: with EP/TP the full device mesh is multi-dimensional, but mfsdp
        # requires an all-FSDP mesh (see experimental/module.py) and never sees the TP/EP
        # axes, so only the DP communicator needs the zero-CTA policy.
        zero_cta_options = dist.ProcessGroupNCCL.Options()
        zero_cta_options.config.cta_policy = dist.ProcessGroupNCCL.NCCL_CTA_POLICY_ZERO
        dp_group = dist.new_group(backend="nccl", pg_options=zero_cta_options)
        # NCCL window registration can fail when symmetric-memory rendezvous is the first
        # operation on a communicator, so initialize this communicator explicitly.
        dist.barrier(group=dp_group, device_ids=[device.index])
    else:
        dp_group = dist.new_group(backend="nccl")

    mesh = DeviceMesh.from_group(dp_group, device.type)
    model = MultiChildModel(dim=dim, num_children=num_children).to(dtype=dtype)
    placements = _flat_placements()
    policy = MixedPrecisionPolicy(main_params_dtype=dtype, main_grads_dtype=dtype)
    for layer in model.layers:
        fully_shard(
            layer,
            mesh=mesh,
            placements=placements,
            mixed_precision_policy=policy,
            use_symm_mem=use_symm_mem,
        )
    fully_shard(
        model,
        mesh=mesh,
        placements=placements,
        mixed_precision_policy=policy,
        use_symm_mem=use_symm_mem,
    )

    x = torch.randn(4096, dim, device=device, dtype=dtype, requires_grad=True)

    def train_one_iteration() -> None:
        model.zero_grad(set_to_none=True)
        model(x).sum().backward()

    train_one_iteration()
    torch.cuda.synchronize(device)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        train_one_iteration()
        # Synchronize inside the profiler context so in-flight device kernels
        # complete and get recorded before the profiler stops on __exit__.
        # Synchronizing after the context would finalize the trace first and
        # drop the CUDA events.
        torch.cuda.synchronize(device)

    gemm_kernels = collect_linked_kernels(prof, _GEMM_OP_NAME_SUBSTRING)
    # Each child Linear runs one forward and two backward matmuls. aten::mm may also
    # launch auxiliary kernels, so check only the matmul lower bound.
    assert len(gemm_kernels) >= 3 * num_children, (
        f"Expected at least {3 * num_children} kernels linked to GEMMs, got "
        f"{len(gemm_kernels)}: "
        f"{[event.name for event in gemm_kernels]}"
    )

    allgather_kernels = collect_linked_kernels(prof, _ALL_GATHER_OP_NAME_SUBSTRING)
    reduce_scatter_kernels = collect_linked_kernels(prof, _REDUCE_SCATTER_OP_NAME_SUBSTRING)
    # The num_children child layers plus the root are each a sharded module; each does a
    # forward and a backward all-gather and one reduce-scatter. Zero-CTA moves the
    # all-gather to copy-engine memcpys, so it should not emit all-gather kernels.
    num_sharded_modules = num_children + 1
    expected_allgather_kernel_count = 0 if use_symm_mem else 2 * num_sharded_modules
    assert len(allgather_kernels) == expected_allgather_kernel_count, (
        f"Expected {expected_allgather_kernel_count} all-gather kernels, got "
        f"{len(allgather_kernels)}: {[event.name for event in allgather_kernels]}"
    )
    assert len(reduce_scatter_kernels) == num_sharded_modules, (
        f"Expected {num_sharded_modules} reduce-scatter kernels, got "
        f"{len(reduce_scatter_kernels)}: {[event.name for event in reduce_scatter_kernels]}"
    )

    allgather_streams = {event.device_resource_id for event in allgather_kernels}
    reduce_scatter_streams = {event.device_resource_id for event in reduce_scatter_kernels}
    gemm_streams = {event.device_resource_id for event in gemm_kernels}
    if allgather_kernels:
        assert len(allgather_streams) == 1
    assert len(reduce_scatter_streams) == 1
    assert allgather_streams.isdisjoint(reduce_scatter_streams)
    assert allgather_streams.isdisjoint(gemm_streams)
    assert reduce_scatter_streams.isdisjoint(gemm_streams)

    allgather_overlap_count = sum(
        any(events_overlap(event, gemm) for gemm in gemm_kernels) for event in allgather_kernels
    )
    reduce_scatter_overlap_count = sum(
        any(events_overlap(event, gemm) for gemm in gemm_kernels)
        for event in reduce_scatter_kernels
    )
    expected_allgather_overlap = 2 * (num_children - 1)
    expected_reduce_scatter_overlap = num_children - 1
    if not use_symm_mem:
        assert allgather_overlap_count >= expected_allgather_overlap, (
            f"Expected at least {expected_allgather_overlap} all-gathers to "
            f"overlap compute, got {allgather_overlap_count}/{len(allgather_kernels)}."
        )
    assert reduce_scatter_overlap_count >= expected_reduce_scatter_overlap, (
        f"Expected at least {expected_reduce_scatter_overlap} reduce-scatters to overlap "
        f"compute, got {reduce_scatter_overlap_count}/{len(reduce_scatter_kernels)}."
    )

    # Release the dedicated communicator so it does not leak into the shared session.
    dist.destroy_process_group(dp_group)


def test_parameterless_parent_with_child_modules_trains(distributed_setup):
    """A parent with no unowned parameters should still root trainable child FsdpModules."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device

    mesh = init_device_mesh(device.type, (world_size,))
    torch.manual_seed(5678)
    model = nn.Sequential(nn.Linear(4, 4, bias=False), nn.Linear(4, 2, bias=False)).to(device)

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
    with torch.no_grad():
        model.weight.fill_(1.0)

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
    with torch.no_grad():
        model.weight.fill_(1.0)

    fully_shard(
        model,
        mesh=mesh,
        placements=_flat_placements(),
        mixed_precision_policy=MixedPrecisionPolicy(main_params_dtype=torch.float32),
    )
    # SGD's foreach/fused CUDA paths require matching parameter and gradient dtypes.
    # Use the scalar path to exercise FP32 main weights with default BF16 main grads.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.25, foreach=False)
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


def test_microbatch_scopes_child_contexts(distributed_setup):
    """microbatch() should scope FSDP child contexts under an unwrapped parent."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device

    mesh = init_device_mesh(device.type, (world_size,))
    model = nn.Sequential(nn.Linear(1, 1, bias=False), nn.Linear(1, 1, bias=False)).to(device)
    for layer in model:
        fully_shard(layer, mesh=mesh, placements=_flat_placements())

    with microbatch(model, is_last=False):
        for layer in model:
            assert not layer.context.is_last_microbatch

    for layer in model:
        assert layer.context.is_last_microbatch


def test_cpu_initialized_parameters_shard_to_mesh_device(distributed_setup):
    """A CPU model should support sharding a child before moving the full model to CUDA."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device

    mesh = init_device_mesh(device.type, (world_size,))
    model = nn.Sequential(nn.Linear(4, 4, bias=False), nn.Linear(4, 4, bias=False))
    with torch.no_grad():
        model[0].weight.fill_(2.0)
        model[1].weight.fill_(3.0)
    x = torch.ones(1, 4)
    expected_output = model(x).to(device)

    # Shard the second layer's parameters onto the mesh device; the unwrapped
    # first layer's parameters remain on CPU until model.to(device) below.
    fully_shard(model[1], mesh=mesh, placements=_flat_placements())

    assert model[0].weight.device.type == "cpu"
    assert isinstance(model[1].weight, DTensor)
    assert model[1].weight.device == device

    model.to(device)

    output = model(x.to(device))
    torch.testing.assert_close(output, expected_output)


def test_non_leaf_parameter_view_survives_storage_resize(distributed_setup):
    """A non-leaf parameter view saved for backward should survive full-storage resize."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    model = NonLeafViewModel().to(device)
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
    torch.cuda.synchronize(device)
    baseline_peak = torch.cuda.max_memory_allocated(device)

    del baseline_optimizer
    del baseline
    del x
    torch.cuda.empty_cache()

    torch.manual_seed(4321)
    model = nn.Sequential(*[nn.Linear(dim, dim, dtype=dtype) for _ in range(layers)]).to(device)
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
    torch.cuda.empty_cache()

    x = torch.randn(batch, dim, device=device, dtype=dtype)
    torch.cuda.reset_peak_memory_stats(device)
    train_steps(model, optimizer, x)
    torch.cuda.synchronize(device)
    sharded_peak = torch.cuda.max_memory_allocated(device)
    logger.info(
        "FSDP peak memory: rank=%s, baseline=%s, sharded=%s",
        rank,
        _mb(baseline_peak),
        _mb(sharded_peak),
    )

    assert sharded_peak < baseline_peak
