# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the minimal Megatron-FSDP path."""

import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import cast

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.profiler import ProfilerActivity, profile

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Flat,
    Placements,
    fully_shard,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.fully_shard import (
    DelayedRelease,
    FsdpContext,
    FsdpModule,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.mixed_precision import MixedPrecisionPolicy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DistributedSetup:
    """Per-rank distributed test setup."""

    rank: int
    world_size: int
    device: torch.device


@pytest.fixture(scope="module")
def setup() -> Iterator[DistributedSetup]:
    """Read torchrun rank state and set up this rank's local device."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("Not running under torchrun.")
    if not torch.cuda.is_available():
        pytest.skip("Experimental FSDP tests require CUDA.")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    yield DistributedSetup(rank=rank, world_size=world_size, device=device)


@pytest.fixture(autouse=True)
def synchronize_before_test(setup: DistributedSetup) -> Iterator[None]:
    """Keep distributed tests aligned when previous tests left async CUDA work queued."""
    if setup.device.type == "cuda":
        torch.cuda.synchronize(setup.device)
    if dist.is_initialized():
        if setup.device.type == "cuda":
            dist.barrier(device_ids=[setup.device.index])
        else:
            dist.barrier()
    yield


class TinyModel(nn.Module):
    """Small model with two separately shardable units."""

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
    """Model with direct parameters and multiple child FSDP units."""

    def __init__(
        self, dim: int = 4, num_children: int = 3, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.ones(dim, dtype=dtype))
        self.layers = nn.ModuleList(
            [nn.Linear(dim, dim, bias=False, dtype=dtype) for _ in range(num_children)]
        )

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


class ConstantMetaModel(nn.Module):
    """Model whose meta parameter is initialized by reset_parameters()."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(4, 4, device="meta"))

    def reset_parameters(self) -> None:
        """Initialize the weight to a deterministic value."""
        self.weight.fill_(3.0)


def _flat_placements() -> Placements:
    return Placements(dp_axes=[0], parameter=[Flat()], gradient=[Flat()], optimizer=[Flat()])


def _mb(num_bytes: int) -> str:
    return f"{num_bytes / 1024**2:.2f} MB"


def _unsharded_storage_nbytes(module: nn.Module) -> int:
    return sum(
        group._unsharded_model_weight.local_buffer.untyped_storage().nbytes()
        for group in module.parameter_groups()
    )


def _set_unsharded_grads(module: nn.Module) -> None:
    for group in module.parameter_groups():
        for parameter in group.unsharded_parameters:
            parameter.grad = torch.ones_like(parameter)


def _fully_shard_children_then_parent(
    model: MultiChildModel, mesh: DeviceMesh, placements: Placements
) -> None:
    for layer in model.layers:
        fully_shard(layer, mesh=mesh, placements=placements)
    fully_shard(model, mesh=mesh, placements=placements)
    model._lazy_init_context()


def _require_context(module: FsdpModule) -> FsdpContext:
    context = module._context
    assert context is not None
    return context


def _run_training_iteration(model: MultiChildModel, x: torch.Tensor) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    if x.grad is not None:
        x.grad = None
    output = model(x)
    loss = output.float().square().mean()
    loss.backward()
    return loss


def _events_overlap(first, second) -> bool:
    return (
        first.time_range.start < second.time_range.end
        and second.time_range.start < first.time_range.end
    )


@pytest.mark.distributed
def test_fully_shard_losses_match_baseline(setup: DistributedSetup):
    """Minimal per-module FSDP training should match single-rank SGD."""
    if setup.world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    torch.manual_seed(1234)
    baseline = TinyModel().to(setup.device)
    model = TinyModel().to(setup.device)
    model.load_state_dict(baseline.state_dict())

    fully_shard(model.fc1, mesh=mesh, placements=_flat_placements())
    fully_shard(model.fc2, mesh=mesh, placements=_flat_placements())
    baseline_optimizer = torch.optim.SGD(baseline.parameters(), lr=0.05)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    x = torch.randn(3, 8, device=setup.device)
    target = torch.randn(3, 4, device=setup.device)

    for step in range(5):
        baseline_optimizer.zero_grad()
        optimizer.zero_grad()

        baseline_loss = torch.nn.functional.mse_loss(baseline(x), target)
        loss = torch.nn.functional.mse_loss(model(x), target)
        logger.info(
            "FSDP train parity: rank=%s, step=%s, baseline_loss=%s, sharded_loss=%s",
            setup.rank,
            step,
            baseline_loss.item(),
            loss.item(),
        )
        torch.testing.assert_close(loss, baseline_loss, msg=f"Loss mismatch at step {step}.")

        baseline_loss.backward()
        loss.backward()
        baseline_optimizer.step()
        optimizer.step()


@pytest.mark.distributed
def test_nested_fully_shard_excludes_child_owned_parameters(setup: DistributedSetup):
    """An outer FSDP unit owns direct parameters but not nested child-unit parameters."""
    if setup.world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    model = NestedModel().to(setup.device)

    fully_shard(model.inner, mesh=mesh, placements=_flat_placements())
    fully_shard(model, mesh=mesh, placements=_flat_placements())

    inner_names = [
        name for group in model.inner.parameter_groups() for name in group.parameter_names
    ]
    outer_names = [name for group in model.parameter_groups() for name in group.parameter_names]

    assert inner_names == ["weight"]
    assert outer_names == ["bias"]


@pytest.mark.distributed
def test_child_then_parent_share_one_context(setup: DistributedSetup):
    """A parent FSDP unit should lazily create one context for its subtree."""
    if setup.world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    model = NestedModel().to(setup.device)

    fully_shard(model.inner, mesh=mesh, placements=_flat_placements())
    assert model.inner._context is None
    fully_shard(model, mesh=mesh, placements=_flat_placements())
    assert model._context is None
    assert model.inner._context is None

    model._lazy_init_context()
    context = _require_context(model)

    assert model.inner._context is context
    assert context.is_root_module(model)
    assert not context.is_root_module(model.inner)


@pytest.mark.distributed
def test_two_child_subtrees_then_parent_collapse_to_one_context(setup: DistributedSetup):
    """Sharding a parent should lazily assign one context across child subtrees."""
    if setup.world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    model = MultiChildModel(dim=4, num_children=2).to(setup.device)

    fully_shard(model.layers[0], mesh=mesh, placements=_flat_placements())
    fully_shard(model.layers[1], mesh=mesh, placements=_flat_placements())

    assert model.layers[0]._context is None
    assert model.layers[1]._context is None

    fully_shard(model, mesh=mesh, placements=_flat_placements())
    model._lazy_init_context()
    context = _require_context(model)

    assert model.layers[0]._context is context
    assert model.layers[1]._context is context


@pytest.mark.distributed
def test_sibling_roots_without_parent_keep_separate_contexts(setup: DistributedSetup):
    """Independent FSDP roots should not share runtime scheduling state."""
    if setup.world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    model = MultiChildModel(dim=4, num_children=2).to(setup.device)

    fully_shard(model.layers[0], mesh=mesh, placements=_flat_placements())
    fully_shard(model.layers[1], mesh=mesh, placements=_flat_placements())

    assert model.layers[0]._context is None
    assert model.layers[1]._context is None

    model.layers[0]._lazy_init_context()
    model.layers[1]._lazy_init_context()
    first_context = _require_context(model.layers[0])
    second_context = _require_context(model.layers[1])

    assert first_context is not second_context
    assert first_context.is_root_module(model.layers[0])
    assert second_context.is_root_module(model.layers[1])


@pytest.mark.distributed
def test_only_parent_is_root_in_shared_context(setup: DistributedSetup):
    """Only the outermost FSDP unit should be root in a shared context."""
    if setup.world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    model = MultiChildModel(dim=4, num_children=1).to(setup.device)
    _fully_shard_children_then_parent(model, mesh, _flat_placements())
    context = _require_context(model)

    assert context.is_root_module(model)
    assert not context.is_root_module(model.layers[0])


@pytest.mark.distributed
def test_normal_pre_unshard_drain_retains_release_delay_minus_one(setup: DistributedSetup):
    """Normal pre-unshard should keep only the newest queued delayed release."""
    if setup.world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    model = MultiChildModel(dim=8, num_children=3).to(setup.device)
    _fully_shard_children_then_parent(model, mesh, _flat_placements())
    context = _require_context(model)

    model.layers[0].pre_forward()
    first_storage_nbytes = _unsharded_storage_nbytes(model.layers[0])
    model.layers[0].post_forward()
    assert first_storage_nbytes > 0
    assert _unsharded_storage_nbytes(model.layers[0]) == first_storage_nbytes
    assert len(context.delayed_releases) == 1

    model.layers[1].pre_forward()
    model.layers[1].post_forward()
    assert len(context.delayed_releases) == context.release_delay

    model.layers[2].pre_forward()

    assert len(context.delayed_releases) == context.release_delay - 1
    assert _unsharded_storage_nbytes(model.layers[0]) == 0
    assert _unsharded_storage_nbytes(model.layers[1]) > 0
    assert _unsharded_storage_nbytes(model.layers[2]) > 0

    model.layers[2].post_forward()
    context.drain_delayed_releases(target_length=0)


@pytest.mark.distributed
def test_root_post_forward_drains_delayed_releases_to_zero(setup: DistributedSetup):
    """Root post-forward should release every queued unsharded storage."""
    if setup.world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    model = MultiChildModel(dim=8, num_children=2).to(setup.device)
    _fully_shard_children_then_parent(model, mesh, _flat_placements())
    context = _require_context(model)

    for layer in model.layers:
        layer.pre_forward()
        layer.post_forward()
    assert len(context.delayed_releases) == context.release_delay

    model.pre_forward()
    model.post_forward()

    assert len(context.delayed_releases) == 0
    assert _unsharded_storage_nbytes(model) == 0
    assert all(_unsharded_storage_nbytes(layer) == 0 for layer in model.layers)


@pytest.mark.distributed
def test_root_post_backward_drains_delayed_releases_to_zero(setup: DistributedSetup):
    """Root post-backward should release every queued unsharded storage."""
    if setup.world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    model = MultiChildModel(dim=8, num_children=1).to(setup.device)
    _fully_shard_children_then_parent(model, mesh, _flat_placements())
    context = _require_context(model)
    layer = model.layers[0]

    layer.pre_backward()
    _set_unsharded_grads(layer)
    layer.post_backward()
    assert len(context.delayed_releases) == 1
    assert _unsharded_storage_nbytes(layer) > 0

    model.pre_backward()
    _set_unsharded_grads(model)
    model.post_backward()

    assert len(context.delayed_releases) == 0
    assert _unsharded_storage_nbytes(model) == 0
    assert _unsharded_storage_nbytes(layer) == 0


@pytest.mark.distributed
def test_delayed_release_waits_on_recorded_cuda_event(setup: DistributedSetup):
    """Delayed release should be stream-ordered after its recorded consumer event."""
    if setup.device.type != "cuda":
        pytest.skip("CUDA event ordering verification requires CUDA.")
    if not hasattr(torch.cuda, "_sleep"):
        pytest.skip("CUDA sleep kernel is unavailable in this PyTorch build.")

    context = FsdpContext(setup.device)
    producer_stream = torch.cuda.Stream(device=setup.device)
    release_event: torch.cuda.Event | None = None

    class ReleaseRecorder:
        def release_unsharded_storage(self) -> None:
            nonlocal release_event
            release_event = torch.cuda.current_stream(setup.device).record_event()

    with torch.cuda.stream(producer_stream):
        torch.cuda._sleep(50_000_000)
        consumer_event = producer_stream.record_event()

    context.delayed_releases.append(
        DelayedRelease(consumer_event=consumer_event, module=cast(FsdpModule, ReleaseRecorder()))
    )
    context.drain_delayed_releases(target_length=0)

    assert release_event is not None
    release_event.synchronize()
    assert consumer_event.query()


@pytest.mark.distributed
def test_fully_sharded_root_with_child_units_overlaps_all_gather_and_compute(
    setup: DistributedSetup,
):
    """A shared root context should let child all-gathers overlap GEMM compute."""
    if setup.world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")
    if setup.device.type != "cuda":
        pytest.skip("CUDA profiler verification requires CUDA.")

    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    dim = 4096
    dtype = torch.bfloat16
    model = MultiChildModel(dim=dim, num_children=4, dtype=dtype).to(setup.device)
    placements = _flat_placements()
    policy = MixedPrecisionPolicy(main_params_dtype=dtype, main_grads_dtype=dtype)
    for layer in model.layers:
        fully_shard(layer, mesh=mesh, placements=placements, mixed_precision_policy=policy)
    fully_shard(model, mesh=mesh, placements=placements, mixed_precision_policy=policy)
    assert model._context is None
    assert all(layer._context is None for layer in model.layers)

    x = torch.randn(4096, dim, device=setup.device, dtype=dtype, requires_grad=True)

    _run_training_iteration(model, x)
    torch.cuda.synchronize(setup.device)
    context = _require_context(model)
    assert context.is_root_module(model)
    assert all(layer._context is context for layer in model.layers)
    assert all(not context.is_root_module(layer) for layer in model.layers)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], acc_events=True) as prof:
        _run_training_iteration(model, x)
        prof.step()
    torch.cuda.synchronize(setup.device)

    cuda_events = [event for event in prof.events() if event.device_type.name == "CUDA"]
    all_gather_events = [
        event
        for event in cuda_events
        if "nccl" in event.name.lower() and "allgather" in event.name.lower()
    ]
    compute_events = [
        event
        for event in cuda_events
        if any(token in event.name.lower() for token in ("gemm", "cutlass", "cublas"))
    ]
    assert all_gather_events, [event.name for event in cuda_events]
    assert compute_events, [event.name for event in cuda_events]

    all_gather_streams = {event.device_resource_id for event in all_gather_events}
    compute_streams = {event.device_resource_id for event in compute_events}
    assert len(all_gather_streams) == 1
    assert all_gather_streams.isdisjoint(compute_streams)

    assert any(
        _events_overlap(all_gather_event, compute_event)
        for all_gather_event in all_gather_events
        for compute_event in compute_events
    )


@pytest.mark.distributed
def test_frozen_parameter_group_does_not_allocate_main_grad(setup: DistributedSetup):
    """A non-trainable parameter group should not allocate persistent main gradients."""
    if setup.world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    model = nn.Linear(4, 4, bias=False).to(setup.device)
    model.weight.requires_grad_(False)

    fully_shard(model, mesh=mesh, placements=_flat_placements())

    (group,) = model.parameter_groups()
    assert not group.requires_grad
    assert group.main_grad is None


pytest.mark.distributed


def test_backward_averages_across_dp_and_accumulates_across_calls(setup: DistributedSetup):
    """Each backward averages over DP ranks; repeated backwards accumulate by summing."""
    if setup.world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    model = nn.Linear(1, setup.world_size, bias=False).to(setup.device)
    with torch.no_grad():
        model.weight.fill_(1.0)

    fully_shard(model, mesh=mesh, placements=_flat_placements())

    x = torch.full((1, 1), float(setup.rank + 1), device=setup.device)
    model(x).sum().backward()
    model(x).sum().backward()

    assert isinstance(model.weight.grad, DTensor)
    local_grad = model.weight.grad.to_local()
    expected = torch.full_like(local_grad, float(setup.world_size + 1))
    torch.testing.assert_close(local_grad, expected, rtol=0, atol=0)


@pytest.mark.distributed
def test_next_forward_uses_optimizer_updated_weights(setup: DistributedSetup):
    """The next forward should observe weights updated by the previous optimizer step."""
    if setup.world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    model = nn.Linear(1, setup.world_size, bias=False, dtype=torch.bfloat16).to(setup.device)
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
    x = torch.ones(1, 1, device=setup.device, dtype=torch.bfloat16)

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


@pytest.mark.distributed
def test_meta_parameters_initialize_with_reset_parameters(setup: DistributedSetup):
    """Meta parameters should be replaced by sharded DTensors and initialized in place."""
    if setup.world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    model = ConstantMetaModel()

    fully_shard(model, mesh=mesh, placements=_flat_placements())

    (group,) = model.parameter_groups()
    full_weight = group.model_weight.allgather(0).get_local_tensor(0)
    assert not full_weight.is_meta
    torch.testing.assert_close(full_weight, torch.full_like(full_weight, 3.0))


@pytest.mark.distributed
def test_non_leaf_parameter_view_survives_storage_resize(setup: DistributedSetup):
    """A non-leaf parameter view saved for backward should survive full-storage resize."""
    if setup.world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    model = NonLeafViewModel().to(setup.device)
    fully_shard(model, mesh=mesh, placements=_flat_placements())

    group = model.parameter_groups()[0]
    x = torch.randn(8, device=setup.device, requires_grad=True)
    loss = model(x).sum()

    assert group._unsharded_model_weight is not None
    assert group._unsharded_model_weight.local_buffer.untyped_storage().nbytes() == 0

    loss.backward()

    assert group.main_grad is not None
    assert group._unsharded_model_weight is not None
    assert group._unsharded_model_weight.local_buffer.untyped_storage().nbytes() == 0


@pytest.mark.distributed
def test_fully_shard_reduces_peak_training_memory(setup: DistributedSetup):
    """Per-layer FSDP should reduce peak CUDA memory during training."""
    if setup.world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")
    if setup.device.type != "cuda":
        pytest.skip("Peak memory verification requires CUDA.")

    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
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
    baseline = nn.Sequential(*[nn.Linear(dim, dim, dtype=dtype) for _ in range(layers)]).to(
        setup.device
    )
    baseline_optimizer = torch.optim.AdamW(baseline.parameters(), lr=0.01)
    x = torch.randn(batch, dim, device=setup.device, dtype=dtype)
    torch.cuda.reset_peak_memory_stats(setup.device)
    train_steps(baseline, baseline_optimizer, x)
    torch.cuda.synchronize(setup.device)
    baseline_peak = torch.cuda.max_memory_allocated(setup.device)

    del baseline_optimizer
    del baseline
    del x
    torch.cuda.empty_cache()

    torch.manual_seed(4321)
    model = nn.Sequential(*[nn.Linear(dim, dim, dtype=dtype) for _ in range(layers)]).to(
        setup.device
    )
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

    x = torch.randn(batch, dim, device=setup.device, dtype=dtype)
    torch.cuda.reset_peak_memory_stats(setup.device)
    train_steps(model, optimizer, x)
    torch.cuda.synchronize(setup.device)
    sharded_peak = torch.cuda.max_memory_allocated(setup.device)
    logger.info(
        "FSDP peak memory: rank=%s, baseline=%s, sharded=%s",
        setup.rank,
        _mb(baseline_peak),
        _mb(sharded_peak),
    )

    assert sharded_peak < baseline_peak
