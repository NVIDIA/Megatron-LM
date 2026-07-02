# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the minimal Megatron-FSDP path."""

import logging
from typing import cast

import pytest
import torch
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
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import (
    DelayedRelease,
    FsdpContext,
    FsdpModule,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.mixed_precision import MixedPrecisionPolicy

logger = logging.getLogger(__name__)


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


@pytest.mark.parametrize("num_microbatches", [1, 3])
def test_fully_shard_losses_match_baseline(distributed_setup, num_microbatches):
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


def test_nested_fully_shard_excludes_child_owned_parameters(distributed_setup):
    """An outer FSDP unit owns direct parameters but not nested child-unit parameters."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    model = NestedModel().to(device)

    fully_shard(model.inner, mesh=mesh, placements=_flat_placements())
    fully_shard(model, mesh=mesh, placements=_flat_placements())

    inner_names = [
        name for group in model.inner.parameter_groups() for name in group.parameter_names
    ]
    outer_names = [name for group in model.parameter_groups() for name in group.parameter_names]

    assert inner_names == ["weight"]
    assert outer_names == ["bias"]


def test_child_then_parent_share_one_context(distributed_setup):
    """A parent FSDP unit should lazily create one context for its subtree."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    model = NestedModel().to(device)
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


def test_two_child_subtrees_then_parent_collapse_to_one_context(distributed_setup):
    """Sharding a parent should lazily assign one context across child subtrees."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    model = MultiChildModel(dim=4, num_children=2).to(device)
    for layer in model.layers:
        fully_shard(layer, mesh=mesh, placements=_flat_placements())
    assert model.layers[0]._context is None
    assert model.layers[1]._context is None
    fully_shard(model, mesh=mesh, placements=_flat_placements())

    model._lazy_init_context()
    context = _require_context(model)

    assert model.layers[0]._context is context
    assert model.layers[1]._context is context


def test_sibling_roots_without_parent_keep_separate_contexts(distributed_setup):
    """Independent FSDP roots should not share runtime scheduling state."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    model = MultiChildModel(dim=4, num_children=2).to(device)
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


def test_fsdp_nvtx_ranges_use_named_module_paths(distributed_setup, monkeypatch):
    """FSDP compute annotations should use root-relative module paths."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")
    if device.type != "cuda":
        pytest.skip("FSDP NVTX annotations require CUDA.")

    mesh = init_device_mesh(device.type, (world_size,))
    model = MultiChildModel(dim=8, num_children=2).to(device)
    _fully_shard_children_then_parent(model, mesh, _flat_placements())

    pushed_ranges: list[str] = []
    popped_ranges = 0

    def record_range_push(message: str) -> None:
        pushed_ranges.append(message)

    def record_range_pop() -> None:
        nonlocal popped_ranges
        popped_ranges += 1

    monkeypatch.setattr(torch.cuda.nvtx, "range_push", record_range_push)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", record_range_pop)

    x = torch.randn(2, 8, device=device, requires_grad=True)
    _run_training_iteration(model, x)

    expected_ranges = [
        "FSDP:<root>:forward_compute",
        "FSDP:layers.0:forward_compute",
        "FSDP:layers.1:forward_compute",
        "FSDP:<root>:backward_compute",
        "FSDP:layers.0:backward_compute",
        "FSDP:layers.1:backward_compute",
    ]
    for expected_range in expected_ranges:
        assert pushed_ranges.count(expected_range) == 1
    assert len(pushed_ranges) == popped_ranges
    assert not model._forward_compute_range_active
    assert not model._backward_compute_range_active
    for layer in model.layers:
        assert not layer._forward_compute_range_active
        assert not layer._backward_compute_range_active


def test_only_parent_is_root_in_shared_context(distributed_setup):
    """Only the outermost FSDP unit should be root in a shared context."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    model = MultiChildModel(dim=4, num_children=1).to(device)
    _fully_shard_children_then_parent(model, mesh, _flat_placements())
    context = _require_context(model)

    assert context.is_root_module(model)
    assert not context.is_root_module(model.layers[0])


def test_normal_pre_unshard_drain_retains_release_delay_minus_one(distributed_setup):
    """Normal pre-unshard should keep only the newest queued delayed release."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    model = MultiChildModel(dim=8, num_children=3).to(device)
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


def test_root_post_forward_drains_delayed_releases_to_zero(distributed_setup):
    """Root post-forward should release every queued unsharded storage."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    model = MultiChildModel(dim=8, num_children=2).to(device)
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


def test_root_post_backward_drains_delayed_releases_to_zero(distributed_setup):
    """Root post-backward should release every queued unsharded storage."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    model = MultiChildModel(dim=8, num_children=1).to(device)
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


def test_delayed_release_waits_on_recorded_cuda_event(distributed_setup):
    """Delayed release should be stream-ordered after its recorded consumer event."""
    device = distributed_setup.device
    if device.type != "cuda":
        pytest.skip("CUDA event ordering verification requires CUDA.")
    if not hasattr(torch.cuda, "_sleep"):
        pytest.skip("CUDA sleep kernel is unavailable in this PyTorch build.")

    context = FsdpContext(device)
    producer_stream = torch.cuda.Stream(device=device)
    release_event: torch.cuda.Event | None = None

    class ReleaseRecorder:
        def release_unsharded_storage(self) -> None:
            nonlocal release_event
            release_event = torch.cuda.current_stream(device).record_event()

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


def test_fully_sharded_root_with_child_units_overlaps_all_gather_and_compute(distributed_setup):
    """A shared root context should let child collectives overlap GEMM compute."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")
    if device.type != "cuda":
        pytest.skip("CUDA profiler verification requires CUDA.")

    mesh = init_device_mesh(device.type, (world_size,))
    dim = 4096
    dtype = torch.bfloat16
    model = MultiChildModel(dim=dim, num_children=4, dtype=dtype).to(device)
    placements = _flat_placements()
    policy = MixedPrecisionPolicy(main_params_dtype=dtype, main_grads_dtype=dtype)
    for layer in model.layers:
        fully_shard(layer, mesh=mesh, placements=placements, mixed_precision_policy=policy)
    fully_shard(model, mesh=mesh, placements=placements, mixed_precision_policy=policy)
    assert model._context is None
    assert all(layer._context is None for layer in model.layers)

    x = torch.randn(4096, dim, device=device, dtype=dtype, requires_grad=True)

    _run_training_iteration(model, x)
    torch.cuda.synchronize(device)
    context = _require_context(model)
    assert context.is_root_module(model)
    assert all(layer._context is context for layer in model.layers)
    assert all(not context.is_root_module(layer) for layer in model.layers)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], acc_events=True) as prof:
        _run_training_iteration(model, x)
        prof.step()
    torch.cuda.synchronize(device)

    cuda_events = [event for event in prof.events() if event.device_type.name == "CUDA"]
    all_gather_events = [
        event
        for event in cuda_events
        if "nccl" in event.name.lower() and "allgather" in event.name.lower()
    ]
    reduce_scatter_events = [
        event
        for event in cuda_events
        if "nccl" in event.name.lower()
        and ("reducescatter" in event.name.lower() or "reduce_scatter" in event.name.lower())
    ]
    compute_events = [
        event
        for event in cuda_events
        if any(token in event.name.lower() for token in ("gemm", "cutlass", "cublas"))
    ]
    assert all_gather_events, [event.name for event in cuda_events]
    assert reduce_scatter_events, [event.name for event in cuda_events]
    assert compute_events, [event.name for event in cuda_events]

    all_gather_streams = {event.device_resource_id for event in all_gather_events}
    reduce_scatter_streams = {event.device_resource_id for event in reduce_scatter_events}
    compute_streams = {event.device_resource_id for event in compute_events}
    assert len(all_gather_streams) == 1
    assert len(reduce_scatter_streams) == 1
    assert all_gather_streams == reduce_scatter_streams
    assert all_gather_streams.isdisjoint(compute_streams)
    assert reduce_scatter_streams.isdisjoint(compute_streams)

    assert any(
        _events_overlap(all_gather_event, compute_event)
        for all_gather_event in all_gather_events
        for compute_event in compute_events
    )
    assert any(
        _events_overlap(reduce_scatter_event, compute_event)
        for reduce_scatter_event in reduce_scatter_events
        for compute_event in compute_events
    )


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

    (group,) = model.parameter_groups()
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


def test_cpu_initialized_parameters_shard_to_mesh_device(distributed_setup):
    """CPU-initialized parameters should be sharded with their real values."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    model = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        model.weight.fill_(3.0)
    expected_weight = model.weight.detach().to(device)

    fully_shard(model, mesh=mesh, placements=_flat_placements())

    (group,) = model.parameter_groups()
    full_weight = group.model_weight.allgather(0).get_local_tensor(0)
    assert full_weight.device.type == device.type
    torch.testing.assert_close(full_weight, expected_weight)


def test_non_leaf_parameter_view_survives_storage_resize(distributed_setup):
    """A non-leaf parameter view saved for backward should survive full-storage resize."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    model = NonLeafViewModel().to(device)
    fully_shard(model, mesh=mesh, placements=_flat_placements())

    group = model.parameter_groups()[0]
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
