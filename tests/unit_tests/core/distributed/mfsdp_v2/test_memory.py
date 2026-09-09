# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Memory-accounting tests for Megatron-FSDP."""

import logging

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Partial, Replicate, Shard

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Placements,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.placement import Flat
from megatron.core.distributed.fsdp.src.megatron_fsdp.mixed_precision import MixedPrecisionPolicy

logger = logging.getLogger(__name__)


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


def _flat_placements() -> Placements:
    return Placements(dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)])


def _zero1_placements() -> Placements:
    return Placements(
        dp_axes=[0], parameter=[Replicate()], gradient=[Partial("avg")], optimizer=[Shard(0)]
    )


def _zero2_placements() -> Placements:
    return Placements(
        dp_axes=[0], parameter=[Replicate()], gradient=[Shard(0)], optimizer=[Shard(0)]
    )


def _mb(num_bytes: int) -> str:
    return f"{num_bytes / 1024**2:.2f} MB"


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
        for layer_index, layer in enumerate(model.layers):
            for group_index, group in enumerate(layer.parameter_groups):
                assert group.model_weight is group.main_weight, (
                    f"Layer {layer_index}, parameter group {group_index} should alias "
                    "model and main weights."
                )
                assert group.post_optimizer_model_weight is group.model_weight, (
                    f"ZeRO-3 layer {layer_index}, parameter group {group_index} should use "
                    "model_weight itself as its post-optimizer model weight."
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


@pytest.mark.parametrize(
    "unify_communication_stream", [False, True], ids=["separate_streams", "unified_stream"]
)
def test_training_step_peak_memory_bounds_full_size_buffers(
    distributed_setup, unify_communication_stream
):
    """A training step should stay within its full-size-buffer bound."""
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
    with fully_shard_context(device=device, unify_communication_stream=unify_communication_stream):
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

    # Warm up so cuBLAS's workspaces land in resting_allocated rather than in peak_delta,
    # which would otherwise depend on whether an earlier test already allocated them.
    train_step()

    resting_allocated = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    train_step()
    peak_delta = torch.cuda.max_memory_allocated(device) - resting_allocated

    # Backward keeps the current child and one prefetched child unsharded. The current
    # child also has a full wgrad until it is copied into a full reduce-scatter input.
    # With separate streams, their allocation cannot reuse the released full-weight
    # storage, for a four-buffer peak. With a unified stream, the release precedes the
    # allocation on that stream, reducing the peak to three. The slack on top covers
    # allocator granularity and small temporaries, measured at ~169 KiB.
    child_weight_nbytes = dim * dim * torch.empty((), dtype=dtype).element_size()
    full_buffer_bound = 3 if unify_communication_stream else 4
    bound_nbytes = full_buffer_bound * child_weight_nbytes + 1024**2

    assert peak_delta < bound_nbytes, (
        "FSDP training-step peak memory exceeded the full-size-buffer bound: "
        f"rank={rank}, peak_delta={_mb(peak_delta)}, "
        f"bound={_mb(bound_nbytes)} ({full_buffer_bound} full child buffers + 1.00 MB)"
    )


def test_zero1_memory_uses_sharded_optimizer_and_replicated_weight(distributed_setup):
    """ZeRO-1 keeps optimizer state sharded while model weights are replicated."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    dim = 4096
    num_tokens = 256
    dtype = torch.bfloat16
    x = torch.ones(num_tokens, dim, device=device, dtype=dtype)
    allocated_before_setup = torch.cuda.memory_allocated(device)
    model = ElementwiseModel(dim).to(device=device, dtype=dtype)
    mesh = init_device_mesh(device.type, (world_size,))
    placements = _zero1_placements()
    with fully_shard_context(device=device):
        fully_shard(model, mesh=mesh, placements=placements)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, foreach=False)
    fully_shard_optimizer(optimizer)
    (parameter_group,) = model.parameter_groups

    full_bf16_weight_nbytes = dim * dim * torch.empty((), dtype=dtype).element_size()
    sharded_bf16_weight_nbytes = full_bf16_weight_nbytes // world_size
    sharded_fp32_weight_nbytes = (
        dim * dim // world_size * torch.empty((), dtype=torch.float32).element_size()
    )
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.reset_peak_memory_stats(device)

    def train_step() -> None:
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

    train_step()
    peak_nbytes = torch.cuda.max_memory_allocated(device) - allocated_before_setup
    resting_nbytes = torch.cuda.memory_allocated(device) - allocated_before_setup
    assert parameter_group.model_weight.placements == (Replicate(),)
    assert parameter_group.post_optimizer_model_weight.placements == (Flat(),)

    optimizer_state_nbytes = sum(
        state["exp_avg"].to_local().nbytes + state["exp_avg_sq"].to_local().nbytes
        for state in optimizer.state.values()
    )
    # Resting memory holds one replicated BF16 model weight, one sharded BF16
    # main gradient, and three sharded FP32 buffers: main weight and two Adam states.
    expected_resting_nbytes = (
        full_bf16_weight_nbytes + sharded_bf16_weight_nbytes + 3 * sharded_fp32_weight_nbytes
    )
    # Peak memory additionally holds the casted gradient and the sqrt and division
    # intermediates in ``denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)``.
    expected_peak_nbytes = expected_resting_nbytes + 3 * sharded_fp32_weight_nbytes
    assert optimizer_state_nbytes == 2 * sharded_fp32_weight_nbytes
    assert resting_nbytes < expected_resting_nbytes + 1024**2
    assert peak_nbytes < expected_peak_nbytes + 1024**2


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


@pytest.mark.parametrize(
    "placements_factory",
    [
        pytest.param(_zero1_placements, id="zero1"),
        pytest.param(_zero2_placements, id="zero2"),
        pytest.param(_flat_placements, id="zero3"),
    ],
)
def test_fully_shard_reduces_peak_training_memory(distributed_setup, placements_factory):
    """Per-layer FSDP should reduce peak CUDA memory for each sharding strategy."""
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
                placements=placements_factory(),
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

    assert sharded_peak < baseline_peak, (
        f"Expected FSDP to reduce peak training memory on rank {rank}: "
        f"baseline={_mb(baseline_peak)}, sharded={_mb(sharded_peak)}"
    )
