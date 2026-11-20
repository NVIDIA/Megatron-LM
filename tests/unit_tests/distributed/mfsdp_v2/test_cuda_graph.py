# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CUDA graph tests for Megatron-FSDP."""

import logging
from copy import deepcopy

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Partial, Replicate, Shard

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    FsdpContext,
    GradientFinalizationMode,
    Placements,
    PrefetchOrder,
    RuntimePolicy,
    fully_shard,
    fully_shard_context,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.execution_order import (
    ExecutionAction,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.weight_retention import (
    build_retain_mask,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
    uneven_dtensor_to_full_tensor,
)

logger = logging.getLogger(__name__)

# Sized so NCCL selects its symmetric-memory (ncclSymk*) kernels over ring for the
# sharded Linear collectives; see test_symmetric_memory.py for the rationale.
_HIDDEN = 1024


class NestedModel(nn.Module):
    """Model with a root FSDP unit and multiple child FSDP units."""

    def __init__(self, dim: int, num_children: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))
        self.layers = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(num_children)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run through every child layer with a root-owned bias."""
        x = x + self.bias
        for layer in self.layers:
            x = layer(x)
        return x


def _default_placements() -> Placements:
    return Placements(dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)])


# CUDA graph capture without symmetric memory is supported and runs by default. The symmetric
# memory case tracks https://github.com/NVIDIA/Megatron-LM/issues/6154: its MemPool is separate
# from CUDA graph's private pool, so allocation addresses and lifetimes are not yet guaranteed
# across capture and replay. Mark only that case flaky until the required PyTorch allocator and
# buffer-registration support is available.
@pytest.mark.parametrize(
    "use_symmetric_memory",
    [
        pytest.param(False, id="default"),
        pytest.param(
            True, id="symmetric_memory", marks=[pytest.mark.flaky, pytest.mark.flaky_in_dev]
        ),
    ],
)
def test_captures_full_iteration(distributed_setup, use_symmetric_memory):
    """Two chunks and microbatches replay one complete U/R schedule under CUDA graphs.

    With ``use_symmetric_memory=True`` the FSDP collective path changes: unshard allocates the
    all-gather buffer from a symmetric-memory ``MemPool`` and calls
    ``symm_mem.rendezvous()``, and the backward reduce-scatter does the same for the
    partial-gradient buffer. Both allocation-from-pool and rendezvous are normally
    illegal inside CUDA-graph capture, so this confirms the warmup establishes reusable,
    already-rendezvoused buffers that capture and replay cleanly, matching eager
    losses and parameter updates.
    """
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if use_symmetric_memory and world_size < 2:
        pytest.skip("Symmetric memory requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    torch.manual_seed(1234)
    dim = _HIDDEN
    model = nn.Sequential(*(NestedModel(dim=dim, num_children=2) for _ in range(2))).to(device)

    static_input = torch.eye(dim, device=device)
    static_target = torch.zeros_like(static_input)

    placements = _default_placements()
    with fully_shard_context(
        FsdpContext(
            device=device,
            runtime_policy=RuntimePolicy(
                use_symmetric_memory=use_symmetric_memory,
                prefetch_order=PrefetchOrder.RECORDED,
                autograd_completes_execution=False,
            ),
        )
    ) as context:
        for chunk in model:
            for layer in chunk.layers:
                fully_shard(layer, mesh=mesh, placements=placements)
            fully_shard(chunk, mesh=mesh, placements=placements)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.25, foreach=False)

    def train_iteration() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=False)
        losses = []
        for _ in range(2):
            output = model(static_input)
            loss = torch.nn.functional.mse_loss(output, static_target) / 2
            loss.backward()
            losses.append(loss.detach())
        if not context.is_recording_execution_order:
            assert context.execution_order_view.position == len(context.execution_order_view)
        context.post_backward()
        optimizer.step()
        return torch.stack(losses).sum()

    if use_symmetric_memory:
        # NCCL symmetric-memory window registration can fail when a rendezvous is the
        # first collective on a communicator, so establish it before capture.
        dist.barrier(group=mesh.get_group(0), device_ids=[device.index])

    source = context.execution_order
    model.eval()
    with torch.no_grad():
        model(static_input)
    model.train()
    assert context.is_recording_execution_order and len(source) == 0

    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())

    # Warmup
    with torch.cuda.stream(capture_stream):
        # See: https://docs.nvidia.com/dl-cuda-graph/troubleshooting/memory-issues.html#gradient-accumulator-cross-stream-memory-growth
        # Warm up on the same stream used for capture so autograd's accumulation
        # path does not create cross-stream gradient-memory growth.
        # The first warmup installs the reusable sharded gradient views; subsequent
        # iterations zero them in place for CUDA graph replay.
        for _ in range(3):
            train_iteration()

    events = tuple(source)
    full_view = context.execution_order_view
    assert full_view.position == 0
    assert {event.action for event in events} == {ExecutionAction.UNSHARD, ExecutionAction.RESHARD}
    assert [event.event_id for event in events] == list(range(len(events)))
    assert all(
        event.allocated_bytes is not None and event.unsharded_bytes is not None for event in events
    )
    with torch.cuda.stream(capture_stream):
        initial_parameters = [
            parameter.to_local().detach().clone() for parameter in model.parameters()
        ]
        reference_losses = []
        for step in range(5):
            reference_losses.append(train_iteration())
            if step == 0:
                model.eval()
                with torch.no_grad():
                    model(static_input)
                model.train()
                assert context.execution_order is source
                assert context.execution_order_view is full_view and full_view.position == 0
                assert tuple(source) == events
        reference_parameters = [
            parameter.to_local().detach().clone() for parameter in model.parameters()
        ]
        with torch.no_grad():
            for parameter, initial in zip(model.parameters(), initial_parameters):
                parameter.to_local().copy_(initial)

    # Capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        static_loss = train_iteration()
    torch.cuda.current_stream().wait_stream(capture_stream)

    # Replay
    losses = []
    for _ in range(5):
        graph.replay()
        # Each replay rewrites static_loss's fixed graph output storage; clone
        # keeps a per-replay GPU snapshot without the CPU sync from .item().
        losses.append(static_loss.clone())
    loss_values = torch.stack(losses).tolist()
    torch.testing.assert_close(torch.stack(losses), torch.stack(reference_losses))
    for parameter, reference in zip(model.parameters(), reference_parameters):
        torch.testing.assert_close(parameter.to_local(), reference)
    assert context.execution_order is source and tuple(source) == events
    assert context.execution_order_view is full_view and full_view.position == 0

    logger.info("CUDA graph replay losses: %s", loss_values)
    assert loss_values[-1] < loss_values[0], (
        "CUDA graph replay did not reduce the fixed-input loss: "
        f"first={loss_values[0]:.6f}, "
        f"last={loss_values[-1]:.6f}, trace={loss_values}"
    )


def test_local_accumulation_captures_full_iteration(distributed_setup):
    """Graph replay matches eager training with singleton-inner accumulation and outer overlap."""
    device = distributed_setup.device
    if distributed_setup.world_size < 2:
        pytest.skip("Requires a nontrivial outer data-parallel axis.")
    mesh = init_device_mesh(device.type, (distributed_setup.world_size, 1))
    placements = Placements(
        [0, 1], [Replicate(), Shard(0)], [Partial("avg"), Shard(0)], [Replicate(), Shard(0)]
    )
    torch.manual_seed(1234)
    model = NestedModel(dim=16, num_children=2).to(device)
    reference_model = NestedModel(dim=16, num_children=2).to(device)
    reference_model.load_state_dict(model.state_dict())
    for candidate in (model, reference_model):
        with fully_shard_context(
            FsdpContext(
                device=device, runtime_policy=RuntimePolicy(overlap_outer_communication=True)
            )
        ):
            for layer in candidate.layers:
                fully_shard(layer, mesh=mesh, placements=placements)
            fully_shard(candidate, mesh=mesh, placements=placements)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.25, foreach=False)
    reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=0.25, foreach=False)
    static_input = torch.eye(16, device=device)
    static_target = torch.zeros_like(static_input)

    def train_iteration(model, optimizer) -> torch.Tensor:
        optimizer.zero_grad(set_to_none=False)
        loss = torch.nn.functional.mse_loss(model(static_input), static_target)
        loss.backward()
        optimizer.step()
        return loss.detach()

    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    # Warm up on the capture stream to establish reusable gradient storage.
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            train_iteration(model, optimizer)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        static_loss = train_iteration(model, optimizer)
    torch.cuda.current_stream().wait_stream(capture_stream)

    # Capture records kernels without executing them; only warmup advanced the model.
    for _ in range(3):
        train_iteration(reference_model, reference_optimizer)

    losses = []
    reference_losses = []
    for _ in range(5):
        graph.replay()
        # Preserve each replay's loss before the next replay overwrites it.
        losses.append(static_loss.clone())
        reference_losses.append(train_iteration(reference_model, reference_optimizer))
    torch.testing.assert_close(torch.stack(losses), torch.stack(reference_losses))
    for parameter, reference_parameter in zip(model.parameters(), reference_model.parameters()):
        torch.testing.assert_close(parameter.to_local(), reference_parameter.to_local())


def test_weight_retention_eager_and_capture_match_baseline(distributed_setup):
    """Keep losses, gradients and updates correct through retained eager and graph reuse."""
    device, world_size = distributed_setup.device, distributed_setup.world_size
    if world_size < 2:
        pytest.skip("Weight-retention coverage requires a nontrivial data-parallel axis.")
    mesh = init_device_mesh(device.type, (world_size,))
    sharded = _default_placements()
    replicated = Placements([0], [Replicate()], [Replicate()], [Replicate()])
    torch.manual_seed(1234)
    initial = NestedModel(dim=8, num_children=2).to(device)
    models, contexts, optimizers = [], [], []
    for retained in (False, True, True):
        model = deepcopy(initial)
        with fully_shard_context(
            FsdpContext(
                device=device,
                runtime_policy=RuntimePolicy(
                    prefetch_order=(
                        PrefetchOrder.RECORDED if retained else PrefetchOrder.REGISTRATION
                    ),
                    autograd_completes_execution=False,
                ),
                gradient_finalization_mode=GradientFinalizationMode.REDUCE_OUTER,
            )
        ) as context:
            fully_shard(model.layers[0], mesh=mesh, placements=sharded)
            # This child's full weights alias persistent storage instead of a gather buffer.
            fully_shard(model.layers[1], mesh=mesh, placements=replicated)
            fully_shard(model, mesh=mesh, placements=sharded)
        models.append(model)
        contexts.append(context)
        optimizers.append(torch.optim.SGD(model.parameters(), lr=0.01, foreach=False))
    inputs = torch.ones(2, 8, device=device)

    def step(index):
        model, context, optimizer = models[index], contexts[index], optimizers[index]
        optimizer.zero_grad(set_to_none=False)
        losses = []
        for microbatch in range(2):
            mode = (
                GradientFinalizationMode.SKIP_OUTER
                if microbatch == 0
                else GradientFinalizationMode.REDUCE_OUTER
            )
            with context.finalize_gradients(mode):
                loss = model(inputs).square().mean()
                loss.backward()
                losses.append(loss.detach())
        was_recording = context.is_recording_execution_order
        context.post_backward()
        if was_recording:
            context.retain_mask = build_retain_mask(
                context.execution_order, dist.group.WORLD, device=device
            )
        context.finish_grad_sync()
        optimizer.step()
        return torch.stack(losses).sum()

    stream = torch.cuda.Stream(device)
    stream.wait_stream(torch.cuda.current_stream(device))
    graph = torch.cuda.CUDAGraph()
    try:
        for iteration in range(6):
            inputs.fill_((iteration + 1) * (distributed_setup.rank + 1) / world_size)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                expected, eager = step(0), step(1)
                # Record once, then warm the planned prefetch/reuse path before capture.
                if iteration < 2:
                    actual = step(2)
                else:
                    if iteration == 2:
                        with torch.cuda.graph(graph, stream=stream):
                            actual = step(2)
                    graph.replay()
            torch.cuda.current_stream(device).wait_stream(stream)
            torch.testing.assert_close(eager, expected)
            torch.testing.assert_close(actual, expected)
            for model in models[1:]:
                for parameter, reference in zip(model.parameters(), models[0].parameters()):
                    torch.testing.assert_close(
                        uneven_dtensor_to_full_tensor(parameter.grad),
                        uneven_dtensor_to_full_tensor(reference.grad),
                    )
                    torch.testing.assert_close(
                        uneven_dtensor_to_full_tensor(parameter),
                        uneven_dtensor_to_full_tensor(reference),
                    )
            for model, context in zip(models[1:], contexts[1:]):
                assert any(context.retain_mask.values())
                for module in (*model.layers, model):
                    for group in module.parameter_groups:
                        assert not group._unshard_events
                        if group.unsharded_storage_bytes:
                            assert not group._unsharded_model_weight.is_allocated
    finally:
        torch.cuda.synchronize(device)
        graph.reset()
