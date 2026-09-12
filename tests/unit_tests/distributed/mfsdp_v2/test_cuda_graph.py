# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CUDA graph tests for Megatron-FSDP."""

import logging

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Placements,
    fully_shard,
    fully_shard_context,
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


def _flat_placements() -> Placements:
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
    """A full training iteration should be CUDA-graphable, with and without symmetric memory.

    With ``use_symmetric_memory=True`` the FSDP collective path changes: unshard allocates the
    all-gather buffer from a symmetric-memory ``MemPool`` and calls
    ``symm_mem.rendezvous()``, and the backward reduce-scatter does the same for the
    partial-gradient buffer. Both allocation-from-pool and rendezvous are normally
    illegal inside CUDA-graph capture, so this confirms the warmup establishes reusable,
    already-rendezvoused buffers that capture and replay cleanly -- reproducing the
    non-symmetric-memory loss trajectory exactly.
    """
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if use_symmetric_memory and world_size < 2:
        pytest.skip("Symmetric memory requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    torch.manual_seed(1234)
    dim = _HIDDEN
    model = NestedModel(dim=dim, num_children=2).to(device)

    static_input = torch.eye(dim, device=device)
    static_target = torch.zeros_like(static_input)

    placements = _flat_placements()
    with fully_shard_context(device=device, use_symmetric_memory=use_symmetric_memory):
        for layer in model.layers:
            fully_shard(layer, mesh=mesh, placements=placements)
        fully_shard(model, mesh=mesh, placements=placements)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.25, foreach=False)

    def train_iteration() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=False)
        output = model(static_input)
        loss = torch.nn.functional.mse_loss(output, static_target)
        loss.backward()
        optimizer.step()
        return loss.detach()

    if use_symmetric_memory:
        # NCCL symmetric-memory window registration can fail when a rendezvous is the
        # first collective on a communicator, so establish it before capture.
        dist.barrier(group=mesh.get_group(0), device_ids=[device.index])

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

    logger.info("CUDA graph replay losses: %s", loss_values)
    assert loss_values[-1] < loss_values[0], (
        "CUDA graph replay did not reduce the fixed-input loss: "
        f"first={loss_values[0]:.6f}, "
        f"last={loss_values[-1]:.6f}, trace={loss_values}"
    )


def test_mixed_captured_and_eager_forward_schedule(distributed_setup):
    """Capture layers 0 and 2 while keeping layer 1 eager.

    The communication stream follows the forward schedule from
    ``runtime_schedule.md``: ``pre_0, pre_1, post_0, pre_2, post_1, post_2``.
    External events bridge eager communication with the captured layers, while
    ordinary events synchronize the eager middle layer.
    """
    device = distributed_setup.device
    default_stream = torch.cuda.current_stream(device)
    communication_stream = torch.cuda.Stream(device=device)

    pre_tokens = [torch.zeros(1, device=device) for _ in range(3)]
    forward_tokens = [torch.zeros(1, device=device) for _ in range(3)]
    post_tokens = [torch.zeros(1, device=device) for _ in range(3)]

    # Captured forwards need graph-external event nodes to synchronize with
    # eager communication. The eager middle forward uses ordinary events.
    pre_done = [
        torch.cuda.Event(external=True),
        torch.cuda.Event(),
        torch.cuda.Event(external=True),
    ]
    forward_done = [
        torch.cuda.Event(external=True),
        torch.cuda.Event(),
        torch.cuda.Event(external=True),
    ]
    forward_start = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    forward_end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    post_start = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    post_end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    sleep_cycles = 200_000_000

    def capture_forward(layer: int, pool: object) -> torch.cuda.CUDAGraph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            pre_done[layer].wait()
            torch.add(pre_tokens[layer], 1, out=forward_tokens[layer])
            forward_done[layer].record()
        return graph

    graph_pool = torch.cuda.graph_pool_handle()
    forward_0 = capture_forward(0, graph_pool)
    forward_2 = capture_forward(2, graph_pool)

    communication_schedule = []
    default_schedule = []
    timeline_start = torch.cuda.Event(enable_timing=True)
    timeline_start.record()

    with torch.cuda.stream(communication_stream):
        communication_stream.wait_event(timeline_start)
        pre_tokens[0].fill_(1)
        pre_done[0].record()
        communication_schedule.append("pre_0")
        pre_tokens[1].fill_(2)
        pre_done[1].record()
        communication_schedule.append("pre_1")

    default_stream.wait_event(pre_done[0])
    forward_start[0].record()
    forward_0.replay()
    forward_end[0].record()
    default_schedule.append("forward_0")

    with torch.cuda.stream(communication_stream):
        communication_stream.wait_event(forward_done[0])
        post_start[0].record()
        torch.add(forward_tokens[0], 1, out=post_tokens[0])
        torch.cuda._sleep(sleep_cycles)
        post_end[0].record()
        communication_schedule.append("post_0")
        pre_tokens[2].fill_(3)
        pre_done[2].record()
        communication_schedule.append("pre_2")

    default_stream.wait_event(pre_done[1])
    forward_start[1].record()
    torch.add(pre_tokens[1], 1, out=forward_tokens[1])
    torch.cuda._sleep(sleep_cycles)
    forward_end[1].record()
    forward_done[1].record()
    default_schedule.append("forward_1")

    with torch.cuda.stream(communication_stream):
        communication_stream.wait_event(forward_done[1])
        post_start[1].record()
        torch.add(forward_tokens[1], 1, out=post_tokens[1])
        torch.cuda._sleep(sleep_cycles)
        post_end[1].record()
        communication_schedule.append("post_1")

    default_stream.wait_event(pre_done[2])
    forward_start[2].record()
    forward_2.replay()
    torch.cuda._sleep(sleep_cycles)
    forward_end[2].record()
    default_schedule.append("forward_2")

    with torch.cuda.stream(communication_stream):
        communication_stream.wait_event(forward_done[2])
        post_start[2].record()
        torch.add(forward_tokens[2], 1, out=post_tokens[2])
        torch.cuda._sleep(sleep_cycles)
        post_end[2].record()
        communication_schedule.append("post_2")

    torch.cuda.synchronize(device)

    assert communication_schedule == ["pre_0", "pre_1", "post_0", "pre_2", "post_1", "post_2"]
    assert default_schedule == ["forward_0", "forward_1", "forward_2"]
    torch.testing.assert_close(
        torch.cat(forward_tokens), torch.tensor([2.0, 3.0, 4.0], device=device)
    )
    torch.testing.assert_close(
        torch.cat(post_tokens), torch.tensor([3.0, 4.0, 5.0], device=device)
    )

    def timestamp(event: torch.cuda.Event) -> float:
        return timeline_start.elapsed_time(event)

    def assert_overlaps(
        first_start: torch.cuda.Event,
        first_end: torch.cuda.Event,
        second_start: torch.cuda.Event,
        second_end: torch.cuda.Event,
    ) -> None:
        overlap = min(timestamp(first_end), timestamp(second_end)) - max(
            timestamp(first_start), timestamp(second_start)
        )
        assert overlap > 0, f"expected overlap, got {overlap:.3f} ms"

    assert_overlaps(post_start[0], post_end[0], forward_start[1], forward_end[1])
    assert_overlaps(post_start[1], post_end[1], forward_start[2], forward_end[2])


def test_partially_captures_forward_with_graphed_callables(distributed_setup):
    """Capture compatible layers as shared-pool graph islands around an eager layer.

    ``make_graphed_callables`` is PyTorch's partial-network capture API. It
    captures the compatible callables in their execution order into separate
    graphs that share one pool; layer 1 remains eager.
    """
    device = distributed_setup.device
    default_stream = torch.cuda.current_stream(device)
    communication_stream = torch.cuda.Stream(device=device)

    pre_tokens = [torch.zeros(1, device=device) for _ in range(3)]
    post_tokens = [torch.zeros(1, device=device) for _ in range(3)]
    pre_done = [
        torch.cuda.Event(external=True),
        torch.cuda.Event(),
        torch.cuda.Event(external=True),
    ]
    forward_done = [
        torch.cuda.Event(external=True),
        torch.cuda.Event(),
        torch.cuda.Event(external=True),
    ]
    forward_start = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    forward_end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    post_start = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    post_end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    sleep_cycles = 200_000_000

    def captured_forward(layer: int):
        def forward(pre_token: torch.Tensor) -> torch.Tensor:
            pre_done[layer].wait()
            output = pre_token + 1
            forward_done[layer].record()
            return output

        return forward

    graph_pool = torch.cuda.graph_pool_handle()
    forward_0, forward_2 = torch.cuda.make_graphed_callables(
        (captured_forward(0), captured_forward(2)),
        ((pre_tokens[0],), (pre_tokens[2],)),
        num_warmup_iters=1,
        pool=graph_pool,
    )
    timeline_start = torch.cuda.Event(enable_timing=True)
    timeline_start.record()

    with torch.cuda.stream(communication_stream):
        communication_stream.wait_event(timeline_start)
        pre_tokens[0].fill_(1)
        pre_done[0].record()
        pre_tokens[1].fill_(2)
        pre_done[1].record()

    default_stream.wait_event(pre_done[0])
    forward_start[0].record()
    forward_output_0 = forward_0(pre_tokens[0])
    forward_end[0].record()

    with torch.cuda.stream(communication_stream):
        communication_stream.wait_event(forward_done[0])
        post_start[0].record()
        torch.add(forward_output_0, 1, out=post_tokens[0])
        torch.cuda._sleep(sleep_cycles)
        post_end[0].record()
        pre_tokens[2].fill_(3)
        pre_done[2].record()

    default_stream.wait_event(pre_done[1])
    forward_start[1].record()
    forward_output_1 = pre_tokens[1] + 1
    torch.cuda._sleep(sleep_cycles)
    forward_end[1].record()
    forward_done[1].record()

    with torch.cuda.stream(communication_stream):
        communication_stream.wait_event(forward_done[1])
        post_start[1].record()
        torch.add(forward_output_1, 1, out=post_tokens[1])
        torch.cuda._sleep(sleep_cycles)
        post_end[1].record()

    default_stream.wait_event(pre_done[2])
    forward_start[2].record()
    forward_output_2 = forward_2(pre_tokens[2])
    torch.cuda._sleep(sleep_cycles)
    forward_end[2].record()

    with torch.cuda.stream(communication_stream):
        communication_stream.wait_event(forward_done[2])
        torch.add(forward_output_2, 1, out=post_tokens[2])

    torch.cuda.synchronize(device)

    torch.testing.assert_close(
        torch.cat(post_tokens), torch.tensor([3.0, 4.0, 5.0], device=device)
    )

    def timestamp(event: torch.cuda.Event) -> float:
        return timeline_start.elapsed_time(event)

    def assert_overlaps(
        first_start: torch.cuda.Event,
        first_end: torch.cuda.Event,
        second_start: torch.cuda.Event,
        second_end: torch.cuda.Event,
    ) -> None:
        overlap = min(timestamp(first_end), timestamp(second_end)) - max(
            timestamp(first_start), timestamp(second_start)
        )
        assert overlap > 0, f"expected overlap, got {overlap:.3f} ms"

    assert_overlaps(post_start[0], post_end[0], forward_start[1], forward_end[1])
    assert_overlaps(post_start[1], post_end[1], forward_start[2], forward_end[2])
