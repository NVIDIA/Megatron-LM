# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Collective retirement decisions and identity-safe replay of gradient inputs."""

from weakref import ref

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    FsdpContext,
    Placements,
    PrefetchOrder,
    RuntimePolicy,
    fully_shard,
    fully_shard_context,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.execution_order import (
    ExecutionEvent,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.phase import Phase


def _context(setup, *, group=True):
    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    model = nn.Identity()
    with fully_shard_context(
        FsdpContext(
            device=setup.device,
            prefetch_order=PrefetchOrder.RECORDED,
            prefetch_process_group=mesh.get_group(0) if group else None,
            runtime_policy=RuntimePolicy(caller_managed_grad_sync=True),
        )
    ) as context:
        fully_shard(
            model,
            mesh=mesh,
            placements=Placements(
                dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)]
            ),
        )
    return model, context, mesh


@pytest.mark.parametrize("case", ["latest", "missing", "already-drained", "no-group"])
def test_retirement_consensus_preserves_producers(distributed_setup, monkeypatch, case):
    """Agree on latest release, omit missing votes, and never free a replacement batch."""
    setup = distributed_setup
    if setup.world_size < 2:
        pytest.skip("Retirement consensus requires multiple ranks.")
    model, context, _ = _context(setup, group=case != "no-group")
    stream = torch.cuda.Stream(setup.device)
    stream.wait_stream(torch.cuda.current_stream(setup.device))
    with torch.cuda.stream(stream):
        first = torch.empty(4096, device=setup.device)
        second = torch.empty_like(first)

    def forward():
        model(first)

    def submit(result, value):
        source = torch.full_like(result, value)
        context.inner_reduction_stream.wait_stream(stream)
        with torch.cuda.stream(context.inner_reduction_stream):
            torch.cuda._sleep(100_000)
            result.copy_(source)
        context._retain_grad_buffers([source], stream)
        return ref(source), context._pending_grad_buffers[-1].done

    with monkeypatch.context() as recording:
        first_done = None

        def candidate_ready(event):
            # The real GPU copies complete before this conservative query model.
            # Only producer0 gets a local forward candidate; producer1 stays live.
            late = setup.rank == setup.world_size - 1
            if event is not first_done or (case == "missing" and late):
                return False
            return len(context.execution_order) >= (2 if late else 1)

        recording.setattr(torch.cuda.Event, "query", candidate_ready)
        with torch.cuda.stream(stream):
            forward()
            first_ref, first_done = submit(first, 7.0)
            first_done.synchronize()
            forward()
            second_ref, second_done = submit(second, 9.0)
            second_done.synchronize()
            forward()
            local_candidates = dict(context._grad_buffer_release_plan)
            context.finish_grad_sync()
    late = setup.rank == setup.world_size - 1
    expected_local = {} if case == "missing" and late else {2 if late else 1: (0,)}
    assert local_candidates == expected_local
    expected = {} if case in {"missing", "no-group"} else {2: (0,)}
    assert context._grad_buffer_release_plan == expected
    assert context._recorded_grad_buffer_positions == [0, 1]
    assert first_ref() is None and second_ref() is None

    def forbidden(*args, **kwargs):
        raise AssertionError("Replay must not query completion or run consensus collectives.")

    monkeypatch.setattr(torch.cuda.Event, "query", forbidden)
    monkeypatch.setattr(dist, "all_reduce", forbidden)

    def replay():
        forward()
        first_ref, _ = submit(first, 7.0)
        assert context._pending_grad_buffers[-1].position == 0
        forward()
        assert first_ref() is not None
        second_ref, _ = submit(second, 9.0)
        assert context._pending_grad_buffers[-1].position == 1
        if case == "already-drained":
            context._release_grad_buffers(keep=1)
            assert first_ref() is None
        forward()
        if expected:
            assert first_ref() is None
        else:
            assert first_ref() is not None
        # Producer0 may be absent, but the next FIFO entry must not inherit its release.
        assert second_ref() is not None
        context.finish_grad_sync()
        assert not context._pending_grad_buffers
        assert first_ref() is None and second_ref() is None
        return torch.empty_like(first).fill_(-9.0)

    with torch.cuda.stream(stream):
        replacement = replay()
    torch.cuda.current_stream(setup.device).wait_stream(stream)
    torch.testing.assert_close(first, torch.full_like(first, 7.0), rtol=0, atol=0)
    torch.testing.assert_close(second, torch.full_like(second, 9.0), rtol=0, atol=0)
    stream.wait_stream(torch.cuda.current_stream(setup.device))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        replacement = replay()
    for _ in range(3):
        with torch.cuda.stream(stream):
            graph.replay()
        torch.cuda.current_stream(setup.device).wait_stream(stream)
        torch.testing.assert_close(first, torch.full_like(first, 7.0), rtol=0, atol=0)
        torch.testing.assert_close(second, torch.full_like(second, 9.0), rtol=0, atol=0)
        torch.testing.assert_close(replacement, torch.full_like(replacement, -9.0), rtol=0, atol=0)
        assert context._grad_buffer_release_plan == expected
        stream.wait_stream(torch.cuda.current_stream(setup.device))


@pytest.mark.parametrize("difference", ["trace", "producer"])
def test_retirement_consensus_rejects_rank_divergence(distributed_setup, difference):
    """Every rank rejects different nonempty traces or producer origins before replay."""
    setup = distributed_setup
    if setup.world_size < 2:
        pytest.skip("Retirement consensus requires multiple ranks.")
    model, context, mesh = _context(setup)
    late = setup.rank == setup.world_size - 1
    context.execution_order.append(ExecutionEvent(model, Phase.FORWARD))
    source = torch.ones(1, device=setup.device)
    stream = context.current_stream()
    context.inner_reduction_stream.wait_stream(stream)
    if difference != "producer" or not late:
        context._retain_grad_buffers([source], stream)
    phase = Phase.BACKWARD if difference == "trace" and late else Phase.FORWARD
    context.execution_order.append(ExecutionEvent(model, phase))
    if difference == "producer" and late:
        context._retain_grad_buffers([source], stream)
    with pytest.raises(
        RuntimeError, match="events or gradient-input producers differ across ranks"
    ):
        context.finish_grad_sync()
    assert not context._pending_grad_buffers
    # A collective mismatch check must fail on all ranks, not strand peers in consensus.
    dist.barrier(group=mesh.get_group(0))
