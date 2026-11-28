# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Execution-context semantics for boundary observers, independent of devices."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, asdict, replace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from megatron.core.determinism.context import (
    DeterminismContext,
    determinism_context,
    get_determinism_context,
)
from megatron.core.determinism.trace import RankLocalTrace, TraceConfig


def test_default_context_is_unspecified_and_immutable():
    context = get_determinism_context()
    assert context == DeterminismContext(None, None, "unspecified")
    with pytest.raises(FrozenInstanceError):
        context.iteration = 1


def test_nested_context_restores_after_exception():
    original = get_determinism_context()
    outer = DeterminismContext(7, 2, "forward")
    with determinism_context(outer) as active:
        assert active is outer
        assert get_determinism_context() is outer
        with pytest.raises(RuntimeError, match="boundary failed"):
            with determinism_context(replace(outer, microbatch=3)):
                assert get_determinism_context() == DeterminismContext(7, 3, "forward")
                raise RuntimeError("boundary failed")
        assert get_determinism_context() is outer
        with determinism_context(DeterminismContext()):
            assert get_determinism_context().iteration is None
            assert get_determinism_context().microbatch is None
            assert get_determinism_context().phase == "unspecified"
    assert get_determinism_context() is original


def test_saved_forward_context_survives_recompute_and_interleaved_backward():
    snapshots = []
    original = get_determinism_context()
    for microbatch in (0, 1):
        with determinism_context(DeterminismContext(7, microbatch, "forward")):
            snapshots.append(get_determinism_context())

    observed = []
    with determinism_context(DeterminismContext(8, 3, "forward")):
        unrelated = get_determinism_context()
        for snapshot in reversed(snapshots):
            with determinism_context(snapshot):
                observed.append(get_determinism_context())
            with determinism_context(replace(snapshot, phase="backward")):
                observed.append(get_determinism_context())
            assert get_determinism_context() is unrelated

    assert observed == [
        DeterminismContext(7, 1, "forward"),
        DeterminismContext(7, 1, "backward"),
        DeterminismContext(7, 0, "forward"),
        DeterminismContext(7, 0, "backward"),
    ]
    assert [snapshot.phase for snapshot in snapshots] == ["forward", "forward"]
    assert get_determinism_context() is original


@pytest.mark.parametrize("recompute", [False, True])
def test_tensor_backward_hooks_bind_forward_snapshot(recompute):
    observed = []
    recomputations = []
    losses = []
    inputs = []

    def layer(tensor, snapshot):
        with determinism_context(snapshot):
            recomputations.append(get_determinism_context())
            output = tensor * 2
            if output.requires_grad:
                saved = get_determinism_context()

                def backward(gradient):
                    with determinism_context(replace(saved, phase="backward")):
                        observed.append(get_determinism_context())
                    return gradient

                output.register_hook(backward)
            return output

    for microbatch in (0, 1):
        snapshot = DeterminismContext(7, microbatch, "forward")
        tensor = torch.ones(1, requires_grad=True)
        inputs.append(tensor)
        with determinism_context(snapshot):
            output = (
                checkpoint(layer, tensor, snapshot, use_reentrant=True)
                if recompute
                else layer(tensor, snapshot)
            )
        losses.append(output.sum())

    with determinism_context(DeterminismContext(8, 3, "forward")):
        unrelated = get_determinism_context()
        for loss in reversed(losses):
            loss.backward()
            assert get_determinism_context() is unrelated

    assert observed == [DeterminismContext(7, 1, "backward"), DeterminismContext(7, 0, "backward")]
    expected_forward = [DeterminismContext(7, 0, "forward"), DeterminismContext(7, 1, "forward")]
    assert recomputations == expected_forward + (
        list(reversed(expected_forward)) if recompute else []
    )
    for tensor in inputs:
        torch.testing.assert_close(tensor.grad, torch.full_like(tensor, 2))


def test_context_preserves_recorder_selection_and_missing_iteration(tmp_path):
    with RankLocalTrace(
        TraceConfig(output_dir=tmp_path, rank=0, mode="metadata", iteration_spec="2-4")
    ) as trace:
        missing = asdict(get_determinism_context())
        trace.record_event("lifecycle", **missing)
        trace.record_tensor(
            "boundary", torch.ones(1), **missing, metadata={"missing_context": True}
        )
        with pytest.raises(ValueError, match="iteration is required for value-bearing captures"):
            trace.record_tensor("boundary", torch.ones(1), **missing, mode="full")
        with pytest.raises(ValueError, match="iteration is required for value-bearing captures"):
            trace.record_scalar("scalar", torch.tensor(1.0), **missing)
        with determinism_context(DeterminismContext(1, 0, "forward")):
            trace.record_tensor("boundary", torch.ones(1), **asdict(get_determinism_context()))
        for _ in range(2):
            with determinism_context(DeterminismContext(2, 0, "forward")):
                trace.record_tensor("boundary", torch.ones(1), **asdict(get_determinism_context()))

    records = [json.loads(line) for line in trace.output_path.read_text().splitlines()]
    assert [record["name"] for record in records] == [
        "lifecycle",
        "boundary",
        "boundary",
        "boundary",
    ]
    assert records[1]["iteration"] is None
    assert records[1]["metadata"] == {"missing_context": True}
    assert [record["occurrence"] for record in records[2:]] == [0, 1]
    assert [record["iteration"] for record in records[2:]] == [2, 2]


def test_threads_require_explicit_snapshot_binding():
    snapshot = DeterminismContext(7, 2, "forward")

    def worker():
        initial = get_determinism_context()
        with determinism_context(snapshot):
            bound = get_determinism_context()
        return initial, bound, get_determinism_context()

    with determinism_context(snapshot), ThreadPoolExecutor(max_workers=1) as executor:
        initial, bound, restored = executor.submit(worker).result()
        assert initial == restored == DeterminismContext()
        assert bound is snapshot
        assert get_determinism_context() is snapshot


def test_async_tasks_inherit_then_isolate_context():
    async def run():
        ready = [asyncio.Event(), asyncio.Event()]
        parent = DeterminismContext(7, None, "forward")

        async def worker(microbatch):
            assert get_determinism_context() is parent
            with determinism_context(replace(parent, microbatch=microbatch)):
                ready[microbatch].set()
                await ready[1 - microbatch].wait()
                assert get_determinism_context().microbatch == microbatch
            assert get_determinism_context() is parent

        with determinism_context(parent):
            await asyncio.gather(worker(0), worker(1))
            assert get_determinism_context() is parent

    original = get_determinism_context()
    asyncio.run(run())
    assert get_determinism_context() is original
