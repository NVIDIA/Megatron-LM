# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CUDA stream and observer-validity checks for the generic trace primitives."""

import os

import pytest
import torch

from megatron.core.determinism.trace import RankLocalTrace, TraceConfig
from tools.determinism.trace_comparison import compare_traces, load_trace

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
    pytest.mark.launch_on_gb200,
]


@pytest.fixture
def device():
    """Use the worker's local GPU when launched through torch.distributed.run."""
    index = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(index)
    return torch.device("cuda", index)


@pytest.mark.parametrize("mode", ["summary", "sampled", "full"])
def test_capture_on_side_stream_flush_on_another_stream(tmp_path, device, mode):
    producer = torch.cuda.Stream(device=device)
    consumer = torch.cuda.Stream(device=device)
    original = torch.arange(4096, device=device, dtype=torch.float32).reshape(64, 64).t()
    producer.wait_stream(torch.cuda.current_stream(device))
    with RankLocalTrace(TraceConfig(output_dir=tmp_path / "left", rank=0, mode=mode)) as trace:
        with torch.cuda.stream(producer):
            mutable = original.clone()
            # Keep capture pending long enough to expose a missing cross-stream
            # dependency. No device-wide synchronization before flush.
            torch.cuda._sleep(2_000_000)
            trace.record_tensor("input", mutable, iteration=1)
            mutable.add_(100)
        with torch.cuda.stream(consumer):
            trace.flush()
    with RankLocalTrace(TraceConfig(output_dir=tmp_path / "right", rank=0, mode=mode)) as trace:
        trace.record_tensor("input", original, iteration=1)
    assert compare_traces(tmp_path / "left", tmp_path / "right")["match"]
    producer.synchronize()


@pytest.mark.parametrize("mode", ["summary", "sampled", "full"])
def test_mixed_producer_streams(tmp_path, device, mode):
    streams = [torch.cuda.Stream(device=device) for _ in range(2)]
    with RankLocalTrace(TraceConfig(output_dir=tmp_path / "left", rank=0, mode=mode)) as trace:
        for index, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                tensor = torch.full((1024,), float(index), device=device)
                trace.record_tensor(str(index), tensor, iteration=1)
    with RankLocalTrace(TraceConfig(output_dir=tmp_path / "right", rank=0, mode=mode)) as trace:
        for index in range(2):
            trace.record_tensor(
                str(index), torch.full((1024,), float(index), device=device), iteration=1
            )
    assert compare_traces(tmp_path / "left", tmp_path / "right")["match"]


@pytest.mark.parametrize("mode", ["metadata", "summary", "sampled", "full"])
def test_observer_preserves_equality_and_injected_divergence(tmp_path, device, mode):
    """A small validity gate, not a claim about application-scale observer cost."""

    def run(path, observed, perturb):
        state = torch.arange(4096, dtype=torch.float32, device=device)
        trace = (
            RankLocalTrace(TraceConfig(output_dir=path, rank=0, mode=mode)) if observed else None
        )
        try:
            for iteration in range(1, 4):
                if perturb and iteration == 2:
                    state[0].add_(1)
                state.mul_(0.5).add_(0.25)
                if trace is not None:
                    trace.record_tensor("state", state, iteration=iteration)
                    trace.flush()
            return state.cpu()
        finally:
            if trace is not None:
                trace.close()

    baseline = run(tmp_path / "off", False, False)
    baseline_changed = run(tmp_path / "off_changed", False, True)
    observed = run(tmp_path / "on", True, False)
    observed_repeat = run(tmp_path / "repeat", True, False)
    observed_changed = run(tmp_path / "on_changed", True, True)
    assert torch.equal(baseline, observed)
    assert torch.equal(observed, observed_repeat)
    assert torch.equal(baseline_changed, observed_changed)
    assert not torch.equal(baseline, baseline_changed)
    assert compare_traces(tmp_path / "on", tmp_path / "repeat")["match"]
    report = compare_traces(tmp_path / "on", tmp_path / "on_changed")
    if mode == "metadata":
        assert report["match"]  # Structure-only evidence must not claim to detect value changes.
    else:
        assert report["rank_results"]["0"]["first_divergence"]["key"]["iteration"] == 2


def test_metadata_cuda_graph_replay_does_not_reexecute_python(tmp_path, device):
    tensor = torch.ones(8, device=device)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    graph = torch.cuda.CUDAGraph()
    with RankLocalTrace(TraceConfig(output_dir=tmp_path, rank=0)) as trace:
        with torch.cuda.graph(graph, stream=stream):
            tensor.add_(1)
            trace.record_tensor("capture_only", tensor, iteration=1)
        graph.replay()
        graph.replay()
    assert len(load_trace(tmp_path)) == 1


@pytest.mark.parametrize("mode", ["summary", "sampled", "full"])
def test_value_capture_is_rejected_inside_cuda_graph(tmp_path, device, mode):
    tensor = torch.ones(8, device=device)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    graph = torch.cuda.CUDAGraph()
    with RankLocalTrace(TraceConfig(output_dir=tmp_path, rank=0, mode=mode)) as trace:
        with torch.cuda.graph(graph, stream=stream):
            tensor.add_(1)
            with pytest.raises(RuntimeError, match="not CUDA-graph-capture safe"):
                trace.record_tensor("input", tensor, iteration=1)
        trace.record_event("after_capture", iteration=1)
    assert load_trace(tmp_path)[0]["sequence"] == 0
