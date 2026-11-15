# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from megatron.core.full_cuda_graph import FullCudaGraphWrapper
from megatron.core.rerun_state_machine import RerunMode, RerunStateMachine
from megatron.core.tensor_parallel import generalized_tensor_parallelism as gtp
from megatron.core.tensor_parallel.gtp_api import HAVE_GTP
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (
    _make_gtp_linear,
    reset_gtp_globals,
)
from tests.unit_tests.test_utilities import Utils


def test_capture_does_not_borrow_eager_wgrad_storage(monkeypatch):
    """A graph must not keep a scratch address that eager work can later overwrite."""
    Utils.initialize_distributed()
    monkeypatch.setattr(gtp, "_wgrad_buf_pool", {})
    eager = gtp._wgrad_pool_get((1024,), torch.float32, "cuda")
    assert eager._from_gtp_wgrad_pool
    gtp._wgrad_pool_put(eager)
    result = torch.empty_like(eager)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        scratch = gtp._wgrad_pool_get(eager.shape, eager.dtype, eager.device)
        scratch.fill_(3)
        result.copy_(scratch)
        # Graph-owned scratch is never cached for eager reuse, whatever release path sees it.
        assert not scratch._from_gtp_wgrad_pool
        gtp._wgrad_pool_put(scratch)
    assert scratch.data_ptr() != eager.data_ptr()
    assert gtp._wgrad_pool_get(eager.shape, eager.dtype, eager.device) is eager
    assert not gtp._wgrad_buf_pool.get((eager.shape, eager.dtype))
    for _ in range(3):
        eager.fill_(9)
        graph.replay()
        torch.testing.assert_close(result, torch.full_like(result, 3))


@pytest.mark.skipif(not HAVE_GTP, reason="GTP requires TransformerEngine >= 2.19")
def test_full_iteration_fp32_accumulation_scratch_lifetime(monkeypatch):
    """Captured send scratch must outlive its RS-stream reader under allocation pressure.

    The RS stream still reads the compute-stream send buffer after _wait_reduce_scatter
    releases it; without last-reader tracking the graph pool hands that storage to the next
    compute-stream allocation. (The all-to-all scratch is allocated on the RS stream itself,
    so compute-stream pressure cannot alias it.)
    """
    Utils.initialize_distributed()
    group = torch.distributed.group.WORLD
    world_size = group.size()
    if world_size <= 2:
        pytest.skip("FP32-accumulation reduce-scatter requires more than two ranks")
    assert not gtp.uses_gtp_symm_pool(group, collective="rs")
    monkeypatch.setattr(gtp, "_wgrad_buf_pool", {})
    monkeypatch.setattr(gtp, "_FULL_ITERATION", True)
    monkeypatch.setattr(gtp.GTP_CONFIG, "async_reduction", True)
    monkeypatch.setattr(gtp.GTP_CONFIG, "reduce_scatter_with_fp32_accumulation", True)
    monkeypatch.setattr(gtp.GTP_CONFIG, "calculate_per_token_loss", False)
    monkeypatch.setattr("megatron.core.full_cuda_graph.get_all_rng_states", lambda: {})
    layer = _make_gtp_linear(128, 32 * world_size, group)
    gtp.classify_gtp_chains(layer)
    weight = layer.weight
    weight.main_grad = torch.zeros_like(weight)
    source = torch.empty(weight._unsharded_shape, dtype=torch.bfloat16, device="cuda")
    source.fill_(group.rank())
    calls = []

    def forward_backward_func(**_):
        weight.main_grad.zero_()
        buffer = weight.get_wgrad_tensor()
        buffer.copy_(source)
        weight.wgrad_reduce_scatter(buffer, async_op=True)
        del buffer
        # This list is populated only by the non-registered all-to-all/FP32-sum path.
        assert len(weight._rs_a2a_bufs) == 1
        calls.append(torch.cuda.is_current_stream_capturing())
        rs_stream = gtp.get_rs_stream(weight.chain_id, group)
        with torch.cuda.stream(rs_stream):
            torch.cuda._sleep(5_000_000)
        weight._wait_reduce_scatter(finalize_grad=True)
        assert weight._rs_a2a_bufs is None and weight._wgrad_input_bufs is None
        # Do not join the RS stream until after these overwrites. Incorrectly freed send
        # scratch can be recycled while the reduce-scatter is still reading it.
        pressure = [torch.empty_like(source) for _ in range(8)]
        for allocation in pressure:
            allocation.fill_(-100)
        torch.cuda.current_stream().wait_stream(rs_stream)
        return [{"loss": weight.main_grad}]

    wrapper = FullCudaGraphWrapper(forward_backward_func, cuda_graph_warmup_steps=1)
    wrapper.reset_cuda_graph()
    kwargs = dict(
        model=[torch.nn.Identity()],
        data_iterator=None,
        num_microbatches=1,
        seq_length=1,
        forward_only=False,
    )
    try:
        wrapper(**kwargs)
        wrapper(**kwargs)
        for step in range(10):
            source.fill_(group.rank() + step)
            result = wrapper(**kwargs)
            torch.cuda.synchronize()
            expected = step + (world_size - 1) / 2
            torch.testing.assert_close(
                result[0]["loss"], torch.full_like(weight.main_grad, expected), atol=0, rtol=0
            )
        assert calls == [False, True], "Replay must not repeat Python scratch allocation"
    finally:
        torch.cuda.synchronize()
        wrapper.reset_cuda_graph()


def _spy_on_stream_waits(monkeypatch):
    """Record ``(waiting stream, awaited stream)`` for every ``wait_stream`` call."""
    waits = []
    original = torch.cuda.Stream.wait_stream

    def wait_stream(self, stream):
        waits.append((self, stream))
        return original(self, stream)

    monkeypatch.setattr(torch.cuda.Stream, "wait_stream", wait_stream)
    return waits


def test_all_gather_join_waits_on_every_side_stream_when_eager(monkeypatch):
    """A forward-only step has no finalize_model_grads to join the all-gather streams."""
    Utils.initialize_distributed()
    side_streams = {"attn": torch.cuda.Stream(), "mlp": torch.cuda.Stream()}
    monkeypatch.setattr(gtp, "_AG_STREAMS", side_streams)
    waits = _spy_on_stream_waits(monkeypatch)

    gtp.wait_for_gtp_all_gather_on_current_stream()

    current = torch.cuda.current_stream()
    assert waits == [(current, stream) for stream in side_streams.values()]


def test_all_gather_join_in_capture_skips_streams_outside_the_capture(monkeypatch):
    """Joining a side stream that did no captured work would add a dependency on uncaptured
    work; leaving one that did unjoined fails the capture."""
    Utils.initialize_distributed()
    used, unused = torch.cuda.Stream(), torch.cuda.Stream()
    monkeypatch.setattr(gtp, "_AG_STREAMS", {"used": used, "unused": unused})
    source = torch.ones(1024, device="cuda")
    result = torch.empty_like(source)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        doubled = source * 2
        used.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(used):
            gathered = doubled + 1
        waits = _spy_on_stream_waits(monkeypatch)
        gtp.wait_for_gtp_all_gather_on_current_stream()
        monkeypatch.undo()
        result.copy_(gathered)

    assert [awaited for _, awaited in waits] == [used]
    for value in (1.0, 4.0):
        source.fill_(value)
        graph.replay()
        torch.testing.assert_close(result, torch.full_like(result, 2 * value + 1))


@pytest.mark.parametrize(
    "full_iteration,graphed,capturing,expected",
    [
        (False, False, False, True),
        (False, True, False, False),  # a partial graph bakes in one backward's schedule
        (False, False, True, False),
        (True, True, True, True),  # a whole-step graph holds every backward and its fence
        (True, True, False, True),
    ],
)
def test_wgrad_accumulation_eligibility_under_full_iteration(
    monkeypatch, full_iteration, graphed, capturing, expected
):
    monkeypatch.setattr(gtp, "_FULL_ITERATION", full_iteration)
    monkeypatch.setattr(gtp, "_chain_is_graphed", lambda chain_id: graphed)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    param = SimpleNamespace(_wgrad_accum_enabled=True, chain_id="chain")

    assert gtp.GTPShardedParam._can_accumulate_wgrad(param) is expected

    # Neither mode makes an opted-out param or a ring-slot param eligible.
    param._wgrad_accum_enabled = False
    assert not gtp.GTPShardedParam._can_accumulate_wgrad(param)
    param._wgrad_accum_enabled = True
    param._gtp_graph_wgrad_ring_slot = object()
    assert not gtp.GTPShardedParam._can_accumulate_wgrad(param)


@pytest.mark.parametrize(
    "full_iteration,capturing,fence_runs",
    [(False, False, True), (False, True, False), (True, True, True), (True, False, True)],
)
def test_grad_reduction_fence_runs_accumulation_in_full_iteration_capture(
    monkeypatch, full_iteration, capturing, fence_runs
):
    """Skipping the fence in a whole-step capture would drop a wgrad whose last-consume flush
    was mispredicted; a per-layer capture still leaves it to the eager fence."""
    Utils.initialize_distributed()
    calls = []
    monkeypatch.setattr(gtp, "_FULL_ITERATION", full_iteration)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    monkeypatch.setattr(gtp, "_AG_STREAMS", {})
    monkeypatch.setattr(gtp, "_RS_STREAMS", {})
    monkeypatch.setattr(
        gtp, "_collect_inflight_wgrad_reduce_scatters", lambda: calls.append("collect")
    )
    monkeypatch.setattr(gtp, "wait_async_comms", lambda *args, **kwargs: calls.append("wait"))
    monkeypatch.setattr(gtp, "_close_wgrad_accumulation_windows", lambda: calls.append("close"))

    gtp.wait_for_gtp_grad_reduction_on_current_stream()

    # In-flight reduce-scatters are collected before the drain and the windows closed after it.
    assert calls == (["collect", "wait", "close"] if fence_runs else ["wait"])


@pytest.mark.skipif(not HAVE_GTP, reason="GTP requires TransformerEngine >= 2.19")
def test_only_forward_only_capture_joins_all_gather_streams(monkeypatch):
    """A forward-only step never reaches finalize_model_grads, which joins them otherwise."""
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    # The wrapper keeps its graphs on the class.
    monkeypatch.setattr(FullCudaGraphWrapper, "curr_iteration", {'training': 0, 'validation': 0})
    monkeypatch.setattr(FullCudaGraphWrapper, "cuda_graph", {'training': None, 'validation': None})
    monkeypatch.setattr(FullCudaGraphWrapper, "result", {'training': None, 'validation': None})
    source = torch.ones(1024, device="cuda")
    wrapper = FullCudaGraphWrapper(lambda **_: [{"loss": source * 2}], cuda_graph_warmup_steps=0)

    def step(*, forward_only):
        return wrapper(
            model=[torch.nn.Identity()],
            data_iterator=None,
            num_microbatches=1,
            seq_length=1,
            forward_only=forward_only,
        )

    machine = RerunStateMachine(mode=RerunMode.DISABLED)
    try:
        with (
            patch("megatron.core.full_cuda_graph.get_all_rng_states", return_value={}),
            patch("megatron.core.full_cuda_graph.get_rerun_state_machine", return_value=machine),
            patch(
                "megatron.core.tensor_parallel.gtp_api.wait_for_gtp_all_gather_on_current_stream"
            ) as join,
        ):
            step(forward_only=False)
            assert FullCudaGraphWrapper.cuda_graph['training'] is not None
            join.assert_not_called()

            step(forward_only=True)
            assert FullCudaGraphWrapper.cuda_graph['validation'] is not None
            join.assert_called_once()

            # Replays run the captured join; the Python call happens at capture only.
            step(forward_only=True)
            join.assert_called_once()
            wrapper.reset_cuda_graph()
    finally:
        torch.cuda.synchronize()
        Utils.destroy_model_parallel()
