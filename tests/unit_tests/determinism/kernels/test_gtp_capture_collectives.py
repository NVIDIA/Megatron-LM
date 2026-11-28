# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GTP capture dispatch preserves repeated gathers, reductions and scratch ownership."""

import pytest
import torch

from megatron.core.tensor_parallel import generalized_tensor_parallelism as gtp
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (
    _torchrun_dist_init,
    reset_gtp_globals,
)


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("full_iteration,padded", [(False, False), (True, False), (True, True)])
def test_capture_collectives_replay_bit_exact(monkeypatch, grouped, full_iteration, padded):
    """Only whole-step captures change dispatch; all modes read current inputs exactly."""
    group = torch.distributed.group.WORLD
    if not gtp.HAVE_TE or group.size() < 2:
        pytest.skip("requires TransformerEngine GTP and at least two ranks")
    monkeypatch.setattr(gtp, "_FULL_ITERATION", full_iteration)
    monkeypatch.setattr(gtp, "_AG_STREAMS", {})
    monkeypatch.setattr(gtp, "_RS_STREAMS", {})
    monkeypatch.setattr(gtp.GTP_CONFIG, "check_param_states", False)
    monkeypatch.setattr(gtp.GTP_CONFIG, "calculate_per_token_loss", False)
    monkeypatch.setattr(gtp.GTP_CONFIG, "reduce_scatter_with_fp32_accumulation", False)
    model = torch.nn.Module()
    shape = (32 * group.size() - int(padded), 32)
    names = [f"weight{i}" for i in range(2 if grouped else 1)]
    for name in names:
        model.register_parameter(
            name, torch.nn.Parameter(torch.zeros(shape, device="cuda", dtype=torch.bfloat16))
        )
    gtp.wrap_module_params_gtp(model, names, group, is_grouped=grouped)
    gtp.classify_gtp_remat_chains(model, cuda_graph_impl="full_iteration")
    monkeypatch.setattr(gtp, "_FULL_ITERATION", full_iteration)
    params = list(model.parameters())
    leader = params[0]
    assert all((param.pad_length > 0) == padded for param in params)
    for param in params:
        param.main_grad = torch.zeros_like(param)
    ag_stream = gtp.get_ag_stream(leader.chain_id, group)
    rs_stream = gtp.get_rs_stream(leader.chain_id, group)
    assert ag_stream.priority == rs_stream.priority == torch.cuda.Stream.priority_range()[1]
    offset = torch.zeros((), device="cuda", dtype=torch.bfloat16)
    observations = []
    rs_observations = []
    gather_name = "grouped_gather_along_first_dim" if grouped else "gather_along_first_dim"
    gather = getattr(gtp, gather_name)

    def observed_gather(*args, **kwargs):
        observations.append((torch.cuda.is_current_stream_capturing(), kwargs["async_op"]))
        assert torch.cuda.current_stream() == ag_stream
        return gather(*args, **kwargs)

    monkeypatch.setattr(gtp, gather_name, observed_gather)
    if grouped:
        coalescing_manager = torch.distributed._coalescing_manager

        def observed_manager(*args, **kwargs):
            if torch.cuda.current_stream() == rs_stream:
                rs_observations.append(
                    (torch.cuda.is_current_stream_capturing(), kwargs["async_ops"])
                )
                if padded:
                    torch.cuda._sleep(20_000_000)
            return coalescing_manager(*args, **kwargs)

        monkeypatch.setattr(torch.distributed, "_coalescing_manager", observed_manager)
    else:
        reduce_scatter = gtp.reduce_scatter_along_first_dim

        def observed_reduce_scatter(*args, **kwargs):
            assert torch.cuda.current_stream() == rs_stream
            rs_observations.append((torch.cuda.is_current_stream_capturing(), kwargs["async_op"]))
            if padded:
                torch.cuda._sleep(20_000_000)
            return reduce_scatter(*args, **kwargs)

        monkeypatch.setattr(gtp, "reduce_scatter_along_first_dim", observed_reduce_scatter)

    def step():
        for param in params:
            param.main_grad.zero_()
        results = []
        # Repeat the same weights to exercise ticket reuse and side-stream handoffs.
        for use in range(3):
            with torch.no_grad():
                for index, param in enumerate(params):
                    param.fill_(group.rank() + 8 * index + use).add_(offset)
            gathered, handle = leader._all_gather_weight(async_op=True, fwd=True)
            handle.wait()
            torch.cuda.current_stream().wait_event(leader.ag_event)
            tensors = gathered if grouped else [gathered]
            results.extend(tensor.clone() for tensor in tensors)
            packed = torch.empty((len(params), *shape), device="cuda", dtype=torch.bfloat16)
            for index in range(len(params)):
                packed[index].fill_(group.rank() + 8 * index + use).add_(offset)
            leader.wgrad_reduce_scatter(list(packed.unbind()), async_op=True)
            del packed
            # The RS stream is delayed above. Overwrite matching allocator blocks before
            # joining it: a padded send copy must not be recycled for these allocations.
            pressure = [
                torch.empty(
                    (param.shape[0] * group.size(), shape[1]), device="cuda", dtype=torch.bfloat16
                )
                for param in params
            ]
            for tensor in pressure:
                tensor.fill_(-100)
            assert leader._wait_reduce_scatter(finalize_grad=True)
            torch.cuda.current_stream().wait_event(leader.rs_event)
            # Reuse graph-pool space after releasing the RS inputs.
            pressure = torch.empty((len(params), *shape), device="cuda", dtype=torch.bfloat16)
            pressure.fill_(-100)
        # Join side-stream allocator bookkeeping, as the full-iteration fence does.
        gtp.wait_for_gtp_grad_reduction_on_current_stream()
        results.extend(param.main_grad.clone() for param in params)
        return results

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(stream):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        for calls in (observations, rs_observations):
            assert calls and all(
                not capturing and asynchronous for capturing, asynchronous in calls
            )
            calls.clear()
        with torch.cuda.graph(graph, stream=stream):
            results = step()
        for calls in (observations, rs_observations):
            assert len(calls) == 3
            assert all(
                capturing and asynchronous == (not full_iteration)
                for capturing, asynchronous in calls
            )
        for value in (0, 4, 8, 0):
            offset.fill_(value)
            graph.replay()
            torch.cuda.synchronize()
            for use in range(3):
                for index in range(len(params)):
                    actual = results[use * len(params) + index]
                    expected = torch.cat(
                        [
                            torch.full_like(params[index], rank + 8 * index + use + value)
                            for rank in range(group.size())
                        ]
                    )
                    assert torch.equal(actual, expected)
            for index, actual in enumerate(results[3 * len(params) :]):
                expected = 3 * ((group.size() - 1) / 2 + 8 * index + value) + 3
                expected_grad = torch.full_like(actual, expected)
                # Padding contributes zero, including the tail of the last rank's shard.
                valid_rows = max(0, min(actual.shape[0], shape[0] - group.rank() * actual.shape[0]))
                expected_grad[valid_rows:].zero_()
                assert torch.equal(actual, expected_grad)
    finally:
        torch.cuda.synchronize()
        graph.reset()
