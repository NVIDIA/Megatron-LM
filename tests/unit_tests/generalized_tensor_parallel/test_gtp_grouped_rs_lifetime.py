# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Grouped reduce-scatter must release eager inputs before graph-only replay."""

import pytest
import torch

from megatron.core.tensor_parallel import generalized_tensor_parallelism as gtp
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (
    _torchrun_dist_init,
    reset_gtp_globals,
)


@pytest.mark.parametrize("async_op", [False, True])
def test_grouped_rs_releases_inputs_after_wait(monkeypatch, async_op):
    """Check input lifetime without issuing another collective to flush NCCL's shelf."""
    group = torch.distributed.group.WORLD
    if group.size() < 2:
        pytest.skip("requires at least two GPUs")
    monkeypatch.setattr(gtp.GTP_CONFIG, "reduce_scatter_with_fp32_accumulation", False)
    monkeypatch.setattr(gtp.GTP_CONFIG, "calculate_per_token_loss", False)
    monkeypatch.setattr(gtp.GTP_CONFIG, "check_param_states", False)
    module = torch.nn.Module()
    shape = (64 * group.size(), 64)
    for name in ("weight0", "weight1"):
        module.register_parameter(
            name, torch.nn.Parameter(torch.zeros(shape, dtype=torch.bfloat16, device="cuda"))
        )
    gtp.wrap_module_params_gtp(module, ["weight0", "weight1"], group, is_grouped=True)
    weight = module.weight0

    def reduce(packed):
        outputs, handle, release_bufs = weight._reduce_scatter(
            list(packed.unbind()), async_op=async_op
        )
        if handle is not None:
            handle.wait()
        # Match the caller dropping its input references once the RS has completed.
        del release_bufs
        return outputs

    # Use views into one packed allocation, as TE's grouped wgrad GEMM does.
    packed = torch.full((2, *shape), group.rank() + 1, dtype=torch.bfloat16, device="cuda")
    address = packed.data_ptr()
    outputs = reduce(packed)
    torch.cuda.synchronize()
    expected = (group.size() + 1) / 2
    for output in outputs:
        torch.testing.assert_close(output, torch.full_like(output, expected), atol=0, rtol=0)
    del packed
    torch.cuda.synchronize()
    # Inspect storage, not Python weakrefs: a Work can retain TensorImpls in C++.
    # Do not enqueue more collectives here; that can release an orphan Work's shelf
    # and mask the retention that persists when training switches to graph replay.
    for segment in torch.cuda.memory_snapshot():
        for block in segment["blocks"]:
            if block["address"] == address:
                assert block["state"] != "active_allocated", "wait retained grouped RS input"

    # Captured inputs must remain valid across replay, despite releasing eager inputs.
    offset = torch.zeros((), dtype=torch.bfloat16, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        packed = torch.empty((2, *shape), dtype=torch.bfloat16, device="cuda")
        packed.fill_(group.rank() + 1).add_(offset)
        outputs = reduce(packed)
        del packed
    for value in (0, 4, 8):
        offset.fill_(value)
        graph.replay()
        torch.cuda.synchronize()
        for output in outputs:
            torch.testing.assert_close(
                output, torch.full_like(output, expected + value), atol=0, rtol=0
            )
