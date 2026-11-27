# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Ownership and adjoint checks for GPU-generated GDN packing metadata."""

import pytest
import torch
from tests.unit_tests.test_utilities import Utils

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.context_parallel.gdn_layout import get_chunkwise_layout


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("capture", [False, True])
def test_transport_values_and_adjoint(size, capture):
    """Replay different document lengths through the same communication graph."""
    Utils.initialize_model_parallel(context_parallel_size=size)
    try:
        group = ProcessGroupCollection.use_mpu_process_groups().cp
        rank = group.rank()
        boundaries = (0, 8, 32, 64)
        cu = torch.tensor(boundaries, dtype=torch.int32, device="cuda")
        packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu)
        x = torch.empty(64 // size, 1, 1, device="cuda", requires_grad=True)
        weights = torch.empty_like(x)

        def run():
            transport, *_ = get_chunkwise_layout(x, packed, group, 4, use_cuda_graphs=True)
            y = transport.forward(x)
            restored = transport.inverse(y)
            extended = transport.extend(y)
            (y * weights).sum().backward()
            return y, restored, extended

        with torch.no_grad():
            x.fill_(1)
            weights.fill_(1)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
                x.grad = None
        torch.cuda.current_stream().wait_stream(stream)
        graph = None
        if capture:
            graph = torch.cuda.CUDAGraph()
            # Use a fresh buffer: no warmed-layout dependency is allowed.
            cu = cu.clone()
            packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu)
            with torch.cuda.graph(graph, stream=stream):
                y, restored, extended = run()
        for boundaries in ((0, 8, 32, 64), (0, 24, 40, 64)):
            source, expected, halo = [], [], []
            for start, stop in zip(boundaries[:-1], boundaries[1:]):
                width = (stop - start) // (2 * size)
                for chunk in (rank, 2 * size - 1 - rank):
                    source.extend(range(start + chunk * width, start + (chunk + 1) * width))
                left = start + rank * 2 * width
                right = left + 2 * width
                expected.extend(range(left, right))
                halo.extend(
                    [
                        position + 1 if position >= start else 0
                        for position in range(left - 3, right)
                    ]
                )
            with torch.no_grad():
                cu.copy_(torch.tensor(boundaries, device="cuda", dtype=torch.int32))
                x.copy_(torch.tensor(source, device="cuda").reshape_as(x) + 1)
                weights.copy_(torch.tensor(expected, device="cuda").reshape_as(x) + 1)
                if x.grad is not None:
                    x.grad.zero_()
            if graph is None:
                y, restored, extended = run()
            else:
                graph.replay()
            torch.testing.assert_close(y, weights, rtol=0, atol=0)
            torch.testing.assert_close(restored, x, rtol=0, atol=0)
            torch.testing.assert_close(
                extended,
                torch.tensor(halo, device="cuda").reshape(-1, 1, 1).float(),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(x.grad, x, rtol=0, atol=0)
        if graph is not None:
            graph.reset()
    finally:
        Utils.destroy_model_parallel()
