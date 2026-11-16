# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Batch-invariant MoE inference kernels bounded by the live row count.

The permuted expert buffers are sized for the worst case (max_tokens x EP x topk rows) so CUDA
graphs see static shapes. These tests check that the row-bounded kernels touch only the live
rows and produce exactly the bits of the whole-buffer kernels on those rows, eagerly and under
CUDA-graph replay with a changing device-side count.
"""

import pytest
import torch

from megatron.core.inference.quantization.mxfp8_quantize import (
    MXFP8_SCALE_ROW_BLOCK,
    mxfp8_quantize,
    mxfp8_quantize_rows,
)

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.internal,
]


def _scale_bytes(n_rows: int, K: int) -> int:
    n_col_blocks = (K // 32 + 3) // 4
    return n_rows // MXFP8_SCALE_ROW_BLOCK * n_col_blocks * 512


def _check_rows(x, n, data, scale):
    ref_data, ref_scale = mxfp8_quantize(x)
    assert torch.equal(ref_data[:n].view(torch.uint8), data[:n].view(torch.uint8))
    nb = _scale_bytes(n, x.shape[1])
    assert torch.equal(ref_scale.view(torch.uint8)[:nb], scale.view(torch.uint8)[:nb])


class TestMxfp8QuantizeRows:
    @pytest.mark.parametrize("K", [1856, 2688, 4096])
    @pytest.mark.parametrize("n", [0, 128, 1280, 8192])
    def test_live_rows_bitwise_equal(self, K, n):
        torch.manual_seed(0)
        x = torch.randn(8192, K, device="cuda", dtype=torch.bfloat16) * 4
        n_t = torch.tensor([n], device="cuda", dtype=torch.int32)
        data, scale = mxfp8_quantize_rows(x, n_t)
        _check_rows(x, n, data, scale)
        # Rows past n are never quantized: their scales stay zero.
        assert not scale.view(torch.uint8)[_scale_bytes(n, K) :].any()

    def test_cuda_graph_replay_with_changing_count(self):
        torch.manual_seed(1)
        x = torch.randn(4096, 2688, device="cuda", dtype=torch.bfloat16)
        n_t = torch.tensor([256], device="cuda", dtype=torch.int32)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            mxfp8_quantize_rows(x, n_t)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            data, scale = mxfp8_quantize_rows(x, n_t)
        for n in (256, 3840, 128):
            n_t.fill_(n)
            graph.replay()
            _check_rows(x, n, data, scale)


class TestUnpermuteInExpertOrderLiveTokens:
    @pytest.mark.parametrize("valid", [1, 40, 1000])
    def test_matches_dense_reference(self, valid):
        from megatron.core.inference.moe.batch_invariant import unpermute_tokens_in_expert_order

        torch.manual_seed(2)
        hidden, experts, topk, num_tokens = 512, 16, 4, 4096
        rows = valid * topk + 128 * experts
        expert_out = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
        inverse = torch.full((num_tokens, experts), -1, device="cuda", dtype=torch.int32)
        pos = torch.randperm(rows, device="cuda")[: valid * topk].view(valid, topk).int()
        eid = torch.stack([torch.randperm(experts, device="cuda")[:topk] for _ in range(valid)])
        inverse[torch.arange(valid, device="cuda")[:, None], eid] = pos
        out = torch.full((num_tokens, hidden), 7.0, device="cuda", dtype=torch.float32)
        valid_t = torch.tensor([valid], device="cuda", dtype=torch.int32)
        unpermute_tokens_in_expert_order(expert_out, inverse, valid_t, out)
        # Reference: fp64 sum in increasing local-expert order, rounded once to fp32.
        ref = torch.zeros(valid, hidden, device="cuda", dtype=torch.float64)
        for lid in range(experts):
            p = inverse[:valid, lid].long()
            m = p >= 0
            ref[m] += expert_out[p[m]].double()
        assert torch.equal(out[:valid], ref.float())
        # Tokens past the live count are not written.
        assert torch.all(out[valid:] == 7.0)
