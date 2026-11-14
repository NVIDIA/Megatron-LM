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

    @pytest.mark.parametrize(
        "M,K,n", [(120, 32, 60), (120, 32, 120), (120, 32, 128), (124, 64, 124), (124, 64, 128)]
    )
    def test_count_clamped_to_height(self, M, K, n):
        """Rows [0, min(n, M)) are quantized: a count past M gives exactly mxfp8_quantize(x).

        x is the head of a larger buffer of ones and n ends inside the last 128-row scale tile,
        so a kernel that ran past M would read those ones and write their nonzero scales into
        the tile's padding, where the comparison sees them. The shapes keep such an overrun
        contained: its (128 - M) * K <= 256 data bytes fit the allocator's 512-byte rounding.
        """
        torch.manual_seed(3)
        base = torch.ones(MXFP8_SCALE_ROW_BLOCK + 128, K, device="cuda", dtype=torch.bfloat16)
        base[:M] = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 4
        x = base[:M]
        data, scale = mxfp8_quantize_rows(x, torch.tensor([n], device="cuda", dtype=torch.int32))
        live = min(n, M)
        # The swizzled scales of x[:live] are a prefix of x's; every later byte stays zero.
        ref_data, ref_scale = mxfp8_quantize(x[:live])
        expected_scale = torch.zeros_like(scale.view(torch.uint8))
        expected_scale[: ref_scale.numel()] = ref_scale.view(torch.uint8)
        assert torch.equal(data[:live].view(torch.uint8), ref_data.view(torch.uint8))
        assert torch.equal(scale.view(torch.uint8), expected_scale)

    def test_count_past_last_scale_tile_stays_in_bounds(self):
        """A count past the last scale tile writes nothing past the data and scale buffers.

        The wrapper allocates its own outputs, and its scale buffer is a whole number of
        512-byte tiles with no allocator slack behind it. So this launches the kernel as the
        wrapper does, but into the heads of larger canary-filled buffers that catch an overrun.
        """
        from megatron.core.inference.quantization.mxfp8_quantize import (
            _mxfp8_quant_swizzle_rows_kernel,
        )

        M, K, n = 120, 64, 2 * MXFP8_SCALE_ROW_BLOCK  # n reaches into a second scale tile
        torch.manual_seed(4)
        src = torch.ones(n, K, device="cuda", dtype=torch.bfloat16)
        src[:M] = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 4
        data_buf = torch.full((n, K), 0xA5, device="cuda", dtype=torch.uint8)
        scale_buf = torch.full((2 * 512,), 0xA5, device="cuda", dtype=torch.uint8)
        scale_buf[:512].zero_()  # one 128 x 4 tile holds the K / 32 = 2 scales of all M rows
        _mxfp8_quant_swizzle_rows_kernel[(M,)](
            data_buf.view(torch.float8_e4m3fn),
            scale_buf,
            src,
            torch.tensor([n], device="cuda", dtype=torch.int32),
            M,
            K,
            1,  # n_col_blocks
            REAL_GROUPS=K // 32,
            BLOCK_K=K,
            BLOCK_GROUPS=K // 32,
            NUM_PROGRAMS=M,
        )
        ref_data, ref_scale = mxfp8_quantize(src[:M])
        assert torch.equal(data_buf[:M], ref_data.view(torch.uint8))
        assert torch.equal(scale_buf[:512], ref_scale.view(torch.uint8))
        assert torch.all(data_buf[M:] == 0xA5) and torch.all(scale_buf[512:] == 0xA5)

    def test_rejects_non_contiguous_input(self):
        x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)[:, :64]
        with pytest.raises(AssertionError, match="contiguous"):
            mxfp8_quantize_rows(x, torch.tensor([128], device="cuda", dtype=torch.int32))

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
