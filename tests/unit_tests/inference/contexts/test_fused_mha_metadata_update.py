# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the fused Triton MHA metadata update kernel.

Tests verify that the fused kernel produces identical results to the original
multi-op PyTorch implementation (copy+pad, cumsum, add) across various batch
sizes and edge cases.
"""

import pytest
import torch


def _reference_mha_metadata_update(
    query_lengths,
    kv_length_offsets,
    query_lengths_buf,
    cu_query_seq_lengths_buf,
    kv_seq_lengths_buf,
    cu_kv_seq_lengths_buf,
    real_batch_size,
    padded_batch_size,
):
    """Reference implementation matching the original MHAMetadata.update() logic."""
    # query_lengths_buf: copy + pad with 0
    query_lengths_buf[:real_batch_size] = query_lengths[:real_batch_size]
    query_lengths_buf[real_batch_size:padded_batch_size] = 0

    # cu_query_seq_lengths_buf: [0] + cumsum + pad with last cumsum value
    cu_query_seq_lengths_buf[0] = 0
    if real_batch_size > 0:
        cumsum_q = torch.cumsum(query_lengths[:real_batch_size], dim=0)
        cu_query_seq_lengths_buf[1 : real_batch_size + 1] = cumsum_q
        cu_query_seq_lengths_buf[real_batch_size + 1 : padded_batch_size + 1] = cumsum_q[-1]
    else:
        cu_query_seq_lengths_buf[1 : padded_batch_size + 1] = 0

    # kv_seq_lengths_buf: (offsets + query_lengths) copy + pad with 0
    kv_seq_lengths = kv_length_offsets[:real_batch_size] + query_lengths[:real_batch_size]
    kv_seq_lengths_buf[:real_batch_size] = kv_seq_lengths
    kv_seq_lengths_buf[real_batch_size:padded_batch_size] = 0

    # cu_kv_seq_lengths_buf: [0] + cumsum(kv_seq_lengths) + pad with last
    cu_kv_seq_lengths_buf[0] = 0
    if real_batch_size > 0:
        cumsum_kv = torch.cumsum(kv_seq_lengths, dim=0)
        cu_kv_seq_lengths_buf[1 : real_batch_size + 1] = cumsum_kv
        cu_kv_seq_lengths_buf[real_batch_size + 1 : padded_batch_size + 1] = cumsum_kv[-1]
    else:
        cu_kv_seq_lengths_buf[1 : padded_batch_size + 1] = 0


def _alloc_buffers(max_bs, device="cuda"):
    """Allocate output buffers matching MHAMetadata shapes."""
    return (
        torch.zeros(max_bs, dtype=torch.int32, device=device),  # query_lengths_buf
        torch.zeros(max_bs + 1, dtype=torch.int32, device=device),  # cu_query_seq_lengths_buf
        torch.zeros(max_bs, dtype=torch.int32, device=device),  # kv_seq_lengths_buf
        torch.zeros(max_bs + 1, dtype=torch.int32, device=device),  # cu_kv_seq_lengths_buf
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestFusedMhaMetadataUpdate:

    def _run_and_compare(self, real_bs, padded_bs, max_bs=None):
        """Run both fused and reference, assert identical outputs."""
        from megatron.core.inference.contexts.attention_context.triton.fused_mha_metadata_update import (
            HAVE_TRITON,
            fused_mha_metadata_update,
        )

        if not HAVE_TRITON:
            pytest.skip("Triton not available")

        if max_bs is None:
            max_bs = padded_bs + 16  # some headroom

        device = "cuda"

        # Random inputs
        query_lengths = torch.randint(1, 128, (real_bs,), dtype=torch.int32, device=device)
        kv_length_offsets = torch.randint(0, 512, (real_bs,), dtype=torch.int32, device=device)

        # Allocate two sets of output buffers
        ql_buf_fused, cu_q_buf_fused, kvl_buf_fused, cu_kv_buf_fused = _alloc_buffers(max_bs, device)
        ql_buf_ref, cu_q_buf_ref, kvl_buf_ref, cu_kv_buf_ref = _alloc_buffers(max_bs, device)

        # Sentinel fill to ensure kernel writes all expected positions
        for buf in (ql_buf_fused, cu_q_buf_fused, kvl_buf_fused, cu_kv_buf_fused):
            buf.fill_(-999)
        for buf in (ql_buf_ref, cu_q_buf_ref, kvl_buf_ref, cu_kv_buf_ref):
            buf.fill_(-999)

        # Run fused kernel
        fused_mha_metadata_update(
            query_lengths,
            kv_length_offsets,
            ql_buf_fused,
            cu_q_buf_fused,
            kvl_buf_fused,
            cu_kv_buf_fused,
            real_bs,
            padded_bs,
        )

        # Run reference
        _reference_mha_metadata_update(
            query_lengths,
            kv_length_offsets,
            ql_buf_ref,
            cu_q_buf_ref,
            kvl_buf_ref,
            cu_kv_buf_ref,
            real_bs,
            padded_bs,
        )

        # Compare the meaningful region (up to padded_bs or padded_bs+1).
        # All ops are int32 (copy, add, cumsum) so results must be bit-exact.
        assert torch.equal(ql_buf_fused[:padded_bs], ql_buf_ref[:padded_bs]), \
            "query_lengths_buf mismatch"
        assert torch.equal(
            cu_q_buf_fused[: padded_bs + 1], cu_q_buf_ref[: padded_bs + 1]
        ), "cu_query_seq_lengths_buf mismatch"
        assert torch.equal(kvl_buf_fused[:padded_bs], kvl_buf_ref[:padded_bs]), \
            "kv_seq_lengths_buf mismatch"
        assert torch.equal(
            cu_kv_buf_fused[: padded_bs + 1], cu_kv_buf_ref[: padded_bs + 1]
        ), "cu_kv_seq_lengths_buf mismatch"

    def test_basic(self):
        """Typical batch: 16 real requests padded to 32."""
        self._run_and_compare(real_bs=16, padded_bs=32)

    def test_no_padding(self):
        """real_bs == padded_bs (no padding needed)."""
        self._run_and_compare(real_bs=8, padded_bs=8)

    def test_single_request(self):
        """Single request, padded to 1."""
        self._run_and_compare(real_bs=1, padded_bs=1)

    def test_single_request_padded(self):
        """Single request, padded to 8."""
        self._run_and_compare(real_bs=1, padded_bs=8)

    def test_empty_batch(self):
        """Zero real requests (edge case)."""
        self._run_and_compare(real_bs=0, padded_bs=0)

    def test_empty_batch_with_padding(self):
        """Zero real requests but nonzero padding."""
        self._run_and_compare(real_bs=0, padded_bs=4)

    def test_large_batch(self):
        """Larger batch size (256 requests)."""
        self._run_and_compare(real_bs=200, padded_bs=256)

    def test_power_of_two_batch(self):
        """Padded batch is an exact power of 2."""
        self._run_and_compare(real_bs=60, padded_bs=64)

    def test_heavy_padding(self):
        """Small real batch with lots of padding."""
        self._run_and_compare(real_bs=3, padded_bs=128)

    @pytest.mark.parametrize("real_bs", [1, 7, 32, 100, 255])
    @pytest.mark.parametrize("pad_extra", [0, 1, 15, 64])
    def test_parametric(self, real_bs, pad_extra):
        """Sweep of (real_bs, padded_bs) combinations."""
        padded_bs = real_bs + pad_extra
        self._run_and_compare(real_bs=real_bs, padded_bs=padded_bs)

    def test_large_values(self):
        """Large query lengths and KV offsets to check int32 overflow safety."""
        device = "cuda"
        from megatron.core.inference.contexts.attention_context.triton.fused_mha_metadata_update import (
            HAVE_TRITON,
            fused_mha_metadata_update,
        )

        if not HAVE_TRITON:
            pytest.skip("Triton not available")

        real_bs = 8
        padded_bs = 16
        max_bs = 32

        query_lengths = torch.full((real_bs,), 10000, dtype=torch.int32, device=device)
        kv_length_offsets = torch.full((real_bs,), 50000, dtype=torch.int32, device=device)

        ql_fused, cu_q_fused, kvl_fused, cu_kv_fused = _alloc_buffers(max_bs, device)
        ql_ref, cu_q_ref, kvl_ref, cu_kv_ref = _alloc_buffers(max_bs, device)

        fused_mha_metadata_update(
            query_lengths, kv_length_offsets, ql_fused, cu_q_fused, kvl_fused, cu_kv_fused,
            real_bs, padded_bs,
        )
        _reference_mha_metadata_update(
            query_lengths, kv_length_offsets, ql_ref, cu_q_ref, kvl_ref, cu_kv_ref,
            real_bs, padded_bs,
        )

        assert torch.equal(ql_fused[:padded_bs], ql_ref[:padded_bs])
        assert torch.equal(cu_q_fused[: padded_bs + 1], cu_q_ref[: padded_bs + 1])
        assert torch.equal(kvl_fused[:padded_bs], kvl_ref[:padded_bs])
        assert torch.equal(cu_kv_fused[: padded_bs + 1], cu_kv_ref[: padded_bs + 1])

    def test_monotonic_cumsum(self):
        """Verify cu_query and cu_kv are non-decreasing (sanity check)."""
        from megatron.core.inference.contexts.attention_context.triton.fused_mha_metadata_update import (
            HAVE_TRITON,
            fused_mha_metadata_update,
        )

        if not HAVE_TRITON:
            pytest.skip("Triton not available")

        device = "cuda"
        real_bs, padded_bs, max_bs = 32, 64, 128
        query_lengths = torch.randint(1, 50, (real_bs,), dtype=torch.int32, device=device)
        kv_length_offsets = torch.randint(0, 200, (real_bs,), dtype=torch.int32, device=device)

        ql, cu_q, kvl, cu_kv = _alloc_buffers(max_bs, device)
        fused_mha_metadata_update(
            query_lengths, kv_length_offsets, ql, cu_q, kvl, cu_kv, real_bs, padded_bs,
        )

        cu_q_vals = cu_q[: padded_bs + 1]
        cu_kv_vals = cu_kv[: padded_bs + 1]

        assert (cu_q_vals[1:] >= cu_q_vals[:-1]).all(), "cu_query not non-decreasing"
        assert (cu_kv_vals[1:] >= cu_kv_vals[:-1]).all(), "cu_kv not non-decreasing"
        assert cu_q_vals[0] == 0, "cu_query[0] should be 0"
        assert cu_kv_vals[0] == 0, "cu_kv[0] should be 0"
