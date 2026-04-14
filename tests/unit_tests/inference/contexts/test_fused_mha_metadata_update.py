# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for update_mha_metadata.

Tests verify that the optimized implementation produces identical results to the
original tensor_copy_and_pad + cumsum implementation across various batch sizes
and edge cases.
"""

import pytest
import torch

from megatron.core.inference.contexts.attention_context.metadata_base import MetadataBase


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
    """Reference using the original MetadataBase.tensor_copy_and_pad path."""
    base = MetadataBase()

    base.tensor_copy_and_pad(query_lengths_buf, query_lengths, real_batch_size, padded_batch_size)
    cu_query_seq_lengths_buf[0] = 0
    base.tensor_copy_and_pad(
        cu_query_seq_lengths_buf[1:],
        (
            torch.cumsum(query_lengths[:real_batch_size], dim=0)
            if real_batch_size > 0
            else torch.zeros(0, dtype=torch.int32, device=query_lengths.device)
        ),
        real_batch_size,
        padded_batch_size,
        is_cumulative_tensor=True,
    )
    base.tensor_copy_and_pad(
        kv_seq_lengths_buf,
        kv_length_offsets[:real_batch_size] + query_lengths[:real_batch_size],
        real_batch_size,
        padded_batch_size,
    )
    cu_kv_seq_lengths_buf[0] = 0
    base.tensor_copy_and_pad(
        cu_kv_seq_lengths_buf[1:],
        (
            torch.cumsum(kv_seq_lengths_buf[:real_batch_size], dim=0)
            if real_batch_size > 0
            else torch.zeros(0, dtype=torch.int32, device=query_lengths.device)
        ),
        real_batch_size,
        padded_batch_size,
        is_cumulative_tensor=True,
    )


def _alloc_buffers(max_bs, device="cuda"):
    """Allocate output buffers matching MHAMetadata shapes."""
    return (
        torch.zeros(max_bs, dtype=torch.int32, device=device),  # query_lengths_buf
        torch.zeros(max_bs + 1, dtype=torch.int32, device=device),  # cu_query_seq_lengths_buf
        torch.zeros(max_bs, dtype=torch.int32, device=device),  # kv_seq_lengths_buf
        torch.zeros(max_bs + 1, dtype=torch.int32, device=device),  # cu_kv_seq_lengths_buf
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestMhaMetadataUpdate:

    def _run_and_compare(self, real_bs, padded_bs, max_bs=None):
        """Run both optimized and reference, assert identical outputs."""
        from megatron.core.inference.contexts.attention_context.mha_metadata import (
            update_mha_metadata,
        )

        if max_bs is None:
            max_bs = padded_bs + 16

        device = "cuda"

        query_lengths = torch.randint(1, 128, (real_bs,), dtype=torch.int32, device=device)
        kv_length_offsets = torch.randint(0, 512, (real_bs,), dtype=torch.int32, device=device)

        ql_opt, cu_q_opt, kvl_opt, cu_kv_opt = _alloc_buffers(max_bs, device)
        ql_ref, cu_q_ref, kvl_ref, cu_kv_ref = _alloc_buffers(max_bs, device)

        for buf in (ql_opt, cu_q_opt, kvl_opt, cu_kv_opt, ql_ref, cu_q_ref, kvl_ref, cu_kv_ref):
            buf.fill_(-999)

        update_mha_metadata(
            query_lengths,
            kv_length_offsets,
            ql_opt,
            cu_q_opt,
            kvl_opt,
            cu_kv_opt,
            real_bs,
            padded_bs,
        )

        _reference_mha_metadata_update(
            query_lengths,
            kv_length_offsets,
            ql_ref,
            cu_q_ref,
            kvl_ref,
            cu_kv_ref,
            real_bs,
            padded_bs,
        )

        assert torch.equal(ql_opt[:padded_bs], ql_ref[:padded_bs]), "query_lengths_buf mismatch"
        assert torch.equal(
            cu_q_opt[: padded_bs + 1], cu_q_ref[: padded_bs + 1]
        ), "cu_query_seq_lengths_buf mismatch"
        assert torch.equal(kvl_opt[:padded_bs], kvl_ref[:padded_bs]), "kv_seq_lengths_buf mismatch"
        assert torch.equal(
            cu_kv_opt[: padded_bs + 1], cu_kv_ref[: padded_bs + 1]
        ), "cu_kv_seq_lengths_buf mismatch"

    def test_empty_batch(self):
        """Zero real requests (edge case)."""
        self._run_and_compare(real_bs=0, padded_bs=0)

    def test_empty_batch_with_padding(self):
        """Zero real requests but nonzero padding."""
        self._run_and_compare(real_bs=0, padded_bs=4)

    def test_large_batch(self):
        """Larger batch size (256 requests)."""
        self._run_and_compare(real_bs=200, padded_bs=256)

    @pytest.mark.parametrize("real_bs", [1, 7, 32, 100, 255])
    @pytest.mark.parametrize("pad_extra", [0, 1, 15, 64])
    def test_parametric(self, real_bs, pad_extra):
        """Sweep of (real_bs, padded_bs) combinations."""
        padded_bs = real_bs + pad_extra
        self._run_and_compare(real_bs=real_bs, padded_bs=padded_bs)

    def test_large_values(self):
        """Large query lengths and KV offsets to check int32 overflow safety."""
        from megatron.core.inference.contexts.attention_context.mha_metadata import (
            update_mha_metadata,
        )

        device = "cuda"
        real_bs, padded_bs, max_bs = 8, 16, 32

        query_lengths = torch.full((real_bs,), 10000, dtype=torch.int32, device=device)
        kv_length_offsets = torch.full((real_bs,), 50000, dtype=torch.int32, device=device)

        ql_opt, cu_q_opt, kvl_opt, cu_kv_opt = _alloc_buffers(max_bs, device)
        ql_ref, cu_q_ref, kvl_ref, cu_kv_ref = _alloc_buffers(max_bs, device)

        update_mha_metadata(
            query_lengths,
            kv_length_offsets,
            ql_opt,
            cu_q_opt,
            kvl_opt,
            cu_kv_opt,
            real_bs,
            padded_bs,
        )
        _reference_mha_metadata_update(
            query_lengths,
            kv_length_offsets,
            ql_ref,
            cu_q_ref,
            kvl_ref,
            cu_kv_ref,
            real_bs,
            padded_bs,
        )

        assert torch.equal(ql_opt[:padded_bs], ql_ref[:padded_bs])
        assert torch.equal(cu_q_opt[: padded_bs + 1], cu_q_ref[: padded_bs + 1])
        assert torch.equal(kvl_opt[:padded_bs], kvl_ref[:padded_bs])
        assert torch.equal(cu_kv_opt[: padded_bs + 1], cu_kv_ref[: padded_bs + 1])

    def test_monotonic_cumsum(self):
        """Verify cu_query and cu_kv are non-decreasing (sanity check)."""
        from megatron.core.inference.contexts.attention_context.mha_metadata import (
            update_mha_metadata,
        )

        device = "cuda"
        real_bs, padded_bs, max_bs = 32, 64, 128
        query_lengths = torch.randint(1, 50, (real_bs,), dtype=torch.int32, device=device)
        kv_length_offsets = torch.randint(0, 200, (real_bs,), dtype=torch.int32, device=device)

        ql, cu_q, kvl, cu_kv = _alloc_buffers(max_bs, device)
        update_mha_metadata(
            query_lengths, kv_length_offsets, ql, cu_q, kvl, cu_kv, real_bs, padded_bs
        )

        cu_q_vals = cu_q[: padded_bs + 1]
        cu_kv_vals = cu_kv[: padded_bs + 1]

        assert (cu_q_vals[1:] >= cu_q_vals[:-1]).all(), "cu_query not non-decreasing"
        assert (cu_kv_vals[1:] >= cu_kv_vals[:-1]).all(), "cu_kv not non-decreasing"
        assert cu_q_vals[0] == 0, "cu_query[0] should be 0"
        assert cu_kv_vals[0] == 0, "cu_kv[0] should be 0"
