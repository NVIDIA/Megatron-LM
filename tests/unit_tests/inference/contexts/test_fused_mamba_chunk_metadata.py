# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the fused Triton mamba chunk metadata kernels.

Verifies that the GPU-only kernel produces identical results to the
original Python-loop + .tolist() implementation.
"""

import pytest
import torch


def _reference_mamba_chunk_metadata(
    cu_seqlens, padded_prefill_count, chunk_size, padded_max_chunks
):
    """Reference implementation matching the original Python loop in MambaMetadata.update()."""
    cu_seqlens_list = cu_seqlens[: padded_prefill_count + 1].tolist()
    chunk_boundaries = [0]
    last_chunk_idx_list = []
    chunk_to_seq_list = []

    for i in range(padded_prefill_count):
        start = cu_seqlens_list[i]
        end = cu_seqlens_list[i + 1]
        seq_len = end - start
        n_chunks = max(1, (seq_len + chunk_size - 1) // chunk_size)
        boundaries = [min(start + (k + 1) * chunk_size, end) for k in range(n_chunks)]
        chunk_boundaries.extend(boundaries)
        chunk_to_seq_list.extend([i] * n_chunks)
        last_chunk_idx_list.append(len(chunk_boundaries) - 2)

    last_boundary = chunk_boundaries[-1]
    pad_b = padded_max_chunks + 1 - len(chunk_boundaries)
    if pad_b > 0:
        chunk_boundaries.extend([last_boundary] * pad_b)
    pad_s = padded_max_chunks - len(chunk_to_seq_list)
    if pad_s > 0:
        chunk_to_seq_list.extend([0] * pad_s)

    n_cu = padded_max_chunks + 1
    cu_chunk_seqlens = torch.tensor(chunk_boundaries[:n_cu], dtype=torch.int32, device="cuda")
    last_chunk_indices = torch.tensor(last_chunk_idx_list, dtype=torch.int32, device="cuda")
    seq_idx_for_varlen = torch.tensor(
        chunk_to_seq_list[:padded_max_chunks], dtype=torch.int32, device="cuda"
    )
    return cu_chunk_seqlens, last_chunk_indices, seq_idx_for_varlen


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestFusedMambaChunkMetadata:

    def _run_and_compare(self, seq_lens_list, padded_prefill_count, chunk_size):
        from megatron.core.inference.contexts.attention_context.triton.fused_mamba_chunk_metadata import (
            HAVE_TRITON,
            fused_mamba_chunk_metadata,
        )

        if not HAVE_TRITON:
            pytest.skip("Triton not available")

        device = "cuda"
        real_prefill = len(seq_lens_list)
        assert real_prefill <= padded_prefill_count

        # Build cu_seqlens: real seqs + zero-length padding seqs
        cu_vals = [0]
        for sl in seq_lens_list:
            cu_vals.append(cu_vals[-1] + sl)
        # Padding seqs have zero length (cu_seqlens stays at last value)
        for _ in range(padded_prefill_count - real_prefill):
            cu_vals.append(cu_vals[-1])

        max_bs = padded_prefill_count + 16
        cu_seqlens_buf = torch.zeros(max_bs + 1, dtype=torch.int32, device=device)
        cu_seqlens_buf[: padded_prefill_count + 1] = torch.tensor(
            cu_vals, dtype=torch.int32, device=device
        )

        padded_token_count = cu_vals[-1] + padded_prefill_count  # conservative
        padded_max_chunks = padded_token_count // chunk_size + padded_prefill_count

        # --- Reference ---
        ref_cu_chunk, ref_last_idx, ref_seq_idx = _reference_mamba_chunk_metadata(
            cu_seqlens_buf, padded_prefill_count, chunk_size, padded_max_chunks
        )

        # --- Fused kernel ---
        cum_chunks_buf = torch.zeros(max_bs + 1, dtype=torch.int32, device=device)
        cu_chunk_seqlens_buf = torch.zeros(
            padded_max_chunks + 16, dtype=torch.int32, device=device
        )
        last_chunk_indices_buf = torch.zeros(max_bs, dtype=torch.int32, device=device)
        seq_idx_for_varlen_buf = torch.zeros(
            padded_max_chunks + 16, dtype=torch.int32, device=device
        )

        fused_mamba_chunk_metadata(
            cu_seqlens_buf=cu_seqlens_buf,
            cum_chunks_buf=cum_chunks_buf,
            cu_chunk_seqlens_buf=cu_chunk_seqlens_buf,
            last_chunk_indices_buf=last_chunk_indices_buf,
            seq_idx_for_varlen_buf=seq_idx_for_varlen_buf,
            padded_prefill_count=padded_prefill_count,
            chunk_size=chunk_size,
            padded_max_chunks=padded_max_chunks,
            padded_token_count=padded_token_count,
        )

        n_cu = padded_max_chunks + 1
        fused_cu_chunk = cu_chunk_seqlens_buf[:n_cu]
        fused_last_idx = last_chunk_indices_buf[:padded_prefill_count]
        fused_seq_idx = seq_idx_for_varlen_buf[:padded_max_chunks]

        assert torch.equal(fused_cu_chunk, ref_cu_chunk), (
            f"cu_chunk_seqlens mismatch\nfused: {fused_cu_chunk}\nref:   {ref_cu_chunk}"
        )
        assert torch.equal(fused_last_idx, ref_last_idx), (
            f"last_chunk_indices mismatch\nfused: {fused_last_idx}\nref:   {ref_last_idx}"
        )
        assert torch.equal(fused_seq_idx, ref_seq_idx), (
            f"seq_idx_for_varlen mismatch\nfused: {fused_seq_idx}\nref:   {ref_seq_idx}"
        )

    def test_single_seq_single_chunk(self):
        """One sequence shorter than chunk_size."""
        self._run_and_compare([50], padded_prefill_count=1, chunk_size=128)

    def test_single_seq_exact_chunk(self):
        """One sequence exactly chunk_size."""
        self._run_and_compare([128], padded_prefill_count=1, chunk_size=128)

    def test_single_seq_multi_chunk(self):
        """One sequence spanning multiple chunks."""
        self._run_and_compare([300], padded_prefill_count=1, chunk_size=128)

    def test_multi_seq_no_padding(self):
        """Multiple sequences, no padded seqs."""
        self._run_and_compare([100, 200, 50], padded_prefill_count=3, chunk_size=128)

    def test_multi_seq_with_padding(self):
        """3 real seqs padded to 8 total (5 zero-length padding seqs)."""
        self._run_and_compare([100, 200, 50], padded_prefill_count=8, chunk_size=128)

    def test_zero_length_sequences(self):
        """All padding sequences (zero real prefill but padded_prefill > 0)."""
        self._run_and_compare([], padded_prefill_count=4, chunk_size=128)

    def test_mixed_lengths(self):
        """Mix of short, exact, and multi-chunk sequences."""
        self._run_and_compare([1, 128, 129, 256, 10, 500], padded_prefill_count=6, chunk_size=128)

    def test_large_batch(self):
        """Larger batch with many sequences."""
        seq_lens = [64 + i * 10 for i in range(32)]
        self._run_and_compare(seq_lens, padded_prefill_count=32, chunk_size=128)

    def test_large_batch_with_padding(self):
        """16 real seqs padded to 64."""
        seq_lens = [100 + i * 20 for i in range(16)]
        self._run_and_compare(seq_lens, padded_prefill_count=64, chunk_size=128)

    def test_small_chunk_size(self):
        """Smaller chunk size generates more chunks per sequence."""
        self._run_and_compare([300, 500], padded_prefill_count=4, chunk_size=32)

    def test_single_token_sequences(self):
        """Sequences with exactly 1 token."""
        self._run_and_compare([1, 1, 1, 1], padded_prefill_count=4, chunk_size=128)

    @pytest.mark.parametrize("n_real", [1, 4, 16])
    @pytest.mark.parametrize("pad_extra", [0, 4, 16])
    @pytest.mark.parametrize("chunk_size", [32, 128])
    def test_parametric(self, n_real, pad_extra, chunk_size):
        seq_lens = [50 + i * 30 for i in range(n_real)]
        self._run_and_compare(
            seq_lens, padded_prefill_count=n_real + pad_extra, chunk_size=chunk_size
        )
