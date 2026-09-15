# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Triton JIT compile-key stability for the packed-THD CSA kernels.

Under THD the packed segment count, the per-microbatch sequence maxima, and the
compressed-K width are all data-dependent. Any of them appearing in a Triton
``tl.constexpr`` slot puts a host-blocking JIT compile on the critical path of
essentially every microbatch, which is what made packed CSA forward several
times slower than the equivalent non-packed run.

Each test drives one kernel across several packed geometries that differ only
in those data-dependent values, and asserts a single compiled variant. The
geometries deliberately share an integer-specialization class (no value is 1 or
a multiple of 16) so that Triton's own divisibility specialization cannot mask a
regression by collapsing distinct keys.
"""

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.csa_utils import (
    csa_indexer_loss_kernels,
    csa_teacher_lse,
    thd_indexer_kernels,
)


def _require_triton_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if not thd_indexer_kernels._TRITON_AVAILABLE:
        pytest.skip("Triton is not available")


def _compiled_variants(kernel) -> int:
    """Count in-memory compiled variants of a ``triton.jit`` function."""
    return sum(len(entry[0]) for entry in kernel.device_caches.values())


def _reset(*kernels):
    for kernel in kernels:
        kernel.device_caches.clear()


def _cu_seqlens(lengths):
    cumulative = [0]
    for length in lengths:
        cumulative.append(cumulative[-1] + length)
    return torch.tensor(cumulative, dtype=torch.int32, device="cuda")


# Segment layouts with distinct segment counts, query maxima, and compressed
# maxima at ratio 4. No derived value is 1 or a multiple of 16.
_SEGMENT_LAYOUTS = ([68, 36], [52, 52, 20], [100, 44, 36, 20])
_RATIO = 4


def test_build_seq_lens_compiles_once_across_packed_geometries():
    _require_triton_cuda()
    kernel = thd_indexer_kernels._dsv4_thd_build_seq_lens_kernel
    _reset(kernel)

    for lengths in _SEGMENT_LAYOUTS:
        cu_seqlens_q = _cu_seqlens(lengths)
        cu_seqlens_kv = _cu_seqlens([length // _RATIO for length in lengths])
        total_q = sum(lengths)

        seq_lens = thd_indexer_kernels.build_seq_lens(cu_seqlens_q, cu_seqlens_kv, total_q, _RATIO)
        expected = thd_indexer_kernels._build_seq_lens_fallback(
            cu_seqlens_q, cu_seqlens_kv, total_q, _RATIO, None
        )
        assert torch.equal(seq_lens, expected)

    assert _compiled_variants(kernel) == 1


def test_sanitize_topk_compiles_once_across_score_widths():
    _require_triton_cuda()
    kernel = thd_indexer_kernels._dsv4_thd_sanitize_topk_kernel
    _reset(kernel)

    # Widths share a power-of-two bucket so ``BLOCK_TOPK`` stays fixed and only
    # the data-dependent widths vary.
    for rows, width, score_width in ((104, 33, 50), (124, 37, 54), (200, 41, 58)):
        candidates = torch.randint(
            -1, score_width + 4, (rows, width), dtype=torch.int32, device="cuda"
        )
        scores = torch.randn(rows, score_width, dtype=torch.float32, device="cuda")
        seq_lens = torch.randint(0, score_width, (rows,), dtype=torch.int32, device="cuda")

        sanitized, topk_length = thd_indexer_kernels.sanitize_topk(candidates, scores, seq_lens)
        expected_sanitized, expected_length = thd_indexer_kernels._sanitize_topk_fallback(
            candidates, scores, seq_lens
        )
        assert torch.equal(sanitized, expected_sanitized)
        assert torch.equal(topk_length, expected_length)

    assert _compiled_variants(kernel) == 1


def test_prepare_sparse_loss_compiles_once_across_score_widths():
    _require_triton_cuda()
    kernel = csa_indexer_loss_kernels._prepare_sparse_loss_kernel
    _reset(kernel)

    topk_width = 33
    for rows, score_width in ((104, 50), (124, 54), (200, 58)):
        scores = torch.randn(rows, score_width, dtype=torch.float32, device="cuda")
        topk = torch.randint(-1, score_width, (rows, topk_width), dtype=torch.int32, device="cuda")
        physical = torch.where(topk >= 0, topk + 1000, topk)
        padding = torch.zeros(rows, dtype=torch.bool, device="cuda")
        padding[::7] = True

        predict, sanitized_topk, sanitized_physical = csa_indexer_loss_kernels.prepare_sparse_loss(
            scores, topk, padding, physical
        )
        expected = csa_indexer_loss_kernels._prepare_sparse_loss_fallback(
            scores, topk, padding, physical
        )
        torch.testing.assert_close(predict, expected[0])
        assert torch.equal(sanitized_topk, expected[1])
        assert torch.equal(sanitized_physical, expected[2])

    assert _compiled_variants(kernel) == 1


def test_teacher_lse_compiles_once_across_packed_geometries():
    _require_triton_cuda()
    window_kernel = csa_teacher_lse._csa_window_lse_kernel
    compressed_kernel = csa_teacher_lse._csa_compressed_lse_thd_kernel
    _reset(window_kernel, compressed_kernel)

    num_heads, head_dim, window_width = 8, 64, 16
    for lengths in _SEGMENT_LAYOUTS:
        total_q = sum(lengths)
        compressed_lengths = [length // _RATIO for length in lengths]
        total_compressed = sum(compressed_lengths)
        total_kv = total_q + total_compressed

        query = torch.randn(total_q, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
        full_kv = torch.randn(total_kv, head_dim, dtype=torch.bfloat16, device="cuda")
        compressed_kv = full_kv[total_q:]
        attn_sink = torch.zeros(num_heads, dtype=torch.float32, device="cuda")
        window_indices = torch.randint(
            -1, total_q, (total_q, window_width), dtype=torch.int32, device="cuda"
        )

        lse = csa_teacher_lse.fused_csa_teacher_lse(
            query,
            full_kv,
            compressed_kv,
            attn_sink,
            window_indices,
            softmax_scale=head_dim**-0.5,
            ratio=_RATIO,
            cu_seqlens_q=_cu_seqlens(lengths),
            cu_seqlens_k=_cu_seqlens(compressed_lengths),
            max_seqlen_q=max(lengths),
            max_seqlen_k=max(compressed_lengths),
        )
        assert lse.shape == (total_q, num_heads)
        assert torch.isfinite(lse).all()

    assert _compiled_variants(window_kernel) == 1
    assert _compiled_variants(compressed_kernel) == 1
