# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for the GQA min-memory DSA cuDNN indexer top-k.

Both tests stub the cuDNN namespace, so they exercise the wrapper logic in
``_cudnn_indexer_topk_full_k`` without requiring cuDNN itself.
"""

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant import dsa_min_memory
from tests.unit_tests.transformer.experimental_attention_variant.dsa_native_parity_utils import (
    assert_similarity,
)


def _make_indexer_inputs(q_len: int, k_total: int, batch: int = 1, heads: int = 2, dim: int = 4):
    """(q_index, weights, k_index_full) in the layouts the tile function expects."""
    q_index = torch.ones((q_len, batch, heads, dim), dtype=torch.float32)
    weights = torch.ones((q_len, batch, heads), dtype=torch.float32)
    k_index_full = torch.ones((k_total, batch, dim), dtype=torch.float32)
    return q_index, weights, k_index_full


@pytest.mark.skipif(not torch.cuda.is_available(), reason="uses torch.cuda.nvtx ranges")
def test_cudnn_indexer_topk_pads_scores_to_full_key_length(monkeypatch):
    """The score width handed to cuDNN must be the full key length, not ``k_total``.

    cuDNN's varlen top-k reads out of bounds when the score width is close to
    ``top_k``. This path truncates the width to the causal limit
    ``k_total = min(q_end, sq)``, which lands in the affected range for most
    query tiles (at seq 8192 / chunk 512 / topk 1024: 1536, 2560, 3584, ...),
    so the scores are padded with -inf to the full key length before the call.
    """
    q_len, k_total, topk, full_key_len = 2, 6, 4, 8
    assert k_total % topk != 0, "the tile under test must have an unpadded width"

    seen = {}

    class FakeDSA:
        @staticmethod
        def indexer_forward_wrapper(q_bshd, k_bshd, w_bsh, ratio, sm_scale, stream, **kwargs):
            b, sq, _, _ = q_bshd.shape
            return {"scores": torch.zeros((b, sq, k_bshd.size(1)), dtype=torch.float32)}

        @staticmethod
        def indexer_top_k_wrapper(scores_flat, seq_lens, top_k, return_val, stream):
            seen["width"] = scores_flat.size(1)
            seen["rows"] = scores_flat.size(0)
            # Everything past k_total must be -inf so it can never be selected.
            seen["pad_is_neg_inf"] = (
                bool(torch.isneginf(scores_flat[:, k_total:]).all())
                if scores_flat.size(1) > k_total
                else True
            )
            indices = torch.arange(top_k, dtype=torch.int32).expand(scores_flat.size(0), top_k)
            return {"indices": indices.contiguous()}

    monkeypatch.setattr(dsa_min_memory, "_DSA", FakeDSA)

    q_index, weights, k_index_full = _make_indexer_inputs(q_len, k_total)
    dsa_min_memory._cudnn_indexer_topk_full_k(
        q_index,
        weights,
        k_index_full,
        topk,
        q_start=k_total - q_len,
        q_end=k_total,
        full_key_len=full_key_len,
    )

    assert seen["width"] == full_key_len, (
        f"scores were passed at width {seen['width']}; cuDNN's top-k reads out of bounds "
        f"when the width is close to top_k, so it must be padded to {full_key_len}"
    )
    assert seen["rows"] == q_len
    assert seen["pad_is_neg_inf"], "padded columns must be -inf so they are never selectable"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="uses torch.cuda.nvtx ranges")
def test_cudnn_indexer_topk_clamps_out_of_range_indices(monkeypatch):
    """Out-of-range indices from cuDNN must never reach a downstream gather.

    The kernel returns -1 for padding slots by design, but it has also been
    observed returning an out-of-range *positive* index without faulting (2304
    for a score width of 2048). Both are replaced with an in-range, causally
    invalid position so the gather is safe and attention still masks the slot.
    """
    # k_total leaves headroom above the last query position, so pad_idx =
    # query_pos + 1 is in range without the clamp binding. That matches
    # production: padding slots only appear when valid < topk <= k_total, which
    # forces query_pos + 1 <= k_total - 1.
    q_len, k_total, topk = 2, 8, 4
    q_start = 4

    class FakeDSA:
        @staticmethod
        def indexer_forward_wrapper(q_bshd, k_bshd, w_bsh, ratio, sm_scale, stream, **kwargs):
            b, sq, _, _ = q_bshd.shape
            return {"scores": torch.zeros((b, sq, k_bshd.size(1)), dtype=torch.float32)}

        @staticmethod
        def indexer_top_k_wrapper(scores_flat, seq_lens, top_k, return_val, stream):
            # Row 0: a -1 padding slot. Row 1: an out-of-range positive index.
            indices = torch.tensor([[0, 1, 2, -1], [0, 1, 2, k_total + 300]], dtype=torch.int32)
            return {"indices": indices}

    monkeypatch.setattr(dsa_min_memory, "_DSA", FakeDSA)

    q_index, weights, k_index_full = _make_indexer_inputs(q_len, k_total)
    _, topk_indices = dsa_min_memory._cudnn_indexer_topk_full_k(
        q_index, weights, k_index_full, topk, q_start=q_start, q_end=q_start + q_len
    )

    assert topk_indices.shape == (1, q_len, topk)
    assert int(topk_indices.min()) >= 0, "negative index would index backwards in a gather"
    assert int(topk_indices.max()) < k_total, "out-of-range index would fault a gather"

    # Both replacements use query_pos + 1: in range, and > query_pos so
    # _selected_causal_invalid_mask drops the slot from attention.
    query_positions = q_start + torch.arange(q_len)
    expected_pad = torch.clamp(query_positions + 1, max=k_total - 1)
    for row in range(q_len):
        assert (
            expected_pad[row] in topk_indices[0, row]
        ), f"row {row} should contain the substituted padding index {int(expected_pad[row])}"
        invalid = topk_indices[0, row] > query_positions[row]
        assert int(invalid.sum()) == 1, "exactly one slot was substituted in this row"


def _skip_unless_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")


@pytest.mark.flaky_in_dev
def test_cudnn_indexer_topk_matches_pytorch_indexer():
    """The cuDNN indexer must select the same keys as the PyTorch reference.

    Both paths run through ``_topk_index_tile`` on identical inputs; only
    ``use_cudnn`` differs. Indices are returned sorted by key position, so the
    selections can be compared directly.
    """
    _skip_unless_cuda()
    if dsa_min_memory._DSA is None:
        pytest.skip("cuDNN DSA namespace is unavailable")
    # indexer_forward_wrapper only supports 32 or 64 indexer heads.
    index_n_heads, index_head_dim, hidden = 32, 64, 256
    sq, q_len, topk = 128, 64, 32
    q_start, q_end = sq - q_len, sq

    torch.manual_seed(1234)
    dev = torch.device("cuda")
    hidden_states = torch.randn((sq, 1, hidden), device=dev, dtype=torch.bfloat16)
    linear_q = (
        torch.randn((index_n_heads * index_head_dim, hidden), device=dev, dtype=torch.bfloat16)
        / hidden**0.5
    )
    linear_k = torch.randn((index_head_dim, hidden), device=dev, dtype=torch.bfloat16) / hidden**0.5
    linear_w = torch.randn((index_n_heads, hidden), device=dev, dtype=torch.bfloat16) / hidden**0.5
    k_norm_w = torch.ones(index_head_dim, device=dev, dtype=torch.bfloat16)
    k_norm_b = hidden_states.new_empty((0,))

    def run(use_cudnn: bool):
        with dsa_min_memory._triton_dispatch_enabled(False):
            _, indices, _, _ = dsa_min_memory._topk_index_tile(
                hidden_states,
                q_start,
                q_end,
                linear_q,
                linear_k,
                k_norm_w,
                k_norm_b,
                False,
                linear_w,
                1e-5,
                index_n_heads,
                index_head_dim,
                topk,
                0,  # index_rotary_dim: rope off keeps the comparison focused
                None,  # rotary_pos_emb
                False,  # rotary_interleaved
                False,  # use_indexer_rope
                False,  # use_hadamard
                sq,  # key_chunk_size: single block, tie-equivalent to a full topk
                use_cudnn=use_cudnn,
            )
        return indices

    reference = run(use_cudnn=False)
    if not dsa_min_memory._cudnn_available_for_indexer(True, index_n_heads):
        pytest.skip("cuDNN indexer not available for this head count")
    actual = run(use_cudnn=True)

    assert actual.shape == reference.shape
    assert int(actual.min()) >= 0 and int(actual.max()) < sq
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.flaky_in_dev
def test_triton_sparse_attention_matches_pytorch_sparse_attention():
    """The Triton sparse-attention tile must match the PyTorch reference.

    ``_sparse_attention_tile`` tries Triton first and falls back to the PyTorch
    implementation when the kernel declines, so toggling Triton dispatch runs
    the same inputs through both.
    """
    _skip_unless_cuda()
    from megatron.core.transformer.experimental_attention_variant.dsa_min_memory_triton import (
        HAVE_TRITON,
    )

    if not HAVE_TRITON:
        pytest.skip("Triton is unavailable")

    q_len, batch, heads, groups, head_dim = 32, 1, 8, 2, 64
    k_total, topk, q_start = 64, 16, 32
    softmax_scale = head_dim**-0.5

    torch.manual_seed(1234)
    dev = torch.device("cuda")
    # bf16 to match production; the kernel selects a different value-dtype path
    # per dtype, so testing anything else would not cover what training runs.
    dtype = torch.bfloat16
    query = torch.randn((q_len, batch, heads, head_dim), device=dev, dtype=dtype)
    key = torch.randn((k_total, batch, groups, head_dim), device=dev, dtype=dtype)
    value = torch.randn((k_total, batch, groups, head_dim), device=dev, dtype=dtype)
    # Every selected key is causally valid: query row i is at absolute position
    # q_start + i, and topk <= q_start.
    topk_indices = (
        torch.arange(topk, device=dev, dtype=torch.long).view(1, 1, topk).expand(batch, q_len, topk)
    ).contiguous()

    with dsa_min_memory._triton_dispatch_enabled(True):
        actual = dsa_min_memory._sparse_attention_tile(
            query, key, value, topk_indices, softmax_scale, q_start
        )
    with dsa_min_memory._triton_dispatch_enabled(False):
        reference = dsa_min_memory._sparse_attention_tile(
            query, key, value, topk_indices, softmax_scale, q_start
        )

    assert actual.shape == (q_len, batch, heads, head_dim)
    # Element-wise tolerances are the wrong instrument here: attention outputs
    # contain near-zero elements, so relative error blows up on those while the
    # tensors agree overall. Use the same similarity check the neighbouring DSA
    # parity tests use. At bf16 the two implementations are expected to differ
    # by roughly one ulp (~4e-3 relative) from differing reduction order, which
    # a similarity check tolerates and an element-wise one would not.
    assert_similarity(actual, reference, label="triton vs pytorch sparse attention")
