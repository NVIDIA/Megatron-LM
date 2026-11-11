# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Short rows of a small-K cuDNN top-k are selected in PyTorch (CPU checks, stand-in kernel).

cuDNN's radix top-k accepts ``top_k`` in (0, 2048], but below ``_cudnn_topk_min_reliable_k``
of the scores' dtype (FP32 6, BF16 14) it gets rows longer than ``top_k`` and at most that many
elements long wrong (the GPU checks are in ``test_dsa_cudnn_topk_small_k_gpu.py``).
``_indexer_top_k_wrapper_chunked`` must hand the kernel those rows with length 0, keep the
kernel's result for every other row, and give those rows the canonical top-k: the
``min(K, selectable)`` best in-row scores (not NaN, not ``-inf``), cutoff ties by ascending
index, in ascending index order, padded with ``-1`` indices and ``finfo.min`` values.

The stand-in kernel is ``dsa_cudnn_fe_fakes.indexer_top_k_wrapper``, which refuses such rows.
"""

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels
from tests.unit_tests.transformer.experimental_attention_variant import dsa_cudnn_fe_fakes as fakes

_MIN_K = {
    dtype: dsa_cudnn_kernels._cudnn_topk_min_reliable_k(dtype)
    for dtype in (torch.float32, torch.bfloat16)
}


def test_reliable_k_follows_the_32_byte_prologue():
    assert _MIN_K == {torch.float32: 6, torch.bfloat16: 14}


class _RecordingTopK:
    """Stand-in ``indexer_top_k_wrapper`` (no ``tie_break``) that records each launch."""

    def __init__(self):
        self.launches = []

    def indexer_top_k_wrapper(self, input_values, seq_lens, top_k, next_n=1, return_val=True):
        self.launches.append((top_k, seq_lens.clone()))
        return fakes.indexer_top_k_wrapper(input_values, seq_lens, top_k, next_n, return_val)


@pytest.fixture
def kernel(monkeypatch):
    fake = _RecordingTopK()
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", fake)
    return fake


def _reference(scores, seq_lens, topk_k):
    """Row by row: the canonical top-k described in the module docstring."""
    rows, width = scores.shape
    indices = torch.full((rows, topk_k), -1, dtype=torch.int32)
    values = torch.full((rows, topk_k), torch.finfo(scores.dtype).min, dtype=scores.dtype)
    for row in range(rows):
        length = max(0, min(int(seq_lens[row]), width))
        row_scores = scores[row].double().tolist()
        selectable = [c for c in range(length) if row_scores[c] > float("-inf")]  # drops NaN too
        chosen = sorted(sorted(selectable, key=lambda c: (-row_scores[c], c))[:topk_k])
        indices[row, : len(chosen)] = torch.tensor(chosen, dtype=torch.int32)
        values[row, : len(chosen)] = scores[row, chosen]
    return indices, values


def _scores(width, dtype, seed, rows=96):
    """Gaussian rows plus rows with ties, NaN, +-inf and signed zeros, and their lengths.

    Lengths cover 0 .. 15 (every short length of both dtypes) on plain and special rows, and
    the row width; the rest are random, one past the width included (clamped).
    """
    generator = torch.Generator().manual_seed(seed)
    scores = torch.randn(rows, width, generator=generator).to(dtype)
    scores[8:16] = torch.randint(-2, 3, (8, width), generator=generator).to(dtype)  # ties
    scores[16:20] = 0.0
    scores[16:20, ::2] = -0.0
    special = [float("nan"), float("inf"), float("-inf")]
    positions = torch.randint(0, width, (12, 2), generator=generator)
    for i in range(12):
        scores[20 + i, positions[i]] = special[i % 3]
    scores[24, : min(width, 3)] = float("nan")
    scores[25, : min(width, 4)] = float("-inf")
    seq_lens = torch.randint(0, width + 2, (rows,), generator=generator)
    seq_lens[:32] = torch.arange(32) % 16
    seq_lens[48:52] = width
    return scores, seq_lens.to(torch.int32)


def _bits(values):
    """Bit patterns, so NaN slots the stand-in kernel ranked compare equal to themselves."""
    return values.contiguous().view(torch.int16 if values.element_size() == 2 else torch.int32)


def _short_rows(seq_lens, topk_k, dtype):
    return (seq_lens > topk_k) & (seq_lens <= _MIN_K[dtype])


def _run(scores, seq_lens, topk_k, **kwargs):
    return dsa_cudnn_kernels._indexer_top_k_wrapper_chunked(
        scores, seq_lens, topk_k=topk_k, **kwargs
    )


_SMALL_K = [
    pytest.param(dtype, topk_k, id=f"{name}-k{topk_k}")
    for dtype, name in ((torch.float32, "fp32"), (torch.bfloat16, "bf16"))
    for topk_k in range(1, _MIN_K[dtype])
]


@pytest.mark.parametrize("dtype, topk_k", _SMALL_K)
@pytest.mark.parametrize("width", [5, 17, 130])
@pytest.mark.parametrize("deterministic", [False, True], ids=["non_det", "det"])
@pytest.mark.parametrize("return_topk_scores", [False, True], ids=["indices", "values"])
def test_short_rows_skip_the_kernel_and_get_the_canonical_top_k(
    kernel, monkeypatch, dtype, topk_k, width, deterministic, return_topk_scores
):
    if topk_k > width:
        pytest.skip("callers never ask for more than the row width")
    scores, seq_lens = _scores(width, dtype, seed=31 * width + topk_k)
    short = _short_rows(seq_lens, topk_k, dtype)
    assert bool(short.any())
    kwargs = dict(return_topk_scores=return_topk_scores, deterministic=deterministic)

    result = _run(scores, seq_lens, topk_k, **kwargs)
    (launched_k, launched_lens), *_ = kernel.launches
    with monkeypatch.context() as unguarded:
        # The stand-in deciding every row, as before the guard: what the other rows keep.
        fakes.trust_stand_in_top_k_for_any_k(unguarded)
        before = _run(scores, seq_lens, topk_k, **kwargs)

    assert launched_k == topk_k
    assert torch.equal(launched_lens, torch.where(short, 0, seq_lens))
    reference_indices, reference_values = _reference(scores, seq_lens, topk_k)
    rows = short.view(-1, 1)
    assert result["indices"].dtype == torch.int32
    assert torch.equal(result["indices"], torch.where(rows, reference_indices, before["indices"]))
    if return_topk_scores:
        assert result["values"].dtype == dtype
        expected = torch.where(rows, reference_values, before["values"])
        assert torch.equal(_bits(result["values"]), _bits(expected))
    else:
        assert result["values"] is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("deterministic", [False, True], ids=["non_det", "det"])
def test_the_reliable_k_hands_the_kernel_every_row(kernel, dtype, deterministic):
    scores, seq_lens = _scores(17, dtype, seed=7)

    _run(scores, seq_lens, _MIN_K[dtype], return_topk_scores=False, deterministic=deterministic)

    assert [(k, lens.tolist()) for k, lens in kernel.launches] == [
        (_MIN_K[dtype], seq_lens.tolist())
    ]


@pytest.mark.parametrize("deterministic", [False, True], ids=["non_det", "det"])
def test_short_rows_are_selected_across_kernel_row_chunks(kernel, deterministic):
    scores, seq_lens = _scores(130, torch.float32, seed=11)
    kwargs = dict(return_topk_scores=True, deterministic=deterministic)
    whole = _run(scores, seq_lens, 3, **kwargs)
    launches = len(kernel.launches)

    # A budget of 8 rows' scratch (rows x 2 x columns int32) per launch.
    chunked = _run(scores, seq_lens, 3, topk_scratch_bytes=8 * 2 * 130 * 4, **kwargs)

    assert len(kernel.launches) - launches > 1
    assert torch.equal(chunked["indices"], whole["indices"])
    assert torch.equal(_bits(chunked["values"]), _bits(whole["values"]))


@pytest.mark.parametrize("budget_rows", [1, 3, 96])
def test_exact_small_topk_matches_the_reference_in_row_chunks(monkeypatch, budget_rows):
    scores, seq_lens = _scores(130, torch.float32, seed=13)
    bytes_per_row = 130 * (4 + 16) + 40 * 3
    monkeypatch.setattr(
        dsa_cudnn_kernels, "_DETERMINISTIC_TOPK_REPAIR_MAX_BYTES", budget_rows * bytes_per_row
    )

    result = dsa_cudnn_kernels._exact_small_topk(
        scores, seq_lens, topk_k=3, return_topk_scores=True
    )

    reference_indices, reference_values = _reference(scores, seq_lens, 3)
    assert torch.equal(result["indices"], reference_indices)
    assert torch.equal(result["values"], reference_values)


def test_stand_in_refuses_short_rows_below_the_reliable_k():
    scores, seq_lens = _scores(17, torch.float32, seed=5)
    short = _short_rows(seq_lens, 2, torch.float32)

    with pytest.raises(ValueError, match="gets wrong"):
        fakes.indexer_top_k_wrapper(scores, seq_lens, 2)
    fakes.indexer_top_k_wrapper(scores, torch.where(short, 0, seq_lens), 2)
