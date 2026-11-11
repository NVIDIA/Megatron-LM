# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Real cuDNN checks of the small-K top-k guard (``_cudnn_topk_min_reliable_k``).

cuDNN's radix top-k reads each row as a scalar prologue up to the next 32-byte boundary, a
vectorized body and a scalar tail, and does not clamp the prologue to the row's length. A row
of length ``L`` whose prologue ``P`` (up to 32 bytes / score size - 1 elements, set by the
row's address) exceeds ``L`` while ``L > top_k`` is selected from a corrupt set: duplicate ids,
columns past ``L``, in the next row or before the row's start (wrapped to 655xx ids). That needs
``top_k <= 32 / score size - 3``: 5 for FP32, 13 for BF16 (both measured on frontends 1.29 and
1.31, H100 and GB300). Below the guard the wrapper gives the kernel such rows with length 0 and
selects them in PyTorch. On one SM90+ GPU with the installed frontend:

* ``_indexer_top_k_wrapper_chunked`` below the guard selects a right top-k on every row: in
  deterministic mode exactly the canonical one (the ``min(K, selectable)`` best in-row scores,
  cutoff ties by ascending index, ascending index order, ``-1`` / ``finfo.min`` padding), and
  otherwise the same scores as the canonical one with distinct in-row ids.
* The raw kernel selects the right set for every ``top_k`` from the guard up, on rows that
  cover every (length, prologue) pair of short rows longer than ``top_k``, in its many-row and
  few-row (at most 148 rows) compilations and with ``tie_break=1``: what lets the guard stop
  there.
* The raw kernel's wrong rows below the guard are reported (``[topk-small-k]`` lines), not
  asserted, so the validation log records whether a frontend still has the defect.
"""

import os
import warnings

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels

_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
# Odd widths walk every row-start residue modulo the vector width (7 and 15 step through them
# in row order); 4097 is wider than one 4096-column bucket.
_WIDTHS = [3, 5, 7, 15, 17, 129, 4097]


def _min_k(dtype):
    return dsa_cudnn_kernels._cudnn_topk_min_reliable_k(dtype)


def _vector(dtype):
    return dsa_cudnn_kernels._CUDNN_TOPK_VECTOR_BYTES // torch.empty((), dtype=dtype).element_size()


@pytest.fixture(autouse=True)
def cudnn_top_k():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("cuDNN radix top-k needs SM90+")
    try:
        dsa_cudnn_kernels._ensure_dsa_namespace()
    except ImportError as exc:
        pytest.skip(f"cuDNN frontend DSA unavailable: {exc}")


def _report(message):
    """Print and warn, so the line also reaches the warnings summary of a captured run."""
    print(f"[topk-small-k] {message}", flush=True)
    warnings.warn(f"[topk-small-k] {message}", stacklevel=2)


def _rows(width, dtype, seed, *, tie_free, few_rows_for_k=None):
    """Score rows on the GPU and their lengths.

    Lengths: every short length (0 .. vector width) on as many consecutive rows as the vector
    width, so with an odd width each short length meets every prologue length; then every length
    0 .. width twice, then random lengths. With ``few_rows_for_k`` there are 128 rows (the
    kernel's low-occupancy compilation takes at most 148): the short lengths from that K - 1 up,
    each on vector-width consecutive rows, then random lengths above K. ``tie_free`` rows hold a
    random permutation of 0 .. width - 1 (exact in BF16 up to 256 columns); otherwise Gaussian
    rows with every third row coarse (many exact ties).
    """
    generator = torch.Generator().manual_seed(seed)
    vector = _vector(dtype)
    if few_rows_for_k is None:
        short = torch.arange(min(width, vector) + 1).repeat_interleave(vector)
        lengths = torch.cat(
            (
                short,
                torch.arange(width + 1).repeat(2),
                torch.randint(0, width + 1, (256,), generator=generator),
            )
        )
    else:
        low = max(0, few_rows_for_k - 1)
        high = max(low, min(width, vector) + 1)
        short = torch.arange(low, high).repeat_interleave(vector)[:128]
        filler = torch.randint(
            min(few_rows_for_k + 1, width), width + 1, (128 - short.numel(),), generator=generator
        )
        lengths = torch.cat((short, filler))
    rows = lengths.numel()
    if tie_free:
        scores = torch.argsort(torch.rand(rows, width, generator=generator), dim=1).float()
    else:
        scores = torch.randn(rows, width, generator=generator)
        scores[::3] = torch.randint(-2, 3, scores[::3].shape, generator=generator).float()
    return scores.to(dtype).cuda(), lengths.to(torch.int32).cuda()


def _canonical(scores, seq_lens, topk_k):
    """Independent reference: a stable descending sort keeps lower indices first among ties."""
    rows, width = scores.shape
    columns = torch.arange(width, device=scores.device).view(1, width)
    in_row = columns < seq_lens.long().clamp(0, width).view(rows, 1)
    masked = torch.where(in_row & (scores > float("-inf")), scores, float("-inf"))
    order = torch.sort(masked.float(), dim=1, descending=True, stable=True).indices[:, :topk_k]
    valid = torch.gather(masked, 1, order) > float("-inf")
    big = torch.iinfo(torch.int64).max
    indices = torch.sort(torch.where(valid, order, big), dim=1).values
    valid = indices != big
    values = torch.gather(scores, 1, indices.clamp(max=width - 1))
    return (
        torch.where(valid, indices, -1).to(torch.int32),
        torch.where(valid, values, torch.finfo(scores.dtype).min),
    )


def _assert_a_right_top_k(result, scores, seq_lens, topk_k):
    """Every row: distinct in-row ids whose scores are the canonical top-k's scores."""
    expected_indices, expected_values = _canonical(scores, seq_lens, topk_k)
    indices = result["indices"].long()
    valid = indices >= 0
    assert torch.equal(valid.sum(dim=1), (expected_indices >= 0).sum(dim=1))
    sentinel = torch.iinfo(torch.int64).max
    ordered = torch.sort(torch.where(valid, indices, sentinel), dim=1).values
    ordered_valid = ordered != sentinel
    assert not bool((ordered_valid[:, 1:] & (ordered[:, 1:] == ordered[:, :-1])).any())
    assert not bool((valid & (indices >= seq_lens.long().view(-1, 1))).any())
    picked = torch.gather(scores, 1, indices.clamp(min=0, max=scores.size(1) - 1))
    picked = torch.where(valid, picked, torch.finfo(scores.dtype).min)
    assert torch.equal(
        torch.sort(picked.float(), dim=1, descending=True).values,
        torch.sort(expected_values.float(), dim=1, descending=True).values,
    )
    values = torch.where(valid, result["values"], torch.finfo(scores.dtype).min)
    assert torch.equal(values, picked)


def _wrong_rows(indices, scores, seq_lens, topk_k):
    """Rows whose valid ids are not exactly the row's top ``min(K, len)`` set (tie-free rows)."""
    expected, _ = _canonical(scores, seq_lens, topk_k)
    sentinel = torch.iinfo(torch.int32).max
    got = torch.sort(torch.where(indices >= 0, indices, sentinel), dim=1).values
    got = torch.where(got == sentinel, -1, got)
    return int((got != expected).any(dim=1).sum())


_BELOW_GUARD = [
    pytest.param(dtype, topk_k, width, id=f"{name}-k{topk_k}-w{width}")
    for name, dtype in _DTYPES.items()
    for topk_k in range(1, dsa_cudnn_kernels._cudnn_topk_min_reliable_k(dtype))
    for width in _WIDTHS
    if topk_k < width  # rows longer than top_k exist
]


@pytest.mark.parametrize("dtype, topk_k, width", _BELOW_GUARD)
@pytest.mark.parametrize("deterministic", [False, True], ids=["non_det", "det"])
@pytest.mark.parametrize("few_rows", [False, True], ids=["many_rows", "few_rows"])
def test_wrapper_selects_a_right_top_k_below_the_guard(
    dtype, topk_k, width, deterministic, few_rows
):
    scores, seq_lens = _rows(
        width,
        dtype,
        seed=width * 16 + topk_k,
        tie_free=False,
        few_rows_for_k=topk_k if few_rows else None,
    )
    assert bool((seq_lens > topk_k).any())

    result = dsa_cudnn_kernels._indexer_top_k_wrapper_chunked(
        scores, seq_lens, topk_k=topk_k, return_topk_scores=True, deterministic=deterministic
    )

    _assert_a_right_top_k(result, scores, seq_lens, topk_k)
    if deterministic:
        expected_indices, expected_values = _canonical(scores, seq_lens, topk_k)
        assert torch.equal(result["indices"], expected_indices)
        assert torch.equal(result["values"], expected_values)


_FROM_GUARD = [
    pytest.param(dtype, topk_k, width, id=f"{name}-k{topk_k}-w{width}")
    for name, dtype in _DTYPES.items()
    for topk_k in sorted(set(range(_min_k(dtype), _min_k(dtype) + 11)) | {24, 32, 64, 128})
    for width in _WIDTHS
    # Rows longer than top_k exist; tie-free BF16 rows need at most 256 columns.
    if topk_k < width and (dtype != torch.bfloat16 or width <= 256)
]


@pytest.mark.parametrize("dtype, topk_k, width", _FROM_GUARD)
@pytest.mark.parametrize("masked", [False, True], ids=["tail_garbage", "tail_neg_inf"])
@pytest.mark.parametrize("variant", ["many_rows", "few_rows", "tie_break"])
def test_raw_kernel_is_exact_from_the_guard_up(dtype, topk_k, width, masked, variant):
    if not masked and not dsa_cudnn_kernels._cudnn_frontend_at_least(
        dsa_cudnn_kernels._TOPK_OOB_LANE_GUARD_FE_VERSION
    ):
        pytest.skip("frontends before 1.29 may read past a row's length")
    kwargs = {}
    if variant == "tie_break":
        if not dsa_cudnn_kernels._supports_lowest_index_topk_tie_break():
            pytest.skip("this cuDNN frontend has no tie_break")
        kwargs["tie_break"] = dsa_cudnn_kernels._TOPK_LOWEST_INDEX_TIE_BREAK
    scores, seq_lens = _rows(
        width,
        dtype,
        seed=width * 1000 + topk_k,
        tie_free=True,
        few_rows_for_k=topk_k if variant == "few_rows" else None,
    )
    # Rows longer than top_k leave the kernel's trivial branch; without them nothing is checked.
    assert bool((seq_lens > topk_k).any())
    if masked:
        columns = torch.arange(width, device=scores.device).view(1, width)
        scores = scores.masked_fill(columns >= seq_lens.long().view(-1, 1), float("-inf"))

    raw = dsa_cudnn_kernels._cudnn_dsa.indexer_top_k_wrapper(
        scores.contiguous(), seq_lens, top_k=topk_k, next_n=1, return_val=False, **kwargs
    )

    assert _wrong_rows(raw["indices"], scores, seq_lens, topk_k) == 0


def test_report_raw_kernel_rows_below_the_guard():
    """Not asserted: records whether the installed frontend still gets small K wrong."""
    frontend = dsa_cudnn_kernels._cudnn_frontend_version()
    for name, dtype in _DTYPES.items():
        for topk_k in range(1, _min_k(dtype)):
            wrong = []
            for width in (7, 15, 129):
                if topk_k > width:
                    continue
                scores, seq_lens = _rows(width, dtype, seed=width + topk_k, tie_free=True)
                raw = dsa_cudnn_kernels._cudnn_dsa.indexer_top_k_wrapper(
                    scores, seq_lens, top_k=topk_k, next_n=1, return_val=False
                )
                wrong.append(f"w{width}:{_wrong_rows(raw['indices'], scores, seq_lens, topk_k)}")
            _report(f"frontend {frontend} {name} K={topk_k}: wrong rows {' '.join(wrong)}")
