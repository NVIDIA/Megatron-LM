# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Deterministic DSA top-k from cuDNN ``tie_break=1`` versus the full-row canonical rule.

``RadixTopK`` models the cuDNN frontend radix top-k on CPU: scores are ranked by the
frontend's ordered radix key (bit pattern, so ``+0.0`` ranks above ``-0.0``), ``tie_break``
0 picks random copies of the cutoff key, 1 the lowest columns, rows no longer than K return
``0..len-1`` then ``-1``, and the output order is shuffled (the kernel does not sort).
"""

import functools
import logging
import weakref

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels

_INT64_MAX = torch.iinfo(torch.int64).max


def _radix_keys(scores: torch.Tensor, nan_mode: str) -> torch.Tensor:
    """cuDNN ``to_ordered`` key for FP32 scores; smaller keys rank higher."""
    bits = scores.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    keys = torch.where(bits >= 0x80000000, bits, (~bits) & 0x7FFFFFFF)
    if nan_mode == "top":
        # Another legal NaN placement: every NaN above every number (fp16 coarse bins).
        keys = torch.where(torch.isnan(scores), torch.full_like(keys, -1), keys)
    else:
        assert nan_mode == "bits"
    return keys


class RadixTopK:
    """CPU model of ``cudnn.DSA.indexer_top_k_wrapper`` (frontend 1.31 signature)."""

    def __init__(self, seed: int = 0, nan_mode: str = "bits"):
        self.generator = torch.Generator().manual_seed(seed)
        self.nan_mode = nan_mode
        self.calls = []

    def indexer_top_k_wrapper(
        self,
        input_values,
        seq_lens,
        top_k,
        next_n=1,
        return_val=True,
        num_copy_bits=256,
        stream=None,
        tie_break=0,
    ):
        assert next_n == 1 and tie_break in (0, 1)
        self.calls.append(
            {
                "rows": input_values.size(0),
                "tie_break": tie_break,
                "val": return_val,
                "offset_bytes": input_values.storage_offset() * input_values.element_size(),
            }
        )
        n_rows, sk = input_values.shape
        keys = _radix_keys(input_values, self.nan_mode)
        if tie_break == 1:
            tie_rank = torch.arange(sk, dtype=torch.int64).expand(n_rows, sk)
        else:
            tie_rank = torch.argsort(torch.rand((n_rows, sk), generator=self.generator), dim=1)
        lengths = seq_lens.to(torch.int64).view(n_rows, 1)
        columns = torch.arange(sk).view(1, sk)
        composite = torch.where(columns < lengths, keys * sk + tie_rank, _INT64_MAX)
        indices = torch.argsort(composite, dim=1)[:, :top_k].to(torch.int32)
        shuffle = torch.argsort(torch.rand((n_rows, top_k), generator=self.generator), dim=1)
        indices = torch.gather(indices, 1, shuffle)
        trivial = lengths.view(n_rows) <= top_k
        prefix = torch.arange(top_k, dtype=torch.int32).view(1, top_k)
        prefix = torch.where(prefix < lengths, prefix, torch.full_like(prefix, -1))
        indices = torch.where(trivial.view(n_rows, 1), prefix, indices)
        values = None
        if return_val:
            values = torch.gather(input_values, 1, indices.clamp(min=0).to(torch.int64))
            values = torch.where(indices >= 0, values, torch.full_like(values, float("-inf")))
        return {"indices": indices, "values": values}


class LegacyTopK(RadixTopK):
    """Frontend 1.21.1 signature: no ``tie_break``."""

    def indexer_top_k_wrapper(
        self,
        input_values,
        seq_lens,
        top_k,
        next_n=1,
        return_val=True,
        num_copy_bits=256,
        stream=None,
    ):
        return super().indexer_top_k_wrapper(
            input_values, seq_lens, top_k, next_n, return_val, num_copy_bits, stream
        )


class KwargsTopK(RadixTopK):
    """A wrapper that forwards ``**kwargs`` cannot be trusted to honor ``tie_break``."""

    def indexer_top_k_wrapper(self, input_values, seq_lens, top_k, **kwargs):
        return super().indexer_top_k_wrapper(input_values, seq_lens, top_k, **kwargs)


def _bits(values: torch.Tensor) -> torch.Tensor:
    return values.contiguous().view(torch.int32)


def _assert_bitwise_equal(actual: dict, expected: dict, return_topk_scores: bool) -> None:
    assert actual["indices"].dtype == expected["indices"].dtype == torch.int32
    torch.testing.assert_close(actual["indices"], expected["indices"], rtol=0, atol=0)
    if return_topk_scores:
        # Compare bit patterns: assert_close treats -0.0 == +0.0.
        assert torch.equal(_bits(actual["values"]), _bits(expected["values"]))
    else:
        assert actual["values"] is None and expected["values"] is None


def _canonical(scores, seq_lens, topk_k, return_topk_scores, fake):
    raw = fake.indexer_top_k_wrapper(scores, seq_lens, topk_k, 1, False, tie_break=0)
    return dsa_cudnn_kernels._canonicalize_deterministic_topk_membership(
        scores, seq_lens, raw["indices"], topk_k=topk_k, return_topk_scores=return_topk_scores
    )


def _tie_break(scores, seq_lens, topk_k, return_topk_scores, fake):
    raw = fake.indexer_top_k_wrapper(scores, seq_lens, topk_k, 1, False, tie_break=1)
    return dsa_cudnn_kernels._canonicalize_lowest_index_tie_break_topk(
        scores, seq_lens, raw["indices"], topk_k=topk_k, return_topk_scores=return_topk_scores
    )


def _random_rows(seed: int, distribution: str, n_rows: int = 48, sk: int = 96):
    generator = torch.Generator().manual_seed(seed)
    if distribution == "gauss":
        scores = torch.randn((n_rows, sk), generator=generator)
    elif distribution == "heavy_ties":
        scores = torch.round(torch.randn((n_rows, sk), generator=generator) * 4) / 64
    elif distribution == "signed_zeros":
        # Mostly zeros of both signs plus a few positives: the cutoff is usually zero.
        choices = torch.tensor([0.0, -0.0, 0.25, -0.5, 1.0])
        weights = torch.tensor([10.0, 10.0, 1.0, 3.0, 1.0])
        picks = torch.multinomial(weights, n_rows * sk, replacement=True, generator=generator)
        scores = choices[picks].view(n_rows, sk)
    elif distribution == "relu_heads":
        # GLM index score sum_h w_h * relu(q_h . k), accumulated from the first head's term:
        # rows whose head weights are all negative score -0.0 wherever every logit is <= 0.
        heads = 4
        logits = torch.round(torch.randn((n_rows, heads, sk), generator=generator) * 2) / 2
        exact_negative_zero = torch.rand((n_rows, heads, sk), generator=generator) < 0.1
        logits = torch.where(exact_negative_zero, torch.full_like(logits, -0.0), logits)
        weights = torch.randn((n_rows, heads, 1), generator=generator).sign() * 0.5
        weights[::2] = -weights[::2].abs()
        terms = torch.relu(logits) * weights
        scores = terms[:, 0]
        for head in range(1, heads):
            scores = scores + terms[:, head]
    elif distribution == "denormal_zeros":
        # Zeros of both signs next to +-denormals: denormals must never count as zero ties.
        tiny = torch.finfo(torch.float32).tiny
        choices = torch.tensor([0.0, -0.0, tiny / 4, -tiny / 4, tiny / 1024, -tiny / 1024, 0.5])
        weights = torch.tensor([8.0, 8.0, 1.0, 1.0, 1.0, 1.0, 0.5])
        picks = torch.multinomial(weights, n_rows * sk, replacement=True, generator=generator)
        scores = choices[picks].view(n_rows, sk)
        assert bool(((scores != 0) & (scores.abs() < tiny)).any()), "denormals were flushed"
    elif distribution == "specials":
        choices = torch.tensor(
            [0.0, -0.0, 1.0, -1.0, 2.0, float("nan"), -float("nan"), float("inf"), -float("inf")]
        )
        weights = torch.tensor([4.0, 4.0, 3.0, 3.0, 2.0, 0.7, 0.7, 0.4, 0.8])
        picks = torch.multinomial(weights, n_rows * sk, replacement=True, generator=generator)
        scores = choices[picks].view(n_rows, sk)
    else:
        raise AssertionError(distribution)
    scores = scores.to(torch.float32).contiguous()
    seq_lens = torch.randint(0, sk + 1, (n_rows,), generator=generator, dtype=torch.int32)
    seq_lens[:4] = torch.tensor([sk, 0, 1, sk], dtype=torch.int32)
    # Past each row length the buffer is undefined (clean_logits=False): poison it.
    columns = torch.arange(sk).view(1, sk)
    garbage = torch.where(
        torch.rand((n_rows, sk), generator=generator) < 0.5,
        torch.full((n_rows, sk), float("nan")),
        torch.full((n_rows, sk), -0.0),
    )
    scores = torch.where(columns < seq_lens.view(n_rows, 1), scores, garbage)
    return scores, seq_lens


@pytest.mark.parametrize(
    "distribution",
    ["gauss", "heavy_ties", "signed_zeros", "relu_heads", "denormal_zeros", "specials"],
)
@pytest.mark.parametrize("topk_k", [1, 7, 32, 95, 96])
@pytest.mark.parametrize("nan_mode", ["bits", "top"])
@pytest.mark.parametrize("seed", range(3))
def test_tie_break_topk_matches_canonical_rule_bitwise(distribution, topk_k, nan_mode, seed):
    scores, seq_lens = _random_rows(seed, distribution)
    fake = RadixTopK(seed=seed + 100, nan_mode=nan_mode)

    expected = _canonical(scores, seq_lens, topk_k, True, fake)
    actual = _tie_break(scores, seq_lens, topk_k, True, fake)

    _assert_bitwise_equal(actual, expected, True)
    # The full-row rule is itself independent of how the raw kernel broke ties.
    raw_tb1 = fake.indexer_top_k_wrapper(scores, seq_lens, topk_k, 1, False, tie_break=1)
    _assert_bitwise_equal(
        dsa_cudnn_kernels._canonicalize_deterministic_topk_membership(
            scores, seq_lens, raw_tb1["indices"], topk_k=topk_k, return_topk_scores=True
        ),
        expected,
        True,
    )


def test_heavy_tie_rows_keep_the_lowest_index_cutoff_ties():
    sk, topk_k = 4096, 2048
    generator = torch.Generator().manual_seed(7)
    # ~13 distinct values per row: about 300 copies of the cutoff value per row.
    scores = (torch.randint(-6, 7, (8, sk), generator=generator).to(torch.float32)) / 64
    seq_lens = torch.tensor([sk, sk - 1, 3000, 2049, 2048, 2047, 1, 0], dtype=torch.int32)
    fake = RadixTopK(seed=3)

    actual = _tie_break(scores, seq_lens, topk_k, True, fake)
    expected = _canonical(scores, seq_lens, topk_k, True, fake)

    _assert_bitwise_equal(actual, expected, True)
    row = scores[0]
    cutoff = torch.sort(row, descending=True).values[topk_k - 1]
    above = int((row > cutoff).sum())
    ties = torch.nonzero(row == cutoff).view(-1)[: topk_k - above]
    want = torch.sort(torch.cat((torch.nonzero(row > cutoff).view(-1), ties))).values
    torch.testing.assert_close(actual["indices"][0].to(torch.int64), want, rtol=0, atol=0)


_POS = 1.0
_NEG = -1.0
_PZ = 0.0
_NZ = -0.0


@pytest.mark.parametrize(
    "row,seq_len,topk_k,expected_indices",
    [
        # +0 cutoff: radix keeps the +0 copies (2, 4); the rule keeps the lowest zeros (1, 2).
        ([_POS, _NZ, _PZ, _NZ, _PZ, 2.0, _NEG], 7, 4, [0, 1, 2, 5]),
        # -0 cutoff: radix keeps +0 (3) and the lowest -0 (0); the rule keeps zeros 0 and 1.
        ([_NZ, _NZ, _NEG, _PZ, _NZ], 5, 2, [0, 1]),
        # Every score is a zero of either sign.
        ([_PZ, _NZ, _PZ, _NZ, _PZ, _NZ, _PZ, _NZ], 8, 3, [0, 1, 2]),
        ([_NZ] * 8, 8, 5, [0, 1, 2, 3, 4]),
        ([_PZ] * 8, 8, 5, [0, 1, 2, 3, 4]),
        # Only -0 before the +0 copies.
        ([_NZ, _NZ, _NZ, _PZ, _PZ, _PZ], 6, 3, [0, 1, 2]),
        # NaN and +inf rank above 3.0 and take two of the K slots. +inf is a score and stays;
        # the rule drops the NaN without refilling its slot, and keeps 3.0 plus the
        # lowest-index zeros of the remaining slots.
        ([float("-inf"), _NZ, float("nan"), _PZ, float("inf"), _NZ, 3.0, _PZ], 8, 4, [1, 4, 6]),
        (
            [float("-inf"), _NZ, float("nan"), _PZ, float("inf"), _NZ, 3.0, _PZ],
            8,
            6,
            [1, 3, 4, 5, 6],
        ),
        # More +inf copies than K: the lowest-index ones, like any other cutoff value.
        ([1.0, float("inf"), 2.0, float("inf"), float("inf"), float("inf")], 6, 2, [1, 3]),
        # Zero cutoff on a row no longer than K: every finite in-row score, no repair.
        ([_NZ, _PZ, _NEG, _NZ], 4, 4, [0, 1, 2, 3]),
        # Zeros past the row length are not ties.
        ([_PZ, 5.0, _NEG, _NZ, _NZ, _PZ], 3, 2, [0, 1]),
        # A negative cutoff: every zero is strictly above it.
        ([_NZ, _NEG, _PZ, -2.0, _NEG, _NZ], 6, 4, [0, 1, 2, 5]),
    ],
)
@pytest.mark.parametrize("nan_mode", ["bits", "top"])
def test_signed_zero_cutoffs_follow_value_ties(row, seq_len, topk_k, expected_indices, nan_mode):
    scores = torch.tensor([row], dtype=torch.float32)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32)
    fake = RadixTopK(seed=11, nan_mode=nan_mode)

    actual = _tie_break(scores, seq_lens, topk_k, True, fake)

    padded = expected_indices + [-1] * (topk_k - len(expected_indices))
    torch.testing.assert_close(
        actual["indices"], torch.tensor([padded], dtype=torch.int32), rtol=0, atol=0
    )
    expected_values = [row[i] if i >= 0 else torch.finfo(torch.float32).min for i in padded]
    assert torch.equal(_bits(actual["values"]), _bits(torch.tensor([expected_values])))
    _assert_bitwise_equal(actual, _canonical(scores, seq_lens, topk_k, True, fake), True)


def _sort_reference(scores, seq_lens, topk_k):
    """Independent rule: drop NaN and -inf, take the top in-row scores by value with the
    lowest index first among equal values (so -0.0 ties +0.0), minus one slot per positive
    NaN the radix kernel ranks into the top K, then sort by index; pad -1 / finfo.min."""
    n_rows, sk = scores.shape
    columns = torch.arange(sk).view(1, sk)
    in_row = columns < seq_lens.view(n_rows, 1).long()
    selectable = in_row & (scores > float("-inf"))
    keyed = torch.where(selectable, scores + 0.0, torch.full_like(scores, float("-inf")))
    order = torch.sort(keyed, dim=1, descending=True, stable=True).indices
    positive_nan = in_row & torch.isnan(scores) & (_bits(scores) >= 0)
    long_row = seq_lens.long() > topk_k
    budget = torch.where(
        long_row, topk_k - positive_nan.sum(dim=1).clamp(max=topk_k), selectable.sum(dim=1)
    )
    budget = torch.minimum(budget, selectable.sum(dim=1))
    keep = torch.arange(sk).view(1, sk) < budget.view(n_rows, 1)
    chosen = torch.sort(torch.where(keep, order, sk), dim=1).values[:, :topk_k]
    indices = torch.where(chosen == sk, -1, chosen).to(torch.int32)
    values = torch.gather(scores, 1, indices.clamp(min=0).long())
    values = torch.where(indices >= 0, values, torch.finfo(torch.float32).min)
    return {"indices": indices, "values": values}


@pytest.mark.parametrize("canonicalize", [_tie_break, _canonical], ids=["tie_break", "full_row"])
def test_positive_infinity_is_selected_and_nan_slots_stay_empty(canonicalize):
    """Head-to-head N2 rows: +inf members were dropped, so +inf rows came back count-short.

    +inf is the largest score and must be selected (by both the ``tie_break=1`` derivation and
    the full-row rule it reproduces); a NaN still costs the slot the kernel ranked it into.
    """
    sk, topk_k = 64, 16
    scores = torch.randn((6, sk), generator=torch.Generator().manual_seed(5))
    scores[0, [0, sk // 7, sk // 3, sk // 2, sk - 1]] = float("inf")  # +inf x5, no NaN
    scores[1, 10] = float("nan")  # NaN inside the row: one slot short, by rule
    scores[2, [3, 9]] = float("inf")
    scores[2, 20] = float("nan")
    scores[3, :] = float("inf")  # more +inf than K: the lowest-index copies
    scores[4, :8] = float("inf")  # short row (len <= K) of +inf and finite scores
    scores[5, [1, 2]] = float("-inf")  # -inf is the mask value and is never selected
    seq_lens = torch.tensor([sk, sk, sk, sk, 12, sk], dtype=torch.int32)

    actual = canonicalize(scores, seq_lens, topk_k, True, RadixTopK(seed=7))

    _assert_bitwise_equal(actual, _sort_reference(scores, seq_lens, topk_k), True)
    counts = (actual["indices"] >= 0).sum(dim=1).tolist()
    assert counts == [topk_k, topk_k - 1, topk_k - 1, topk_k, 12, topk_k]
    assert {0, sk // 7, sk // 3, sk // 2, sk - 1} <= set(actual["indices"][0].tolist())
    assert actual["indices"][3].tolist() == list(range(topk_k))
    assert not {1, 2} & set(actual["indices"][5].tolist())


@pytest.mark.parametrize("seed", range(4))
def test_specials_match_an_independent_sort_reference(seed):
    """Random rows of +-0, +-inf, positive NaN and heavy ties against a sort reference."""
    generator = torch.Generator().manual_seed(seed)
    n_rows, sk, topk_k = 40, 96, 24
    choices = torch.tensor([0.0, -0.0, 1.0, -1.0, 2.0, float("nan"), float("inf"), float("-inf")])
    weights = torch.tensor([4.0, 4.0, 3.0, 3.0, 2.0, 0.3, 0.6, 0.8])
    picks = torch.multinomial(weights, n_rows * sk, replacement=True, generator=generator)
    scores = choices[picks].view(n_rows, sk).contiguous()
    seq_lens = torch.randint(0, sk + 1, (n_rows,), generator=generator, dtype=torch.int32)
    expected = _sort_reference(scores, seq_lens, topk_k)

    for nan_mode in ("bits", "top"):
        fake = RadixTopK(seed=seed, nan_mode=nan_mode)
        _assert_bitwise_equal(_tie_break(scores, seq_lens, topk_k, True, fake), expected, True)
        _assert_bitwise_equal(_canonical(scores, seq_lens, topk_k, True, fake), expected, True)


def test_zero_cutoff_rows_preserve_negative_zero_values():
    scores = torch.tensor([[_NZ, _PZ, _NZ, _PZ, 1.0]], dtype=torch.float32)
    seq_lens = torch.tensor([5], dtype=torch.int32)

    actual = _tie_break(scores, seq_lens, 3, True, RadixTopK())

    torch.testing.assert_close(
        actual["indices"], torch.tensor([[0, 1, 4]], dtype=torch.int32), rtol=0, atol=0
    )
    assert torch.equal(_bits(actual["values"]), _bits(torch.tensor([[_NZ, _PZ, 1.0]])))


@pytest.mark.parametrize("seq_len", [0, 1, 6, 7, 8, 9, 12])
@pytest.mark.parametrize("return_topk_scores", [False, True])
def test_short_rows_pad_with_minus_one_and_finfo_min(seq_len, return_topk_scores):
    scores = torch.tensor(
        [[3.0, _NZ, float("-inf"), 1.0, _PZ, float("nan"), 2.0, 4.0, 0.5, 9.0, 9.0, 9.0]]
    )
    seq_lens = torch.tensor([seq_len], dtype=torch.int32)
    fake = RadixTopK(seed=5)

    actual = _tie_break(scores, seq_lens, 8, return_topk_scores, fake)
    expected = _canonical(scores, seq_lens, 8, return_topk_scores, fake)

    _assert_bitwise_equal(actual, expected, return_topk_scores)
    valid = actual["indices"] >= 0
    assert torch.all(actual["indices"][valid] < seq_len)
    if return_topk_scores:
        assert torch.all(actual["values"][~valid] == torch.finfo(torch.float32).min)


def test_tie_break_topk_rejects_invalid_geometry():
    scores = torch.zeros((2, 4))
    with pytest.raises(RuntimeError, match="geometry"):
        dsa_cudnn_kernels._canonicalize_lowest_index_tie_break_topk(
            scores,
            torch.tensor([4, 4], dtype=torch.int32),
            torch.zeros((2, 5), dtype=torch.int32),
            topk_k=5,
            return_topk_scores=False,
        )
    with pytest.raises(RuntimeError, match="unexpected number of elements"):
        dsa_cudnn_kernels._canonicalize_lowest_index_tie_break_topk(
            scores,
            torch.tensor([4, 4], dtype=torch.int32),
            torch.zeros((2, 3), dtype=torch.int32),
            topk_k=2,
            return_topk_scores=False,
        )


def test_tie_break_topk_does_not_synchronize_with_the_host(monkeypatch):
    scores, seq_lens = _random_rows(0, "signed_zeros")
    fake = RadixTopK(seed=1)
    raw = fake.indexer_top_k_wrapper(scores, seq_lens, 32, 1, False, tie_break=1)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("host synchronization in the deterministic top-k")

    for name in ("item", "tolist", "__bool__", "__int__", "nonzero"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    monkeypatch.setattr(torch, "nonzero", forbidden)

    result = dsa_cudnn_kernels._canonicalize_lowest_index_tie_break_topk(
        scores, seq_lens, raw["indices"], topk_k=32, return_topk_scores=True
    )

    assert result["indices"].shape == (scores.size(0), 32)


@pytest.fixture
def frontend_131(monkeypatch):
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_frontend_version", lambda: (1, 31))
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: False)


@pytest.mark.parametrize("return_topk_scores", [False, True])
@pytest.mark.parametrize("distribution", ["heavy_ties", "signed_zeros", "specials"])
def test_wrapper_uses_frontend_tie_break_in_deterministic_mode(
    monkeypatch, frontend_131, return_topk_scores, distribution
):
    scores, seq_lens = _random_rows(4, distribution, n_rows=40, sk=64)
    fake = RadixTopK(seed=9)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", fake)

    assert dsa_cudnn_kernels._supports_lowest_index_topk_tie_break()
    actual = dsa_cudnn_kernels._indexer_top_k_wrapper_chunked(
        scores, seq_lens, topk_k=16, return_topk_scores=return_topk_scores, deterministic=True
    )

    assert [call["tie_break"] for call in fake.calls] == [1]
    assert [call["val"] for call in fake.calls] == [False]
    expected = _canonical(scores, seq_lens, 16, return_topk_scores, RadixTopK(seed=2))
    _assert_bitwise_equal(actual, expected, return_topk_scores)


def test_wrapper_tie_break_chunks_match_a_single_launch(monkeypatch, frontend_131):
    scores, seq_lens = _random_rows(8, "signed_zeros", n_rows=40, sk=64)
    fake = RadixTopK(seed=3)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", fake)
    single = dsa_cudnn_kernels._indexer_top_k_wrapper_chunked(
        scores, seq_lens, topk_k=16, return_topk_scores=True, deterministic=True
    )
    fake.calls.clear()
    # 64 columns * 2 * 4 bytes per row: a 7-row budget, rounded to the 1-row alignment.
    monkeypatch.setattr(dsa_cudnn_kernels, "_TOPK_WRAPPER_MAX_SCRATCH_BYTES", 7 * 64 * 8)

    chunked = dsa_cudnn_kernels._indexer_top_k_wrapper_chunked(
        scores, seq_lens, topk_k=16, return_topk_scores=True, deterministic=True
    )

    assert [call["rows"] for call in fake.calls] == [7] * 5 + [5]
    _assert_bitwise_equal(chunked, single, True)


def test_wrapper_tie_break_path_drops_the_full_row_repair_chunk_cap(monkeypatch, frontend_131):
    fake = RadixTopK()
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", fake)
    # Full-row repair would allow 512 MiB / (16 * sk + 40 * K) = 2 rows per launch here.
    monkeypatch.setattr(
        dsa_cudnn_kernels, "_DETERMINISTIC_TOPK_REPAIR_MAX_BYTES", 2 * 16 * 64 + 2 * 40 * 8
    )
    scores, seq_lens = _random_rows(1, "gauss", n_rows=12, sk=64)

    dsa_cudnn_kernels._indexer_top_k_wrapper_chunked(
        scores, seq_lens, topk_k=8, return_topk_scores=False, deterministic=True
    )

    assert [call["rows"] for call in fake.calls] == [12]


@pytest.mark.parametrize(
    "fake_type,version,expect_tie_break",
    [
        (RadixTopK, (1, 31), True),
        (RadixTopK, (1, 32), True),
        (RadixTopK, (1, 30), False),
        (RadixTopK, None, False),
        (LegacyTopK, (1, 21), False),
        (LegacyTopK, (1, 31), False),
        (KwargsTopK, (1, 31), False),
    ],
)
def test_capability_probe_and_full_row_fallback(monkeypatch, fake_type, version, expect_tie_break):
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_frontend_version", lambda: version)
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: False)
    fake = fake_type(seed=6)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", fake)
    monkeypatch.setattr(
        dsa_cudnn_kernels, "_DETERMINISTIC_TOPK_REPAIR_MAX_BYTES", 2 * 16 * 64 + 2 * 40 * 8
    )
    scores, seq_lens = _random_rows(2, "signed_zeros", n_rows=6, sk=64)

    assert dsa_cudnn_kernels._supports_lowest_index_topk_tie_break() is expect_tie_break
    result = dsa_cudnn_kernels._indexer_top_k_wrapper_chunked(
        scores, seq_lens, topk_k=8, return_topk_scores=True, deterministic=True
    )

    if expect_tie_break:
        assert [(call["rows"], call["tie_break"]) for call in fake.calls] == [(6, 1)]
    else:
        # The full-row canonicalization keeps its repair-sized chunks and the default tie_break.
        assert [(call["rows"], call["tie_break"]) for call in fake.calls] == [(2, 0)] * 3
    _assert_bitwise_equal(result, _canonical(scores, seq_lens, 8, True, RadixTopK(seed=4)), True)


def test_probe_is_false_without_a_namespace(monkeypatch):
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_frontend_version", lambda: (1, 31))

    assert not dsa_cudnn_kernels._supports_lowest_index_topk_tie_break()


def test_non_deterministic_mode_never_requests_a_tie_break(monkeypatch, frontend_131):
    fake = RadixTopK()
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", fake)
    scores, seq_lens = _random_rows(3, "gauss", n_rows=5, sk=64)

    result = dsa_cudnn_kernels._indexer_top_k_wrapper_chunked(
        scores, seq_lens, topk_k=8, return_topk_scores=True
    )

    assert [(call["tie_break"], call["val"]) for call in fake.calls] == [(0, True)]
    assert result["values"] is not None


def test_torch_deterministic_algorithms_select_the_tie_break_path(monkeypatch):
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_frontend_version", lambda: (1, 31))
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: True)
    fake = RadixTopK()
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", fake)
    scores, seq_lens = _random_rows(5, "heavy_ties", n_rows=5, sk=64)

    dsa_cudnn_kernels._indexer_top_k_wrapper_chunked(
        scores, seq_lens, topk_k=8, return_topk_scores=False
    )

    assert [call["tie_break"] for call in fake.calls] == [1]


def test_full_row_fallback_chunks_keep_32_byte_aligned_starts(monkeypatch):
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_frontend_version", lambda: (1, 21))
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: False)
    fake = LegacyTopK(seed=12)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", fake)
    # 67 FP32 columns = 268 bytes per row: chunk starts must be multiples of 8 rows. The repair
    # budget allows 13 rows, which would start the second chunk 3484 bytes (28 mod 32) in.
    sk, topk_k = 67, 8
    monkeypatch.setattr(
        dsa_cudnn_kernels, "_DETERMINISTIC_TOPK_REPAIR_MAX_BYTES", 13 * (16 * sk + 40 * topk_k)
    )
    scores, seq_lens = _random_rows(13, "signed_zeros", n_rows=30, sk=sk)

    result = dsa_cudnn_kernels._indexer_top_k_wrapper_chunked(
        scores, seq_lens, topk_k=topk_k, return_topk_scores=True, deterministic=True
    )

    assert [call["rows"] for call in fake.calls] == [8, 8, 8, 6]
    assert all(call["offset_bytes"] % 32 == 0 for call in fake.calls)
    _assert_bitwise_equal(
        result, _canonical(scores, seq_lens, topk_k, True, RadixTopK(seed=1)), True
    )


class _ReleaseTracker:
    """Weak references to every chunk-level tensor; all must be dead at the next launch."""

    def __init__(self):
        self.refs = []
        self.launches = 0

    def track(self, result):
        for tensor in (result["indices"], result["values"]):
            if tensor is not None:
                self.refs.append(weakref.ref(tensor))
        return result

    def assert_released(self):
        alive = sum(ref() is not None for ref in self.refs)
        assert alive == 0, f"{alive} chunk tensors are still alive at launch {self.launches}"
        self.launches += 1


@pytest.mark.parametrize(
    "mode,fake_type", [("non_det", RadixTopK), ("tie_break", RadixTopK), ("full_row", LegacyTopK)]
)
@pytest.mark.parametrize("return_topk_scores", [False, True])
def test_chunk_results_are_released_before_the_next_launch(
    monkeypatch, frontend_131, mode, fake_type, return_topk_scores
):
    """A chunk result alive at the next launch would sit in the freed scratch block.

    The frontend allocates a launch's outputs before its scratch, so results kept for a final
    concatenation split the previous scratch block and every later launch needed a second
    scratch-sized segment (two 8 GiB blocks in a CUDA graph pool at 1M).
    """
    tracker = _ReleaseTracker()
    fake = fake_type(seed=21)
    real_top_k = fake.indexer_top_k_wrapper

    def top_k(input_values, seq_lens, top_k, *args, **kwargs):
        tracker.assert_released()
        return tracker.track(real_top_k(input_values, seq_lens, top_k, *args, **kwargs))

    for name in (
        "_canonicalize_lowest_index_tie_break_topk",
        "_canonicalize_deterministic_topk_membership",
    ):
        canonicalize = getattr(dsa_cudnn_kernels, name)
        monkeypatch.setattr(
            dsa_cudnn_kernels,
            name,
            lambda *args, _canonicalize=canonicalize, **kwargs: tracker.track(
                _canonicalize(*args, **kwargs)
            ),
        )
    scores, seq_lens = _random_rows(17, "heavy_ties", n_rows=30, sk=64)
    deterministic = mode != "non_det"
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", fake)
    single = dsa_cudnn_kernels._indexer_top_k_wrapper_chunked(
        scores,
        seq_lens,
        topk_k=8,
        return_topk_scores=return_topk_scores,
        deterministic=deterministic,
    )
    tracker.refs.clear()
    # Keep the frontend signature visible to the probe; chunk at 7 rows.
    fake.indexer_top_k_wrapper = functools.wraps(real_top_k)(top_k)
    monkeypatch.setattr(dsa_cudnn_kernels, "_TOPK_WRAPPER_MAX_SCRATCH_BYTES", 7 * 64 * 8)
    monkeypatch.setattr(dsa_cudnn_kernels, "_DETERMINISTIC_TOPK_REPAIR_MAX_BYTES", 1 << 30)
    assert dsa_cudnn_kernels._supports_lowest_index_topk_tie_break() is (fake_type is RadixTopK)

    chunked = dsa_cudnn_kernels._indexer_top_k_wrapper_chunked(
        scores,
        seq_lens,
        topk_k=8,
        return_topk_scores=return_topk_scores,
        deterministic=deterministic,
    )

    assert tracker.launches == 5
    assert chunked["indices"].dtype == torch.int32
    if deterministic:
        _assert_bitwise_equal(chunked, single, return_topk_scores)
    else:
        # Random tie choices differ between runs; the value multiset per row does not.
        assert chunked["indices"].shape == single["indices"].shape
        if return_topk_scores:
            assert torch.equal(
                torch.sort(chunked["values"], dim=1).values,
                torch.sort(single["values"], dim=1).values,
            )


def test_chunked_wrapper_rejects_unexpected_chunk_dtypes(monkeypatch, frontend_131):
    class Int64TopK(LegacyTopK):
        def indexer_top_k_wrapper(self, input_values, seq_lens, top_k, next_n=1, return_val=True):
            result = super().indexer_top_k_wrapper(
                input_values, seq_lens, top_k, next_n, return_val
            )
            return {"indices": result["indices"].long(), "values": result["values"]}

    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", Int64TopK())
    monkeypatch.setattr(dsa_cudnn_kernels, "_TOPK_WRAPPER_MAX_SCRATCH_BYTES", 7 * 64 * 8)
    scores, seq_lens = _random_rows(3, "gauss", n_rows=20, sk=64)

    with pytest.raises(RuntimeError, match="torch.int64"):
        dsa_cudnn_kernels._indexer_top_k_wrapper_chunked(
            scores, seq_lens, topk_k=8, return_topk_scores=True
        )


@pytest.mark.parametrize(
    "fake_type,version,level,needle",
    [
        (RadixTopK, (1, 31), logging.INFO, "tie_break=1"),
        (LegacyTopK, (1, 21), logging.WARNING, "full-row canonicalization"),
        (RadixTopK, (1, 30), logging.WARNING, "full-row canonicalization"),
        (KwargsTopK, (1, 31), logging.WARNING, "names tie_break=False"),
    ],
)
def test_deterministic_path_choice_is_logged_once(
    monkeypatch, caplog, fake_type, version, level, needle
):
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_frontend_version", lambda: version)
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: False)
    monkeypatch.setattr(dsa_cudnn_kernels, "_DETERMINISTIC_TOPK_PATHS_LOGGED", set())
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", fake_type(seed=1))
    scores, seq_lens = _random_rows(6, "gauss", n_rows=6, sk=64)

    with caplog.at_level(logging.INFO, logger=dsa_cudnn_kernels.__name__):
        for deterministic in (True, True, False):
            dsa_cudnn_kernels._indexer_top_k_wrapper_chunked(
                scores, seq_lens, topk_k=8, return_topk_scores=True, deterministic=deterministic
            )

    records = [r for r in caplog.records if "Deterministic DSA top-k" in r.getMessage()]
    assert len(records) == 1
    assert records[0].levelno == level and needle in records[0].getMessage()


def test_missing_triton_on_cuda_is_logged_as_a_warning(monkeypatch, caplog):
    monkeypatch.setattr(dsa_cudnn_kernels, "_DETERMINISTIC_TOPK_PATHS_LOGGED", set())
    monkeypatch.setattr(dsa_cudnn_kernels.dsa_topk_zero_ties, "HAVE_TRITON", False)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", RadixTopK())
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_frontend_version", lambda: (1, 31))

    with caplog.at_level(logging.INFO, logger=dsa_cudnn_kernels.__name__):
        dsa_cudnn_kernels._log_deterministic_topk_path_once(True, True)
        dsa_cudnn_kernels._log_deterministic_topk_path_once(True, True)

    assert [(r.levelno, "Triton is not installed" in r.getMessage()) for r in caplog.records] == [
        (logging.WARNING, True)
    ]
