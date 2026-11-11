# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Deterministic DSA top-k membership with infinite and NaN scores.

``RadixTopK`` models the cuDNN frontend radix top-k on CPU: scores are ranked by the
frontend's ordered radix key (bit pattern, so ``+0.0`` ranks above ``-0.0``), the copies
of the cutoff key that make the top K are picked at random, rows no longer than K return
``0..len-1`` then ``-1``, and the output order is shuffled (the kernel does not sort).
"""

import functools
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
    """CPU model of ``cudnn.DSA.indexer_top_k_wrapper``."""

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
    ):
        assert next_n == 1
        self.calls.append(
            {
                "rows": input_values.size(0),
                "offset_bytes": input_values.storage_offset() * input_values.element_size(),
            }
        )
        n_rows, sk = input_values.shape
        keys = _radix_keys(input_values, self.nan_mode)
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
    raw = fake.indexer_top_k_wrapper(scores, seq_lens, topk_k, 1, False)
    return dsa_cudnn_kernels._canonicalize_deterministic_topk_membership(
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


def test_positive_infinity_is_selected_and_nan_slots_stay_empty():
    """+inf/NaN rows: a rule that drops +inf members returns the +inf rows count-short.

    +inf is the largest score and must be selected; a NaN still costs the slot the kernel
    ranked it into.
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

    actual = _canonical(scores, seq_lens, topk_k, True, RadixTopK(seed=7))

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
        _assert_bitwise_equal(_canonical(scores, seq_lens, topk_k, True, fake), expected, True)


def test_full_row_fallback_chunks_keep_32_byte_aligned_starts(monkeypatch):
    fake = RadixTopK(seed=12)
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


def test_full_row_fallback_takes_one_aligned_row_group_below_its_repair_budget(monkeypatch):
    fake = RadixTopK(seed=14)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", fake)
    # 67 FP32 columns need 8-row chunk starts, but the repair budget fits only 5 rows (as the
    # default budget does above about 4.2M columns): each launch takes one 8-row group.
    sk, topk_k = 67, 8
    monkeypatch.setattr(
        dsa_cudnn_kernels, "_DETERMINISTIC_TOPK_REPAIR_MAX_BYTES", 5 * (16 * sk + 40 * topk_k)
    )
    scores, seq_lens = _random_rows(14, "signed_zeros", n_rows=20, sk=sk)

    result = dsa_cudnn_kernels._indexer_top_k_wrapper_chunked(
        scores, seq_lens, topk_k=topk_k, return_topk_scores=True, deterministic=True
    )

    assert [call["rows"] for call in fake.calls] == [8, 8, 4]
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


@pytest.mark.parametrize("mode", ["non_det", "full_row"])
@pytest.mark.parametrize("return_topk_scores", [False, True])
def test_chunk_results_are_released_before_the_next_launch(monkeypatch, mode, return_topk_scores):
    """A chunk result alive at the next launch would sit in the freed scratch block.

    The frontend allocates a launch's outputs before its scratch, so results kept for a final
    concatenation split the previous scratch block and every later launch needed a second
    scratch-sized segment (with an 8 GiB budget, two 8 GiB blocks in a CUDA graph pool at 1M).
    """
    tracker = _ReleaseTracker()
    fake = RadixTopK(seed=21)
    real_top_k = fake.indexer_top_k_wrapper

    def top_k(input_values, seq_lens, top_k, *args, **kwargs):
        tracker.assert_released()
        return tracker.track(real_top_k(input_values, seq_lens, top_k, *args, **kwargs))

    for name in ("_canonicalize_deterministic_topk_membership",):
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
    fake.indexer_top_k_wrapper = functools.wraps(real_top_k)(top_k)
    monkeypatch.setattr(dsa_cudnn_kernels, "_TOPK_WRAPPER_MAX_SCRATCH_BYTES", 7 * 64 * 8)
    monkeypatch.setattr(dsa_cudnn_kernels, "_DETERMINISTIC_TOPK_REPAIR_MAX_BYTES", 1 << 30)

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


def test_chunked_wrapper_rejects_unexpected_chunk_dtypes(monkeypatch):
    class Int64TopK(RadixTopK):
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
