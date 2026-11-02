# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Deterministic DSA top-k membership with infinite and NaN scores.

``RadixTopK`` models the cuDNN frontend radix top-k on CPU: scores are ranked by the
frontend's ordered radix key (bit pattern, so ``+0.0`` ranks above ``-0.0``), ``tie_break``
0 picks random copies of the cutoff key, 1 the lowest columns, rows no longer than K return
``0..len-1`` then ``-1``, and the output order is shuffled (the kernel does not sort).
"""

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


@pytest.mark.parametrize("canonicalize", [_canonical], ids=["full_row"])
def test_positive_infinity_is_selected_and_nan_slots_stay_empty(canonicalize):
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
        _assert_bitwise_equal(_canonical(scores, seq_lens, topk_k, True, fake), expected, True)
