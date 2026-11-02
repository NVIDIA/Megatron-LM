# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Skipping the ``-inf`` prefill of the cuDNN packed-THD indexer score buffer.

With ``clean_logits=False`` the cuDNN scorer leaves every column at or past a row's causal
limit unspecified. The fake frontend poisons those columns (NaN or +inf); MCore must then
select exactly the same top-k as with the prefill.
"""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels
from tests.unit_tests.transformer.experimental_attention_variant import dsa_cudnn_fe_fakes as fakes

SKIP = dsa_cudnn_kernels.IndexerScorerOptions(skip_masked_score_fill=True)
KEEP = dsa_cudnn_kernels.IndexerScorerOptions(skip_masked_score_fill=False)


@pytest.fixture
def frontend(monkeypatch):
    def install(namespace, version=(1, 31), poison=float("nan")):
        if hasattr(namespace, "reset"):
            namespace.reset()
        monkeypatch.setattr(namespace, "poison", poison, raising=False)
        monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
        monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", namespace)
        monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_frontend_version", lambda: version)
        return namespace

    return install


def _packed_layout(seq_lens, cp_size, cp_rank, max_num_seqs=None):
    """Local rows, bounds and padded cu_seqlens of a zigzag-CP packed THD batch."""
    divisor = 2 * cp_size if cp_size > 1 else 1
    assert all(length % divisor == 0 for length in seq_lens)
    cu = [0]
    for length in seq_lens:
        cu.append(cu[-1] + length)
    if max_num_seqs is not None:
        cu += [cu[-1]] * (max_num_seqs + 1 - len(cu))
    starts, ends = [], []
    for seq, length in enumerate(seq_lens):
        if cp_size == 1:
            positions = range(length)
        else:
            half = length // divisor
            front = range(cp_rank * half, (cp_rank + 1) * half)
            back = range((divisor - 1 - cp_rank) * half, (divisor - cp_rank) * half)
            positions = list(front) + list(back)
        for position in positions:
            starts.append(cu[seq])
            ends.append(cu[seq] + position + 1)
    cu_tensor = torch.tensor(cu, dtype=torch.int32)
    return (
        torch.tensor(starts, dtype=torch.int64),
        torch.tensor(ends, dtype=torch.int64),
        cu_tensor,
        max(seq_lens),
    )


def _inputs(sq, sk, heads=4, dim=8, seed=0):
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn((1, sq, heads, dim), generator=generator).to(torch.bfloat16)
    k = torch.randn((1, sk, dim), generator=generator).to(torch.bfloat16)
    w = torch.rand((1, sq, heads), generator=generator).to(torch.bfloat16)
    return q, k, w


def _score(
    q, k, w, starts, ends, cu, max_seqlen, cp_size, cp_rank, topk, return_topk_scores, options
):
    return dsa_cudnn_kernels._indexer_topk_bshd(
        q,
        k,
        w,
        topk,
        varlen_starts=starts,
        varlen_ends=ends,
        return_scores=False,
        return_topk_scores=return_topk_scores,
        use_local_indexer_varlen=True,
        packed_cu_seqlens_q=cu,
        packed_cu_seqlens_k=cu,
        packed_max_seqlen_q=max_seqlen,
        packed_max_seqlen_k=max_seqlen,
        packed_cp_size=cp_size,
        local_packed_cp_rank=cp_rank,
        scorer_options=options,
    )


def test_clean_logits_probe_requires_named_parameter_and_topk_guard(frontend):
    frontend(fakes.Fe131CleanLogitsDSA, (1, 31))
    assert dsa_cudnn_kernels._supports_indexer_clean_logits()

    frontend(fakes.Fe131DSA, (1, 31))
    assert not dsa_cudnn_kernels._supports_indexer_clean_logits()

    # An older frontend that only added clean_logits still has the unguarded radix top-k.
    frontend(fakes.Fe131CleanLogitsDSA, (1, 21))
    assert not dsa_cudnn_kernels._supports_indexer_clean_logits()

    class KwargsOnly:
        @staticmethod
        def indexer_forward_wrapper(q, k, w, **kwargs):
            raise AssertionError("not called")

    frontend(KwargsOnly, (1, 31))
    assert not dsa_cudnn_kernels._supports_indexer_clean_logits()


def test_scorer_options_follow_the_config():
    config = SimpleNamespace(dsa_indexer_skip_masked_score_fill=True)
    assert dsa_cudnn_kernels._scorer_options_from_config(config).skip_masked_score_fill
    config.dsa_indexer_skip_masked_score_fill = False
    assert not dsa_cudnn_kernels._scorer_options_from_config(config).skip_masked_score_fill
    assert not dsa_cudnn_kernels._scorer_options_from_config(None).skip_masked_score_fill


@pytest.mark.parametrize("poison", [float("nan"), float("inf"), 1.0e30])
@pytest.mark.parametrize("return_topk_scores", [False, True])
@pytest.mark.parametrize("topk", [3, 64], ids=["ranked", "full_cover"])
@pytest.mark.parametrize(
    "seq_lens, cp_size, cp_rank, max_num_seqs",
    [
        ((8, 4, 12), 1, 0, None),
        ((8, 4, 12), 1, 0, 6),
        ((8, 16), 2, 0, None),
        ((8, 16), 2, 1, 4),
        ((16, 8, 8), 4, 3, None),
    ],
)
def test_skipping_the_prefill_selects_identical_topk(
    frontend, poison, return_topk_scores, topk, seq_lens, cp_size, cp_rank, max_num_seqs
):
    starts, ends, cu, max_seqlen = _packed_layout(seq_lens, cp_size, cp_rank, max_num_seqs)
    sq = starts.numel()
    q, k, w = _inputs(sq, sq * cp_size, seed=sum(seq_lens) + cp_rank)
    args = (q, k, w, starts, ends, cu, max_seqlen, cp_size, cp_rank, topk, return_topk_scores)

    frontend(fakes.Fe131DSA)
    reference = _score(*args, KEEP)

    dsa = frontend(fakes.Fe131CleanLogitsDSA, poison=poison)
    skipped = _score(*args, SKIP)
    assert [call["clean_logits"] for call in dsa.calls] == [False]

    for expected, actual in zip(reference, skipped):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_skip_is_ignored_without_frontend_support(frontend):
    starts, ends, cu, max_seqlen = _packed_layout((8, 8), 1, 0)
    q, k, w = _inputs(16, 16)
    dsa = frontend(fakes.Fe131DSA)
    _score(q, k, w, starts, ends, cu, max_seqlen, 1, 0, 3, False, SKIP)
    assert [call["clean_logits"] for call in dsa.calls] == [True]


def test_keep_option_never_requests_clean_logits(frontend):
    starts, ends, cu, max_seqlen = _packed_layout((8, 8), 1, 0)
    q, k, w = _inputs(16, 16)
    dsa = frontend(fakes.Fe131CleanLogitsDSA)
    _score(q, k, w, starts, ends, cu, max_seqlen, 1, 0, 3, False, KEEP)
    assert [call["clean_logits"] for call in dsa.calls] == [True]


@pytest.mark.parametrize("topk", [3, 64], ids=["ranked", "full_cover"])
def test_rows_past_the_last_cu_seqlens_entry_select_nothing(frontend, topk):
    """A pack whose cu_seqlens do not cover every row fails closed instead of reading poison."""
    starts, ends, cu, max_seqlen = _packed_layout((8, 4), 1, 0)
    # Four trailing rows that the bounds treat as part of the last sequence, but that no
    # cu_seqlens segment covers, so the kernel never writes them.
    starts = torch.cat((starts, torch.full((4,), 8)))
    ends = torch.cat((ends, torch.arange(13, 17)))
    q, k, w = _inputs(16, 16)
    frontend(fakes.Fe131CleanLogitsDSA, poison=float("inf"))
    indices, lengths, _ = _score(q, k, w, starts, ends, cu, max_seqlen, 1, 0, topk, False, SKIP)
    assert torch.all(lengths[0, 12:] == 0)
    assert torch.all(indices[0, 12:] == -1)
    assert torch.all(lengths[0, :12] == (ends[:12] - starts[:12]).clamp(max=min(topk, 8)))


def test_column_limits_match_the_kernel_contract():
    cu_q = torch.tensor([0, 2, 4, 4, 6], dtype=torch.int32)
    cu_k = torch.tensor([0, 2, 6, 6, 8], dtype=torch.int32)
    offsets = torch.tensor([0, 2, 0, 0], dtype=torch.int32)
    limits = dsa_cudnn_kernels._packed_thd_score_column_limits(
        cu_q, cu_k, offsets, num_rows=8, ratio=1
    )
    torch.testing.assert_close(limits, torch.tensor([1, 2, 3, 4, 1, 2, 0, 0]))
    no_offsets = dsa_cudnn_kernels._packed_thd_score_column_limits(
        cu_q, cu_k, None, num_rows=8, ratio=1
    )
    torch.testing.assert_close(no_offsets, torch.tensor([1, 2, 1, 2, 1, 2, 0, 0]))


def test_skip_path_does_not_synchronize_with_the_host(frontend, monkeypatch):
    starts, ends, cu, max_seqlen = _packed_layout((8, 16), 2, 1, 4)
    sq = starts.numel()
    q, k, w = _inputs(sq, sq * 2)
    frontend(fakes.Fe131CleanLogitsDSA)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("host synchronization in the packed THD scorer")

    monkeypatch.setattr(torch.Tensor, "item", forbidden)
    monkeypatch.setattr(torch.Tensor, "tolist", forbidden)
    _score(q, k, w, starts, ends, cu, max_seqlen, 2, 1, 3, True, SKIP)
