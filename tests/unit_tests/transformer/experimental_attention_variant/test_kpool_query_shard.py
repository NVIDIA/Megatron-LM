# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""KPool selection with query rows split across a process group.

Ranks are simulated in one process: a fake group reports a size and rank, and the
all-gather is replaced by one that returns the shards every simulated rank produced.
"""

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant import dsa
from megatron.core.transformer.experimental_attention_variant.dsa import (
    _kpool_query_shard_bounds,
    fused_qk_topk_kpool,
)
from megatron.core.transformer.experimental_attention_variant.dsa_masking import (
    generate_varlen_mask_params_for_positions,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class _FakeGroup:
    def __init__(self, size, rank):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


def _make_inputs(sq, batch, n_heads, dim, seed):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(sq, batch, n_heads, dim, generator=gen).to(DEVICE)
    k = torch.randn(sq, batch, dim, generator=gen).to(DEVICE)
    weights = torch.randn(sq, batch, n_heads, generator=gen).to(DEVICE)
    gate = torch.randn(sq, batch, dim, generator=gen).to(DEVICE)
    ape = torch.randn(4, dim, generator=gen).to(DEVICE)
    return q, k, weights, gate, ape


def _sharded_kpool(monkeypatch, shard_size, **kwargs):
    """Run ``fused_qk_topk_kpool`` on every simulated rank; return each rank's output."""
    local_shards = {}

    def record_gather(output, local, group):
        local_shards[group.rank()] = local.clone()
        output.fill_(-7)

    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", record_gather)
    for rank in range(shard_size):
        fused_qk_topk_kpool(
            **kwargs, return_index_scores=False, query_shard_group=_FakeGroup(shard_size, rank)
        )

    def replay_gather(output, local, group):
        torch.testing.assert_close(local, local_shards[group.rank()], rtol=0, atol=0)
        output.copy_(torch.cat([local_shards[r] for r in range(group.size())], dim=0))

    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", replay_gather)
    return [
        fused_qk_topk_kpool(
            **kwargs, return_index_scores=False, query_shard_group=_FakeGroup(shard_size, rank)
        )[1]
        for rank in range(shard_size)
    ]


@pytest.mark.parametrize(
    "sq,shard_size,expected",
    [
        (16, 4, [(0, 4), (4, 8), (8, 12), (12, 16)]),
        (13, 4, [(0, 4), (4, 8), (8, 12), (12, 13)]),
        (5, 8, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 5), (5, 5), (5, 5)]),
    ],
)
def test_query_shard_bounds_cover_every_row_once(sq, shard_size, expected):
    bounds = [_kpool_query_shard_bounds(sq, shard_size, rank)[:2] for rank in range(shard_size)]
    assert bounds == expected


@pytest.mark.parametrize("shard_size", [2, 3, 8])
@pytest.mark.parametrize("sq", [5, 24, 37])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("small_score_chunks", [False, True])
def test_sharded_selection_matches_unsharded_causal(
    monkeypatch, shard_size, sq, batch, small_score_chunks
):
    if small_score_chunks:
        # A few query rows per score chunk, so shard edges fall inside the chunk loop.
        monkeypatch.setattr(dsa, "_KPOOL_SCORE_CHUNK_BYTES", 3 * batch * 2 * (sq // 4) * 4)
    q, k, weights, gate, ape = _make_inputs(sq, batch, n_heads=2, dim=8, seed=sq + shard_size)
    mask = torch.triu(torch.full((sq, sq), float("-inf"), device=DEVICE), diagonal=1)
    kwargs = dict(
        q=q,
        k=k,
        weights=weights,
        index_topk=8,
        pool_size=4,
        gate_score=gate,
        ape=ape,
        mask=mask,
        rotate_activation_enabled=False,
    )
    _, expected = fused_qk_topk_kpool(**kwargs, return_index_scores=False)
    for actual in _sharded_kpool(monkeypatch, shard_size, **kwargs):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("shard_size", [2, 4])
def test_sharded_selection_matches_unsharded_packed(monkeypatch, shard_size):
    lengths = (7, 12, 3, 9)
    total = sum(lengths)
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device=DEVICE)
    starts, ends = generate_varlen_mask_params_for_positions(cu, torch.arange(total, device=DEVICE))
    q, k, weights, gate, ape = _make_inputs(total, 1, n_heads=2, dim=8, seed=7)
    kwargs = dict(
        q=q,
        k=k,
        weights=weights,
        index_topk=8,
        pool_size=4,
        gate_score=gate,
        ape=ape,
        varlen_starts=starts,
        varlen_ends=ends,
        cu_seqlens_kv=cu,
        rotate_activation_enabled=False,
    )
    _, expected = fused_qk_topk_kpool(**kwargs, return_index_scores=False)
    for actual in _sharded_kpool(monkeypatch, shard_size, **kwargs):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_sharded_selection_scores_only_local_rows(monkeypatch):
    """Each rank scores ceil(sq / shard_size) query rows, not the full sequence."""
    sq, shard_size = 24, 4
    q, k, weights, gate, ape = _make_inputs(sq, 1, n_heads=2, dim=8, seed=11)
    scored_rows = []
    real_einsum = torch.einsum

    def counting_einsum(equation, *operands):
        if equation == 'sbhd,tbd->sbht':
            scored_rows.append(operands[0].size(0))
        return real_einsum(equation, *operands)

    monkeypatch.setattr(torch, "einsum", counting_einsum)
    _sharded_kpool(
        monkeypatch,
        shard_size,
        q=q,
        k=k,
        weights=weights,
        index_topk=8,
        pool_size=4,
        gate_score=gate,
        ape=ape,
        rotate_activation_enabled=False,
    )
    assert scored_rows and all(rows == sq // shard_size for rows in scored_rows)


def test_query_shard_group_requires_chunked_selection():
    q, k, weights, gate, ape = _make_inputs(8, 1, n_heads=2, dim=8, seed=0)
    with pytest.raises(ValueError, match="return_index_scores=False"):
        fused_qk_topk_kpool(
            q,
            k,
            weights,
            index_topk=8,
            pool_size=4,
            gate_score=gate,
            ape=ape,
            return_index_scores=True,
            query_shard_group=_FakeGroup(2, 0),
        )
