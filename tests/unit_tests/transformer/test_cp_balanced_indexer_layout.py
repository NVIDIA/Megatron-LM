# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Layout parity between the balanced CP indexer and megatron.core.context_parallel_layout.

The balanced DSA indexer packs each rank's per-sequence zigzag chunks as
``[all head chunks | all tail chunks]`` (the fused indexer kernel allows only one
segment per sequence per packed call). ``context_parallel_layout`` stores the same
zigzag layout per-sequence-interleaved (``[seq0 head, seq0 tail, seq1 head, ...]``).
These tests pin the two implementations to one canonical layout definition.
"""

import pytest
import torch

# Deliberate coupling to the framework's zigzag ownership definition: these tests exist to
# pin the balanced indexer to it. ``routes`` only exposes the segment builder privately
# today; switch to a public per-rank ownership helper once one is exported again.
from megatron.core.context_parallel_layout.routes import _build_thd_layout_segments
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.experimental_attention_variant.cp_balanced_indexer import (
    _zigzag_plan,
    prebuild_balanced_layouts,
)

_S = 262144
_CASES = [
    # (name, cu list, cp_size, capacity)
    ("uneq-2seq", [0, 163840, 262144], 16, _S),
    ("eq-8seq", [32768 * i for i in range(9)], 16, _S),
    ("eq-16seq", [16384 * i for i in range(17)], 16, _S),
    ("padded-tail", [0, 196608], 16, _S),
    ("uneq-2seq-cp32", [0, 163840, 262144], 32, _S),
    ("small", [0, 1024, 3072, 4096], 4, 4096),
]


class _StubGroup:
    def __init__(self, size, rank):
        self._size, self._rank = size, rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


def _comp_cu(cu, ratio=4):
    lens = torch.div(cu[1:] - cu[:-1], ratio, rounding_mode="floor")
    return torch.cat((torch.zeros_like(cu[:1]), torch.cumsum(lens, dim=0, dtype=torch.int32)))


def _canonical_cu(cu_list, capacity):
    return cu_list if cu_list[-1] == capacity else cu_list + [capacity]


def _canonical_rank_indices(cu_list, cp_size, cp_rank):
    """Expand the framework's canonical zigzag segments into rank-local index order."""
    segments, _ = _build_thd_layout_segments(cu_list, cp_size, cp_rank, "zigzag")
    rows = []
    for global_start, length, _ in sorted(segments, key=lambda item: item[2]):
        rows.extend(range(global_start, global_start + length))
    return torch.tensor(rows, dtype=torch.long)


@pytest.mark.parametrize("name,cu_list,cp_size,capacity", _CASES)
def test_zigzag_plan_matches_canonical_layout(name, cu_list, cp_size, capacity):
    """gather_idx must equal the canonical zigzag ownership, reordered [heads|tails]."""
    cu = torch.tensor(cu_list, dtype=torch.int32)
    l_local = capacity // cp_size
    canon_cu_list = _canonical_cu(cu_list, capacity)
    for r in range(cp_size):
        plan = _zigzag_plan(cu, _comp_cu(cu), cp_size, l_local, r, torch.device("cpu"), None)
        canon = _canonical_rank_indices(canon_cu_list, cp_size, r)
        half = plan["half"]
        g = plan["gather_idx"]
        heads, tails = g[:half], g[half:]
        pos = hoff = toff = 0
        for s, e in zip(canon_cu_list[:-1], canon_cu_list[1:]):
            c = (e - s) // (2 * cp_size)
            assert torch.equal(canon[pos : pos + c], heads[hoff : hoff + c]), (name, r, s)
            assert torch.equal(canon[pos + c : pos + 2 * c], tails[toff : toff + c]), (name, r, s)
            pos += 2 * c
            hoff += c
            toff += c
        assert pos == canon.numel() and hoff == half and toff == half
        assert torch.equal(torch.sort(canon).values, torch.sort(g).values)


@pytest.mark.parametrize("name,cu_list,cp_size,capacity", _CASES)
def test_inverse_roundtrip(name, cu_list, cp_size, capacity):
    """inv_idx must recover each rank's contiguous rows from the rank-major concat."""
    cu = torch.tensor(cu_list, dtype=torch.int32)
    l_local = capacity // cp_size
    plans = [
        _zigzag_plan(cu, _comp_cu(cu), cp_size, l_local, r, torch.device("cpu"), None)
        for r in range(cp_size)
    ]
    z = torch.cat([p["gather_idx"] for p in plans])
    for r in range(cp_size):
        mine = torch.arange(r * l_local, (r + 1) * l_local, dtype=torch.int64)
        assert torch.equal(z[plans[r]["inv_idx"]], mine), (name, r)


def _packed_params(cu_list, capacity):
    # Production padded-cu semantics: the capacity tail is merged into the LAST
    # sequence (cu_padded[-1] always equals the pack length), unlike the explicit
    # trailing-padding-sequence convention used by the canonical-layout tests.
    merged = cu_list if cu_list[-1] == capacity else cu_list[:-1] + [capacity]
    cu = torch.tensor(merged, dtype=torch.int32)
    cu_real = torch.tensor(cu_list, dtype=torch.int32)
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_real,
        cu_seqlens_kv=cu_real,
        cu_seqlens_q_padded=cu,
        cu_seqlens_kv_padded=cu,
        max_seqlen_q=capacity,
        max_seqlen_kv=capacity,
    )


@pytest.mark.parametrize(
    "name,cu_list,cp_size,capacity", [c for c in _CASES if len(c[1]) > 2 or c[1][-1] != c[3]]
)
def test_prebuild_matches_runtime_plan(name, cu_list, cp_size, capacity):
    """The data-prep prebuild (built on context_parallel_layout primitives) must
    produce exactly the plan the capture-safe runtime fallback would build."""
    l_local = capacity // cp_size
    for r in range(cp_size):
        psp = _packed_params(cu_list, capacity)
        prebuild_balanced_layouts(psp, cp_group=_StubGroup(cp_size, r))
        assert psp._dsa_cp_multi_seq is True
        prebuilt = psp._dsa_cp_balance_layout_cache[("zigzag", r)]
        cu = psp.cu_seqlens_q_padded
        ref = _zigzag_plan(cu, _comp_cu(cu), cp_size, l_local, r, torch.device("cpu"), None)
        assert prebuilt["half"] == ref["half"]
        for key in ("gather_idx", "inv_idx", "pos_head", "pos_tail"):
            assert torch.equal(prebuilt[key], ref[key]), (name, r, key)
        for key in ("head_layout", "tail_layout"):
            for a, b in zip(prebuilt[key], ref[key]):
                assert torch.equal(a.to(torch.int64), b.to(torch.int64)), (name, r, key)


def test_prebuild_single_full_seq_sets_gate_only():
    """A single sequence filling the pack uses folding + K-slice: gate False, no plan."""
    psp = _packed_params([0, _S], _S)
    prebuild_balanced_layouts(psp, cp_group=_StubGroup(16, 3))
    assert psp._dsa_cp_multi_seq is False
    assert getattr(psp, "_dsa_cp_balance_layout_cache", None) is None
