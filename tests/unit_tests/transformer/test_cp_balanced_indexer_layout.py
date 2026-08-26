# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Layout parity between the balanced CP indexer and megatron.core.context_parallel_layout.

The balanced DSA indexer packs each rank's per-sequence zigzag chunks as
``[all head chunks | all tail chunks]`` (the fused indexer kernel allows only one
segment per sequence per packed call). ``context_parallel_layout`` stores the same
zigzag layout per-sequence-interleaved (``[seq0 head, seq0 tail, seq1 head, ...]``).
These tests pin the two implementations to one canonical layout definition.
"""

import os
import subprocess
import sys

import pytest
import torch

# Deliberate coupling to the framework's zigzag ownership definition: these tests exist to
# pin the balanced indexer to it. ``routes`` only exposes the segment builder privately
# today; switch to a public per-rank ownership helper once one is exported again.
from megatron.core.context_parallel_layout.routes import _build_thd_layout_segments
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.experimental_attention_variant.cp_balanced_indexer import (
    _ZZ_PACK_OK,
    _ensure_pack_zigzag_ok,
    _zigzag_plan,
    pack_eligible_for_zigzag,
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
    ("odd-cp3", [0, 1536, 4608], 3, 4608),
]


class _StubGroup:
    # Unique group_name per instance: module-level registries in cp_balanced_indexer
    # (_LAST_PLAN / _ZZ_PACK_OK) key on it, and CPython id() reuse after GC could
    # otherwise leak plan/verdict state between tests.
    _count = 0

    def __init__(self, size, rank):
        self._size, self._rank = size, rank
        _StubGroup._count += 1
        self.group_name = f"stub-group-{_StubGroup._count}"

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
        # Production semantics: max_seqlen_* is the PER-SEQUENCE maximum, not the
        # pack capacity (get_thd_batch_on_this_cp_rank). A fixture that used the
        # capacity here masked a consume-time assert bug for packs of short
        # sequences.
        max_seqlen_q=int((cu[1:] - cu[:-1]).max()),
        max_seqlen_kv=int((cu[1:] - cu[:-1]).max()),
    )


@pytest.mark.parametrize("name,cu_list,cp_size,capacity", _CASES)
def test_prebuild_matches_runtime_plan(name, cu_list, cp_size, capacity):
    """The data-prep prebuild (built on context_parallel_layout primitives) must
    produce exactly the plan the capture-safe runtime fallback would build."""
    l_local = capacity // cp_size
    for r in range(cp_size):
        psp = _packed_params(cu_list, capacity)
        prebuild_balanced_layouts(psp, cp_group=_StubGroup(cp_size, r))
        cu = psp.cu_seqlens_q_padded
        prebuilt = psp._dsa_cp_balance_layout_cache[("zigzag", r)]
        ref = _zigzag_plan(cu, _comp_cu(cu), cp_size, l_local, r, torch.device("cpu"), None)
        assert prebuilt["half"] == ref["half"]
        for key in ("gather_idx", "inv_idx", "pos_head", "pos_tail"):
            assert torch.equal(prebuilt[key], ref[key]), (name, r, key)
        for key in ("head_layout", "tail_layout"):
            for a, b in zip(prebuilt[key], ref[key]):
                assert torch.equal(a.to(torch.int64), b.to(torch.int64)), (name, r, key)


def test_prebuild_single_full_seq_builds_unified_plan():
    """A single pack-spanning sequence is the nseg==1 case of the unified zigzag path:
    the plan + routes are built like any other composition."""
    psp = _packed_params([0, _S], _S)
    prebuild_balanced_layouts(psp, cp_group=_StubGroup(16, 3))
    plan = psp._dsa_cp_balance_layout_cache[("zigzag", 3)]
    assert "disp_send_rows" in plan and "cmb_recv_rows" in plan


def _sim_all_to_all(sends, in_splits_all):
    """CPU emulation of all_to_all_single across N ranks (blocked by destination)."""
    n = len(sends)
    chunks = []
    for r in range(n):
        offs = [0]
        for s in in_splits_all[r]:
            offs.append(offs[-1] + s)
        chunks.append([sends[r][offs[d] : offs[d + 1]] for d in range(n)])
    return [torch.cat([chunks[s][r] for s in range(n)]) for r in range(n)]


@pytest.mark.parametrize("name,cu_list,cp_size,capacity", _CASES)
def test_route_a2a_roundtrip(name, cu_list, cp_size, capacity):
    """Dispatch route must reproduce the [heads|tails] gather exactly, and the combine
    route must return every computed row to its contiguous owner (bit-exact inverse)."""
    l_local = capacity // cp_size
    plans = []
    for r in range(cp_size):
        psp = _packed_params(cu_list, capacity)
        prebuild_balanced_layouts(psp, cp_group=_StubGroup(cp_size, r))
        plans.append(psp._dsa_cp_balance_layout_cache[("zigzag", r)])
    payload = torch.arange(capacity, dtype=torch.int64).unsqueeze(1)
    # dispatch: contiguous -> [heads|tails]
    sends = [
        payload[r * l_local : (r + 1) * l_local].index_select(0, plans[r]["disp_send_rows"])
        for r in range(cp_size)
    ]
    recvs = _sim_all_to_all(sends, [p["disp_in_splits"] for p in plans])
    ordered = []
    for r in range(cp_size):
        o = torch.empty_like(recvs[r])
        o.index_copy_(0, plans[r]["disp_recv_rows"], recvs[r])
        ordered.append(o)
        assert torch.equal(o.squeeze(1), plans[r]["gather_idx"]), (name, r)
    # combine: [heads|tails] -> contiguous owners
    sends2 = [ordered[r].index_select(0, plans[r]["cmb_send_rows"]) for r in range(cp_size)]
    recvs2 = _sim_all_to_all(sends2, [p["disp_out_splits"] for p in plans])
    for r in range(cp_size):
        out = torch.empty_like(recvs2[r])
        out.index_copy_(0, plans[r]["cmb_recv_rows"], recvs2[r])
        mine = torch.arange(r * l_local, (r + 1) * l_local, dtype=torch.int64)
        assert torch.equal(out.squeeze(1), mine), (name, r)


def test_prebuild_records_pack_verdicts_for_routing():
    """Prebuild records the per-pack verdict (microbatch cache + module registry):
    conforming packs are eligible; a pack that violates per-sequence 2N divisibility
    records False so the forward routes that microbatch to the contiguous reference
    path."""
    group = _StubGroup(4, 0)
    g = getattr(group, "group_name", None) or id(group)  # mirrors _group_key

    aligned = _packed_params([0, 1024, 4096], 4096)
    prebuild_balanced_layouts(aligned, cp_group=group)
    assert _ZZ_PACK_OK[(g, 1024)] is True
    assert aligned._dsa_cp_balance_layout_cache["zz_pack_ok"] == (1024, True)
    assert pack_eligible_for_zigzag(aligned, None, group, 4, 1024)
    _ensure_pack_zigzag_ok(None, group, 4, 1024, aligned._dsa_cp_balance_layout_cache)

    # Sequence lengths 500/508 are not divisible by 2 * cp_size = 8: verdict False,
    # no plan, and the router sends the microbatch to the reference path.
    unaligned = _packed_params([0, 500, 1008], 1008)
    prebuild_balanced_layouts(unaligned, cp_group=group)
    assert unaligned._dsa_cp_balance_layout_cache["zz_pack_ok"] == (252, False)
    assert ("zigzag", 0) not in unaligned._dsa_cp_balance_layout_cache
    assert not pack_eligible_for_zigzag(unaligned, None, group, 4, 252)
    with pytest.raises(ValueError, match="not zigzag-representable"):
        _ensure_pack_zigzag_ok(None, group, 4, 252, unaligned._dsa_cp_balance_layout_cache)
    # The aligned capacity's verdict is untouched.
    assert _ZZ_PACK_OK[(g, 1024)] is True

    prebuild_balanced_layouts(aligned, cp_group=group)
    assert _ZZ_PACK_OK[(g, 1024)] is True


def test_eager_prebuild_switches_paths_at_the_same_capacity():
    """Eligibility is a property of the current pack, not its row capacity."""
    group = _StubGroup(4, 0)
    gkey = group.group_name

    eligible_a = _packed_params([0, 1024, 4096], 4096)
    prebuild_balanced_layouts(eligible_a, cp_group=group, pad_alignment=8)
    assert eligible_a._dsa_cp_balance_layout_cache["zz_pack_ok"] == (1024, True)

    ineligible = _packed_params([0, 1001, 4096], 4096)
    prebuild_balanced_layouts(ineligible, cp_group=group, pad_alignment=8)
    assert ineligible._dsa_cp_balance_layout_cache["zz_pack_ok"] == (1024, False)
    assert ("zigzag", 0) not in ineligible._dsa_cp_balance_layout_cache
    assert _ZZ_PACK_OK[(gkey, 1024)] is False

    eligible_b = _packed_params([0, 2048, 4096], 4096)
    prebuild_balanced_layouts(eligible_b, cp_group=group, pad_alignment=8)
    assert eligible_b._dsa_cp_balance_layout_cache["zz_pack_ok"] == (1024, True)
    assert ("zigzag", 0) in eligible_b._dsa_cp_balance_layout_cache
    assert _ZZ_PACK_OK[(gkey, 1024)] is True


def test_eager_prebuild_accepts_capacity_changes():
    """Two eligible eager packs may build different per-rank capacities."""
    group = _StubGroup(4, 0)
    short = _packed_params([0, 1024, 4096], 4096)
    long = _packed_params([0, 2048, 6144], 6144)

    prebuild_balanced_layouts(short, cp_group=group, pad_alignment=8)
    prebuild_balanced_layouts(long, cp_group=group, pad_alignment=8)

    assert short._dsa_cp_balance_layout_cache["zz_pack_ok"] == (1024, True)
    assert long._dsa_cp_balance_layout_cache["zz_pack_ok"] == (1536, True)
    assert short._dsa_cp_balance_layout_cache[("zigzag", 0)]["half"] == 512
    assert long._dsa_cp_balance_layout_cache[("zigzag", 0)]["half"] == 768


def test_eager_probe_routes_unprebuilt_packs():
    """A frontend that never prebuilds (legacy varlen ``get_batch``) can hand the
    router a raw, non-2N-aligned pack: the eager probe records a False verdict (the
    microbatch routes to the reference path), while the internal consistency check
    still raises if a caller bypasses that routing."""
    group = _StubGroup(4, 0)
    cache = {}

    # 500 % (2 * 4) != 0 -> not zigzag-representable. l_local = 252 (raw pack).
    cu = torch.tensor([0, 500, 1008], dtype=torch.int32)
    psp = _packed_params([0, 500, 1008], 1008)
    psp._dsa_cp_balance_layout_cache = cache
    assert not pack_eligible_for_zigzag(psp, cu, group, 4, 252)
    assert cache["zz_pack_ok"] == (252, False)
    with pytest.raises(ValueError, match="not zigzag-representable"):
        _ensure_pack_zigzag_ok(cu, group, 4, 252, cache)

    # Aligned pack with a capacity-padding tail (both 2N-aligned) -> zigzag OK.
    cu2 = torch.tensor([0, 1024, 3072], dtype=torch.int32)
    cache2 = {}
    _ensure_pack_zigzag_ok(cu2, group, 4, 1024, cache2)
    assert cache2["zz_pack_ok"] == (1024, True)

    # A cached verdict probed at a different capacity does not apply: re-probe.
    stale = {"zz_pack_ok": (128, True)}
    _ensure_pack_zigzag_ok(cu2, group, 4, 1024, stale)
    assert stale["zz_pack_ok"] == (1024, True)

    # With no cu to probe and no recorded verdict, the check cannot verify: raise.
    with pytest.raises(RuntimeError, match="cannot verify pack eligibility"):
        _ensure_pack_zigzag_ok(None, _StubGroup(4, 1), 4, 64, None)


def test_zigzag_plan_rejects_unaligned_pack_eagerly():
    """The capture-safe runtime builder must refuse packs it cannot represent
    instead of emitting out-of-range gather indices."""
    # 500 % (2 * 4) != 0: the ragged chunk enumeration cannot represent this pack.
    cu = torch.tensor([0, 500, 1008], dtype=torch.int32)
    with pytest.raises(ValueError, match="divisible by 2 \\* cp_size"):
        _zigzag_plan(cu, _comp_cu(cu), 4, 252, 0, torch.device("cpu"), None)


def test_zigzag_plan_clamps_pseudo_sequence_positions():
    """Rows in the capacity-padding pseudo-sequence [cu[-1], S) carry RoPE
    position 0, matching the reference clamp (their top-k is discarded)."""
    cp_size, capacity = 4, 4096
    cu = torch.tensor([0, 2048], dtype=torch.int32)  # pseudo-seq = [2048, 4096)
    l_local = capacity // cp_size
    for r in range(cp_size):
        plan = _zigzag_plan(cu, _comp_cu(cu), cp_size, l_local, r, torch.device("cpu"), None)
        g = plan["gather_idx"]
        pos = torch.cat((plan["pos_head"], plan["pos_tail"]))
        pseudo = g >= 2048
        assert bool(pseudo.any()), "every rank owns pseudo-sequence chunks"
        assert bool((pos[pseudo] == 0).all())
        assert bool((pos[~pseudo] < 2048).all())


def test_prebuild_refresh_preserves_old_plan_objects():
    """Changing packs must not mutate a previously published plan in place
    (recompute of an earlier microbatch may still read it): the module slot is
    replaced with a fresh object and the old one keeps its values. In eager this
    is silent (normal varlen operation); once a CUDA graph captured a plan, a
    composition change must raise instead."""
    group = _StubGroup(4, 0)
    pack_a = _packed_params([0, 1024, 4096], 4096)
    prebuild_balanced_layouts(pack_a, cp_group=group)
    plan_a = pack_a._dsa_cp_balance_layout_cache[("zigzag", 0)]
    gather_a = plan_a["gather_idx"].clone()

    pack_b = _packed_params([0, 2048, 4096], 4096)
    prebuild_balanced_layouts(pack_b, cp_group=group)
    plan_b = pack_b._dsa_cp_balance_layout_cache[("zigzag", 0)]
    assert plan_b is not plan_a
    assert torch.equal(plan_a["gather_idx"], gather_a), "old plan mutated in place"
    assert not torch.equal(plan_b["gather_idx"], gather_a)

    # Idempotent path: same pack again reuses the slot object (pointer-stable).
    pack_b2 = _packed_params([0, 2048, 4096], 4096)
    prebuild_balanced_layouts(pack_b2, cp_group=group)
    assert pack_b2._dsa_cp_balance_layout_cache[("zigzag", 0)] is plan_b

    # Under CUDA graphs the composition is contract-static: with graphs enabled a
    # change raises at data-prep time, before any early return.
    pack_c = _packed_params([0, 1024, 2048, 4096], 4096)
    with pytest.raises(RuntimeError, match="static pack composition"):
        prebuild_balanced_layouts(pack_c, cp_group=group, graphs_enabled=True)
    # Without graphs, changing packs is ordinary varlen operation: silent.
    prebuild_balanced_layouts(pack_c, cp_group=group)
    assert pack_c._dsa_cp_balance_layout_cache[("zigzag", 0)] is not plan_b


@pytest.mark.parametrize("name,cu_list,cp_size,capacity", _CASES)
def test_kv_bounds_cover_every_visible_key(name, cu_list, cp_size, capacity):
    """The tight per-call compressed-K bounds must cover, for every owned row, the
    full causally visible compressed prefix (a clipped visible key would silently
    change the top-k). Ground truth is recomputed from first principles."""
    ratio = 4
    for r in range(cp_size):
        psp = _packed_params(cu_list, capacity)
        prebuild_balanced_layouts(psp, cp_group=_StubGroup(cp_size, r))
        plan = psp._dsa_cp_balance_layout_cache[("zigzag", r)]
        cu = torch.tensor(plan["_cu_list"], dtype=torch.long)
        comp = _comp_cu(cu.to(torch.int32)).to(torch.long)
        gather = plan["gather_idx"]
        half = plan["half"]
        pos = torch.cat((plan["pos_head"], plan["pos_tail"]))
        seq_of = torch.bucketize(gather, cu[1:], right=True).clamp_max(cu.numel() - 2)
        real = gather < cu[-1]
        comp_len = comp[1:] - comp[:-1]
        visible = torch.minimum((pos + 1) // ratio, comp_len[seq_of])
        need_end = comp[seq_of] + visible  # absolute compressed-row end per row
        for which, mkv_key, kend_key, sl in (
            ("head", "mkv_head", "k_end_head", slice(0, half)),
            ("tail", "mkv_tail", "k_end_tail", slice(half, None)),
        ):
            v = visible[sl][real[sl]]
            e = need_end[sl][real[sl]]
            if v.numel() == 0:
                continue
            assert int(v.max()) <= plan[mkv_key], (name, r, which, int(v.max()), plan[mkv_key])
            assert int(e.max()) <= plan[kend_key], (name, r, which, int(e.max()), plan[kend_key])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="indexer scoring runs on device")
def test_chunk_scoring_matches_full_call():
    """The mathematical foundation of the balanced path: a query's top-k depends only
    on its own position and K, so scoring a chunk of rows at its global offset selects
    the same keys as the same rows of one full-sequence call. Asserted exactly on the
    unfused reference path, which pins the shared layout/causal-offset math.

    The fused kernel is deliberately NOT asserted here: its score numerics differ from
    the reference (fp32 dense score buffer vs the reference matmul), and on random
    bf16 inputs the top-k boundary gaps are so tight that most rows flip an index —
    measured on GB200: fused-vs-unfused disagrees on ~87% of rows for the SAME full
    call, and the disagreement rate of a chunk call matches the full call's row for
    row (512/512 vs 512/512, 764 vs 765, 1024 vs 1024 across three offsets), i.e. the
    fused deviation is call-shape-independent and chunk==full holds up to that ambient
    kernel noise (production-data divergence is ~1 row / 4096)."""
    # Unfused only: exact. The fused kernel's tie resolution is call-shape
    # dependent (measured ~87% row flips on random bf16 data, rate identical
    # between chunk and full calls); its layout semantics are pinned separately
    # by test_fused_multi_offset_packed_layout on tie-free data.
    use_fused = False
    from megatron.core.transformer.experimental_attention_variant.csa_utils import cp_utils as _cu

    torch.manual_seed(1234)
    dev = torch.device("cuda")
    T, heads, dim, ratio, topk = 4096, 4, 64, 4, 64
    cu = torch.tensor([0, 1536, T], dtype=torch.int32, device=dev)
    cu_comp = _comp_cu(cu.cpu()).to(dev)
    comp = int(cu_comp[-1])
    q = torch.randn(T, heads, dim, dtype=torch.bfloat16, device=dev)
    # bf16 weights: the fused kernel asserts bf16 inputs (production weights are the
    # bf16 linear_weights_proj output).
    w = (torch.rand(T, heads, dtype=torch.float32, device=dev) + 0.5).to(torch.bfloat16)
    k = torch.randn(comp, dim, dtype=torch.bfloat16, device=dev)
    scale = dim**-0.5

    def _call(rows_q, rows_w, gs, mq):
        tk, _ = _cu.compute_cp_indexer_topk(
            rows_q,
            rows_w,
            k,
            cu,
            cu_comp,
            gs,
            ratio,
            topk,
            scale,
            max_seqlen_q=mq,
            use_fused=use_fused,
        )
        return tk

    full = _call(q, w, 0, T)
    for gs, sz in ((1024, 512), (1536, 1024), (3072, 1024)):
        chunk = _call(q[gs : gs + sz], w[gs : gs + sz], gs, T)
        a, _ = torch.sort(full[gs : gs + sz], dim=-1)
        b, _ = torch.sort(chunk, dim=-1)
        assert torch.equal(a, b), (use_fused, gs, sz)


def test_prebuild_capacity_probe_for_raw_cu():
    """Middle-PP-stage PackedSeqParams carry the raw (unpadded) cu while the hidden
    states are padded to capacity: with capacity=..., prebuild probes and builds at
    the PHYSICAL pack size (capacity tail merged into the last sequence), so its
    plans and verdicts match the forward's l_local instead of being rejected by the
    capacity guards (which would silently degrade middle stages to folding)."""
    raw = torch.tensor([0, 1024, 3000], dtype=torch.int32)
    psp = PackedSeqParams(
        qkv_format="thd", cu_seqlens_q=raw, cu_seqlens_kv=raw, max_seqlen_q=2048, max_seqlen_kv=2048
    )
    group = _StubGroup(4, 0)
    prebuild_balanced_layouts(psp, cp_group=group, capacity=4096)
    l_local = 4096 // 4
    plan = psp._dsa_cp_balance_layout_cache[("zigzag", 0)]
    assert plan["half"] * 2 == l_local
    assert psp._dsa_cp_balance_layout_cache["zz_pack_ok"] == (l_local, True)
    # [raw_total, capacity) is the zero-K pseudo-sequence: the compressed-K
    # geometry must match the K buffer the forward builds from the RAW cu
    # (1024//4 + 1976//4 = 750 physical rows) — merging the tail into the last
    # sequence would inflate k_end past it and trip the consume-side guard.
    raw_comp_total = 1024 // 4 + (3000 - 1024) // 4
    assert plan["k_end_head"] <= raw_comp_total
    assert plan["k_end_tail"] <= raw_comp_total
    # Pseudo-sequence rows carry RoPE position 0 (their top-k is discarded).
    pos = torch.cat((plan["pos_head"], plan["pos_tail"]))
    assert bool((pos[plan["gather_idx"] >= 3000] == 0).all())

    # Without the capacity hint prebuild probes at the RAW length (l_local 750):
    # its artifacts are internally valid but capacity-tagged at 750, so a forward
    # running on capacity-padded hidden states (l_local 1024) rejects them via the
    # capacity guards and re-probes — the capacity hint is what makes them match.
    psp2 = PackedSeqParams(
        qkv_format="thd", cu_seqlens_q=raw, cu_seqlens_kv=raw, max_seqlen_q=2048, max_seqlen_kv=2048
    )
    prebuild_balanced_layouts(psp2, cp_group=_StubGroup(4, 0))
    assert psp2._dsa_cp_balance_layout_cache["zz_pack_ok"][0] == 3000 // 4


def test_balanced_compute_smoke_runs_production_path(monkeypatch):
    """Invoke the PRODUCTION balanced_compute_cp_indexer_topk (unfused, CPU): the
    cp_size <= 1 exit still runs the real projection -> _no_fp8_ctx -> RoPE ->
    reference-delegation pipeline, so module-level integration failures (e.g. a
    missing global read inside _no_fp8_ctx) surface here rather than only on
    GPU. Verified against the reference call on the same inputs."""
    from megatron.core.transformer.experimental_attention_variant import cp_balanced_indexer as M
    from megatron.core.transformer.experimental_attention_variant.csa_utils import cp_utils

    torch.manual_seed(3)
    rows, q_lora, heads, dim, pos_dim, ratio, topk = 256, 16, 4, 8, 4, 4, 8
    dev = torch.device("cpu")

    class _Rotary:
        def __init__(self):
            self._rpe = torch.randn(rows, 1, 1, pos_dim, dtype=torch.float32)

        def __call__(self, length, packed_seq=True):
            return self._rpe

    class _Indexer:
        def __init__(self):
            self._w = torch.randint(-2, 3, (heads * dim, q_lora)).to(torch.bfloat16)
            self.rotary_pos_emb = _Rotary()
            self.index_n_heads, self.index_head_dim = heads, dim
            self.qk_pos_emb_head_dim = pos_dim

        def linear_wq_b(self, x):
            return x.reshape(-1, q_lora) @ self._w.t(), None

    class _Cfg:
        apply_rope_fusion = False
        cuda_graph_impl = "none"
        rotary_interleaved = False

    # fast_hadamard_transform is GPU-only; the smoke targets the integration
    # wiring (projection/_no_fp8_ctx/RoPE/delegation), so make rotation identity
    # on BOTH the production call and the reference reproduction below.
    monkeypatch.setattr(M, "rotate_activation", lambda x: x)
    idx = _Indexer()
    qr = torch.randint(-2, 3, (rows, 1, q_lora)).to(torch.bfloat16)
    w = (torch.rand(rows, heads) + 0.5).to(torch.bfloat16)
    cu = torch.tensor([0, rows], dtype=torch.int32)
    cu_comp = torch.tensor([0, rows // ratio], dtype=torch.int32)
    k = torch.randn(rows // ratio, dim).to(torch.bfloat16)

    tk, layout = M.balanced_compute_cp_indexer_topk(
        qr,
        w,
        idx,
        k,
        cu,
        cu_comp,
        _Cfg(),
        _StubGroup(1, 0),
        1,
        rows,
        0,
        ratio,
        topk,
        dim**-0.5,
        rows,
        use_fused=False,
    )
    assert tk is not None and tk.shape == (rows, topk)
    # The cp1 exit is the reference call over own rows: reproduce it directly.
    from megatron.core.transformer.experimental_attention_variant.csa_utils.cp_utils import (
        apply_thd_cp_local_rope_unfused,
    )

    q_ref, _ = idx.linear_wq_b(qr)
    q_ref = q_ref.reshape(rows, heads, dim)
    q_ref = apply_thd_cp_local_rope_unfused(
        q_ref, idx.rotary_pos_emb(rows), dim - pos_dim, pos_dim, cu, 0, _Cfg()
    )
    ref, _ = cp_utils.compute_cp_indexer_topk(
        q_ref, w, k, cu, cu_comp, 0, ratio, topk, dim**-0.5, max_seqlen_q=rows, use_fused=False
    )
    fs, _ = torch.sort(tk, dim=-1)
    rs, _ = torch.sort(ref, dim=-1)
    assert torch.equal(fs, rs)


def test_prebuild_routes_ineligible_above_limit_pack():
    """An INELIGIBLE pack above the balanced row limit must route to the
    reference path (verdict False), not raise: it never issues the two synthetic
    half-row calls, so the balanced-only limit does not apply to it."""
    from megatron.core.transformer.experimental_attention_variant.csa_utils import cp_utils

    group = _StubGroup(4, 0)
    cap = 4 * (cp_utils.FUSED_INDEXER_MAX_SAFE_ROWS + 2048) * 2  # l_local > 2 * limit
    # First sequence length is odd -> not divisible by 2 * cp_size = 8.
    psp = _packed_params([0, 100001, cap], cap)
    prebuild_balanced_layouts(psp, cp_group=group, pad_alignment="max")
    l_local = cap // 4
    assert psp._dsa_cp_balance_layout_cache["zz_pack_ok"] == (l_local, False)
    assert ("zigzag", 0) not in psp._dsa_cp_balance_layout_cache


def test_prebuild_rejects_capacity_above_fused_row_limit():
    """The balanced path issues two fused calls of l_local // 2 rows; prebuild
    fails at data-prep time when that exceeds FUSED_INDEXER_MAX_SAFE_ROWS (the
    verified fused-kernel defect boundary, see the WORKSPACE NOTE below)."""
    group = _StubGroup(4, 0)
    cap = 4 * 4 * 32768  # l_local = 131072 -> half = 65536 > 32768
    psp = _packed_params([0, cap], cap)
    with pytest.raises(RuntimeError, match="verified-safe limit"):
        prebuild_balanced_layouts(psp, cp_group=group, pad_alignment=8)
    # At the boundary (half == 32768) the plan builds normally.
    cap = 4 * 2 * 32768  # l_local = 65536 -> half = 32768
    psp = _packed_params([0, cap], cap)
    prebuild_balanced_layouts(psp, cp_group=group, pad_alignment=8)
    assert ("zigzag", 0) in psp._dsa_cp_balance_layout_cache


def test_row_limit_guard_in_compute_cp_indexer_topk():
    """Above FUSED_INDEXER_MAX_SAFE_ROWS the policy splits: a synthetic layout
    (balanced path) fails closed, while zero-work exits still run first so no-op
    calls never trip the guard. Legacy above-limit calls proceed fused with a
    once-per-process correctness warning (exercised by the GPU suite), so they
    are not run here on CPU."""
    from megatron.core.transformer.experimental_attention_variant.csa_utils import cp_utils

    rows = cp_utils.FUSED_INDEXER_MAX_SAFE_ROWS + 4
    heads, dim, ratio = 4, 8, 4
    q = torch.zeros((rows, heads, dim), dtype=torch.bfloat16)
    w = torch.ones((rows, heads), dtype=torch.bfloat16)
    k = torch.zeros((rows // ratio, dim), dtype=torch.bfloat16)
    cu = torch.tensor([0, rows], dtype=torch.int32)
    cc = torch.tensor([0, rows // ratio], dtype=torch.int32)
    fake_layout = (cu, cc, torch.tensor([0, 0], dtype=torch.int32))
    with pytest.raises(RuntimeError, match="verified-safe limit"):
        cp_utils.compute_cp_indexer_topk(
            q,
            w,
            k,
            cu,
            cc,
            0,
            ratio,
            8,
            1.0,
            max_seqlen_q=rows,
            use_fused=True,
            prebuilt_layout=fake_layout,
            synthetic_layout=True,
        )
    # Zero-work exits run before the guard: a no-op call never raises even with
    # the synthetic flag set (topk_width == 0 -> (None, None)).
    tk, layout = cp_utils.compute_cp_indexer_topk(
        q,
        w,
        k,
        cu,
        cc,
        0,
        ratio,
        0,
        1.0,
        max_seqlen_q=rows,
        use_fused=True,
        prebuilt_layout=fake_layout,
        synthetic_layout=True,
    )
    assert tk is None and layout is None


# WORKSPACE NOTE — the fused kernel package's cross-call defect, fully
# characterized on GB200 / cudnn-frontend 1.26.0 (three controlled matrices,
# fresh process per scenario, fused output vs an in-process unfused torch
# reference on tie-free signature data):
#   * A fused call with total_q > 32768 rows is correct ONLY as the process's
#     first fused call. After any fused call — of any shape, including a
#     bit-identical one — every output row >= 32768 is silently wrong
#     (deterministic; first bad row exactly 32768).
#   * Calls with total_q <= 32768 are immune to process history entirely:
#     row-count transitions, sequence count, max_seqlen_kv, causal offsets and
#     the predecessor's shape were each varied and none is causal.
#   * Discarded warmup/priming calls cannot help (a same-shape priming call is
#     itself a predecessor — falsified in CI run 32712450771).
# Production consequence: the balanced synthetic-layout calls fail closed above
# the limit (prebuild bounds l_local at 2 * 32768), balanced-internal
# ordinary-layout calls take the unfused path, and pre-existing callers keep
# their behavior with a once-per-process correctness warning (no rejection, no
# silent rerouting). In shared-process CI lanes, other
# suites' tiny fused calls run first (they don't observe the defect themselves:
# they compare fused output against equally-degenerate references at sub-32
# head counts, where the kernel silently returns all-zero scores), so the
# 65536-row multi-offset test below would be corrupted in-process:
# ``test_fused_kernel_suite_isolated`` re-runs the five fused tests in a fresh
# subprocess (clean kernel-package state), where the in-file ordering — the
# multi-offset test's 65536-row call first — is sufficient (pinned green
# standalone on GB200). In-process those five tests skip unless
# MCORE_DSA_FUSED_CHILD=1. To be raised with the kernel owners together with
# the sub-32-head silent-zero mode; do not add a version exemption until a
# fixed kernel package is demonstrated against the reproducer.
_FUSED_ISOLATED_TESTS = (
    "test_fused_multi_offset_packed_layout",
    "test_fused_tight_width_smoke",
    "test_fused_sliced_k_prefix_view_smoke",
    "test_fused_tight_width_ceiling_smoke",
    "test_fused_reduced_mq_matches_full_mq",
)
_IN_FUSED_CHILD = os.environ.get("MCORE_DSA_FUSED_CHILD") == "1"

_fused_kernel_test = pytest.mark.skipif(
    not torch.cuda.is_available() or not _IN_FUSED_CHILD,
    reason="fused indexer kernel required; runs inside the isolated subprocess "
    "(cross-call kernel-package state, see WORKSPACE NOTE)",
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused indexer kernel required")
def test_fused_kernel_suite_isolated():
    """Run the five fused-kernel tests in a fresh process (see WORKSPACE NOTE)."""
    env = dict(os.environ)
    env["MCORE_DSA_FUSED_CHILD"] = "1"
    # The child is a plain single-process pytest: drop the launcher's
    # distributed environment so nothing tries to re-join the parent's job,
    # and pin the child to this rank's GPU.
    for k in list(env):
        if k in (
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "LOCAL_WORLD_SIZE",
            "GROUP_RANK",
            "MASTER_ADDR",
            "MASTER_PORT",
        ) or k.startswith("TORCHELASTIC"):
            env.pop(k)
    visible = env.get("CUDA_VISIBLE_DEVICES")
    if visible:
        env["CUDA_VISIBLE_DEVICES"] = visible.split(",")[torch.cuda.current_device()]
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(torch.cuda.current_device())
    node_ids = [f"{__file__}::{name}" for name in _FUSED_ISOLATED_TESTS]
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *node_ids],
        env=env,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    assert proc.returncode == 0, (
        f"isolated fused-kernel suite failed (rc={proc.returncode}):\n"
        f"{proc.stdout[-4000:]}\n{proc.stderr[-2000:]}"
    )


@_fused_kernel_test
def test_fused_multi_offset_packed_layout():
    """The production zigzag path feeds the fused kernel synthetic packed layouts in
    which EVERY real segment carries a non-zero q_causal_offset (r*c_i per sequence)
    plus per-sequence K ranges and a K-prefix view — an input regime the reference
    layout builder can never produce (it emits at most one non-zero offset). Verify
    on the REAL kernel, with tie-free signature data, that a plan's head/tail packed
    calls select exactly the same keys as one full fused reference call."""
    from megatron.core.transformer.experimental_attention_variant.csa_utils import cp_utils as _cu

    torch.manual_seed(29)
    dev = torch.device("cuda")
    cp_size, cap = 4, 65536
    cu_list = [0, 40960, 65536]  # both sequences 2N-divisible; large enough that
    # the head chunks get TIGHT quantized widths (mkv < gkv) and K-PREFIX slices
    # (k_end < comp) — the exact production regime of the K-bound optimization.
    heads, dim, ratio, topk = 64, 128, 4, 64  # PRODUCTION head shape (see _signature_qkw)
    l_local, half = cap // cp_size, cap // cp_size // 2
    cu = torch.tensor(cu_list, dtype=torch.int32, device=dev)
    cu_comp = _comp_cu(cu.cpu()).to(dev)
    comp = int(cu_comp[-1])
    q, w, k, _sig_perm = _signature_qkw(cap, comp, heads, dim, dev)
    scale = dim**-0.5

    tk_ref, _ = _cu.compute_cp_indexer_topk(
        q, w, k, cu, cu_comp, 0, ratio, topk, scale, max_seqlen_q=cap, use_fused=True
    )

    Z = torch.empty((cap, topk), dtype=tk_ref.dtype, device=dev)
    plans = []
    for r in range(cp_size):
        psp = _packed_params(cu_list, cap)
        psp.cu_seqlens_q = psp.cu_seqlens_q.to(dev)
        psp.cu_seqlens_q_padded = psp.cu_seqlens_q_padded.to(dev)
        prebuild_balanced_layouts(psp, cp_group=_StubGroup(cp_size, r))
        plans.append(psp._dsa_cp_balance_layout_cache[("zigzag", r)])
    gkv = max(1, cap // ratio)
    comp_total = int(cu_comp[-1])
    # Guard against this test silently degenerating to full-width calls again:
    # at least one packed call must exercise a tight width AND a K-prefix slice.
    assert any(
        p[m] < gkv for p in plans for m in ("mkv_head", "mkv_tail")
    ), "no tight width engaged — enlarge the case"
    assert any(
        p[e] < comp_total for p in plans for e in ("k_end_head", "k_end_tail")
    ), "no K-prefix slice engaged — enlarge the case"
    mq = max(1, min(cap, half))
    for r in range(cp_size):
        plan = plans[r]
        rows_q = q.index_select(0, plan["gather_idx"])
        rows_w = w.index_select(0, plan["gather_idx"])
        parts = []
        for sl, lay, mkv_k, kend_k in (
            (slice(0, half), "head_layout", "mkv_head", "k_end_head"),
            (slice(half, None), "tail_layout", "mkv_tail", "k_end_tail"),
        ):
            k_end = plan[kend_k]
            k_pass = k[:k_end] if k_end < comp else k
            tk, _ = _cu.compute_cp_indexer_topk(
                rows_q[sl],
                rows_w[sl],
                k_pass,
                cu,
                cu_comp,
                0,
                ratio,
                topk,
                scale,
                max_seqlen_q=mq,
                use_fused=True,
                max_seqlen_kv=plan[mkv_k],
                prebuilt_layout=plan[lay],
            )
            parts.append(tk)
        Z[r * l_local : (r + 1) * l_local] = torch.cat(parts)
    torch.cuda.synchronize()
    for r in range(cp_size):
        mine = Z.index_select(0, plans[r]["inv_idx"])
        ref = tk_ref[r * l_local : (r + 1) * l_local]
        a, _ = torch.sort(mine, dim=-1)
        b, _ = torch.sort(ref, dim=-1)
        assert torch.equal(a, b), (r, int((a != b).any(dim=-1).sum()))
    # Anti-degeneration anchor (see _signature_qkw): pin a few rows against the
    # computable ground truth so a kernel mode that zeroes all scores cannot make
    # both sides agree on garbage.
    comp_l = [int(v) for v in cu_comp.tolist()]
    for g_row in (cu_list[1] - 1, cu_list[1] + 5000, cap - 1):
        seq = max(i for i in range(len(cu_list)) if cu_list[i] <= g_row)
        pos = g_row - cu_list[seq]
        seg_len = comp_l[seq + 1] - comp_l[seq]
        vis = min((pos + 1) // ratio, seg_len)
        if vis < topk:
            continue
        window = _sig_perm[comp_l[seq] : comp_l[seq] + vis].float()
        gt, _ = torch.sort(torch.topk(window, topk).indices.to(torch.int32))
        got, _ = torch.sort(tk_ref[g_row])
        assert torch.equal(got, gt), (g_row, got[:8].tolist(), gt[:8].tolist())


@_fused_kernel_test
def test_fused_tight_width_smoke():
    """Exercise the empirical tight-width kernel contract (_KV_TIGHT_WIDTH_CEILING):
    a fused call whose score width (16384) is narrower than the declared per-sequence
    compressed KV length must complete without an illegal memory access and return
    causally valid indices. Guards the contract the K-bound optimization rests on."""
    from megatron.core.transformer.experimental_attention_variant.csa_utils import cp_utils as _cu

    torch.manual_seed(7)
    dev = torch.device("cuda")
    T, heads, dim, ratio, topk = 131072, 64, 128, 4, 64  # production head shape
    gs, sz = 63488, 1024
    cu = torch.tensor([0, T], dtype=torch.int32, device=dev)
    cu_comp = _comp_cu(cu.cpu()).to(dev)
    q = torch.randn(sz, heads, dim, dtype=torch.bfloat16, device=dev)
    w = (torch.rand(sz, heads, dtype=torch.float32, device=dev) + 0.5).to(torch.bfloat16)
    k = torch.randn(int(cu_comp[-1]), dim, dtype=torch.bfloat16, device=dev)
    tk, _ = _cu.compute_cp_indexer_topk(
        q,
        w,
        k,
        cu,
        cu_comp,
        gs,
        ratio,
        topk,
        dim**-0.5,
        max_seqlen_q=T,
        use_fused=True,
        max_seqlen_kv=16384,
    )
    torch.cuda.synchronize()
    assert tk.shape == (sz, topk)
    # Causal bound: a row at global position gs+i sees at most (gs+i+1)//ratio keys.
    assert int(tk.max()) < (gs + sz) // ratio
    assert int(tk.min()) >= 0


@_fused_kernel_test
def test_fused_sliced_k_prefix_view_smoke():
    """The zigzag consumer feeds the fused kernel a PREFIX VIEW of the gathered K
    (k_seq_major[:k_end]) together with a layout whose declared per-sequence K
    ranges may extend past the slice (mkv is a capacity, k_end the causal need).
    Reads past the slice must land on valid full-buffer memory — exercise that
    combination end to end and check the indices stay causally bounded."""
    from megatron.core.transformer.experimental_attention_variant.csa_utils import cp_utils as _cu

    torch.manual_seed(11)
    dev = torch.device("cuda")
    T, heads, dim, ratio, topk = 65536, 64, 128, 4, 64  # production head shape
    gs, sz = 16384, 1024  # rows see at most (gs+sz)//ratio = 4352 keys
    cu = torch.tensor([0, T], dtype=torch.int32, device=dev)
    cu_comp = _comp_cu(cu.cpu()).to(dev)
    k_full = torch.randn(int(cu_comp[-1]), dim, dtype=torch.bfloat16, device=dev)
    k_end = 8192
    k_slice = k_full[:k_end]  # prefix view, NOT contiguous-ized
    assert k_slice.data_ptr() == k_full.data_ptr()
    q = torch.randn(sz, heads, dim, dtype=torch.bfloat16, device=dev)
    w = (torch.rand(sz, heads, dtype=torch.float32, device=dev) + 0.5).to(torch.bfloat16)
    tk, _ = _cu.compute_cp_indexer_topk(
        q,
        w,
        k_slice,
        cu,
        cu_comp,
        gs,
        ratio,
        topk,
        dim**-0.5,
        max_seqlen_q=T,
        use_fused=True,
        max_seqlen_kv=8192,
    )
    torch.cuda.synchronize()
    assert tk.shape == (sz, topk)
    assert int(tk.max()) < (gs + sz) // ratio
    assert int(tk.min()) >= 0


@_fused_kernel_test
def test_fused_tight_width_ceiling_smoke():
    """Boundary case of the empirical kernel contract: a tight score width EXACTLY at
    _KV_TIGHT_WIDTH_CEILING (65536), narrower than the declared per-sequence compressed
    KV length, must complete without an illegal memory access. Guards the exact edge
    the K-bound optimization is allowed to reach."""
    from megatron.core.transformer.experimental_attention_variant.csa_utils import cp_utils as _cu

    torch.manual_seed(13)
    dev = torch.device("cuda")
    T, heads, dim, ratio, topk = 524288, 64, 128, 4, 64  # production head shape
    gs, sz = 261120, 1024  # (gs + sz) // ratio == 65536 == the ceiling
    cu = torch.tensor([0, T], dtype=torch.int32, device=dev)
    cu_comp = _comp_cu(cu.cpu()).to(dev)
    assert int(cu_comp[-1]) > 65536  # declared per-seq KV length exceeds the width
    k = torch.randn(int(cu_comp[-1]), dim, dtype=torch.bfloat16, device=dev)
    q = torch.randn(sz, heads, dim, dtype=torch.bfloat16, device=dev)
    w = (torch.rand(sz, heads, dtype=torch.float32, device=dev) + 0.5).to(torch.bfloat16)
    tk, _ = _cu.compute_cp_indexer_topk(
        q,
        w,
        k,
        cu,
        cu_comp,
        gs,
        ratio,
        topk,
        dim**-0.5,
        max_seqlen_q=T,
        use_fused=True,
        max_seqlen_kv=65536,
    )
    torch.cuda.synchronize()
    assert tk.shape == (sz, topk)
    assert int(tk.max()) < (gs + sz) // ratio
    assert int(tk.min()) >= 0


def test_zigzag_plan_rejects_wrong_capacity_cache():
    """_zigzag_plan must rebuild when the per-batch cache holds a plan probed at a
    different capacity (e.g. prebuild saw a raw cu ending short of the physical
    pack); resurrecting it would feed wrong-sized routes to the consumer."""
    cu = torch.tensor([0, 2048, 4096], dtype=torch.int32)
    l_local = 4096 // 4
    stale = {"half": 128}  # built for l_local = 256, not 1024
    cache = {("zigzag", 0): stale}
    plan = _zigzag_plan(cu, _comp_cu(cu), 4, l_local, 0, torch.device("cpu"), cache)
    assert plan is not stale
    assert plan["half"] * 2 == l_local
    # The freshly built plan replaces the stale entry in the cache.
    assert cache[("zigzag", 0)] is plan


def test_prebuild_uses_actual_pack_instead_of_alignment_for_eligibility():
    """Alignment is a capacity-rounding hint, not a zigzag contract.

    Every supported alignment spelling builds an eligible actual pack, while an
    ineligible actual pack records False even when its configured alignment is a
    multiple of ``2 * cp_size``.
    """
    for alignment in (None, "max", 12, 8):
        group = _StubGroup(4, 0)
        pack = _packed_params([0, 1024, 4096], 4096)
        prebuild_balanced_layouts(
            pack, cp_group=group, pad_alignment=alignment, graphs_enabled=True
        )
        assert pack._dsa_cp_balance_layout_cache["zz_pack_ok"] == (1024, True)
        assert ("zigzag", 0) in pack._dsa_cp_balance_layout_cache

    group = _StubGroup(4, 0)
    pack = _packed_params([0, 500, 1008], 1008)
    prebuild_balanced_layouts(pack, cp_group=group, pad_alignment=8)
    assert pack._dsa_cp_balance_layout_cache["zz_pack_ok"] == (252, False)
    assert ("zigzag", 0) not in pack._dsa_cp_balance_layout_cache


def test_prebuild_routes_odd_local_capacity_to_reference():
    """A capacity divisible by CP can still leave an odd number of rows per rank.

    That pack is merely ineligible for two equal synthetic half calls; it is not a
    malformed physical partition and must route to the contiguous reference path.
    """
    group = _StubGroup(4, 0)
    pack = _packed_params([0, 500, 1004], 1004)  # l_local = 251 (odd)
    prebuild_balanced_layouts(pack, cp_group=group, pad_alignment=None)
    assert pack._dsa_cp_balance_layout_cache["zz_pack_ok"] == (251, False)
    assert ("zigzag", 0) not in pack._dsa_cp_balance_layout_cache


def test_prebuild_rejects_capacity_not_divisible_by_cp():
    """A physical capacity that CP cannot partition evenly remains a hard error."""
    pack = _packed_params([0, 500, 1003], 1003)
    with pytest.raises(ValueError, match="physical pack capacity must be divisible"):
        prebuild_balanced_layouts(pack, cp_group=_StubGroup(4, 0))


def _make_balanced_transformer_config(monkeypatch, **overrides):
    """Construct the minimal DSv4 config while isolating optional backend probes."""
    from megatron.core.transformer import transformer_config as transformer_config_module
    from megatron.core.transformer.experimental_attention_variant import dsa_kernels
    from tests.unit_tests.transformer.experimental_attention_variant.test_dsv4_hybrid_attention import (
        _make_config,
    )

    monkeypatch.setattr(dsa_kernels, "use_fused_dsa_kernels", lambda _config: True)
    monkeypatch.setattr(transformer_config_module, "HAVE_PACKAGING", True)
    monkeypatch.setattr(transformer_config_module, "is_te_min_version", lambda _version: True)
    kwargs = dict(
        context_parallel_size=4,
        cp_partition_mode="contiguous",
        sequence_packing_scheduler="dp_balanced",
        max_seqlen_per_dp_cp_rank=4096,
        csa_compress_ratios=[4, 4, 4, 4],
        csa_dense_mode=False,
        dsa_kernel_backend="none",
        dsa_cp_balance_indexer=True,
    )
    kwargs.update(overrides)
    return _make_config(**kwargs)


@pytest.mark.parametrize("alignment", [None, "max", 12, 8])
def test_balanced_config_accepts_all_supported_alignment_policies(monkeypatch, alignment):
    """Startup validation must not pretend alignment proves per-pack eligibility."""
    config = _make_balanced_transformer_config(monkeypatch, pad_packed_seq_alignment=alignment)
    assert config.pad_packed_seq_alignment == alignment


def test_balanced_config_requires_active_context_parallelism(monkeypatch):
    with pytest.raises(ValueError, match="requires active context parallelism"):
        _make_balanced_transformer_config(monkeypatch, context_parallel_size=1)


def test_balanced_config_accepts_dynamic_context_parallelism(monkeypatch):
    config = _make_balanced_transformer_config(
        monkeypatch,
        context_parallel_size=1,
        dynamic_context_parallel=True,
        sequence_packing_scheduler="default_dynamic_cp",
    )
    assert config.dynamic_context_parallel


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"csa_dense_mode": True}, "requires a DSA indexer to exist"),
        ({"csa_compress_ratios": [128, 128, 128, 128]}, "requires a DSA indexer to exist"),
    ],
)
def test_balanced_config_requires_ratio4_indexer(monkeypatch, overrides, match):
    with pytest.raises(ValueError, match=match):
        _make_balanced_transformer_config(monkeypatch, **overrides)


def test_balanced_ratio4_layer_rejects_custom_spec_without_indexer(monkeypatch):
    """A custom module spec cannot silently turn the enabled feature into a no-op."""
    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.transformer.experimental_attention_variant.csa import (
        CompressedSparseAttention,
        CompressedSparseAttentionSubmodules,
    )

    config = _make_balanced_transformer_config(monkeypatch, csa_compress_ratios=[4, 128, 4, 128])
    submodules = CompressedSparseAttentionSubmodules(compressor=None, indexer=None)
    kwargs = dict(
        config=config,
        submodules=submodules,
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        pg_collection=object(),
        rotary_pos_emb=None,
    )
    with pytest.raises(ValueError, match="selected module spec provides none"):
        CompressedSparseAttention(**kwargs, compress_ratio=4)

    # Ratio-128 layers legitimately have no indexer even when another layer in
    # the same model uses the balanced ratio-4 path.
    csa = CompressedSparseAttention(**kwargs, compress_ratio=128)
    assert csa.indexer is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="indexer scoring runs on device")
def test_zigzag_scoring_matches_reference():
    """Numerical parity of the ZIGZAG scoring path (the flagship path): per-rank
    [heads|tails] rows, projected and RoPE'd at the plan's EXPLICIT positions and
    scored against the plan's synthetic per-sequence layouts and prebuilt K
    bounds, then combined via the inverse permutation, must select the same top-k
    as one reference contiguous call over the full pack.

    The packed calls are scored by a layout-HONORING reimplementation of the
    unfused reference formula (relu(q@k) weighted-sum, fp32, torch.topk): the
    in-tree unfused scorer recomputes its masking from (cu_seqlens, global_start)
    and only returns prebuilt_layout as metadata, so it cannot consume synthetic
    layouts — the fused kernel is the layout consumer, and this scorer mirrors
    its documented semantics (per-segment causal offsets and K ranges). Integer
    projection weights make the projection exact in bf16 under any batching, so
    a mismatch isolates position/layout math (e.g. swapped pos_head/pos_tail or
    a RoPE flag divergence from apply_thd_cp_local_rope_unfused)."""
    import megatron.core.transformer.experimental_attention_variant.cp_balanced_indexer as M
    from megatron.core.transformer.experimental_attention_variant.csa_utils import cp_utils as _cu
    from megatron.core.transformer.experimental_attention_variant.dsa import rotate_activation

    torch.manual_seed(3)
    dev = torch.device("cuda")
    cp_size, cap = 4, 4096
    cu_list = [0, 1024, 3072, 4096]
    heads, dim, pos_dim, q_lora, ratio, topk = 4, 64, 32, 128, 4, 64
    nope = dim - pos_dim
    l_local, half = cap // cp_size, cap // cp_size // 2
    msq = 2048  # per-sequence maximum

    class _Cfg2:
        apply_rope_fusion = False
        rotary_interleaved = False

    class _StubRotary:
        def __init__(self, rpe):
            self._rpe = rpe

        def __call__(self, length, packed_seq=True):
            return self._rpe

    class _StubIndexer:
        def __init__(self):
            self._w = torch.randint(-2, 3, (heads * dim, q_lora), device=dev).to(torch.bfloat16)
            self.rotary_pos_emb = _StubRotary(
                torch.randn(msq, 1, 1, pos_dim, dtype=torch.float32, device=dev)
            )
            self.index_n_heads, self.index_head_dim, self.qk_pos_emb_head_dim = heads, dim, pos_dim

        def linear_wq_b(self, x):
            return x.reshape(-1, q_lora) @ self._w.t(), None

    stub = _StubIndexer()
    qr = torch.randint(-2, 3, (cap, 1, q_lora), device=dev).to(torch.bfloat16)
    w = (torch.rand(cap, heads, dtype=torch.float32, device=dev) + 0.5).to(torch.bfloat16)
    cu = torch.tensor(cu_list, dtype=torch.int32, device=dev)
    cu_comp = _comp_cu(cu.cpu()).to(dev)
    comp = int(cu_comp[-1])
    k = torch.randn(comp, dim, dtype=torch.bfloat16, device=dev)
    scale = dim**-0.5

    def _layout_topk(q, w_rows, k_pass, layout3):
        """Score exactly like the unfused reference formula, but with masking taken
        from the synthetic (cu_q, cu_k, q_causal_offsets) layout — the fused
        kernel's contract: a row at in-call position p of segment i has true
        sequence position p + offsets[i] and sees keys cu_k[i]:cu_k[i]+visible."""
        cu_q, cu_k, offs = (t.long() for t in layout3)
        rows = q.shape[0]
        nseg = cu_q.shape[0] - 1
        ar = torch.arange(rows, device=q.device)
        seg = torch.bucketize(ar, cu_q[1:], right=True).clamp_max(nseg - 1)
        pos = ar - cu_q[seg] + offs[seg]
        seg_k_len = (
            torch.minimum(cu_k[1:], torch.full_like(cu_k[1:], k_pass.shape[0]))[seg] - cu_k[seg]
        )
        visible = torch.minimum(
            torch.div(pos + 1, ratio, rounding_mode="floor"), seg_k_len
        ).clamp_min(0)
        k_rows = torch.arange(k_pass.shape[0], device=q.device)
        k_seg = torch.bucketize(k_rows, cu_k[1:], right=True).clamp_max(nseg - 1)
        k_pos = k_rows - cu_k[k_seg]
        scores = torch.einsum("rhd,kd->rhk", q.float(), k_pass.float())
        scores = torch.relu(scores) * w_rows.float().unsqueeze(-1)
        scores = scores.sum(dim=1) * float(scale)
        valid = (k_seg.unsqueeze(0) == seg.unsqueeze(1)) & (
            k_pos.unsqueeze(0) < visible.unsqueeze(1)
        )
        scores = scores.masked_fill(~valid, float("-inf"))
        width = min(topk, k_pass.shape[0])
        values, sel = torch.topk(scores, width, dim=-1)
        out = torch.full((rows, topk), -1, dtype=torch.int32, device=q.device)
        out[:, :width] = torch.where(torch.isfinite(values), k_pos[sel].to(torch.int32), -1)
        return out

    def _project_rope(rows_q, pos, layout3):
        q, _ = stub.linear_wq_b(rows_q)
        q = q.reshape(-1, heads, dim)
        if pos is None:  # reference: implicit positions from (cu, gs=0)
            rpe = stub.rotary_pos_emb(msq, packed_seq=True)
            q = _cu.apply_thd_cp_local_rope_unfused(q, rpe, nope, pos_dim, cu, 0, _Cfg2())
        else:  # zigzag: explicit per-row positions from the plan
            q = M._rope_positions(q, pos, layout3[0], nope, pos_dim, stub, _Cfg2(), msq)
        return rotate_activation(q)

    q_ref = _project_rope(qr, None, None)
    tk_ref, _ = _cu.compute_cp_indexer_topk(
        q_ref,
        w.reshape(-1, heads),
        k,
        cu,
        cu_comp,
        0,
        ratio,
        topk,
        scale,
        max_seqlen_q=msq,
        use_fused=False,
    )

    # Per-rank zigzag: gather -> two packed layout-honoring calls -> inverse stitch.
    Z = torch.empty((cap, topk), dtype=tk_ref.dtype, device=dev)
    plans = []
    for r in range(cp_size):
        psp = _packed_params(cu_list, cap)
        psp.cu_seqlens_q = psp.cu_seqlens_q.to(dev)
        psp.cu_seqlens_q_padded = psp.cu_seqlens_q_padded.to(dev)
        prebuild_balanced_layouts(psp, cp_group=_StubGroup(cp_size, r))
        plans.append(psp._dsa_cp_balance_layout_cache[("zigzag", r)])
    for r in range(cp_size):
        plan = plans[r]
        rows_q = qr.index_select(0, plan["gather_idx"])
        rows_w = w.index_select(0, plan["gather_idx"])
        parts = []
        for sl, lay, pos, kend_k in (
            (slice(0, half), "head_layout", "pos_head", "k_end_head"),
            (slice(half, None), "tail_layout", "pos_tail", "k_end_tail"),
        ):
            k_end = plan[kend_k]
            k_pass = k[:k_end] if k_end < comp else k
            q_c = _project_rope(rows_q[sl], plan[pos], plan[lay])
            parts.append(_layout_topk(q_c, rows_w[sl], k_pass, plan[lay]))
        Z[r * l_local : (r + 1) * l_local] = torch.cat(parts)
    for r in range(cp_size):
        mine = Z.index_select(0, plans[r]["inv_idx"])
        ref = tk_ref[r * l_local : (r + 1) * l_local]
        a, _ = torch.sort(mine, dim=-1)
        b, _ = torch.sort(ref, dim=-1)
        assert torch.equal(a, b), (r, int((a != b).any(dim=-1).sum()))


@_fused_kernel_test
def test_fused_reduced_mq_matches_full_mq():
    """The production balanced path always passes a REDUCED max_seqlen_q (the
    chunk's row count) with causal offsets reaching far beyond it. Verify on the
    real fused kernel that max_seqlen_q does not participate in masking: the same
    rows scored with mq=rows and with mq=T must select identical keys. Integer
    q/w/k make the scores tie-free so the comparison is exact."""
    from megatron.core.transformer.experimental_attention_variant.csa_utils import cp_utils as _cu

    torch.manual_seed(23)
    dev = torch.device("cuda")
    T, heads, dim, ratio, topk = 262144, 64, 128, 4, 64  # production head shape
    gs, sz = 61440, 1024
    cu = torch.tensor([0, T], dtype=torch.int32, device=dev)
    cu_comp = _comp_cu(cu.cpu()).to(dev)
    # Tie-free signature data (see _signature_qkw): every key scores a unique
    # integer, so the comparison is exact under any kernel tiling.
    q, w, k, sig_perm = _signature_qkw(sz, int(cu_comp[-1]), heads, dim, dev)

    def _call(mq):
        tk, _ = _cu.compute_cp_indexer_topk(
            q,
            w,
            k,
            cu,
            cu_comp,
            gs,
            ratio,
            topk,
            dim**-0.5,
            max_seqlen_q=mq,
            use_fused=True,
            max_seqlen_kv=16384,
        )
        return tk

    a, _ = torch.sort(_call(sz), dim=-1)
    b, _ = torch.sort(_call(T), dim=-1)
    torch.cuda.synchronize()
    assert torch.equal(a, b), int((a != b).any(dim=-1).sum())
    # Anti-degeneration anchor: the signature data has a computable ground truth;
    # a kernel mode that zeroes all scores (e.g. unsupported head counts) would
    # make both calls agree on garbage — pin one row against the truth.
    r = sz - 1
    vis = min((gs + r + 1) // ratio, int(cu_comp[-1]))
    gt, _ = torch.sort(torch.topk(sig_perm[:vis].float(), topk).indices.to(torch.int32))
    assert torch.equal(a[r], gt), (a[r][:8].tolist(), gt[:8].tolist())


def _signature_qkw(rows, comp, heads, dim, dev):
    """Tie-free scoring inputs: head h (h < 4) reads base-16 digit h of a random
    permutation via a one-hot q; the base-16 place values live in w, so
    score(key i) == perm[i] exactly (every bf16 product is a sub-16 integer, the
    place multiplication happens in the wrapper's weight path). The top-k of any
    causal window is then fully determined — any per-segment offset or K-range
    error changes the visible window and therefore the selected set.

    NOTE: callers must use the PRODUCTION head count (64). The fused kernel
    silently returns all-zero scores (degenerating the top-k to the first
    indices in order) for head counts below its tile width — measured: 4 heads
    degenerate, 32/64 correct. That silent mode also poisoned earlier versions
    of these tests into false passes/failures.
    """
    assert heads >= 32, "fused kernel silently degenerates below ~32 heads"
    assert comp <= 16**4
    perm = torch.randperm(comp, device=dev)
    q = torch.zeros(rows, heads, dim, dtype=torch.bfloat16, device=dev)
    for h in range(4):
        q[:, h, h] = 1.0
    k = torch.zeros(comp, dim, dtype=torch.bfloat16, device=dev)
    for h in range(4):
        k[:, h] = ((perm // (16 ** (3 - h))) % 16).to(torch.bfloat16)
    w = torch.zeros(rows, heads, dtype=torch.bfloat16, device=dev)
    for h in range(4):
        w[:, h] = float(16 ** (3 - h))
    return q, w, k, perm
