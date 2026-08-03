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
    _a2a_meta,
    _zigzag_plan,
    _ZZ_PACK_OK,
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
    # Unique group_name per instance: module-level registries in cp_balanced_indexer
    # ( _LAST_PLAN / _ZZ_PACK_OK / _MULTI_SEQ_LAST ) key on it, and CPython id() reuse
    # after GC could otherwise leak captured-plan state between tests.
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
        cu = psp.cu_seqlens_q_padded
        # The gate mirrors the forward probe (csa.py), which sees the PADDED cu:
        # a capacity tail merged into the last sequence counts as one sequence.
        expected_multi = int(((cu[1:] - cu[:-1]) > 0).sum()) != 1
        assert psp._dsa_cp_multi_seq is expected_multi
        prebuilt = psp._dsa_cp_balance_layout_cache[("zigzag", r)]
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

@pytest.mark.skipif(not torch.cuda.is_available(), reason="indexer scoring runs on device")
@pytest.mark.parametrize("use_fused", [False])
def test_chunk_scoring_matches_full_call(use_fused):
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
@pytest.mark.parametrize("cp_size,l_local", [(4, 1024), (5, 10), (16, 16384), (3, 7)])
def test_folding_a2a_meta_roundtrip(cp_size, l_local):
    """The folding fallback's fixed chunk-permutation all_to_all metadata: dispatch must
    hand every rank exactly its head chunk (r) and tail chunk (2N-1-r) rows in order,
    and combine must return each computed chunk to its contiguous owner (bit-exact
    inverse), including odd cp_size (merged-pair peer), odd l_local (uneven chunk
    sizes), and the swap_pair send convention."""
    N, nch = cp_size, 2 * cp_size
    S = N * l_local
    bounds = [(k * S) // nch for k in range(nch + 1)]
    groups = [_StubGroup(N, r) for r in range(N)]
    metas = [_a2a_meta(groups[r], N, l_local) for r in range(N)]
    payload = torch.arange(S, dtype=torch.int64).unsqueeze(1)

    # Dispatch: each owner sends its two half-chunks, peer-ordered (swap_pair).
    sends = []
    for r in range(N):
        rows = payload[r * l_local : (r + 1) * l_local]
        s0 = metas[r]["s0"]
        sends.append(torch.cat((rows[s0:], rows[:s0])) if metas[r]["swap_pair"] else rows)
    recvs = _sim_all_to_all(sends, [m["d_in"] for m in metas])
    heads_tails = []
    for r in range(N):
        sh, st = metas[r]["sh"], metas[r]["st"]
        head_c, tail_c = r, nch - 1 - r
        head, tail = recvs[r][:sh], recvs[r][sh:]
        assert torch.equal(head.squeeze(1), payload[bounds[head_c] : bounds[head_c + 1]].squeeze(1))
        assert torch.equal(tail.squeeze(1), payload[bounds[tail_c] : bounds[tail_c + 1]].squeeze(1))
        assert st == bounds[tail_c + 1] - bounds[tail_c]
        heads_tails.append((head, tail))

    # Combine: computed [head | tail] rows return to their contiguous owners.
    sends2 = [torch.cat(ht) for ht in heads_tails]
    recvs2 = _sim_all_to_all(sends2, [m["c_in"] for m in metas])
    for r in range(N):
        s0, s1 = metas[r]["s0"], metas[r]["s1"]
        rec = recvs2[r]
        out = torch.cat((rec[s1:], rec[:s1])) if metas[r]["swap_pair"] else rec
        mine = torch.arange(r * l_local, (r + 1) * l_local, dtype=torch.int64)
        assert torch.equal(out.squeeze(1), mine), (cp_size, l_local, r)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused indexer kernel required")
def test_fused_tight_width_smoke():
    """Exercise the empirical tight-width kernel contract (_KV_TIGHT_WIDTH_CEILING):
    a fused call whose score width (16384) is narrower than the declared per-sequence
    compressed KV length must complete without an illegal memory access and return
    causally valid indices. Guards the contract the K-bound optimization rests on."""
    from megatron.core.transformer.experimental_attention_variant.csa_utils import cp_utils as _cu

    torch.manual_seed(7)
    dev = torch.device("cuda")
    T, heads, dim, ratio, topk = 131072, 4, 64, 4, 64
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

@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused indexer kernel required")
def test_fused_sliced_k_prefix_view_smoke():
    """The zigzag consumer feeds the fused kernel a PREFIX VIEW of the gathered K
    (k_seq_major[:k_end]) together with a layout whose declared per-sequence K
    ranges may extend past the slice (mkv is a capacity, k_end the causal need).
    Reads past the slice must land on valid full-buffer memory — exercise that
    combination end to end and check the indices stay causally bounded."""
    from megatron.core.transformer.experimental_attention_variant.csa_utils import cp_utils as _cu

    torch.manual_seed(11)
    dev = torch.device("cuda")
    T, heads, dim, ratio, topk = 65536, 4, 64, 4, 64
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

@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused indexer kernel required")
def test_fused_tight_width_ceiling_smoke():
    """Boundary case of the empirical kernel contract: a tight score width EXACTLY at
    _KV_TIGHT_WIDTH_CEILING (65536), narrower than the declared per-sequence compressed
    KV length, must complete without an illegal memory access. Guards the exact edge
    the K-bound optimization is allowed to reach."""
    from megatron.core.transformer.experimental_attention_variant.csa_utils import cp_utils as _cu

    torch.manual_seed(13)
    dev = torch.device("cuda")
    T, heads, dim, ratio, topk = 524288, 4, 64, 4, 64
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
