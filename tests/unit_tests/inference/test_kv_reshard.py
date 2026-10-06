# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Correctness of hetero TP/PP/EP KV resharding (single process).

We materialize a global KV tensor, split it into a *source* layout's
shards, run the reshard plan to assemble a *destination* layout's
shards, and assert each dst shard equals the direct split of the global
KV. Sweeping many (Tp,Pp,Td,Pd) combos -- divisible, non-divisible,
PP-changing, and EP-replicated -- exercises the range-intersection
planner end to end without any distributed runtime.
"""

import pytest
import torch

from megatron.core.inference.disaggregation.kv_reshard import KVShardLayout, plan_kv_reshard
from megatron.core.inference.disaggregation.utils import transfers_for_dst

# global model
L, Hh, BC, BS, HD = 12, 8, 2, 4, 5  # layers, kv-heads, block_count, block_size, head_dim


def _global_kv():
    # [2(K/V), L, BC, BS, H, HD] with unique values per (kv, layer, head)
    g = torch.zeros(2, L, BC, BS, Hh, HD)
    for kv in range(2):
        for l in range(L):
            for h in range(Hh):
                g[kv, l, :, :, h, :] = (kv * 1_000_000) + l * 1000 + h
    return g


def _shard_of(global_kv, lay: KVShardLayout):
    """The dst staging tensor a worker with layout `lay` should hold:
    [BC, 2, local_layers, BS, local_heads, HD] (export's attn layout)."""
    l0, l1 = lay.layer_range()
    h0, h1 = lay.head_range()
    # global_kv is [2, L, BC, BS, H, HD]; export layout is
    # [BC, 2, layers, BS, heads, HD]
    sub = global_kv[:, l0:l1, :, :, h0:h1, :]  # [2, ll, BC, BS, hh, HD]
    return sub.permute(2, 0, 1, 3, 4, 5).contiguous()  # [BC,2,ll,BS,hh,HD]


def _make_layouts(tp, pp, ep=1, etp=1):
    outs = []
    rank = 0
    for p in range(pp):
        for t in range(tp):
            for e in range(ep):
                for et in range(etp):
                    outs.append(
                        KVShardLayout(
                            num_layers=L,
                            num_heads=Hh,
                            tp_size=tp,
                            tp_rank=t,
                            pp_size=pp,
                            pp_rank=p,
                            global_rank=rank,
                            ep_size=ep,
                            ep_rank=e,
                            etp_size=etp,
                            etp_rank=et,
                        )
                    )
                    rank += 1
    return outs


def _run_reshard(src_layouts, dst_layouts):
    g = _global_kv()
    # src buffers = each src's correct shard of the global KV
    src_buf = {s.global_rank: _shard_of(g, s) for s in src_layouts}
    plan = plan_kv_reshard(src_layouts, dst_layouts)
    by_rank = {s.global_rank: s for s in src_layouts}
    out = {}
    for d in dst_layouts:
        dst = torch.full((BC, 2, d.local_num_layers(), BS, d.local_num_heads(), HD), -999.0)
        for t in transfers_for_dst(plan, d.global_rank):
            s = by_rank[t.src_rank]
            block = src_buf[t.src_rank][:, :, t.src_layer_slice(s), :, t.src_head_slice(s), :]
            dst[:, :, t.dst_layer_slice(d), :, t.dst_head_slice(d), :] = block
        out[d.global_rank] = dst
    return g, out


@pytest.mark.parametrize(
    "src,dst",
    [
        ((1, 1), (1, 1)),  # homogeneous
        ((2, 1), (4, 1)),  # TP fan-out (divisible)
        ((4, 1), (2, 1)),  # TP merge (divisible)
        ((1, 2), (1, 3)),  # PP change (divisible both)
        ((2, 2), (4, 3)),  # both change
        ((2, 3), (4, 2)),  # TP + PP mixed
    ],
)
def test_reshard_matches_direct_split(src, dst):
    tp_s, pp_s = src
    tp_d, pp_d = dst
    # skip layouts that violate divisibility of the GLOBAL dims
    if Hh % tp_s or Hh % tp_d or L % pp_s or L % pp_d:
        pytest.skip("layout not divisible for this global model")
    src_layouts = _make_layouts(tp_s, pp_s)
    dst_layouts = _make_layouts(tp_d, pp_d)
    g, out = _run_reshard(src_layouts, dst_layouts)
    for d in dst_layouts:
        expected = _shard_of(g, d)
        got = out[d.global_rank]
        assert torch.equal(got, expected), f"dst rank {d.global_rank} mismatch"
        assert (got != -999.0).all(), "some dst entries never received"


def _assert_one_source_per_shard(plan, src_layouts):
    """Each attention shard (tp_rank, pp_rank) must be sourced by exactly
    one rank -- no duplicate sends from EP/ETP replicas."""
    src_by_rank = {s.global_rank: s for s in src_layouts}
    shard_sources = {}
    for t in plan:
        s = src_by_rank[t.src_rank]
        shard_sources.setdefault(s.kv_shard_key(), set()).add(t.src_rank)
    for key, ranks in shard_sources.items():
        assert len(ranks) == 1, f"shard {key} sourced by {ranks}"


@pytest.mark.parametrize("ep,etp", [(2, 1), (1, 2), (2, 2)])
def test_expert_replication_picks_single_source(ep, etp):
    """EP- and/or ETP-replicated sources: each attention shard is sourced
    once; every dst (any EP/ETP replica) still gets correct, complete data.
    EP and ETP shard the expert FFN, not the KV, so they're pure replicas."""
    src_layouts = _make_layouts(tp=2, pp=1, ep=ep, etp=etp)
    dst_layouts = _make_layouts(tp=2, pp=1, ep=ep, etp=etp)
    plan = plan_kv_reshard(src_layouts, dst_layouts)
    _assert_one_source_per_shard(plan, src_layouts)
    g, out = _run_reshard(src_layouts, dst_layouts)
    for d in dst_layouts:
        assert torch.equal(out[d.global_rank], _shard_of(g, d))


def test_hetero_tp_with_expert_replication():
    """Hetero attention TP merge (4->2) while sources are also ETP-replicated:
    the reshard still merges heads correctly and dedupes the ETP replicas."""
    src_layouts = _make_layouts(tp=4, pp=1, etp=2)  # 8 ranks, 4 attn shards x2
    dst_layouts = _make_layouts(tp=2, pp=1)
    plan = plan_kv_reshard(src_layouts, dst_layouts)
    _assert_one_source_per_shard(plan, src_layouts)
    g, out = _run_reshard(src_layouts, dst_layouts)
    for d in dst_layouts:
        assert torch.equal(out[d.global_rank], _shard_of(g, d))


def test_one_prefill_to_multiple_decode_targets_of_different_parallelism():
    """A single prefill source set reshards correctly to several decode
    targets that each use a DIFFERENT (Tp,Pp) -- e.g. a heterogeneous
    decode pool. Each target is an independent reshard (one plan call per
    target replica); the planner imposes no shared parallelism across
    targets."""
    src_layouts = _make_layouts(tp=2, pp=2)  # prefill: TP2 x PP2
    targets = [(4, 1), (2, 1), (1, 3), (4, 3)]  # decode replicas, all different
    g = _global_kv()
    for tp_d, pp_d in targets:
        dst_layouts = _make_layouts(tp_d, pp_d)
        _, out = _run_reshard(src_layouts, dst_layouts)
        for d in dst_layouts:
            assert torch.equal(
                out[d.global_rank], _shard_of(g, d)
            ), f"decode target TP{tp_d}xPP{pp_d} rank {d.global_rank} mismatch"


def test_uneven_pp_attention_window():
    """Attention layers split UNEVENLY across PP (hybrid-style) via explicit
    (layer_start, num_local_layers); reshard to pp=1 still reconstructs the
    global KV. The even-split default would map the wrong global layers here."""
    src = [
        KVShardLayout(L, Hh, 1, 0, 2, 0, 0, layer_start=0, num_local_layers=5),
        KVShardLayout(L, Hh, 1, 0, 2, 1, 1, layer_start=5, num_local_layers=7),
    ]
    dst = [KVShardLayout(L, Hh, 1, 0, 1, 0, 2)]  # pp=1: all L layers on one rank
    assert src[0].layer_range() == (0, 5) and src[1].layer_range() == (5, 12)
    g, out = _run_reshard(src, dst)
    for d in dst:
        assert torch.equal(out[d.global_rank], _shard_of(g, d))


def test_explicit_layer_window_is_all_or_nothing():
    # Setting only one of (layer_start, num_local_layers) would silently fall
    # back to the even-split count -- reject it.
    with pytest.raises(ValueError):
        KVShardLayout(L, Hh, 1, 0, 2, 0, 0, layer_start=0)
    with pytest.raises(ValueError):
        KVShardLayout(L, Hh, 1, 0, 2, 0, 0, num_local_layers=5)
