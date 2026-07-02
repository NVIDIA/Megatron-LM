# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Hetero TP/PP reshard of Mamba snapshot state.

Builds a known global Mamba state, shards it to a source (tp, pp) the way
mamba_mixer does ([x|B|C] conv bands, head-sharded ssm, layers split by PP),
and asserts every destination rank reconstructs a direct shard of the global
state through three paths: the raw plan, the push-transport slice/scatter,
and the pull-transport byte-fragment reads.
"""

import dataclasses

import pytest
import torch

from megatron.core.inference.disaggregation.kv_transfer_pull import _snapshot_fragment_pulls
from megatron.core.inference.disaggregation.mamba_reshard import (
    MambaShardLayout,
    MambaStateDims,
    plan_mamba_reshard,
)

NHEADS, HEADDIM, DSTATE, NGROUPS, DCONV = 8, 4, 2, 2, 3
M = 4  # global Mamba layers
N = 2  # snapshots per request
D_INNER = NHEADS * HEADDIM
G = NGROUPS * DSTATE  # B and C band global size
CONV_DIM = D_INNER + 2 * G


def _dims():
    return MambaStateDims(
        nheads=NHEADS, headdim=HEADDIM, d_state=DSTATE, ngroups=NGROUPS, d_conv=DCONV
    )


def _layouts(tp, pp, base=0):
    """One MambaShardLayout per rank for a (tp, pp) instance; rank = base+p*tp+r.
    PP splits the M layers evenly (contiguous per stage). `base` offsets the
    global ranks so prefill and decode occupy disjoint rank windows."""
    per = M // pp
    out = {}
    for p in range(pp):
        for r in range(tp):
            rank = base + p * tp + r
            out[rank] = MambaShardLayout(
                global_rank=rank,
                tp_size=tp,
                tp_rank=r,
                layer_start=p * per,
                num_layers=per,
                dims=_dims(),
            )
    return out


def _global_snapshots():
    """Distinct value per (snapshot, layer, channel/head) so any mis-slice is
    caught. Staging layout: conv (N, M, CONV_DIM, DCONV), ssm (N, M, NHEADS,
    HEADDIM, DSTATE)."""
    conv = torch.arange(N * M * CONV_DIM * DCONV, dtype=torch.float32).reshape(
        N, M, CONV_DIM, DCONV
    )
    ssm = (
        torch.arange(N * M * NHEADS * HEADDIM * DSTATE, dtype=torch.float32).reshape(
            N, M, NHEADS, HEADDIM, DSTATE
        )
        + 100_000.0
    )
    return conv, ssm


def _shard(conv_g, ssm_g, lay: MambaShardLayout):
    """Shard the global snapshot staging to one rank exactly as mamba_mixer
    packs it: conv channel axis = [x | B | C] local bands, ssm head axis."""
    s, e = lay.layer_range()
    r, tp = lay.tp_rank, lay.tp_size
    di_l = D_INNER // tp
    g_l = (NGROUPS // tp) * DSTATE
    x = conv_g[:, s:e, 0:D_INNER][:, :, r * di_l : (r + 1) * di_l]
    b = conv_g[:, s:e, D_INNER : D_INNER + G][:, :, r * g_l : (r + 1) * g_l]
    c = conv_g[:, s:e, D_INNER + G : D_INNER + 2 * G][:, :, r * g_l : (r + 1) * g_l]
    conv_l = torch.cat([x, b, c], dim=2).contiguous()
    nh_l = NHEADS // tp
    ssm_l = ssm_g[:, s:e, r * nh_l : (r + 1) * nh_l].contiguous()
    return conv_l, ssm_l


PARALLELISM_SWEEP = [
    ((2, 1), (1, 1)),  # TP2 -> TP1 (band merge)
    ((1, 1), (2, 1)),  # TP1 -> TP2 (band split)
    ((1, 2), (1, 1)),  # PP2 -> PP1 (layer merge)
    ((1, 1), (1, 2)),  # PP1 -> PP2 (layer split)
    ((2, 2), (1, 1)),  # both axes hetero
    ((2, 1), (2, 1)),  # identity
]


@pytest.mark.parametrize("src,dst", PARALLELISM_SWEEP)
def test_push_snapshot_reshard_reconstructs_destination(src, dst):
    """The push path's slice/scatter: each transfer moves
    staging[:, src_layer, src_lo:src_hi] into the destination staging at
    [:, dst_layer, dst_lo:dst_hi] (the exact indexing kv_transfer_push
    uses). Every destination rank must end up a direct shard of the global
    state."""
    conv_g, ssm_g = _global_snapshots()
    src_lay, dst_lay = _layouts(*src), _layouts(*dst, base=8)
    src_t = {rk: _shard(conv_g, ssm_g, lay) for rk, lay in src_lay.items()}
    dst_t = {
        rk: (
            torch.zeros(N, lay.num_layers, lay.conv_dim_local, DCONV),
            torch.zeros(N, lay.num_layers, lay.nheads_local, HEADDIM, DSTATE),
        )
        for rk, lay in dst_lay.items()
    }

    plan = plan_mamba_reshard(list(src_lay.values()), list(dst_lay.values()))
    for t in plan:
        src_conv, src_ssm = src_t[t.src_rank]
        dst_conv, dst_ssm = dst_t[t.dst_rank]
        sub = (src_conv if t.is_conv else src_ssm)[:, t.src_layer, t.src_lo : t.src_hi]
        (dst_conv if t.is_conv else dst_ssm)[:, t.dst_layer, t.dst_lo : t.dst_hi] = sub

    for rk, lay in dst_lay.items():
        want_conv, want_ssm = _shard(conv_g, ssm_g, lay)
        assert torch.equal(dst_t[rk][0], want_conv), f"conv mismatch at rank {rk} ({src}->{dst})"
        assert torch.equal(dst_t[rk][1], want_ssm), f"ssm mismatch at rank {rk} ({src}->{dst})"


class _FakeByteBackend:
    """Applies begin_pull_raw triples by copying bytes between the pools it
    was given, byte-offset-relative to each pool's start (what the region
    base_addr is on the real backend)."""

    class _Done:
        def wait(self):
            pass

    def __init__(self, src_pools):
        # src_pools: agent_name -> {region: tensor}
        self._src_pools = src_pools
        self._dst_pools = {}  # region -> tensor (set by the test)

    def begin_pull_raw(self, peer_meta, region, triples):
        src = self._src_pools[peer_meta["agent_name"]][region].flatten().view(torch.uint8)
        dst = self._dst_pools[region].flatten().view(torch.uint8)
        for lo, ro, nb in triples:
            dst[lo : lo + nb] = src[ro : ro + nb]
        return self._Done()


class _FakeSlotAllocator:
    def __init__(self, num_layers, slots):
        self.conv_states = torch.zeros(num_layers, slots, CONV_DIM, DCONV)
        self.ssm_states = torch.zeros(num_layers, slots, NHEADS, HEADDIM, DSTATE)


class _FakeCtx:
    def __init__(self, sa):
        self.mamba_slot_allocator = sa


def _pool_meta(tensor):
    """The region layout the real backend exports for a (layers, slots,
    *state) pool: per-layer and per-slot strides in bytes."""
    elem = tensor.element_size()
    return {"outer_stride_bytes": tensor.stride(0) * elem, "inner_bytes": tensor.stride(1) * elem}


@pytest.mark.parametrize("src,dst", PARALLELISM_SWEEP)
def test_pull_snapshot_fragments_reconstruct_destination(src, dst):
    """The pull path's byte-fragment reads: each destination rank reads its
    band slices of every snapshot out of the source ranks' slot pools via
    _snapshot_fragment_pulls, landing them in its own pool slots."""
    conv_g, ssm_g = _global_snapshots()
    src_lay, dst_lay = _layouts(*src), _layouts(*dst, base=8)
    hashes = [1111, 2222]
    src_slot_of = {1111: 3, 2222: 5}  # arbitrary non-contiguous source slots
    dst_slots = [4, 1]  # arbitrary destination slots

    # Source pools: each rank's snapshot shards parked at the source slots.
    src_pools, src_region_meta, src_snap_slots = {}, {}, {}
    for rk, lay in src_lay.items():
        conv_l, ssm_l = _shard(conv_g, ssm_g, lay)
        conv_pool = torch.zeros(lay.num_layers, 8, lay.conv_dim_local, DCONV)
        ssm_pool = torch.zeros(lay.num_layers, 8, lay.nheads_local, HEADDIM, DSTATE)
        for k, h in enumerate(hashes):
            conv_pool[:, src_slot_of[h]] = conv_l[k]
            ssm_pool[:, src_slot_of[h]] = ssm_l[k]
        name = f"src{rk}"
        src_pools[name] = {"snap_conv": conv_pool, "snap_ssm": ssm_pool}
        src_region_meta[rk] = {
            "agent_name": name,
            "regions": {"snap_conv": _pool_meta(conv_pool), "snap_ssm": _pool_meta(ssm_pool)},
        }
        src_snap_slots[rk] = dict(src_slot_of)

    backend = _FakeByteBackend(src_pools)
    plan = plan_mamba_reshard(list(src_lay.values()), list(dst_lay.values()))
    alloc = {"hashes": hashes, "slots": dst_slots, "block_ids": [10, 11], "keep_indices": [0, 1]}

    for rk, lay in dst_lay.items():
        sa = _FakeSlotAllocator(lay.num_layers, 8)
        sa.conv_states = torch.zeros(lay.num_layers, 8, lay.conv_dim_local, DCONV)
        sa.ssm_states = torch.zeros(lay.num_layers, 8, lay.nheads_local, HEADDIM, DSTATE)
        backend._dst_pools = {"snap_conv": sa.conv_states, "snap_ssm": sa.ssm_states}
        transfers = [t for t in plan if t.dst_rank == rk]
        handles = _snapshot_fragment_pulls(
            _FakeCtx(sa), backend, lay, transfers, alloc, src_snap_slots, src_region_meta
        )
        for h in handles:
            h.wait()

        want_conv, want_ssm = _shard(conv_g, ssm_g, lay)
        for k in range(N):
            slot = dst_slots[k]
            assert torch.equal(
                sa.conv_states[:, slot], want_conv[k]
            ), f"conv mismatch at rank {rk} snapshot {k} ({src}->{dst})"
            assert torch.equal(
                sa.ssm_states[:, slot], want_ssm[k]
            ), f"ssm mismatch at rank {rk} snapshot {k} ({src}->{dst})"


def test_mamba_dedupes_replica_sources():
    """Two source ranks holding the same Mamba shard (same tp_rank +
    layer_start, e.g. EP/DP replicas) are deduped: the shard is sourced from
    exactly one of them (smallest global_rank)."""

    def _lay(gr):
        return MambaShardLayout(
            global_rank=gr, tp_size=1, tp_rank=0, layer_start=0, num_layers=M, dims=_dims()
        )

    plan = plan_mamba_reshard([_lay(0), _lay(1)], [_lay(2)])
    assert {t.src_rank for t in plan} == {0}


def test_mamba_rejects_indivisible_groups():
    """ngroups < tp_size would shard the B/C groups to zero width; reject it
    up front."""
    with pytest.raises(ValueError):
        MambaShardLayout(
            global_rank=0,
            tp_size=4,
            tp_rank=0,
            layer_start=0,
            num_layers=1,
            dims=MambaStateDims(nheads=8, headdim=HEADDIM, d_state=DSTATE, ngroups=2, d_conv=DCONV),
        )


def test_layout_wire_roundtrip():
    """Layouts cross the coordinator as plain dicts (asdict) and are rebuilt
    via MambaShardLayout(**dict); the nested dims dict must coerce back to
    MambaStateDims."""
    lay = MambaShardLayout(
        global_rank=1, tp_size=2, tp_rank=1, layer_start=0, num_layers=M, dims=_dims()
    )
    rebuilt = MambaShardLayout(**dataclasses.asdict(lay))
    assert rebuilt == lay
    assert rebuilt.conv_dim_local == lay.conv_dim_local
