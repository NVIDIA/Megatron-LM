# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Prefill->decode KV transfer, pull family: one-sided (RDMA, NIXL) hand-off.

The prefill registers its paged KV (+ Mamba) buffers ONCE and the decode rank
READs the prefill's blocks straight into its own freshly-allocated blocks -- no
staging copy, no per-request registration. The prefill publishes only references
(block ids + Mamba slot + the static region meta); the coordinator relays them
opaque. Mirrors the reference NIXL backend. The two-sided (push) family lives in
``kv_transfer_push.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import torch

from megatron.core.inference.disaggregation import kv_reshard, mamba_reshard, utils


def publish_request_kv(backend, ref_payload, my_layout):
    """(prefill, one-sided) Build this rank's per-request hand-off: the static
    region meta plus the request's source block ids / Mamba slot. No copy -- the
    decode READs these from the registered ``memory_buffer``. ``ref_payload``
    comes from ``context.export_request_kv_ref``."""
    return {
        "transport": "nixl",
        "shard_key": list(my_layout.kv_shard_key()),
        "global_rank": int(my_layout.global_rank),
        "region_meta": backend.export_regions_meta(),
        "block_ids": ref_payload["block_ids"],
        "block_hashes": ref_payload["block_hashes"],
        "block_count": ref_payload["block_count"],
        "mamba_src_slot": ref_payload.get("mamba_src_slot", -1),
        "mamba_dims": ref_payload.get("mamba_dims"),
        "snapshots": ref_payload.get("snapshots", []),
        "layout": ref_payload["layout"],
        # geometry of the prefill rank's KV buffer, for hetero (fragment) reads
        "kv_dims": {
            "num_layers": ref_payload["num_layers"],
            "total_blocks": ref_payload["total_blocks"],
            "block_size": ref_payload["block_size_tokens"],
            "heads": ref_payload["num_heads_per_partition"],
            "hidden": ref_payload["hidden_per_head"],
            "elem": ref_payload["elem_size"],
        },
    }


@dataclass
class NixlPullRecv:
    """Decode side: an in-flight one-sided READ of a request's blocks.
    :meth:`finish` waits the read, then registers block hashes + binds the Mamba
    slot (the KV already landed in the allocated blocks), mirroring
    ``DecodeRecv.finish``. If the prefill published Mamba prefix-cache
    ``snapshots``, a second READ pulls them after the KV commit (their slots are
    protected by the KV pin)."""

    handles: List[Any]
    block_ids: List[int]
    block_hashes: List[int]
    mamba_dst_slot: int
    backend: Any = None
    peer_meta: Optional[dict] = None
    snapshots: List = field(default_factory=list)

    def finish(self, engine: Any) -> Optional[dict]:
        for h in self.handles:
            h.wait()
        with torch.inference_mode():
            result = engine.context.disagg_pull_commit(
                self.block_ids, self.block_hashes, self.mamba_dst_slot
            )
            if self.snapshots and self.backend is not None:
                # Resolve hashes -> local blocks (now registered by the commit),
                # allocate snapshot slots, READ the peer's snapshots into them by
                # reference, then register hash->block for prefix-cache sub-hits.
                plan = engine.context.disagg_snapshot_pull_plan(self.snapshots)
                if plan is not None:
                    self.backend.begin_pull(self.peer_meta, plan["transfers"]).wait()
                    engine.context.disagg_snapshot_commit(plan["block_ids"], plan["hashes"])
            return result


def _ctx_kv_dims(ctx) -> dict:
    """KV ``memory_buffer`` geometry for the local rank: ``(2, L, total_blocks,
    block_size, heads, hidden)``."""
    mb = ctx.memory_buffer
    return {
        "num_layers": int(mb.shape[1]),
        "total_blocks": int(mb.shape[2]),
        "block_size": int(mb.shape[3]),
        "heads": int(mb.shape[4]),
        "hidden": int(mb.shape[5]),
        "elem": int(mb.element_size()),
    }


def _kv_fragment_triples(src_dims, dst_dims, src_block, dst_block,
                         src_lslice, dst_lslice, src_hslice, dst_hslice):
    """Byte ``(local_off, remote_off, nbytes)`` triples to READ a head/layer
    fragment of one block. A head range is contiguous only per ``(k, layer,
    token)`` in the row-major buffer, so emit one descriptor per ``(k, layer,
    token)`` of ``head_count*hidden`` elements."""
    HD = dst_dims["hidden"]
    elem = dst_dims["elem"]
    head_n = src_hslice.stop - src_hslice.start
    nbytes = head_n * HD * elem
    sL, sTB, sBS, sH = src_dims["num_layers"], src_dims["total_blocks"], src_dims["block_size"], src_dims["heads"]
    dL, dTB, dBS, dH = dst_dims["num_layers"], dst_dims["total_blocks"], dst_dims["block_size"], dst_dims["heads"]
    src_layers = range(src_lslice.start, src_lslice.stop)
    dst_layers = range(dst_lslice.start, dst_lslice.stop)
    triples = []
    for k in (0, 1):
        for sl, dl in zip(src_layers, dst_layers):
            for t in range(sBS):
                soff = ((((k * sL + sl) * sTB + src_block) * sBS + t) * sH + src_hslice.start) * HD * elem
                doff = ((((k * dL + dl) * dTB + dst_block) * dBS + t) * dH + dst_hslice.start) * HD * elem
                triples.append((doff, soff, nbytes))  # (local=dst, remote=src, nbytes)
    return triples


def _mamba_dims_from(conv, ssm) -> dict:
    """Mamba conv/ssm geometry for a local live buffer pair. conv:
    ``(num_mamba_layers, n_slots, conv_dim, d_conv)``; ssm:
    ``(num_mamba_layers, n_slots, nheads, headdim, d_state)``."""
    return {
        "n_slots": int(conv.shape[1]), "conv_dim": int(conv.shape[2]), "d_conv": int(conv.shape[3]),
        "nheads": int(ssm.shape[2]), "headdim": int(ssm.shape[3]), "d_state": int(ssm.shape[4]),
        "elem": int(conv.element_size()),
    }


def _mamba_band_triple(src_dims, dst_dims, is_conv, t, src_slot, dst_slot):
    """One byte ``(local_off, remote_off, nbytes)`` triple for a Mamba reshard
    transfer ``t`` (a conv-channel or ssm-head band of one layer). The band is
    contiguous at fixed ``(layer, slot)``, so it's a single descriptor."""
    elem = dst_dims["elem"]
    if is_conv:
        d_conv = dst_dims["d_conv"]
        s_cd, d_cd = src_dims["conv_dim"], dst_dims["conv_dim"]
        s_ns, d_ns = src_dims["n_slots"], dst_dims["n_slots"]
        nbytes = (t.src_hi - t.src_lo) * d_conv * elem
        soff = ((t.src_layer * s_ns + src_slot) * s_cd + t.src_lo) * d_conv * elem
        doff = ((t.dst_layer * d_ns + dst_slot) * d_cd + t.dst_lo) * d_conv * elem
    else:
        hd, ds = dst_dims["headdim"], dst_dims["d_state"]
        s_nh, d_nh = src_dims["nheads"], dst_dims["nheads"]
        s_ns, d_ns = src_dims["n_slots"], dst_dims["n_slots"]
        nbytes = (t.src_hi - t.src_lo) * hd * ds * elem
        soff = ((t.src_layer * s_ns + src_slot) * s_nh + t.src_lo) * hd * ds * elem
        doff = ((t.dst_layer * d_ns + dst_slot) * d_nh + t.dst_lo) * hd * ds * elem
    return (doff, soff, nbytes)


def post_pull_request_kv(engine, backend, rank_handoffs, my_layout,
                         src_layouts=None, dst_layouts=None,
                         src_mamba_layouts=None, dst_mamba_layouts=None, my_mamba_layout=None):
    """(decode, one-sided) Allocate destination blocks and issue the one-sided
    READ(s) pulling the request's KV into them. Returns a :class:`NixlPullRecv`,
    or ``None`` if the decode KV cache is full. Identity reshard pulls the 1:1
    counterpart's whole blocks (+ Mamba slot/snapshots); hetero reshard (TP remap)
    assembles this rank's head/layer shard from per-source-rank fragments."""
    if not rank_handoffs:
        return None
    block_count = int(rank_handoffs[0]["block_count"])
    identity = True
    if src_layouts:
        identity = (
            src_layouts[0].tp_size == my_layout.tp_size
            and src_layouts[0].pp_size == my_layout.pp_size
        )

    if identity:
        my_key = list(my_layout.kv_shard_key())
        src = next((h for h in rank_handoffs if h.get("shard_key") == my_key), None)
        if src is None:
            raise NotImplementedError(f"pull hand-off: no source shard matching key {my_key}")
        want_mamba = int(src.get("mamba_src_slot", -1)) >= 0
        alloc = engine.context.disagg_pull_alloc(block_count, want_mamba=want_mamba)
        if alloc is None:
            return None
        dst_block_ids = alloc["block_ids"]
        mamba_dst_slot = alloc["mamba_dst_slot"]
        transfers = [("kv", s, d) for s, d in zip(src["block_ids"], dst_block_ids)]
        if want_mamba and mamba_dst_slot >= 0:
            ms = int(src["mamba_src_slot"])
            transfers.append(("mamba_conv", ms, mamba_dst_slot))
            transfers.append(("mamba_ssm", ms, mamba_dst_slot))
        handle = backend.begin_pull(src["region_meta"], transfers)
        return NixlPullRecv(
            handles=[handle], block_ids=dst_block_ids,
            block_hashes=list(src.get("block_hashes") or []), mamba_dst_slot=mamba_dst_slot,
            backend=backend, peer_meta=src["region_meta"],
            snapshots=list(src.get("snapshots") or []),
        )

    # --- hetero (TP-remap) fragment pull ---
    if not src_layouts or not dst_layouts:
        raise NotImplementedError("hetero pull requires src/dst layouts")
    want_mamba = any(int(h.get("mamba_src_slot", -1)) >= 0 for h in rank_handoffs)
    if want_mamba and (not src_mamba_layouts or not dst_mamba_layouts or my_mamba_layout is None):
        raise NotImplementedError("hetero Mamba pull requires Mamba shard layouts")
    alloc = engine.context.disagg_pull_alloc(block_count, want_mamba=want_mamba)
    if alloc is None:
        return None
    dst_block_ids = alloc["block_ids"]
    mamba_dst_slot = alloc["mamba_dst_slot"]
    dst_dims = _ctx_kv_dims(engine.context)
    by_rank = {int(h["global_rank"]): h for h in rank_handoffs}
    src_by_rank = {l.global_rank: l for l in src_layouts}
    plan = kv_reshard.plan_kv_reshard(src_layouts, dst_layouts)
    handles = []
    # Attention KV: head/layer fragments, one fragment READ per source rank.
    for t in utils.transfers_for_dst(plan, my_layout.global_rank):
        h = by_rank.get(t.src_rank)
        src_layout = src_by_rank.get(t.src_rank)
        if h is None or src_layout is None:
            continue
        sl, sh = t.src_layer_slice(src_layout), t.src_head_slice(src_layout)
        dl, dh = t.dst_layer_slice(my_layout), t.dst_head_slice(my_layout)
        triples = []
        for i in range(block_count):
            triples += _kv_fragment_triples(
                h["kv_dims"], dst_dims, int(h["block_ids"][i]), int(dst_block_ids[i]),
                sl, dl, sh, dh,
            )
        if triples:
            handles.append(backend.begin_pull_raw(h["region_meta"], "kv", triples))

    # Mamba state: conv-channel / ssm-head bands from each source rank's hold-ring.
    if want_mamba and mamba_dst_slot >= 0:
        ctx = engine.context
        m_dst = _mamba_dims_from(ctx.mamba_conv_states, ctx.mamba_ssm_states)
        plan_m = mamba_reshard.plan_mamba_reshard(src_mamba_layouts, dst_mamba_layouts)
        banded: dict = {}  # (region_name, src_rank) -> [triple, ...]
        for t in utils.transfers_for_dst(plan_m, my_mamba_layout.global_rank):
            h = by_rank.get(t.src_rank)
            if h is None or h.get("mamba_dims") is None:
                continue
            region = "mamba_conv" if t.is_conv else "mamba_ssm"
            triple = _mamba_band_triple(
                h["mamba_dims"], m_dst, t.is_conv, t, int(h["mamba_src_slot"]), mamba_dst_slot
            )
            banded.setdefault((region, t.src_rank), []).append(triple)
        for (region, src_rank), triples in banded.items():
            handles.append(backend.begin_pull_raw(by_rank[src_rank]["region_meta"], region, triples))

    return NixlPullRecv(
        handles=handles, block_ids=dst_block_ids,
        block_hashes=list(rank_handoffs[0].get("block_hashes") or []),
        mamba_dst_slot=mamba_dst_slot, backend=backend, peer_meta=None, snapshots=[],
    )
