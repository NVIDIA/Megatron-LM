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


def pull_static_meta(backend, my_layout, kv_dims, mamba_dims=None):
    """(prefill) This rank's request-invariant pull metadata. Built once and
    merged with :func:`pull_request_meta` (``{**static, **request}``) per request,
    so the per-request hand-off carries only block references, not this blob.

    Pipeline: engine gathers it once at registration -> attached to PREFILL_DONE
    -> consumed by the decode's ``post_pull_request_kv``.

    Args:
        backend: this rank's one-sided (NIXL) transport, regions already registered.
        my_layout: this rank's ``KVShardLayout`` (gives the shard identity).
        kv_dims: KV buffer geometry ``{num_layers, total_blocks, block_size, heads,
            hidden, elem}``.
        mamba_dims: Mamba hold-ring geometry, or ``None`` for non-hybrid models.
    """
    return {
        "transport": "nixl",
        "shard_key": list(my_layout.kv_shard_key()),
        "global_rank": int(my_layout.global_rank),
        "region_meta": backend.export_regions_meta(),
        "kv_dims": kv_dims,
        "mamba_dims": mamba_dims,
    }


def pull_request_meta(ref_payload):
    """(prefill) A request's block-level references -- the per-request half of the
    hand-off. Selects only the fields that vary per request, dropping the static
    geometry in ``ref_payload`` (that goes in :func:`pull_static_meta` instead).
    Replicated across the MP group (every rank schedules identically), so one
    rank's copy is authoritative for all; merged onto each rank's
    :func:`pull_static_meta` to form the hand-off.

    Pipeline: engine builds it per request in ``_disagg_publish_kv`` -> attached
    to PREFILL_DONE -> the decode's ``post_pull_request_kv`` READs these blocks
    from the registered ``memory_buffer`` (no copy).

    Args:
        ref_payload: the dict returned by ``context.export_request_kv_ref(request_id)``.

    Returns:
        ``{layout, block_ids, block_hashes, block_count, mamba_src_slot, snapshots}``.
    """
    return {
        "layout": ref_payload["layout"],
        "block_ids": ref_payload["block_ids"],
        "block_hashes": ref_payload["block_hashes"],
        "block_count": ref_payload["block_count"],
        "mamba_src_slot": ref_payload.get("mamba_src_slot", -1),
        "snapshots": ref_payload.get("snapshots", []),
    }


@dataclass
class NixlPullRecv:
    """Decode side: an in-flight one-sided READ of a request's blocks.
    :meth:`finish` waits the read, then commits -- registers block hashes + binds
    the Mamba slot (the KV already landed in the allocated blocks), mirroring
    ``DecodeRecv.finish``. If the prefill published Mamba prefix-cache
    ``snapshots``, a second READ pulls them after the KV commit.

    The Mamba *end-state* is never pinned -- it moves by value through the
    reset-safe hold-ring. The only pin is the **KV block** ref-count pin (held on
    the prefill during the hand-off); the prefill's snapshot slots stay valid for
    the second READ only as a side effect of it, since snapshot eviction is gated
    on the KV block ref-count being 0.

    Attributes:
        handles: pollable transport handles for the in-flight READ(s).
        block_ids: destination KV block ids (reused prefix + newly pulled).
        block_hashes: per-block hashes to register on commit.
        mamba_dst_slot: destination Mamba slot, or -1 if not hybrid.
        backend: the one-sided transport (for the second, snapshot READ).
        peer_meta: the single source's region meta (snapshot READ); None if multi-source.
        snapshots: prefill's Mamba snapshot refs to pull, or empty.
    """

    handles: List[Any]
    block_ids: List[int]
    block_hashes: List[int]
    mamba_dst_slot: int
    backend: Any = None
    peer_meta: Optional[dict] = None
    snapshots: List = field(default_factory=list)

    def poll(self) -> bool:
        """Non-blocking: ``True`` iff every in-flight READ for this request has
        drained. Lets the engine admit completed pulls without blocking the loop
        on a slower one (it just rechecks next step). Empty handles (e.g. a full
        prefix hit) read as done."""
        return all(h.poll() for h in self.handles)

    def finish(self, engine: Any) -> Optional[dict]:
        """Wait the READ(s), commit the KV (register hashes + bind Mamba slot),
        and pull any Mamba snapshots.

        Args:
            engine: the decode engine (uses ``engine.context`` to commit).

        Returns:
            The import result dict from ``disagg_pull_commit``.
        """
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
    """(decode) This rank's live KV ``memory_buffer`` geometry ``{num_layers,
    total_blocks, block_size, heads, hidden, elem}`` -- the *destination*
    (``dst_dims``) side of the byte-offset math in
    :func:`_kv_fragment_descriptors`. The matching *source* geometry arrives over
    the wire as the hand-off's ``kv_dims`` (built by :func:`pull_static_meta`):
    same schema, two sources (local read here vs. relayed from the prefill).

    Args:
        ctx: the decode :class:`DynamicInferenceContext` (reads ``memory_buffer``).
    """
    mb = ctx.memory_buffer
    return {
        "num_layers": int(mb.shape[1]),
        "total_blocks": int(mb.shape[2]),
        "block_size": int(mb.shape[3]),
        "heads": int(mb.shape[4]),
        "hidden": int(mb.shape[5]),
        "elem": int(mb.element_size()),
    }


def _kv_fragment_descriptors(src_dims, dst_dims, src_block, dst_block,
                             src_layer_slice, dst_layer_slice, src_head_slice, dst_head_slice):
    """Byte ``(local_offset, remote_offset, nbytes)`` READ descriptors for one block's
    head/layer fragment, coalesced to the contiguous minimum.

    In the row-major ``(2, num_layers, total_blocks, block_size, heads, hidden)``
    buffer a head range is contiguous across tokens only when it spans *all*
    heads. So when the fragment takes the full head dim on both sides (the
    same-TP case), the whole ``(block_size, heads, hidden)`` block region is one
    contiguous run per ``(kv, layer)`` -> one descriptor. A partial head range (a
    TP remap) is contiguous only per ``(kv, layer, token)`` -> one descriptor
    each. The full-head form reproduces exactly the whole-block stride math, so
    the same-TP case moves blocks at the minimal descriptor count without a
    separate code path. ``kv`` indexes the key (0) / value (1) cache.

    Args:
        src_dims: source (prefill) KV geometry, from the hand-off's ``kv_dims``.
        dst_dims: destination (decode) KV geometry, from :func:`_ctx_kv_dims`.
        src_block: source physical block id.
        dst_block: destination physical block id.
        src_layer_slice / dst_layer_slice: layer range on each side (a ``slice``).
        src_head_slice / dst_head_slice: head range on each side (a ``slice``).

    Returns:
        A list of ``(local_offset, remote_offset, nbytes)`` byte descriptors.
    """
    hidden = dst_dims["hidden"]
    elem = dst_dims["elem"]
    src_num_layers, src_total_blocks, src_block_size, src_heads = (
        src_dims["num_layers"], src_dims["total_blocks"], src_dims["block_size"], src_dims["heads"]
    )
    dst_num_layers, dst_total_blocks, dst_block_size, dst_heads = (
        dst_dims["num_layers"], dst_dims["total_blocks"], dst_dims["block_size"], dst_dims["heads"]
    )
    src_layer_ids = range(src_layer_slice.start, src_layer_slice.stop)
    dst_layer_ids = range(dst_layer_slice.start, dst_layer_slice.stop)
    full_head_span = (
        src_head_slice.start == 0 and src_head_slice.stop == src_heads
        and dst_head_slice.start == 0 and dst_head_slice.stop == dst_heads
        and src_heads == dst_heads
    )
    descriptors = []
    if full_head_span:
        nbytes = src_block_size * src_heads * hidden * elem
        for kv in (0, 1):
            for src_layer, dst_layer in zip(src_layer_ids, dst_layer_ids):
                src_offset = (((kv * src_num_layers + src_layer) * src_total_blocks + src_block) * src_block_size) * src_heads * hidden * elem
                dst_offset = (((kv * dst_num_layers + dst_layer) * dst_total_blocks + dst_block) * dst_block_size) * dst_heads * hidden * elem
                descriptors.append((dst_offset, src_offset, nbytes))  # (local=dst, remote=src, nbytes)
        return descriptors
    head_count = src_head_slice.stop - src_head_slice.start
    nbytes = head_count * hidden * elem
    for kv in (0, 1):
        for src_layer, dst_layer in zip(src_layer_ids, dst_layer_ids):
            for token in range(src_block_size):
                src_offset = ((((kv * src_num_layers + src_layer) * src_total_blocks + src_block) * src_block_size + token) * src_heads + src_head_slice.start) * hidden * elem
                dst_offset = ((((kv * dst_num_layers + dst_layer) * dst_total_blocks + dst_block) * dst_block_size + token) * dst_heads + dst_head_slice.start) * hidden * elem
                descriptors.append((dst_offset, src_offset, nbytes))
    return descriptors


def _mamba_dims_from(conv, ssm) -> dict:
    """Mamba conv/ssm geometry for a local live buffer pair, as
    ``{n_slots, conv_dim, d_conv, nheads, headdim, d_state, elem}``.

    Args:
        conv: conv-state buffer ``(num_mamba_layers, n_slots, conv_dim, d_conv)``.
        ssm: ssm-state buffer ``(num_mamba_layers, n_slots, nheads, headdim, d_state)``.
    """
    return {
        "n_slots": int(conv.shape[1]), "conv_dim": int(conv.shape[2]), "d_conv": int(conv.shape[3]),
        "nheads": int(ssm.shape[2]), "headdim": int(ssm.shape[3]), "d_state": int(ssm.shape[4]),
        "elem": int(conv.element_size()),
    }


def _mamba_band_descriptor(src_dims, dst_dims, is_conv, transfer, src_slot, dst_slot):
    """One byte ``(local_offset, remote_offset, nbytes)`` READ descriptor for a Mamba
    reshard ``transfer`` -- a band of one layer in the conv (channel) or ssm
    (head) buffer. The band is contiguous at fixed ``(layer, slot)``, so it's a
    single descriptor; conv and ssm share the stride math and differ only in the
    indexed dim and the contiguous per-unit size.

    Args:
        src_dims / dst_dims: source / destination Mamba geometry (:func:`_mamba_dims_from`).
        is_conv: True for a conv-state band, False for an ssm-state band.
        transfer: the Mamba reshard transfer (layer + ``[lo, hi)`` band on each side).
        src_slot / dst_slot: source / destination Mamba slot index.

    Returns:
        One ``(local_offset, remote_offset, nbytes)`` byte descriptor.
    """
    if is_conv:
        src_index_dim, dst_index_dim = src_dims["conv_dim"], dst_dims["conv_dim"]
        unit = dst_dims["d_conv"]
    else:
        src_index_dim, dst_index_dim = src_dims["nheads"], dst_dims["nheads"]
        unit = dst_dims["headdim"] * dst_dims["d_state"]
    src_n_slots, dst_n_slots = src_dims["n_slots"], dst_dims["n_slots"]
    unit_bytes = unit * dst_dims["elem"]
    nbytes = (transfer.src_hi - transfer.src_lo) * unit_bytes
    src_offset = ((transfer.src_layer * src_n_slots + src_slot) * src_index_dim + transfer.src_lo) * unit_bytes
    dst_offset = ((transfer.dst_layer * dst_n_slots + dst_slot) * dst_index_dim + transfer.dst_lo) * unit_bytes
    return (dst_offset, src_offset, nbytes)


def post_pull_request_kv(engine, backend, rank_handoffs, my_layout,
                         src_layouts=None, dst_layouts=None,
                         src_mamba_layouts=None, dst_mamba_layouts=None, my_mamba_layout=None):
    """(decode, one-sided) Allocate destination blocks and issue the one-sided
    READ(s) pulling the request's KV into them. Returns a :class:`NixlPullRecv`,
    or ``None`` if the decode KV cache is full.

    One path, driven by ``kv_reshard.plan_kv_reshard``: each decode rank reads its
    ``(layer x head)`` shard as byte fragments from the prefill ranks that hold
    it. ``_kv_fragment_descriptors`` coalesces a full-head fragment to one
    descriptor per ``(kv, layer)``, so the same-TP case moves whole blocks at the
    minimal descriptor count while a TP remap splits per token -- no separate fast
    path. Mamba state is pulled as conv/ssm bands. Mamba prefix-cache snapshots
    are carried only when a single prefill rank holds this request's shard (so the
    second, snapshot READ resolves against one peer); a multi-source TP remap
    skips them.

    Args:
        engine: the decode engine (uses ``engine.context`` to alloc/match blocks).
        backend: this rank's one-sided (NIXL) transport.
        rank_handoffs: the per-prefill-rank hand-offs relayed from PREFILL_DONE
            (each ``{**pull_static_meta, **pull_request_meta}``).
        my_layout: this decode rank's :class:`KVShardLayout`.
        src_layouts / dst_layouts: full prefill / decode KV layout lists (drive
            the reshard plan).
        src_mamba_layouts / dst_mamba_layouts / my_mamba_layout: the Mamba
            analogues, required only for hybrid models.

    Returns:
        A :class:`NixlPullRecv` to complete later, or ``None`` if the decode KV
        cache is full.
    """
    if not rank_handoffs:
        return None
    if not src_layouts or not dst_layouts:
        raise NotImplementedError("pull hand-off requires src/dst layouts")
    block_count = int(rank_handoffs[0]["block_count"])
    want_mamba = any(int(h.get("mamba_src_slot", -1)) >= 0 for h in rank_handoffs)
    if want_mamba and (not src_mamba_layouts or not dst_mamba_layouts or my_mamba_layout is None):
        raise NotImplementedError("Mamba pull requires Mamba shard layouts")
    # Partial transfer: reuse the longest block prefix the decode already has
    # cached (hashes are TP-independent) and pull only blocks
    # [match_len, block_count). dst block table = reused (in order) + new.
    match = engine.context.disagg_pull_match_prefix(rank_handoffs[0].get("block_hashes") or [])
    reused, match_len = match["reused_block_ids"], match["match_len"]
    alloc = engine.context.disagg_pull_alloc(block_count - match_len, want_mamba=want_mamba)
    if alloc is None:
        engine.context.disagg_pull_unmatch(reused)
        return None
    dst_block_ids = list(reused) + list(alloc["block_ids"])
    mamba_dst_slot = alloc["mamba_dst_slot"]
    dst_dims = _ctx_kv_dims(engine.context)
    handoff_by_rank = {int(h["global_rank"]): h for h in rank_handoffs}
    src_layout_by_rank = {layout.global_rank: layout for layout in src_layouts}
    kv_plan = kv_reshard.plan_kv_reshard(src_layouts, dst_layouts)
    handles = []
    source_handoffs: dict = {}  # src_rank -> handoff (the ranks this dst reads from)
    # Attention KV: per source rank, READ its head/layer fragments of the missing
    # suffix [match_len, block_count). Full-head fragments coalesce to whole blocks.
    for transfer in utils.transfers_for_dst(kv_plan, my_layout.global_rank):
        src_handoff = handoff_by_rank.get(transfer.src_rank)
        src_layout = src_layout_by_rank.get(transfer.src_rank)
        if src_handoff is None or src_layout is None:
            continue
        source_handoffs[transfer.src_rank] = src_handoff
        src_layer_slice = transfer.src_layer_slice(src_layout)
        src_head_slice = transfer.src_head_slice(src_layout)
        dst_layer_slice = transfer.dst_layer_slice(my_layout)
        dst_head_slice = transfer.dst_head_slice(my_layout)
        descriptors = []
        for block_idx in range(match_len, block_count):
            descriptors += _kv_fragment_descriptors(
                src_handoff["kv_dims"], dst_dims,
                int(src_handoff["block_ids"][block_idx]), int(dst_block_ids[block_idx]),
                src_layer_slice, dst_layer_slice, src_head_slice, dst_head_slice,
            )
        if descriptors:
            handles.append(backend.begin_pull_raw(src_handoff["region_meta"], "kv", descriptors))

    # Mamba state: conv-channel / ssm-head bands from each source rank's hold-ring.
    if want_mamba and mamba_dst_slot >= 0:
        ctx = engine.context
        dst_mamba_dims = _mamba_dims_from(ctx.mamba_conv_states, ctx.mamba_ssm_states)
        mamba_plan = mamba_reshard.plan_mamba_reshard(src_mamba_layouts, dst_mamba_layouts)
        bands_by_region: dict = {}  # (region_name, src_rank) -> [descriptor, ...]
        for transfer in utils.transfers_for_dst(mamba_plan, my_mamba_layout.global_rank):
            src_handoff = handoff_by_rank.get(transfer.src_rank)
            if src_handoff is None or src_handoff.get("mamba_dims") is None:
                continue
            region = "mamba_conv" if transfer.is_conv else "mamba_ssm"
            descriptor = _mamba_band_descriptor(
                src_handoff["mamba_dims"], dst_mamba_dims, transfer.is_conv, transfer,
                int(src_handoff["mamba_src_slot"]), mamba_dst_slot,
            )
            bands_by_region.setdefault((region, transfer.src_rank), []).append(descriptor)
        for (region, src_rank), descriptors in bands_by_region.items():
            handles.append(
                backend.begin_pull_raw(handoff_by_rank[src_rank]["region_meta"], region, descriptors)
            )

    # Snapshots: only when a single prefill rank holds this request's shard, so
    # the second snapshot READ in finish() resolves against one peer.
    peer_meta, snapshots = None, []
    if len(source_handoffs) == 1:
        single_source = next(iter(source_handoffs.values()))
        peer_meta = single_source["region_meta"]
        snapshots = list(single_source.get("snapshots") or [])
    return NixlPullRecv(
        handles=handles, block_ids=dst_block_ids,
        block_hashes=list(rank_handoffs[0].get("block_hashes") or []),
        mamba_dst_slot=mamba_dst_slot, backend=backend, peer_meta=peer_meta,
        snapshots=snapshots,
    )
