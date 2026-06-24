# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Prefill->decode KV transfer: stage, move, import."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

import torch

from megatron.core.inference.disaggregation import kv_reshard, mamba_reshard, utils
from megatron.core.inference.disaggregation.transfer_backends.base import (
    KVTransportBackend,
    TransferHandle,
    get_kv_transport_backend,
)
from megatron.core.inference.inference_request import compute_block_hashes_batched

logger = logging.getLogger(__name__)


def derive_decode_schema(engine: Any, prompt_token_ids) -> Optional[dict]:
    """Reconstruct the KV schema on the decode side with no control message.

    Header-free: the metadata (tensor shapes/dtypes, block count, optional
    Mamba dims) is computed locally from the engine's static config + the prompt
    tokens the decode worker already holds, so only KV tensors cross the wire --
    no descriptor header. Returns ``None`` for the MLA latent cache, which isn't
    derivable this way; MLA disaggregation is unsupported.

    Assumes a homogeneous, fresh prefill: ``block_count`` and the snapshot count
    follow directly from the prompt length and block size.
    """

    ctx = engine.context
    if getattr(ctx, "cache_mla_latent", False):
        return None

    bs = int(ctx.block_size_tokens)
    if isinstance(prompt_token_ids, torch.Tensor):
        toks = prompt_token_ids
        prompt_len = int(toks.numel())
    else:
        prompt_len = len(prompt_token_ids)
        toks = torch.tensor(list(prompt_token_ids), dtype=torch.int64)
    block_count = (prompt_len + bs - 1) // bs
    block_hashes = list(compute_block_hashes_batched(toks, bs))

    mb = ctx.memory_buffer  # (2, num_layers, total_blocks, block_size, heads, hidden)
    _, num_layers, _, _, heads, hidden = mb.shape
    meta: dict = {
        "layout": "std_attn_v1",
        "block_count": block_count,
        "block_size_tokens": bs,
        "num_layers": int(num_layers),
        "num_heads_per_partition": int(heads),
        "hidden_per_head": int(hidden),
        "block_hashes": block_hashes,
        "attn_shape": (block_count, 2, int(num_layers), bs, int(heads), int(hidden)),
        "attn_dtype": mb.dtype,
        "has_mamba": False,
        "has_snapshots": False,
        "empty": False,
    }
    if getattr(ctx, "is_hybrid_model", False):
        conv = ctx.mamba_conv_states  # (num_mamba_layers, max_requests, *conv_state)
        ssm = ctx.mamba_ssm_states
        nml = int(conv.shape[0])
        conv_state = tuple(int(x) for x in conv.shape[2:])
        ssm_state = tuple(int(x) for x in ssm.shape[2:])
        meta["has_mamba"] = True
        meta["mamba"] = {
            "num_mamba_layers": nml,
            "conv_shape": (nml, *conv_state),
            "conv_dtype": conv.dtype,
            "ssm_shape": (nml, *ssm_state),
            "ssm_dtype": ssm.dtype,
        }
        n_snap = len(block_hashes)  # one snapshot per complete block
        if getattr(ctx, "mamba_slot_allocator", None) is not None and n_snap > 0:
            meta["has_snapshots"] = True
            meta["snapshots"] = {
                "block_hashes": block_hashes,
                "conv_shape": (n_snap, nml, *conv_state),
                "conv_dtype": conv.dtype,
                "ssm_shape": (n_snap, nml, *ssm_state),
                "ssm_dtype": ssm.dtype,
            }
    return meta


@dataclass
class PrefillHandoff:
    """Bookkeeping the prefill side holds until the transfer drains.

    Keeps the staged tensors alive (so the backend's in-flight sends
    don't reference freed memory) until :meth:`wait` completes.
    """

    handles: List[TransferHandle]
    keepalive: List[torch.Tensor] = field(default_factory=list)

    def wait(self) -> None:
        for h in self.handles:
            h.wait()
        self.keepalive.clear()


# Transfer matching is by POST-ORDER, not by tag. Every transport backend we
# use matches a recv to a send by the order they are posted on a (src, dst)
# pair: NCCL and NVSHMEM ignore the P2P tag entirely, and gloo matches same-tag
# ops FIFO. So correctness rests on a single invariant: the send side and the
# recv side enumerate the transfers for each (src, dst) pair in the SAME order.
# They do -- both iterate the deterministic reshard plan (attention transfers,
# then Mamba), and the coordinator emits each request's SEND_KV/RECV_KV in a
# consistent request order, so concurrent in-flight requests stay ordered too.
# No tags are needed; a new state-bearing layer type just extends the ordered
# enumeration on both sides.


def send_request_kv_resharded(
    engine: Any,
    request_id: int,
    my_layout,
    src_layouts: list,
    dst_layouts: list,
    *,
    backend: Optional[KVTransportBackend] = None,
    group: Optional[object] = None,
    payload: Optional[dict] = None,
    my_mamba_layout=None,
    src_mamba_layouts: Optional[list] = None,
    dst_mamba_layouts: Optional[list] = None,
) -> Optional["PrefillHandoff"]:
    """Hetero-layout prefill send: reshard this rank's KV sub-blocks to
    the decode layout via global-coordinate range intersection.

    ``my_layout`` is this prefill rank's :class:`KVShardLayout`;
    ``src_layouts`` / ``dst_layouts`` are the full prefill / decode attention
    layout lists. For hybrid models the analogous Mamba layouts
    (:class:`MambaShardLayout`) are passed too, and the conv/ssm state is
    resharded alongside the attention KV. Header-free: the decode side derives
    shapes from config + prompt.

    ``payload`` may be a previously exported KV staging dict (from
    ``export_request_kv``); when given it is shipped instead of re-exporting
    from the live context -- used by the coordinator-native path, which
    exports + holds the KV at prefill completion and ships it later on SEND_KV.
    """

    backend = backend or get_kv_transport_backend()
    if payload is None:
        payload = engine.context.export_request_kv(request_id)
    if payload is None:
        raise ValueError(f"send_request_kv_resharded: request {request_id} has no exportable KV")
    if payload.get("mamba_payload") is not None and my_mamba_layout is None:
        raise NotImplementedError(
            "hybrid (Mamba) hetero handoff requires Mamba shard layouts; "
            "the coordinator-native path supplies them."
        )

    attn = payload["staging_tensor"]  # [BC, 2, local_layers, BS, local_heads, HD]
    plan = kv_reshard.plan_kv_reshard(src_layouts, dst_layouts)
    mine = utils.transfers_for_src(plan, my_layout.global_rank)
    # Collect every sub-block this request ships (attention, then Mamba) and
    # issue them as ONE coalesced batch. Posting dozens of separate isends for a
    # single request races on NCCL (un-grouped concurrent P2P -> illegal memory
    # access); batching wraps them in one ncclGroup so the request's transfer is
    # atomic. ``keep`` holds the staged slices alive until the batch drains.
    sends: List[tuple] = []  # (tensor, dst)
    keep: List[torch.Tensor] = []
    for t in mine:
        sub = attn[
            :, :, t.src_layer_slice(my_layout), :, t.src_head_slice(my_layout), :
        ].contiguous()
        keep.append(sub)
        sends.append((sub, t.dst_rank))

    if payload.get("mamba_payload") is not None:
        _send_mamba_resharded(
            payload["mamba_payload"],
            my_mamba_layout,
            src_mamba_layouts,
            dst_mamba_layouts,
            sends,
            keep,
        )
    handle, _ = backend.batch(sends, [])
    return PrefillHandoff(handles=[handle], keepalive=keep)


def _send_mamba_resharded(mp, my_mamba, src_mamba, dst_mamba, sends, keep):
    """Reshard this rank's conv/ssm sub-blocks, appending each ``(tensor, dst)``
    to ``sends`` (enumerated after the attention sends so the recv side, which
    posts its Mamba receives after its attention receives, matches them in the
    same order within the request's coalesced batch)."""

    conv = mp["conv_states_tensor"]  # (local_layers, conv_dim_local, d_conv)
    ssm = mp["ssm_states_tensor"]  # (local_layers, nheads_local, headdim, d_state)
    plan = mamba_reshard.plan_mamba_reshard(src_mamba, dst_mamba)
    for t in utils.transfers_for_src(plan, my_mamba.global_rank):
        if t.is_conv:
            sub = conv[t.src_layer, t.src_lo : t.src_hi, :].contiguous()
        else:
            sub = ssm[t.src_layer, t.src_lo : t.src_hi, :, :].contiguous()
        keep.append(sub)
        sends.append((sub, t.dst_rank))


@dataclass
class DecodeRecv:
    """In-flight decode receive: the irecv handles + the staging buffer they
    fill. :meth:`finish` waits the transfers, assembles the local KV tensor and
    imports it. Lets the caller post the receive and defer completion (so the
    transfer overlaps with an engine step) instead of blocking inline."""

    meta: dict
    staging: torch.Tensor
    pending: List[tuple]  # [(KVReshardTransfer, recv_buffer)]
    my_layout: Any
    # Single coalesced handle for the whole request's batched receives.
    handle: Optional[TransferHandle] = None
    # Mamba (hybrid only): the local conv/ssm buffers + their received sub-blocks.
    mamba_conv: Optional[torch.Tensor] = None
    mamba_ssm: Optional[torch.Tensor] = None
    mamba_pending: Optional[List[tuple]] = None  # [(MambaReshardTransfer, recv_buffer)]
    my_mamba_layout: Any = None

    def finish(self, engine: Any) -> Optional[dict]:
        """Wait the (single, coalesced) receive, assemble the staging
        tensor(s), and import them."""
        if self.handle is not None:
            self.handle.wait()
        for t, sub in self.pending:
            dst = self.staging[
                :, :, t.dst_layer_slice(self.my_layout), :, t.dst_head_slice(self.my_layout), :
            ]
            assert dst.shape == sub.shape, (
                f"DISAGG_RECV attn shape mismatch: dst={tuple(dst.shape)} recv={tuple(sub.shape)} "
                f"transfer=({t.global_layer_lo}:{t.global_layer_hi},{t.global_head_lo}:{t.global_head_hi}) "
                f"src={t.src_rank} dst_rank={t.dst_rank}"
            )
            self.staging[
                :, :, t.dst_layer_slice(self.my_layout), :, t.dst_head_slice(self.my_layout), :
            ] = sub
        m = self.meta
        payload = {
            "layout": "std_attn_v1",
            "block_count": m["block_count"],
            "block_size_tokens": m["block_size_tokens"],
            "num_layers": self.my_layout.local_num_layers(),
            "num_heads_per_partition": self.my_layout.local_num_heads(),
            "hidden_per_head": m["hidden_per_head"],
            "block_hashes": list(m.get("block_hashes") or []),
            "staging_tensor": self.staging,
        }
        if self.mamba_pending is not None:
            for t, sub in self.mamba_pending:
                if t.is_conv:
                    dst = self.mamba_conv[t.dst_layer, t.dst_lo : t.dst_hi, :]
                    assert (
                        0 <= t.dst_layer < self.mamba_conv.shape[0]
                        and t.dst_hi <= self.mamba_conv.shape[1]
                        and dst.shape == sub.shape
                    ), (
                        f"DISAGG_RECV mamba-conv mismatch: dst_layer={t.dst_layer} "
                        f"dst=[{t.dst_lo}:{t.dst_hi}] buf={tuple(self.mamba_conv.shape)} "
                        f"dst_shape={tuple(dst.shape)} recv={tuple(sub.shape)} src={t.src_rank}"
                    )
                    self.mamba_conv[t.dst_layer, t.dst_lo : t.dst_hi, :] = sub
                else:
                    dst = self.mamba_ssm[t.dst_layer, t.dst_lo : t.dst_hi, :, :]
                    assert (
                        0 <= t.dst_layer < self.mamba_ssm.shape[0]
                        and t.dst_hi <= self.mamba_ssm.shape[1]
                        and dst.shape == sub.shape
                    ), (
                        f"DISAGG_RECV mamba-ssm mismatch: dst_layer={t.dst_layer} "
                        f"dst=[{t.dst_lo}:{t.dst_hi}] buf={tuple(self.mamba_ssm.shape)} "
                        f"dst_shape={tuple(dst.shape)} recv={tuple(sub.shape)} src={t.src_rank}"
                    )
                    self.mamba_ssm[t.dst_layer, t.dst_lo : t.dst_hi, :, :] = sub
            ml = self.my_mamba_layout
            payload["layout"] = "hybrid_v1"
            payload["mamba_payload"] = {
                "num_mamba_layers": ml.num_layers,
                "conv_states_shape": [ml.conv_dim_local, ml.d_conv],
                "ssm_states_shape": [ml.nheads_local, ml.headdim, ml.d_state],
                "conv_states_dtype": str(self.mamba_conv.dtype),
                "ssm_states_dtype": str(self.mamba_ssm.dtype),
                "conv_states_tensor": self.mamba_conv,
                "ssm_states_tensor": self.mamba_ssm,
            }
        # import_request_kv writes the received KV into the cache + block
        # bookkeeping (inference tensors), but runs from the engine's message
        # loop (schedule_requests), outside the inference_mode the model step
        # uses -- re-enter it so the in-place writes are permitted.
        with torch.inference_mode():
            return engine.context.import_request_kv(payload)


def post_recv_request_kv_resharded(
    engine: Any,
    my_layout,
    src_layouts: list,
    dst_layouts: list,
    prompt_token_ids,
    *,
    backend: Optional[KVTransportBackend] = None,
    device: Optional[torch.device] = None,
    my_mamba_layout=None,
    src_mamba_layouts: Optional[list] = None,
    dst_mamba_layouts: Optional[list] = None,
) -> Optional["DecodeRecv"]:
    """Hetero-layout decode receive (non-blocking): post the irecv for every
    KV sub-block covering this rank's (layer x head) rectangle and return a
    :class:`DecodeRecv` to complete later. For hybrid models the conv/ssm
    sub-blocks are posted too (Mamba layouts supplied). Header-free (schema from
    config + prompt). Returns ``None`` if there is no KV to receive."""

    backend = backend or get_kv_transport_backend()
    meta = derive_decode_schema(engine, prompt_token_ids)
    if meta is None:
        return None
    if meta["has_mamba"] and my_mamba_layout is None:
        raise NotImplementedError(
            "hybrid (Mamba) hetero handoff requires Mamba shard layouts; "
            "the coordinator-native path supplies them."
        )

    bc = meta["block_count"]
    bs = meta["block_size_tokens"]
    hd = meta["hidden_per_head"]
    dtype = meta["attn_dtype"]
    if device is None:
        mb = getattr(engine.context, "memory_buffer", None)
        device = mb.device if mb is not None else None

    staging = torch.empty(
        bc,
        2,
        my_layout.local_num_layers(),
        bs,
        my_layout.local_num_heads(),
        hd,
        dtype=dtype,
        device=device,
    )

    # Collect every sub-block this request receives (attention, then Mamba) and
    # post them as ONE coalesced batch -- mirrors the send side's single batch
    # so the request's transfer is atomic and ordered (un-grouped concurrent
    # irecvs race on NCCL -> illegal memory access).
    plan = kv_reshard.plan_kv_reshard(src_layouts, dst_layouts)
    attn_transfers = utils.transfers_for_dst(plan, my_layout.global_rank)
    recvs: List[tuple] = []  # (shape, dtype, src)
    for t in attn_transfers:
        n_lay = t.global_layer_hi - t.global_layer_lo
        n_head = t.global_head_hi - t.global_head_lo
        recvs.append(((bc, 2, n_lay, bs, n_head, hd), dtype, t.src_rank))

    recv = DecodeRecv(meta=meta, staging=staging, pending=[], my_layout=my_layout)
    mamba_transfers: List = []
    if meta["has_mamba"]:
        mamba_transfers = _build_mamba_recvs(
            recv, meta, my_mamba_layout, src_mamba_layouts, dst_mamba_layouts, device, recvs
        )

    handle, bufs = backend.batch([], recvs, device=device)
    recv.handle = handle
    n_attn = len(attn_transfers)
    recv.pending = list(zip(attn_transfers, bufs[:n_attn]))
    if meta["has_mamba"]:
        recv.mamba_pending = list(zip(mamba_transfers, bufs[n_attn:]))
    return recv


def _build_mamba_recvs(recv, meta, my_mamba, src_mamba, dst_mamba, device, recvs):
    """Allocate the decode conv/ssm buffers and APPEND this request's Mamba
    recv specs (shape, dtype, src) to ``recvs`` (after the attention recvs, to
    match the send side's order within the coalesced batch). Returns the Mamba
    transfer list, in the same order as the appended specs."""

    ml = my_mamba
    conv_dtype = meta["mamba"]["conv_dtype"]
    ssm_dtype = meta["mamba"]["ssm_dtype"]
    recv.my_mamba_layout = ml
    recv.mamba_conv = torch.zeros(
        ml.num_layers, ml.conv_dim_local, ml.d_conv, dtype=conv_dtype, device=device
    )
    recv.mamba_ssm = torch.zeros(
        ml.num_layers, ml.nheads_local, ml.headdim, ml.d_state, dtype=ssm_dtype, device=device
    )
    plan = mamba_reshard.plan_mamba_reshard(src_mamba, dst_mamba)
    mamba_transfers = list(utils.transfers_for_dst(plan, ml.global_rank))
    for t in mamba_transfers:
        if t.is_conv:
            shape = (t.dst_hi - t.dst_lo, ml.d_conv)
            dt = conv_dtype
        else:
            shape = (t.dst_hi - t.dst_lo, ml.headdim, ml.d_state)
            dt = ssm_dtype
        recvs.append((shape, dt, t.src_rank))
    return mamba_transfers
