# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Prefill->decode KV transfer, push family: two-sided (NCCL) hand-off.

Both ranks post matched send/recv ops (the coordinator triggers both sides).
The prefill gathers its KV into a staging tensor and ships it; the decode
scatters the received sub-blocks into its paged cache. The one-sided (pull)
family lives in kv_transfer_pull.py; derive_decode_schema lives here because
only the push receive path needs it.

Hybrid (Mamba) models hand off block-boundary snapshots rather than the live
end-state (see the kv_transfer_pull docstring for why). The snapshot count is
not derivable from static config, so PREFILL_DONE carries the snapshot hashes
and the decode sizes its receives from them. Snapshots reshard across
arbitrary TP/PP changes via plan_mamba_reshard, band by band, alongside the
attention KV.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import torch

from megatron.core.inference.disaggregation import kv_reshard, mamba_reshard, utils
from megatron.core.inference.disaggregation.transfer_backends.base import (
    KVTransportBackend,
    TransferHandle,
)
from megatron.core.inference.inference_request import compute_block_hashes_batched


def derive_decode_schema(engine: Any, prompt_token_ids) -> dict:
    """Reconstruct the KV schema (shapes/dtypes, block count) on the decode
    side from the engine's static config and the prompt tokens, so only KV
    tensors cross the wire. Assumes the whole prompt is handed off into a
    uniform KV layout. Raises NotImplementedError for the MLA latent cache
    (unsupported)."""

    ctx = engine.context
    if ctx.cache_mla_latent:
        raise NotImplementedError(
            "disaggregated KV transfer does not support the MLA latent KV cache"
        )

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
    # The decode rebuilds its own staging tensor from its KVShardLayout
    # (local_num_layers/heads), so only the dtype + per-head width are needed here.
    return {
        "block_count": block_count,
        "block_size_tokens": bs,
        "hidden_per_head": int(mb.shape[5]),
        "block_hashes": block_hashes,
        "attn_dtype": mb.dtype,
    }


def snapshot_transfer_shape(t: "mamba_reshard.MambaReshardTransfer", n: int, dims) -> tuple:
    """Shape of the buffer one snapshot reshard transfer moves: n snapshots of
    one local layer's band slice. Conv slices the channel axis (trailing
    d_conv); ssm slices the head axis (trailing headdim x d_state)."""
    width = t.dst_hi - t.dst_lo
    if t.is_conv:
        return (n, width, dims.d_conv)
    return (n, width, dims.headdim, dims.d_state)


@dataclass
class PrefillHandoff:
    """Prefill-side bookkeeping that keeps the staged tensors alive until
    wait() completes."""

    handles: List[TransferHandle]
    keepalive: List[torch.Tensor] = field(default_factory=list)

    def wait(self) -> None:
        """Wait every handle, then drop the keepalive references."""
        for h in self.handles:
            h.wait()
        self.keepalive.clear()


def send_request_kv_resharded(
    my_layout,
    src_layouts: list,
    dst_layouts: list,
    *,
    backend: KVTransportBackend,
    payload: dict,
    src_mamba_layouts: list,
    dst_mamba_layouts: list,
) -> "PrefillHandoff":
    """Reshard this rank's pre-exported KV to the decode layout and ship it.

    `my_layout` is this rank's KVShardLayout; `src_layouts` / `dst_layouts`
    are the full prefill / decode KV layout lists, and the Mamba layout lists
    drive the snapshot reshard for hybrid models. `payload` is the staging
    dict the context exported when the request was staged.
    """

    attn = payload["staging_tensor"]  # [BC, 2, local_layers, BS, local_heads, HD]
    plan = kv_reshard.plan_kv_reshard(src_layouts, dst_layouts)
    mine = utils.transfers_for_src(plan, my_layout.global_rank)
    # All of the request's sends (attention sub-blocks, then Mamba snapshots)
    # go out as one coalesced batch; `keep` holds the staged slices alive
    # until it drains.
    sends: List[tuple] = []  # (tensor, dst)
    keep: List[torch.Tensor] = []
    for t in mine:
        sub = attn[
            :, :, t.src_layer_slice(my_layout), :, t.src_head_slice(my_layout), :
        ].contiguous()
        keep.append(sub)
        sends.append((sub, t.dst_rank))

    snapshots = payload["mamba_snapshots"]
    if snapshots is not None:
        # Snapshot band slices follow the attention sends; the recv side
        # enumerates the same plan, so the post orders match per peer.
        conv = snapshots["conv_states_tensor"]  # (n, layers, conv_dim_local, d_conv)
        ssm = snapshots["ssm_states_tensor"]  # (n, layers, nheads_local, headdim, d_state)
        for t in mamba_reshard.plan_mamba_reshard(src_mamba_layouts, dst_mamba_layouts):
            if t.src_rank != my_layout.global_rank:
                continue
            src = conv if t.is_conv else ssm
            sub = src[:, t.src_layer, t.src_lo : t.src_hi].contiguous()
            keep.append(sub)
            sends.append((sub, t.dst_rank))
    handle, _ = backend.batch(sends, [])
    return PrefillHandoff(handles=[handle], keepalive=keep)


@dataclass
class DecodeRecv:
    """In-flight decode receive: the irecv handle and the staging buffer it
    fills. finish() waits the transfer, assembles the local KV tensor, and
    imports it, so the caller can overlap completion with an engine step."""

    meta: dict
    staging: torch.Tensor
    pending: List[tuple]  # [(KVReshardTransfer, recv_buffer)]
    my_layout: Any
    # Single coalesced handle for the whole request's batched receives.
    handle: Optional[TransferHandle] = None
    # Mamba snapshots (hybrid): hashes, per-band receives, and the staging
    # tensors they assemble into.
    snapshot_hashes: List[int] = field(default_factory=list)
    pending_snap: List[tuple] = field(default_factory=list)  # [(transfer, buf)]
    snapshot_conv: Optional[torch.Tensor] = None
    snapshot_ssm: Optional[torch.Tensor] = None

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
                f"DISAGG_RECV attn shape mismatch: dst={tuple(dst.shape)} "
                f"recv={tuple(sub.shape)} transfer=({t.global_layer_lo}:"
                f"{t.global_layer_hi},{t.global_head_lo}:{t.global_head_hi}) "
                f"src={t.src_rank} dst_rank={t.dst_rank}"
            )
            dst.copy_(sub)
        mamba_snapshots = None
        if self.snapshot_hashes:
            for t, buf in self.pending_snap:
                dst = self.snapshot_conv if t.is_conv else self.snapshot_ssm
                dst[:, t.dst_layer, t.dst_lo : t.dst_hi] = buf
            mamba_snapshots = {
                "block_hashes": list(self.snapshot_hashes),
                "conv_states_tensor": self.snapshot_conv,
                "ssm_states_tensor": self.snapshot_ssm,
            }
        # import_request_kv writes into inference tensors but runs from the
        # engine's message loop, outside the model step's inference_mode;
        # re-enter it so the in-place writes are permitted.
        with torch.inference_mode():
            return engine.context.import_request_kv(
                self.staging, list(self.meta["block_hashes"]), mamba_snapshots=mamba_snapshots
            )


def post_recv_request_kv_resharded(
    engine: Any,
    my_layout,
    src_layouts: list,
    dst_layouts: list,
    prompt_token_ids,
    *,
    backend: KVTransportBackend,
    handoff: Optional[dict],
    src_mamba_layouts: list,
    dst_mamba_layouts: list,
) -> "DecodeRecv":
    """Post the irecvs for every KV sub-block covering this rank's
    (layer x head) rectangle, plus this rank's Mamba snapshot band slices for
    hybrid models (sized from the handoff's snapshot hashes and the reshard
    plan), and return a DecodeRecv to complete later."""

    meta = derive_decode_schema(engine, prompt_token_ids)
    bc = meta["block_count"]
    bs = meta["block_size_tokens"]
    hd = meta["hidden_per_head"]
    dtype = meta["attn_dtype"]
    device = engine.context.memory_buffer.device

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

    # Mirror the send side's single coalesced batch: attention sub-blocks,
    # then Mamba snapshots.
    plan = kv_reshard.plan_kv_reshard(src_layouts, dst_layouts)
    attn_transfers = utils.transfers_for_dst(plan, my_layout.global_rank)
    recvs: List[tuple] = []  # (shape, dtype, src)
    for t in attn_transfers:
        n_lay = t.global_layer_hi - t.global_layer_lo
        n_head = t.global_head_hi - t.global_head_lo
        recvs.append(((bc, 2, n_lay, bs, n_head, hd), dtype, t.src_rank))

    recv = DecodeRecv(meta=meta, staging=staging, pending=[], my_layout=my_layout)
    snapshot_hashes = list((handoff or {}).get("snapshot_hashes") or [])
    snap_transfers: List[Any] = []
    if snapshot_hashes:
        sa = engine.context.mamba_slot_allocator
        n = len(snapshot_hashes)
        my_mamba = next(m for m in dst_mamba_layouts if m.global_rank == my_layout.global_rank)
        # Band slices this rank receives, in plan order (mirrors the sender).
        plan = mamba_reshard.plan_mamba_reshard(src_mamba_layouts, dst_mamba_layouts)
        snap_transfers = [t for t in plan if t.dst_rank == my_layout.global_rank]
        for t in snap_transfers:
            dtype = sa.conv_states.dtype if t.is_conv else sa.ssm_states.dtype
            recvs.append((snapshot_transfer_shape(t, n, my_mamba.dims), dtype, t.src_rank))
        recv.snapshot_hashes = snapshot_hashes
        recv.snapshot_conv = torch.empty(
            (n, my_mamba.num_layers, *sa.conv_states.shape[2:]),
            dtype=sa.conv_states.dtype,
            device=device,
        )
        recv.snapshot_ssm = torch.empty(
            (n, my_mamba.num_layers, *sa.ssm_states.shape[2:]),
            dtype=sa.ssm_states.dtype,
            device=device,
        )

    handle, bufs = backend.batch([], recvs, device=device)
    recv.handle = handle
    n_attn = len(attn_transfers)
    recv.pending = list(zip(attn_transfers, bufs[:n_attn]))
    recv.pending_snap = list(zip(snap_transfers, bufs[n_attn:]))
    return recv
