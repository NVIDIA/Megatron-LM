# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Two-sided (NCCL) KV transfer backend for disaggregated prefill/decode.

Unlike the one-sided NIXL backend, both peers participate: the decode posts
receives when the hand-off request arrives (begin_pull_blocks) and the prefill
posts the matching sends when the coordinator's SEND_KV names the decode
instance (begin_push_blocks). Both sides enumerate the same reshard plan in
the same deterministic order, so the point-to-point operations match by post
order per peer pair. Data moves straight out of the prefill's pinned blocks;
there is no staging copy on the send side.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist

from megatron.core.inference.disaggregation.kv_reshard import KVShardLayout, plan_kv_reshard
from megatron.core.inference.disaggregation.mamba_reshard import (
    MambaShardLayout,
    plan_mamba_reshard,
)
from megatron.core.inference.disaggregation.transfer_backends.base import (
    compute_buffer_geometry,
    export_geometry_meta,
)
from megatron.core.inference.disaggregation.utils import transfer_peer_records

logger = logging.getLogger(__name__)


class NcclTransferHandle:
    """Pollable handle for one batched NCCL transfer.

    Receives land in temporary contiguous buffers; on completion the handle
    runs its scatter closures once to place the data into the paged buffers.
    Send handles keep the gathered source slices alive until reaped.
    """

    def __init__(self, works: List[Any], keepalive: List[torch.Tensor], scatters: List[Any]):
        self._works = works
        self._keepalive = keepalive
        self._scatters = scatters
        self._done = not works and not scatters

    def poll(self) -> bool:
        """Return True if the transfer has settled, scattering received data
        into the paged buffers on first completion."""
        if self._done:
            return True
        if not all(w.is_completed() for w in self._works):
            return False
        self._finish()
        return True

    def wait(self) -> None:
        """Block until the transfer completes, then scatter."""
        for w in self._works:
            w.wait()
        self._finish()

    def _finish(self) -> None:
        if self._done:
            return
        with torch.inference_mode():
            for scatter in self._scatters:
                scatter()
        self._keepalive.clear()
        self._works = []
        self._done = True


def _make_copy(view: torch.Tensor, buf: torch.Tensor):
    def _copy():
        view.copy_(buf.view(view.shape))

    return _copy


def _kv_layout_from_meta(meta: Dict[str, Any]) -> KVShardLayout:
    """Rebuild a peer's KVShardLayout from its exported metadata."""
    return KVShardLayout(
        num_layers=int(meta["num_layers_global"]),
        num_heads=int(meta["num_kv_heads_global"]),
        tp_size=int(meta["tp_size"]),
        tp_rank=int(meta["tp_rank"]),
        pp_size=int(meta["pp_size"]),
        pp_rank=int(meta["pp_rank"]),
        global_rank=int(meta["global_rank"]),
        layer_start=int(meta["layer_start"]),
        num_local_layers=int(meta["layer_end"]) - int(meta["layer_start"]),
    )


class NcclTransferBackend:
    """Per-buffer NCCL transport over the default process group.

    Mirrors the NIXL backend's construction and metadata schema so the
    hand-off layer treats the two interchangeably; only the transfer calls
    differ (two-sided matched send/recv instead of one-sided reads).
    """

    name = "nccl"
    is_push = True

    def __init__(
        self,
        agent_name: str,
        memory_buffer: torch.Tensor,
        expected_num_blocks: int,
        tp_size: Optional[int] = None,
        tp_rank: Optional[int] = None,
        num_kv_heads_global: Optional[int] = None,
        heads_per_partition: Optional[int] = None,
        head_dim: Optional[int] = None,
        tokens_per_block: Optional[int] = None,
        global_rank: Optional[int] = None,
        pp_size: Optional[int] = None,
        pp_rank: Optional[int] = None,
        num_layers_global: Optional[int] = None,
        layer_start: Optional[int] = None,
        layer_end: Optional[int] = None,
        mamba_layout: Optional[MambaShardLayout] = None,
        mamba_state_kind: Optional[str] = None,
    ):
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                "NcclTransferBackend requires torch.distributed to be initialized; "
                "the prefill and decode workers must share a process group."
            )
        self.agent_name = agent_name
        self._memory_buffer = memory_buffer
        geometry = compute_buffer_geometry(
            memory_buffer,
            expected_num_blocks,
            backend_name="NcclTransferBackend",
            tp_size=tp_size,
            tp_rank=tp_rank,
            num_kv_heads_global=num_kv_heads_global,
            heads_per_partition=heads_per_partition,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            global_rank=global_rank,
            pp_size=pp_size,
            pp_rank=pp_rank,
            num_layers_global=num_layers_global,
            layer_start=layer_start,
            layer_end=layer_end,
            mamba_layout=mamba_layout,
            mamba_state_kind=mamba_state_kind,
        )
        self._geometry = geometry
        self._layout = geometry.layout
        self._mamba_layout = mamba_layout
        self._mamba_state_kind = mamba_state_kind
        logger.info(
            "NcclTransferBackend[%s] over %d-block buffer (rank=%d, shape=%s)",
            agent_name,
            geometry.num_blocks,
            dist.get_rank(),
            list(memory_buffer.shape),
        )

    def export_meta(self) -> Dict[str, Any]:
        """The shared geometry schema plus this rank's NCCL address."""
        meta = export_geometry_meta(self._geometry, self._mamba_layout)
        meta["transport"] = "nccl"
        meta["nccl_rank"] = dist.get_rank()
        return meta

    # --- shared enumeration -------------------------------------------------
    def _kv_transfers(self, peer_records, mine_is_src: bool):
        """Yield (peer_meta, layers, heads) for this rank's part of the KV
        reshard plan, in deterministic plan order."""
        sources: list = []
        peers_by_rank: dict = {}
        for meta, blocks in peer_records:
            layout = _kv_layout_from_meta(meta)
            if layout.global_rank in peers_by_rank:
                raise ValueError(f"duplicate peer global_rank={layout.global_rank} in KV metadata")
            peers_by_rank[layout.global_rank] = meta
            sources.append(layout)
        if mine_is_src:
            plan = plan_kv_reshard([self._layout], sources)
        else:
            plan = plan_kv_reshard(sources, [self._layout])
        for transfer in plan:
            peer_rank = transfer.dst_rank if mine_is_src else transfer.src_rank
            meta = peers_by_rank[peer_rank]
            if mine_is_src:
                layers = transfer.src_layer_slice(self._layout)
                heads = transfer.src_head_slice(self._layout)
            else:
                layers = transfer.dst_layer_slice(self._layout)
                heads = transfer.dst_head_slice(self._layout)
            yield meta, layers, heads

    def _mamba_transfers(self, peer_records, mine_is_src: bool):
        """Yield (peer_meta, lo, hi) band slices of this rank's Mamba state,
        in deterministic plan order."""
        for meta, _ in peer_records:
            raw_layout = meta.get("mamba_layout")
            if not isinstance(raw_layout, dict):
                raise ValueError("peer metadata is missing mamba_layout")
            peer_layout = MambaShardLayout(**raw_layout)
            if mine_is_src:
                plan = plan_mamba_reshard([self._mamba_layout], [peer_layout])
            else:
                plan = plan_mamba_reshard([peer_layout], [self._mamba_layout])
            for t in plan:
                if t.is_conv != (self._mamba_state_kind == "conv"):
                    continue
                if mine_is_src:
                    yield meta, t.src_layer, t.src_lo, t.src_hi
                else:
                    yield meta, t.dst_layer, t.dst_lo, t.dst_hi

    def _kv_block_view(self, block_id: int, layers: slice, heads: slice) -> torch.Tensor:
        """One block's (kv, layer, token, head, dim) fragment in the
        [2, L, B, T, H, d] paged buffer."""
        return self._memory_buffer[:, layers, block_id, :, heads, :]

    # --- decode side ---------------------------------------------------------
    def begin_pull_blocks(
        self, peer_meta: Any, src_block_ids: List[int], dst_block_ids: List[int]
    ) -> NcclTransferHandle:
        """Post the receives matching the prefill's sends; the handle scatters
        into the destination blocks (or Mamba slots) on completion."""
        if not dst_block_ids:
            return NcclTransferHandle([], [], [])
        records = transfer_peer_records(peer_meta, src_block_ids)

        ops: List[Any] = []
        buffers: List[torch.Tensor] = []
        scatters: List[Any] = []
        device = self._memory_buffer.device
        dtype = self._memory_buffer.dtype

        if self._mamba_layout is not None:
            for meta, layer, lo, hi in self._mamba_transfers(records, mine_is_src=False):
                for slot in dst_block_ids:
                    view = self._memory_buffer[layer, int(slot), lo:hi]
                    buf = torch.empty(view.shape, dtype=dtype, device=device)
                    buffers.append(buf)
                    ops.append(dist.P2POp(dist.irecv, buf, int(meta["nccl_rank"])))
                    scatters.append(_make_copy(view, buf))
        else:
            geo = self._geometry
            for meta, layers, heads in self._kv_transfers(records, mine_is_src=False):
                n_layers = layers.stop - layers.start
                n_heads = heads.stop - heads.start
                for block in dst_block_ids:
                    buf = torch.empty(
                        (2, n_layers, geo.tokens_per_block, n_heads, geo.head_dim),
                        dtype=dtype,
                        device=device,
                    )
                    buffers.append(buf)
                    ops.append(dist.P2POp(dist.irecv, buf, int(meta["nccl_rank"])))
                    scatters.append(_make_copy(self._kv_block_view(int(block), layers, heads), buf))

        works = dist.batch_isend_irecv(ops) if ops else []
        return NcclTransferHandle(works, buffers, scatters)

    # --- prefill side ----------------------------------------------------------
    def begin_push_blocks(self, peer_meta: Any, src_block_ids: List[int]) -> NcclTransferHandle:
        """Post the sends matching the decode's receives, straight out of the
        pinned source blocks (or Mamba slots). `peer_meta` is the decode
        instance's per-rank metadata in the same nested shape as a hand-off's
        kv_meta."""
        if not src_block_ids:
            return NcclTransferHandle([], [], [])
        records = transfer_peer_records(peer_meta, [])

        ops: List[Any] = []
        keep: List[torch.Tensor] = []

        if self._mamba_layout is not None:
            for meta, layer, lo, hi in self._mamba_transfers(records, mine_is_src=True):
                for slot in src_block_ids:
                    sub = self._memory_buffer[layer, int(slot), lo:hi].contiguous()
                    keep.append(sub)
                    ops.append(dist.P2POp(dist.isend, sub, int(meta["nccl_rank"])))
        else:
            for meta, layers, heads in self._kv_transfers(records, mine_is_src=True):
                for block in src_block_ids:
                    sub = self._kv_block_view(int(block), layers, heads).contiguous()
                    keep.append(sub)
                    ops.append(dist.P2POp(dist.isend, sub, int(meta["nccl_rank"])))

        works = dist.batch_isend_irecv(ops) if ops else []
        return NcclTransferHandle(works, keep, [])
