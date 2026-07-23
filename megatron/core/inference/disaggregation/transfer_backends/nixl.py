# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct NIXL backend for disaggregated prefill/decode KV transfer.

Each rank registers its paged KV buffer once, exports NIXL peer metadata, and
the decode side pulls source block ranges directly into its local KV blocks.

Backend selection belongs in ``transfer_backends.base`` and is supplied
explicitly by the launcher.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import torch

from megatron.core.inference.disaggregation.kv_reshard import KVShardLayout, plan_kv_reshard
from megatron.core.inference.disaggregation.mamba_reshard import (
    MambaShardLayout,
    plan_mamba_reshard,
)
from megatron.core.inference.disaggregation.transfer_backends.base import compute_buffer_geometry
from megatron.core.inference.disaggregation.utils import transfer_peer_records

logger = logging.getLogger(__name__)

try:
    from nixl._api import nixl_agent  # type: ignore[import-not-found]

    _HAVE_NIXL = True
except ImportError:
    nixl_agent = None  # type: ignore[assignment]
    _HAVE_NIXL = False


# NIXL exposes polling, not a blocking wait. A long stall usually means peer or
# fabric failure, so cap the wait.
_POLL_INTERVAL_S = 0.0005  # 0.5 ms
_POLL_TIMEOUT_S = 30.0


@dataclass
class NixlPullHandle:
    """Pollable handle for one logical pull made of one or more NIXL transfers."""

    agent: Any
    xfers: List[Any]
    contexts: List[str]
    submitted_at: float
    timeout_s: float = _POLL_TIMEOUT_S
    done: bool = False
    error: Optional[str] = None

    def poll(self) -> bool:
        """Return True if every transfer has settled, without blocking."""
        if self.done:
            if self.error is not None:
                raise RuntimeError(self.error)
            return True
        if not self.xfers:
            self.done = True
            return True

        errors: List[str] = []
        pending: List[str] = []
        for xfer, ctx in zip(self.xfers, self.contexts):
            state = self.agent.check_xfer_state(xfer)
            if state == "DONE":
                continue
            if state == "ERR":
                errors.append(ctx)
                continue
            pending.append(f"{ctx}: {state}")

        if not pending:
            self.done = True
            if errors:
                self.error = f"NIXL transfer failed ({', '.join(errors)})"
                raise RuntimeError(self.error)
            return True
        if time.perf_counter() - self.submitted_at > self.timeout_s:
            raise TimeoutError(
                f"NIXL transfer timed out after {self.timeout_s}s; pending={pending}"
            )
        return False

    def wait(self) -> None:
        """Block until the transfer completes; NIXL has no blocking wait, so
        this polls."""
        while not self.poll():
            time.sleep(_POLL_INTERVAL_S)


class NixlTransferBackend:
    """Per-rank NIXL agent owning a registration over the paged KV buffer.

    Per-block transfers are descriptor ranges over that registration. Peer
    metadata is exchanged by the control plane and registered lazily on first
    pull.
    """

    name = "nixl"

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
        if not _HAVE_NIXL:
            raise RuntimeError(
                "NixlTransferBackend requires the nixl Python package. Install the "
                "NIXL runtime and `pip install nixl` before launching "
                "disaggregated workers."
            )
        self.agent_name = agent_name
        self._memory_buffer = memory_buffer

        # Addressing geometry shared with the other backends.
        geometry = compute_buffer_geometry(
            memory_buffer,
            expected_num_blocks,
            backend_name="NixlTransferBackend",
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
        shape = list(memory_buffer.shape)
        self._buf_ptr = geometry.buf_ptr
        self._element_size = geometry.element_size
        self._device_id = geometry.device_id
        self._outer_stride_bytes = geometry.outer_stride_bytes
        self._num_outer = geometry.num_outer
        self._bytes_per_slice = geometry.bytes_per_slice
        self._blocks_axis = geometry.blocks_axis
        self._num_blocks = geometry.num_blocks
        self._heads_per_partition = geometry.heads_per_partition
        self._head_dim = geometry.head_dim
        self._tokens_per_block = geometry.tokens_per_block
        self._layout = geometry.layout
        self._mamba_layout = mamba_layout
        self._mamba_state_kind = mamba_state_kind

        # Configure UCX before agent construction. Avoid TCP for VRAM addresses;
        # operators may override this by setting UCX_TLS before launch.
        os.environ.setdefault("UCX_TLS", "cuda_ipc,cuda_copy,cma,shm,self")
        # Explicit registration makes the UCX memtype cache unnecessary and
        # avoids stale VRAM/host classifications.
        os.environ.setdefault("UCX_MEMTYPE_CACHE", "n")

        self._agent = nixl_agent(agent_name)
        self._reg_handle = self._agent.register_memory(memory_buffer)

        # Base64 keeps NIXL metadata safe for msgpack/json control messages.
        self._agent_metadata = self._agent.get_agent_metadata()

        # Peer agent_name -> id returned by add_remote_agent.
        self._known_peers: Dict[str, Any] = {}

        logger.info(
            "NixlTransferBackend[%s] registered %d-block buffer "
            "(blocks_axis=%d, %d outer-slices/block × %d bytes/slice = "
            "%d bytes/block, device=%d, shape=%s)",
            agent_name,
            self._num_blocks,
            self._blocks_axis,
            self._num_outer,
            self._bytes_per_slice,
            self._num_outer * self._bytes_per_slice,
            self._device_id,
            shape,
        )

    def export_meta(self) -> Dict[str, Any]:
        """Return JSON/msgpack-safe metadata for shipping to a decode peer.

        Layout fields describe the scatter-gather address ranges needed to pull
        source blocks into decode-owned blocks.
        """
        meta = {
            "agent_name": self.agent_name,
            "agent_metadata_b64": base64.b64encode(self._agent_metadata).decode("ascii"),
            "base_addr": self._buf_ptr,
            "outer_stride_bytes": self._outer_stride_bytes,
            "device_id": self._device_id,
            "num_outer": self._num_outer,
            "bytes_per_slice": self._bytes_per_slice,
            "blocks_axis": self._blocks_axis,
            "num_blocks": self._num_blocks,
            "heads_per_partition": self._heads_per_partition,
            "head_dim": self._head_dim,
            "tokens_per_block": self._tokens_per_block,
            "element_size": self._element_size,
        }
        if self._layout is not None:
            layer_start, layer_end = self._layout.layer_range()
            meta.update(
                {
                    "global_rank": self._layout.global_rank,
                    "tp_size": self._layout.tp_size,
                    "tp_rank": self._layout.tp_rank,
                    "pp_size": self._layout.pp_size,
                    "pp_rank": self._layout.pp_rank,
                    "num_layers_global": self._layout.num_layers,
                    "num_kv_heads_global": self._layout.num_heads,
                    "layer_start": layer_start,
                    "layer_end": layer_end,
                }
            )
        if self._mamba_layout is not None:
            meta["mamba_layout"] = asdict(self._mamba_layout)
        return meta

    def _ensure_peer_registered(self, peer_meta: Dict[str, Any]) -> str:
        """Register the peer with NIXL on first use; return its agent id."""
        peer_name = peer_meta["agent_name"]
        existing = self._known_peers.get(peer_name)
        if existing is not None:
            return existing
        metadata_b64 = peer_meta.get("agent_metadata_b64")
        if not metadata_b64:
            raise ValueError(f"peer_meta for {peer_name!r} is missing agent_metadata_b64")
        peer_id = self._agent.add_remote_agent(base64.b64decode(metadata_b64))
        resolved = peer_id if peer_id else peer_name
        self._known_peers[peer_name] = resolved
        logger.info("NixlTransferBackend[%s] registered peer %s", self.agent_name, peer_name)
        return resolved

    def _validate_peer(
        self,
        meta: Dict[str, Any],
        src_block_ids: List[int],
        dst_block_ids: List[int],
        *,
        matched_layout: bool = False,
    ) -> None:
        """Validate block mappings and physical transfer compatibility."""

        if len(src_block_ids) != len(dst_block_ids):
            raise ValueError(
                f"source/destination block_id length mismatch for peer "
                f"{meta.get('agent_name')!r}: {len(src_block_ids)} vs {len(dst_block_ids)}"
            )
        for block in src_block_ids:
            if not 0 <= block < int(meta["num_blocks"]):
                raise ValueError(f"source block {block} is outside pool [0, {meta['num_blocks']})")
        for block in dst_block_ids:
            if not 0 <= block < self._num_blocks:
                raise ValueError(
                    f"destination block {block} is outside pool [0, {self._num_blocks})"
                )

        local = {
            "head_dim": self._head_dim,
            "tokens_per_block": self._tokens_per_block,
            "element_size": self._element_size,
            "num_outer": self._num_outer,
            "bytes_per_slice": self._bytes_per_slice,
            "blocks_axis": self._blocks_axis,
            "heads_per_partition": self._heads_per_partition,
        }
        fields = ["head_dim", "tokens_per_block", "element_size"]
        if matched_layout:
            fields.extend(["num_outer", "bytes_per_slice", "blocks_axis", "heads_per_partition"])
        mismatches = [
            f"{field}: peer={meta.get(field)} local={local[field]}"
            for field in fields
            if meta.get(field) is not None
            and local[field] is not None
            and meta.get(field) != local[field]
        ]
        if mismatches:
            kind = "matched-layout" if matched_layout else "transfer"
            raise ValueError(f"{kind} geometry mismatch: {', '.join(mismatches)}")

    @staticmethod
    def _kv_layout_from_meta(meta: Dict[str, Any]) -> KVShardLayout:
        """Reconstruct a main-planner KV layout from peer wire metadata."""

        keys = (
            "global_rank",
            "tp_size",
            "tp_rank",
            "pp_size",
            "pp_rank",
            "num_layers_global",
            "num_kv_heads_global",
            "layer_start",
            "layer_end",
        )
        missing = [key for key in keys if meta.get(key) is None]
        if missing:
            raise ValueError(f"peer metadata missing KV layout fields: {missing}")
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

    def begin_pull_blocks(
        self, peer_meta: Any, src_block_ids: List[int], dst_block_ids: List[int]
    ) -> NixlPullHandle:
        """Submit a pull and return a handle that can be polled later."""
        if not isinstance(peer_meta, dict) or "pp_metas" not in peer_meta:
            if not src_block_ids and not dst_block_ids:
                return NixlPullHandle(
                    agent=self._agent,
                    xfers=[],
                    contexts=[],
                    submitted_at=time.perf_counter(),
                    done=True,
                )

        xfers: List[Any] = []
        contexts: List[str] = []
        submitted_at = time.perf_counter()
        try:
            if self._mamba_layout is not None:
                state_kind = self._mamba_state_kind
                assert state_kind is not None
                width = (
                    self._mamba_layout.conv_dim_local
                    if state_kind == "conv"
                    else self._mamba_layout.nheads_local
                )
                if (
                    self._heads_per_partition != width
                    or self._num_outer != self._mamba_layout.num_layers
                    or self._blocks_axis != 1
                ):
                    raise ValueError(f"local {state_kind} geometry does not match its Mamba layout")

                sources = []
                peers_by_rank = {}
                for meta, blocks in transfer_peer_records(peer_meta, src_block_ids):
                    raw_layout = meta.get("mamba_layout")
                    if not isinstance(raw_layout, dict):
                        raise ValueError("peer metadata is missing mamba_layout")
                    layout = MambaShardLayout(**raw_layout)
                    self._validate_peer(meta, blocks, dst_block_ids)
                    peer_width = (
                        layout.conv_dim_local if state_kind == "conv" else layout.nheads_local
                    )
                    if (
                        meta.get("heads_per_partition") != peer_width
                        or int(meta["num_outer"]) != layout.num_layers
                        or int(meta["blocks_axis"]) != 1
                    ):
                        raise ValueError(
                            f"peer {state_kind} geometry does not match its Mamba layout"
                        )
                    if layout.global_rank in peers_by_rank:
                        raise ValueError(
                            f"duplicate source global_rank={layout.global_rank} "
                            "in Mamba metadata"
                        )
                    sources.append(layout)
                    peers_by_rank[layout.global_rank] = (meta, blocks)
                if not sources:
                    raise ValueError("Mamba handoff contains no source peer metadata")

                transfers = [
                    transfer
                    for transfer in plan_mamba_reshard(sources, [self._mamba_layout])
                    if transfer.is_conv == (state_kind == "conv")
                ]
                for layer in range(self._mamba_layout.num_layers):
                    intervals = sorted(
                        (transfer.dst_lo, transfer.dst_hi)
                        for transfer in transfers
                        if transfer.dst_layer == layer
                    )
                    if not intervals or intervals[0][0] != 0 or intervals[-1][1] != width:
                        raise ValueError(
                            f"incomplete Mamba {state_kind} coverage for layer {layer}"
                        )
                    if any(a[1] != b[0] for a, b in zip(intervals, intervals[1:])):
                        raise ValueError(
                            f"non-contiguous Mamba {state_kind} coverage for layer {layer}"
                        )

                for transfer in transfers:
                    meta, blocks = peers_by_rank[transfer.src_rank]
                    xfer, ctx = self._begin_transfer(
                        meta,
                        blocks,
                        dst_block_ids,
                        transfer.src_layer,
                        transfer.dst_layer,
                        1,
                        transfer.src_lo,
                        transfer.dst_lo,
                        transfer.src_hi - transfer.src_lo,
                    )
                    xfers.append(xfer)
                    contexts.append(ctx)
            elif self._layout is not None:
                sources = []
                peers_by_rank = {}
                for meta, blocks in transfer_peer_records(peer_meta, src_block_ids):
                    layout = self._kv_layout_from_meta(meta)
                    self._validate_peer(meta, blocks, dst_block_ids)
                    if meta.get("heads_per_partition") != layout.local_num_heads():
                        raise ValueError("peer heads_per_partition does not match its KV layout")
                    if int(meta["num_outer"]) % layout.local_num_layers():
                        raise ValueError("peer num_outer is not divisible by its local layer count")
                    if layout.global_rank in peers_by_rank:
                        raise ValueError(
                            f"duplicate source global_rank={layout.global_rank} in KV metadata"
                        )
                    sources.append(layout)
                    peers_by_rank[layout.global_rank] = (meta, blocks, layout)
                if not sources:
                    raise ValueError("KV handoff contains no source peer metadata")

                local_planes = self._num_outer // self._layout.local_num_layers()
                for transfer in plan_kv_reshard(sources, [self._layout]):
                    meta, blocks, source_layout = peers_by_rank[transfer.src_rank]
                    source_planes = int(meta["num_outer"]) // source_layout.local_num_layers()
                    if source_planes != local_planes:
                        raise ValueError(
                            f"outer-plane mismatch peer={source_planes} local={local_planes}"
                        )

                    src_layers = transfer.src_layer_slice(source_layout)
                    dst_layers = transfer.dst_layer_slice(self._layout)
                    src_heads = transfer.src_head_slice(source_layout)
                    dst_heads = transfer.dst_head_slice(self._layout)
                    layer_count = src_layers.stop - src_layers.start
                    head_count = src_heads.stop - src_heads.start
                    full_heads = (
                        src_heads.start == 0
                        and src_heads.stop == source_layout.local_num_heads()
                        and dst_heads.start == 0
                        and dst_heads.stop == self._layout.local_num_heads()
                        and int(meta["bytes_per_slice"]) == self._bytes_per_slice
                    )
                    full_layers = (
                        src_layers.start == 0
                        and src_layers.stop == source_layout.local_num_layers()
                        and dst_layers.start == 0
                        and dst_layers.stop == self._layout.local_num_layers()
                    )
                    if full_heads and full_layers:
                        xfer, ctx = self._begin_transfer(
                            meta, blocks, dst_block_ids, 0, 0, self._num_outer
                        )
                        xfers.append(xfer)
                        contexts.append(ctx)
                        continue
                    if not full_heads and (int(meta["blocks_axis"]) != 2 or self._blocks_axis != 2):
                        raise NotImplementedError(
                            "KV head resharding requires the [2, L, B, T, H, d] layout"
                        )

                    for plane in range(local_planes):
                        xfer, ctx = self._begin_transfer(
                            meta,
                            blocks,
                            dst_block_ids,
                            plane * source_layout.local_num_layers() + src_layers.start,
                            plane * self._layout.local_num_layers() + dst_layers.start,
                            layer_count,
                            0 if full_heads else src_heads.start,
                            0 if full_heads else dst_heads.start,
                            0 if full_heads else head_count,
                        )
                        xfers.append(xfer)
                        contexts.append(ctx)
            else:
                records = transfer_peer_records(peer_meta, src_block_ids)
                if len(records) != 1:
                    raise ValueError("matched-layout transfer requires exactly one source peer")
                meta, blocks = records[0]
                self._validate_peer(meta, blocks, dst_block_ids, matched_layout=True)
                xfer, ctx = self._begin_transfer(meta, blocks, dst_block_ids, 0, 0, self._num_outer)
                xfers.append(xfer)
                contexts.append(ctx)
        except Exception as exc:
            if xfers:
                cleanup = NixlPullHandle(
                    agent=self._agent, xfers=xfers, contexts=contexts, submitted_at=submitted_at
                )
                try:
                    cleanup.wait()
                except TimeoutError:
                    # Tell the owner not to recycle the destination storage while
                    # an already-submitted transfer may still write to it.
                    setattr(exc, "transfer_destinations_safe", False)
                except Exception:
                    # Transfer errors are reported only after every submitted
                    # transfer has reached a terminal state.
                    pass
            raise
        return NixlPullHandle(
            agent=self._agent, xfers=xfers, contexts=contexts, submitted_at=submitted_at
        )

    def _begin_transfer(
        self,
        peer_meta: Dict[str, Any],
        src_block_ids: List[int],
        dst_block_ids: List[int],
        src_o_start: int,
        dst_o_start: int,
        n_outer: int,
        src_h0: int = 0,
        dst_h0: int = 0,
        n_heads: int = 0,
    ) -> tuple[Any, str]:
        """Submit one full-slice or head-fragment NIXL transfer."""
        pm = peer_meta
        peer_base = pm["base_addr"]
        peer_device_id = pm.get("device_id", 0)
        peer_bps = pm["bytes_per_slice"]
        peer_os = pm["outer_stride_bytes"]
        peer_id = self._ensure_peer_registered(pm)

        bps = self._bytes_per_slice
        local_os = self._outer_stride_bytes

        src_tuples: List[Any] = []
        dst_tuples: List[Any] = []

        if n_heads == 0:
            # One descriptor per block and outer slice.
            for src_b, dst_b in zip(src_block_ids, dst_block_ids):
                for i in range(n_outer):
                    src_o = src_o_start + i
                    dst_o = dst_o_start + i
                    src_tuples.append(
                        (peer_base + src_o * peer_os + src_b * peer_bps, peer_bps, peer_device_id)
                    )
                    dst_tuples.append(
                        (self._buf_ptr + dst_o * local_os + dst_b * bps, bps, self._device_id)
                    )
            ctx = (
                f"matched peer={peer_id} outer[{src_o_start}:+{n_outer}] "
                f"blocks={len(src_block_ids)}"
            )
        else:
            # Head sub-range copy: one descriptor per token.
            assert self._head_dim is not None
            assert self._heads_per_partition is not None
            assert self._tokens_per_block is not None
            d_bytes = self._head_dim * self._element_size
            local_token_stride = self._heads_per_partition * d_bytes
            peer_token_stride = pm["heads_per_partition"] * d_bytes
            T = self._tokens_per_block
            frag_bytes = n_heads * d_bytes
            src_h_off = src_h0 * d_bytes
            dst_h_off = dst_h0 * d_bytes

            for src_b, dst_b in zip(src_block_ids, dst_block_ids):
                for i in range(n_outer):
                    src_o = src_o_start + i
                    dst_o = dst_o_start + i
                    src_slice = peer_base + src_o * peer_os + src_b * peer_bps + src_h_off
                    dst_slice = self._buf_ptr + dst_o * local_os + dst_b * bps + dst_h_off
                    for t in range(T):
                        src_tuples.append(
                            (src_slice + t * peer_token_stride, frag_bytes, peer_device_id)
                        )
                        dst_tuples.append(
                            (dst_slice + t * local_token_stride, frag_bytes, self._device_id)
                        )
            ctx = (
                f"reshard peer={peer_id} outer[{src_o_start}:+{n_outer}] "
                f"heads[{src_h0}:+{n_heads}] blocks={len(src_block_ids)}"
            )

        src_descs = self._agent.get_xfer_descs(src_tuples, mem_type="VRAM")
        dst_descs = self._agent.get_xfer_descs(dst_tuples, mem_type="VRAM")
        # READ pulls remote -> local. Signature is (op, local, remote, peer).
        xfer = self._agent.initialize_xfer("READ", dst_descs, src_descs, peer_id)
        try:
            self._agent.transfer(xfer)
        except Exception as exc:
            # The transport may have accepted the operation before surfacing an
            # error, so its destination cannot be proven safe for immediate reuse.
            setattr(exc, "transfer_destinations_safe", False)
            raise
        return xfer, ctx

    def close(self) -> None:
        """Release the registration and agent."""
        if self._agent is None:
            return
        try:
            self._agent.deregister_memory(self._reg_handle)
        except Exception:  # noqa: BLE001 - shutdown path
            logger.exception("NixlTransferBackend: deregister_memory failed")
        self._agent = None
