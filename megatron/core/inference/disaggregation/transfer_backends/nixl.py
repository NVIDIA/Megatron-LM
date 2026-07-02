# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""NIXL (one-sided RDMA) pull-based KV transfer backend.

Each rank registers its paged KV (and Mamba) buffers once; the decode rank
then reads entries straight out of the prefill's buffers by index, with no
staging copy or per-request registration. ``nixl`` is imported lazily;
selected via ``--disagg-kv-transport-backend nixl``.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from typing import Any, Dict, Optional

from megatron.core.inference.disaggregation.transfer_backends.base import (
    KVTransportBackend,
    PullRegion,
)

logger = logging.getLogger(__name__)

# UCX env required for GPU KV transfer (inspected, never set):
# UCX_TLS must include a CUDA transport (host-only UCX segfaults on the first
# GPU hand-off), and UCX_MEMTYPE_CACHE=n so UCX re-queries pointer types
# instead of misclassifying CUDA pointers as host memory.
_UCX_RECOMMENDED = {"UCX_TLS": "cuda_ipc,cuda_copy,tcp,sm,self", "UCX_MEMTYPE_CACHE": "n"}

_POLL_INTERVAL_S = 0.0005


def _check_ucx_env() -> None:
    """Raise on a UCX environment known to crash GPU KV transfer; warn on a
    merely suboptimal one."""
    tls = os.environ.get("UCX_TLS")
    if tls is not None and "cuda" not in tls.lower():
        raise RuntimeError(
            f"NIXL KV transport: UCX_TLS={tls!r} has no CUDA transport, which "
            f"crashes GPU hand-offs. Set UCX_TLS={_UCX_RECOMMENDED['UCX_TLS']} "
            f"(or unset it) before launch."
        )
    cache = os.environ.get("UCX_MEMTYPE_CACHE")
    if cache is None or cache.lower() not in ("n", "no"):
        logger.warning(
            "NIXL KV transport: UCX_MEMTYPE_CACHE=%r; with the memtype cache on, "
            "UCX may misclassify CUDA pointers as host memory. Recommended: "
            "UCX_MEMTYPE_CACHE=%s",
            cache,
            _UCX_RECOMMENDED["UCX_MEMTYPE_CACHE"],
        )


class NixlPullHandle:
    """Pollable handle for one decode-side pull (one NIXL READ transfer)."""

    def __init__(self, agent: Any, xfer: Any):
        self._agent = agent
        self._xfer = xfer
        self._done = xfer is None

    def poll(self) -> bool:
        """Return True if the transfer has settled, without blocking."""
        if self._done:
            return True
        st = self._agent.check_xfer_state(self._xfer)
        if st == "ERR":
            self._release()
            raise RuntimeError("NIXL transfer entered ERR state")
        if st == "DONE":
            self._release()
            self._done = True
        return self._done

    def _release(self) -> None:
        # Free the agent-side transfer handle as soon as the transfer settles;
        # otherwise handles accumulate per request.
        self._agent.release_xfer_handle(self._xfer)
        self._xfer = None

    def wait(self) -> None:
        """Block until the transfer completes.

        Used on backpressure paths that must drain a transfer now; NIXL has
        no blocking wait, so this polls. The steady state polls in-flight
        transfers once per engine step instead and never blocks here.
        """
        while not self.poll():
            time.sleep(_POLL_INTERVAL_S)


class NixlTransportBackend(KVTransportBackend):
    """Per-rank NIXL agent owning one registration over the paged KV buffers.

    One-sided: register once, then read entries by index (`begin_pull`) or raw
    byte fragments (`begin_pull_raw`). The two-sided push interface is left
    unimplemented.
    """

    is_pull = True

    def __init__(self) -> None:
        self._agent = None
        self._agent_name: Optional[str] = None
        self._agent_meta_b64: Optional[str] = None
        self._regions: Dict[str, PullRegion] = {}
        self._layouts: Dict[str, dict] = {}
        self._reg_handles: list = []
        self._known_peers: Dict[str, str] = {}
        self._init = False

    def init(self) -> None:
        if self._init:
            return
        _check_ucx_env()
        import torch.distributed as dist
        from nixl._api import nixl_agent, nixl_agent_config

        rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
        self._agent_name = f"megatron-disagg-rank{rank}"
        self._agent = nixl_agent(self._agent_name, nixl_agent_config(backends=["UCX"]))
        self._init = True

    def register_regions(self, regions: Dict[str, PullRegion]) -> None:
        """Register each region's buffer with the agent, once for the
        backend's lifetime, and export the agent metadata."""
        assert self._init, "NixlTransportBackend.init() not called"
        assert not self._regions, "register_regions called more than once"
        for name, region in regions.items():
            self._reg_handles.append(self._agent.register_memory(region.tensor))
            self._regions[name] = region
            # Layouts are immutable after registration; compute once.
            self._layouts[name] = region.layout()
        self._agent_meta_b64 = base64.b64encode(self._agent.get_agent_metadata()).decode("ascii")

    def export_regions_meta(self) -> dict:
        assert self._regions, "register_regions() not called"
        return {
            "agent_name": self._agent_name,
            "agent_metadata_b64": self._agent_meta_b64,
            "regions": dict(self._layouts),
        }

    def begin_pull(self, peer_meta: dict, transfers: list) -> NixlPullHandle:
        """Read whole entries from a peer. `transfers` is a list of
        (region_name, peer_src_index, local_dst_index)."""
        return self._begin(
            peer_meta, [(region, local_dst, peer_src) for region, peer_src, local_dst in transfers]
        )

    def begin_pull_raw(self, peer_meta: dict, region_name: str, triples: list) -> NixlPullHandle:
        """Read byte sub-ranges of one region, for hetero-TP fragment reads.

        `triples` is a list of (local_byte_off, remote_byte_off, nbytes),
        offsets relative to each side's region base; all batched into one
        transfer.
        """
        assert self._init, "NixlTransportBackend.init() not called"
        if not triples:
            return NixlPullHandle(self._agent, None)
        peer = self._ensure_peer(peer_meta)
        ld = self._layouts[region_name]
        pr = peer_meta["regions"][region_name]
        local_tuples = [(ld["base_addr"] + lo, nb, ld["device_id"]) for lo, ro, nb in triples]
        remote_tuples = [(pr["base_addr"] + ro, nb, pr["device_id"]) for lo, ro, nb in triples]
        local_descs = self._agent.get_xfer_descs(local_tuples, mem_type="VRAM")
        remote_descs = self._agent.get_xfer_descs(remote_tuples, mem_type="VRAM")
        xfer = self._agent.initialize_xfer("READ", local_descs, remote_descs, peer)
        self._agent.transfer(xfer)
        return NixlPullHandle(self._agent, xfer)

    def _begin(self, peer_meta: dict, items: list) -> NixlPullHandle:
        """Issue one remote read batching every (region, local_index,
        remote_index) in `items`. Peer and local region must share
        num_outer / inner_bytes."""
        assert self._init, "NixlTransportBackend.init() not called"
        if not items:
            return NixlPullHandle(self._agent, None)
        peer = self._ensure_peer(peer_meta)
        peer_regions = peer_meta["regions"]

        local_tuples: list = []
        remote_tuples: list = []
        for region_name, local_idx, remote_idx in items:
            pr = peer_regions[region_name]
            ld = self._layouts[region_name]
            assert pr["num_outer"] == ld["num_outer"] and pr["inner_bytes"] == ld["inner_bytes"], (
                f"NIXL region {region_name!r}: peer/local layout mismatch "
                f"({pr['num_outer']}x{pr['inner_bytes']} vs "
                f"{ld['num_outer']}x{ld['inner_bytes']}); hetero reshard not supported"
            )
            inner = ld["inner_bytes"]
            for o in range(ld["num_outer"]):
                local_tuples.append(
                    (
                        ld["base_addr"] + o * ld["outer_stride_bytes"] + local_idx * inner,
                        inner,
                        ld["device_id"],
                    )
                )
                remote_tuples.append(
                    (
                        pr["base_addr"] + o * pr["outer_stride_bytes"] + remote_idx * inner,
                        inner,
                        pr["device_id"],
                    )
                )
        local_descs = self._agent.get_xfer_descs(local_tuples, mem_type="VRAM")
        remote_descs = self._agent.get_xfer_descs(remote_tuples, mem_type="VRAM")
        xfer = self._agent.initialize_xfer("READ", local_descs, remote_descs, peer)
        self._agent.transfer(xfer)
        return NixlPullHandle(self._agent, xfer)

    def _ensure_peer(self, peer_meta: dict) -> str:
        # Load each peer's metadata exactly once: registrations are stable, so
        # a peer always advertises the same regions, and loadRemoteMD rejects
        # re-loading an already-known agent (NIXL_ERR_NOT_ALLOWED).
        name = peer_meta["agent_name"]
        cached = self._known_peers.get(name)
        if cached is not None:
            return cached
        peer_id = self._agent.add_remote_agent(base64.b64decode(peer_meta["agent_metadata_b64"]))
        self._known_peers[name] = peer_id
        return peer_id
