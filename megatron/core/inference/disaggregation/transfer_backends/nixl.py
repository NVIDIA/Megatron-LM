# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""NIXL (one-sided RDMA) pull-based KV transfer backend for disaggregated
inference, following the reference NIXL backend.

Each rank registers its paged KV (and Mamba) buffers once; the decode rank then
READs specific entries straight from the prefill's buffer by index, with no
staging copy or per-request registration. ``nixl`` is imported lazily
(:func:`is_available` reports availability); selected via
``--disagg-kv-transport-backend nixl``.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from typing import Any, Dict, List, Optional

from megatron.core.inference.disaggregation.transfer_backends.base import (
    KVTransportBackend,
    PullRegion,
)

logger = logging.getLogger(__name__)

# Recommended UCX env for GPU KV transfer (we only inspect + warn, never set):
#   * UCX_TLS must include a CUDA transport; ``UCX_TLS=tcp`` (host-only) degrades
#     NIXL to host memory and stalls.
#   * UCX_MEMTYPE_CACHE=n so UCX re-queries pointer types instead of
#     misclassifying CUDA pointers as host memory.
_UCX_RECOMMENDED = {
    "UCX_TLS": "cuda_ipc,cuda_copy,tcp,sm,self",
    "UCX_MEMTYPE_CACHE": "n",
}

_POLL_INTERVAL_S = 0.0005
_POLL_TIMEOUT_S = 30.0


def is_available() -> bool:
    """Whether ``nixl`` (and torch under it) can be imported in this container."""
    try:
        import nixl._api  # noqa: F401

        return True
    except Exception:
        return False


def _check_ucx_env() -> None:
    """Warn (never mutate) if the UCX environment looks unfit for GPU KV
    transfer, leaving the decision to the operator."""
    issues = []
    tls = os.environ.get("UCX_TLS")
    if not tls or "cuda" not in tls.lower():
        issues.append(
            f"UCX_TLS={tls!r} has no CUDA transport; GPU KV may degrade to host "
            f"memory and stall. Recommended: UCX_TLS={_UCX_RECOMMENDED['UCX_TLS']}"
        )
    cache = os.environ.get("UCX_MEMTYPE_CACHE")
    if cache is None or cache.lower() not in ("n", "no"):
        issues.append(
            f"UCX_MEMTYPE_CACHE={cache!r}; with the memtype cache on, UCX may "
            f"misclassify CUDA pointers as host memory. Recommended: "
            f"UCX_MEMTYPE_CACHE={_UCX_RECOMMENDED['UCX_MEMTYPE_CACHE']}"
        )
    if issues:
        logger.warning(
            "NIXL KV transport: UCX environment may be unfit for GPU transfer; "
            "set these before launch to avoid stalls:\n  - %s",
            "\n  - ".join(issues),
        )


class NixlPullHandle:
    """Pollable handle for one decode-side pull (one NIXL READ transfer)."""

    def __init__(self, agent: Any, xfer: Any):
        self._agent = agent
        self._xfer = xfer
        self._done = xfer is None

    def poll(self) -> bool:
        if self._done:
            return True
        st = self._agent.check_xfer_state(self._xfer)
        if st == "ERR":
            raise RuntimeError("NIXL transfer entered ERR state")
        self._done = st == "DONE"
        return self._done

    def wait(self, timeout_s: float = _POLL_TIMEOUT_S) -> None:
        deadline = time.monotonic() + timeout_s
        while not self.poll():
            if time.monotonic() > deadline:
                raise TimeoutError("NIXL transfer did not complete within timeout")
            time.sleep(_POLL_INTERVAL_S)


class NixlTransportBackend(KVTransportBackend):
    """Per-rank NIXL agent owning one registration over the paged KV buffers.
    One-sided: register once, then :meth:`begin_pull` (READ) / :meth:`begin_push`
    (WRITE) move entries by index in either direction. The two-sided send/recv
    interface is left unimplemented."""

    is_pull = True

    def __init__(self, agent_name: Optional[str] = None) -> None:
        self._agent = None
        self._agent_name = agent_name
        self._agent_meta_b64: Optional[str] = None
        self._regions: Dict[str, PullRegion] = {}
        self._reg_handles: list = []
        self._known_peers: Dict[str, str] = {}
        self._init = False

    def is_initialized(self) -> bool:
        return self._init

    def init(self, *, group: Optional[object] = None, **kwargs) -> None:
        if self._init:
            return
        _check_ucx_env()
        from nixl._api import nixl_agent, nixl_agent_config

        if self._agent_name is None:
            import torch.distributed as dist

            rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
            self._agent_name = f"megatron-disagg-rank{rank}"
        self._agent = nixl_agent(self._agent_name, nixl_agent_config(backends=["UCX"]))
        self._init = True

    # --- one-sided family: register once, READ or WRITE entries -----------
    def register_regions(self, regions: Dict[str, PullRegion]) -> None:
        """Register each region's buffer with the agent once (for the backend's
        lifetime) and export the agent metadata, so peers load it exactly once."""
        assert self._init, "NixlTransportBackend.init() not called"
        assert not self._regions, "register_regions called more than once"
        for name, region in regions.items():
            self._reg_handles.append(self._agent.register_memory(region.tensor))
            self._regions[name] = region
        self._agent_meta_b64 = base64.b64encode(
            self._agent.get_agent_metadata()
        ).decode("ascii")

    def export_regions_meta(self) -> dict:
        assert self._regions, "register_regions() not called"
        return {
            "agent_name": self._agent_name,
            "agent_metadata_b64": self._agent_meta_b64,
            "regions": {name: r.layout() for name, r in self._regions.items()},
        }

    def begin_pull(self, peer_meta: dict, transfers: list) -> NixlPullHandle:
        """Remote READ peer->local. ``transfers``: ``(region, peer_src, local_dst)``."""
        return self._begin(
            "READ", peer_meta,
            [(region, local_dst, peer_src) for region, peer_src, local_dst in transfers],
        )

    def begin_push(self, peer_meta: dict, transfers: list) -> NixlPullHandle:
        """Remote WRITE local->peer. ``transfers``: ``(region, local_src, peer_dst)``."""
        return self._begin(
            "WRITE", peer_meta,
            [(region, local_src, peer_dst) for region, local_src, peer_dst in transfers],
        )

    def begin_pull_raw(self, peer_meta: dict, region_name: str, triples: list) -> NixlPullHandle:
        """Remote READ of arbitrary byte sub-ranges of one region -- for hetero-TP
        fragment reads (head/layer sub-ranges, not whole entries). ``triples``:
        ``(local_byte_off, remote_byte_off, nbytes)`` relative to each side's
        region base; all batched into one transfer."""
        assert self._init, "NixlTransportBackend.init() not called"
        if not triples:
            return NixlPullHandle(self._agent, None)
        peer = self._ensure_peer(peer_meta)
        ld = self._regions[region_name].layout()
        pr = peer_meta["regions"][region_name]
        local_tuples = [(ld["base_addr"] + lo, nb, ld["device_id"]) for lo, ro, nb in triples]
        remote_tuples = [(pr["base_addr"] + ro, nb, pr["device_id"]) for lo, ro, nb in triples]
        local_descs = self._agent.get_xfer_descs(local_tuples, mem_type="VRAM")
        remote_descs = self._agent.get_xfer_descs(remote_tuples, mem_type="VRAM")
        xfer = self._agent.initialize_xfer("READ", local_descs, remote_descs, peer)
        self._agent.transfer(xfer)
        return NixlPullHandle(self._agent, xfer)

    def _begin(self, op: str, peer_meta: dict, items: list) -> NixlPullHandle:
        """Issue one one-sided transfer (``op`` = READ/WRITE) batching every
        ``(region, local_index, remote_index)`` in ``items``. Identity layout
        only -- peer and local region must share num_outer / inner_bytes."""
        assert self._init, "NixlTransportBackend.init() not called"
        if not items:
            return NixlPullHandle(self._agent, None)
        peer = self._ensure_peer(peer_meta)
        peer_regions = peer_meta["regions"]

        local_tuples: list = []
        remote_tuples: list = []
        for region_name, local_idx, remote_idx in items:
            pr = peer_regions[region_name]
            ld = self._regions[region_name].layout()
            assert pr["num_outer"] == ld["num_outer"] and pr["inner_bytes"] == ld["inner_bytes"], (
                f"NIXL region {region_name!r}: peer/local layout mismatch "
                f"({pr['num_outer']}x{pr['inner_bytes']} vs "
                f"{ld['num_outer']}x{ld['inner_bytes']}); hetero reshard not supported"
            )
            inner = ld["inner_bytes"]
            for o in range(ld["num_outer"]):
                local_tuples.append(
                    (ld["base_addr"] + o * ld["outer_stride_bytes"] + local_idx * inner,
                     inner, ld["device_id"])
                )
                remote_tuples.append(
                    (pr["base_addr"] + o * pr["outer_stride_bytes"] + remote_idx * inner,
                     inner, pr["device_id"])
                )
        local_descs = self._agent.get_xfer_descs(local_tuples, mem_type="VRAM")
        remote_descs = self._agent.get_xfer_descs(remote_tuples, mem_type="VRAM")
        xfer = self._agent.initialize_xfer(op, local_descs, remote_descs, peer)
        self._agent.transfer(xfer)
        return NixlPullHandle(self._agent, xfer)

    def _ensure_peer(self, peer_meta: dict) -> str:
        # Load the peer's metadata exactly once. Registrations are stable, so a
        # given peer always advertises the same regions; loadRemoteMD rejects
        # re-loading an already-known agent (NIXL_ERR_NOT_ALLOWED).
        name = peer_meta["agent_name"]
        cached = self._known_peers.get(name)
        if cached is not None:
            return cached
        peer_id = self._agent.add_remote_agent(
            base64.b64decode(peer_meta["agent_metadata_b64"])
        )
        resolved = peer_id if peer_id else name
        self._known_peers[name] = resolved
        return resolved

    def close(self) -> None:
        if self._agent is None:
            return
        for h in self._reg_handles:
            try:
                self._agent.deregister_memory(h)
            except Exception:
                pass
        self._reg_handles.clear()
        self._regions.clear()
        self._agent = None
        self._init = False
