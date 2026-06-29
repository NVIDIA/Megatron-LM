# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""NIXL (one-sided RDMA) KV transfer backend for disaggregated inference.

Unlike the NCCL backend (two-sided ``isend``/``irecv``, post-order matched),
NIXL is **pull**-based: the prefill rank registers its staged KV with a local
NIXL agent and exports descriptors; the decode rank ``READ``s those bytes
directly into its own buffers. The control plane (our in-job coordinator, or
Dynamo's frontend) only relays the opaque ``handoff_meta`` from the prefill's
:meth:`publish_descs` to the decode's :meth:`begin_read` -- it never touches
the KV bytes.

This backend is optional: it imports ``nixl`` lazily and :func:`is_available`
reports whether the container provides it, so the disaggregation package does
not hard-depend on NIXL. Selected via ``MEGATRON_KV_TRANSFER_BACKEND=nixl`` (or
``auto`` when NIXL is importable).
"""

from __future__ import annotations

import base64
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from megatron.core.inference.disaggregation.transfer_backends.base import (
    KVTransportBackend,
    TransferHandle,
)

# UCX transports/config NIXL needs for intra-node GPU<->GPU transfer. The
# container often ships ``UCX_TLS=tcp`` (host-only) and leaves the memtype
# cache on, which makes UCX treat CUDA pointers as host memory and stall. We
# set sane defaults *before the agent is created* unless the operator already
# pinned them. cuda_ipc/cuda_copy carry GPU memory; tcp/sm/self are needed for
# the agent's own intra-agent setup; the memtype cache must be off so UCX
# re-queries pointer types (torch may have allocated GPU memory before UCX's
# hooks installed).
_UCX_DEFAULTS = {
    "UCX_TLS": "cuda_ipc,cuda_copy,tcp,sm,self",
    "UCX_MEMTYPE_CACHE": "n",
}

_POLL_INTERVAL_S = 0.0005


def is_available() -> bool:
    """Whether ``nixl`` (and torch under it) can be imported in this container."""
    try:
        import nixl._api  # noqa: F401

        return True
    except Exception:
        return False


def _apply_ucx_defaults() -> None:
    # Memtype cache off (safe default; respect an explicit operator setting).
    os.environ.setdefault("UCX_MEMTYPE_CACHE", _UCX_DEFAULTS["UCX_MEMTYPE_CACHE"])
    # UCX_TLS must include CUDA transports for GPU KV. Containers frequently
    # pin ``UCX_TLS=tcp`` (host-only), which silently degrades NIXL to host
    # memory and stalls; force the GPU-capable set when CUDA transports are
    # absent, but leave an already-CUDA-capable value untouched.
    tls = os.environ.get("UCX_TLS")
    if not tls or "cuda" not in tls.lower():
        os.environ["UCX_TLS"] = _UCX_DEFAULTS["UCX_TLS"]


class NixlPullHandle:
    """Pollable handle for one decode-side READ (one or more NIXL transfers)."""

    def __init__(self, agent: Any, xfers: List[Any], keepalive: List[Any]):
        self._agent = agent
        self._xfers = xfers
        self._keepalive = keepalive  # tensors/regs kept alive until the read drains

    def poll(self) -> bool:
        done = True
        for x in self._xfers:
            st = self._agent.check_xfer_state(x)
            if st == "ERR":
                raise RuntimeError("NIXL transfer entered ERR state")
            done = done and (st == "DONE")
        return done

    def wait(self, timeout_s: float = 30.0) -> None:
        deadline = time.monotonic() + timeout_s
        while not self.poll():
            if time.monotonic() > deadline:
                raise TimeoutError("NIXL transfer did not complete within timeout")
            time.sleep(_POLL_INTERVAL_S)
        # keepalive can drop now; transfers are done.
        self._keepalive.clear()


class NixlTransportBackend(KVTransportBackend):
    """Per-rank NIXL agent over the disaggregation KV buffers.

    Pull-based, so it implements :meth:`publish_descs` (prefill) and
    :meth:`begin_read` (decode) rather than ``send``/``recv``. Worker-level
    agent metadata rides in each request's ``handoff_meta`` and peers are
    registered lazily on first read (cheap; metadata is exchanged once per
    (prefill, decode) pair).
    """

    is_pull = True

    def __init__(self, agent_name: Optional[str] = None) -> None:
        self._agent = None
        self._agent_name = agent_name
        self._agent_metadata_b64: Optional[str] = None
        self._known_peers: Dict[str, str] = {}
        self._init = False

    def is_initialized(self) -> bool:
        return self._init

    def init(self, *, group: Optional[object] = None, **kwargs) -> None:
        if self._init:
            return
        _apply_ucx_defaults()
        from nixl._api import nixl_agent, nixl_agent_config

        if self._agent_name is None:
            import torch.distributed as dist

            rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
            self._agent_name = f"megatron-disagg-rank{rank}"
        self._agent = nixl_agent(self._agent_name, nixl_agent_config(backends=["UCX"]))
        self._agent_metadata_b64 = base64.b64encode(
            self._agent.get_agent_metadata()
        ).decode("ascii")
        self._init = True

    # --- push interface: unsupported (this is a pull backend) -------------
    def send(self, *a, **k):
        raise NotImplementedError(
            "NixlTransportBackend is pull-based; use publish_descs()/begin_read()."
        )

    def recv(self, *a, **k):
        raise NotImplementedError(
            "NixlTransportBackend is pull-based; use publish_descs()/begin_read()."
        )

    # --- prefill side ------------------------------------------------------
    def publish_descs(self, tensors: List[torch.Tensor]) -> Tuple[dict, list]:
        """Register ``tensors`` and return ``(handoff_meta, keepalive)``.

        ``handoff_meta`` is JSON/msgpack-safe (this rank's agent metadata + one
        descriptor per tensor: ``(addr, nbytes, device_id)`` plus shape/dtype so
        the decode side can build a matching local buffer). ``keepalive`` holds
        the registrations; the caller must keep it alive until the decode peer
        has finished reading (the registration must outlive the remote READ).
        """
        assert self._init, "NixlTransportBackend.init() not called"
        keepalive: list = []
        descs: List[dict] = []
        for t in tensors:
            t = t.contiguous()
            reg = self._agent.register_memory(t)
            keepalive.append((t, reg))
            descs.append(
                {
                    "addr": t.data_ptr(),
                    "nbytes": t.element_size() * t.numel(),
                    "device_id": t.device.index or 0,
                    "shape": list(t.shape),
                    "dtype": str(t.dtype),
                }
            )
        # Re-export AFTER registering: NIXL agent metadata advertises this
        # rank's registered memory regions, and the decode peer can only READ
        # regions present in the metadata it imported. Our staging tensors are
        # allocated per request, so the metadata exported at init() predates
        # them -- re-export here so the fresh registrations are reachable.
        meta = {
            "agent_name": self._agent_name,
            "agent_metadata_b64": base64.b64encode(
                self._agent.get_agent_metadata()
            ).decode("ascii"),
            "descs": descs,
        }
        return meta, keepalive

    # --- decode side -------------------------------------------------------
    def begin_read(
        self,
        handoff_meta: dict,
        local_tensors: List[torch.Tensor],
    ) -> NixlPullHandle:
        """Issue one-sided READs pulling the prefill's published descriptors
        into ``local_tensors`` (same count/sizes, in order). Returns a pollable
        handle; the caller waits it before consuming ``local_tensors``."""
        assert self._init, "NixlTransportBackend.init() not called"
        peer_name = self._ensure_peer(handoff_meta)
        src = handoff_meta["descs"]
        assert len(src) == len(local_tensors), (
            f"NIXL begin_read: {len(src)} published descs vs {len(local_tensors)} "
            "local tensors"
        )
        keepalive: list = []
        xfers: list = []
        src_tuples, dst_tuples = [], []
        for sd, lt in zip(src, local_tensors):
            lt = lt.contiguous()
            reg = self._agent.register_memory(lt)
            keepalive.append((lt, reg))
            nbytes = lt.element_size() * lt.numel()
            assert nbytes == sd["nbytes"], (
                f"NIXL begin_read size mismatch: local {nbytes} vs remote {sd['nbytes']}"
            )
            src_tuples.append((sd["addr"], sd["nbytes"], sd["device_id"]))
            dst_tuples.append((lt.data_ptr(), nbytes, lt.device.index or 0))
        src_descs = self._agent.get_xfer_descs(src_tuples, mem_type="VRAM")
        dst_descs = self._agent.get_xfer_descs(dst_tuples, mem_type="VRAM")
        xfer = self._agent.initialize_xfer("READ", dst_descs, src_descs, peer_name)
        self._agent.transfer(xfer)
        xfers.append(xfer)
        return NixlPullHandle(self._agent, xfers, keepalive)

    def _ensure_peer(self, handoff_meta: dict) -> str:
        # Re-import the peer's metadata each request: the prefill re-exports it
        # after registering the request's staging tensors, so the advertised
        # memory regions change request to request. add_remote_agent updates the
        # peer's known regions; without the refresh the READ targets a region
        # NIXL hasn't seen and fails with "no potential backend".
        name = handoff_meta["agent_name"]
        meta_b64 = handoff_meta["agent_metadata_b64"]
        peer_id = self._agent.add_remote_agent(base64.b64decode(meta_b64))
        resolved = peer_id if peer_id else name
        self._known_peers[name] = resolved
        return resolved
