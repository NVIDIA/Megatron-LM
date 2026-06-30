# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""KV transfer backend interface + the active-backend factory."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class TransferHandle:
    """Opaque handle for an in-flight non-blocking transfer.

    ``wait()`` blocks until the transfer completes and, for receives,
    returns the received tensor. Backends populate ``tensor`` for
    receives so the caller need not pre-know where the bytes landed.
    """

    wait_fn: object  # Callable[[], None]
    tensor: Optional[torch.Tensor] = None

    def wait(self) -> Optional[torch.Tensor]:
        if self.wait_fn is not None:
            self.wait_fn()
        return self.tensor


@dataclass
class PullRegion:
    """A paged buffer registered for one-sided remote READ, whose entries
    (KV blocks, Mamba slots) are addressed by index along ``index_axis``.

    Given a row-major ``tensor`` and the axis whose index selects an entry, an
    entry's bytes are laid out as ``num_outer`` equal-size contiguous slices (the
    product of the dims *before* ``index_axis``), each ``inner_bytes`` long (the
    product of the dims *after* it), spaced ``outer_stride_bytes`` apart. So for
    entry ``i`` and outer slice ``o`` the address is
    ``base_addr + o*outer_stride_bytes + i*inner_bytes`` for ``inner_bytes``. This
    is the stride math the pull backend uses to READ arbitrary entries without a
    staging copy."""

    tensor: torch.Tensor
    index_axis: int

    def layout(self) -> dict:
        """JSON-safe per-region layout for a remote peer to compute addresses."""
        shape = self.tensor.shape
        elem = self.tensor.element_size()
        num_outer = 1
        for d in shape[: self.index_axis]:
            num_outer *= int(d)
        inner = 1
        for d in shape[self.index_axis + 1 :]:
            inner *= int(d)
        return {
            "base_addr": self.tensor.data_ptr(),
            "num_outer": num_outer,
            "outer_stride_bytes": int(shape[self.index_axis]) * inner * elem,
            "inner_bytes": inner * elem,
            "device_id": self.tensor.device.index or 0,
        }


class KVTransportBackend(abc.ABC):
    """Backend interface for moving KV-cache blobs between workers.

    Two transport families, distinguished by :attr:`is_pull`:

    * **Push** (two-sided: NCCL). Both peers post matched ``send``/``recv`` ops;
      the coordinator triggers both sides. Multiple transfers on a ``(src, dst)``
      pair are matched by POST-ORDER (the order posted), so the send and recv
      sides must enumerate a pair's transfers in the same order. The ``tag`` arg
      mirrors ``isend``/``irecv`` and may be ignored (NCCL does; gloo is same-tag
      FIFO) -- callers must not rely on it for matching.

    * **One-sided** (RDMA: NIXL). Each rank registers its paged KV buffers
      *once* (:meth:`register_regions`) and exports their layout
      (:meth:`export_regions_meta`); a single rank then moves entries with no
      action from the peer, in *either direction*: :meth:`begin_pull` issues a
      remote READ (copy peer entries into local buffers) and :meth:`begin_push`
      issues a remote WRITE (copy local entries into the peer's buffers). No
      staging copy and no per-request registration -- the control plane only
      relays the opaque region meta. The orchestration chooses the direction
      (the disagg flow has the decode pull after PREFILL_DONE). This mirrors the
      reference NIXL backend and vLLM's NIXL connector.

    A backend implements exactly one family and leaves the other raising
    ``NotImplementedError``; callers branch on :attr:`is_pull`.
    """

    # Transport family (see class docstring): True for one-sided backends whose
    # hand-off is publish + remote-initiated transfer (pull/push), False for
    # two-sided send/recv. Callers branch on this.
    is_pull: bool = False

    @abc.abstractmethod
    def is_initialized(self) -> bool:
        """Whether :meth:`init` has run."""

    @abc.abstractmethod
    def init(self, *, group: Optional[object] = None, **kwargs) -> None:
        """One-shot, idempotent init. ``group`` scopes the collective
        for backends that need it (NCCL); ignored otherwise."""

    # --- push family (two-sided) ------------------------------------------
    def send(self, tensor: torch.Tensor, dst: int, tag: int = 0) -> TransferHandle:
        """(push) Send ``tensor`` to ``dst`` (non-blocking). Returns a handle
        whose ``wait()`` ensures the transfer has drained; the caller must keep
        ``tensor`` alive until then."""
        raise NotImplementedError(f"{type(self).__name__} does not implement the push interface")

    def recv(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        src: int,
        tag: int = 0,
        *,
        device: Optional[torch.device] = None,
    ) -> TransferHandle:
        """(push) Receive a tensor of the given shape/dtype from ``src``
        (non-blocking). ``handle.wait()`` returns the received tensor."""
        raise NotImplementedError(f"{type(self).__name__} does not implement the push interface")

    # --- one-sided family (RDMA, either direction) ------------------------
    def register_regions(self, regions: dict) -> None:
        """(one-sided) Register this rank's KV buffers for one-sided remote
        access (READ and WRITE), once, for the backend's lifetime. ``regions``
        maps a name to a :class:`PullRegion` describing a paged buffer whose
        entries (blocks / slots) are addressed by index along one axis."""
        raise NotImplementedError(f"{type(self).__name__} does not implement the one-sided interface")

    def export_regions_meta(self) -> dict:
        """(one-sided) Return the JSON/msgpack-safe metadata a remote peer needs
        to READ from / WRITE to this rank's registered regions (agent metadata +
        per-region base address, per-entry stride, count). Exported once; reused
        per request, so the peer loads it exactly once."""
        raise NotImplementedError(f"{type(self).__name__} does not implement the one-sided interface")

    def begin_pull(self, peer_meta: dict, transfers: list):
        """(one-sided) Remote READ: copy entries FROM a peer's registered regions
        INTO this rank's. ``transfers`` is a list of
        ``(region_name, peer_src_index, local_dst_index)``. Returns a pollable
        handle whose ``wait()`` blocks until the reads drain."""
        raise NotImplementedError(f"{type(self).__name__} does not implement the one-sided interface")

    def begin_push(self, peer_meta: dict, transfers: list):
        """(one-sided) Remote WRITE: copy entries FROM this rank's registered
        regions INTO a peer's. ``transfers`` is a list of
        ``(region_name, local_src_index, peer_dst_index)``. Returns a pollable
        handle whose ``wait()`` blocks until the writes drain. The mirror of
        :meth:`begin_pull`; the orchestration picks whichever direction it wants."""
        raise NotImplementedError(f"{type(self).__name__} does not implement the one-sided interface")

    def batch(self, sends, recvs, *, device: Optional[torch.device] = None):
        """Issue all of one request's point-to-point ops as a single atomic
        group, returning ``(handle, recv_buffers)``.

        ``sends``: list of ``(tensor, dst)``. ``recvs``: list of
        ``(shape, dtype, src)``; the buffers are allocated here and returned in
        order. The default issues them sequentially via :meth:`send`/:meth:`recv`
        (preserving post-order); backends with a native grouped primitive (NCCL)
        override this to avoid the un-grouped-concurrent-P2P hazard.
        """
        handles = []
        for tensor, dst in sends:
            handles.append(self.send(tensor, dst))
        bufs = []
        for shape, dtype, src in recvs:
            h = self.recv(shape, dtype, src, device=device)
            bufs.append(h.tensor)
            handles.append(h)

        def _wait(_hs=handles):
            for h in _hs:
                h.wait()

        return TransferHandle(wait_fn=_wait), bufs


def construct_kv_transport_backend(name: str) -> KVTransportBackend:
    """Build a KV transport backend by explicit name -- ``"nccl"`` (two-sided
    push) or ``"nixl"`` (one-sided pull). The choice is passed down from the
    disaggregation config (no env var, no auto-detection): the caller decides.

    Lazy imports avoid a base <-> backend import cycle and keep the optional
    NIXL dep from being a hard requirement of the disaggregation package.
    """
    if name == "nccl":
        from megatron.core.inference.disaggregation.transfer_backends.nccl import (
            NcclTransportBackend,
        )
        return NcclTransportBackend()
    if name == "nixl":
        from megatron.core.inference.disaggregation.transfer_backends.nixl import (
            NixlTransportBackend,
        )
        return NixlTransportBackend()
    raise ValueError(f"Unknown KV transfer backend {name!r}; expected 'nccl' or 'nixl'")
