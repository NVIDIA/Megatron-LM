# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""KV transfer backend interface + the active-backend factory."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class TransferHandle:
    """Handle for an in-flight non-blocking transfer; ``wait()`` blocks until it
    completes and returns the received tensor (for receives)."""

    wait_fn: object  # Callable[[], None]
    tensor: Optional[torch.Tensor] = None

    def wait(self) -> Optional[torch.Tensor]:
        if self.wait_fn is not None:
            self.wait_fn()
        return self.tensor


@dataclass
class PullRegion:
    """A paged buffer registered for one-sided remote READ, whose entries (KV
    blocks, Mamba slots) are addressed by index along ``index_axis``.

    Entry ``i``'s bytes are ``num_outer`` slices (product of dims before
    ``index_axis``), each ``inner_bytes`` long (product of dims after), spaced
    ``outer_stride_bytes`` apart -- so slice ``o`` lives at
    ``base_addr + o*outer_stride_bytes + i*inner_bytes``. This is the stride math
    the pull backend uses to READ entries without a staging copy."""

    tensor: torch.Tensor
    index_axis: int

    def layout(self) -> dict:
        """Per-region layout (plain ints) a remote peer uses to compute
        addresses; crosses the control plane, so no tensors/dtypes."""
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
    """Interface for moving KV-cache blobs between workers. Two families,
    distinguished by :attr:`is_pull`:

    * **Push** (two-sided: NCCL). Both peers post matched ``send``/``recv``;
      transfers on a ``(src, dst)`` pair match by POST-ORDER, so both sides must
      enumerate them in the same order (``tag`` is not used for matching).
    * **One-sided** (RDMA: NIXL). Each rank registers its buffers once
      (:meth:`register_regions` / :meth:`export_regions_meta`); one rank then
      moves entries with no peer action, either direction -- :meth:`begin_pull`
      (remote READ) or :meth:`begin_push` (remote WRITE). No staging copy, no
      per-request registration.

    A backend implements one family and leaves the other raising
    ``NotImplementedError``; callers branch on :attr:`is_pull`.
    """

    # True for one-sided (pull/push) backends, False for two-sided send/recv.
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
        """(push) Non-blocking send to ``dst``; ``wait()`` drains it and the
        caller must keep ``tensor`` alive until then."""
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
        """(push) Non-blocking receive from ``src``; ``wait()`` returns the
        received tensor."""
        raise NotImplementedError(f"{type(self).__name__} does not implement the push interface")

    # --- one-sided family (RDMA, either direction) ------------------------
    def register_regions(self, regions: dict) -> None:
        """(one-sided) Register this rank's KV buffers once for remote READ/WRITE.
        ``regions`` maps a name to a :class:`PullRegion`."""
        raise NotImplementedError(f"{type(self).__name__} does not implement the one-sided interface")

    def export_regions_meta(self) -> dict:
        """(one-sided) Metadata a remote peer needs to READ/WRITE this rank's
        regions (agent metadata + per-region layout). Exported once."""
        raise NotImplementedError(f"{type(self).__name__} does not implement the one-sided interface")

    def begin_pull(self, peer_meta: dict, transfers: list):
        """(one-sided) Remote READ: copy entries from a peer's regions into this
        rank's. ``transfers``: ``(region_name, peer_src_index, local_dst_index)``.
        Returns a pollable handle."""
        raise NotImplementedError(f"{type(self).__name__} does not implement the one-sided interface")

    def begin_push(self, peer_meta: dict, transfers: list):
        """(one-sided) Remote WRITE: the mirror of :meth:`begin_pull`, copying
        this rank's entries into a peer's. ``transfers``:
        ``(region_name, local_src_index, peer_dst_index)``. Returns a handle."""
        raise NotImplementedError(f"{type(self).__name__} does not implement the one-sided interface")

    def batch(self, sends, recvs, *, device: Optional[torch.device] = None):
        """Issue one request's point-to-point ops as a group, returning
        ``(handle, recv_buffers)``. ``sends``: ``(tensor, dst)``; ``recvs``:
        ``(shape, dtype, src)`` with buffers allocated here and returned in order.
        The default runs them sequentially via :meth:`send`/:meth:`recv`; NCCL
        overrides this with a grouped primitive to avoid the concurrent-P2P hazard.
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
    """Build a KV transport backend by explicit name: ``"nccl"`` (two-sided push)
    or ``"nixl"`` (one-sided pull). Lazy imports avoid a base<->backend cycle and
    keep NIXL an optional dependency."""
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
