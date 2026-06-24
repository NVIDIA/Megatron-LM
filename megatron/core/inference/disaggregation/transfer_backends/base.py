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


class KVTransportBackend(abc.ABC):
    """Backend interface for moving KV-cache blobs between workers.

    PE identity is the backend's choice (NCCL uses the process group's
    rank space; NVSHMEM uses global PE ids — equal in our setup). All ops
    are point-to-point between a ``(src, dst)`` pair; multiple transfers on
    a pair are matched by POST-ORDER (the order they are posted), so the send
    and recv sides must enumerate a pair's transfers in the same order. The
    ``tag`` arg mirrors ``isend``/``irecv`` and may be ignored (NCCL/NVSHMEM
    do; gloo is same-tag FIFO) -- callers must not rely on it for matching.
    """

    @abc.abstractmethod
    def is_initialized(self) -> bool:
        """Whether :meth:`init` has run."""

    @abc.abstractmethod
    def init(self, *, group: Optional[object] = None, **kwargs) -> None:
        """One-shot, idempotent init. ``group`` scopes the collective
        for backends that need it (NCCL); ignored otherwise."""

    @abc.abstractmethod
    def send(self, tensor: torch.Tensor, dst: int, tag: int = 0) -> TransferHandle:
        """Send ``tensor`` to ``dst`` (non-blocking). Returns a handle whose
        ``wait()`` ensures the transfer has drained; the caller must keep
        ``tensor`` alive until then."""

    @abc.abstractmethod
    def recv(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        src: int,
        tag: int = 0,
        *,
        device: Optional[torch.device] = None,
    ) -> TransferHandle:
        """Receive a tensor of the given shape/dtype from ``src`` (non-blocking).
        ``handle.wait()`` returns the received tensor."""

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

    def stream(self) -> Optional[torch.cuda.Stream]:
        """Optional dedicated stream; default ``None`` (use current)."""
        return None


# Module-level singleton. Defaults to NCCL (portable / CI-friendly).
_backend: Optional[KVTransportBackend] = None


def get_kv_transport_backend() -> KVTransportBackend:
    """Return the active backend, constructing the default (NCCL) on
    first call."""
    global _backend
    if _backend is None:
        # Lazy import avoids a base <-> nccl import cycle.
        from megatron.core.inference.disaggregation.transfer_backends.nccl import (
            NcclTransportBackend,
        )

        _backend = NcclTransportBackend()
    return _backend


def set_kv_transport_backend(backend: Optional[KVTransportBackend]) -> None:
    """Override the active backend. ``None`` resets to the NCCL default
    on next access. Used by tests, or to select NVSHMEM at startup."""
    global _backend
    _backend = backend
