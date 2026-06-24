# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""``torch.distributed`` point-to-point KV transfer backend (NCCL / gloo)."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.distributed as dist

from megatron.core.inference.disaggregation.transfer_backends.base import (
    KVTransportBackend,
    TransferHandle,
)


class NcclTransportBackend(KVTransportBackend):
    """``torch.distributed`` point-to-point transport.

    Uses ``isend`` / ``irecv`` so it works identically under the
    ``nccl`` backend (GPU) and ``gloo`` backend (CPU/CI). Tensors are
    moved on whatever device they live on; the receive side allocates
    the destination buffer.
    """

    def __init__(self, group: Optional[object] = None) -> None:
        self._group = group
        self._init = False

    def is_initialized(self) -> bool:
        return self._init

    def init(self, *, group: Optional[object] = None, **kwargs) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "NcclTransportBackend.init: torch.distributed is not initialized; "
                "the prefill/decode workers must share a process group."
            )
        if group is not None:
            self._group = group
        self._init = True

    def send(self, tensor: torch.Tensor, dst: int, tag: int = 0) -> TransferHandle:
        t = tensor.contiguous()
        work = dist.isend(t, dst=dst, tag=tag, group=self._group)
        # Keep a reference to ``t`` so it isn't freed before the send drains.
        return TransferHandle(wait_fn=lambda _t=t, _w=work: _w.wait())

    def recv(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        src: int,
        tag: int = 0,
        *,
        device: Optional[torch.device] = None,
    ) -> TransferHandle:
        buf = torch.empty(shape, dtype=dtype, device=device)
        work = dist.irecv(buf, src=src, tag=tag, group=self._group)
        return TransferHandle(wait_fn=work.wait, tensor=buf)

    def batch(self, sends, recvs, *, device: Optional[torch.device] = None):
        """Issue many point-to-point ops for one request as a SINGLE coalesced
        NCCL group, returning ``(handle, recv_buffers)``.

        Posting each request's sub-block sends/recvs as separate ``isend`` /
        ``irecv`` calls races on NCCL: a single rank with dozens of concurrent
        ungrouped P2P ops to the same peer can corrupt memory (illegal access)
        because the ops are not issued atomically. ``batch_isend_irecv`` wraps
        them in one ``ncclGroupStart/End`` so the whole request's transfer is
        one atomic, correctly-ordered operation -- and it still overlaps the
        engine step (the returned handle is waited a step later).

        ``sends``: list of ``(tensor, dst)``. ``recvs``: list of
        ``(shape, dtype, src)`` -- buffers are allocated here and returned in
        order so the caller can map them back to its transfers.
        """
        ops = []
        for tensor, dst in sends:
            ops.append(dist.P2POp(dist.isend, tensor.contiguous(), dst, group=self._group))
        bufs = []
        for shape, dtype, src in recvs:
            buf = torch.empty(shape, dtype=dtype, device=device)
            bufs.append(buf)
            ops.append(dist.P2POp(dist.irecv, buf, src, group=self._group))
        works = dist.batch_isend_irecv(ops) if ops else []

        def _wait(_works=works):
            for w in _works:
                w.wait()

        return TransferHandle(wait_fn=_wait), bufs
