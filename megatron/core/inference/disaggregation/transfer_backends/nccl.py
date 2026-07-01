# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""``torch.distributed`` point-to-point (NCCL) KV transfer backend."""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist

from megatron.core.inference.disaggregation.transfer_backends.base import (
    KVTransportBackend,
    TransferHandle,
)


class NcclTransportBackend(KVTransportBackend):
    """``torch.distributed`` point-to-point transport via ``isend``/``irecv`` over
    the NCCL backend. The receive side allocates the destination buffer."""

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

    def batch(self, sends, recvs, *, device: Optional[torch.device] = None):
        """Issue one request's point-to-point ops as a single coalesced NCCL
        group (``batch_isend_irecv`` -> one ``ncclGroupStart/End``), returning
        ``(handle, recv_buffers)``. Grouping is required for correctness: dozens
        of concurrent ungrouped P2P ops to one peer can corrupt memory. ``sends``:
        ``(tensor, dst)``; ``recvs``: ``(shape, dtype, src)`` with buffers
        allocated here and returned in order.
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
