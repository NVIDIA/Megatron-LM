# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""torch.distributed point-to-point (NCCL) KV transfer backend."""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist

from megatron.core.inference.disaggregation.transfer_backends.base import (
    KVTransportBackend,
    TransferHandle,
)


class NcclTransportBackend(KVTransportBackend):
    """Point-to-point transport via isend/irecv over the default (NCCL)
    process group. The receive side allocates the destination buffers; send
    tensors must be contiguous."""

    def init(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "NcclTransportBackend.init: torch.distributed is not initialized; "
                "the prefill/decode workers must share a process group."
            )

    def batch(self, sends, recvs, *, device: Optional[torch.device] = None):
        """Issue one request's point-to-point ops as a single coalesced NCCL
        group and return (handle, recv_buffers).

        `sends` is a list of (tensor, dst); `recvs` is a list of
        (shape, dtype, src), whose buffers are allocated here and returned in
        order. Grouping via batch_isend_irecv is required: concurrent
        ungrouped P2P ops to the same peer can corrupt memory.
        """
        ops = []
        for tensor, dst in sends:
            ops.append(dist.P2POp(dist.isend, tensor, dst))
        bufs = []
        for shape, dtype, src in recvs:
            buf = torch.empty(shape, dtype=dtype, device=device)
            bufs.append(buf)
            ops.append(dist.P2POp(dist.irecv, buf, src))
        works = dist.batch_isend_irecv(ops) if ops else []

        def _wait(_works=works):
            for w in _works:
                w.wait()

        return TransferHandle(wait_fn=_wait), bufs
