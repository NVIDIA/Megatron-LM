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
