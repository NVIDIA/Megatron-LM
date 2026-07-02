# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""KV transfer backend interface and the backend factory."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Callable, Optional

import torch


@dataclass
class TransferHandle:
    """Handle for an in-flight non-blocking transfer.

    `wait()` blocks until the transfer completes; received data lands in the
    buffers returned by `KVTransportBackend.batch`.
    """

    wait_fn: Callable[[], None]

    def wait(self) -> None:
        """Block until the transfer completes."""
        self.wait_fn()


@dataclass
class PullRegion:
    """A paged buffer registered for one-sided remote read.

    Entries (KV blocks, Mamba slots) are addressed by index along
    `index_axis`. Entry i consists of `num_outer` slices (product of the dims
    before `index_axis`), each `inner_bytes` long (product of the dims after),
    spaced `outer_stride_bytes` apart: slice o starts at
    `base_addr + o * outer_stride_bytes + i * inner_bytes`. The pull backend
    uses this to read entries directly, without a staging copy.
    """

    tensor: torch.Tensor
    index_axis: int

    def layout(self) -> dict:
        """Layout a remote peer needs to compute addresses in this region.

        Plain ints only; this crosses the control plane.
        """
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
            "device_id": self.tensor.device.index,
        }


class KVTransportBackend(abc.ABC):
    """Interface for moving KV-cache blobs between workers.

    There are two backend families, distinguished by `is_pull`:

    * Push (two-sided, NCCL): both peers post one coalesced group of
      point-to-point ops via `batch`. Transfers on a (src, dst) pair match by
      post order, so both sides must enumerate them identically.
    * Pull (one-sided, NIXL/RDMA): each rank registers its buffers once
      (`register_regions` / `export_regions_meta`); a peer then reads entries
      (`begin_pull`) or raw byte ranges (`begin_pull_raw`) with no action from
      this rank.

    A backend implements one family and leaves the other raising
    NotImplementedError; callers branch on `is_pull`.
    """

    # True for one-sided (pull) backends, False for two-sided push.
    is_pull: bool = False

    @abc.abstractmethod
    def init(self) -> None:
        """One-shot, idempotent init."""

    # --- push family (two-sided) ------------------------------------------
    def batch(self, sends, recvs, *, device: Optional[torch.device] = None):
        """Issue one request's point-to-point ops as a single coalesced group.

        `sends` is a list of (tensor, dst); `recvs` is a list of
        (shape, dtype, src), whose buffers are allocated here and returned in
        order. Returns (handle, recv_buffers). Ops must be grouped: concurrent
        ungrouped P2P ops to the same peer can race in NCCL.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement the push interface")

    # --- pull family (one-sided) ------------------------------------------
    def register_regions(self, regions: dict) -> None:
        """Register this rank's KV buffers once for remote read.

        `regions` maps a name to a PullRegion.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement the pull interface")

    def export_regions_meta(self) -> dict:
        """Metadata a remote peer needs to read this rank's regions
        (agent metadata plus per-region layout). Exported once."""
        raise NotImplementedError(f"{type(self).__name__} does not implement the pull interface")

    def begin_pull(self, peer_meta: dict, transfers: list):
        """Read whole entries from a peer's regions into this rank's.

        `transfers` is a list of (region_name, peer_src_index,
        local_dst_index). Returns a pollable handle.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement the pull interface")

    def begin_pull_raw(self, peer_meta: dict, region_name: str, descriptors: list):
        """Read raw byte fragments from one peer region, for resharded
        hand-offs. `descriptors` is a list of (local_offset_bytes,
        peer_offset_bytes, num_bytes), offsets relative to each side's region
        base. Returns a pollable handle."""
        raise NotImplementedError(f"{type(self).__name__} does not implement the pull interface")


def construct_kv_transport_backend(name: str) -> KVTransportBackend:
    """Build a KV transport backend by name: "nccl" (push) or "nixl" (pull).

    Imports are lazy to avoid a base<->backend cycle and to keep NIXL an
    optional dependency.
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
