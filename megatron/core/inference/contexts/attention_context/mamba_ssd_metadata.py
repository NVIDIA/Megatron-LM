# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Per-step ragged tiling for the CuteDSL varlen SSD kernel.

Everything the CuteDSL backend needs beyond the metadata the Triton path
already publishes lives here. Under the default `mamba_prefill_backend="triton"`,
`MambaSSDMetadata.create` and `MambaSSDBufferLayout.create` both return None and
no buffer, computation, or bookkeeping-buffer space is spent.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from megatron.core.utils import round_up_to_nearest_multiple

# Every SSD array, in bookkeeping-buffer order, mapped to:
#   capacity: which bound sizes the array ("requests" or "chunks"),
#   count_key: the per-step length, as published in the compute_cpu_metadata dict.
_SSD_ARRAYS = {
    "seq_chunk_start": ("requests", "ssd_num_active_seqs"),
    "seq_chunk_count": ("requests", "ssd_num_active_seqs"),
    "seq_chunk_base": ("requests", "ssd_num_active_seqs"),
    "active_seq_idx": ("requests", "ssd_num_active_seqs"),
    "empty_seq_idx": ("requests", "ssd_num_empty_seqs"),
    "chunk_token_base": ("chunks", "ssd_num_chunks"),
    "chunk_valid_start": ("chunks", "ssd_num_chunks"),
    "chunk_valid_end": ("chunks", "ssd_num_chunks"),
}

# Scalars the tiling publishes alongside the arrays.
_SSD_SCALARS = ("ssd_starts_aligned", "ssd_active_is_prefix")

# CuteDSL descriptors require 16-byte aligned data pointers.
_SSD_ALIGNMENT = 16


@dataclass(frozen=True)
class SSDTilingLists:
    """How the varlen SSD kernel tiles one prefill batch, as host lists.

    Attributes:
        arrays: One list per name in `_SSD_ARRAYS`, in that key space.
        starts_aligned: True when no sequence starts mid-chunk.
        active_is_prefix: True when the active slots come first in the batch.
    """

    arrays: Dict[str, List[int]]
    starts_aligned: bool
    active_is_prefix: bool


def compute_ssd_tiling(
    cu_seqlens_all: List[int], padded_prefill_count: int, chunk_size: int
) -> SSDTilingLists:
    """Derive how the varlen SSD kernel tiles this prefill batch.

    Args:
        cu_seqlens_all: Cumulative token counts over all padded prefill slots.
        padded_prefill_count: Number of padded prefill slots.
        chunk_size: The kernel's tile length.

    Returns:
        The tiling, as host lists.
    """
    active, chunk_base, chunk_count, chunk_start = [], [], [], []
    token_base, valid_start, valid_end = [], [], []
    acc = 0
    for i in range(padded_prefill_count):
        start, end = cu_seqlens_all[i], cu_seqlens_all[i + 1]
        if end <= start:
            continue
        active.append(i)

        base = start // chunk_size
        count = -(-(end - base * chunk_size) // chunk_size)
        chunk_base.append(base)
        chunk_count.append(count)
        chunk_start.append(acc)
        acc += count

        for c in range(count):
            token_base.append((base + c) * chunk_size)
            valid_start.append(start)
            valid_end.append(end)

    active_set = set(active)
    empty = [i for i in range(padded_prefill_count) if i not in active_set]

    return SSDTilingLists(
        arrays={
            "seq_chunk_start": chunk_start,
            "seq_chunk_count": chunk_count,
            "seq_chunk_base": chunk_base,
            "active_seq_idx": active,
            "empty_seq_idx": empty,
            "chunk_token_base": token_base,
            "chunk_valid_start": valid_start,
            "chunk_valid_end": valid_end,
        },
        starts_aligned=all(cu_seqlens_all[i] % chunk_size == 0 for i in active),
        active_is_prefix=active == list(range(len(active))),
    )


class MambaSSDBufferLayout:
    """Byte layout of the SSD arrays inside a coalesced bookkeeping buffer.

    One instance describes the same block on both sides of the H2D, so
    `DynamicInferenceContext` (pinned CPU) and `ContextGPUView` (device) size
    and bind it with the same two calls.
    """

    def __init__(self, max_requests: int, max_mamba_chunks: int):
        """
        Args:
            max_requests: Maximum number of concurrent requests.
            max_mamba_chunks: Maximum chunk count across batch configurations.
        """
        max_ssd_chunks = max_mamba_chunks + max_requests
        capacities = {"requests": max_requests, "chunks": max_ssd_chunks}
        # Round each field to the alignment so that, given an aligned block
        # start, every field inside it starts aligned too.
        self.field_bytes = {
            name: round_up_to_nearest_multiple(capacities[capacity] * 4, _SSD_ALIGNMENT)
            for name, (capacity, _) in _SSD_ARRAYS.items()
        }

    @staticmethod
    def view_names() -> Tuple[str, ...]:
        """The keys `bind` returns views under, whether or not the backend is on."""
        return tuple(f"ssd_{name}" for name in _SSD_ARRAYS)

    @staticmethod
    def create(
        enabled: bool, max_requests: int, max_mamba_chunks: int
    ) -> Optional["MambaSSDBufferLayout"]:
        """The layout, or None when the CuteDSL backend is not in use.

        Args:
            enabled: Whether the CuteDSL SSD backend is selected.
            max_requests: Maximum number of concurrent requests.
            max_mamba_chunks: Maximum chunk count across batch configurations.
        """
        if not enabled:
            return None
        return MambaSSDBufferLayout(max_requests, max_mamba_chunks)

    def bytes_after(self, preceding_bytes: int) -> int:
        """Bytes this block adds after `preceding_bytes`, alignment padding included."""
        return self._pad(preceding_bytes) + sum(self.field_bytes.values())

    def bind(self, offset: int, buf: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], int]:
        """Carve the SSD arrays out of `buf`.

        Args:
            offset: Byte offset just past the preceding block; the same value
                passed to `bytes_after`.
            buf: The uint8 bookkeeping buffer to carve views from.

        Returns:
            A (views keyed `ssd_<name>`, offset past the block) pair.
        """
        offset += self._pad(offset)
        views = {}
        for name, nbytes in self.field_bytes.items():
            views[f"ssd_{name}"] = buf[offset : offset + nbytes].view(torch.int32)
            offset += nbytes
        return views, offset

    @staticmethod
    def _pad(offset: int) -> int:
        """Bytes to skip from `offset` to reach the next aligned block start."""
        return (-offset) % _SSD_ALIGNMENT


class MambaSSDMetadata:
    """The per-step SSD tiling, as the arrays and scalars `SSDTiling` reads.

    Held by `MambaMetadata` only when the CuteDSL backend is enabled, so the
    Triton path allocates none of these buffers and runs none of this
    bookkeeping. Attribute names are part of the op-layer contract: `SSDTiling`
    reads `ssd_*`, `mamba_chunk_size`, `cu_seqlens` and
    `real_prefill_token_count` off an instance by attribute access only.
    """

    def __init__(self, max_requests: int, max_chunks: int, chunk_size: int, device):
        """
        Args:
            max_requests: Maximum number of concurrent requests.
            max_chunks: Maximum chunk count across batch configurations. A
                sequence starting mid-chunk adds one chunk, so the SSD arrays
                are sized for one extra chunk per request.
            chunk_size: The Mamba chunk size, which must equal the kernel's
                tile length.
            device: Where the standalone buffers live.
        """
        self.mamba_chunk_size = chunk_size
        self.max_ssd_chunks = max_chunks + max_requests
        capacities = {"requests": max_requests, "chunks": self.max_ssd_chunks}
        self._buffers = {
            name: torch.zeros(capacities[capacity], dtype=torch.int32, device=device)
            for name, (capacity, _) in _SSD_ARRAYS.items()
        }
        self.reset()

    @staticmethod
    def create(
        enabled: bool, max_requests: int, max_chunks: int, chunk_size: int, device
    ) -> Optional["MambaSSDMetadata"]:
        """The per-step tiling holder, or None when the CuteDSL backend is not in use.

        Args:
            enabled: Whether the CuteDSL SSD backend is selected.
            max_requests: Maximum number of concurrent requests.
            max_chunks: Maximum chunk count across batch configurations.
            chunk_size: The Mamba chunk size.
            device: Where the standalone buffers live.
        """
        if not enabled:
            return None
        return MambaSSDMetadata(max_requests, max_chunks, chunk_size, device)

    def reset(self) -> None:
        """Drop the previous step's views."""
        self.cu_seqlens = None
        self.real_prefill_token_count = 0
        self.ssd_starts_aligned = True
        self.ssd_active_is_prefix = True
        for name in _SSD_ARRAYS:
            setattr(self, f"ssd_{name}", None)

    def update(
        self,
        cu_seqlens_all: List[int],
        padded_prefill_count: int,
        cu_seqlens: torch.Tensor,
        real_prefill_token_count: int,
    ) -> None:
        """Compute the tiling into the standalone buffers.

        This is the counterpart of `MambaMetadata.update`: the path used when
        no context bound its coalesced buffers (unit tests).

        Args:
            cu_seqlens_all: Cumulative token counts over all padded prefill slots.
            padded_prefill_count: Number of padded prefill slots.
            cu_seqlens: The step's device cu_seqlens, exposed to `SSDTiling`.
            real_prefill_token_count: Tokens actually covered by the sequences.
        """
        tiling = compute_ssd_tiling(cu_seqlens_all, padded_prefill_count, self.mamba_chunk_size)
        for name, values in tiling.arrays.items():
            buf = self._buffers[name]
            buf[: len(values)].copy_(torch.tensor(values, dtype=torch.int32))
            setattr(self, f"ssd_{name}", buf[: len(values)])
        self._set_scalars(tiling, cu_seqlens, real_prefill_token_count)

    def write_cpu_buffers(
        self, bufs: Dict[str, torch.Tensor], cu_seqlens_all: List[int], padded_prefill_count: int
    ) -> Dict[str, int]:
        """Compute the tiling into the bound pinned CPU views.

        Args:
            bufs: Pinned CPU views, keyed as `MambaSSDBufferLayout.bind` keys them.
            cu_seqlens_all: Cumulative token counts over all padded prefill slots.
            padded_prefill_count: Number of padded prefill slots.

        Returns:
            The scalars and per-step array lengths `load_from_gpu_view` needs
            after the H2D, for merging into the compute_cpu_metadata dict.
        """
        tiling = compute_ssd_tiling(cu_seqlens_all, padded_prefill_count, self.mamba_chunk_size)
        for name, values in tiling.arrays.items():
            bufs[f"ssd_{name}"][: len(values)] = torch.tensor(values, dtype=torch.int32)
        return {
            "ssd_starts_aligned": tiling.starts_aligned,
            "ssd_active_is_prefix": tiling.active_is_prefix,
            "ssd_num_active_seqs": len(tiling.arrays["active_seq_idx"]),
            "ssd_num_empty_seqs": len(tiling.arrays["empty_seq_idx"]),
            "ssd_num_chunks": len(tiling.arrays["chunk_token_base"]),
        }

    def load_from_gpu_view(
        self, gpu_view, d: Dict[str, int], cu_seqlens: torch.Tensor, real_prefill_token_count: int
    ) -> None:
        """Point the arrays at the freshly-transferred shared GPU views.

        Args:
            gpu_view: The context's `ContextGPUView`.
            d: The dict returned by `MambaMetadata.compute_cpu_metadata`.
            cu_seqlens: The step's device cu_seqlens, exposed to `SSDTiling`.
            real_prefill_token_count: Tokens actually covered by the sequences.
        """
        for name, (_, count_key) in _SSD_ARRAYS.items():
            view = getattr(gpu_view, f"mamba_ssd_{name}")
            setattr(self, f"ssd_{name}", view[: d[count_key]])
        for name in _SSD_SCALARS:
            setattr(self, name, d[name])
        self.cu_seqlens = cu_seqlens
        self.real_prefill_token_count = real_prefill_token_count

    def _set_scalars(
        self, tiling: SSDTilingLists, cu_seqlens: torch.Tensor, real_prefill_token_count: int
    ) -> None:
        """Publish the host-side values `SSDTiling` reads alongside the arrays."""
        self.ssd_starts_aligned = tiling.starts_aligned
        self.ssd_active_is_prefix = tiling.active_is_prefix
        self.cu_seqlens = cu_seqlens
        self.real_prefill_token_count = real_prefill_token_count
