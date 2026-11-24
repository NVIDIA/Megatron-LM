# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Per-document row, halo and compression maps for packed zigzag hybrid attention."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .thd_layout_kernels import CPCompressorLayout


@dataclass
class PackedZigzagLayout:
    """Rank-local query geometry and an invertible exchange for its document-local halos."""

    q_position_ids: torch.Tensor
    kv_position_ids: torch.Tensor
    row_geometry: torch.Tensor
    send_indices: torch.Tensor
    recv_order: torch.Tensor
    send_splits: list
    recv_splits: list
    compressor_layout: Optional[CPCompressorLayout]
    d_window: int

    def exchange(self, hidden, group):
        """Gather left context for each segment and sum its gradients at their source."""
        from megatron.core.tensor_parallel.mappings import all_to_all

        sends = hidden.index_select(0, self.send_indices)
        received = all_to_all(group, sends, self.recv_splits, self.send_splits)
        return received.index_select(0, self.recv_order)


def _transfers(boundaries, chunks, cp_size, src, dst, window):
    """Intervals shared by src's local segments and dst's requested halos, in wire order."""
    for doc, chunk in zip(boundaries, chunks):
        if not chunk:
            continue
        for dst_v in (dst, 2 * cp_size - 1 - dst):
            end = dst_v * chunk
            left = max(0, end - window)
            for side, src_v in enumerate((src, 2 * cp_size - 1 - src)):
                start = max(left, src_v * chunk)
                stop = min(end, (src_v + 1) * chunk)
                if start < stop:
                    local = doc // cp_size + side * chunk + start - src_v * chunk
                    yield local, doc + start, stop - start


def build_packed_zigzag_layout(boundaries, cp_rank, cp_size, local_rows, ratio, window, device):
    """Build index maps from the host-known document boundaries for this microbatch."""
    lengths = np.diff(np.asarray(boundaries, dtype=np.int64))
    if (
        not len(lengths)
        or boundaries[0] != 0
        or boundaries[-1] != local_rows * cp_size
        or np.any(lengths < 0)
        or np.any(lengths % (2 * cp_size))
    ):
        raise ValueError(
            f"Packed zigzag needs padded document lengths divisible by 2 * CP ({2 * cp_size}) "
            f"covering {local_rows * cp_size} tokens, got {boundaries}."
        )
    if ratio == 4:
        raise NotImplementedError(
            "Packed zigzag does not support the overlapping ratio-4 CSA indexer."
        )
    chunks = (lengths // (2 * cp_size)).tolist()
    # cu_seqlens may contain repeated boundaries for unused metadata slots. Preserve their
    # sequence IDs in the compressed prefixes, and omit their empty segments from transport.
    q_pos = np.empty(local_rows, dtype=np.int32)
    geometry = np.empty((local_rows, 4), dtype=np.int32)
    halo_positions = []
    halo_global = []
    halo_rows = 0
    for doc, chunk in zip(boundaries, chunks):
        for side, v in enumerate((cp_rank, 2 * cp_size - 1 - cp_rank)):
            start = doc + v * chunk
            left = max(doc, start - window)
            local = doc // cp_size + side * chunk
            rows = slice(local, local + chunk)
            positions = np.arange(start, start + chunk, dtype=np.int32)
            q_pos[rows] = positions - doc
            geometry[rows, 0] = positions
            geometry[rows, 1] = start
            geometry[rows, 2] = halo_rows - left
            geometry[rows, 3] = local - start
            halo = np.arange(left, start, dtype=np.int32)
            halo_global.append(halo)
            halo_positions.append(halo - doc)
            halo_rows += len(halo)
    geometry[:, 3] += halo_rows

    def on_device(array):
        return torch.as_tensor(array, device=device).contiguous()

    # Send each global row once per destination, even if two halos share it. The receive
    # permutation restores document/segment order and differentiably duplicates shared rows.
    sends, send_splits, received_global, recv_splits = [], [], [], []
    for peer in range(cp_size):
        outgoing = _transfers(boundaries, chunks, cp_size, cp_rank, peer, window)
        indices = {idx for local, _, size in outgoing for idx in range(local, local + size)}
        send_splits.append(len(indices))
        sends.extend(sorted(indices))
        incoming = _transfers(boundaries, chunks, cp_size, peer, cp_rank, window)
        # Per-document [front, back] is ascending in both local and global row order.
        indices = {idx for _, pos, size in incoming for idx in range(pos, pos + size)}
        recv_splits.append(len(indices))
        received_global.extend(sorted(indices))

    wire_slot = {pos: slot for slot, pos in enumerate(received_global)}
    recv_order = [wire_slot[pos] for pos in np.concatenate(halo_global).tolist()]

    compressor = None
    if ratio > 1:
        # The owner of a block's final token has its entire input in [halo | local].
        # All ranks must publish equally sized buffers to the existing KV all-gather.
        prefix = np.concatenate(([0], np.cumsum(lengths // ratio))).astype(np.int32)
        counts = [0] * cp_size
        owners, slots, own_blocks, sources = [], [], [], []
        for doc, chunk, n_groups in zip(boundaries, chunks, lengths // ratio):
            for group_id in range(n_groups):
                last = (group_id + 1) * ratio - 1
                virtual = last // chunk
                owner = min(virtual, 2 * cp_size - 1 - virtual)
                owners.append(owner)
                slots.append(counts[owner])
                counts[owner] += 1
                if owner != cp_rank:
                    continue
                local = doc // cp_size + int(virtual >= cp_size) * chunk
                _, start, halo_base, local_base = geometry[local]
                block = np.arange(doc + group_id * ratio, doc + (group_id + 1) * ratio)
                sources.append(np.where(block < start, block + halo_base, block + local_base))
                own_blocks.append(group_id)
        capacity = max(1, max(counts))
        group_ids = np.full(capacity, -1, dtype=np.int32)
        group_ids[: len(own_blocks)] = own_blocks
        forward_map = np.full(capacity * ratio, -1, dtype=np.int32)
        inverse = np.full(halo_rows + local_rows, -1, dtype=np.int32)
        if sources:
            source = np.concatenate(sources)
            forward_map[: len(source)] = source
            inverse[source] = np.arange(len(source), dtype=np.int32)
        rank_rows = np.asarray(owners, dtype=np.int32) * capacity + np.asarray(
            slots, dtype=np.int32
        )
        # Each compact slot is an independent, complete, non-overlapping ratio-sized group.
        # Group IDs retain document-relative positions for RoPE. No padded slot is indexed.
        compact_cu = np.array([0, capacity], dtype=np.int32)
        compressor = CPCompressorLayout(
            on_device(forward_map),
            on_device(inverse),
            on_device(group_ids),
            on_device(np.maximum(group_ids, 0) * ratio),
            on_device(compact_cu * ratio),
            on_device(compact_cu),
            on_device(prefix),
            on_device(rank_rows),
            halo_rows,
            0,
            cp_size,
            boundary_segments=2 * len(lengths),
        )
    return PackedZigzagLayout(
        on_device(q_pos),
        on_device(np.concatenate([*halo_positions, q_pos])),
        on_device(geometry),
        on_device(np.asarray(sends, dtype=np.int64)),
        on_device(np.asarray(recv_order, dtype=np.int64)),
        send_splits,
        recv_splits,
        compressor,
        window,
    )
