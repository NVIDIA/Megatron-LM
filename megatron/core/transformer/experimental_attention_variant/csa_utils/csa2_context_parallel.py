# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CSA2 local compression and canonical shared-K addressing on the DSv4 CP path."""

from dataclasses import dataclass

import torch
from torch import Tensor

from .cp_utils import get_cp_compressor_capacity
from .thd_utils import CSA2THDCompressionLayout, CSA2THDLayout


@dataclass(frozen=True)
class CSA2CPCompressionLayout:
    """Local DSv4 compact groups and their canonical sequence-major gather map.

    DSv4 compaction can include an extra group entirely in the left halo. Each
    global key is read only from the rank owning its last source token, so that
    redundant group receives no gradient. Physical and logical padding keep
    their original addresses; only floating-point K tensors are communicated.
    """

    local: CSA2THDCompressionLayout
    global_layout: CSA2THDCompressionLayout
    gather_indices: Tensor

    def to_sequence_major(self, gathered: Tensor) -> Tensor:
        """Select each group's unique owner, retaining the all-gather autograd edge."""
        if gathered.shape[0] != self.local.cp_size * self.local.capacity:
            raise ValueError("CSA2 gathered K must match the rank-major compression capacity")
        output = gathered.index_select(0, self.gather_indices)
        invalid = ~self.global_layout.valid_groups.reshape(-1, *([1] * (gathered.ndim - 1)))
        return output.masked_fill(invalid, 0)


def build_csa2_cp_compression_layout(
    tokens: CSA2THDLayout, ratio: int, halo_rows: int
) -> CSA2CPCompressionLayout:
    """Mirror DSv4's compact group order using device-only prefix arithmetic.

    Local source indices address ``cat([left_halo, local_hidden])``. Group
    positions remain relative to the original sequence, including a group
    straddling a CP cut. Capacity depends only on host-known tensor geometry.
    """
    if tokens.cp_size <= 1 or tokens.total_tokens <= 0:
        raise ValueError("CSA2 CP compression requires nonempty contiguous CP partitions")
    if ratio not in (1, 2) or halo_rows < ratio:
        raise ValueError("CSA2 CP compression requires r1/r2 and the DSv4 compression halo")
    global_layout = tokens.for_compression(ratio)
    capacity = get_cp_compressor_capacity(tokens.total_tokens, ratio)
    cu = tokens.cu_seqlens_padded
    starts, ends = cu[:-1].long(), cu[1:].long()
    begin, end = tokens.global_start, tokens.global_start + tokens.total_tokens
    # Match CompressorInputCompact, including the preceding complete group
    # when a sequence crosses an aligned CP boundary.
    first = (begin - ratio - starts).clamp_min(0).add(ratio - 1) // ratio
    stop = (ends.clamp_max(end) - starts).clamp_min(0) // ratio
    counts = (stop - first).clamp_min(0).masked_fill(ends.clamp_max(end) <= begin, 0)
    zero = cu.new_zeros(1)
    local_cu = torch.cat((zero, counts.cumsum(0).to(cu.dtype)))
    logical_counts = torch.minimum(
        (tokens.cu_seqlens.diff().long() // ratio - first).clamp_min(0), counts
    )
    logical_cu = torch.cat((zero, logical_counts.cumsum(0).to(cu.dtype)))
    rows = torch.arange(capacity, device=cu.device, dtype=torch.int64)
    if starts.numel() == 0:
        sequence_ids = torch.full_like(rows, -1)
        valid = torch.zeros_like(rows, dtype=torch.bool)
        positions = torch.zeros_like(rows)
        sources = torch.zeros((capacity, ratio), dtype=torch.int64, device=cu.device)
    else:
        sequence_ids = torch.bucketize(rows, local_cu[1:], right=True).clamp_max(starts.numel() - 1)
        group_ids = rows - local_cu[sequence_ids] + first[sequence_ids]
        assigned = rows < local_cu[-1]
        valid = assigned & ((group_ids + 1) * ratio <= tokens.cu_seqlens.diff()[sequence_ids])
        positions = (group_ids * ratio).masked_fill(~valid, 0)
        sources = (
            starts[sequence_ids, None]
            + group_ids[:, None] * ratio
            + torch.arange(ratio, device=cu.device)
            - begin
            + halo_rows
        ).masked_fill(~valid[:, None], 0)
        sequence_ids = sequence_ids.masked_fill(~assigned, -1)
    local = CSA2THDCompressionLayout(
        ratio=ratio,
        total_tokens=tokens.total_tokens + halo_rows,
        capacity=capacity,
        max_seqlen=global_layout.max_seqlen,
        cu_seqlens=logical_cu,
        cu_seqlens_padded=local_cu,
        sequence_ids=sequence_ids,
        position_ids=positions,
        source_indices=sources,
        valid_groups=valid,
        cp_rank=tokens.cp_rank,
        cp_size=tokens.cp_size,
    )
    if global_layout.capacity == 0:
        gather_indices = rows[:0]
    else:
        owner = global_layout.source_indices[:, -1] // tokens.total_tokens
        rank_start = owner * tokens.total_tokens
        first_seq = torch.bucketize(rank_start, cu[1:], right=True).clamp_max(starts.numel() - 1)
        first_group = (rank_start - ratio - starts[first_seq]).clamp_min(0).add(ratio - 1) // ratio
        first_global = global_layout.cu_seqlens_padded[first_seq] + first_group
        gather_indices = (
            owner * capacity + torch.arange(global_layout.capacity, device=cu.device) - first_global
        ).masked_fill(~global_layout.valid_groups, 0)
    return CSA2CPCompressionLayout(local, global_layout, gather_indices)
