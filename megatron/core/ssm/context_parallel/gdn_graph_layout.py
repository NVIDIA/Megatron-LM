# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""GPU packing metadata and differentiable per-document CP redistribution."""

import torch

from megatron.core.ssm.context_parallel.chunkwise import PackedSequenceCPMetadata
from megatron.core.ssm.context_parallel.gdn_layout import _AllToAll


class _GatherHalo(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tails, group):
        ctx.group = group
        gathered = tails.new_empty((group.size() * tails.shape[0], *tails.shape[1:]))
        torch.distributed.all_gather_into_tensor(gathered, tails.contiguous(), group=group)
        return gathered.view(group.size(), *tails.shape)

    @staticmethod
    def backward(ctx, grad):
        result = torch.empty_like(grad[0])
        torch.distributed.reduce_scatter_tensor(
            result, grad.flatten(0, 1).contiguous(), group=ctx.group
        )
        return result, None


class _Redistributor:
    def __init__(self, cu, group, local, halo):
        self.group = group
        size, rank = group.size(), group.rank()
        local_cu = cu // size
        half_cu = cu // (2 * size)
        positions = torch.arange(local, device=cu.device)
        sequence = torch.searchsorted(local_cu[1:], positions, right=True)
        offset = positions - local_cu[sequence]
        lengths = local_cu[1:] - local_cu[:-1]
        half_length = lengths[sequence] // 2
        half = offset // half_length
        packed_offset = half_cu[sequence] + offset.remainder(half_length)
        self.unsend = half * (local // 2) + packed_offset
        self.receive = (half if 2 * rank < size else 1 - half) * (local // 2) + packed_offset
        half_positions = torch.arange(local // 2, device=cu.device)
        half_sequence = torch.searchsorted(half_cu[1:], half_positions, right=True)
        first = local_cu[half_sequence] + half_positions - half_cu[half_sequence]
        second = first + lengths[half_sequence] // 2
        self.send = torch.cat((first, second))
        self.unreceive = self.send if 2 * rank < size else torch.cat((second, first))
        sends, receives = [0] * size, [0] * size
        for destination in (rank // 2, size - 1 - rank // 2):
            sends[destination] += local // 2
        for chunk in (2 * rank, 2 * rank + 1):
            receives[min(chunk, 2 * size - 1 - chunk)] += local // 2
        self.send_counts, self.receive_counts = tuple(sends), tuple(receives)
        self.local_cu, self.sequence = local_cu, sequence
        self.halo = halo
        self.local_indices = positions + (sequence + 1) * halo
        count = cu.numel() - 1
        self.extended_cu = (
            local_cu + torch.arange(count + 1, device=cu.device, dtype=torch.int32) * halo
        )
        extended_positions = torch.arange(local + count * halo, device=cu.device)
        extended_sequence = torch.searchsorted(self.extended_cu[1:], extended_positions, right=True)
        extended_offset = extended_positions - self.extended_cu[extended_sequence] - halo
        self.extended_ids = extended_sequence.to(torch.int32).unsqueeze(0)
        self.extend_indices = torch.where(
            extended_offset < 0,
            extended_sequence * halo + extended_offset + halo,
            count * halo + local_cu[extended_sequence] + extended_offset,
        )
        tail_offset = torch.arange(-halo, 0, device=cu.device)
        tail_positions = local_cu[1:, None] + tail_offset
        self.tail_indices = tail_positions.clamp(min=0)
        self.tail_valid = tail_positions >= local_cu[:-1, None]
        preceding = rank * lengths[:, None] + tail_offset
        source = preceding.div(lengths[:, None], rounding_mode="floor").clamp(min=0)
        tail_slot = preceding.remainder(lengths[:, None]) - lengths[:, None] + halo
        documents = torch.arange(count, device=cu.device)[:, None]
        self.halo_indices = (source * count + documents) * halo + tail_slot
        self.halo_valid = preceding >= 0

    def forward(self, x):
        transferred = _AllToAll.apply(
            x.index_select(0, self.send), self.send_counts, self.receive_counts, self.group
        )
        return transferred.index_select(0, self.receive)

    def inverse(self, x):
        transferred = _AllToAll.apply(
            x.index_select(0, self.unreceive), self.receive_counts, self.send_counts, self.group
        )
        return transferred.index_select(0, self.unsend)

    def extend(self, x):
        if not self.halo:
            return x
        tails = x[self.tail_indices].masked_fill(~self.tail_valid[..., None, None], 0)
        gathered = _GatherHalo.apply(tails, self.group)
        halo = gathered.flatten(0, 2).index_select(0, self.halo_indices.flatten())
        halo = halo.masked_fill(~self.halo_valid.flatten()[:, None, None], 0)
        return torch.cat((halo, x)).index_select(0, self.extend_indices)


def _build_layout(hidden, cu, kv, group, conv_kernel):
    total = hidden.shape[0] * group.size()
    if cu is None:
        cu = torch.arange(2, device=hidden.device, dtype=torch.int32) * total
        kv = cu
    if (
        cu.ndim != 1
        or cu.dtype != torch.int32
        or cu.numel() < 2
        or cu.device != hidden.device
        or kv is None
        or kv.shape != cu.shape
        or kv.dtype != cu.dtype
        or kv.device != cu.device
    ):
        raise ValueError("GDN physical Q/KV boundaries must be matching CUDA int32 vectors")
    lengths = cu[1:] - cu[:-1]
    torch._assert_async(
        (cu[0] == 0)
        & (cu[-1] == total)
        & (lengths > 0).all()
        & (lengths.remainder(2 * group.size()) == 0).all()
        & (cu == kv).all(),
        "GDN physical Q/KV packing must match and cover the input in multiples of 2*CP",
    )
    transport = _Redistributor(cu, group, hidden.shape[0], conv_kernel - 1)
    metadata = PackedSequenceCPMetadata(
        local_seq_idx=transport.sequence.to(torch.int32).unsqueeze(0),
        local_cu_seqlens=transport.local_cu,
        preceding_rank_start=0,
        following_rank_stop=group.size(),
    )
    return transport, metadata, transport.extended_cu, transport.extended_ids
