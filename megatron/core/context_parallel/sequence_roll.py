# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Contiguous packed sequence rolls with explicit groups and differentiable boundary exchange."""

import torch
import torch.distributed as dist


def _continuation_mask(tensor, packed_seq_params, cp_group, dims):
    """Return the local rows whose next token belongs to the same real document."""
    size = cp_group.size() if cp_group is not None else 1
    rank = cp_group.rank() if cp_group is not None else 0
    length = tensor.shape[dims]
    rows = torch.arange(rank * length, (rank + 1) * length, device=tensor.device)
    if packed_seq_params is None:
        ends = torch.full_like(rows, length * size)
        return rows + 1 < ends
    real = packed_seq_params.cu_seqlens_q
    physical = packed_seq_params.cu_seqlens_q_padded
    physical = real if physical is None else physical
    seq = torch.bucketize(rows, physical[1:], right=True)
    in_pack = seq < physical.numel() - 1
    seq = seq.clamp_max(physical.numel() - 2)
    ends = physical[seq] + (real[1:] - real[:-1])[seq]
    valid = in_pack & (rows >= physical[seq]) & (rows + 1 < ends)
    return valid


class _RollLeft(torch.autograd.Function):
    """One differentiable left roll, including the reverse neighbor exchange."""

    @staticmethod
    def forward(ctx, tensor, group, dims):
        """Shift local values and receive the next rank's first row."""
        ctx.group, ctx.dims = group, dims
        result = torch.roll(tensor, -1, dims)
        result.select(dims, -1).zero_()
        if group is not None and group.size() > 1:
            rank, size = group.rank(), group.size()
            send = tensor.select(dims, 0).contiguous()
            recv = torch.empty_like(send)
            ops = []
            if rank > 0:
                ops.append(
                    dist.P2POp(dist.isend, send, dist.get_global_rank(group, rank - 1), group)
                )
            if rank + 1 < size:
                ops.append(
                    dist.P2POp(dist.irecv, recv, dist.get_global_rank(group, rank + 1), group)
                )
            for work in dist.batch_isend_irecv(ops):
                work.wait()
            if rank + 1 < size:
                result.select(dims, -1).copy_(recv)
        return result

    @staticmethod
    def backward(ctx, grad):
        """Return the boundary gradient to its original token owner."""
        group, dims = ctx.group, ctx.dims
        result = torch.roll(grad, 1, dims)
        result.select(dims, 0).zero_()
        if group is not None and group.size() > 1:
            rank, size = group.rank(), group.size()
            send = grad.select(dims, -1).contiguous()
            recv = torch.empty_like(send)
            ops = []
            if rank + 1 < size:
                ops.append(
                    dist.P2POp(dist.isend, send, dist.get_global_rank(group, rank + 1), group)
                )
            if rank > 0:
                ops.append(
                    dist.P2POp(dist.irecv, recv, dist.get_global_rank(group, rank - 1), group)
                )
            for work in dist.batch_isend_irecv(ops):
                work.wait()
            if rank > 0:
                result.select(dims, 0).copy_(recv)
        return result, None, None


def roll_contiguous(tensor, dims, cp_group, packed_seq_params=None, fill_value=0):
    """Roll a dense or packed contiguous shard by one, preserving real document boundaries."""
    if tensor.shape[dims] == 0:
        return tensor.clone()
    valid = _continuation_mask(tensor, packed_seq_params, cp_group, dims)
    shape = [1] * tensor.ndim
    shape[dims] = tensor.shape[dims]
    return _RollLeft.apply(tensor, cp_group, dims).masked_fill(~valid.view(shape), fill_value)
