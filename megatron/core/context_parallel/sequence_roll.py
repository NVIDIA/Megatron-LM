# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Contiguous packed sequence rolls with explicit groups and differentiable boundary exchange."""

import torch
import torch.distributed as dist


def _continuation_mask(tensor, packed_seq_params, cp_group, dims, sp_group=None):
    """Return the local rows whose next token belongs to the same real document."""
    rank = cp_group.rank() if cp_group is not None else 0
    if sp_group is not None:
        rank = rank * sp_group.size() + sp_group.rank()
    length = tensor.shape[dims]
    real = packed_seq_params.cu_seqlens_q
    physical = packed_seq_params.cu_seqlens_q_padded
    physical = real if physical is None else physical
    rows = torch.arange(rank * length, (rank + 1) * length, device=tensor.device)
    seq = torch.bucketize(rows, physical[1:], right=True)
    in_pack = seq < physical.numel() - 1
    seq = seq.clamp_max(physical.numel() - 2)
    ends = physical[seq] + (real[1:] - real[:-1])[seq]
    valid = in_pack & (rows >= physical[seq]) & (rows + 1 < ends)
    return valid


def exchange_neighbor_tokens(tokens, group, directions, *, wrap=False):
    """Batch boundary exchanges, preserving each token's shape and dtype.

    Each direction is -1 (send left, receive right) or +1 (send right, receive left).
    Missing neighbors return None unless wrap is enabled. Callers own boundary fills
    and autograd; this helper only performs communication on the supplied group.
    """
    rank = group.rank() if group is not None else 0
    size = group.size() if group is not None else 1
    received, ops = [], []
    for token, direction in zip(tokens, directions):
        send_rank, recv_rank = rank + direction, rank - direction
        if wrap:
            send_rank, recv_rank = send_rank % size, recv_rank % size
        if 0 <= send_rank < size:
            ops.append(
                dist.P2POp(
                    dist.isend, token.contiguous(), dist.get_global_rank(group, send_rank), group
                )
            )
        recv = None
        if 0 <= recv_rank < size:
            recv = torch.empty_like(token, memory_format=torch.contiguous_format)
            ops.append(dist.P2POp(dist.irecv, recv, dist.get_global_rank(group, recv_rank), group))
        received.append(recv)
    # P2POp retains each contiguous send buffer until the batch completes.
    for work in dist.batch_isend_irecv(ops) if ops else []:
        work.wait()
    return received


class _RollLeft(torch.autograd.Function):
    """One differentiable left roll, including the reverse neighbor exchange."""

    @staticmethod
    def forward(ctx, tensor, group, dims, sp_group):
        """Shift local values and receive the next rank's first row."""
        ctx.group, ctx.dims, ctx.sp_group = group, dims, sp_group
        result = torch.roll(tensor, -1, dims)
        boundary = tensor.select(dims, 0)
        if sp_group is None or sp_group.rank() == 0:
            (boundary,) = exchange_neighbor_tokens((boundary,), group, (-1,))
            if boundary is None:
                boundary = torch.zeros_like(tensor.select(dims, 0))
        # Owners are ordered CP rank first, then SP rank. Relay the next CP
        # shard's first token from SP rank zero to the last SP rank.
        if sp_group is not None and sp_group.size() > 1:
            (boundary,) = exchange_neighbor_tokens((boundary,), sp_group, (-1,), wrap=True)
        result.select(dims, -1).copy_(boundary)
        return result

    @staticmethod
    def backward(ctx, grad):
        """Return the boundary gradient to its original token owner."""
        group, dims, sp_group = ctx.group, ctx.dims, ctx.sp_group
        result = torch.roll(grad, 1, dims)
        boundary = grad.select(dims, -1)
        if sp_group is not None and sp_group.size() > 1:
            (boundary,) = exchange_neighbor_tokens((boundary,), sp_group, (1,), wrap=True)
        if sp_group is None or sp_group.rank() == 0:
            (boundary,) = exchange_neighbor_tokens((boundary,), group, (1,))
            if boundary is None:
                boundary = torch.zeros_like(grad.select(dims, -1))
        result.select(dims, 0).copy_(boundary)
        return result, None, None, None


def roll_contiguous(tensor, dims, cp_group, packed_seq_params=None, fill_value=0, *, sp_group=None):
    """Roll a contiguous shard by one, preserving real document boundaries.

    With sequence parallelism, each CP shard is further split in SP rank order.
    Both boundary exchanges are differentiable and use the supplied groups.
    """
    if tensor.shape[dims] == 0:
        return tensor.clone()
    cp_size = cp_group.size() if cp_group is not None else 1
    sp_size = sp_group.size() if sp_group is not None else 1
    if cp_size == sp_size == 1:
        result = torch.roll(tensor, -1, dims)
        result.select(dims, -1).fill_(fill_value)
    else:
        result = _RollLeft.apply(tensor, cp_group, dims, sp_group)
        if fill_value != 0:
            last_cp = cp_group is None or cp_group.rank() == cp_size - 1
            last_sp = sp_group is None or sp_group.rank() == sp_size - 1
            if last_cp and last_sp:
                result.select(dims, -1).fill_(fill_value)
    if packed_seq_params is None:
        return result
    valid = _continuation_mask(tensor, packed_seq_params, cp_group, dims, sp_group)
    shape = [1] * tensor.ndim
    shape[dims] = tensor.shape[dims]
    return result.masked_fill_(~valid.view(shape), fill_value)


def roll_contiguous_fields(tensors, cp_group, packed_seq_params=None, fill_values=None):
    """Roll non-differentiable token fields together, preserving each dtype and fill value.

    This batches one depth's IDs/positions/masks; activations continue to use the
    differentiable roll above.
    """
    present = [tensor for tensor in tensors if tensor is not None]
    if not present:
        return tuple(tensors)
    reference = present[0]
    length = reference.shape[-1]
    if any(
        t.requires_grad or t.shape[-1] != length or t.device != reference.device for t in present
    ):
        raise ValueError("Token fields require matching sequence lengths/devices and no gradients.")
    if fill_values is None:
        fill_values = (0,) * len(tensors)
    if len(fill_values) != len(tensors):
        raise ValueError("Each token field needs one fill value.")
    if length == 0:
        return tuple(None if t is None else t.clone() for t in tensors)
    invalid = (
        ~_continuation_mask(reference, packed_seq_params, cp_group, -1)
        if packed_seq_params is not None
        else None
    )
    boundaries = exchange_neighbor_tokens(
        [tensor[..., 0] for tensor in present], cp_group, [-1] * len(present)
    )
    received = iter(boundaries)
    outputs = []
    for tensor, fill in zip(tensors, fill_values):
        if tensor is None:
            outputs.append(None)
            continue
        result = torch.roll(tensor, -1, -1)
        boundary = next(received)
        if boundary is not None:
            result[..., -1].copy_(boundary)
        else:
            result[..., -1].fill_(fill)
        if invalid is not None:
            result.masked_fill_(invalid, fill)
        outputs.append(result)
    return tuple(outputs)
