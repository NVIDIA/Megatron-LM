"""Bounded NCCL return staging with token accumulation and its transpose."""

from itertools import accumulate

import torch
import torch.distributed as dist
from torch.autograd.function import once_differentiable


def _round(splits: tuple[int, ...], q: int, k: int) -> tuple[list[int], list[int]]:
    counts = [min(q, max(0, count - k * q)) for count in splits]
    starts = [start + min(count, k * q)
              for start, count in zip(accumulate((0, *splits[:-1])), splits)]
    return counts, starts


def _peer_order(
    rows: torch.Tensor, spans: tuple[tuple[tuple[int, int], ...], ...]
) -> torch.Tensor:
    """Map peer-major communication rows to expert-major input rows."""
    base = torch.arange(rows.shape[0], device=rows.device)
    order = torch.cat(
        [base[start : start + length] for peer_spans in spans for start, length in peer_spans]
    )
    assert order.numel() == rows.shape[0]
    return order


class _ChunkedReturn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rows, mapping, group, send_splits, recv_splits, tokens, q, rounds, spans):
        order = _peer_order(rows, spans)
        ctx.save_for_backward(mapping, order)
        ctx.group = group
        ctx.send_splits = send_splits
        ctx.recv_splits = recv_splits
        ctx.q = q
        ctx.rounds = rounds
        ctx.input_shape = rows.shape
        send = rows.new_empty((min(group.size() * q, sum(send_splits)), rows.shape[1]))
        recv = rows.new_empty((min(group.size() * q, sum(recv_splits)), rows.shape[1]))
        if rounds == 1:
            output = rows.new_empty((tokens, rows.shape[1]))
            torch.index_select(rows, 0, order, out=send)
            work = dist.all_to_all_single(recv, send, output_split_sizes=recv_splits,
                                          input_split_sizes=send_splits, group=group,
                                          async_op=True)
            output.zero_()
            work.wait()
            if torch.are_deterministic_algorithms_enabled():
                output.index_add_(0, mapping, recv)
            else:
                output.scatter_add_(0, mapping[:, None].expand_as(recv), recv)
            return output
        output = torch.zeros((tokens, rows.shape[1]), dtype=rows.dtype, device=rows.device)
        for k in range(rounds):
            sends, send_starts = _round(send_splits, q, k)
            recvs, recv_starts = _round(recv_splits, q, k)
            packed = 0
            for start, count in zip(send_starts, sends):
                torch.index_select(rows, 0, order[start:start + count],
                                   out=send[packed:packed + count])
                packed += count
            dist.all_to_all_single(recv[:sum(recvs)], send[:sum(sends)],
                                   output_split_sizes=recvs, input_split_sizes=sends, group=group)
            packed = 0
            for start, count in zip(recv_starts, recvs):
                output.index_add_(0, mapping[start:start + count],
                                  recv[packed:packed + count])
                packed += count
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        mapping, order = ctx.saved_tensors
        group, q = ctx.group, ctx.q
        send_splits, recv_splits = ctx.recv_splits, ctx.send_splits
        send = grad_output.new_empty((min(group.size() * q, sum(send_splits)), grad_output.shape[1]))
        recv = grad_output.new_empty((min(group.size() * q, sum(recv_splits)), grad_output.shape[1]))
        grad_rows = grad_output.new_empty(ctx.input_shape)
        if ctx.rounds == 1:
            torch.index_select(grad_output, 0, mapping, out=send)
            dist.all_to_all_single(recv, send, output_split_sizes=recv_splits,
                                   input_split_sizes=send_splits, group=group)
            grad_rows.index_copy_(0, order, recv)
            return grad_rows, None, None, None, None, None, None, None, None
        for k in range(ctx.rounds):
            sends, send_starts = _round(send_splits, q, k)
            recvs, recv_starts = _round(recv_splits, q, k)
            packed = 0
            for start, count in zip(send_starts, sends):
                torch.index_select(grad_output, 0, mapping[start:start + count],
                                   out=send[packed:packed + count])
                packed += count
            dist.all_to_all_single(recv[:sum(recvs)], send[:sum(sends)],
                                   output_split_sizes=recvs, input_split_sizes=sends, group=group)
            packed = 0
            for start, count in zip(recv_starts, recvs):
                grad_rows.index_copy_(0, order[start:start + count],
                                      recv[packed:packed + count])
                packed += count
        return grad_rows, None, None, None, None, None, None, None, None


def chunked_return(
    rows: torch.Tensor,
    mapping: torch.Tensor,
    group: dist.ProcessGroup,
    send_splits: tuple[int, ...],
    recv_splits: tuple[int, ...],
    tokens: int,
    q: int,
    rounds: int,
    spans: tuple[tuple[tuple[int, int], ...], ...] | None = None,
) -> torch.Tensor:
    """Return expert rows to token owners using a group-consistent chunk plan.

    ``mapping`` maps the ordinary full receive row order to local token indices.
    ``rounds`` must equal ceil(global maximum peer count / q) on every rank.
    Only the token mapping and compact peer-order index are saved for backward.
    Communication buffers contain at most ``group.size() * q`` rows each.
    Accumulation uses the input dtype, as in the ordinary unpermute path.
    """
    if spans is None:
        starts = accumulate((0, *send_splits[:-1]))
        spans = tuple(((start, count),) for start, count in zip(starts, send_splits))
    return _ChunkedReturn.apply(rows, mapping, group, send_splits, recv_splits, tokens, q, rounds, spans)
