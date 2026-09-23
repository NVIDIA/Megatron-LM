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


def _pieces(spans: tuple[tuple[int, int], ...], offset: int, count: int):
    """Return source slices for one bounded interval of a peer's expert chunks."""
    for start, length in spans:
        skipped = min(offset, length)
        offset -= skipped
        take = min(count, length - skipped)
        if take:
            yield start + skipped, take
            count -= take
        if not count:
            break


class _ChunkedReturn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rows, mapping, group, send_splits, recv_splits, tokens, q, rounds, spans):
        ctx.save_for_backward(mapping)
        ctx.group = group
        ctx.send_splits = send_splits
        ctx.recv_splits = recv_splits
        ctx.q = q
        ctx.rounds = rounds
        ctx.input_shape = rows.shape
        ctx.spans = spans
        send = rows.new_empty((min(group.size() * q, sum(send_splits)), rows.shape[1]))
        recv = rows.new_empty((min(group.size() * q, sum(recv_splits)), rows.shape[1]))
        accumulator_dtype = torch.float32 if rows.dtype in (torch.float16, torch.bfloat16) else rows.dtype
        output = torch.zeros((tokens, rows.shape[1]), dtype=accumulator_dtype, device=rows.device)
        for k in range(rounds):
            sends, _ = _round(send_splits, q, k)
            recvs, recv_starts = _round(recv_splits, q, k)
            packed = 0
            for peer, count in enumerate(sends):
                for start, length in _pieces(spans[peer], k * q, count):
                    send[packed:packed + length].copy_(rows[start:start + length])
                    packed += length
            dist.all_to_all_single(recv[:sum(recvs)], send[:sum(sends)],
                                   output_split_sizes=recvs, input_split_sizes=sends, group=group)
            packed = 0
            for start, count in zip(recv_starts, recvs):
                output.index_add_(0, mapping[start:start + count],
                                  recv[packed:packed + count].to(accumulator_dtype))
                packed += count
        return output.to(rows.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (mapping,) = ctx.saved_tensors
        group, q = ctx.group, ctx.q
        send_splits, recv_splits = ctx.recv_splits, ctx.send_splits
        send = grad_output.new_empty((min(group.size() * q, sum(send_splits)), grad_output.shape[1]))
        recv = grad_output.new_empty((min(group.size() * q, sum(recv_splits)), grad_output.shape[1]))
        grad_rows = grad_output.new_empty(ctx.input_shape)
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
            for peer, count in enumerate(recvs):
                for start, length in _pieces(ctx.spans[peer], k * q, count):
                    grad_rows[start:start + length].copy_(recv[packed:packed + length])
                    packed += length
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
    Only the mapping and host metadata are saved for first-order backward.
    Communication buffers contain at most ``group.size() * q`` rows each.
    """
    if spans is None:
        starts = accumulate((0, *send_splits[:-1]))
        spans = tuple(((start, count),) for start, count in zip(starts, send_splits))
    return _ChunkedReturn.apply(rows, mapping, group, send_splits, recv_splits, tokens, q, rounds, spans)
