# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Differentiable zigzag/contiguous ownership and shared GDN packing metadata."""

import weakref
from collections import OrderedDict
from dataclasses import dataclass, fields, is_dataclass
from functools import lru_cache
from threading import RLock

import torch

from megatron.core.ssm.context_parallel.chunkwise import build_packed_sequence_cp_metadata
from megatron.core.transformer.cuda_graphs import register_capture_constant


@dataclass(frozen=True)
class _LayoutPlan:
    send_order: tuple[int, ...]
    send_counts: tuple[int, ...]
    receive_order: tuple[int, ...]
    receive_counts: tuple[int, ...]


def _zigzag_positions(boundaries, size, rank):
    positions = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if (stop - start) % (2 * size):
            raise ValueError("GDN physical packed lengths must be divisible by 2*CP")
        chunk = (stop - start) // (2 * size)
        for index in (rank, 2 * size - 1 - rank):
            positions.extend(range(start + index * chunk, start + (index + 1) * chunk))
    return tuple(positions)


@lru_cache(maxsize=32)
def build_layout_plan(boundaries, size, rank):
    """Plan a bijection from per-document zigzag to globally contiguous shards."""
    if size < 1 or not 0 <= rank < size:
        raise ValueError("Invalid CP rank or size")
    if (
        len(boundaries) < 2
        or boundaries[0] != 0
        or any(a >= b for a, b in zip(boundaries[:-1], boundaries[1:]))
    ):
        raise ValueError("Expected strictly increasing physical boundaries starting at zero")
    sources = [_zigzag_positions(boundaries, size, src) for src in range(size)]
    local = boundaries[-1] // size
    positions = sources[rank]
    send_order = tuple(sorted(range(local), key=lambda i: positions[i] // local))
    send_counts = tuple(sum(p // local == dest for p in positions) for dest in range(size))
    received = [p for source in sources for p in source if p // local == rank]
    receive_counts = tuple(sum(p // local == rank for p in source) for source in sources)
    if sorted(received) != list(range(rank * local, (rank + 1) * local)):
        raise ValueError("Packing does not cover the destination shard exactly")
    receive_order = tuple(sorted(range(local), key=received.__getitem__))
    return _LayoutPlan(send_order, send_counts, receive_order, receive_counts)


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, input_splits, output_splits, group):
        ctx.splits = (input_splits, output_splits)
        ctx.group = group
        result = x.new_empty((sum(output_splits), *x.shape[1:]))
        torch.distributed.all_to_all_single(
            result, x.contiguous(), list(output_splits), list(input_splits), group=group
        )
        return result

    @staticmethod
    def backward(ctx, grad):
        input_splits, output_splits = ctx.splits
        result = grad.new_empty((sum(input_splits), *grad.shape[1:]))
        torch.distributed.all_to_all_single(
            result, grad.contiguous(), list(input_splits), list(output_splits), group=ctx.group
        )
        return result, None, None, None


class _Redistributor:
    def __init__(self, plan, group, device):
        self.plan, self.group = plan, group
        self.send = torch.tensor(plan.send_order, dtype=torch.long, device=device)
        self.receive = torch.tensor(plan.receive_order, dtype=torch.long, device=device)
        self.unsend = torch.argsort(self.send)
        self.unreceive = torch.argsort(self.receive)

    def forward(self, x):
        transferred = _AllToAll.apply(
            x.index_select(0, self.send),
            self.plan.send_counts,
            self.plan.receive_counts,
            self.group,
        )
        return transferred.index_select(0, self.receive)

    def inverse(self, x):
        transferred = _AllToAll.apply(
            x.index_select(0, self.unreceive),
            self.plan.receive_counts,
            self.plan.send_counts,
            self.group,
        )
        return transferred.index_select(0, self.unsend)


_entries = OrderedDict()
_lock = RLock()


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, _Redistributor):
        yield from (value.send, value.receive, value.unsend, value.unreceive)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item)
    elif is_dataclass(value):
        for field in fields(value):
            yield from _tensors(getattr(value, field.name))


def _build_layout(hidden, cu, kv, group, conv_kernel):
    total = hidden.shape[0] * group.size()
    if cu is not None and (cu.ndim != 1 or cu.dtype != torch.int32):
        raise ValueError("GDN physical boundaries must be a one-dimensional int32 tensor")
    boundaries = (0, total) if cu is None else tuple(cu.cpu().tolist())
    plan = build_layout_plan(boundaries, group.size(), group.rank())
    if boundaries[-1] != total:
        raise ValueError("GDN packed physical boundaries do not cover the sharded input")
    if cu is not None and (kv is None or not torch.equal(cu, kv)):
        raise ValueError("GDN requires equal physical Q and KV packed boundaries")
    transport = _Redistributor(plan, group, hidden.device)
    lengths = torch.tensor(
        [b - a for a, b in zip(boundaries[:-1], boundaries[1:])], device=hidden.device
    )
    ids = torch.repeat_interleave(
        torch.arange(len(boundaries) - 1, dtype=torch.int32, device=hidden.device),
        lengths,
        output_size=total,
    )[None, :]
    metadata = build_packed_sequence_cp_metadata(ids, group.rank(), group.size())
    halo = conv_kernel - 1
    start, stop = group.rank() * hidden.shape[0] - halo, (group.rank() + 1) * hidden.shape[0]
    # Include boundaries inside the halo to prevent cross-document convolution.
    extended = [0] + [b - start for b in boundaries if start < b < stop] + [stop - start]
    extended_cu = torch.tensor(extended, dtype=torch.int32, device=hidden.device)
    lengths = torch.tensor(
        [b - a for a, b in zip(extended[:-1], extended[1:])], device=hidden.device
    )
    extended_ids = torch.repeat_interleave(
        torch.arange(len(lengths), dtype=torch.int32, device=hidden.device),
        lengths,
        output_size=hidden.shape[0] + halo,
    )[None, :]
    return transport, metadata, extended_cu, extended_ids


def get_chunkwise_layout(hidden, packed, group, conv_kernel):
    """Reuse immutable packing across GDN layers with bounded, stream-safe storage.

    Weak references and tensor mutation versions prevent stale identity reuse.
    Inference tensors without versions bypass caching. Capture reuses a layout
    warmed with these input buffers; physical packing is constant until recapture.
    """
    capturing = torch.cuda.is_current_stream_capturing()
    cu, kv = None, None
    if packed is not None:
        if packed.qkv_format != "thd":
            raise ValueError("GDN chunkwise CP supports packed THD or batch-one dense input")
        cu = (
            packed.cu_seqlens_q_padded
            if packed.cu_seqlens_q_padded is not None
            else packed.cu_seqlens_q
        )
        kv = (
            packed.cu_seqlens_kv_padded
            if packed.cu_seqlens_kv_padded is not None
            else packed.cu_seqlens_kv
        )
        if cu is None or kv is None:
            raise ValueError("Packed THD GDN requires physical Q and KV boundaries")
    try:
        versions = tuple(t._version if t is not None else None for t in (cu, kv))
    except RuntimeError:
        if capturing:
            raise RuntimeError("GDN CUDA graph packing tensors must have mutation versions")
        return _build_layout(hidden, cu, kv, group, conv_kernel)
    key = (
        id(cu),
        id(kv),
        versions,
        hidden.shape[0],
        hidden.device,
        id(group),
        group.size(),
        conv_kernel,
    )
    with _lock:
        entry = _entries.get(key)
        if entry is not None and all(
            ref is None or ref() is value for ref, value in zip(entry[0], (cu, kv))
        ):
            _entries.move_to_end(key)
            _, result, tensors, ready, creator = entry
            current = torch.cuda.current_stream(hidden.device)
            if current.cuda_stream != creator:
                current.wait_event(ready)
                for tensor in tensors:
                    tensor.record_stream(current)
            if capturing:
                for tensor in tensors:
                    register_capture_constant(tensor)
            return result
        if capturing:
            raise RuntimeError(
                "Warm the GDN chunkwise layout outside capture with the graph's physical "
                "packing buffers; changed physical packing requires recapture"
            )
        result = _build_layout(hidden, cu, kv, group, conv_kernel)
        tensors = tuple(_tensors(result))
        current = torch.cuda.current_stream(hidden.device)
        ready = torch.cuda.Event()
        ready.record(current)

        def discard(ref):
            with _lock:
                existing = _entries.get(key)
                if existing is not None and any(item is ref for item in existing[0]):
                    _entries.pop(key)

        refs = tuple(weakref.ref(t, discard) if t is not None else None for t in (cu, kv))
        _entries[key] = (refs, result, tensors, ready, current.cuda_stream)
        _entries.move_to_end(key)
        while len(_entries) > 8:
            _entries.popitem(last=False)
        return result
