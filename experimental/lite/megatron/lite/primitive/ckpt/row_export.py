# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Bounded native row export alongside the existing HF parameter shard writer."""
import json
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import save


@dataclass
class TensorRowChunk:
    """One contiguous row interval of a tensor too large to gather in memory."""

    tensor: torch.Tensor
    start: int
    rows: int


def iter_row_shards(local_rows, *, max_bytes=64 * 1024**2, rank0_only=False):
    """Stream owner rows over the default export group, retaining native storage."""
    local = {name: (value, start, rows) for name, value, start, rows in local_rows}
    rank, world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
    metadata = [(n, r, v.shape, v.dtype, total) for n, (v, r, total) in local.items()]
    owners = [metadata]
    if world > 1:
        owners = [None] * world
        dist.all_gather_object(owners, metadata)
    layout = {}
    for owner, entries in enumerate(owners):
        for name, start, shape, dtype, total in entries:
            layout.setdefault(name, {}).setdefault(start, (owner, shape, dtype, total))
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if world > 1 and dist.get_backend() == "nccl"
        else torch.device("cpu")
    )
    for name, shards in sorted(layout.items()):
        cursor = 0
        for start, (owner, shape, dtype, total) in sorted(shards.items()):
            if shape[0] == 0:
                continue
            if start != cursor:
                raise ValueError(f"Noncontiguous export rows for {name}: {start} != {cursor}")
            step = max(1, max_bytes // (math.prod(shape[1:]) * dtype.itemsize))
            for offset in range(0, shape[0], step):
                count = min(step, shape[0] - offset)
                payload = (
                    local[name][0][offset : offset + count]
                    .contiguous()
                    .view(torch.uint8)
                    .to(device)
                    if rank == owner
                    else torch.empty((count, *shape[1:]), dtype=dtype, device=device).view(
                        torch.uint8
                    )
                )
                if world > 1:
                    dist.broadcast(payload, owner)
                if not rank0_only or rank == 0:
                    yield name, TensorRowChunk(payload.cpu().view(dtype), start + offset, total)
            cursor += shape[0]
        if cursor != total:
            raise ValueError(f"Incomplete export rows for {name}: {cursor} != {total}")


def save_row_shards(export_iter, path):
    """Append streamed row tensors to an HF export without assembling full tables."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    root = Path(path)
    index = root / "model.safetensors.index.json"
    weights, files, total_size = {}, [], 0
    if rank == 0:
        if index.exists():
            previous = json.loads(index.read_text())
            weights, total_size = previous["weight_map"], previous["metadata"]["total_size"]
        else:
            with safe_open(root / "model.safetensors", framework="pt") as reader:
                weights = dict.fromkeys(reader.keys(), "model.safetensors")
            with open(root / "model.safetensors", "rb") as raw:
                header_bytes = 8 + struct.unpack("<Q", raw.read(8))[0]
            total_size = (root / "model.safetensors").stat().st_size - header_bytes
        files = list(dict.fromkeys(weights.values()))
    output, cursor, current = None, 0, None
    for name, part in export_iter:
        if rank != 0:
            continue
        value = part.tensor.contiguous().cpu()
        if output is None:
            if part.start != 0 or name in weights:
                raise ValueError(f"Invalid first row or duplicate tensor: {name}")
            stub = save({name: torch.empty((0,), dtype=value.dtype)})
            header = json.loads(stub[8 : 8 + struct.unpack("<Q", stub[:8])[0]])
            size = part.rows * math.prod(value.shape[1:]) * value.element_size()
            header[name].update(shape=[part.rows, *value.shape[1:]], data_offsets=[0, size])
            encoded = json.dumps(header, separators=(",", ":")).encode()
            encoded += b" " * (-len(encoded) % 8)
            filename = f".model-shard-{len(files) + 1:05d}.safetensors"
            output = open(root / filename, "wb")
            output.write(struct.pack("<Q", len(encoded)) + encoded)
            files.append(filename)
            weights[name], total_size = filename, total_size + size
            current, cursor = name, 0
        if name != current or part.start != cursor:
            output.close()
            raise ValueError("Interleaved or noncontiguous row export")
        output.write(value.view(torch.uint8).numpy().tobytes())
        cursor += value.shape[0]
        if cursor == part.rows:
            output.close()
            output = None
    if output is not None:
        output.close()
        raise ValueError("Incomplete final row tensor")
    if rank == 0:
        renamed = {}
        for i, old in enumerate(files, 1):
            new = f"model-{i:05d}-of-{len(files):05d}.safetensors"
            temporary = f".model-export-{i:05d}.safetensors"
            os.replace(root / old, root / temporary)
            renamed[old] = (temporary, new)
        for temporary, new in renamed.values():
            os.replace(root / temporary, root / new)
        index.write_text(
            json.dumps(
                {
                    "metadata": {"total_size": total_size},
                    "weight_map": {k: renamed[v][1] for k, v in weights.items()},
                }
            )
        )
    if dist.is_initialized():
        dist.barrier()
