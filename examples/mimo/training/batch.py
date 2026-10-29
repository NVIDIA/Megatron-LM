# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Batch tensor utilities for MIMO training."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields
from math import isqrt

import torch

from megatron.core.packed_seq_params import PackedSeqParams


def collate_encoder_inputs(items: list[dict[str, torch.Tensor]]) -> dict:
    """Combine fixed or patchified RADIO images with matching packed attention metadata."""
    x = items[0]["x"]
    batch = {"x": torch.cat([item["x"] for item in items], dim=1 if x.ndim == 3 else 0)}
    if "imgs_sizes" in items[0]:
        sizes = torch.cat([item["imgs_sizes"] for item in items])
        patch_dim = isqrt(x.shape[-1] // 3)
        if patch_dim == 0 or 3 * patch_dim**2 != x.shape[-1]:
            raise ValueError("patchified image feature width must be 3 * patch_dim^2")
        if sizes.device.type != "cpu" or sizes.ndim != 2 or sizes.shape[1] != 2:
            raise ValueError("imgs_sizes must be a CPU tensor of (height, width) rows")
        if (
            sizes.dtype not in (torch.int32, torch.int64)
            or torch.any(sizes <= 0)
            or torch.any(sizes % patch_dim)
        ):
            raise ValueError("image dimensions must be positive integer multiples of patch_dim")
        lengths = torch.prod(sizes // patch_dim, dim=-1, dtype=torch.int32)
        if int(lengths.sum()) != batch["x"].shape[1]:
            raise ValueError("imgs_sizes disagree with packed input patch count")
        cu = torch.cat((torch.zeros(1, dtype=torch.int32), lengths.cumsum(0, dtype=torch.int32)))
        maximum = int(lengths.max())
        batch.update(
            imgs_sizes=sizes,
            packed_seq_params=PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=cu,
                cu_seqlens_kv=cu.clone(),
                max_seqlen_q=maximum,
                max_seqlen_kv=maximum,
            ),
        )
    return batch


def map_batch_tensors(value, transform: Callable[[torch.Tensor], torch.Tensor]):
    """Apply a transform to tensor leaves, including PackedSeqParams fields."""
    if isinstance(value, torch.Tensor):
        return transform(value)
    if isinstance(value, dict):
        return {key: map_batch_tensors(item, transform) for key, item in value.items()}
    if isinstance(value, list):
        return [map_batch_tensors(item, transform) for item in value]
    if isinstance(value, tuple):
        return tuple(map_batch_tensors(item, transform) for item in value)
    if isinstance(value, PackedSeqParams):
        for field in fields(value):
            item = getattr(value, field.name)
            if isinstance(item, torch.Tensor):
                setattr(value, field.name, transform(item))
    return value


def move_batch_to_cuda(value):
    """Move tensor leaves, including PackedSeqParams tensor fields, to CUDA."""
    return map_batch_tensors(value, lambda tensor: tensor.cuda(non_blocking=True))
