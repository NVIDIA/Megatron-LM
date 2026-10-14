# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Batch tensor utilities for MIMO training."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields

import torch

from megatron.core.packed_seq_params import PackedSeqParams


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
