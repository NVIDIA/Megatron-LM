# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Batch tensor utilities for MIMO training."""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import fields

import torch

from megatron.core.packed_seq_params import PackedSeqParams


def map_batch_tensors(
    value, transform: Callable[[torch.Tensor], torch.Tensor], *, preserve_keys: Collection[str] = ()
):
    """Transform tensor leaves, excluding subtrees under named dictionary keys.

    ``PackedSeqParams`` tensor fields are included. The default transforms every
    tensor; callers may preserve host metadata such as per-image sizes.
    """
    if isinstance(value, torch.Tensor):
        return transform(value)
    if isinstance(value, dict):
        return {
            key: (
                item
                if key in preserve_keys
                else map_batch_tensors(item, transform, preserve_keys=preserve_keys)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [map_batch_tensors(item, transform, preserve_keys=preserve_keys) for item in value]
    if isinstance(value, tuple):
        return tuple(
            map_batch_tensors(item, transform, preserve_keys=preserve_keys) for item in value
        )
    if isinstance(value, PackedSeqParams):
        for field in fields(value):
            item = getattr(value, field.name)
            if isinstance(item, torch.Tensor):
                setattr(value, field.name, transform(item))
    return value


def move_batch_to_cuda(value):
    """Move model tensors to CUDA, retaining ``imgs_sizes`` host metadata.

    Dynamic vision encoders read the CPU per-image sizes in Python. Attention
    indices in ``PackedSeqParams`` still move to CUDA with pixels and token IDs.
    Callers must supply ``imgs_sizes`` on CPU.
    """
    return map_batch_tensors(
        value, lambda tensor: tensor.cuda(non_blocking=True), preserve_keys={"imgs_sizes"}
    )
