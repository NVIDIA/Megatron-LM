# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Pinned-CPU shadow state dicts for async checkpoint loading.

A "shadow" is a tree shaped like a model's sharded_state_dict, but with every
ShardedTensor/Factory ``.data`` replaced by a pinned CPU buffer. Async loads
read checkpoint bytes into the shadow instead of live GPU weights.
"""

from __future__ import annotations

import copy
from typing import Any

import torch

from .dict_utils import dict_list_map_outplace
from .mapping import ShardedTensor, ShardedTensorFactory


def build_cpu_shadow_sharded_state_dict(model) -> dict:
    """Build a shadow of ``model``'s sharded state dict backed by pinned CPU buffers.

    Replacing ``factory.data`` (the GPU param) with a CPU buffer keeps the later
    ``apply_factories`` chunking from aliasing into live GPU storage.

    Args:
        model: list of model chunks (as passed to Megatron checkpointing).

    Returns: dict keyed like Megatron checkpoints ('model' or 'model{i}').
    """
    shadow: dict[str, Any] = {}
    for i, model_i in enumerate(model):
        key = "model" if len(model) == 1 else f"model{i}"
        model_sd = unwrap_for_sharded_state_dict(model_i).sharded_state_dict()
        shadow[key] = dict_list_map_outplace(_shadow_leaf, model_sd)
    return shadow


def unwrap_for_sharded_state_dict(module):
    """Strip DDP / Float16Module wrappers so the produced keys match the keys
    in the on-disk checkpoint metadata (which was saved without DDP wrap)."""
    while hasattr(module, "module") and isinstance(getattr(module, "module"), torch.nn.Module):
        module = module.module
    return module


def _shadow_leaf(node: Any) -> Any:
    """Copy a ShardedTensor/Factory with ``.data`` replaced by a pinned CPU
    buffer; pass any other leaf through untouched."""
    if isinstance(node, (ShardedTensor, ShardedTensorFactory)):
        new_node = copy.copy(node)
        new_node.data = _new_pinned_like(node.data)
        return new_node
    return node


def _new_pinned_like(original: torch.Tensor) -> torch.Tensor:
    try:
        return torch.empty(original.shape, dtype=original.dtype, device="cpu", pin_memory=True)
    except RuntimeError:
        return torch.empty(original.shape, dtype=original.dtype, device="cpu")


class ShadowBufferPool:
    """Pool of ``num_buffers`` pinned-CPU shadow trees shared by every load.

    A shadow only lives between kick and commit, so a small pool suffices:
    ``num_buffers=1`` (default) keeps host RAM independent of the number of
    distinct checkpoints; ``num_buffers>1`` enables double-buffering.

    ``acquire`` raises rather than blocks when no buffer is free — blocking
    inside a collective-ordered kick would deadlock, since the release that
    frees a buffer is also caller-driven and collective.
    """

    def __init__(self, model, num_buffers: int = 1):
        if num_buffers < 1:
            raise ValueError(f"ShadowBufferPool needs num_buffers >= 1, got {num_buffers}")
        self._all: list[dict] = [
            build_cpu_shadow_sharded_state_dict(model) for _ in range(num_buffers)
        ]
        self._free: list[dict] = list(self._all)

    def num_buffers(self) -> int:
        """Total number of shadow buffers owned by the pool."""
        return len(self._all)

    def num_free(self) -> int:
        """Number of currently free shadow buffers."""
        return len(self._free)

    def acquire(self) -> dict:
        """Lease a free shadow tree; raises RuntimeError when none is free."""
        if not self._free:
            raise RuntimeError(
                "ShadowBufferPool exhausted — finalize and release an outstanding "
                "async load before acquiring another shadow."
            )
        return self._free.pop()

    def release(self, shadow: dict) -> None:
        """Return a previously acquired shadow tree to the pool."""
        if not any(shadow is buf for buf in self._all):
            raise RuntimeError("ShadowBufferPool.release() called with a foreign shadow tree.")
        if any(shadow is buf for buf in self._free):
            raise RuntimeError("ShadowBufferPool.release() called twice for the same shadow tree.")
        self._free.append(shadow)
