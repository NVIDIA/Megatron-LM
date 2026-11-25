# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Shared replay contexts, block planning, and finalization.

Connections choose which operations can be replayed and which state stays live.
Callers register operations in forward order and finalize each block at its live boundary.
Only immutable boundary plans may be cached; managers belong to one forward invocation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from torch import Tensor

from megatron.core.tensor_parallel.random import (
    CheckpointWithoutOutput,
    CheckpointWithoutOutputManager,
)

_R = TypeVar("_R")


def build_recompute_block_end_plan(
    num_layers: int, block_size: int | None, *, atomic_layer_pairs: Sequence[tuple[int, int]] = ()
) -> tuple[bool, ...]:
    """Mark block ends without splitting disjoint adjacent atomic layer pairs.

    Indices use the caller's local layer order. A pair may exceed a block size of
    one; otherwise blocks contain at most ``block_size`` layers. ``None`` uses
    the entire local stack. Empty stacks return an empty plan.
    """
    if num_layers < 0:
        raise ValueError("Residual recompute plan requires a non-negative layer count.")
    if num_layers == 0:
        return ()
    if block_size is not None and (
        isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1
    ):
        raise ValueError("Residual recompute block size must be a positive integer or None.")

    effective_block_size = block_size or num_layers
    if not atomic_layer_pairs:
        return tuple(
            (layer_index + 1) % effective_block_size == 0 or layer_index == num_layers - 1
            for layer_index in range(num_layers)
        )

    pair_starts: dict[int, int] = {}
    paired_indices: set[int] = set()
    for pair in atomic_layer_pairs:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError("Atomic residual recompute layer pairs must be two-item tuples.")
        start, end = pair
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
        ):
            raise ValueError("Atomic residual recompute layer-pair indices must be integers.")
        if start < 0 or end >= num_layers or end != start + 1:
            raise ValueError(
                "Atomic residual recompute layer pairs must contain adjacent in-range indices."
            )
        if start in paired_indices or end in paired_indices:
            raise ValueError("Atomic residual recompute layer pairs must not overlap.")
        pair_starts[start] = end
        paired_indices.update((start, end))

    # Treat each unpaired layer or complete pair as one indivisible unit.
    atomic_units = []
    layer_index = 0
    while layer_index < num_layers:
        unit_end = pair_starts.get(layer_index, layer_index)
        atomic_units.append((layer_index, unit_end))
        layer_index = unit_end + 1

    # Pack whole units in order, shortening a block instead of splitting a pair.
    block_ends: set[int] = set()
    layers_in_block = 0
    for unit_start, unit_end in atomic_units:
        unit_size = unit_end - unit_start + 1
        if layers_in_block and layers_in_block + unit_size > effective_block_size:
            block_ends.add(unit_start - 1)
            layers_in_block = 0
        layers_in_block += unit_size
        if layers_in_block >= effective_block_size:
            block_ends.add(unit_end)
            layers_in_block = 0
    block_ends.add(num_layers - 1)
    return tuple(layer_index in block_ends for layer_index in range(num_layers))


def build_recompute_layer_managers(
    block_ends: Sequence[bool],
) -> list[CheckpointWithoutOutputManager]:
    """Allocate fresh managers for an active plan, sharing one within each block."""
    if not block_ends:
        return []

    managers = []
    manager = CheckpointWithoutOutputManager()
    for layer_index, is_block_end in enumerate(block_ends):
        managers.append(manager)
        if is_block_end and layer_index + 1 < len(block_ends):
            manager = CheckpointWithoutOutputManager()
    return managers


def finalize_recompute_block(
    manager: CheckpointWithoutOutputManager | None, hidden_states: Tensor, is_block_end: bool
) -> None:
    """Discard registered outputs and attach replay at the block's live boundary."""
    if manager is not None and is_block_end:
        manager.discard_all_outputs_and_register_unified_recompute(hidden_states)


@dataclass(frozen=True)
class ResidualStreamRecomputeContext:
    """One layer's immutable view of a shared residual-stream replay block."""

    manager: CheckpointWithoutOutputManager
    is_block_end: bool

    def checkpoint(self, function: Callable[..., _R], *args: Any, fp8: bool = False) -> _R:
        """Run one cheap operation and register it for ordered replay."""

        return CheckpointWithoutOutput(fp8=fp8, ckpt_manager=self.manager).checkpoint(
            function, *args
        )

    def finalize(self, hidden_states: Tensor) -> None:
        """Discard this block's registered outputs once its live boundary exists."""

        finalize_recompute_block(self.manager, hidden_states, self.is_block_end)


def build_residual_stream_recompute_plan(
    num_layers: int, block_size: int | None, *, atomic_layer_pairs: Sequence[tuple[int, int]] = ()
) -> list[ResidualStreamRecomputeContext]:
    """Partition local physical layers into independent ordered replay blocks.

    Shortcut MoE's predecessor (e.g. attention or Mamba) and paired MoE layer share
    replay-owned intermediates, so a replay block must not end between them.

    Args:
        num_layers: Local physical layer count, before shortcut wrapper grouping.
        block_size: Target maximum physical layers per block; None uses the entire local stack.
            A pair stays intact even when block_size=1, forming a two-layer block.
        atomic_layer_pairs: Disjoint adjacent pairs of zero-based local physical layer indices
            that must share one replay manager.

    Returns:
        One context per physical layer, sharing a manager within each block. Only the block's
        final layer has is_block_end=True.

    For example, four layers with block_size=3 and pairs (0, 1), (2, 3) form blocks
    [0, 1] and [2, 3], not [0, 1, 2] and [3].
    """

    block_ends = build_recompute_block_end_plan(
        num_layers, block_size, atomic_layer_pairs=atomic_layer_pairs
    )
    managers = build_recompute_layer_managers(block_ends)
    return [
        ResidualStreamRecomputeContext(manager=manager, is_block_end=is_block_end)
        for manager, is_block_end in zip(managers, block_ends)
    ]
