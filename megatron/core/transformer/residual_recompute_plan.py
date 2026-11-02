# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Residual recompute planning functions.

Only immutable boundary plans may be cached, such as a plan that keeps a
ShortcutMoE producer/consumer pair in the same block. Managers own checkpoints
from one forward invocation; callers choose which operations to register.
"""

from collections.abc import Sequence

from torch import Tensor

from megatron.core.tensor_parallel.random import CheckpointWithoutOutputManager


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
