# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Shared replay contexts, planning, and residual-connection dispatch.

Connections choose which operations can be replayed and which state stays live.
Callers register operations in forward order and finalize each block at its live boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import torch
from torch import Tensor

from megatron.core.tensor_parallel.random import (
    CheckpointWithoutOutput,
    CheckpointWithoutOutputManager,
)
from megatron.core.transformer.residual_connection import (
    ResidualBranchOutput,
    ResidualConnection,
    ResidualConnectionState,
)
from megatron.core.transformer.residual_recompute_plan import (
    build_recompute_block_end_plan,
    build_recompute_layer_managers,
    finalize_recompute_block,
)

_R = TypeVar("_R")


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


def checkpoint_residual_read(
    connection: ResidualConnection,
    hidden_states: Tensor,
    context: ResidualStreamRecomputeContext,
    *,
    fp32_residual_connection: bool,
    branch_input_dtype: torch.dtype | None = None,
) -> tuple[Tensor, ResidualConnectionState]:
    """Checkpoint a residual read with its implementation's state and replay policy."""

    return connection._checkpoint_read(
        hidden_states,
        context,
        fp32_residual_connection=fp32_residual_connection,
        branch_input_dtype=branch_input_dtype,
    )


def checkpoint_residual_write(
    connection: ResidualConnection,
    branch_output: ResidualBranchOutput,
    state: ResidualConnectionState,
    context: ResidualStreamRecomputeContext,
    *,
    dropout_probability: float,
    training: bool,
) -> Tensor:
    """Checkpoint a residual write with its implementation's replay policy."""

    return connection._checkpoint_write(
        branch_output, state, context, dropout_probability=dropout_probability, training=training
    )
