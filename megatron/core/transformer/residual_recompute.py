# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Ordered output-discard replay for residual-stream operations.

Wide-residual maps are static model parameters, so replay only needs the
residual-stream inputs and branch outputs already present in the autograd graph.
Within each replay block, cheap reads, connected norms, and non-terminal writes
are reconstructed in forward order during backward.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

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
from megatron.core.typed_torch import apply_module

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig

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

        if self.is_block_end:
            self.manager.discard_all_outputs_and_register_unified_recompute(hidden_states)


def residual_stream_recompute_enabled(config: TransformerConfig, training: bool) -> bool:
    """Return whether selective residual-stream replay is active for this forward."""

    return bool(
        training
        and torch.is_grad_enabled()
        and config.recompute_granularity == "selective"
        and config.recompute_modules is not None
        and "residual_stream" in config.recompute_modules
    )


def build_residual_stream_recompute_plan(
    num_layers: int, block_size: int | None, *, atomic_layer_pairs: Sequence[tuple[int, int]] = ()
) -> list[ResidualStreamRecomputeContext]:
    """Partition local layers into independent ordered replay blocks.

    ``atomic_layer_pairs`` keeps adjacent physical layers in the same replay block when the
    first layer produces a replay-owned activation consumed by the second.
    """

    if num_layers < 0:
        raise ValueError("Residual recompute plan requires a non-negative layer count.")
    if num_layers == 0:
        return []
    if block_size is not None and (
        isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1
    ):
        raise ValueError("Residual recompute block size must be a positive integer or None.")

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

    effective_block_size = block_size or num_layers
    atomic_units = []
    layer_index = 0
    while layer_index < num_layers:
        unit_end = pair_starts.get(layer_index, layer_index)
        atomic_units.append((layer_index, unit_end))
        layer_index = unit_end + 1

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

    contexts = []
    manager = CheckpointWithoutOutputManager()
    for layer_index in range(num_layers):
        is_block_end = layer_index in block_ends
        contexts.append(ResidualStreamRecomputeContext(manager=manager, is_block_end=is_block_end))
        if is_block_end and layer_index + 1 < num_layers:
            manager = CheckpointWithoutOutputManager()
    return contexts


def checkpoint_residual_read(
    connection: ResidualConnection,
    hidden_states: Tensor,
    context: ResidualStreamRecomputeContext,
    *,
    fp32_residual_connection: bool,
    branch_input_dtype: torch.dtype | None = None,
) -> tuple[Tensor, ResidualConnectionState]:
    """Checkpoint a residual read while retaining its carried stream as state.

    An optional branch-input dtype request is captured by the replay closure so eager execution
    and backward reconstruction produce the same branch dtype.
    """

    def run_read(stream: Tensor) -> tuple[Tensor, ...]:
        branch_input, state = apply_module(connection)(
            stream,
            operation="read",
            fp32_residual_connection=False,
            branch_input_dtype=branch_input_dtype,
        )
        if state[0].shape != stream.shape:
            raise ValueError("Residual connection read returned an incompatible carried stream.")
        return (branch_input, *state[1:])

    outputs = context.checkpoint(run_read, hidden_states)
    if torch.is_tensor(outputs):
        outputs = (outputs,)
    if not isinstance(outputs, tuple) or not outputs:
        raise TypeError("Checkpointed residual read must return a non-empty tensor tuple.")
    if not all(torch.is_tensor(output) for output in outputs):
        raise TypeError("Checkpointed residual read state must contain only tensors.")

    # This mirrors ResidualConnection's defensive state promotion. Normal full-model
    # FP32-residual inputs are already FP32, making .float() an alias rather than a cast kernel;
    # direct/custom low-precision inputs still need the conversion to preserve replay state.
    residual_stream = hidden_states.float() if fp32_residual_connection else hidden_states
    return outputs[0], (residual_stream, *outputs[1:])


def checkpoint_residual_write(
    connection: ResidualConnection,
    branch_output: ResidualBranchOutput,
    state: ResidualConnectionState,
    context: ResidualStreamRecomputeContext,
    *,
    dropout_probability: float,
    training: bool,
) -> Tensor:
    """Checkpoint one residual write while preserving standard module hooks."""

    if isinstance(branch_output, tuple):
        output, bias = branch_output
        return_tuple = True
    else:
        output, bias = branch_output, None
        return_tuple = False

    def run_write(
        branch_update: Tensor, branch_bias: Tensor | None, *connection_state: Tensor
    ) -> Tensor:
        value = (branch_update, branch_bias) if return_tuple else branch_update
        return apply_module(connection)(
            value,
            operation="write",
            state=connection_state,
            dropout_probability=dropout_probability,
            training=training,
        )

    hidden_states = context.checkpoint(run_write, output, bias, *state)
    if not torch.is_tensor(hidden_states):
        raise TypeError("Checkpointed residual write must return a tensor.")
    return hidden_states
