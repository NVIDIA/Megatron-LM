# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The set of operations a backend can own."""

from enum import Enum


class Operation(str, Enum):
    """One :class:`~megatron.core.ops.BackendSpecProvider` method a backend can own.

    Each value is the name of the provider method that fills the slot, so an operation
    and the method implementing it cannot drift apart. Add a member here only when an
    existing class, callable, or builder already owns that boundary at construction time.
    """

    LINEAR = "linear"
    COLUMN_PARALLEL_LINEAR = "column_parallel_linear"
    ROW_PARALLEL_LINEAR = "row_parallel_linear"
    COLUMN_PARALLEL_LAYER_NORM_LINEAR = "column_parallel_layer_norm_linear"
    LAYER_NORM = "layer_norm"
    CORE_ATTENTION = "core_attention"
    GROUPED_MLP_MODULES = "grouped_mlp_modules"
    ACTIVATION_FUNC = "activation_func"
    MOE_ROUTER = "moe_router"
    VOCAB_PARALLEL_CROSS_ENTROPY = "vocab_parallel_cross_entropy"

    def __str__(self) -> str:
        return self.value


def parse_operation(name: str) -> Operation:
    """Return the operation named ``name``, or raise with the valid choices listed."""
    try:
        return Operation(name)
    except ValueError:
        choices = ", ".join(operation.value for operation in Operation)
        raise ValueError(f"Unknown operation '{name}'. Valid operations: {choices}") from None
