# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import torch

    from megatron.core.transformer.transformer_config import TransformerConfig


class ColumnParallelLinearInterface(Protocol):
    """Protocol for ColumnParallelLinear modules."""

    def forward(self, hidden_states: torch.Tensor, /) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Applies the column parallel linear module to the input hidden states."""
        ...

    def backward_dw(self) -> None:
        """Compute weight gradients during the backward pass if delay_wgrad_compute is enabled."""
        ...


class ColumnParallelLinearBuilder(Protocol):
    """Protocol for building column_parallel_linear layers."""

    def __call__(
        self,
        input_size: int,
        output_size: int,
        /,
        *,
        config: TransformerConfig,
        init_method: Callable[[torch.Tensor], None],
        gather_output: bool,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: str | None,
        tp_group: torch.distributed.ProcessGroup | None,
        stride: int = 1,
    ) -> ColumnParallelLinearInterface: ...


class RowParallelLinearInterface(Protocol):
    """Protocol for RowParallelLinear modules."""

    def forward(self, hidden_states: torch.Tensor, /) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Applies the row parallel linear module to the input hidden states."""
        ...

    def backward_dw(self) -> None:
        """Compute weight gradients during the backward pass if delay_wgrad_compute is enabled."""
        ...


class RowParallelLinearBuilder(Protocol):
    """Protocol for building row_parallel_linear layers."""

    def __call__(
        self,
        input_size: int,
        output_size: int,
        /,
        *,
        config: TransformerConfig,
        init_method: Callable[[torch.Tensor], None],
        bias: bool,
        input_is_parallel: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: str | None,
        tp_group: torch.distributed.ProcessGroup | None,
    ) -> RowParallelLinearInterface: ...
