# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import torch

    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.transformer.transformer_config import TransformerConfig


class LinearInterface(Protocol):
    """Interface for linear layers."""

    def forward(self, hidden_states: torch.Tensor, /) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Applies the linear module to the input hidden states."""
        ...

    def backward_dw(self) -> None:
        """Compute weight gradients during the backward pass if delay_wgrad_compute is enabled."""
        ...


class LinearBuilder(Protocol):
    """Interface for building linear layers."""

    def __call__(
        self,
        input_size: int,
        output_size: int,
        /,
        *,
        parallel_mode: str | None,
        config: TransformerConfig,
        init_method: Callable[[torch.Tensor], None],
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: str | None,
        skip_weight_param_allocation: bool,
    ) -> LinearInterface: ...


class ColumnParallelLinearBuilder(Protocol):
    """Interface for building column_parallel_linear layers."""

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
    ) -> LinearInterface: ...


class RowParallelLinearBuilder(Protocol):
    """Interface for building row_parallel_linear layers."""

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
    ) -> LinearInterface: ...


class CoreAttentionInterface(Protocol):
    """Interface for core_attention layers."""

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        /,
        *,
        attn_mask_type: AttnMaskType,
        attention_bias: torch.Tensor | None,
        packed_seq_params: PackedSeqParams | None,
    ) -> torch.Tensor:
        """Applies dot product attention."""
        ...


class CoreAttentionBuilder(Protocol):
    """Interface for building core_attention layers."""

    def __call__(
        self,
        *,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        cp_comm_type: str | None,
        softmax_scale: float | None,
        pg_collection: ProcessGroupCollection | None,
    ) -> CoreAttentionInterface: ...
