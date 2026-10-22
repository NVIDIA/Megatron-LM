# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
"""Transformer adapter for wide-residual read/write connections."""

from __future__ import annotations

from typing import Optional

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.residual_connection import (
    ResidualConnection,
    ResidualConnectionOperation,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.transformer.wide_residual import (
    StreamwiseSigmoidWideResidualConnection,
    load_residual_state,
)


class WideResidualTransformerLayer(TransformerLayer):
    """Transformer layer carrying a wide stream around ordinary-width branches."""

    supports_wide_residual_connections: bool = True

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: Optional[float] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
        is_mtp_layer: bool = False,
        add_layer_offset: bool = True,
        pp_layer_offset: Optional[int] = None,
        hash_moe_layer_threshold: Optional[int] = None,
        name: str | None = None,
    ) -> None:
        if is_mtp_layer:
            raise ValueError(
                "MTP auxiliary stacks must use ordinary-width TransformerLayer, not "
                "WideResidualTransformerLayer."
            )
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            hidden_dropout=hidden_dropout,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
            is_mtp_layer=is_mtp_layer,
            add_layer_offset=add_layer_offset,
            pp_layer_offset=pp_layer_offset,
            hash_moe_layer_threshold=hash_moe_layer_threshold,
            name=name,
        )

        if config.wide_residual is None:
            raise ValueError("WideResidualTransformerLayer requires wide_residual config.")
        if not isinstance(self.cross_attention, IdentityOp):
            raise NotImplementedError(
                "WideResidualTransformerLayer does not support cross-attention branches."
            )

        self.residual_read_self_attn = None
        self.residual_write_self_attn = None
        if not isinstance(self.self_attention, IdentityOp):
            self.residual_read_self_attn = StreamwiseSigmoidWideResidualConnection(
                config, self.layer_number, "self_attention", operation="read"
            )
            self.residual_write_self_attn = StreamwiseSigmoidWideResidualConnection(
                config, self.layer_number, "self_attention", operation="write"
            )

        self.residual_read_mlp = None
        self.residual_write_mlp = None
        if not isinstance(self.mlp, IdentityOp):
            self.residual_read_mlp = StreamwiseSigmoidWideResidualConnection(
                config, self.layer_number, "mlp", operation="read"
            )
            self.residual_write_mlp = StreamwiseSigmoidWideResidualConnection(
                config, self.layer_number, "mlp", operation="write"
            )

        self.register_load_state_dict_pre_hook(load_residual_state)
        residual_connections = [
            reader
            for reader in (self.residual_read_self_attn, self.residual_read_mlp)
            if reader is not None
        ]

        if self.residual_read_self_attn is not None and self._input_layernorm_returns_residual:
            raise ValueError(
                "A self-attention residual connection cannot be combined with a layer norm "
                "that returns its own residual."
            )
        if self.residual_read_mlp is not None and self._pre_mlp_layernorm_returns_residual:
            raise ValueError(
                "An MLP residual connection cannot be combined with a layer norm that returns "
                "its own residual."
            )
        if residual_connections and self.config.inference_fuse_tp_communication:
            raise NotImplementedError(
                "Wide-residual connections do not support fused tensor-parallel inference."
            )

        self.residual_stream_hidden_size = (
            config.wide_residual.num_streams * self.config.hidden_size
        )
        for connection in residual_connections:
            if connection.residual_stream_hidden_size != self.residual_stream_hidden_size:
                raise ValueError(
                    "All wide-residual connections must carry num_streams * hidden_size "
                    f"features, expected {self.residual_stream_hidden_size}, got "
                    f"{connection.residual_stream_hidden_size}."
                )
            if connection.branch_hidden_size != self.config.hidden_size:
                raise ValueError(
                    "Wide-residual connections around TransformerLayer branches must produce "
                    f"hidden_size={self.config.hidden_size}, got "
                    f"{connection.branch_hidden_size}."
                )

    def _get_self_attention_residual_connection(
        self, operation: ResidualConnectionOperation
    ) -> ResidualConnection | None:
        """Return the reader or writer surrounding the self-attention branch."""

        match operation:
            case "read":
                return self.residual_read_self_attn
            case "write":
                return self.residual_write_self_attn
            case _:
                raise ValueError(f"Unsupported residual connection operation: {operation}.")

    def _get_mlp_residual_connection(
        self, operation: ResidualConnectionOperation
    ) -> ResidualConnection | None:
        """Return the reader or writer surrounding the MLP or MoE branch."""

        match operation:
            case "read":
                return self.residual_read_mlp
            case "write":
                return self.residual_write_mlp
            case _:
                raise ValueError(f"Unsupported residual connection operation: {operation}.")
