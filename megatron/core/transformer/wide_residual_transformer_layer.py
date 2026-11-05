# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
"""Transformer adapter for wide-residual read/write connections."""

from __future__ import annotations

from typing import Optional

from megatron.core.fusions.wide_residual_shortcut.api import PendingShortcutAdd
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.residual_connection import (
    ResidualConnection,
    ResidualConnectionOperation,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.transformer.wide_residual import (
    SeamInput,
    StreamwiseSigmoidWideResidualConnection,
    apply_pending_write,
    defer_write,
    fuses_lone_seams,
    load_residual_state,
)


class WideResidualTransformerLayer(TransformerLayer):
    """Transformer layer carrying a wide stream around ordinary-width branches."""

    supports_wide_residual_connections: bool = True
    # Fused residual seams (see wide_residual.py): set when the next layer applies this write.
    defers_residual_write: bool = False

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

    @property
    def fuses_residual_seams(self) -> bool:
        """Whether neighbouring seam kernels may apply this lone MLP-only layer's read and write."""
        return (
            isinstance(self.self_attention, IdentityOp)
            and not isinstance(self.mlp, IdentityOp)
            and not self.is_moe_layer
            and fuses_lone_seams(self, self.pre_mlp_layernorm)
        )

    def seam_read(self):
        """The read, RMSNorm and MXFP8 consumer a fused shortcut block applies for this layer."""
        attention = self.self_attention
        projection = getattr(attention, "in_proj", getattr(attention, "linear_qkv", None))
        return self.residual_read_self_attn, self.input_layernorm, (attention, projection)

    def _run_input_layernorm(self, hidden_states, *args, **kwargs):
        if isinstance(hidden_states, SeamInput):
            self._input_layernorm_checkpoint_active = False
            return hidden_states.branch_input, hidden_states.residual, (hidden_states.residual,)
        return super()._run_input_layernorm(hidden_states, *args, **kwargs)

    def attention_bda_and_cross_attention(self, attention_output_with_bias, *args, **kwargs):
        # Only a compute-only layer's attention write is its last write; an MLP-only layer defers
        # its MLP write instead.
        if self.defers_residual_write and isinstance(self.mlp, IdentityOp):
            write = defer_write(
                kwargs["attn_state"][0], attention_output_with_bias, self.residual_write_self_attn
            )
            return write, kwargs.get("context")
        return super().attention_bda_and_cross_attention(
            attention_output_with_bias, *args, **kwargs
        )

    def _forward_attention(self, hidden_states, *args, **kwargs):
        if isinstance(hidden_states, PendingShortcutAdd):
            # An MLP-only layer applies a deferred write at its MLP read.
            return hidden_states, kwargs.get("context")
        return super()._forward_attention(hidden_states, *args, **kwargs)

    def _pre_mlp_layernorm_and_residual(self, hidden_states, *args, **kwargs):
        if not isinstance(hidden_states, PendingShortcutAdd):
            return super()._pre_mlp_layernorm_and_residual(hidden_states, *args, **kwargs)
        from megatron.core.transformer.moe.shared_experts import SharedExpertMLP

        # In a fused shortcut block the shared expert's first Linear alone consumes this input.
        shared = getattr(self.mlp, "shared_experts", None)
        consumer = None
        if (
            shared is not None
            and type(shared).forward is SharedExpertMLP.forward
            and not shared.use_shared_expert_gate
        ):
            consumer = (shared, shared.linear_fc1)
        residual, branch_input = apply_pending_write(
            hidden_states, (self.residual_read_mlp,), (self.pre_mlp_layernorm,), consumer
        )
        return branch_input, residual, (residual,)

    def _apply_mlp_bda_step(self, mlp_output_with_bias, residual, mlp_state=(), **kwargs):
        if self.defers_residual_write:
            return defer_write(mlp_state[0], mlp_output_with_bias, self.residual_write_mlp)
        return super()._apply_mlp_bda_step(mlp_output_with_bias, residual, mlp_state, **kwargs)
