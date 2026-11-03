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
    StreamwiseSigmoidWideResidualConnection,
    is_fusable_rmsnorm,
    load_residual_state,
    pending_write,
    write_pending_and_read_norm,
)


class WideResidualTransformerLayer(TransformerLayer):
    """Transformer layer carrying a wide stream around ordinary-width branches."""

    supports_wide_residual_connections: bool = True
    # Set by ShortcutMoEBlock at init when the fused shortcut kernels own the self-attention
    # residual read/norm and write: the layer takes a normalized input and returns the raw
    # attention output with bias.
    using_shortcut_moe_fused_residual_kernel: bool = False
    # Set by group_layers_into_shortcut_blocks on an MLP-only layer when the next layer applies
    # its write: forward returns a PendingShortcutAdd instead of the written residual.
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

    def _run_input_layernorm(self, hidden_states, *args, **kwargs):
        """Skip the read and input norm when the fused shortcut kernel already applied them."""

        if self.using_shortcut_moe_fused_residual_kernel:
            self._input_layernorm_checkpoint_active = False
            return hidden_states, None, ()
        return super()._run_input_layernorm(hidden_states, *args, **kwargs)

    def attention_bda_and_cross_attention(self, attention_output_with_bias, *args, **kwargs):
        """Return the raw attention output when the fused shortcut kernel owns the write."""

        if self.using_shortcut_moe_fused_residual_kernel:
            return attention_output_with_bias, kwargs.get("context")
        return super().attention_bda_and_cross_attention(
            attention_output_with_bias, *args, **kwargs
        )

    def _is_lone_mlp(self) -> bool:
        return (
            isinstance(self.self_attention, IdentityOp)
            and not isinstance(self.mlp, IdentityOp)
            and not self.is_moe_layer
            and not self.recompute_pre_mlp_layernorm
        )

    def pending_write_reader(self):
        """The MLP read and pre-MLP RMSNorm a fused seam kernel can apply, or None.

        Only MLP-only layers qualify; with a reader the layer accepts a PendingShortcutAdd as
        ``hidden_states``.
        """
        if not self._is_lone_mlp() or not is_fusable_rmsnorm(self.pre_mlp_layernorm):
            return None
        return self.residual_read_mlp, self.pre_mlp_layernorm

    def pending_write_writer(self):
        """The MLP residual write an MLP-only layer can defer to the next seam, or None."""
        return self.residual_write_mlp if self._is_lone_mlp() else None

    def _forward_attention(self, hidden_states, *args, **kwargs):
        """An MLP-only layer passes a deferred write straight to its MLP read."""

        if isinstance(hidden_states, PendingShortcutAdd):
            if self.pending_write_reader() is None:
                raise RuntimeError("this layer cannot apply a deferred residual write")
            return hidden_states, kwargs.get("context")
        return super()._forward_attention(hidden_states, *args, **kwargs)

    def _pre_mlp_layernorm_and_residual(self, hidden_states, *args, **kwargs):
        """Apply a deferred write together with the MLP read and pre-MLP RMSNorm."""

        if isinstance(hidden_states, PendingShortcutAdd):
            if kwargs.get("residual_stream_recompute_context") is not None:
                raise RuntimeError("a deferred residual write cannot be replayed here")
            residual, branch_input = write_pending_and_read_norm(
                hidden_states, *self.pending_write_reader()
            )
            return branch_input, residual, (residual,)
        return super()._pre_mlp_layernorm_and_residual(hidden_states, *args, **kwargs)

    def _apply_mlp_bda_step(self, mlp_output_with_bias, residual, mlp_state=(), **kwargs):
        """Return the MLP write as a PendingShortcutAdd when the next layer applies it."""

        if not self.defers_residual_write:
            return super()._apply_mlp_bda_step(mlp_output_with_bias, residual, mlp_state, **kwargs)
        if not mlp_state:
            raise RuntimeError("Missing state for the MLP residual connection.")
        return pending_write(mlp_state[0], mlp_output_with_bias, self.residual_write_mlp)
