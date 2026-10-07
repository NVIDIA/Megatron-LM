# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Mamba layer specialization for streamwise wide-residual connections."""

from torch import Tensor

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_layer import MambaLayer, MambaLayerSubmodules
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.residual_connection import ResidualConnectionState
from megatron.core.transformer.residual_recompute import (
    ResidualStreamRecomputeContext,
    checkpoint_residual_read,
    checkpoint_residual_write,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.wide_residual_layer import (
    StreamwiseSigmoidWideResidualConnection,
    _load_residual_state,
)
from megatron.core.typed_torch import apply_module


class WideResidualMambaLayer(MambaLayer):
    """Mamba layer carrying a wide stream around its ordinary-width mixer."""

    supports_wide_residual_connections: bool = True

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MambaLayerSubmodules,
        layer_number: int = 1,
        pg_collection: ProcessGroupCollection = None,
        pp_layer_offset: int = 0,
        name: str | None = None,
        is_mtp_layer: bool = False,
    ) -> None:
        if is_mtp_layer:
            raise ValueError(
                "MTP auxiliary stacks must use ordinary-width MambaLayer, not "
                "WideResidualMambaLayer."
            )
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            pg_collection=pg_collection,
            pp_layer_offset=pp_layer_offset,
            name=name,
            is_mtp_layer=is_mtp_layer,
        )

        if config.wide_residual is None:
            raise ValueError("WideResidualMambaLayer requires wide_residual config.")
        if getattr(self.norm, "returns_residual", False):
            raise ValueError(
                "A Mamba residual connection cannot be combined with a layer norm that "
                "returns its own residual."
            )

        self.residual_read = StreamwiseSigmoidWideResidualConnection(
            config, self.layer_number, "mamba", mode="read"
        )
        self.residual_write = StreamwiseSigmoidWideResidualConnection(
            config, self.layer_number, "mamba", mode="write"
        )
        self.register_load_state_dict_pre_hook(_load_residual_state)
        self.residual_stream_hidden_size = (
            config.wide_residual.num_streams * self.config.hidden_size
        )
        if self.residual_read.residual_stream_hidden_size != self.residual_stream_hidden_size:
            raise ValueError(
                "The wide-residual Mamba connection must carry num_streams * hidden_size "
                f"features, expected {self.residual_stream_hidden_size}, got "
                f"{self.residual_read.residual_stream_hidden_size}."
            )
        if self.residual_read.branch_hidden_size != self.config.hidden_size:
            raise ValueError(
                "The wide-residual Mamba connection must produce "
                f"hidden_size={self.config.hidden_size}, got "
                f"{self.residual_read.branch_hidden_size}."
            )

    def _get_residual_connection(self):
        """Return the connection surrounding the Mamba mixer."""

        return self.residual_read, self.residual_write

    def _prepare_mixer_state(
        self,
        hidden_states: Tensor,
        residual_stream_recompute_context: ResidualStreamRecomputeContext | None = None,
    ) -> tuple[Tensor, Tensor, ResidualConnectionState, ResidualStreamRecomputeContext | None]:
        """Read the wide stream and optionally replay its connected input normalization."""

        recompute_context = residual_stream_recompute_context
        if recompute_context is None:
            hidden_states, connection_state = apply_module(self.residual_read)(
                hidden_states,
                fp32_residual_connection=self.config.fp32_residual_connection,
                branch_input_dtype=self.config.params_dtype,
            )
        else:
            hidden_states, connection_state = checkpoint_residual_read(
                self.residual_read,
                hidden_states,
                recompute_context,
                fp32_residual_connection=self.config.fp32_residual_connection,
                branch_input_dtype=self.config.params_dtype,
            )
        residual = connection_state[0]

        hidden_states = hidden_states.to(dtype=self.config.params_dtype)
        if recompute_context is not None and not isinstance(self.norm, IdentityOp):
            hidden_states = recompute_context.checkpoint(apply_module(self.norm), hidden_states)
        else:
            hidden_states = apply_module(self.norm)(hidden_states)
        return hidden_states, residual, connection_state, recompute_context

    def _apply_mixer_update(
        self,
        mixer_out_with_bias,
        residual: Tensor,
        connection_state: ResidualConnectionState | None = None,
        recompute_context: ResidualStreamRecomputeContext | None = None,
    ) -> Tensor:
        """Write the mixer update, replaying only nonterminal residual writes."""

        if connection_state is None:
            raise RuntimeError("Missing state for the Mamba residual connection.")
        if recompute_context is not None and not recompute_context.is_block_end:
            return checkpoint_residual_write(
                self.residual_write,
                mixer_out_with_bias,
                connection_state,
                recompute_context,
                dropout_probability=self.hidden_dropout,
                training=self.training,
            )
        with self.bias_dropout_add_exec_handler():
            return apply_module(self.residual_write)(
                mixer_out_with_bias,
                state=connection_state,
                dropout_probability=self.hidden_dropout,
                training=self.training,
            )

    def forward_post_core_attn(
        self,
        ssm_output: Tensor,
        residual: Tensor,
        connection_state: ResidualConnectionState,
        *,
        residual_stream_recompute_context: ResidualStreamRecomputeContext | None = None,
    ) -> Tensor:
        """Project the mixer output and write it to the saved wide-residual stream."""

        mixer_out_with_bias = self.mixer.forward_post_core_attn(ssm_output)
        return self._apply_mixer_update(
            mixer_out_with_bias,
            residual,
            connection_state,
            recompute_context=residual_stream_recompute_context,
        )
