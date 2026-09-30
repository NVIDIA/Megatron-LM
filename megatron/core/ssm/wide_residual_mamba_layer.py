# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Mamba layer specialization for streamwise wide-residual connections."""

from torch import Tensor

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_layer import MambaLayer, MambaLayerSubmodules
from megatron.core.transformer.residual_connection import ResidualConnectionState
from megatron.core.transformer.residual_recompute import ResidualStreamRecomputeContext
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.wide_residual_layer import StreamwiseSigmoidWideResidualConnection


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

        self.residual_connection = StreamwiseSigmoidWideResidualConnection(
            config=self.config,
            layer_number=self.layer_number,
            branch_name="mamba",
            pg_collection=pg_collection,
            name=(name + ".residual_connection") if name is not None else None,
        )
        self.residual_stream_hidden_size = (
            config.wide_residual.num_streams * self.config.hidden_size
        )
        if self.residual_connection.residual_stream_hidden_size != self.residual_stream_hidden_size:
            raise ValueError(
                "The wide-residual Mamba connection must carry num_streams * hidden_size "
                f"features, expected {self.residual_stream_hidden_size}, got "
                f"{self.residual_connection.residual_stream_hidden_size}."
            )
        if self.residual_connection.branch_hidden_size != self.config.hidden_size:
            raise ValueError(
                "The wide-residual Mamba connection must produce "
                f"hidden_size={self.config.hidden_size}, got "
                f"{self.residual_connection.branch_hidden_size}."
            )

    def _get_residual_connection(self):
        """Return the connection surrounding the Mamba mixer."""

        return self.residual_connection

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
        return self._apply_mixer_bda(
            mixer_out_with_bias,
            residual,
            connection_state,
            recompute_context=residual_stream_recompute_context,
        )
