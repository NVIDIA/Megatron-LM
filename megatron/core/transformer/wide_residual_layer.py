# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
"""Construction-time wide-residual connections and model-boundary helpers."""

from __future__ import annotations

import math
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.residual_connection import (
    ResidualBranchOutput,
    ResidualConnection,
    ResidualConnectionState,
    ResidualConnectionWriteState,
)
from megatron.core.transformer.streamwise_residual_ops import (
    streamwise_sigmoid_read,
    streamwise_sigmoid_writeback,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules

_MIN_CONTROLLER_NUMEL = 128


def _make_padded_vector(values: Tensor) -> nn.Parameter:
    """Create a parameter vector padded for small distributed-optimizer shards."""

    if values.ndim != 1:
        raise ValueError(f"Expected one-dimensional initial values, got {values.shape}.")
    if values.numel() >= _MIN_CONTROLLER_NUMEL:
        return nn.Parameter(values.clone())
    padded = torch.zeros(_MIN_CONTROLLER_NUMEL, dtype=values.dtype)
    padded[: values.numel()] = values
    return nn.Parameter(padded)


def _active_vector(param: Tensor, length: int) -> Tensor:
    """Return the active prefix of a potentially padded vector parameter."""

    return param[:length]


def _mark_residual_map_parameter(param: nn.Parameter, config: TransformerConfig) -> None:
    """Attach optimizer and TP-gradient metadata to a replicated residual map."""

    setattr(param, "is_residual_map_parameter", True)
    setattr(param, "use_muon", False)
    setattr(param, "sequence_parallel", config.sequence_parallel)


def _mark_replicated_tp_parameter(param: nn.Parameter, config: TransformerConfig) -> None:
    """Mark a replicated controller for the correct TP gradient reduction."""

    setattr(param, "allreduce", True)
    setattr(param, "tensor_model_parallel", False)
    setattr(param, "average_gradients_across_tp_domain", not config.sequence_parallel)


def _mark_retention_parameter(param: nn.Parameter, config: TransformerConfig) -> None:
    """Attach optimizer and TP-gradient metadata to a replicated retention parameter."""

    setattr(param, "is_wide_residual_retention_parameter", True)
    setattr(param, "sequence_parallel", config.sequence_parallel)
    _mark_replicated_tp_parameter(param, config)


class LearnedWideResidualRetention(nn.Module):
    """Bounded learned retention with one controller per full-width stream."""

    def __init__(
        self, config: TransformerConfig, layer_number: int, branch_name: str, *, num_streams: int
    ) -> None:
        super().__init__()
        if config.wide_residual is None:
            raise ValueError("LearnedWideResidualRetention requires wide_residual config.")
        wr = config.wide_residual
        if num_streams != wr.num_streams:
            raise ValueError(
                "Retention must define one controller per full-width stream, got "
                f"{num_streams} controllers for {wr.num_streams} streams."
            )
        self.layer_number = layer_number
        self.branch_name = branch_name
        self.num_streams = num_streams
        self.max_forget = wr.retention_max_forget
        forget_ratio = (1.0 - wr.retention_init) / self.max_forget
        initial_logit = math.log((1.0 - forget_ratio) / forget_ratio)
        self.retention_logit = _make_padded_vector(
            torch.full((num_streams,), initial_logit, dtype=torch.float32)
        )
        _mark_retention_parameter(self.retention_logit, config)

    def factors(self) -> Tensor:
        """Return one FP32 retention factor for every full-width stream."""

        logits = _active_vector(self.retention_logit, self.num_streams).float()
        return 1.0 - self.max_forget * torch.sigmoid(-logits)

    def forward(self, *, return_logits: bool = False) -> Tensor:
        """Return padded logits or materialized retention factors."""

        return self.retention_logit if return_logits else self.factors()


class StreamwiseSigmoidMap(nn.Module):
    """Positive scalar controller for each contiguous full-width stream."""

    def __init__(self, config: TransformerConfig, *, map_kind: Literal["read", "write"]) -> None:
        super().__init__()
        if config.wide_residual is None:
            raise ValueError("StreamwiseSigmoidMap requires wide_residual config.")
        wr = config.wide_residual

        self.map_kind = map_kind
        self.num_streams = wr.num_streams

        if map_kind == "read":
            initial_factor = 1.0 / self.num_streams
            initial_logit = math.log(initial_factor / (1.0 - initial_factor))
            initial_logits = torch.full((self.num_streams,), initial_logit, dtype=torch.float32)
        elif map_kind == "write":
            initial_logits = torch.linspace(
                -wr.streamwise_sigmoid_init_scale,
                wr.streamwise_sigmoid_init_scale,
                self.num_streams,
                dtype=torch.float32,
            )
        else:
            raise ValueError(f"Unsupported streamwise sigmoid map kind: {map_kind!r}.")

        self.logit = _make_padded_vector(initial_logits)
        _mark_residual_map_parameter(self.logit, config)
        _mark_replicated_tp_parameter(self.logit, config)

    def factors(self) -> Tensor:
        """Return one positive factor for every full-width residual stream."""

        factors = torch.sigmoid(_active_vector(self.logit, self.num_streams).float())
        if self.map_kind == "write":
            factors = 2.0 * factors
        return factors

    def forward(self, *, return_logits: bool = False) -> Tensor:
        """Return padded logits or one positive factor per full-width stream."""

        return self.logit if return_logits else self.factors()


class StreamwiseSigmoidWideResidualConnection(ResidualConnection):
    """Positive streamwise maps around one ordinary-width residual branch."""

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        branch_name: str,
        pg_collection: ProcessGroupCollection,
        name: str | None = None,
    ) -> None:
        del pg_collection, name
        if config.wide_residual is None:
            raise ValueError(
                "StreamwiseSigmoidWideResidualConnection requires wide_residual config."
            )
        wr = config.wide_residual
        super().__init__(
            residual_stream_hidden_size=wr.num_streams * config.hidden_size,
            branch_hidden_size=config.hidden_size,
        )
        self.layer_number = layer_number
        self.branch_name = branch_name
        self.num_streams = wr.num_streams
        self.read_map = StreamwiseSigmoidMap(config, map_kind="read")
        self.write_map = StreamwiseSigmoidMap(config, map_kind="write")
        self.retention = (
            LearnedWideResidualRetention(
                config, layer_number, branch_name, num_streams=self.num_streams
            )
            if wr.learned_retention
            else None
        )

    def _read(self, hidden_states: Tensor) -> tuple[Tensor, ResidualConnectionWriteState]:
        return (
            streamwise_sigmoid_read(
                hidden_states, self.read_map(return_logits=True), self.num_streams
            ),
            (),
        )

    def _write(
        self,
        branch_output: ResidualBranchOutput,
        state: ResidualConnectionState,
        *,
        dropout_probability: float,
        training: bool,
    ) -> Tensor:
        residual_stream = state[0]
        if isinstance(branch_output, tuple):
            branch_update, bias = branch_output
        else:
            branch_update, bias = branch_output, None

        branch_update = branch_update.to(dtype=residual_stream.dtype)
        if bias is not None:
            branch_update = branch_update + bias.to(
                device=branch_update.device, dtype=branch_update.dtype
            )
        branch_update = F.dropout(branch_update, p=dropout_probability, training=training)
        retention_logits = (
            self.retention(return_logits=True) if self.retention is not None else None
        )
        return streamwise_sigmoid_writeback(
            residual_stream,
            branch_update,
            self.write_map(return_logits=True),
            self.num_streams,
            retention_logits=retention_logits,
            retention_max_forget=(self.retention.max_forget if self.retention is not None else 0.0),
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
        name: str | None = None,
    ) -> None:
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
            name=name,
        )

        if config.wide_residual is None:
            raise ValueError("WideResidualTransformerLayer requires wide_residual config.")
        if not isinstance(self.cross_attention, IdentityOp):
            raise NotImplementedError(
                "WideResidualTransformerLayer does not support cross-attention branches."
            )

        self.residual_connection_self_attn = (
            StreamwiseSigmoidWideResidualConnection(
                config=self.config,
                layer_number=self.layer_number,
                branch_name="self_attention",
                pg_collection=self.pg_collection,
                name=(name + ".residual_connection_self_attn") if name is not None else None,
            )
            if not isinstance(self.self_attention, IdentityOp)
            else None
        )
        self.residual_connection_mlp = (
            StreamwiseSigmoidWideResidualConnection(
                config=self.config,
                layer_number=self.layer_number,
                branch_name="mlp",
                pg_collection=self.pg_collection,
                name=(name + ".residual_connection_mlp") if name is not None else None,
            )
            if not isinstance(self.mlp, IdentityOp)
            else None
        )

        residual_connections = [
            connection
            for connection in (self.residual_connection_self_attn, self.residual_connection_mlp)
            if connection is not None
        ]
        if (
            self.residual_connection_self_attn is not None
            and self._input_layernorm_returns_residual
        ):
            raise ValueError(
                "A self-attention residual connection cannot be combined with a layer norm "
                "that returns its own residual."
            )
        if self.residual_connection_mlp is not None and self._pre_mlp_layernorm_returns_residual:
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

    def _get_self_attention_residual_connection(self) -> ResidualConnection | None:
        """Return the connection surrounding the self-attention branch."""

        return self.residual_connection_self_attn

    def _get_mlp_residual_connection(self) -> ResidualConnection | None:
        """Return the connection surrounding the MLP or MoE branch."""

        return self.residual_connection_mlp


class StreamwiseSigmoidResidualReadout(nn.Module):
    """Learned streamwise readout from D' to D before final normalization."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        if config.wide_residual is None:
            raise ValueError("StreamwiseSigmoidResidualReadout requires wide_residual config.")
        wr = config.wide_residual
        self.input_hidden_size = wr.num_streams * config.hidden_size
        self.output_hidden_size = config.hidden_size
        self.exit_map = StreamwiseSigmoidMap(config, map_kind="read")

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Mix the full-width streams into one backbone-width activation."""

        if hidden_states.shape[-1] != self.input_hidden_size:
            raise ValueError(
                "Streamwise sigmoid readout expected hidden size "
                f"{self.input_hidden_size}, got {hidden_states.shape[-1]}."
            )
        return streamwise_sigmoid_read(
            hidden_states, self.exit_map(return_logits=True), self.exit_map.num_streams
        )


def build_wide_residual_readout(
    config: TransformerConfig,
) -> StreamwiseSigmoidResidualReadout | None:
    """Build the streamwise model-boundary readout when wide residuals are enabled."""

    if config.wide_residual is None:
        return None
    return StreamwiseSigmoidResidualReadout(config)


def expand_wide_residual_stream(hidden_states: Tensor, num_streams: int) -> Tensor:
    """Replicate an ordinary-width activation into contiguous residual streams."""

    if num_streams <= 1:
        raise ValueError(f"num_streams must be greater than one, got {num_streams}.")
    repeat_shape = [1] * hidden_states.ndim
    repeat_shape[-1] = num_streams
    return hidden_states.repeat(*repeat_shape)
