# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
"""Construction-time wide-residual connections and model-boundary helpers."""

from __future__ import annotations

import copy
import functools
import math
from typing import Any, Literal

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
from megatron.core.transformer.spec_utils import ModuleSpec, get_module
from megatron.core.transformer.streamwise_residual_ops import (
    streamwise_sigmoid_read,
    streamwise_sigmoid_writeback,
)
from megatron.core.transformer.transformer_config import TransformerConfig

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


def _resolve_module(spec_or_builder: Any) -> Any:
    """Resolve ModuleSpec and partial wrappers without constructing a module."""

    while isinstance(spec_or_builder, functools.partial):
        spec_or_builder = spec_or_builder.func
    if isinstance(spec_or_builder, ModuleSpec):
        return get_module(spec_or_builder)
    return spec_or_builder


def _is_identity_spec(spec_or_builder: Any) -> bool:
    module = _resolve_module(spec_or_builder)
    return isinstance(module, type) and issubclass(module, IdentityOp)


def _is_moe_builder(spec_or_builder: Any) -> bool:
    from megatron.core.transformer.moe.moe_layer import MoELayer

    module = _resolve_module(spec_or_builder)
    return isinstance(module, type) and issubclass(module, MoELayer)


def specialize_wide_residual_layer_spec(
    layer_spec: ModuleSpec, config: TransformerConfig
) -> ModuleSpec:
    """Return a copied layer spec with explicit wide-residual branch connections.

    - Self-attention, Mamba, and dense MLP branches receive a D'->D->D' connection.
    - MoE branches receive the same outer connection, so the existing router,
      routed experts, and optional shared experts all continue to operate at D.
    """

    if config.wide_residual is None:
        return layer_spec
    if not isinstance(layer_spec, ModuleSpec):
        raise TypeError(
            "wide_residual requires ModuleSpec layer definitions so residual connections "
            "can be installed before module construction."
        )

    specialized = copy.deepcopy(layer_spec)
    layer_module = _resolve_module(specialized)
    connection_spec = ModuleSpec(module=StreamwiseSigmoidWideResidualConnection)

    from megatron.core.ssm.mamba_layer import MambaLayer, MambaLayerSubmodules

    if isinstance(layer_module, type) and issubclass(layer_module, MambaLayer):
        if not isinstance(specialized.submodules, MambaLayerSubmodules):
            raise TypeError("Wide-residual Mamba specs require MambaLayerSubmodules.")
        specialized.submodules.residual_connection = copy.deepcopy(connection_spec)
        return specialized

    from megatron.core.transformer.transformer_layer import (
        MoETransformerLayer,
        TransformerLayer,
        TransformerLayerSubmodules,
    )

    if not (isinstance(layer_module, type) and issubclass(layer_module, TransformerLayer)):
        raise TypeError(
            "wide_residual submodule connections currently support TransformerLayer specs, "
            f"got {layer_module!r}."
        )
    if not isinstance(specialized.submodules, TransformerLayerSubmodules):
        raise TypeError("Wide-residual Transformer specs require TransformerLayerSubmodules.")

    submodules = specialized.submodules
    has_self_attention = not _is_identity_spec(submodules.self_attention)
    has_cross_attention = not _is_identity_spec(submodules.cross_attention)
    if has_cross_attention:
        raise NotImplementedError("wide_residual does not yet support cross-attention branches.")
    if has_self_attention:
        submodules.residual_connection_self_attn = copy.deepcopy(connection_spec)

    is_moe_layer = issubclass(layer_module, MoETransformerLayer)
    is_moe_builder = _is_moe_builder(submodules.mlp)
    if is_moe_layer and not is_moe_builder:
        raise TypeError(
            "Wide-residual MoETransformerLayer specs must provide a MoELayer MLP builder."
        )
    if not _is_identity_spec(submodules.mlp):
        submodules.residual_connection_mlp = copy.deepcopy(connection_spec)
    return specialized
