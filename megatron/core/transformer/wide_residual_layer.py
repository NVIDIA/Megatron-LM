# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
"""Compatibility exports for the wide-residual implementation modules."""

# pylint: disable=unused-import

from megatron.core.transformer.wide_residual import _MIN_CONTROLLER_NUMEL as _MIN_CONTROLLER_NUMEL
from megatron.core.transformer.wide_residual import (
    LearnedWideResidualRetention as LearnedWideResidualRetention,
)
from megatron.core.transformer.wide_residual import StreamwiseSigmoidMap as StreamwiseSigmoidMap
from megatron.core.transformer.wide_residual import (
    StreamwiseSigmoidResidualReadout as StreamwiseSigmoidResidualReadout,
)
from megatron.core.transformer.wide_residual import (
    StreamwiseSigmoidWideResidualRead as StreamwiseSigmoidWideResidualRead,
)
from megatron.core.transformer.wide_residual import _active_vector as _active_vector
from megatron.core.transformer.wide_residual import _make_padded_vector as _make_padded_vector
from megatron.core.transformer.wide_residual import (
    _mark_replicated_tp_parameter as _mark_replicated_tp_parameter,
)
from megatron.core.transformer.wide_residual import (
    _mark_residual_map_parameter as _mark_residual_map_parameter,
)
from megatron.core.transformer.wide_residual import (
    _mark_retention_parameter as _mark_retention_parameter,
)
from megatron.core.transformer.wide_residual import (
    build_wide_residual_readout as build_wide_residual_readout,
)
from megatron.core.transformer.wide_residual import (
    expand_wide_residual_stream as expand_wide_residual_stream,
)
from megatron.core.transformer.wide_residual_policy import (
    _RESIDUAL_STATE_PREFIXES as _RESIDUAL_STATE_PREFIXES,
)
from megatron.core.transformer.wide_residual_policy import (
    StreamwiseSigmoidWideResidualConnection as StreamwiseSigmoidWideResidualConnection,
)
from megatron.core.transformer.wide_residual_policy import (
    load_residual_state as load_residual_state,
)
from megatron.core.transformer.wide_residual_policy import (
    wide_residual_recompute_enabled as wide_residual_recompute_enabled,
)
from megatron.core.transformer.wide_residual_transformer_layer import (
    WideResidualTransformerLayer as WideResidualTransformerLayer,
)
