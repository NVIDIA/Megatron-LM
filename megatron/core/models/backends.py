# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

import warnings
from abc import abstractmethod
from functools import partial
from typing import Optional, Protocol, cast

from typing_extensions import final, override

from megatron.core.extensions.transformer_engine import (
    TEColumnParallelGroupedLinear,
    TERowParallelGroupedLinear,
)
from megatron.core.models.protocols import (
    ColumnParallelLinearBuilder,
    LinearBuilder,
    RowParallelLinearBuilder,
)
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.mlp import MLPSubmodules, TEActivationFunctionBuilder
from megatron.core.transformer.moe.experts import (
    GroupedMLPSubmodules,
    InferenceGroupedMLP,
    SequentialMLP,
)
from megatron.core.transformer.moe.moe_layer import ExpertsBuilder
from megatron.core.transformer.torch_norm import LayerNormBuilder, WrappedTorchNorm
from megatron.core.typed_torch import not_none
from megatron.core.utils import is_te_min_version

try:
    import apex  # pylint: disable=unused-import

    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

    HAVE_APEX = True
    LNImpl = FusedLayerNorm
except ImportError:
    warnings.warn("Apex is not installed. Falling back to Torch Norm")
    FusedLayerNorm = None
    HAVE_APEX = False
    LNImpl = WrappedTorchNorm

from megatron.core.extensions.transformer_engine import (
    TEActivationOp,
    TEDotProductAttention,
    TELinear,
    TENorm,
)
from megatron.core.tensor_parallel.inference_layers import (
    InferenceColumnParallelLinear,
    InferenceLayerNormColumnParallelLinear,
    InferenceRowParallelLinear,
)
from megatron.core.utils import is_te_min_version


class BackendSpecProvider(Protocol):
    """A protocol for providing the submodules used in Spec building."""

    @abstractmethod
    def linear(self) -> LinearBuilder:
        """Which linear module the backend uses"""
        ...

    @abstractmethod
    def column_parallel_linear(self) -> ColumnParallelLinearBuilder:
        """Which column parallel linear module the backend uses"""
        ...

    @abstractmethod
    def row_parallel_linear(self) -> RowParallelLinearBuilder:
        """Which row parallel linear module the backend uses"""
        ...

    @abstractmethod
    def fuse_layernorm_and_linear(self) -> bool:
        """Does the backend support a single module for layernorm and linear"""
        ...

    @abstractmethod
    def column_parallel_layer_norm_linear(self) -> Optional[type]:
        """Which module for sequential layernorm and linear"""
        ...

    @abstractmethod
    def layer_norm(
        self, rms_norm: bool = False, for_qk: bool = False, has_residual: bool = False
    ) -> LayerNormBuilder:
        """Which module for layernorm"""
        ...

    @abstractmethod
    def core_attention(self) -> type:
        """Which module to use for attention"""
        ...

    @abstractmethod
    def grouped_mlp_modules(self, moe_use_grouped_gemm: bool) -> ExpertsBuilder:
        """Which module and submodules to use for grouped mlp"""
        ...

    @abstractmethod
    def activation_func(self) -> TEActivationFunctionBuilder | None:
        """Which module to use for activation function"""
        ...


@final
class LocalSpecProvider(BackendSpecProvider):
    """A protocol for providing Local submodules used in Spec building."""

    @override
    def linear(self) -> LinearBuilder:
        """Which linear module the backend uses"""
        raise NotImplementedError("LocalSpecProvider does not have a linear module")

    @override
    def column_parallel_linear(self) -> ColumnParallelLinearBuilder:
        """Which column parallel linear module the backend uses"""
        return ColumnParallelLinear

    @override
    def row_parallel_linear(self) -> RowParallelLinearBuilder:
        """Which row parallel linear module the backend uses"""
        return RowParallelLinear

    @override
    def fuse_layernorm_and_linear(self) -> bool:
        """Does the backend choose a single module for layernorm and linear"""
        return False

    @override
    def column_parallel_layer_norm_linear(self) -> Optional[type]:
        """Which module for sequential layernorm and linear"""
        return None

    @override
    def layer_norm(
        self, rms_norm: bool = False, for_qk: bool = False, has_residual: bool = False
    ) -> LayerNormBuilder:
        """Which module to use for layer norm"""
        if rms_norm:
            # Matching get_gpt_layer_local_spec.
            # Why does the global need to be updated?
            global LNImpl
            LNImpl = WrappedTorchNorm
        return LNImpl

    @override
    def core_attention(self) -> type:
        """Which module to use for attention"""
        return DotProductAttention

    @override
    def grouped_mlp_modules(self, moe_use_grouped_gemm: bool) -> ExpertsBuilder:
        """Which module and submodules to use for grouped mlp"""
        return partial(
            SequentialMLP,
            submodules=MLPSubmodules(
                linear_fc1=ColumnParallelLinear,
                linear_fc2=RowParallelLinear,
                activation_func=self.activation_func(),
            ),
        )

    @override
    def activation_func(self) -> TEActivationFunctionBuilder | None:
        """Which module to use for activation function"""
        return None


@final
class InferenceSpecProvider(BackendSpecProvider):
    """A protocol for providing the submodules used in Spec building."""

    @override
    def linear(self) -> LinearBuilder:
        """Which linear module TE backend uses"""
        return TELinear

    @override
    def column_parallel_linear(self) -> ColumnParallelLinearBuilder:
        """Which column parallel linear module TE backend uses"""
        return InferenceColumnParallelLinear

    @override
    def row_parallel_linear(self) -> RowParallelLinearBuilder:
        """Which row parallel linear module TE backend uses"""
        return InferenceRowParallelLinear

    @override
    def fuse_layernorm_and_linear(self) -> bool:
        """TE backend chooses a single module for layernorm and linear"""
        return True

    @override
    def column_parallel_layer_norm_linear(self) -> type[InferenceLayerNormColumnParallelLinear]:
        """Which module for sequential layernorm and linear"""
        return InferenceLayerNormColumnParallelLinear

    @override
    def layer_norm(
        self, rms_norm: bool = False, for_qk: bool = False, has_residual: bool = False
    ) -> LayerNormBuilder:
        """Which module to use for layer norm"""
        if for_qk and not is_te_min_version("1.9.0"):
            # TENorm significantly harms convergence when used
            # for QKLayerNorm if TE Version < 1.9;
            # we instead use the Apex implementation.
            return not_none(FusedLayerNorm)
        return TENorm

    @override
    def core_attention(self) -> type[TEDotProductAttention]:
        """Which module to use for attention"""
        return TEDotProductAttention

    @override
    def activation_func(self) -> TEActivationFunctionBuilder | None:
        """Which module to use for activation function"""
        # transformer_engine.BasicOperation.forward has an overly permissive return type, but by
        # design these classes always meet the interface.
        return cast(TEActivationFunctionBuilder, TEActivationOp)

    @override
    def grouped_mlp_modules(self, moe_use_grouped_gemm: bool) -> ExpertsBuilder:
        """Which module and submodules to use for grouped mlp"""
        return partial(
            InferenceGroupedMLP,
            submodules=GroupedMLPSubmodules(
                linear_fc1=TEColumnParallelGroupedLinear,
                linear_fc2=TERowParallelGroupedLinear,
                activation_func=self.activation_func(),
            ),
        )
