# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import warnings
from abc import abstractmethod
from typing import Optional, Protocol, Tuple

from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP
from megatron.core.transformer.torch_norm import WrappedTorchNorm

try:
    import apex  # pylint: disable=unused-import

    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

    HAVE_APEX = True
    LNImpl = FusedLayerNorm
except ImportError:

    warnings.warn('Apex is not installed. Falling back to Torch Norm')
    LNImpl = WrappedTorchNorm


class BackendSpecProvider(Protocol):
    """A protocol for providing the submodules used in Spec building."""

    @abstractmethod
    def column_parallel_linear(self) -> type:
        """Which column parallel linear module the backend uses"""
        ...

    @abstractmethod
    def row_parallel_linear(self) -> type:
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
    def layer_norm(self, rms_norm: bool = False, for_qk: bool = False) -> type:
        """Which module for layernorm"""
        ...

    @abstractmethod
    def core_attention(self) -> type:
        """Which module to use for attention"""
        ...

    @abstractmethod
    def grouped_mlp_modules(
        self, moe_use_grouped_gemm: bool, moe_use_legacy_grouped_gemm: bool
    ) -> Tuple[type, Optional[MLPSubmodules]]:
        """Which module and submodules to use for grouped mlp"""
        ...


class LocalSpecProvider(BackendSpecProvider):
    """A protocol for providing Local submodules used in Spec building."""

    def column_parallel_linear(self) -> type:
        """Which column parallel linear module the backend uses"""
        return ColumnParallelLinear

    def row_parallel_linear(self) -> type:
        """Which row parallel linear module the backend uses"""
        return RowParallelLinear

    def fuse_layernorm_and_linear(self) -> bool:
        """Does the backend choose a single module for layernorm and linear"""
        return False

    def column_parallel_layer_norm_linear(self) -> Optional[type]:
        """Which module for sequential layernorm and linear"""
        return None

    def layer_norm(self, rms_norm: bool = False, for_qk: bool = False) -> type:
        """Which module to use for layer norm"""
        if rms_norm:
            # Matching get_gpt_layer_local_spec.
            # Why does the global need to be updated?
            global LNImpl
            LNImpl = WrappedTorchNorm
        return LNImpl

    def core_attention(self) -> type:
        """Which module to use for attention"""
        return DotProductAttention

    def grouped_mlp_modules(
        self, moe_use_grouped_gemm: bool, moe_use_legacy_grouped_gemm: bool
    ) -> Tuple[type, Optional[MLPSubmodules]]:
        """Which module and submodules to use for grouped mlp"""
        if moe_use_grouped_gemm:
            warnings.warn(
                'The legacy GroupedMLP will be deprecated in Megatron-Core v0.12.0. '
                'Please update the TransformerEngine to version>=1.7.0 and use TEGroupedMLP.'
            )
            return GroupedMLP, None
        else:
            return SequentialMLP, MLPSubmodules(
                linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear
            )
