# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import warnings
from functools import partial
from typing import Optional, cast

from megatron.core.extensions.transformer_engine import (
    TEActivationOp,
    TEColumnParallelGroupedLinear,
    TEColumnParallelLinear,
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TELinear,
    TENorm,
    TERowParallelGroupedLinear,
    TERowParallelLinear,
)
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.models.backends import (
    BackendSpecProvider,
    CrossEntropyTarget,
    select_cross_entropy,
)
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.mlp import MLPSubmodules, TEActivationFunctionBuilder
from megatron.core.transformer.moe.experts import GroupedMLPSubmodules, SequentialMLP, TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import ExpertsBuilder
from megatron.core.transformer.torch_norm import LayerNormBuilder
from megatron.core.utils import get_te_version, is_te_min_version


class _TENormWithResidual:
    """Class adapter for TENorm with residual fusion enabled."""

    def __new__(cls, *args, **kwargs):
        return TENorm(*args, has_residual=True, **kwargs)


class TESpecProvider(BackendSpecProvider):
    """A protocol for providing the submodules used in Spec building."""

    #: The optional package this backend needs. Declared here so there is one place to read
    #: it from, and checked by ``require`` at the point a caller wants an early, clear
    #: refusal. It is deliberately *not* checked when the provider is built: a spec may be
    #: assembled without Transformer Engine installed -- several module-level specs are, at
    #: import time -- and it is instantiating a TE module that fails.
    REQUIRES = "transformer_engine"

    def __init__(
        self,
        use_te_op_fuser: bool = False,
        cross_entropy_loss_fusion: bool = False,
        cross_entropy_fusion_impl: str = "native",
        cuda_graph_impl: Optional[str] = None,
    ) -> None:
        self._use_te_op_fuser = use_te_op_fuser
        self._cross_entropy_loss_fusion = cross_entropy_loss_fusion
        self._cross_entropy_fusion_impl = cross_entropy_fusion_impl
        self._cuda_graph_impl = cuda_graph_impl

    def linear(self) -> type:
        """Which linear module TE backend uses"""
        return TELinear

    def column_parallel_linear(self) -> type:
        """Which column parallel linear module TE backend uses"""
        return TEColumnParallelLinear

    def row_parallel_linear(self) -> type[TERowParallelLinear]:
        """Which row parallel linear module TE backend uses"""
        return TERowParallelLinear

    def fuse_layernorm_and_linear(self) -> bool:
        """TE backend chooses a single module for layernorm and linear"""
        return True

    def column_parallel_layer_norm_linear(self) -> Optional[type]:
        """Which module for sequential layernorm and linear"""
        return TELayerNormColumnParallelLinear

    def layer_norm(
        self, rms_norm: bool = False, for_qk: bool = False, has_residual: bool = False
    ) -> LayerNormBuilder:
        """Which module to use for layer norm"""
        if for_qk and not is_te_min_version("1.9.0"):
            # TENorm significantly harms convergence when used
            # for QKLayerNorm if TE Version < 1.9;
            # we instead use the Apex implementation.
            return FusedLayerNorm
        # Keep returning a class so this path stays aligned with build_module's class handling.
        return _TENormWithResidual if has_residual else TENorm

    def core_attention(self) -> type:
        """Which module to use for attention"""
        return TEDotProductAttention

    def grouped_mlp_modules(self, moe_use_grouped_gemm: bool) -> ExpertsBuilder:
        """Which module and submodules to use for grouped mlp"""
        if moe_use_grouped_gemm and TEColumnParallelGroupedLinear is not None:
            return partial(
                TEGroupedMLP,
                submodules=GroupedMLPSubmodules(
                    linear_fc1=TEColumnParallelGroupedLinear,
                    linear_fc2=TERowParallelGroupedLinear,
                    activation_func=self.activation_func(),
                ),
            )
        else:
            if not is_te_min_version("1.7.0.dev0"):
                warnings.warn(
                    "Only transformer-engine>=1.7.0 supports MoE experts, "
                    f"but your version is {get_te_version()}. "
                    "Use local linear implementation instead."
                )
                return partial(
                    SequentialMLP,
                    submodules=MLPSubmodules(
                        linear_fc1=ColumnParallelLinear,
                        linear_fc2=RowParallelLinear,
                        activation_func=self.activation_func(),
                    ),
                )
            return partial(
                SequentialMLP,
                submodules=MLPSubmodules(
                    linear_fc1=TEColumnParallelLinear,
                    linear_fc2=TERowParallelLinear,
                    activation_func=self.activation_func(),
                ),
            )

    def activation_func(self) -> TEActivationFunctionBuilder | None:
        """Which module to use for activation function"""
        # transformer_engine.BasicOperation.forward has an overly permissive return type, but by
        # design these classes always meet the interface.
        return cast(TEActivationFunctionBuilder, TEActivationOp)

    def mlp_module(self, grouped: bool = False) -> type:
        """The dense MLP block, fused into TE operations when the fuser is on."""
        if not self._use_te_op_fuser:
            from megatron.core.transformer.mlp import MLP

            return MLP
        from megatron.core.extensions.transformer_engine import (
            TEFusedMLP,
            TEFusedMLPWithGroupedLinear,
        )

        target = TEFusedMLPWithGroupedLinear if grouped else TEFusedMLP
        if target is None:
            raise ImportError(
                "Transformer Engine is installed but does not expose the operation-fused MLP."
            )
        return target

    def moe_router(self) -> Optional[type]:
        """Keep the MoESubmodules default."""
        return None

    def vocab_parallel_cross_entropy(self) -> CrossEntropyTarget:
        """Which vocab-parallel cross entropy to use.

        Which one depends on the config rather than on this being the TE provider, so the
        same settings give the same kernel whichever backend supplies the rest of the model.
        ``megatron/training/arguments.py`` rejects ``cross_entropy_fusion_impl='te'``, but a
        config built directly can still ask for it.
        """
        return select_cross_entropy(
            self._cross_entropy_loss_fusion, self._cross_entropy_fusion_impl, self._cuda_graph_impl
        )
