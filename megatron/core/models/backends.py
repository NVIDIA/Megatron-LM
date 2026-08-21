# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

from abc import abstractmethod
from functools import partial
from typing import Callable, Literal, Optional, Protocol, cast

import torch

from megatron.core.extensions.transformer_engine import (
    TEActivationOp,
    TEColumnParallelGroupedLinear,
    TEDotProductAttention,
    TELinear,
    TENorm,
    TERowParallelGroupedLinear,
)
from megatron.core.fusions.fused_cross_entropy import (
    fused_vocab_parallel_cross_entropy as _fused_ce,
)
from megatron.core.tensor_parallel.cross_entropy import (
    vocab_parallel_cross_entropy as _reference_ce,
)
from megatron.core.tensor_parallel.inference_layers import (
    InferenceColumnParallelLinear,
    InferenceLayerNormColumnParallelLinear,
    InferenceRowParallelLinear,
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

VocabParallelCrossEntropy = Callable[
    [torch.Tensor, torch.Tensor, Optional[torch.distributed.ProcessGroup]], torch.Tensor
]
"""``loss = target(logits, labels, tp_group)``, with ``logits`` [s, b, vocab/tp] and the
target owning every reduction over ``tp_group``. Targets *consume* ``logits``: they subtract
the row max and exponentiate in place, so a caller that needs them afterwards passes a clone.
"""


def _default_tp_group(tp_group):
    """Fill in the default tensor-parallel group for kernels that cannot take ``None``."""
    if tp_group is not None:
        return tp_group
    from megatron.core.parallel_state import get_tensor_model_parallel_group

    return get_tensor_model_parallel_group()


def vocab_parallel_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """Megatron's custom-autograd cross entropy, in provider-contract argument order."""
    return _reference_ce(logits, labels, tp_group=tp_group)


def fused_vocab_parallel_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """Megatron's compiled cross entropy, in provider-contract argument order.

    The kernel reads rank and size straight off the group, so unlike the reference target it
    cannot take ``None``. Filling that in is this adapter's job, not the caller's.
    """
    return _fused_ce(logits, labels, _default_tp_group(tp_group))


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

    @abstractmethod
    def mlp_module(self, grouped: bool = False) -> type:
        """Which module to use for the dense MLP block."""

    @abstractmethod
    def moe_router(self) -> Optional[type]:
        """Which MoE router to use, or None to keep the MoESubmodules default."""

    @abstractmethod
    def vocab_parallel_cross_entropy(self) -> VocabParallelCrossEntropy:
        """Which vocab-parallel cross entropy to use."""


class LocalSpecProvider(BackendSpecProvider):
    """Every backend a Megatron-Core-only run uses."""

    def __init__(self, cross_entropy_loss_fusion: bool = False) -> None:
        self._cross_entropy_loss_fusion = cross_entropy_loss_fusion

    def column_parallel_linear(self) -> type:
        """Which column parallel linear module the backend uses"""
        return ColumnParallelLinear

    def row_parallel_linear(self) -> type[RowParallelLinear]:
        """Which row parallel linear module the backend uses"""
        return RowParallelLinear

    def fuse_layernorm_and_linear(self) -> bool:
        """Does the backend choose a single module for layernorm and linear"""
        return False

    def column_parallel_layer_norm_linear(self) -> Optional[type]:
        """Which module for sequential layernorm and linear"""
        return None

    def layer_norm(
        self, rms_norm: bool = False, for_qk: bool = False, has_residual: bool = False
    ) -> LayerNormBuilder:
        """Which module to use for layer norm.

        Torch, not Apex-if-installed: whether an optional package happens to be present must
        not change which kernel a run gets. WrappedTorchNorm reads ``config.normalization``,
        so one target covers both LayerNorm and RMSNorm.
        """
        del rms_norm, for_qk, has_residual
        return WrappedTorchNorm

    def core_attention(self) -> type:
        """Which module to use for attention"""
        return DotProductAttention

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

    def activation_func(self) -> TEActivationFunctionBuilder | None:
        """Which module to use for activation function"""
        return None

    def linear(self) -> type:
        """Megatron Core has no non-parallel linear; Transformer Engine is the one that does."""
        raise NotImplementedError(
            "This backend has no non-parallel linear. Use --transformer-impl transformer_engine."
        )

    def mlp_module(self, grouped: bool = False) -> type:
        """Megatron Core has one MLP block; grouped GEMM is a Transformer Engine feature."""
        del grouped
        from megatron.core.transformer.mlp import MLP

        return MLP

    def moe_router(self) -> Optional[type]:
        """Keep the MoESubmodules default."""
        return None

    def vocab_parallel_cross_entropy(self) -> VocabParallelCrossEntropy:
        """Which vocab-parallel cross entropy to use."""
        if self._cross_entropy_loss_fusion:
            return fused_vocab_parallel_cross_entropy
        return vocab_parallel_cross_entropy


class InferenceSpecProvider(LocalSpecProvider):
    """Every backend an inference-optimized run uses."""

    def linear(self) -> type:
        """Which linear module TE backend uses"""
        return TELinear

    def column_parallel_linear(self) -> type:
        """Which column parallel linear module TE backend uses"""
        return InferenceColumnParallelLinear

    def row_parallel_linear(self) -> type[InferenceRowParallelLinear]:
        """Which row parallel linear module Inference backend uses"""
        return InferenceRowParallelLinear

    def fuse_layernorm_and_linear(self) -> bool:
        """TE backend chooses a single module for layernorm and linear"""
        return True

    def column_parallel_layer_norm_linear(self) -> type[InferenceLayerNormColumnParallelLinear]:
        """Which module for sequential layernorm and linear"""
        return InferenceLayerNormColumnParallelLinear

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

    def core_attention(self) -> type[TEDotProductAttention]:
        """Which module to use for attention"""
        return TEDotProductAttention

    def activation_func(self) -> TEActivationFunctionBuilder | None:
        """Which module to use for activation function"""
        # transformer_engine.BasicOperation.forward has an overly permissive return type, but by
        # design these classes always meet the interface.
        return cast(TEActivationFunctionBuilder, TEActivationOp)

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


def get_backend(
    transformer_impl: Literal["local", "transformer_engine", "inference_optimized"],
) -> BackendSpecProvider:
    """Return the backend that's selected with the given `transformer_impl`."""
    if transformer_impl == "transformer_engine":
        from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

        return TESpecProvider()
    elif transformer_impl == "inference_optimized":
        return InferenceSpecProvider()
    elif transformer_impl == "local":
        return LocalSpecProvider()
    else:
        raise ValueError(f"unknown transformer_impl='{transformer_impl}'")

    def moe_router(self) -> Optional[type]:
        """Inference needs compact [tokens, topk] index routing."""
        from megatron.core.transformer.moe.router import InferenceTopKRouter

        return InferenceTopKRouter


def get_backend(
    transformer_impl: str,
    *,
    use_kitchen: bool = False,
    use_kitchen_attention: bool = False,
    kitchen_attention_backend: str = "sdpa",
    **settings,
) -> BackendSpecProvider:
    """Build the provider for a named backend.

    The whole of backend selection is here: three names, three providers, plus Kitchen layered
    over whichever was chosen.
    """
    if transformer_impl == "transformer_engine":
        from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

        base: BackendSpecProvider = TESpecProvider(**settings)
    elif transformer_impl == "inference_optimized":
        settings.pop("use_te_op_fuser", None)
        base = InferenceSpecProvider(**settings)
    elif transformer_impl == "local":
        settings.pop("use_te_op_fuser", None)
        base = LocalSpecProvider(**settings)
    else:
        raise ValueError(
            f"unknown transformer_impl='{transformer_impl}'. "
            "Valid choices: local, transformer_engine, inference_optimized"
        )

    if not use_kitchen:
        return base
    # Kitchen takes the operations it implements and forwards the rest to the provider that
    # would otherwise have owned them.
    from megatron.core.extensions.kitchen import KitchenSpecProvider

    return KitchenSpecProvider(
        fallback=base,
        use_kitchen_attention=use_kitchen_attention,
        kitchen_attention_backend=kitchen_attention_backend,
    )


def get_backend_spec_provider(
    config: object, *, transformer_impl: Optional[str] = None
) -> BackendSpecProvider:
    """Build the provider a TransformerConfig asks for.

    ``transformer_impl`` overrides ``config.transformer_impl``, for the callers that build a
    Transformer Engine spec regardless of what the config says.
    """
    return get_backend(
        transformer_impl or getattr(config, "transformer_impl", "local"),
        use_kitchen=getattr(config, "use_kitchen", False),
        use_kitchen_attention=getattr(config, "use_kitchen_attention", False),
        kitchen_attention_backend=getattr(config, "kitchen_attention_backend", "sdpa"),
        use_te_op_fuser=getattr(config, "use_transformer_engine_op_fuser", False),
        cross_entropy_loss_fusion=getattr(config, "cross_entropy_loss_fusion", False),
    )
