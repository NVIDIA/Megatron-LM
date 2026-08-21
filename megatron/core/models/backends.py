# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""The construction-time API for choosing a kernel backend.

A provider answers, once while the model is built, which implementation each operation
should use. Model code asks the provider and never branches on what it got back, so a spec
builder needs no ``HAVE_*`` flag and no import guard of its own.

The implementations themselves live in :mod:`megatron.core.ops`, one package per operation
family. A provider is the short, readable list of which of them this configuration uses --
reading :class:`LocalSpecProvider` tells you every backend a local run gets.
"""

from __future__ import annotations

from typing import Optional, Protocol

from megatron.core.ops.attention import AttentionLocal
from megatron.core.ops.linear import LinearLocal
from megatron.core.ops.loss import LossMegatron, LossMegatronFused, VocabParallelCrossEntropy
from megatron.core.ops.mlp import MlpMegatron
from megatron.core.ops.moe import MoeLocal
from megatron.core.transformer.mlp import TEActivationFunctionBuilder
from megatron.core.transformer.moe.moe_layer import ExpertsBuilder
from megatron.core.transformer.torch_norm import LayerNormBuilder


class BackendSpecProvider(Protocol):
    """A protocol for providing the submodules used in spec building."""

    def linear(self) -> type:
        """Which linear module the backend uses."""
        ...

    def column_parallel_linear(self) -> type:
        """Which column parallel linear module the backend uses."""
        ...

    def row_parallel_linear(self) -> type:
        """Which row parallel linear module the backend uses."""
        ...

    def fuse_layernorm_and_linear(self) -> bool:
        """Does the backend choose a single module for layernorm and linear."""
        ...

    def column_parallel_layer_norm_linear(self) -> Optional[type]:
        """Which module for sequential layernorm and linear."""
        ...

    def layer_norm(
        self, rms_norm: bool = False, for_qk: bool = False, has_residual: bool = False
    ) -> LayerNormBuilder:
        """Which module to use for layer norm."""
        ...

    def core_attention(self) -> type:
        """Which module to use for attention."""
        ...

    def grouped_mlp_modules(self, moe_use_grouped_gemm: bool) -> ExpertsBuilder:
        """Which module and submodules to use for grouped mlp."""
        ...

    def activation_func(self) -> TEActivationFunctionBuilder | None:
        """Which module to use for activation function."""
        ...

    def mlp_module(self, grouped: bool = False) -> type:
        """Which module to use for the dense MLP block."""
        ...

    def moe_router(self) -> Optional[type]:
        """Which MoE router to use, or None to keep the MoESubmodules default."""
        ...

    def vocab_parallel_cross_entropy(self) -> VocabParallelCrossEntropy:
        """Which vocab-parallel cross entropy to use."""
        ...


class LocalSpecProvider(BackendSpecProvider):
    """Every backend a Megatron-Core-only run uses.

    Norm is Torch rather than Apex-if-installed: whether an optional package happens to be
    present must not silently change which kernel a run gets. Apex is a deliberate choice,
    made by passing ``use_apex_norm=True``.
    """

    def __init__(
        self, use_apex_norm: bool = False, cross_entropy_loss_fusion: bool = False
    ) -> None:
        # Imported here so that selecting Apex is what pulls Apex in.
        if use_apex_norm:
            from megatron.core.ops.norm import NormApex

            self._norm = NormApex()
        else:
            from megatron.core.ops.norm import NormTorch

            self._norm = NormTorch()
        self._linear = LinearLocal()
        self._attention = AttentionLocal()
        self._mlp = MlpMegatron()
        self._moe = MoeLocal()
        self._loss = LossMegatronFused() if cross_entropy_loss_fusion else LossMegatron()

    def linear(self) -> type:
        """Megatron Core has no non-parallel linear; Transformer Engine is the one that does."""
        raise NotImplementedError(
            "This backend has no non-parallel linear. Use --transformer-impl transformer_engine."
        )

    def column_parallel_linear(self) -> type:
        """Which column parallel linear module the backend uses."""
        return self._linear.column_parallel_linear()

    def row_parallel_linear(self) -> type:
        """Which row parallel linear module the backend uses."""
        return self._linear.row_parallel_linear()

    def fuse_layernorm_and_linear(self) -> bool:
        """Does the backend choose a single module for layernorm and linear."""
        return False

    def column_parallel_layer_norm_linear(self) -> Optional[type]:
        """Which module for sequential layernorm and linear."""
        return self._linear.column_parallel_layer_norm_linear()

    def layer_norm(
        self, rms_norm: bool = False, for_qk: bool = False, has_residual: bool = False
    ) -> LayerNormBuilder:
        """Which module to use for layer norm."""
        return self._norm.layer_norm(rms_norm=rms_norm, for_qk=for_qk, has_residual=has_residual)

    def core_attention(self) -> type:
        """Which module to use for attention."""
        return self._attention.core_attention()

    def grouped_mlp_modules(self, moe_use_grouped_gemm: bool) -> ExpertsBuilder:
        """Which module and submodules to use for grouped mlp."""
        return self._moe.grouped_mlp_modules(moe_use_grouped_gemm)

    def activation_func(self) -> TEActivationFunctionBuilder | None:
        """Which module to use for activation function."""
        return self._moe.activation_func()

    def mlp_module(self, grouped: bool = False) -> type:
        """Which module to use for the dense MLP block."""
        return self._mlp.mlp_module(grouped=grouped)

    def moe_router(self) -> Optional[type]:
        """Which MoE router to use, or None to keep the MoESubmodules default."""
        return self._moe.moe_router()

    def vocab_parallel_cross_entropy(self) -> VocabParallelCrossEntropy:
        """Which vocab-parallel cross entropy to use."""
        return self._loss.vocab_parallel_cross_entropy()


class InferenceSpecProvider(LocalSpecProvider):
    """Every backend an inference-optimized run uses.

    Attention comes from Transformer Engine; the inference-specific gains are in the linear
    and mixture-of-experts layers.
    """

    def __init__(self, cross_entropy_loss_fusion: bool = False) -> None:
        super().__init__(cross_entropy_loss_fusion=cross_entropy_loss_fusion)
        from megatron.core.inference.ops import LinearInference, MoeInference, NormInference
        from megatron.core.ops.attention import AttentionTE

        self._norm = NormInference()
        self._linear = LinearInference()
        self._attention = AttentionTE()
        self._moe = MoeInference()

    def linear(self) -> type:
        """Which non-parallel linear module the backend uses."""
        return self._linear.linear()

    def fuse_layernorm_and_linear(self) -> bool:
        """The inference linear fuses layernorm into the column-parallel linear."""
        return True


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
    over whichever was chosen. A caller that has a ``TransformerConfig`` should use
    :func:`get_backend_spec_provider` instead, which reads the settings off it.
    """
    base = _base_backend(transformer_impl, **settings)
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


def _base_backend(transformer_impl: str, **settings) -> BackendSpecProvider:
    if transformer_impl == "transformer_engine":
        from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

        return TESpecProvider(**settings)
    if transformer_impl == "inference_optimized":
        settings.pop("use_te_op_fuser", None)
        return InferenceSpecProvider(**settings)
    if transformer_impl == "local":
        settings.pop("use_te_op_fuser", None)
        return LocalSpecProvider(**settings)
    raise ValueError(
        f"unknown transformer_impl='{transformer_impl}'. "
        "Valid choices: local, transformer_engine, inference_optimized"
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
