# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.

"""Common (model-wide) configuration for the HybridModel Python DSL.

:class:`CommonLayerConfig` is a focused dataclass exposing the fields that
typical hybrid-model authors actually customise per-model. It is *not* a
subclass of :class:`TransformerConfig`; instead, it normalises a small set of
human-friendly fields (e.g. ``mixed_precision_dtype="bf16"`` rather than the
``bf16: bool`` / ``fp16: bool`` pair) and converts to a full
:class:`TransformerConfig` only at layer-construction time.

All fields here are *defaults* shared by at least two decoder layer families
in the HybridModel stack. Per-layer :class:`LayerConfig` instances may
override or augment via either:

- :meth:`update`: a sibling :class:`CommonLayerConfig` produced by
  :func:`dataclasses.replace`. Use this for layer-class-wide overrides
  (e.g. an MoE-specific common with a different ``ffn_hidden_size``).
- per-layer fields on the layer-specific config (e.g.
  :class:`MambaLayerConfig.head_dim`).

Anything derivable from the model architecture (``num_layers`` from pattern
length, ``num_query_groups`` from ``num_attention_heads``,
``kv_channels`` from ``hidden_size // num_attention_heads``,
``mamba_num_heads`` from ``hidden_size * expand // mamba_head_dim``) does
**not** appear here — it is computed at compile time.
"""

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import torch

from megatron.core.activations import squared_relu
from megatron.core.utils import init_method_normal

# --- string-keyed lookups for human-friendly recipe authoring -------------

_DTYPE_BY_NAME: Dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}


_ACTIVATION_BY_NAME: Dict[str, Callable] = {
    "gelu": torch.nn.functional.gelu,
    "relu": torch.nn.functional.relu,
    "silu": torch.nn.functional.silu,
    "swish": torch.nn.functional.silu,
    "swiglu": torch.nn.functional.silu,  # SwiGLU = SiLU + GLU; gated_linear_unit=True separately
    "squared_relu": squared_relu,
}


def _resolve_dtype(name_or_dtype: Any) -> torch.dtype:
    if isinstance(name_or_dtype, torch.dtype):
        return name_or_dtype
    try:
        return _DTYPE_BY_NAME[name_or_dtype]
    except KeyError as e:
        raise ValueError(
            f"Unknown dtype name {name_or_dtype!r}; expected one of "
            f"{sorted(_DTYPE_BY_NAME)}."
        ) from e


def _resolve_activation(name_or_callable: Any) -> Callable:
    if callable(name_or_callable):
        return name_or_callable
    try:
        return _ACTIVATION_BY_NAME[name_or_callable]
    except KeyError as e:
        raise ValueError(
            f"Unknown activation {name_or_callable!r}; expected a callable or "
            f"one of {sorted(_ACTIVATION_BY_NAME)}."
        ) from e


# --- the config itself ----------------------------------------------------


@dataclass
class CommonLayerConfig:
    """Model-wide defaults shared across every layer in a HybridModel recipe."""

    # === Architecture ===
    hidden_size: int = 0
    """Layer hidden size. Required (must be set by the recipe)."""

    ffn_hidden_size: Optional[int] = None
    """MLP intermediate size. If None, ``TransformerConfig`` derives ``4 * hidden_size``."""

    # === Precision ===
    mixed_precision_dtype: str = "fp32"
    """Mixed-precision compute dtype. One of ``"fp32"``, ``"fp16"``, ``"bf16"``."""

    params_dtype: str = "fp32"
    """Parameter storage dtype."""

    first_last_layers_bf16: bool = False
    """Retain the first and last N TransformerBlocks in BF16 instead of FP8."""

    # === Parallelism (layer-level only) ===
    # Model-level parallelism sizes (TP/PP/CP/EP/ETP) live on
    # :class:`HybridModelConfig`, not here — they're job-/model-level
    # concerns and can't actually vary per-layer in the current
    # implementation. ``sequence_parallel`` stays here because it is a
    # per-layer behaviour toggle for layer norms / dropout.
    sequence_parallel: bool = False
    """Enable sequence-parallel layer norms / dropout (paired with TP)."""

    # === Initialisation ===
    init_method_std: float = 0.02
    """Standard deviation for the default normal-init method."""

    perform_initialization: bool = True
    """If False, skip weight init at construction (useful when loading checkpoints)."""

    use_cpu_initialization: bool = False
    """Initialise on CPU instead of GPU (slower but TP-deterministic)."""

    # === Norms ===
    normalization: str = "LayerNorm"
    """``"LayerNorm"`` or ``"RMSNorm"``."""

    layernorm_epsilon: float = 1e-5
    """Epsilon for normalisation layers."""

    layernorm_zero_centered_gamma: bool = False
    """Centre LayerNorm gamma around 0 for numerical stability."""

    # === Bias / activation ===
    add_bias_linear: bool = True
    """Bias in supported projection / MLP / expert linear layers."""

    gated_linear_unit: bool = False
    """Use GLU on the first MLP linear layer."""

    activation_func: str = "gelu"
    """MLP activation function name (looked up in :data:`_ACTIVATION_BY_NAME`)."""

    # === Dropout ===
    hidden_dropout: float = 0.0
    """Dropout on the transformer hidden state."""

    # === Residual ===
    fp32_residual_connection: bool = False
    """Cast residual connections to fp32."""

    # === Fusions ===
    apply_rope_fusion: bool = False
    """Use the fused RoPE kernel for attention-like layers."""

    persist_layer_norm: bool = False
    """Use the persistent fused LayerNorm kernel (only supports a fixed set of hidden sizes)."""

    # === Model family ===
    # ``is_hybrid_model`` is intentionally NOT exposed here — every HybridModel
    # recipe is a hybrid model by construction, so :meth:`HybridModelConfig.compile`
    # forces it to ``True`` on every per-layer / stack-level TC. Putting it on the
    # user surface would just be a redundant ``=True`` users have to write.

    # ─────────────────────────────────────────────────────────────────────────
    # TransformerConfig-specific fields below this line.
    #
    # These knobs map directly to ``TransformerConfig`` settings that select
    # implementation backends or kernel-fusion details specific to the
    # ``TransformerConfig`` / ``TransformerLayer`` machinery. They exist to
    # let recipes faithfully reproduce existing models that already set them.
    # When the underlying layer infrastructure is rewritten with dedicated
    # per-layer configs, these fields will move, change shape, or disappear —
    # they are not a stable part of the DSL's contract. New recipes should
    # only set them if they actually need to override the default.
    # ─────────────────────────────────────────────────────────────────────────

    transformer_impl: str = "transformer_engine"
    """Transformer implementation: ``"transformer_engine"``, ``"local"``, or ``"inference_optimized"``."""

    cuda_graph_impl: Optional[str] = None
    """CUDA graph capture implementation: ``"none"``, ``"local"``, or ``"transformer_engine"``."""

    cuda_graph_scope: str = "full"
    """CUDA graph capture scope (e.g. ``"full"``, ``"attn"``, ``"mlp"``, ``"full_iteration"``)."""

    cuda_graph_warmup_steps: int = 3
    """Number of warmup steps for CUDA graphs."""

    # === Escape hatch for any TransformerConfig field not curated above ===
    extra: Dict[str, Any] = field(default_factory=dict)
    """Passthrough kwargs forwarded to :class:`TransformerConfig`.

    Use this to set any TransformerConfig field that isn't a first-class
    attribute on :class:`CommonLayerConfig`. New fields added to
    TransformerConfig upstream propagate to recipes immediately without
    needing a DSL update::

        common = CommonLayerConfig(
            hidden_size=2688,
            extra={"qk_layernorm": True, "moe_expert_capacity_factor": 1.5},
        )

    Keys are validated against ``dataclasses.fields(TransformerConfig)`` at
    compile time — typos raise ``ValueError``. Curated fields (e.g.
    ``hidden_size``) take precedence; ``extra`` only fills in fields the
    curated set does not cover.
    """

    # ---------------------------------------------------------------- helpers

    def update(self, **overrides) -> "CommonLayerConfig":
        """Return a fresh CommonLayerConfig with overrides applied."""
        return dataclasses.replace(self, **overrides)

    def to_transformer_config_kwargs(self) -> Dict[str, Any]:
        """Translate to keyword arguments accepted by :class:`TransformerConfig`.

        Performs string→torch.dtype and string→callable mapping. Does **not**
        include layer-specific fields (``num_attention_heads``,
        ``mamba_head_dim``, ``num_moe_experts``, ...) — those come from the
        per-layer :class:`LayerConfig` and are merged on top.

        .. note::

            This method is the bridge between the DSL's user-facing fields
            and the underlying :class:`TransformerConfig` machinery the
            current layer classes consume. When the layer classes are
            rewritten with dedicated per-layer configs, this method's
            output type and name change but the DSL's user-facing surface
            (``CommonLayerConfig`` fields, ``LayerConfig`` subclasses) is
            preserved. Recipe code does not need to touch this method.
        """
        kwargs: Dict[str, Any] = {
            "hidden_size": self.hidden_size,
            "params_dtype": _resolve_dtype(self.params_dtype),
            "sequence_parallel": self.sequence_parallel,
            "init_method_std": self.init_method_std,
            "perform_initialization": self.perform_initialization,
            "use_cpu_initialization": self.use_cpu_initialization,
            "normalization": self.normalization,
            "layernorm_epsilon": self.layernorm_epsilon,
            "layernorm_zero_centered_gamma": self.layernorm_zero_centered_gamma,
            "add_bias_linear": self.add_bias_linear,
            "gated_linear_unit": self.gated_linear_unit,
            "activation_func": _resolve_activation(self.activation_func),
            "hidden_dropout": self.hidden_dropout,
            "fp32_residual_connection": self.fp32_residual_connection,
            "first_last_layers_bf16": self.first_last_layers_bf16,
            "apply_rope_fusion": self.apply_rope_fusion,
            "persist_layer_norm": self.persist_layer_norm,
            "transformer_impl": self.transformer_impl,
            "cuda_graph_scope": self.cuda_graph_scope,
            "cuda_graph_warmup_steps": self.cuda_graph_warmup_steps,
        }
        if self.ffn_hidden_size is not None:
            kwargs["ffn_hidden_size"] = self.ffn_hidden_size
        if self.cuda_graph_impl is not None:
            kwargs["cuda_graph_impl"] = self.cuda_graph_impl
        # mixed_precision_dtype → bf16/fp16 booleans expected by TransformerConfig
        mp = self.mixed_precision_dtype
        if mp == "bf16":
            kwargs["bf16"] = True
        elif mp == "fp16":
            kwargs["fp16"] = True
        elif mp != "fp32":
            raise ValueError(
                f"mixed_precision_dtype must be one of 'fp32' / 'fp16' / 'bf16'; "
                f"got {mp!r}."
            )
        # Escape-hatch passthrough: validate keys then merge over curated kwargs
        # for any TransformerConfig field not exposed as a first-class attribute.
        if self.extra:
            validate_extra_kwargs(self.extra, "CommonLayerConfig.extra")
            kwargs.update(self.extra)
        return kwargs


def validate_extra_kwargs(extra: Dict[str, Any], origin: str) -> None:
    """Validate that every key in ``extra`` names a real TransformerConfig field.

    The DSL accepts an ``extra: Dict[str, Any]`` passthrough on every config
    class to keep the recipe surface complete relative to TransformerConfig
    without monotonically growing the curated DSL field set. To catch typos
    early, this validator rejects keys that don't appear on TransformerConfig.

    Raises:
        ValueError: if any key in ``extra`` is not a TransformerConfig field.
    """
    from megatron.core.transformer.transformer_config import TransformerConfig

    valid_fields = {f.name for f in dataclasses.fields(TransformerConfig)}
    unknown = sorted(set(extra) - valid_fields)
    if unknown:
        raise ValueError(
            f"{origin} contains keys that are not TransformerConfig fields: "
            f"{unknown}. Either correct the spelling or add the field upstream "
            f"in megatron/core/transformer/transformer_config.py."
        )
