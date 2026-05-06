# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.

"""Per-type :class:`LayerConfig` subclasses for the HybridModel Python DSL.

Each layer-type config exposes only the fields a recipe author actually needs
to set: layer-specific architecture knobs and per-layer overrides of the
common config. Anything derivable from other fields (``num_query_groups``,
``kv_channels``, ``mamba_num_heads``, ``num_layers``) is computed at compile
time inside :meth:`LayerConfig.to_transformer_config` rather than required
from the recipe.

Special pattern markers — :class:`EmbeddingLayerConfig` and
:class:`CrossEntropyLayerConfig` — also live here. They participate in
the layer pattern but are not "layers" the :class:`HybridStack`
constructs; they encode model-wrapping metadata (vocab/sequence shape).
"""

from dataclasses import dataclass, field, fields
from typing import Any, ClassVar, Dict, Optional, Union

from megatron.core.models.hybrid.common_layer_config import (
    CommonLayerConfig,
    _resolve_activation,
    validate_extra_kwargs,
)
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import MLATransformerConfig, TransformerConfig

# ------------------------------------------------------------------- LAYERS


@dataclass
class LayerConfig:
    """Base class for layer-pattern leaves.

    Concrete subclasses set a class-level ``SYMBOL`` matching one of
    :class:`Symbols.VALID_LAYERS` and override
    :meth:`_layer_specific_kwargs` to supply layer-class-specific fields
    (e.g. ``mamba_state_dim``) for the per-layer
    :class:`TransformerConfig`.
    """

    common_config: Optional[CommonLayerConfig] = None
    """Per-layer common config; ``None`` (the default) means
    "inherit from the recipe's :attr:`HybridModelConfig.common_config`",
    which the recipe's internal lowering substitutes in. Pass an
    explicit :class:`CommonLayerConfig` to override (e.g. via
    :meth:`CommonLayerConfig.update`). After lowering runs, this
    field is always non-None."""

    extra: Dict[str, Any] = field(default_factory=dict)
    """Per-layer passthrough kwargs forwarded to this layer's
    :class:`TransformerConfig`. Same semantics as
    :attr:`CommonLayerConfig.extra` but applied per-layer instead of
    model-wide. Layer ``extra`` overrides curated layer fields, which
    override common ``extra``, which overrides common curated fields."""

    SYMBOL: ClassVar[str]

    TC_CLASS: ClassVar[type] = TransformerConfig
    """The :class:`TransformerConfig` subclass used to materialise this
    layer's per-layer config. Subclasses override (e.g.
    :class:`DSALayerConfig` uses :class:`MLATransformerConfig`) when they
    need fields that only live on a specialised subclass."""

    def _layer_specific_kwargs(self) -> Dict[str, Any]:
        """Return the layer-class-specific TransformerConfig kwargs."""
        return {}

    def to_transformer_config(
        self,
        num_layers: int,
        parallelism: Optional[Dict[str, Any]] = None,
        placeholders: Optional[Dict[str, Any]] = None,
    ) -> TransformerConfig:
        """Build a per-layer :class:`TransformerConfig`.

        ``num_layers`` is supplied by the caller (usually
        :meth:`HybridModelConfig.compile`) after counting the flattened
        layer pattern, so recipes never need to set it manually.

        ``parallelism`` carries the universal model-level parallelism
        settings (TP/CP) — these **always win** because they're
        job-level. EP/ETP are MoE-only and are injected into MoE
        per-layer TCs by :meth:`HybridModelConfig.compile`, not via this
        path. ``placeholders`` carries values that satisfy
        :meth:`TransformerConfig.__post_init__` invariants on layers that
        don't naturally set them (e.g. ``num_attention_heads`` is required
        for the ``kv_channels`` derivation even on Mamba layers) — these
        **only fill in if absent**, so a recipe author's
        ``common.extra={"num_attention_heads": 32}`` overrides the
        placeholder.

        Final precedence (lowest → highest): common curated → ``common.extra``
        → ``parallelism`` → ``placeholders`` (only if absent) →
        ``layer._layer_specific_kwargs()`` → ``layer.extra``.
        """
        kwargs = self.common_config.to_transformer_config_kwargs()
        if parallelism:
            kwargs.update(parallelism)
        if placeholders:
            for k, v in placeholders.items():
                kwargs.setdefault(k, v)
        kwargs.update(self._layer_specific_kwargs())
        if self.extra:
            origin = f"{type(self).__name__}.extra"
            validate_extra_kwargs(self.extra, origin)
            # Reject ``extra`` keys that shadow this layer's curated fields
            # (silent dataclass shadowing) or model-wide parallelism (process
            # groups are global; a per-layer disagreement is invalid sharding).
            # Layer ``extra`` may still override a value carried over from
            # common scope — more-specific scope wins.
            curated = {f.name for f in fields(self) if f.name not in ("common_config", "extra")}
            shadowed = sorted(set(self.extra) & curated)
            if shadowed:
                raise ValueError(
                    f"{origin} cannot name curated fields: {shadowed}. "
                    f"Set them on the dataclass attribute directly."
                )
            if parallelism:
                shadowed = sorted(set(parallelism) & set(self.extra))
                if shadowed:
                    raise ValueError(
                        f"{origin} cannot override model-wide parallelism fields "
                        f"{shadowed}; set them on HybridModelConfig instead."
                    )
            kwargs.update(self.extra)
        # Test-friendly fallback: the recipe's lowering always supplies a
        # placeholder, but unit tests calling
        # ``lc.to_transformer_config(num_layers=N)`` directly need a non-zero
        # baseline so TC's ``__post_init__`` doesn't divide by zero deriving
        # ``kv_channels``.
        kwargs.setdefault("num_attention_heads", 1)
        kwargs["num_layers"] = num_layers
        # Drop None values so TransformerConfig defaults apply.
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return self.TC_CLASS(**kwargs)


@dataclass
class MambaLayerConfig(LayerConfig):
    """Mamba SSM layer (symbol ``M``).

    ``num_heads`` may be left ``None`` to auto-derive
    ``hidden_size * expand // head_dim`` (TransformerConfig handles this
    derivation when ``mamba_num_heads`` is ``None``).
    """

    head_dim: int = 64
    """Mamba head dimension (``mamba_head_dim``)."""

    state_size: int = 128
    """SSM state dimension (``mamba_state_dim``)."""

    num_groups: int = 8
    """Mamba groups (``mamba_num_groups``)."""

    num_heads: Optional[int] = None
    """Optional explicit head count; defaults to derived value."""

    SYMBOL: ClassVar[str] = Symbols.MAMBA

    def _layer_specific_kwargs(self) -> Dict[str, Any]:
        return {
            "mamba_head_dim": self.head_dim,
            "mamba_state_dim": self.state_size,
            "mamba_num_groups": self.num_groups,
            "mamba_num_heads": self.num_heads,
        }


@dataclass
class AttentionLayerConfig(LayerConfig):
    """Self-attention layer (symbol ``*``).

    ``num_query_groups`` and ``kv_channels`` may be left ``None`` to
    auto-derive from ``num_attention_heads`` and ``hidden_size`` respectively
    (handled in :meth:`TransformerConfig.__post_init__`).
    """

    num_attention_heads: int = 1
    """Number of attention heads."""

    num_query_groups: Optional[int] = None
    """GQA group count; ``None`` → ``num_attention_heads`` (i.e. MHA)."""

    kv_channels: Optional[int] = None
    """KV head dim; ``None`` → ``hidden_size // num_attention_heads``."""

    attention_dropout: Optional[float] = None
    """Dropout on attention weights; ``None`` → TransformerConfig default."""

    attention_softmax_in_fp32: Optional[bool] = None
    """Run attention masking and softmax in fp32; ``None`` keeps the default."""

    add_qkv_bias: Optional[bool] = None
    """Bias only in QKV projections, overriding ``add_bias_linear`` for QKV."""

    masked_softmax_fusion: Optional[bool] = None
    """Use fused masked-softmax; ``None`` keeps the default."""

    attention_backend: Optional[str] = None
    """Attention backend name (``"flash"`` / ``"fused"`` / ``"unfused"`` /
    ``"local"`` / ``"auto"``); ``None`` → TransformerConfig default."""

    SYMBOL: ClassVar[str] = Symbols.ATTENTION

    def _layer_specific_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "num_attention_heads": self.num_attention_heads,
            "num_query_groups": self.num_query_groups,
            "kv_channels": self.kv_channels,
            "attention_dropout": self.attention_dropout,
            "attention_softmax_in_fp32": self.attention_softmax_in_fp32,
            "add_qkv_bias": self.add_qkv_bias,
            "masked_softmax_fusion": self.masked_softmax_fusion,
        }
        if self.attention_backend is not None:
            kwargs["attention_backend"] = AttnBackend[self.attention_backend]
        return kwargs


@dataclass
class DSALayerConfig(LayerConfig):
    """DeepSeek Sparse Attention / MLA layer (symbol ``D``).

    Cannot be combined with :class:`AttentionLayerConfig` in the same pattern.
    The MLA-defining knobs (``q_lora_rank``, ``kv_lora_rank``,
    ``qk_head_dim``, ``qk_pos_emb_head_dim``, ``v_head_dim``,
    ``rotary_scaling_factor``) all default to ``None``, which means
    "fall through to the underlying :class:`TransformerConfig` default";
    DeepSeek-style models override them to the values the model card
    specifies.
    """

    num_attention_heads: int = 1
    """Number of attention heads."""

    num_query_groups: Optional[int] = None
    """GQA group count."""

    kv_channels: Optional[int] = None
    """KV head dim."""

    add_qkv_bias: Optional[bool] = None
    """Bias in the QKV projection only, overriding ``add_bias_linear`` for
    QKV. Mirrors the same field on :class:`AttentionLayerConfig`."""

    q_lora_rank: Optional[int] = None
    """Rank of the query tensor's low-rank representation."""

    kv_lora_rank: Optional[int] = None
    """Rank of the key/value tensors' low-rank representation."""

    qk_head_dim: Optional[int] = None
    """Dimension of the QK projection head. Together with
    ``qk_pos_emb_head_dim`` this gives the full Q head dim
    (``q_head_dim = qk_head_dim + qk_pos_emb_head_dim``)."""

    qk_pos_emb_head_dim: Optional[int] = None
    """Position-embedding portion of the QK projection head dim."""

    v_head_dim: Optional[int] = None
    """Dimension of the V projection head."""

    rotary_scaling_factor: Optional[float] = None
    """Rotary scaling factor for YaRN-style rope inside MLA."""

    SYMBOL: ClassVar[str] = Symbols.DS_ATTENTION
    TC_CLASS: ClassVar[type] = MLATransformerConfig

    def _layer_specific_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "num_attention_heads": self.num_attention_heads,
            "num_query_groups": self.num_query_groups,
            "kv_channels": self.kv_channels,
            "multi_latent_attention": True,
            "q_lora_rank": self.q_lora_rank,
            "kv_lora_rank": self.kv_lora_rank,
            "qk_head_dim": self.qk_head_dim,
            "qk_pos_emb_head_dim": self.qk_pos_emb_head_dim,
            "v_head_dim": self.v_head_dim,
            "rotary_scaling_factor": self.rotary_scaling_factor,
        }
        if self.add_qkv_bias is not None:
            kwargs["add_qkv_bias"] = self.add_qkv_bias
        return kwargs


@dataclass
class GDNLayerConfig(LayerConfig):
    """Gated DeltaNet layer (symbol ``G``)."""

    num_attention_heads: int = 1
    """Used for the attention-shaped paths inside GDN."""

    num_query_groups: Optional[int] = None

    kv_channels: Optional[int] = None

    SYMBOL: ClassVar[str] = Symbols.GDN

    def _layer_specific_kwargs(self) -> Dict[str, Any]:
        return {
            "num_attention_heads": self.num_attention_heads,
            "num_query_groups": self.num_query_groups,
            "kv_channels": self.kv_channels,
        }


@dataclass
class MLPLayerConfig(LayerConfig):
    """Dense MLP layer (symbol ``-``)."""

    ffn_hidden_size: Optional[int] = None
    """MLP intermediate size; ``None`` → ``4 * hidden_size``."""

    SYMBOL: ClassVar[str] = Symbols.MLP

    def _layer_specific_kwargs(self) -> Dict[str, Any]:
        return {"ffn_hidden_size": self.ffn_hidden_size}


@dataclass
class MoELayerConfig(LayerConfig):
    """Mixture-of-Experts layer (symbol ``E``).

    Two distinct :class:`MoELayerConfig` instances with different
    architecture overrides may coexist in the same pattern (e.g. one with
    128 experts/top-6 and one with 64 experts/top-2). Expert parallelism
    (EP/ETP) is a model-wide concept and lives on
    :class:`HybridModelConfig`; :meth:`HybridModelConfig.compile` injects
    those fields into MoE per-layer TransformerConfigs only.
    """

    num_experts: int = 1
    """Total experts (``num_moe_experts``)."""

    top_k: int = 1
    """Top-k routing (``moe_router_topk``)."""

    ffn_hidden_size: Optional[int] = None
    """Per-expert FFN size; falls back to ``moe_ffn_hidden_size`` semantics."""

    router_score_function: str = "softmax"
    """``"softmax"`` or ``"sigmoid"``."""

    router_load_balancing_type: str = "aux_loss"
    """``"aux_loss"`` / ``"seq_aux_loss"`` / ``"global_aux_loss"`` / ``"sinkhorn"`` / ``"none"``."""

    router_topk_scaling_factor: Optional[float] = None
    """Optional scaling factor applied to top-k logits."""

    router_enable_expert_bias: bool = False
    """Enable learnable per-expert bias term in the router."""

    router_dtype: Optional[str] = None
    """Router compute dtype name; ``None`` → input dtype."""

    aux_loss_coeff: float = 0.0
    """Auxiliary load-balancing loss coefficient (``moe_aux_loss_coeff``)."""

    shared_expert_intermediate_size: Optional[int] = None
    """If set, enables shared experts at this size."""

    token_dispatcher_type: str = "allgather"
    """``"allgather"`` / ``"alltoall"`` / ``"flex"``."""

    grouped_gemm: bool = False
    """Use grouped GEMM for expert MLPs."""

    permute_fusion: bool = False
    """Fuse the token permutation kernel."""

    activation: Optional[str] = None
    """Override the MLP activation for this layer (defaults to common)."""

    # === Architecture / Routing / Fusion ===

    router_num_groups: Optional[int] = None
    """Number of router groups for grouped top-k routing (``moe_router_num_groups``)."""

    router_group_topk: Optional[int] = None
    """Per-group top-k for grouped routing (``moe_router_group_topk``)."""

    shared_expert_overlap: bool = False
    """Overlap shared-expert compute with dispatcher comms (``moe_shared_expert_overlap``)."""

    flex_dispatcher_backend: Optional[str] = None
    """Flex token-dispatcher backend (``"deepep"`` / ``"hybridep"``);
    ``None`` → TransformerConfig default."""

    hybridep_num_sms: Optional[int] = None
    """Number of SMs reserved for the HybridEP dispatcher
    (``moe_hybridep_num_sms``); ``None`` → TransformerConfig default."""

    router_padding_for_fp8: bool = False
    """Pad router output for FP8 alignment (``moe_router_padding_for_fp8``)."""

    router_fusion: bool = False
    """Enable fused router kernels (``moe_router_fusion``)."""

    router_force_load_balancing: bool = False
    """Force balanced routing assignment (``moe_router_force_load_balancing``)."""

    use_fused_weighted_squared_relu: bool = False
    """Enable the fused weighted squared-ReLU expert activation kernel."""

    SYMBOL: ClassVar[str] = Symbols.MOE

    def _layer_specific_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "num_moe_experts": self.num_experts,
            "moe_router_topk": self.top_k,
            "moe_ffn_hidden_size": self.ffn_hidden_size,
            "moe_router_score_function": self.router_score_function,
            "moe_router_load_balancing_type": self.router_load_balancing_type,
            "moe_router_topk_scaling_factor": self.router_topk_scaling_factor,
            "moe_router_enable_expert_bias": self.router_enable_expert_bias,
            "moe_router_dtype": self.router_dtype,
            "moe_aux_loss_coeff": self.aux_loss_coeff,
            "moe_shared_expert_intermediate_size": self.shared_expert_intermediate_size,
            "moe_token_dispatcher_type": self.token_dispatcher_type,
            "moe_grouped_gemm": self.grouped_gemm,
            "moe_permute_fusion": self.permute_fusion,
            "moe_router_num_groups": self.router_num_groups,
            "moe_router_group_topk": self.router_group_topk,
            "moe_shared_expert_overlap": self.shared_expert_overlap,
            "moe_flex_dispatcher_backend": self.flex_dispatcher_backend,
            "moe_hybridep_num_sms": self.hybridep_num_sms,
            "moe_router_padding_for_fp8": self.router_padding_for_fp8,
            "moe_router_fusion": self.router_fusion,
            "moe_router_force_load_balancing": self.router_force_load_balancing,
            "use_fused_weighted_squared_relu": self.use_fused_weighted_squared_relu,
        }
        if self.activation is not None:
            kwargs["activation_func"] = _resolve_activation(self.activation)
        return kwargs


# -------------------------------------------------------- PATTERN MARKERS


@dataclass
class EmbeddingLayerConfig:
    """Input embedding marker (must appear at the start of a layer pattern).

    Carries the runtime model-shape parameters that aren't part of any single
    decoder layer: vocab size, max sequence length, position-embedding
    selection, and rotary settings.
    """

    common_config: Optional[CommonLayerConfig] = None
    """Per-marker common config; ``None`` (the default) means "inherit from
    the recipe's :attr:`HybridModelConfig.common_config`", which the
    recipe's internal lowering substitutes in. After lowering runs,
    this field is always non-None."""

    vocab_size: int = 0
    """Vocabulary size (post-padding)."""

    max_sequence_length: int = 0
    """Maximum sequence length the embedding supports."""

    position_embedding_type: str = "none"
    """``"learned_absolute"`` / ``"rope"`` / ``"yarn"`` / ``"none"``."""

    rotary_percent: float = 1.0
    """Fraction of the head dim to apply RoPE to (only when ``rope``/``yarn``)."""

    rotary_base: int = 10000
    """RoPE base period."""

    seq_len_interpolation_factor: Optional[float] = None
    """Linear-interpolation scaling for longer-than-trained sequences."""

    scatter_embedding_sequence_parallel: bool = True
    """Scatter the embedding output along the sequence-parallel dim."""

    yarn: Optional[Dict[str, Any]] = None
    """YARN scaling parameters (consumed only when
    ``position_embedding_type == "yarn"``). These are not
    :class:`TransformerConfig` dataclass fields today; the recipe's
    internal lowering attaches them as ad-hoc attributes to match the
    :func:`getattr` lookups in :class:`HybridModel.__init__`.
    Recognised keys: ``yarn_rotary_scaling_factor``,
    ``yarn_original_max_position_embeddings``, ``yarn_beta_fast``,
    ``yarn_beta_slow``, ``yarn_mscale``, ``yarn_mscale_all_dim``,
    ``yarn_correction_range_round_to_int``. Unknown keys raise at lowering
    time. When upstream lifts these onto :class:`TransformerConfig`, this
    field collapses to a plain ``extra`` passthrough and the setattr loop
    in the lowering goes away."""

    extra: Dict[str, Any] = field(default_factory=dict)
    """Passthrough kwargs forwarded to the stack-level
    :class:`TransformerConfig` (used for the embedding init / final norm).
    Same semantics as :attr:`CommonLayerConfig.extra`."""


@dataclass
class CrossEntropyLayerConfig:
    """Output / loss marker (must appear at the end of a layer pattern)."""

    fp16_lm_cross_entropy: bool = False
    """Compute the LM cross-entropy in fp16."""

    parallel_output: bool = True
    """Keep logits split across tensor-parallel ranks (no all-gather)."""

    loss_fusion: bool = False
    """Enable fused cross-entropy loss (``cross_entropy_loss_fusion``)."""

    fusion_impl: str = "native"
    """Fused CE implementation (``"native"`` / ``"te"``); maps to
    ``cross_entropy_fusion_impl``."""

    calculate_per_token_loss: bool = False
    """Average loss per token rather than per microbatch
    (``calculate_per_token_loss``)."""

    extra: Dict[str, Any] = field(default_factory=dict)
    """Passthrough kwargs forwarded to the stack-level
    :class:`TransformerConfig`. Same semantics as
    :attr:`CommonLayerConfig.extra`."""


# ----------------------------------------------------------- type aliases

#: Anything that may legally appear at a leaf of a layer pattern.
PatternLeaf = Union[LayerConfig, EmbeddingLayerConfig, CrossEntropyLayerConfig]
