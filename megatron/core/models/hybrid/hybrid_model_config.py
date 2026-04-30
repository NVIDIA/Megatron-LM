# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.

"""Recipe-level wrapper for the HybridModel Python DSL.

A complete HybridModel "recipe" is a single :class:`HybridModelConfig`. It
bundles the model-wide common config, the layer pattern (with embedding/loss
markers), and the small set of model-wrapping settings that aren't part of
any individual layer (output untying, parallel-output behaviour, etc.).

The recipe entry point — i.e. the function a ``--model-recipe`` module must
export — returns an instance of this class:

.. code-block:: python

    def make_recipe() -> HybridModelConfig:
        ...
        return HybridModelConfig(
            common_config=common_config,
            layer_pattern=layer_pattern,
            untie_embeddings_and_output_weights=True,
        )

:meth:`compile` walks the layer pattern, extracts the embedding / loss /
pipeline-split markers, derives ``num_layers`` from the flattened layer count,
and returns private compiler output containing exactly the kwargs
:class:`HybridModel` needs. Recipe authors should treat
:class:`HybridModelConfig` as the stable API; the generated
:class:`TransformerConfig` objects are a compatibility layer for the current
implementation.
"""

import dataclasses
from dataclasses import dataclass
from typing import Any, List, Optional

from megatron.core.models.hybrid.common_layer_config import CommonLayerConfig
from megatron.core.models.hybrid.layer_configs import (
    AttentionLayerConfig,
    CrossEntropyLayerConfig,
    DSALayerConfig,
    EmbeddingLayerConfig,
    LayerConfig,
    MoELayerConfig,
    MTPLayerConfig,
    PipelineSplit,
)
from megatron.core.transformer.transformer_config import TransformerConfig

# YARN scaling parameters that HybridModel reads via getattr when
# position_embedding_type == "yarn" (see hybrid_model.py around the
# YarnRotaryEmbedding construction). They are not currently
# TransformerConfig dataclass fields; compile() applies them via setattr to
# match the existing legacy-path pattern.
_YARN_FIELDS = (
    "yarn_rotary_scaling_factor",
    "yarn_original_max_position_embeddings",
    "yarn_beta_fast",
    "yarn_beta_slow",
    "yarn_mscale",
    "yarn_mscale_all_dim",
    "yarn_correction_range_round_to_int",
)


@dataclass
class CompiledRecipe:
    """Internal result of compiling a :class:`HybridModelConfig`.

    Contains exactly the data :class:`HybridModel` needs to construct the
    model: a global :class:`TransformerConfig` for the stack-level concerns
    (final norm, embedding init), per-layer configs and symbols for the
    decoder, and the model-wrapping kwargs derived from the embedding / loss
    markers. This is intentionally compiler output, not the user-facing recipe
    authoring API.
    """

    config: TransformerConfig
    layer_type_list: List[str]
    layer_config_list: List[TransformerConfig]
    vocab_size: int
    max_sequence_length: int
    position_embedding_type: str
    rotary_percent: float
    rotary_base: int
    seq_len_interpolation_factor: Optional[float]
    scatter_embedding_sequence_parallel: bool
    share_embeddings_and_output_weights: bool
    fp16_lm_cross_entropy: bool
    parallel_output: bool
    # MTP plumbing (consumed by the existing MultiTokenPredictionBlock).
    # ``mtp_layer_pattern`` is the single-character per-depth body string —
    # the same form :func:`parse_hybrid_pattern` produces from a string DSL
    # like ``"M*M*/MM/MM"``. ``None`` when the recipe has no MTP markers.
    mtp_layer_pattern: Optional[str]
    mtp_num_depths: int
    # Dotted path to a custom HybridStack ``ModuleSpec``; the builder
    # imports it. ``None`` lets the builder auto-pick by ``transformer_impl``.
    stack_spec: Optional[str]


@dataclass
class HybridModelConfig:
    """A complete HybridModel recipe.

    Returned by a recipe module's ``make_recipe()`` entry point and consumed
    by :func:`load_recipe` and the ``--model-recipe`` path in
    ``hybrid_builder``.
    """

    common_config: CommonLayerConfig
    """Model-wide defaults shared by every layer."""

    layer_pattern: list
    """The layer pattern. Must begin with an :class:`EmbeddingLayerConfig`,
    contain at least one decoder :class:`LayerConfig`, and end with a
    :class:`CrossEntropyLayerConfig`. Decoder layers may be nested in any
    list/tuple structure; :func:`compile` flattens them. Pipeline splits are
    declared with :class:`PipelineSplit` markers (PP > 1 not yet supported)."""

    untie_embeddings_and_output_weights: bool = False
    """Untie input embedding and output projection weights."""

    # === Model-level parallelism ===
    # These live here (not on :class:`CommonLayerConfig`) because they are
    # job-/model-level concerns: process groups are constructed once, not
    # per-layer, and they cannot meaningfully vary per layer. ``compile()``
    # injects them into every per-layer TransformerConfig and the
    # stack-level config.
    tensor_model_parallel_size: int = 1
    """Tensor-model-parallel world size."""

    pipeline_model_parallel_size: int = 1
    """Pipeline-model-parallel world size (PP > 1 not yet supported by the DSL)."""

    context_parallel_size: int = 1
    """Context-parallel world size."""

    expert_model_parallel_size: int = 1
    """Expert-model-parallel world size (used by MoE layers)."""

    expert_tensor_parallel_size: Optional[int] = None
    """Expert tensor-parallel size (defaults to ``tensor_model_parallel_size``)."""

    pipeline_dtype: Optional[str] = None
    """Pipeline P2P communication dtype name. ``None`` keeps TC default."""

    stack_spec: Optional[str] = None
    """Dotted Python path to a custom :class:`ModuleSpec` for
    :class:`HybridStack` (e.g. ``"my_pkg.my_module.my_stack_spec"``). When
    set, the builder imports and uses this spec; otherwise the spec is
    auto-picked from ``transformer_impl`` (the legacy ``--spec`` /
    ``hybrid_inference_stack_spec`` selection). The string form keeps
    recipes serialisable; resolution happens in ``hybrid_builder``."""

    def compile(self) -> CompiledRecipe:
        """Process the layer pattern into a :class:`CompiledRecipe`.

        Extracts the embedding/loss markers, validates pattern structure,
        flattens decoder layers, derives ``num_layers`` from the flattened
        count, and produces per-layer :class:`TransformerConfig` instances.
        """
        from megatron.core.models.hybrid.layer_pattern import flatten_decoder_pattern

        embedding, decoder_leaves, mtp_markers, loss = _split_pattern(self.layer_pattern)

        # Auto-inherit the recipe's common_config into any layer/marker that
        # was constructed without an explicit ``common_config=`` argument.
        # Without this, a layer like ``MambaLayerConfig(head_dim=64)`` carries
        # a default-constructed CommonLayerConfig with ``hidden_size=0``,
        # silently producing an invalid model.
        embedding = _inherit_common_if_default(embedding, self.common_config)
        loss = _inherit_common_if_default(loss, self.common_config)
        decoder_leaves = _inherit_common_in_pattern(decoder_leaves, self.common_config)
        mtp_markers = [_inherit_common_in_marker(m, self.common_config) for m in mtp_markers]

        decoder_flat: List[LayerConfig] = flatten_decoder_pattern(decoder_leaves)
        if not decoder_flat:
            raise ValueError(
                "layer_pattern has no decoder layers between the EmbeddingLayerConfig "
                "and CrossEntropyLayerConfig markers."
            )

        num_layers = len(decoder_flat)

        # ─────────────────────────────────────────────────────────────────
        # Glue between the DSL surface and the underlying TransformerConfig
        # machinery. ``parallelism`` carries the universal job-level
        # settings (TP/PP/CP) that flow into every per-layer config.
        # ``expert_parallelism`` carries EP/ETP, which are MoE-only and
        # therefore only flow into MoE per-layer TCs (and the stack-level
        # TC for the model-wide topology snapshot). ``placeholders`` carries
        # values that exist solely to satisfy
        # ``TransformerConfig.__post_init__`` invariants on layers that
        # don't naturally set them (e.g. ``num_attention_heads`` is
        # required for the kv_channels derivation even on Mamba layers).
        #
        # When the layer infrastructure is rewritten with dedicated
        # per-layer config classes, neither the placeholder logic nor the
        # split parallelism dicts will be needed — the new layers won't
        # share a single config type and won't impose cross-layer
        # invariants. The DSL's user-facing surface stays the same; only
        # this glue block changes.
        # ─────────────────────────────────────────────────────────────────

        # Universal parallelism — flows into every per-layer TC.
        parallelism: dict = {
            "tensor_model_parallel_size": self.tensor_model_parallel_size,
            "pipeline_model_parallel_size": self.pipeline_model_parallel_size,
            "context_parallel_size": self.context_parallel_size,
            # ``is_hybrid_model`` is implicit for any HybridModelConfig recipe;
            # force it on every TC so users don't carry a redundant ``=True``.
            "is_hybrid_model": True,
        }
        if self.pipeline_dtype is not None:
            from megatron.core.models.hybrid.common_layer_config import _resolve_dtype

            parallelism["pipeline_dtype"] = _resolve_dtype(self.pipeline_dtype)

        # MoE-only parallelism — applied to MoE per-layer TCs (after their
        # base build) and to the stack-level TC. Non-MoE per-layer TCs do
        # not receive these fields; TransformerConfig's defaults
        # (``expert_model_parallel_size=1``, ``expert_tensor_parallel_size``
        # auto-derived from TP) take over, which is correct because those
        # layers don't participate in expert parallelism.
        expert_parallelism: dict = {
            "expert_model_parallel_size": self.expert_model_parallel_size,
            "expert_tensor_parallel_size": (
                self.expert_tensor_parallel_size
                if self.expert_tensor_parallel_size is not None
                else self.tensor_model_parallel_size
            ),
        }

        # ``placeholders`` only fill if absent, so a recipe author's
        # ``common.extra={...}`` or per-layer override wins.
        # ``num_attention_heads % tensor_model_parallel_size == 0`` is
        # required by TC; setting the placeholder to TP satisfies the
        # check trivially on non-attention layers. Attention layers override
        # with their real value via ``_layer_specific_kwargs``.
        attention_metadata = _infer_uniform_attention_metadata(decoder_flat)

        # Heterogeneous attention geometry under RoPE/YARN is not supported:
        # HybridModel builds one global rotary embedding from the stack TC
        # and passes it to every attention layer. If layers disagree on
        # ``num_attention_heads`` / ``num_query_groups`` / ``kv_channels``,
        # the rotary tensor is sized for the placeholder value and at
        # runtime layers with different geometry hit shape mismatches.
        # Reject at compile time until per-layer rotary lands.
        if embedding.position_embedding_type in ("rope", "yarn"):
            _reject_heterogeneous_attention_geometry(decoder_flat, attention_metadata)

        placeholders: dict = {"num_attention_heads": max(1, self.tensor_model_parallel_size)}
        # If the recipe contains a uniform attention geometry, use it as the
        # semantic placeholder for non-attention layer TransformerConfigs.
        # This keeps recipe authors out of TransformerConfig compatibility
        # details like "Mamba still needs num_attention_heads to satisfy
        # TC.__post_init__".
        placeholders.update(attention_metadata)

        layer_type_list = [type(lc).SYMBOL for lc in decoder_flat]
        layer_config_list: List[TransformerConfig] = []
        for lc in decoder_flat:
            # Only MoE layers carry expert parallelism; non-MoE TCs default
            # to EP=1 (TC default), which is what they actually consume at
            # runtime. Merging into the initial kwargs (rather than via a
            # follow-up dataclasses.replace) keeps TC.__post_init__'s MoE
            # cross-checks (e.g. ``add_bias_linear`` requires ``ETP==1``)
            # validating against the right ETP value the first time around.
            layer_parallelism = parallelism
            if isinstance(lc, MoELayerConfig):
                layer_parallelism = {**parallelism, **expert_parallelism}
            tc = lc.to_transformer_config(
                num_layers, parallelism=layer_parallelism, placeholders=placeholders
            )
            layer_config_list.append(tc)

        # Reuse the existing pattern validator for the Attention/DSA mix rule.
        from megatron.core.models.hybrid.hybrid_layer_allocation import _validate_pattern

        _validate_pattern("".join(layer_type_list), "main", allow_pipe=False)

        # MTP: compile the per-depth body (shared across all MTP markers in
        # the pattern) into the single-character form the existing
        # MultiTokenPredictionBlock infrastructure consumes.
        (mtp_layer_pattern, mtp_num_depths, mtp_loss_scaling_factor, mtp_use_repeated_layer) = (
            _compile_mtp_markers(mtp_markers, self.common_config)
        )

        # Stack-level config: a model-wide topology snapshot used for the
        # final norm, embedding init, and the args projection in
        # ``_apply_model_recipe_to_args``. Carries EP/ETP because it
        # represents the global topology, even though non-MoE per-layer TCs
        # do not. When EP > 1, TC.__post_init__ requires ``num_moe_experts``
        # on every TC including this one — the real expert count lives on
        # the per-layer MoE TCs, so a non-None placeholder is sufficient.
        stack_kwargs = embedding.common_config.to_transformer_config_kwargs()
        stack_kwargs.update(parallelism)
        stack_kwargs.update(expert_parallelism)
        for k, v in attention_metadata.items():
            stack_kwargs.setdefault(k, v)
        for k, v in placeholders.items():
            stack_kwargs.setdefault(k, v)
        if self.expert_model_parallel_size > 1:
            stack_kwargs.setdefault("num_moe_experts", 1)
        # DSA / MLA layers carry their own decoupled RoPE; HybridModel uses
        # ``self.config.multi_latent_attention`` (the stack TC) to decide
        # whether to construct the global RoPE embedding. Without this flag
        # promoted to the stack, recipes containing DSALayerConfig get a
        # global rotary built and passed in alongside the MLA-internal one.
        if any(isinstance(lc, DSALayerConfig) for lc in decoder_flat):
            stack_kwargs["multi_latent_attention"] = True
        stack_kwargs["num_layers"] = num_layers
        stack_kwargs["cross_entropy_loss_fusion"] = loss.loss_fusion
        stack_kwargs["cross_entropy_fusion_impl"] = loss.fusion_impl
        stack_kwargs["calculate_per_token_loss"] = loss.calculate_per_token_loss
        # MTP plumbing — without ``mtp_num_layers`` set, HybridModel.forward
        # builds the MTP block but skips ``process_mtp_loss`` entirely (the
        # gate is ``self.config.mtp_num_layers is not None``), silently
        # training a model whose MTP head contributes nothing to the
        # objective. Set it whenever the recipe contains MTP markers.
        if mtp_num_depths > 0:
            stack_kwargs["mtp_num_layers"] = mtp_num_depths
            if mtp_loss_scaling_factor is not None:
                stack_kwargs["mtp_loss_scaling_factor"] = mtp_loss_scaling_factor
            if mtp_use_repeated_layer is not None:
                stack_kwargs["mtp_use_repeated_layer"] = mtp_use_repeated_layer
        # Marker-level passthroughs: Embedding and CrossEntropy markers may
        # set TransformerConfig fields the curated DSL surface doesn't cover.
        # ``extra`` may not name a curated TC field — that would silently shadow
        # the field set above and obscure the recipe's authoritative source.
        from megatron.core.models.hybrid.common_layer_config import validate_extra_kwargs

        if embedding.extra:
            validate_extra_kwargs(embedding.extra, "EmbeddingLayerConfig.extra")
            stack_kwargs.update(embedding.extra)
        if loss.extra:
            validate_extra_kwargs(loss.extra, "CrossEntropyLayerConfig.extra")
            curated = {
                "cross_entropy_loss_fusion",
                "cross_entropy_fusion_impl",
                "calculate_per_token_loss",
            }
            shadowed = curated & set(loss.extra)
            if shadowed:
                raise ValueError(
                    f"CrossEntropyLayerConfig.extra cannot override curated fields "
                    f"{sorted(shadowed)}; set them on the marker directly "
                    f"(e.g. CrossEntropyLayerConfig(loss_fusion=...))."
                )
            stack_kwargs.update(loss.extra)
        stack_kwargs = {k: v for k, v in stack_kwargs.items() if v is not None}
        stack_config = TransformerConfig(**stack_kwargs)

        # YARN parameters are not TransformerConfig dataclass fields today
        # (see model_builder.py:196-201 and tests/.../test_hybrid_model.py for
        # the existing setattr pattern). HybridModel.__init__ reads them via
        # getattr when position_embedding_type == "yarn"; mirror that
        # convention here so recipe-built models support YARN without forcing
        # the recipe author to monkey-patch the config.
        if embedding.position_embedding_type == "yarn":
            for name in _YARN_FIELDS:
                value = getattr(embedding, name)
                if value is not None:
                    setattr(stack_config, name, value)

        return CompiledRecipe(
            config=stack_config,
            layer_type_list=layer_type_list,
            layer_config_list=layer_config_list,
            vocab_size=embedding.vocab_size,
            max_sequence_length=embedding.max_sequence_length,
            position_embedding_type=embedding.position_embedding_type,
            rotary_percent=embedding.rotary_percent,
            rotary_base=embedding.rotary_base,
            seq_len_interpolation_factor=embedding.seq_len_interpolation_factor,
            scatter_embedding_sequence_parallel=embedding.scatter_embedding_sequence_parallel,
            share_embeddings_and_output_weights=not self.untie_embeddings_and_output_weights,
            fp16_lm_cross_entropy=loss.fp16_lm_cross_entropy,
            parallel_output=loss.parallel_output,
            mtp_layer_pattern=mtp_layer_pattern,
            mtp_num_depths=mtp_num_depths,
            stack_spec=self.stack_spec,
        )


# --- pattern-structure helpers --------------------------------------------


_DEFAULT_COMMON = CommonLayerConfig()


def _inherit_common_if_default(node: Any, recipe_common: CommonLayerConfig) -> Any:
    """If ``node`` carries a default-constructed common_config, replace it
    with ``recipe_common`` via :func:`dataclasses.replace`. Otherwise return
    unchanged. Applies to LayerConfig instances and pattern markers."""
    if not hasattr(node, "common_config"):
        return node
    if node.common_config == _DEFAULT_COMMON:
        return dataclasses.replace(node, common_config=recipe_common)
    return node


def _inherit_common_in_pattern(node: Any, recipe_common: CommonLayerConfig) -> Any:
    """Recursive variant: walks lists/tuples and applies
    :func:`_inherit_common_if_default` to every leaf."""
    if isinstance(node, list):
        return [_inherit_common_in_pattern(child, recipe_common) for child in node]
    if isinstance(node, tuple):
        return tuple(_inherit_common_in_pattern(child, recipe_common) for child in node)
    return _inherit_common_if_default(node, recipe_common)


def _inherit_common_in_marker(marker: Any, recipe_common: CommonLayerConfig) -> Any:
    """Apply common-inheritance to an MTP marker and recurse into its
    ``mtp_model_layer`` body."""
    marker = _inherit_common_if_default(marker, recipe_common)
    if hasattr(marker, "mtp_model_layer"):
        new_body = _inherit_common_in_pattern(marker.mtp_model_layer, recipe_common)
        marker = dataclasses.replace(marker, mtp_model_layer=new_body)
    return marker


def _split_pattern(pattern: list):
    """Split ``[Embedding, ...decoder..., MTP*, Loss]`` into its four pieces.

    Trailing :class:`MTPLayerConfig` markers (zero or more) sit between the
    decoder body and the :class:`CrossEntropyLayerConfig` marker. The
    returned tuple is ``(embedding, decoder_body, mtp_markers, loss)``;
    ``mtp_markers`` is an empty list when the pattern has no MTP.

    Validates that exactly one :class:`EmbeddingLayerConfig` appears at the
    start, exactly one :class:`CrossEntropyLayerConfig` at the end, and
    raises a clear error if a :class:`PipelineSplit` appears (until PP is
    plumbed through).
    """
    if not isinstance(pattern, list) or not pattern:
        raise TypeError("layer_pattern must be a non-empty list.")

    if not isinstance(pattern[0], EmbeddingLayerConfig):
        raise TypeError(
            "layer_pattern must begin with an EmbeddingLayerConfig; got "
            f"{type(pattern[0]).__name__}."
        )
    if not isinstance(pattern[-1], CrossEntropyLayerConfig):
        raise TypeError(
            "layer_pattern must end with a CrossEntropyLayerConfig; got "
            f"{type(pattern[-1]).__name__}."
        )

    body = pattern[1:-1]

    # Pop trailing MTPLayerConfig markers.
    mtp_markers: list = []
    while body and isinstance(body[-1], MTPLayerConfig):
        mtp_markers.insert(0, body.pop())

    # PipelineSplit anywhere in the body → not yet supported.
    # Embedding/Loss/MTP in the decoder body → wrong slot.
    def _walk(node):
        if isinstance(node, PipelineSplit):
            raise NotImplementedError(
                "PipelineSplit() in the Python layer_pattern is not yet supported. "
                "Use the legacy string DSL (hybrid_layer_pattern with '|' separators) "
                "for pipeline parallelism, or wait for the follow-up that wires "
                "PipelineSplit through the recipe pipeline."
            )
        if isinstance(node, (EmbeddingLayerConfig, CrossEntropyLayerConfig)):
            raise TypeError(
                "EmbeddingLayerConfig / CrossEntropyLayerConfig may only appear "
                "at the start / end of layer_pattern, never in the body."
            )
        if isinstance(node, MTPLayerConfig):
            raise TypeError(
                "MTPLayerConfig markers must form a contiguous trailing run "
                "immediately before the CrossEntropyLayerConfig — they cannot "
                "appear earlier in the decoder body."
            )
        if isinstance(node, (list, tuple)):
            for child in node:
                _walk(child)

    for entry in body:
        _walk(entry)

    return pattern[0], body, mtp_markers, pattern[-1]


_MTP_OVERRIDE_GUIDANCE = (
    "MTPLayerConfig only forwards the layer-symbol pattern to "
    "MultiTokenPredictionBlock today; per-MTP-layer config overrides are not yet "
    "plumbed through and would be silently ignored. Drop the override or wait for "
    "the MTP follow-up that wires per-layer configs through."
)


def _ensure_mtp_lc_uses_defaults(lc: LayerConfig, recipe_common: CommonLayerConfig) -> None:
    """Reject per-MTP-layer LayerConfig overrides that would be silently dropped."""
    if lc.common_config != recipe_common:
        raise ValueError(
            f"{type(lc).__name__} inside an MTPLayerConfig.mtp_model_layer body "
            f"has a custom common_config. {_MTP_OVERRIDE_GUIDANCE}"
        )
    if lc.extra:
        raise ValueError(
            f"{type(lc).__name__} inside an MTPLayerConfig.mtp_model_layer body "
            f"has a non-empty extra={lc.extra!r}. {_MTP_OVERRIDE_GUIDANCE}"
        )
    default_lc = type(lc)(common_config=recipe_common)
    for f in dataclasses.fields(type(lc)):
        if f.name in ("common_config", "extra"):
            continue
        if getattr(lc, f.name) != getattr(default_lc, f.name):
            raise ValueError(
                f"{type(lc).__name__}.{f.name}={getattr(lc, f.name)!r} inside an "
                f"MTPLayerConfig.mtp_model_layer body diverges from the default. "
                f"{_MTP_OVERRIDE_GUIDANCE}"
            )


def _compile_mtp_markers(mtp_markers: list, recipe_common: CommonLayerConfig):
    """Compile trailing MTP markers into stack-level MTP plumbing.

    Returns a 4-tuple ``(mtp_layer_pattern, mtp_num_depths, loss_scaling_factor,
    use_repeated_layer)``. The existing :class:`MultiTokenPredictionBlock`
    infrastructure assumes all MTP depths share an identical body and
    stack-level settings; we enforce that here: every
    :class:`MTPLayerConfig` in ``mtp_markers`` must produce the same
    flattened symbol string and agree on ``loss_scaling_factor`` /
    ``use_repeated_layer``. ``loss_scaling_factor`` and
    ``use_repeated_layer`` are ``None`` when no marker sets them (caller
    should fall back to the :class:`TransformerConfig` defaults). When
    ``mtp_markers`` is empty, returns ``(None, 0, None, None)``.

    Also rejects per-MTP-layer config overrides that would otherwise be
    silently dropped (see :data:`_MTP_OVERRIDE_GUIDANCE`).
    """
    from megatron.core.models.hybrid.layer_pattern import flatten_decoder_pattern

    if not mtp_markers:
        return None, 0, None, None

    bodies: List[str] = []
    for marker in mtp_markers:
        if marker.common_config != recipe_common:
            raise ValueError(
                f"MTPLayerConfig.common_config diverges from the recipe's "
                f"common_config. {_MTP_OVERRIDE_GUIDANCE}"
            )
        flat = flatten_decoder_pattern(marker.mtp_model_layer)
        if not flat:
            raise ValueError(
                "MTPLayerConfig.mtp_model_layer must contain at least one " "decoder LayerConfig."
            )
        for lc in flat:
            _ensure_mtp_lc_uses_defaults(lc, recipe_common)
        bodies.append("".join(type(lc).SYMBOL for lc in flat))

    shared = bodies[0]
    for i, body in enumerate(bodies[1:], start=2):
        if body != shared:
            raise ValueError(
                f"All MTPLayerConfig markers in the pattern must share an "
                f"identical mtp_model_layer body. Marker 1 compiles to "
                f"{shared!r} but marker {i} compiles to {body!r}."
            )

    loss_scaling = _agree_across_mtp_markers(mtp_markers, "loss_scaling_factor")
    use_repeated = _agree_across_mtp_markers(mtp_markers, "use_repeated_layer")
    return shared, len(mtp_markers), loss_scaling, use_repeated


def _agree_across_mtp_markers(mtp_markers: list, field_name: str) -> Any:
    """Return the shared value of ``field_name`` across markers, or raise.

    Stack-level MTP settings (``loss_scaling_factor``, ``use_repeated_layer``)
    cannot meaningfully vary per-depth: there is one
    :class:`MultiTokenPredictionBlock` and one auxiliary loss term.
    Either every marker leaves the field at its default ``None`` (in which
    case we return ``None`` and let TC defaults apply), or every marker
    that sets it agrees on the value.
    """
    values = [getattr(m, field_name) for m in mtp_markers]
    non_none = {v for v in values if v is not None}
    if len(non_none) > 1:
        raise ValueError(
            f"MTPLayerConfig.{field_name} must be identical across all MTP markers "
            f"(it is a stack-level setting). Found values: {sorted(non_none)}."
        )
    return next(iter(non_none), None)


def _infer_uniform_attention_metadata(decoder_flat: List[LayerConfig]) -> dict[str, Any]:
    """Infer model-wide attention geometry from attention-like layers.

    ``TransformerConfig`` still requires attention-shaped fields even for
    non-attention layer configs. Those requirements are an implementation
    artifact of the current lowering target, not DSL concepts recipe authors
    should have to spell in ``CommonLayerConfig.extra``.

    When all attention/DSA layers in the pattern agree on a field, expose that
    value to the compiler as a private placeholder. If attention layers are
    heterogeneous, return only the fields that are actually uniform; the
    remaining TransformerConfig invariants fall back to minimal placeholders.
    """

    attention_layers = [
        lc for lc in decoder_flat if isinstance(lc, (AttentionLayerConfig, DSALayerConfig))
    ]
    if not attention_layers:
        return {}

    candidate_fields = ("num_attention_heads", "num_query_groups", "kv_channels")
    metadata: dict[str, Any] = {}
    for field_name in candidate_fields:
        values = [getattr(lc, field_name) for lc in attention_layers]
        if any(v is None for v in values):
            continue
        first = values[0]
        if all(v == first for v in values[1:]):
            metadata[field_name] = first
    return metadata


def _reject_heterogeneous_attention_geometry(
    decoder_flat: List[LayerConfig], attention_metadata: dict[str, Any]
) -> None:
    """Raise if attention layers under RoPE/YARN disagree on rotary geometry.

    HybridModel builds a single global rotary embedding from the stack TC's
    ``num_attention_heads`` / ``kv_channels``. When per-layer attention
    configs diverge on those fields, the rotary tensor is sized for the
    placeholder and runtime hits shape mismatches. Until per-layer rotary
    lands, force recipes under RoPE/YARN to use uniform attention geometry.
    """

    attention_layers = [
        lc for lc in decoder_flat if isinstance(lc, (AttentionLayerConfig, DSALayerConfig))
    ]
    if not attention_layers:
        return
    for field_name in ("num_attention_heads", "num_query_groups", "kv_channels"):
        # If inference produced this field, all layers agree; skip.
        if field_name in attention_metadata:
            continue
        values = [getattr(lc, field_name) for lc in attention_layers]
        non_none = {v for v in values if v is not None}
        if len(non_none) > 1:
            raise NotImplementedError(
                f"Heterogeneous attention geometry under RoPE/YARN is not "
                f"supported: attention layers disagree on {field_name!r} "
                f"(values: {sorted(non_none)}). HybridModel builds one global "
                f"rotary embedding from the stack TC and passes it to every "
                f"attention layer; per-layer rotary support is a follow-up. "
                f"Use uniform attention geometry, or position_embedding_type="
                f"\"none\" / \"learned_absolute\"."
            )
