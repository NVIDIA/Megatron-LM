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
from dataclasses import dataclass, field
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

    def compile(self) -> CompiledRecipe:
        """Process the layer pattern into a :class:`CompiledRecipe`.

        Extracts the embedding/loss markers, validates pattern structure,
        flattens decoder layers, derives ``num_layers`` from the flattened
        count, and produces per-layer :class:`TransformerConfig` instances.
        """
        from megatron.core.models.hybrid.layer_pattern import flatten_decoder_pattern

        embedding, decoder_leaves, mtp_markers, loss = _split_pattern(self.layer_pattern)

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

        placeholders: dict = {
            "num_attention_heads": max(1, self.tensor_model_parallel_size),
        }
        # If the recipe contains a uniform attention geometry, use it as the
        # semantic placeholder for non-attention layer TransformerConfigs.
        # This keeps recipe authors out of TransformerConfig compatibility
        # details like "Mamba still needs num_attention_heads to satisfy
        # TC.__post_init__".
        placeholders.update(attention_metadata)

        layer_type_list = [type(lc).SYMBOL for lc in decoder_flat]
        layer_config_list: List[TransformerConfig] = []
        for lc in decoder_flat:
            tc = lc.to_transformer_config(
                num_layers, parallelism=parallelism, placeholders=placeholders
            )
            if isinstance(lc, MoELayerConfig):
                # ``dataclasses.replace`` re-runs TC.__post_init__, so the
                # EP > 1 / num_moe_experts invariant validates against the
                # MoE TC (which carries the real ``num_moe_experts``).
                tc = dataclasses.replace(tc, **expert_parallelism)
            layer_config_list.append(tc)

        # Reuse the existing pattern validator for the Attention/DSA mix rule.
        from megatron.core.models.hybrid.hybrid_layer_allocation import _validate_pattern

        _validate_pattern("".join(layer_type_list), "main", allow_pipe=False)

        # MTP: compile the per-depth body (shared across all MTP markers in
        # the pattern) into the single-character form the existing
        # MultiTokenPredictionBlock infrastructure consumes.
        mtp_layer_pattern, mtp_num_depths = _compile_mtp_markers(mtp_markers)

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
        stack_kwargs.setdefault("num_attention_heads", 1)
        if self.expert_model_parallel_size > 1:
            stack_kwargs.setdefault("num_moe_experts", 1)
        stack_kwargs["num_layers"] = num_layers
        stack_kwargs["cross_entropy_loss_fusion"] = loss.loss_fusion
        stack_kwargs["cross_entropy_fusion_impl"] = loss.fusion_impl
        stack_kwargs["calculate_per_token_loss"] = loss.calculate_per_token_loss
        # Marker-level passthroughs: Embedding and CrossEntropy markers may
        # set TransformerConfig fields the curated DSL surface doesn't cover.
        from megatron.core.models.hybrid.common_layer_config import validate_extra_kwargs

        if embedding.extra:
            validate_extra_kwargs(embedding.extra, "EmbeddingLayerConfig.extra")
            stack_kwargs.update(embedding.extra)
        if loss.extra:
            validate_extra_kwargs(loss.extra, "CrossEntropyLayerConfig.extra")
            stack_kwargs.update(loss.extra)
        stack_kwargs = {k: v for k, v in stack_kwargs.items() if v is not None}
        stack_config = TransformerConfig(**stack_kwargs)

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
        )


# --- pattern-structure helpers --------------------------------------------


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


def _compile_mtp_markers(mtp_markers: list):
    """Compile trailing MTP markers into ``(mtp_layer_pattern, mtp_num_depths)``.

    The existing :class:`MultiTokenPredictionBlock` infrastructure assumes all
    MTP depths share an identical body. We enforce the same constraint here:
    every :class:`MTPLayerConfig` in ``mtp_markers`` must produce the same
    flattened symbol string. The returned ``mtp_layer_pattern`` is that
    shared single-character body string (e.g. ``"MM"``); ``mtp_num_depths``
    is ``len(mtp_markers)``. When ``mtp_markers`` is empty, returns
    ``(None, 0)``.
    """
    from megatron.core.models.hybrid.layer_pattern import flatten_decoder_pattern

    if not mtp_markers:
        return None, 0

    bodies: List[str] = []
    for marker in mtp_markers:
        flat = flatten_decoder_pattern(marker.mtp_model_layer)
        if not flat:
            raise ValueError(
                "MTPLayerConfig.mtp_model_layer must contain at least one "
                "decoder LayerConfig."
            )
        bodies.append("".join(type(lc).SYMBOL for lc in flat))

    shared = bodies[0]
    for i, body in enumerate(bodies[1:], start=2):
        if body != shared:
            raise ValueError(
                f"All MTPLayerConfig markers in the pattern must share an "
                f"identical mtp_model_layer body. Marker 1 compiles to "
                f"{shared!r} but marker {i} compiles to {body!r}."
            )
    return shared, len(mtp_markers)


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
