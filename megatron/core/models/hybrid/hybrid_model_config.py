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
    untie_embeddings_and_output_weights: bool
    fp16_lm_cross_entropy: bool
    parallel_output: bool

    # Recipe topology pins, surfaced None-able so the launcher's args
    # projection can distinguish "recipe pinned this value" from "recipe
    # left it unset, defer to the CLI flag". ``compiled.config`` always
    # carries concrete substituted values (TC needs them); these mirror
    # the recipe's authorial intent, not the substituted runtime values.
    tensor_model_parallel_size: Optional[int]
    context_parallel_size: Optional[int]
    expert_model_parallel_size: Optional[int]
    expert_tensor_parallel_size: Optional[int]


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
    list/tuple structure; :func:`compile` flattens them. Pipeline parallelism
    is not part of the recipe DSL — recipes that need PP must use the legacy
    ``hybrid_layer_pattern`` string DSL."""

    untie_embeddings_and_output_weights: bool = False
    """Untie input embedding and output projection weights."""

    # === Model-level parallelism ===
    # These live here (not on :class:`CommonLayerConfig`) because they are
    # job-/model-level concerns: process groups are constructed once, not
    # per-layer, and they cannot meaningfully vary per layer. ``compile()``
    # injects them into every per-layer TransformerConfig and the
    # stack-level config. PP is intentionally absent — the recipe DSL is
    # PP-free; recipes that need PP must use the legacy string DSL.
    #
    # All four default to ``None``, meaning "do not pin from the recipe".
    # When unset, ``_apply_model_recipe_to_args`` skips projecting them onto
    # ``args``, so launcher CLI flags (``--tensor-model-parallel-size`` etc.)
    # win. Recipes for topology-bound models (e.g. Nemotron-3 Nano needs
    # TP=4/EP=8 to fit) should pin these explicitly; ad-hoc / debugging
    # recipes can leave them unset and let the launcher decide.
    tensor_model_parallel_size: Optional[int] = None
    """Tensor-model-parallel world size; ``None`` = use ``--tensor-model-parallel-size``."""

    context_parallel_size: Optional[int] = None
    """Context-parallel world size; ``None`` = use ``--context-parallel-size``."""

    expert_model_parallel_size: Optional[int] = None
    """Expert-model-parallel world size (used by MoE layers); ``None`` = use
    ``--expert-model-parallel-size``."""

    expert_tensor_parallel_size: Optional[int] = None
    """Expert tensor-parallel size; ``None`` = use ``--expert-tensor-parallel-size``
    (which itself defaults to ``tensor_model_parallel_size``)."""

    # === Stack-TC MoE metadata override (heterogeneous-MoE recipes only) ===
    # The stack-level TransformerConfig carries one ``num_moe_experts`` /
    # ``moe_ffn_hidden_size`` pair. With homogeneous MoE the value is
    # unambiguous; with heterogeneous MoE there is no single right answer.
    # Two consumers read these fields from the stack TC: MTP's MoE body and
    # the inference text-generation capacity-factor calculation. Recipes
    # that mix MoE shapes must pin the values they want those consumers to
    # see — ``compile()`` raises if MoE is heterogeneous and these are unset.
    stack_moe_num_experts: Optional[int] = None
    """Stack-TC ``num_moe_experts`` for heterogeneous-MoE recipes.
    Required when MoE layers disagree on ``num_experts``; ignored otherwise."""

    stack_moe_ffn_hidden_size: Optional[int] = None
    """Stack-TC ``moe_ffn_hidden_size`` for heterogeneous-MoE recipes.
    Required when MoE layers disagree on ``ffn_hidden_size``; ignored
    otherwise. ``None`` falls through to ``ffn_hidden_size``."""

    def compile(self) -> CompiledRecipe:
        """Process the layer pattern into a :class:`CompiledRecipe`.

        Extracts the embedding/loss markers, validates pattern structure,
        flattens decoder layers, derives ``num_layers`` from the flattened
        count, and produces per-layer :class:`TransformerConfig` instances.
        """
        from megatron.core.models.hybrid.layer_pattern import flatten_decoder_pattern

        embedding, decoder_leaves, loss = _split_pattern(self.layer_pattern)

        # Auto-inherit the recipe's common_config into any layer/marker that
        # was constructed without an explicit ``common_config=`` argument.
        # Without this, a layer like ``MambaLayerConfig(head_dim=64)`` carries
        # a default-constructed CommonLayerConfig with ``hidden_size=0``,
        # silently producing an invalid model.
        embedding = _inherit_common_if_default(embedding, self.common_config)
        loss = _inherit_common_if_default(loss, self.common_config)
        decoder_leaves = _inherit_common_in_pattern(decoder_leaves, self.common_config)

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

        # Topology fields default to ``None`` on the recipe (= "let the
        # launcher decide"). For per-layer TC construction we still need
        # concrete values, so substitute TC defaults (1 / TP) here. The
        # ``None``-ness is preserved on the recipe itself for the args
        # projection, which only writes to ``args.<field>`` when the recipe
        # pinned a value.
        tp = self.tensor_model_parallel_size if self.tensor_model_parallel_size is not None else 1
        cp = self.context_parallel_size if self.context_parallel_size is not None else 1
        ep = self.expert_model_parallel_size if self.expert_model_parallel_size is not None else 1
        etp = (
            self.expert_tensor_parallel_size if self.expert_tensor_parallel_size is not None else tp
        )

        # Universal parallelism — flows into every per-layer TC. PP is
        # intentionally absent: the recipe DSL is PP-free. ``is_hybrid_model``
        # is implicit for any HybridModelConfig recipe; force it on every TC
        # so users don't carry a redundant ``=True``.
        parallelism: dict = {
            "tensor_model_parallel_size": tp,
            "context_parallel_size": cp,
            "is_hybrid_model": True,
        }

        # MoE-only parallelism — applied to MoE per-layer TCs (after their
        # base build) and to the stack-level TC. Non-MoE per-layer TCs do
        # not receive these fields; TransformerConfig's defaults
        # (``expert_model_parallel_size=1``, ``expert_tensor_parallel_size``
        # auto-derived from TP) take over, which is correct because those
        # layers don't participate in expert parallelism.
        expert_parallelism: dict = {
            "expert_model_parallel_size": ep,
            "expert_tensor_parallel_size": etp,
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

        placeholders: dict = {"num_attention_heads": max(1, tp)}
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

        # Stack-level config: a model-wide topology snapshot used for the
        # final norm, embedding init, inference capacity sizing, and the
        # args projection in ``_apply_model_recipe_to_args``. Carries EP/ETP
        # because it represents the global topology, even though non-MoE
        # per-layer TCs do not.
        stack_kwargs = embedding.common_config.to_transformer_config_kwargs()
        stack_kwargs.update(parallelism)
        stack_kwargs.update(expert_parallelism)
        for k, v in attention_metadata.items():
            stack_kwargs.setdefault(k, v)
        for k, v in placeholders.items():
            stack_kwargs.setdefault(k, v)
        # Derive stack-level MoE metadata from the recipe's actual MoE
        # configs rather than using a placeholder. Two consumers read these
        # fields semantically: the MTP block construction path (when a body
        # symbol contains ``E``, the MoE layer inside MTP is built from the
        # stack TC) and the inference capacity-factor calculation
        # (``capacity_factor = num_moe_experts / moe_router_topk``).
        stack_moe_experts, stack_moe_ffn_hidden_size = _derive_stack_moe_metadata(
            decoder_flat,
            override_num_experts=self.stack_moe_num_experts,
            override_ffn_hidden_size=self.stack_moe_ffn_hidden_size,
        )
        if stack_moe_experts is not None:
            stack_kwargs["num_moe_experts"] = stack_moe_experts
            if stack_moe_ffn_hidden_size is not None:
                stack_kwargs["moe_ffn_hidden_size"] = stack_moe_ffn_hidden_size
        elif ep > 1:
            raise ValueError(
                "expert_model_parallel_size > 1 requires at least one "
                "MoELayerConfig in the layer_pattern; the recipe has none. "
                "Either drop expert parallelism, or add MoE layers."
            )
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
        # Marker-level passthroughs: Embedding and CrossEntropy markers may
        # set TransformerConfig fields the curated DSL surface doesn't cover.
        # Same shadowing rule as CommonLayerConfig.extra and LayerConfig.extra:
        # ``extra`` cannot name a curated field on the same marker.
        from megatron.core.models.hybrid.common_layer_config import validate_extra_kwargs

        if embedding.extra:
            validate_extra_kwargs(embedding.extra, "EmbeddingLayerConfig.extra")
            curated = {
                f.name
                for f in dataclasses.fields(embedding)
                if f.name not in ("common_config", "extra")
            }
            shadowed = sorted(set(embedding.extra) & curated)
            if shadowed:
                raise ValueError(
                    f"EmbeddingLayerConfig.extra cannot name curated fields: {shadowed}. "
                    f"Set them on the dataclass attribute directly."
                )
            stack_kwargs.update(embedding.extra)
        if loss.extra:
            validate_extra_kwargs(loss.extra, "CrossEntropyLayerConfig.extra")
            # CrossEntropy's curated fields are renamed when projected onto
            # TransformerConfig (``loss_fusion`` → ``cross_entropy_loss_fusion``
            # etc.); reject both the recipe-side dataclass names and the
            # TC-side names ``compile`` writes into ``stack_kwargs`` above.
            curated = {f.name for f in dataclasses.fields(loss) if f.name != "extra"} | {
                "cross_entropy_loss_fusion",
                "cross_entropy_fusion_impl",
                "calculate_per_token_loss",
            }
            shadowed = sorted(set(loss.extra) & curated)
            if shadowed:
                raise ValueError(
                    f"CrossEntropyLayerConfig.extra cannot name curated fields: {shadowed}. "
                    f"Set them on the dataclass attribute directly."
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
        if embedding.yarn:
            unknown = set(embedding.yarn) - set(_YARN_FIELDS)
            if unknown:
                raise ValueError(
                    f"EmbeddingLayerConfig.yarn has unknown keys {sorted(unknown)}; "
                    f"valid keys are {sorted(_YARN_FIELDS)}."
                )
            if embedding.position_embedding_type == "yarn":
                for name, value in embedding.yarn.items():
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
            untie_embeddings_and_output_weights=self.untie_embeddings_and_output_weights,
            fp16_lm_cross_entropy=loss.fp16_lm_cross_entropy,
            parallel_output=loss.parallel_output,
            tensor_model_parallel_size=self.tensor_model_parallel_size,
            context_parallel_size=self.context_parallel_size,
            expert_model_parallel_size=self.expert_model_parallel_size,
            expert_tensor_parallel_size=self.expert_tensor_parallel_size,
        )


# --- pattern-structure helpers --------------------------------------------


def _inherit_common_if_default(node: Any, recipe_common: CommonLayerConfig) -> Any:
    """If ``node`` carries no explicit common_config (left as ``None``),
    substitute ``recipe_common`` via :func:`dataclasses.replace`. An
    explicit :class:`CommonLayerConfig` (even a default-constructed one)
    is preserved — recipe authors who write ``CommonLayerConfig()``
    intentionally get the default values they asked for. Applies to
    LayerConfig instances and pattern markers."""
    if not hasattr(node, "common_config"):
        return node
    if node.common_config is None:
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


def _split_pattern(pattern: list):
    """Split ``[Embedding, ...decoder..., Loss]`` into its three pieces.

    The returned tuple is ``(embedding, decoder_body, loss)``.

    Validates that exactly one :class:`EmbeddingLayerConfig` appears at the
    start and exactly one :class:`CrossEntropyLayerConfig` at the end. MTP
    and pipeline parallelism are intentionally not part of the recipe DSL;
    recipes that need either should use the legacy ``hybrid_layer_pattern``
    string DSL.
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

    # Embedding/Loss in the decoder body → wrong slot.
    def _walk(node):
        if isinstance(node, (EmbeddingLayerConfig, CrossEntropyLayerConfig)):
            raise TypeError(
                "EmbeddingLayerConfig / CrossEntropyLayerConfig may only appear "
                "at the start / end of layer_pattern, never in the body."
            )
        if isinstance(node, (list, tuple)):
            for child in node:
                _walk(child)

    for entry in body:
        _walk(entry)

    return pattern[0], body, pattern[-1]


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


def _derive_stack_moe_metadata(
    decoder_flat: List[LayerConfig],
    override_num_experts: Optional[int],
    override_ffn_hidden_size: Optional[int],
) -> tuple[Optional[int], Optional[int]]:
    """Derive ``(num_moe_experts, moe_ffn_hidden_size)`` for the stack-level TC.

    The stack TC is a model-wide topology snapshot, not a per-layer config,
    but two consumers read its MoE fields semantically:
    :class:`MultiTokenPredictionBlock` (when a body symbol contains ``E``,
    the MoE layer inside MTP is built from the stack TC) and the inference
    text-generation controller's capacity-factor calculation. Three cases:

    1. No MoELayerConfig in the decoder → ``(None, None)``. The stack TC is
       dense; the caller is responsible for raising on ``EP > 1``.
    2. Homogeneous MoE → that value. The stack TC matches what every MoE
       layer in the recipe actually uses.
    3. Heterogeneous MoE → recipe must pin
       :attr:`HybridModelConfig.stack_moe_num_experts` (and optionally
       :attr:`stack_moe_ffn_hidden_size`). Without the pin we raise rather
       than silently picking a "max"-shaped value: MTP and the inference
       capacity-factor calc would otherwise consume a config that matches
       no actual layer, and per-consumer wrongness is hard to debug after
       the fact.
    """
    moe_layers = [lc for lc in decoder_flat if isinstance(lc, MoELayerConfig)]
    if not moe_layers:
        return None, None

    distinct_experts = {lc.num_experts for lc in moe_layers}
    distinct_ffn = {lc.ffn_hidden_size for lc in moe_layers if lc.ffn_hidden_size is not None}

    if len(distinct_experts) == 1:
        num_experts = next(iter(distinct_experts))
    elif override_num_experts is not None:
        num_experts = override_num_experts
    else:
        raise ValueError(
            f"Heterogeneous MoE recipe (num_experts values: {sorted(distinct_experts)}) "
            f"requires HybridModelConfig.stack_moe_num_experts to pin the value the "
            f"stack-level TransformerConfig should expose. The stack TC is read by "
            f"MTP and by the inference capacity-factor calculation; without an "
            f"explicit pin those consumers would see a config that matches no "
            f"actual MoE layer."
        )

    if len(distinct_ffn) <= 1:
        # Homogeneous (or all None) — pick the matching layer's value, or fall
        # through to ``ffn_hidden_size`` when every MoE layer leaves it unset.
        ffn_hidden_size = next(
            (
                lc.ffn_hidden_size
                for lc in moe_layers
                if lc.num_experts == num_experts and lc.ffn_hidden_size is not None
            ),
            None,
        )
    elif override_ffn_hidden_size is not None:
        ffn_hidden_size = override_ffn_hidden_size
    else:
        raise ValueError(
            f"Heterogeneous MoE recipe (ffn_hidden_size values: {sorted(distinct_ffn)}) "
            f"requires HybridModelConfig.stack_moe_ffn_hidden_size to pin the value the "
            f"stack-level TransformerConfig should expose."
        )

    return num_experts, ffn_hidden_size


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
