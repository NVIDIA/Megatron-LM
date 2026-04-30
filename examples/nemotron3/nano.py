# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.

"""Nemotron-3 Nano model recipe expressed in the HybridModel Python DSL.

A 52-layer hybrid model interleaving Mamba SSM, attention, and MoE layers.
Faithful 1:1 with the canonical
``Megatron-Bridge/recipes/nemotronh/nemotron_3_nano.py::nemotron_3_nano_pretrain_config``.

Recipe entry point: :func:`make_recipe`. ``num_layers`` is intentionally
**not** set anywhere — it is derived at compile time from the flattened
decoder length of ``layer_pattern``.

Use this recipe via the ``--model-recipe`` flag in any of these forms::

    --model-recipe examples.nemotron3.nano
    --model-recipe examples.nemotron3.nano:make_recipe
    --model-recipe /path/to/this/file.py
    --model-recipe /path/to/this/file.py:make_recipe

Run this file directly to print the resulting ``layer_type_list``::

    uv run python -m examples.nemotron3.nano
"""

from megatron.core.models.hybrid import (
    AttentionLayerConfig,
    CommonLayerConfig,
    CrossEntropyLayerConfig,
    EmbeddingLayerConfig,
    HybridModelConfig,
    MambaLayerConfig,
    MoELayerConfig,
    flatten_decoder_pattern,
)

VOCAB_SIZE: int = 131072
MAX_SEQUENCE_LENGTH: int = 8192


# --- Common (model-wide) defaults ------------------------------------------

#
# Each field below either differs from the TransformerConfig default OR is
# unique to Nemotron-3 Nano. TC-default-matching fields (``apply_rope_fusion``,
# ``gated_linear_unit``, ``transformer_impl``,
# ``cuda_graph_impl``, ``cuda_graph_scope``, ``cuda_graph_warmup_steps``,
# ``moe_layer_freq``, ``moe_router_bias_update_rate``,
# ``mtp_loss_scaling_factor``, ``overlap_p2p_comm``) are intentionally
# omitted — they fall through to TC's own defaults at compile time.
#
# ``is_hybrid_model=True`` is also intentionally absent — every HybridModel
# recipe is a hybrid model by construction; ``HybridModelConfig.compile()``
# forces it on every per-layer TC.
#
# ``make_vocab_size_divisible_by=128`` from the canonical config is NOT a
# TransformerConfig field — it's a launcher-side tokenizer pad. The DSL
# expresses it by setting ``EmbeddingLayerConfig.vocab_size`` to the already-
# padded value (131072 below = 1024 * 128).
common_config = CommonLayerConfig(
    # Architecture
    hidden_size=2688,
    ffn_hidden_size=1856,
    # Precision
    mixed_precision_dtype="bf16",
    first_last_layers_bf16=True,
    # Layer-level parallelism (TP/CP/EP sizes live on HybridModelConfig).
    sequence_parallel=True,
    # Initialisation
    init_method_std=0.0173,
    # Norms
    normalization="RMSNorm",
    # Bias / activation
    add_bias_linear=False,
    activation_func="squared_relu",
    # Fusions
    persist_layer_norm=True,
    # ``tp_comm_overlap`` is not a curated DSL field today; the Bridge config
    # sets it via CommOverlapConfig. Use ``extra`` to project it onto every
    # per-layer TransformerConfig (matches the legacy ``--tp-comm-overlap``
    # CLI flag).
    extra={"tp_comm_overlap": True},
)


# --- Per-layer configs -----------------------------------------------------

Embedding = EmbeddingLayerConfig(
    common_config=common_config,
    vocab_size=VOCAB_SIZE,
    max_sequence_length=MAX_SEQUENCE_LENGTH,
    position_embedding_type="none",
)

Mamba = MambaLayerConfig(
    common_config=common_config, head_dim=64, state_size=128, num_groups=8, num_heads=64
)

Att = AttentionLayerConfig(
    common_config=common_config,
    num_attention_heads=32,
    num_query_groups=2,
    kv_channels=128,
    attention_softmax_in_fp32=False,
    masked_softmax_fusion=True,
    attention_backend="fused",
)

MoE = MoELayerConfig(
    common_config=common_config,
    num_experts=128,
    top_k=6,
    ffn_hidden_size=1856,
    router_score_function="sigmoid",
    router_load_balancing_type="seq_aux_loss",
    router_topk_scaling_factor=2.5,
    router_enable_expert_bias=True,
    router_dtype="fp32",
    aux_loss_coeff=1e-4,
    shared_expert_intermediate_size=3712,
    # DeepEP-backed flex dispatcher (final canonical value).
    token_dispatcher_type="flex",
    flex_dispatcher_backend="deepep",
    hybridep_num_sms=16,
    grouped_gemm=True,
    permute_fusion=True,
    router_num_groups=1,
    router_group_topk=1,
    router_fusion=False,
    shared_expert_overlap=False,
    use_fused_weighted_squared_relu=True,
)

Loss = CrossEntropyLayerConfig(loss_fusion=True, fusion_impl="native")


# --- Layer pattern composition ---------------------------------------------
#
# Decoder layout (52 layers total = 6 + 28 + 9 + 9). ``num_layers`` is never
# set explicitly; HybridModelConfig.compile() derives it from the flattened
# pattern length.

stage0 = [Mamba] + [MoE, Mamba] * 2 + [Att]
stage1 = [MoE, Mamba] * 3 + [Att]
stage2 = [MoE, Mamba] * 4 + [Att]
stage3 = [MoE, Mamba] * 4 + [MoE]

decoder = stage0 + stage1 * 4 + stage2 + stage3

layer_pattern = [Embedding] + decoder + [Loss]


# --- Recipe entry point ----------------------------------------------------


def make_recipe() -> HybridModelConfig:
    """Return the Nemotron-3 Nano HybridModel pretrain recipe.

    ``make_recipe`` is the canonical default entry point used when
    ``--model-recipe`` is passed without a ``:func`` suffix.
    """
    return HybridModelConfig(
        common_config=common_config,
        layer_pattern=layer_pattern,
        untie_embeddings_and_output_weights=True,
        # Model-level parallelism (PP intentionally absent — the recipe DSL
        # is PP-free; recipes that need PP must use the legacy string DSL).
        tensor_model_parallel_size=4,
        expert_model_parallel_size=8,
        expert_tensor_parallel_size=1,
    )


def nemotron_3_nano_pretrain_config():
    """Bridge-style alias retained for explicit ``:func`` selection."""
    return make_recipe()


if __name__ == "__main__":
    # Compile the recipe and print the resulting layer_type_list. This is
    # the same code path ``--model-recipe`` exercises in production via
    # ``load_recipe``.
    #
    # The full :meth:`HybridModelConfig.compile` requires Transformer Engine
    # (``moe_permute_fusion=True``, ``transformer_impl="transformer_engine"``,
    # etc. trigger TE checks at config-construction time). In dev environments
    # without TE, fall back to the pure-Python ``flatten_decoder_pattern`` +
    # ``SYMBOL`` derivation so the smoke test can still verify the pattern
    # composition.
    recipe = make_recipe()
    try:
        layer_type_list = recipe.compile().layer_type_list
    except (ValueError, ImportError, ModuleNotFoundError) as e:
        msg = str(e)
        if "fused permutation" not in msg and "TE" not in msg and "transformer_engine" not in msg:
            raise
        print(
            f"NOTE: full compile() needs Transformer Engine "
            f"({type(e).__name__}: {e}). Falling back to pattern-only "
            f"verification — production runs go through compile()."
        )
        decoder_flat = flatten_decoder_pattern(recipe.layer_pattern[1:-1])
        layer_type_list = [type(lc).SYMBOL for lc in decoder_flat]
    composed_string = "".join(layer_type_list)

    print(f"Composed layer pattern: {composed_string}")
    print(f"Layer count:            {len(layer_type_list)}")
