# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Nemotron-3 Nano model recipe expressed in the HybridModel Python DSL.

The recipe defines the same 52-layer hybrid model as the legacy string DSL
pattern used by :mod:`examples.nemotron3.nano.sh`:

``MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME``

Use it with ``pretrain_hybrid.py --model-recipe examples.nemotron3.nano``.
The SLURM launcher can override topology with ``NANO_TP``, ``NANO_CP``,
``NANO_EP``, ``NANO_ETP``, and ``NANO_SP``.
"""

import os

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


LEGACY_HYBRID_LAYER_PATTERN = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
VOCAB_SIZE = 131072
MAX_SEQUENCE_LENGTH = 8192


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "t", "yes", "y", "on"}


def make_recipe() -> HybridModelConfig:
    """Return the Nemotron-3 Nano HybridModel pretrain recipe."""

    common_config = CommonLayerConfig(
        hidden_size=2688,
        ffn_hidden_size=1856,
        mixed_precision_dtype="bf16",
        first_last_layers_bf16=True,
        sequence_parallel=_env_bool("NANO_SP", True),
        init_method_std=0.0173,
        normalization="RMSNorm",
        add_bias_linear=False,
        activation_func="squared_relu",
        persist_layer_norm=True,
        cuda_graph_impl="none",
    )

    embedding = EmbeddingLayerConfig(
        common_config=common_config,
        vocab_size=_env_int("NANO_VOCAB_SIZE", VOCAB_SIZE),
        max_sequence_length=_env_int("NANO_MAX_SEQUENCE_LENGTH", MAX_SEQUENCE_LENGTH),
        position_embedding_type="none",
        extra={
            "moe_router_topk": 6,
            "moe_router_topk_scaling_factor": 2.5,
            "moe_router_num_groups": 1,
            "moe_router_group_topk": 1,
            "moe_router_score_function": "sigmoid",
            "moe_router_enable_expert_bias": True,
            "moe_router_load_balancing_type": "seq_aux_loss",
            "moe_router_dtype": "fp32",
            "moe_aux_loss_coeff": 0.0001,
            "moe_shared_expert_intermediate_size": 3712,
            "moe_token_dispatcher_type": "flex",
            "moe_flex_dispatcher_backend": "deepep",
            "moe_hybridep_num_sms": 16,
            "moe_grouped_gemm": True,
            "moe_permute_fusion": True,
            "use_fused_weighted_squared_relu": True,
        },
    )
    mamba = MambaLayerConfig(
        common_config=common_config,
        head_dim=64,
        state_size=128,
        num_groups=8,
        num_heads=64,
    )
    attention = AttentionLayerConfig(
        common_config=common_config,
        num_attention_heads=32,
        num_query_groups=2,
        kv_channels=128,
        attention_softmax_in_fp32=False,
        masked_softmax_fusion=True,
        attention_backend="fused",
    )
    moe = MoELayerConfig(
        common_config=common_config,
        num_experts=128,
        top_k=6,
        ffn_hidden_size=1856,
        router_score_function="sigmoid",
        router_load_balancing_type="seq_aux_loss",
        router_topk_scaling_factor=2.5,
        router_enable_expert_bias=True,
        router_dtype="fp32",
        aux_loss_coeff=0.0001,
        shared_expert_intermediate_size=3712,
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
    loss = CrossEntropyLayerConfig(loss_fusion=True, fusion_impl="native")

    stage0 = [mamba] + [moe, mamba] * 2 + [attention]
    stage1 = [moe, mamba] * 3 + [attention]
    stage2 = [moe, mamba] * 4 + [attention]
    stage3 = [moe, mamba] * 4 + [moe]
    decoder = stage0 + stage1 * 4 + stage2 + stage3

    return HybridModelConfig(
        common_config=common_config,
        layer_pattern=[embedding] + decoder + [loss],
        untie_embeddings_and_output_weights=True,
        tensor_model_parallel_size=_env_int("NANO_TP", 4),
        context_parallel_size=_env_int("NANO_CP", 1),
        expert_model_parallel_size=_env_int("NANO_EP", 8),
        expert_tensor_parallel_size=_env_int("NANO_ETP", 1),
    )


def nemotron_3_nano_pretrain_config() -> HybridModelConfig:
    """Bridge-style alias for explicit ``--model-recipe ...:func`` selection."""

    return make_recipe()


if __name__ == "__main__":
    recipe = make_recipe()
    try:
        layer_type_list = recipe.compile().layer_type_list
    except (ValueError, ImportError, ModuleNotFoundError) as exc:
        print(
            "NOTE: full compile() failed "
            f"({type(exc).__name__}: {exc}). Falling back to pattern-only verification."
        )
        decoder_flat = flatten_decoder_pattern(recipe.layer_pattern[1:-1])
        layer_type_list = [type(layer_config).SYMBOL for layer_config in decoder_flat]

    composed = "".join(layer_type_list)
    print(f"Composed layer pattern: {composed}")
    print(f"Layer count:            {len(layer_type_list)}")
    print(f"Legacy reference:       {LEGACY_HYBRID_LAYER_PATTERN}")
    print(f"Match:                  {composed == LEGACY_HYBRID_LAYER_PATTERN}")
    if composed != LEGACY_HYBRID_LAYER_PATTERN:
        raise SystemExit("Composed pattern does not match the legacy string DSL pattern.")
