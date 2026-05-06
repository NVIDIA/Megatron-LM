# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.

"""Smoke-test recipe for the HybridModel Python DSL construction path.

A small Mamba / Attention / MLP hybrid that exercises
``HybridModel(config: HybridModelConfig, ...)`` end-to-end through the
real training launcher (``pretrain_hybrid.py --model-recipe``). The CI
test that consumes this recipe trains for ~50 iterations and locks loss /
gradient-norm golden values on the first successful run; subsequent runs
detect numerical regressions.

The recipe is deliberately small (12 decoder layers, hidden_size=1024) so
the test runs quickly. Architecture style mirrors the existing
``hybrid_mr_mcore_te_tp1_pp1_cp1_dgx_a100_1N8G`` legacy-string-DSL
functional test, but the entry point is the new constructor surface.
"""

from megatron.core.models.hybrid import (
    AttentionLayerConfig,
    CommonLayerConfig,
    CrossEntropyLayerConfig,
    EmbeddingLayerConfig,
    HybridModelConfig,
    MambaLayerConfig,
    MLPLayerConfig,
)

# Model-wide defaults shared by every layer below.
common = CommonLayerConfig(
    hidden_size=1024,
    mixed_precision_dtype="bf16",
    normalization="RMSNorm",
    add_bias_linear=False,
    activation_func="silu",
    gated_linear_unit=True,
)

# Pattern markers — embedding at the head, loss at the tail.
Embedding = EmbeddingLayerConfig(
    common_config=common,
    vocab_size=131072,
    max_sequence_length=1024,
    position_embedding_type="none",
)
Loss = CrossEntropyLayerConfig()

# Layer building blocks. Mamba uses CommonLayerConfig defaults; Attention
# pins the GQA group count to match the legacy functional test.
Mamba = MambaLayerConfig(common_config=common)
Attn = AttentionLayerConfig(
    common_config=common, num_attention_heads=16, num_query_groups=8, attention_softmax_in_fp32=True
)
MLP = MLPLayerConfig(common_config=common)

# 12-layer decoder: two repeats of [Mamba, MLP, Mamba, MLP, Attn, MLP].
# Plain Python list operations replace the legacy string DSL.
decoder = [Mamba, MLP, Mamba, MLP, Attn, MLP] * 2
layer_pattern = [Embedding] + decoder + [Loss]


def make_recipe() -> HybridModelConfig:
    """Recipe entry point consumed by ``--model-recipe``."""
    return HybridModelConfig(
        common_config=common,
        layer_pattern=layer_pattern,
        # Match HybridModel's legacy default
        # (``share_embeddings_and_output_weights=False``).
        untie_embeddings_and_output_weights=True,
    )
