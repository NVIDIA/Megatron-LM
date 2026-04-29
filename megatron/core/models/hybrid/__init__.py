# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.

"""Public surface for the HybridModel Python layer-pattern DSL.

A complete recipe looks like:

.. code-block:: python

    from megatron.core.models.hybrid import (
        AttentionLayerConfig,
        CommonLayerConfig,
        CrossEntropyLayerConfig,
        EmbeddingLayerConfig,
        HybridModelConfig,
        MambaLayerConfig,
        MoELayerConfig,
    )

    common_config = CommonLayerConfig(
        hidden_size=2688, mixed_precision_dtype="bf16",
        normalization="RMSNorm", add_bias_linear=False,
    )
    Embedding = EmbeddingLayerConfig(
        common_config=common_config,
        vocab_size=131072, max_sequence_length=8192,
        position_embedding_type="none",
    )
    Mamba = MambaLayerConfig(common_config=common_config, head_dim=64, state_size=128)
    Att   = AttentionLayerConfig(common_config=common_config, num_attention_heads=32,
                                 num_query_groups=2, kv_channels=128)
    MoE   = MoELayerConfig(common_config=common_config, num_experts=128, top_k=6, ...)
    Loss  = CrossEntropyLayerConfig()

    layer_pattern = [Embedding, Mamba, [MoE, Mamba] * 3, Att, ..., Loss]

    def make_recipe():
        return HybridModelConfig(
            common_config=common_config,
            layer_pattern=layer_pattern,
            untie_embeddings_and_output_weights=True,
        )

The legacy string DSL (``hybrid_layer_pattern="M*M*"``) remains supported
unchanged for callers that prefer it.

``HybridModel`` and ``HybridStack`` are intentionally **not** re-exported
from this package's ``__init__`` to avoid a circular import: this package is
imported transitively from :mod:`megatron.core` initialisation (via
``inference.contexts.dynamic_context``), and the HybridModel import chain
re-enters partially-initialised :mod:`megatron.core`. Import them from their
submodules directly::

    from megatron.core.models.hybrid.hybrid_model import HybridModel
    from megatron.core.models.hybrid.hybrid_block import HybridStack
"""

from megatron.core.models.hybrid.common_layer_config import CommonLayerConfig
from megatron.core.models.hybrid.hybrid_model_config import HybridModelConfig
from megatron.core.models.hybrid.layer_configs import (
    AttentionLayerConfig,
    CrossEntropyLayerConfig,
    DSALayerConfig,
    EmbeddingLayerConfig,
    GDNLayerConfig,
    LayerConfig,
    MambaLayerConfig,
    MLPLayerConfig,
    MoELayerConfig,
    MTPLayerConfig,
    PipelineSplit,
)
from megatron.core.models.hybrid.layer_pattern import (
    RECIPE_ENTRY_POINT,
    flatten_decoder_pattern,
    load_recipe,
)

__all__ = [
    "AttentionLayerConfig",
    "CommonLayerConfig",
    "CrossEntropyLayerConfig",
    "DSALayerConfig",
    "EmbeddingLayerConfig",
    "GDNLayerConfig",
    "HybridModelConfig",
    "LayerConfig",
    "MambaLayerConfig",
    "MLPLayerConfig",
    "MoELayerConfig",
    "MTPLayerConfig",
    "PipelineSplit",
    "RECIPE_ENTRY_POINT",
    "flatten_decoder_pattern",
    "load_recipe",
]
