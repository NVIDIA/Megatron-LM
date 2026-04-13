# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Layer spec helpers for Kimi K2.5 VL language decoder.

The language decoder uses MoE with Multi-Latent Attention (MLA),
which uses the standard ``get_gpt_decoder_block_spec`` with
``use_transformer_engine=True``.

No vision spec is needed here because the MoonViT3d vision encoder
is dynamically loaded from HuggingFace.
"""

from typing import Optional

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_config import MLATransformerConfig


def get_kimi_k25_language_spec(
    config: MLATransformerConfig,
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
) -> TransformerBlockSubmodules:
    """Transformer block spec for the Kimi K2.5 language decoder.

    Uses the standard GPT decoder block spec with Transformer Engine,
    which handles MLA layer construction when the config has
    ``multi_latent_attention=True``.

    Args:
        config: Language decoder MLATransformerConfig.
        vp_stage: Virtual pipeline stage.
        pp_rank: Pipeline parallel rank.

    Returns:
        TransformerBlockSubmodules with per-layer specs.
    """
    return get_gpt_decoder_block_spec(
        config=config,
        use_transformer_engine=True,
        vp_stage=vp_stage,
        pp_rank=pp_rank,
    )
