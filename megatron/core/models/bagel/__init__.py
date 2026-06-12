# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Bagel model components.

Includes MoT (Mixture of Transformers) transformer layer/block primitives, the
Bagel-specific RoPE, FlexAttention, packed-sequence parameters, and the LLM /
MIMO model wrappers used by the Bagel example.
"""
from .attention_mot import (
    SelfAttentionMoT,
    SelfAttentionMoTSubmodules,
)
from .bagel_mimo import BagelMimoModel, gather_pad_to_length
from .bagel_rope import BagelRotaryEmbedding
from .flex_attention import FlexAttention
from .hf_bagel_llm import BagelLLMHuggingFaceModel
from .mcore_bagel_llm import BagelMCoreModel
from .mot_packed_seq_params import MoTPackedSeqParams
from .transformer_mot_block import (
    TransformerMoTBlock,
    TransformerMoTBlockSubmodules,
    get_mot_layer_spec,
)
from .transformer_mot_layer import (
    MoTTransformerLayer,
    MoTTransformerLayerSubmodules,
)

__all__ = [
    'BagelLLMHuggingFaceModel',
    'BagelMCoreModel',
    'BagelMimoModel',
    'BagelRotaryEmbedding',
    'FlexAttention',
    'MoTPackedSeqParams',
    'MoTTransformerLayer',
    'MoTTransformerLayerSubmodules',
    'SelfAttentionMoT',
    'SelfAttentionMoTSubmodules',
    'TransformerMoTBlock',
    'TransformerMoTBlockSubmodules',
    'gather_pad_to_length',
    'get_mot_layer_spec',
]
