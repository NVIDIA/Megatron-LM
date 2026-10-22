# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Any

from megatron.core.transformer import TransformerConfig
from megatron.training.models.base import ModelConfig


@dataclass(kw_only=True)
class LLaVAModelConfig(ModelConfig):
    """LLaVA language-model inputs used by the multimodal provider callbacks.

    Vision and projection specialization remains in those applications; this config
    owns the model settings consumed by the shared training lifecycle.
    """

    transformer: TransformerConfig
    """Base language-transformer architecture, precision, and parallelism."""
    seq_length: int
    """Maximum position capacity, not the image or text data sequence length."""
    vocab_size: int | None = None
    """Language-model vocabulary, or None until tokenizer initialization."""
    should_pad_vocab: bool = False
    """Whether to pad the supplied raw vocabulary size."""
    make_vocab_size_divisible_by: int = 128
    """Vocabulary alignment factor, multiplied by tensor-parallel size."""
    fp16_lm_cross_entropy: bool = False
    """Compute language-model cross entropy in half precision."""
    share_embeddings_and_output_weights: bool = True
    """Tie language embedding and output projection weights."""
    position_embedding_type: str = "learned_absolute"
    """Language position embedding implementation."""
    rotary_percent: float = 1.0
    """Fraction of the language-model head dimension using rotary embeddings."""
    rotary_base: int = 10000
    """Base period for language-model rotary embeddings."""
    rope_scaling: bool = False
    """Enable language-model rotary position scaling."""
    rope_scaling_factor: float = 8.0
    """Rotary position scaling factor when enabled."""
    hybrid_layer_pattern: str | None = None
    """Layer layout when the language backbone is a Hybrid model."""
    tp_comm_overlap_cfg: str | dict[str, Any] | None = None
    """Transformer Engine user-buffer overrides, as a YAML path or dictionary."""
