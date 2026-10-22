# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Any

from megatron.core.transformer import TransformerConfig
from megatron.training.models.base import ModelConfig


@dataclass(kw_only=True)
class T5ModelConfig(ModelConfig):
    """T5 encoder-decoder inputs for the application's model provider."""

    transformer: TransformerConfig
    """Decoder settings shared with the encoder, except its layer count."""
    seq_length: int
    """Maximum position capacity, not the data sequence length."""
    encoder_num_layers: int
    """Number of encoder layers; the decoder uses transformer.num_layers."""
    vocab_size: int | None = None
    """Model vocabulary, or None until tokenizer initialization."""
    should_pad_vocab: bool = False
    """Whether to pad the supplied raw vocabulary size."""
    make_vocab_size_divisible_by: int = 128
    """Vocabulary alignment factor, multiplied by tensor-parallel size."""
    fp16_lm_cross_entropy: bool = False
    """Compute language-model cross entropy in half precision."""
    share_embeddings_and_output_weights: bool = True
    """Tie input embedding and output projection weights."""
    position_embedding_type: str = "learned_absolute"
    """Position embedding implementation."""
    rotary_percent: float = 1.0
    """Fraction of the head dimension using rotary embeddings."""
    relative_attention_num_buckets: int = 32
    """Number of relative-position attention buckets."""
    relative_attention_max_distance: int = 128
    """Maximum distance distinguished by relative-position attention."""
    tp_comm_overlap_cfg: str | dict[str, Any] | None = None
    """Transformer Engine user-buffer overrides, as a YAML path or dictionary."""
