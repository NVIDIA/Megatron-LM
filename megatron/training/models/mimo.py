# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Any

import torch

from megatron.core.transformer import TransformerConfig
from megatron.training.models.base import ModelConfig


@dataclass(kw_only=True)
class MimoModelConfig(ModelConfig):
    """MIMO language-grid inputs for the shared training lifecycle.

    Tower construction and topology remain owned by the MIMO application.
    """

    transformer: TransformerConfig
    """Language-grid architecture, precision, and parallelism."""
    seq_length: int
    """Language-model position capacity, not the data sequence length."""
    img_h: int = 224
    """Vision input height shared by the encoder and its data pipeline."""
    img_w: int = 224
    """Vision input width shared by the encoder and its data pipeline."""
    patch_dim: int = 16
    """Vision patch size."""
    drop_vision_class_token: bool = False
    """Drop vision class tokens before projecting image embeddings."""
    training_world_size: int | None = None
    """Language-grid rank count, or None to use the full launch."""
    vocab_size: int | None = None
    """Language-model vocabulary, or None until tokenizer initialization."""
    should_pad_vocab: bool = False
    """Whether to pad the supplied raw vocabulary size."""
    make_vocab_size_divisible_by: int = 128
    """Vocabulary alignment factor, multiplied by tensor-parallel size."""
    share_embeddings_and_output_weights: bool = True
    """Tie language embedding and output projection weights."""
    logit_dtype: torch.dtype | None = None
    """Optional language-model output-logit dtype override."""
    hybrid_layer_pattern: str | None = None
    """Layer layout when the language backbone is a Hybrid model."""
    tp_comm_overlap_cfg: str | dict[str, Any] | None = None
    """Transformer Engine user-buffer overrides, as a YAML path or dictionary."""
