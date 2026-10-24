# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Any

from megatron.core.transformer import TransformerConfig


@dataclass(kw_only=True)
class BertModelConfig:
    """BERT model inputs; construction remains with the application's provider."""

    transformer: TransformerConfig
    """BERT encoder architecture, precision, and parallelism."""
    seq_length: int
    """Maximum position capacity, not the data sequence length."""
    vocab_size: int | None = None
    """Model vocabulary, or None until tokenizer initialization."""
    should_pad_vocab: bool = False
    """Whether to pad the supplied raw vocabulary size."""
    make_vocab_size_divisible_by: int = 128
    """Vocabulary alignment factor, multiplied by tensor-parallel size."""
    share_embeddings_and_output_weights: bool = True
    """Tie embedding and language-model output weights."""
    bert_binary_head: bool = True
    """Include the sentence-order head and two token types."""
    tp_comm_overlap_cfg: str | dict[str, Any] | None = None
    """Transformer Engine user-buffer overrides, as a YAML path or dictionary."""
