# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Any, Literal

from megatron.core.transformer import TransformerConfig


@dataclass(kw_only=True)
class T5ModelConfig:
    """T5 encoder-decoder inputs for the application's model provider."""

    transformer: TransformerConfig
    """Decoder settings shared with the encoder, except its layer count."""

    seq_length: int
    """Maximum position capacity, not the data sequence length."""

    encoder_num_layers: int
    """Number of encoder layers; the decoder uses transformer.num_layers."""

    encoder_seq_length: int | None = None
    """Encoder input width, independent of the position-embedding capacity."""

    decoder_seq_length: int | None = None
    """T5's independent decoder input width; not a decoder-only model's execution length."""

    vocab_size: int | None = None
    """Model vocabulary, or None until tokenizer initialization."""

    should_pad_vocab: bool = False
    """Whether to pad the supplied raw vocabulary size."""

    make_vocab_size_divisible_by: int = 128
    """Vocabulary alignment factor, multiplied by tensor-parallel size."""

    fp16_lm_cross_entropy: bool = False
    """Compute language-model cross entropy in half precision."""

    share_embeddings_and_output_weights: bool = False
    """Tie input embedding and output projection weights."""

    position_embedding_type: Literal["learned_absolute", "rope", "relative"] = "learned_absolute"
    """Position embedding implementation."""

    rotary_percent: float = 1.0
    """Fraction of the head dimension using rotary embeddings."""

    seq_len_interpolation_factor: float | None = None
    """Scale for linearly interpolating rotary position indices."""

    relative_attention_num_buckets: int = 32
    """Number of relative-position attention buckets."""

    relative_attention_max_distance: int = 128
    """Maximum distance distinguished by relative-position attention."""

    tp_comm_overlap_cfg: str | dict[str, Any] | None = None
    """Transformer Engine user-buffer overrides, as a YAML path or dictionary."""

    def validate(self) -> None:
        """Check the independent encoder and decoder execution widths."""
        for name in ("encoder_seq_length", "decoder_seq_length"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")
            bound = self.transformer.max_seq_length_per_sample
            if value is not None and bound is not None and value > bound:
                raise ValueError(f"max_seq_length_per_sample must cover {name}")

    def finalize(self) -> None:
        """Resolve the encoder width before shared runtime services read the model."""
        if self.encoder_seq_length is None:
            self.encoder_seq_length = self.transformer.max_seq_length_per_sample
        if (
            self.transformer.max_seq_length_per_sample is None
            and self.encoder_seq_length is not None
        ):
            self.transformer.max_seq_length_per_sample = max(
                self.encoder_seq_length, self.decoder_seq_length or self.encoder_seq_length
            )
