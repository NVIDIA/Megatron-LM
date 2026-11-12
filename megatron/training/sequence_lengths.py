# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Resolve configured input widths without confusing them with allocation bounds."""

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig


def get_model_input_lengths(
    model, dataset: GPTDatasetConfig | None = None
) -> tuple[int | None, int | None]:
    """Return the configured encoder/input and optional decoder tensor widths.

    Fixed GPT rows use the dataset width, which can be smaller than the model's
    allocation bound. SFT uses the physical pack width derived during finalization,
    not its per-conversation truncation limit. T5 has two independent input widths.
    Callback pipelines without a registered dataset must emit the configured model
    width or supply variable-shape metadata to the schedule. For variable-length
    batches these are nominal widths, not substitutes for actual batch metadata.
    """
    # Avoid importing model configs while the training package is bootstrapping.
    from megatron.training.models import LLaVAModelConfig, T5ModelConfig

    if isinstance(model, T5ModelConfig):
        return model.encoder_seq_length, model.decoder_seq_length
    if isinstance(model, LLaVAModelConfig):
        # Text-only dataset rows expand into image tokens before entering the LM.
        # The application resolves the padded execution width before startup.
        return model.transformer.max_seq_length_per_sample, None
    if dataset is not None:
        return dataset.packed_sequence_length or dataset.sequence_length, None
    return model.transformer.max_seq_length_per_sample, None
