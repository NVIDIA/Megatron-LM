# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Resolve configured input widths without confusing them with allocation bounds."""

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig


def get_dataset_input_length(dataset: GPTDatasetConfig) -> int:
    """Return a dataset's output row width, not an SFT conversation truncation limit."""
    from megatron.training.datasets.sft_dataset import SFTDatasetConfig

    if isinstance(dataset, SFTDatasetConfig):
        return dataset.packed_sequence_length or dataset.sequence_length
    return dataset.sequence_length


def get_model_input_length(model, dataset: GPTDatasetConfig | None = None) -> int | None:
    """Return the configured input width for shared training runtime consumers.

    Fixed GPT rows use the dataset width, which can be smaller than the model's
    allocation bound. SFT uses the physical pack width derived during finalization,
    not its per-conversation truncation limit. T5 returns its encoder input width;
    its separate decoder length remains a model/dataset concern, not a pipeline
    scheduling input (the T5 entrypoint does not support pipeline parallelism).
    Callback pipelines without a registered dataset must emit the configured model
    width or supply variable-shape metadata to the schedule. For variable-length
    batches these are nominal widths, not substitutes for actual batch metadata.
    """
    # Avoid importing model configs while the training package is bootstrapping.
    from megatron.training.models import LLaVAModelConfig, T5ModelConfig

    if isinstance(model, T5ModelConfig):
        return model.encoder_seq_length
    if isinstance(model, LLaVAModelConfig):
        # Text-only dataset rows expand into image tokens before entering the LM.
        # The application resolves the padded execution width before startup.
        return model.transformer.max_seq_length_per_sample
    if dataset is not None:
        return get_dataset_input_length(dataset)
    return model.transformer.max_seq_length_per_sample
