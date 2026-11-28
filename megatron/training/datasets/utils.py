# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Shared utilities for training-side dataset helpers."""

import json
import os
from typing import TYPE_CHECKING, Any, Optional

from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig

if TYPE_CHECKING:
    from megatron.training.config import PretrainConfigContainer


def get_dataset_input_length(dataset: BlendedMegatronDatasetConfig) -> int:
    """Return a dataset's output row width, using the derived physical pack width for SFT."""
    from megatron.training.datasets.sft_dataset import SFTDatasetConfig

    if isinstance(dataset, SFTDatasetConfig):
        return dataset.packed_sequence_length or dataset.sequence_length
    return dataset.sequence_length


def get_training_data_sequence_length(cfg: "PretrainConfigContainer") -> int | None:
    """Return the data width for token accounting and batch-size schedules.

    Prefer the dataset's row/pack width. Callback datasets have no dataset config:
    use T5's encoder width, or the model's training width for the other callbacks.
    This is not the expanded model input width used for pipeline buffers.
    """
    from megatron.training.models import T5ModelConfig

    if cfg.data.dataset is not None:
        return get_dataset_input_length(cfg.data.dataset)
    if isinstance(cfg.model, T5ModelConfig):
        return cfg.model.encoder_seq_length
    return cfg.model.transformer.max_seq_length_per_sample


def load_json_arg(spec: Optional[str]) -> Optional[Any]:
    """Parse a CLI JSON argument that may be either a JSON literal or a path
    to a JSON file.

    The argument is interpreted as a file path when ``spec`` points to an
    existing regular file on the local filesystem; otherwise it is parsed as
    a JSON literal string. Returns ``None`` when ``spec`` itself is ``None``,
    so callers can use it transparently for optional CLI flags.

    Used by the ``--sft-mock-dataset-config-json`` and
    ``--varlen-mock-dataset-config-json`` flags, which both accept either an
    inline JSON snippet or the path to a file containing the same JSON
    document.
    """
    if spec is None:
        return None
    if os.path.isfile(spec):
        with open(spec, "r") as f:
            return json.load(f)
    return json.loads(spec)
