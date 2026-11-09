# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The seam between the run config and the legacy arguments: the data sequence length, with its
fallback for a section that has no dataset."""

from types import SimpleNamespace

import pytest

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.training.config import DataConfig
from megatron.training.config.compat import data_sequence_length

BOOLS = dict(reset_position_ids=False, reset_attention_mask=False, eod_mask_loss=False)


def _dataset_section(sequence_length=16):
    return DataConfig(
        dataset=GPTDatasetConfig(sequence_length=sequence_length, random_seed=1, **BOOLS)
    )


def test_data_sequence_length_comes_from_the_dataset_and_never_reads_args_then():
    """With a dataset the namespace is not consulted: a different, or absent, args.seq_length
    cannot leak in. Without a dataset (a loader-only section) the namespace is the source."""
    cfg = SimpleNamespace(data=_dataset_section(16))
    assert data_sequence_length(cfg, SimpleNamespace(seq_length=32)) == 16
    assert data_sequence_length(cfg, SimpleNamespace()) == 16  # no seq_length attribute at all
    loader_only = SimpleNamespace(data=DataConfig())
    assert data_sequence_length(loader_only, SimpleNamespace(seq_length=32)) == 32
    with pytest.raises(AttributeError):
        data_sequence_length(loader_only, SimpleNamespace())
