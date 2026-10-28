# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""is_packed_sequence_run() reads the data section of the registered run config: the same source
and the same formula as get_batch's has_cu_seqlens (the SFT family, or inter-document masking)."""

from types import SimpleNamespace

import pytest

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset
from megatron.training.config import DataConfig
from megatron.training.datasets.sft_dataset import MockSFTDataset
from megatron.training.datasets.varlen_dataset import MockVarlenDataset
from megatron.training.distillation import utils as distillation_utils

BOOLS = dict(reset_position_ids=False, reset_attention_mask=False, eod_mask_loss=False)


@pytest.mark.parametrize(
    "dataset_type, inter_document_masking, expected",
    [
        (MockGPTDataset, False, False),
        (MockGPTDataset, True, True),
        (MockSFTDataset, False, True),
        (MockVarlenDataset, False, False),  # packed for the rank rule, no cu_seqlens in get_batch
        (None, False, False),  # a loader-only section
    ],
)
def test_packedness_follows_the_data_section(
    monkeypatch, dataset_type, inter_document_masking, expected
):
    if dataset_type is None:
        data = DataConfig()
    else:
        dataset = GPTDatasetConfig(
            sequence_length=16,
            random_seed=1,
            inter_document_masking=inter_document_masking,
            create_attention_mask=not inter_document_masking,
            **BOOLS,
        )
        data = DataConfig(dataset=dataset, dataset_type=dataset_type)
    monkeypatch.setattr(distillation_utils, "get_run_config", lambda: SimpleNamespace(data=data))
    monkeypatch.setattr(
        distillation_utils, "get_args", lambda: pytest.fail("the predicate must not read args")
    )
    assert distillation_utils.is_packed_sequence_run() is expected
