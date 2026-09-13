# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU tests: pre-tokenised packed bins (``input_ids`` / ``loss_mask`` / ``seq_start_id``)
through the varlen dataset: bins expand into their sequences, and the label mask follows the
packed-SFT convention (``loss_mask[1:]`` after the next-token shift)."""

import types

import numpy as np
import pytest
import torch

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from megatron.training.datasets.sft_dataset import IGNORE_INDEX
from megatron.training.datasets.varlen_dataset import VarlenDataset, VarlenLowLevelDataset

EOD = 1


def _write_bins(path):
    # bin 0: two sequences [0, 7) and [7, 12); bin 1: one sequence [0, 6)
    rows = {
        "input_ids": [
            [0, 10, 11, 12, 13, EOD, EOD, 0, 20, 21, EOD, EOD],
            [0, 30, 31, 32, EOD, EOD],
        ],
        "loss_mask": [[0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0], [0, 1, 1, 1, 0, 0]],
        "seq_start_id": [[0, 7], [0]],
    }
    table = pa.table(
        {
            "input_ids": pa.array(rows["input_ids"], type=pa.list_(pa.int64())),
            "loss_mask": pa.array(rows["loss_mask"], type=pa.list_(pa.int8())),
            "seq_start_id": pa.array(rows["seq_start_id"], type=pa.list_(pa.int64())),
        }
    )
    pq.write_table(table, path, row_group_size=1)


def test_low_level_expands_bins_into_sequences(tmp_path):
    path = str(tmp_path / "bins.parquet")
    _write_bins(path)
    ds = VarlenLowLevelDataset(path)
    assert ds.schema_name == "packed-bins"
    assert len(ds) == 3
    first, second, third = ds[0], ds[1], ds[2]
    assert first["input_ids"].tolist() == [0, 10, 11, 12, 13, EOD, EOD]
    assert first["loss_mask"].tolist() == [0, 0, 1, 1, 1, 0, 0]
    assert second["input_ids"].tolist() == [0, 20, 21, EOD, EOD]
    assert third["input_ids"].tolist() == [0, 30, 31, 32, EOD, EOD]


def test_varlen_getitem_shifts_labels_and_mask(tmp_path):
    path = str(tmp_path / "bins.parquet")
    _write_bins(path)
    low = VarlenLowLevelDataset(path)
    ds = VarlenDataset.__new__(VarlenDataset)
    ds.dataset = low
    ds.indices = np.arange(len(low))
    ds.config = types.SimpleNamespace(
        tokenizer=types.SimpleNamespace(eod=EOD, pad=None),
        sequence_length=64,
        reset_position_ids=False,
        create_attention_mask=False,
        reset_attention_mask=False,
        varlen_sbhd_validation=False,
    )
    ds._calculate_padding_divisor = lambda: 4
    sample = ds[0]
    # tokens = ids[:-1]; labels = ids[1:] with IGNORE_INDEX where the source token is unmasked
    assert sample["tokens"].tolist()[:6] == [0, 10, 11, 12, 13, EOD]
    labels = sample["labels"].tolist()
    assert labels[:6] == [IGNORE_INDEX, 11, 12, 13, IGNORE_INDEX, IGNORE_INDEX]
    mask = sample["loss_mask"].tolist()
    assert mask[:6] == [0.0, 1.0, 1.0, 1.0, 0.0, 0.0]  # == loss_mask[1:] of the source
    assert int(sample["original_seq_len"]) == 6
    assert int(sample["padded_seq_len"]) % 4 == 0
    assert torch.equal(sample["position_ids"], torch.arange(int(sample["padded_seq_len"])))
