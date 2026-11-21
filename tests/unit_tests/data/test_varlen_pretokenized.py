# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the pre-tokenized varlen SFT dataset and preprocessing tool."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from megatron.core.datasets.indexed_dataset import IndexedDataset, IndexedDatasetBuilder
from megatron.training.datasets.sft_dataset import IGNORE_INDEX
from megatron.training.datasets.varlen_dataset import (
    VarlenDataset,
    VarlenPretokenizedLowLevelDataset,
)
from tools.preprocess_varlen_sft import assistant_only_mask, write_pretokenized


def _write_stream(prefix, sequences, dtype) -> None:
    builder = IndexedDatasetBuilder(str(prefix) + ".bin", dtype=dtype)
    for sequence in sequences:
        builder.add_item(torch.tensor(sequence, dtype=torch.int64))
        builder.end_document()
    builder.finalize(str(prefix) + ".idx")


def _config(tokenizer, seq_length=64, cp=1, sbhd=False) -> SimpleNamespace:
    return SimpleNamespace(
        tokenizer=tokenizer,
        sequence_length=seq_length,
        reset_position_ids=False,
        create_attention_mask=False,
        reset_attention_mask=False,
        varlen_sbhd_validation=sbhd,
        data_parallel_size=1,
        context_parallel_size=cp,
        dynamic_context_parallel=False,
        sequence_parallel_size=1,
    )


def _make_dataset(low_level, config) -> VarlenDataset:
    dataset = VarlenDataset.__new__(VarlenDataset)
    dataset.config = config
    dataset.dataset = low_level
    dataset.indices = np.arange(len(low_level))
    return dataset


class _FakeTokenizer:
    """Only the ids needed for padding are used on the pretokenized path."""

    eod = 0
    pad = 7


class _FakeSFTTokenizer:
    def __init__(self):
        self.eod = 0

    def tokenize_conversation(self, messages, return_target, add_generation_prompt):
        tokens = torch.tensor([10, 11, 12, 13], dtype=torch.int64)
        targets = torch.tensor([IGNORE_INDEX, IGNORE_INDEX, 12, 13], dtype=torch.int64)
        return tokens, targets


def test_pretokenized_low_level_round_trip(tmp_path):
    tokens = [[11, 12, 13, 14], [21, 22, 23]]
    masks = [[0, 0, 1, 1], [0, 0, 1]]
    _write_stream(tmp_path / "tokens", tokens, np.int32)
    _write_stream(tmp_path / "tokens.mask", masks, np.uint8)

    dataset = VarlenPretokenizedLowLevelDataset(str(tmp_path / "tokens"))

    assert len(dataset) == 2
    for idx, (expected_tokens, expected_mask) in enumerate(zip(tokens, masks)):
        got_tokens, got_mask = dataset[idx]
        assert got_tokens.tolist() == expected_tokens
        assert got_mask.tolist() == expected_mask
        assert got_mask.dtype == np.uint8


def test_build_low_level_dataset_selects_pretokenized_reader(tmp_path):
    _write_stream(tmp_path / "tokens", [[1, 2, 3]], np.int32)
    _write_stream(tmp_path / "tokens.mask", [[0, 1, 1]], np.uint8)
    config = _config(_FakeTokenizer())
    config.varlen_pretokenized_dataset = True

    dataset = VarlenDataset.build_low_level_dataset(str(tmp_path / "tokens"), config)

    assert isinstance(dataset, VarlenPretokenizedLowLevelDataset)


def test_pretokenized_low_level_rejects_missing_mask(tmp_path):
    _write_stream(tmp_path / "tokens", [[1, 2, 3]], np.int32)

    with pytest.raises(ValueError, match="Missing assistant loss mask sidecar"):
        VarlenPretokenizedLowLevelDataset(str(tmp_path / "tokens"))


def test_pretokenized_low_level_rejects_sample_count_mismatch(tmp_path):
    _write_stream(tmp_path / "tokens", [[1, 2, 3], [4, 5, 6]], np.int32)
    _write_stream(tmp_path / "tokens.mask", [[0, 1, 1]], np.uint8)

    with pytest.raises(ValueError, match="different number of samples"):
        VarlenPretokenizedLowLevelDataset(str(tmp_path / "tokens"))


def test_pretokenized_low_level_rejects_sequence_length_mismatch(tmp_path):
    _write_stream(tmp_path / "tokens", [[1, 2, 3, 4]], np.int32)
    _write_stream(tmp_path / "tokens.mask", [[0, 1, 1]], np.uint8)

    with pytest.raises(ValueError, match="sequence lengths do not match"):
        VarlenPretokenizedLowLevelDataset(str(tmp_path / "tokens"))


def test_pretokenized_low_level_rejects_non_binary_mask(tmp_path):
    _write_stream(tmp_path / "tokens", [[1, 2, 3]], np.int32)
    _write_stream(tmp_path / "tokens.mask", [[0, 2, 1]], np.uint8)

    dataset = VarlenPretokenizedLowLevelDataset(str(tmp_path / "tokens"))
    with pytest.raises(ValueError, match="only 0/1 values"):
        dataset[0]


def test_pretokenized_getitem_shifts_mask_and_reports_seq_lens(tmp_path):
    tokens = [10, 11, 12, 13, 14, 15]
    # Assistant spans are tokens 12-13 and 15.
    mask = [0, 0, 1, 1, 0, 1]
    _write_stream(tmp_path / "tokens", [tokens], np.int32)
    _write_stream(tmp_path / "tokens.mask", [mask], np.uint8)
    low_level = VarlenPretokenizedLowLevelDataset(str(tmp_path / "tokens"))

    # cp=2 -> pad divisor 4; 5 valid targets pad to 8.
    dataset = _make_dataset(low_level, _config(_FakeTokenizer(), cp=2))
    out = dataset[0]

    assert out["tokens"].tolist() == [10, 11, 12, 13, 14, 15, 7, 7]
    assert out["labels"].tolist() == [11, 12, 13, 14, 15, 7, 7, 7]
    assert out["loss_mask"].tolist() == [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    assert int(out["original_seq_len"].item()) == 5
    assert int(out["padded_seq_len"].item()) == 8
    assert out["tokens"].numel() == int(out["padded_seq_len"].item())

    # Every retained assistant target is trainable; prompt targets are not.
    labels = out["labels"]
    assert torch.equal(out["loss_mask"][labels == 12], torch.tensor([1.0]))
    assert torch.equal(out["loss_mask"][labels == 13], torch.tensor([1.0]))
    assert torch.equal(out["loss_mask"][labels == 15], torch.tensor([1.0]))
    assert torch.equal(out["loss_mask"][labels == 11], torch.tensor([0.0]))
    assert torch.equal(out["loss_mask"][labels == 14], torch.tensor([0.0]))


def test_pretokenized_getitem_keeps_tokens_verbatim(tmp_path):
    _write_stream(tmp_path / "tokens", [[1, 2, 3, 4]], np.int32)
    _write_stream(tmp_path / "tokens.mask", [[0, 0, 1, 1]], np.uint8)
    low_level = VarlenPretokenizedLowLevelDataset(str(tmp_path / "tokens"))

    out = _make_dataset(low_level, _config(_FakeTokenizer()))[0]

    # No EOD is appended or substituted on the pretokenized path.
    assert out["tokens"].tolist() == [1, 2, 3]
    assert out["labels"].tolist() == [2, 3, 4]
    assert 0 not in out["tokens"].tolist()


def test_pretokenized_getitem_uses_mask_in_sbhd_mode(tmp_path):
    _write_stream(tmp_path / "tokens", [[10, 11, 12, 13]], np.int32)
    _write_stream(tmp_path / "tokens.mask", [[0, 0, 1, 1]], np.uint8)
    low_level = VarlenPretokenizedLowLevelDataset(str(tmp_path / "tokens"))

    out = _make_dataset(low_level, _config(_FakeTokenizer(), seq_length=6, sbhd=True))[0]

    assert out["loss_mask"].tolist() == [0.0, 1.0, 1.0, 0.0, 0.0, 0.0]


def test_pretokenized_getitem_rejects_single_token(tmp_path):
    _write_stream(tmp_path / "tokens", [[5]], np.int32)
    _write_stream(tmp_path / "tokens.mask", [[1]], np.uint8)
    low_level = VarlenPretokenizedLowLevelDataset(str(tmp_path / "tokens"))

    with pytest.raises(ValueError, match="at least two tokens"):
        _make_dataset(low_level, _config(_FakeTokenizer()))[0]


def test_preprocessor_writes_tokens_and_assistant_mask(tmp_path):
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(json.dumps({"messages": [{"role": "user", "content": "hi"}]}) + "\n")
    output_prefix = tmp_path / "out"

    count = write_pretokenized(
        str(input_path), str(output_prefix), _FakeSFTTokenizer(), "nemotron-h-aligned"
    )

    assert count == 1
    tokens = IndexedDataset(str(output_prefix))
    masks = IndexedDataset(str(output_prefix) + ".mask")
    assert tokens[0].tolist() == [10, 11, 12, 13, 0]
    assert masks[0].tolist() == [0, 0, 1, 1, 1]


def test_assistant_only_mask_rejects_default_format():
    with pytest.raises(ValueError, match="all-1 targets"):
        assistant_only_mask([1, 2, 3], "default")


def test_assistant_only_mask_uses_ignore_index():
    mask = assistant_only_mask([IGNORE_INDEX, IGNORE_INDEX, 2, 3], "identity")
    assert mask.tolist() == [0, 0, 1, 1]


def test_assistant_only_mask_rejects_no_targets():
    with pytest.raises(ValueError, match="no assistant target"):
        assistant_only_mask([IGNORE_INDEX, IGNORE_INDEX], "identity")
