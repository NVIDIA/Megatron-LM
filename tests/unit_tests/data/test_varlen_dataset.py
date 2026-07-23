# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the variable-length raw-text and mock pretraining datasets."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from megatron.training.datasets.varlen_dataset import (
    MockVarlenDataset,
    VarlenDataset,
    VarlenDatasetConfig,
    VarlenLowLevelDataset,
    _looks_like_hf_id,
    _raw_text_loader,
)


@pytest.mark.parametrize(
    "path,expected",
    [
        ("owner/dataset", True),
        ("/tmp/data.jsonl", False),
        ("./data.jsonl", False),
        ("../data.jsonl", False),
        ("data.jsonl", False),
        ("", False),
        (None, False),
    ],
)
def test_looks_like_hf_id(path, expected):
    assert _looks_like_hf_id(path) is expected


def _varlen_args(**overrides):
    values = {
        "use_varlen_dataset": False,
        "varlen_sbhd_validation": False,
        "varlen_mock_dataset_config_json": None,
        "sft": False,
        "fim_data": False,
        "mock_data": False,
        "hybrid_context_parallel": False,
        "num_experts": None,
        "reset_position_ids": False,
        "reset_attention_mask": False,
        "dataloader_inter_document_masking": False,
        "create_attention_mask_in_dataloader": True,
        "sequence_packing_scheduler": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_varlen_validation_selects_default_scheduler():
    from megatron.training.arguments import _validate_varlen_dataset_args

    args = _varlen_args(use_varlen_dataset=True)
    _validate_varlen_dataset_args(args)

    assert args.sequence_packing_scheduler == "dp_balanced"
    assert args.create_attention_mask_in_dataloader is False


def test_varlen_validation_preserves_explicit_scheduler():
    from megatron.training.arguments import _validate_varlen_dataset_args

    args = _varlen_args(use_varlen_dataset=True, sequence_packing_scheduler="dp_balanced")
    _validate_varlen_dataset_args(args)

    assert args.sequence_packing_scheduler == "dp_balanced"


@pytest.mark.parametrize(
    "overrides",
    [
        {"varlen_sbhd_validation": True},
        {"varlen_mock_dataset_config_json": "{}", "mock_data": True},
        {"use_varlen_dataset": True, "varlen_mock_dataset_config_json": "{}"},
        {"use_varlen_dataset": True, "sft": True},
        {"use_varlen_dataset": True, "fim_data": True},
        {"use_varlen_dataset": True, "reset_position_ids": True},
        {"use_varlen_dataset": True, "reset_attention_mask": True},
        {"use_varlen_dataset": True, "dataloader_inter_document_masking": True},
        {
            "use_varlen_dataset": True,
            "varlen_sbhd_validation": True,
            "sequence_packing_scheduler": "dp_balanced",
        },
        {"use_varlen_dataset": True, "varlen_sbhd_validation": True, "mock_data": True},
        {"use_varlen_dataset": True, "hybrid_context_parallel": True},
        {"use_varlen_dataset": True, "varlen_sbhd_validation": True, "num_experts": 8},
    ],
)
def test_varlen_validation_rejects_incompatible_options(overrides):
    from megatron.training.arguments import _validate_varlen_dataset_args

    with pytest.raises(AssertionError):
        _validate_varlen_dataset_args(_varlen_args(**overrides))


def test_raw_text_loader():
    assert _raw_text_loader({"text": "hello"}) == "hello"
    assert _raw_text_loader({"text": None}) == ""
    with pytest.raises(ValueError, match="must be a string"):
        _raw_text_loader({"text": [1, 2]})


def _write_jsonl(tmp_path: Path, rows):
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    return str(path)


def test_low_level_loads_json_array_raw_text(tmp_path):
    pytest.importorskip("datasets")
    pytest.importorskip("pandas")
    path = tmp_path / "data.json"
    path.write_text(json.dumps([{"text": "one"}, {"text": "two", "id": 2}]))

    dataset = VarlenLowLevelDataset(str(path))

    assert dataset.schema_name == "pretrain-text"
    assert [dataset[0], dataset[1]] == ["one", "two"]


def test_low_level_loads_jsonl_raw_text(tmp_path):
    pytest.importorskip("datasets")
    pytest.importorskip("pandas")
    path = _write_jsonl(
        tmp_path,
        [
            {"text": "document one", "url": "https://example/1"},
            {"text": "document two", "url": "https://example/2"},
        ],
    )

    dataset = VarlenLowLevelDataset(path)

    assert len(dataset) == 2
    assert dataset[1] == "document two"


def test_low_level_rejects_non_text_schema(tmp_path):
    pytest.importorskip("datasets")
    pytest.importorskip("pandas")
    path = _write_jsonl(tmp_path, [{"messages": [{"role": "user", "content": "hello"}]}])

    with pytest.raises(ValueError, match="requires a raw pretraining 'text' column"):
        VarlenLowLevelDataset(path)


class _FakeTokenizer:
    def __init__(self, eod: int = 0, pad=None):
        self._eod = eod
        self._pad = pad

    @property
    def eod(self):
        return self._eod

    @property
    def pad(self):
        return self._pad

    @property
    def vocab_size(self):
        return 128

    def tokenize(self, text):
        return [ord(char) % 100 + 1 for char in text]


def test_varlen_dataset_config_owns_mock_and_sbhd_fields():
    config_args = {
        "random_seed": 123,
        "sequence_length": 8,
        "tokenizer": _FakeTokenizer(),
        "reset_position_ids": False,
        "reset_attention_mask": False,
        "eod_mask_loss": False,
        "create_attention_mask": False,
        "context_parallel_size": 1,
    }
    mock_config = {"mode": "file", "path": "lengths.csv"}

    config = VarlenDatasetConfig(
        **config_args, mock_dataset_config=mock_config, sbhd_validation=True
    )

    assert config.mock_dataset_config == mock_config
    assert config.sbhd_validation is True
    with pytest.raises(AssertionError, match="hybrid context parallelism"):
        VarlenDatasetConfig(**config_args, hybrid_context_parallel=True)


def _make_config(tokenizer, seq_length=64, *, cp=1, dp=1, sp=1, sbhd=False, eod_mask=False):
    return SimpleNamespace(
        tokenizer=tokenizer,
        sequence_length=seq_length,
        reset_position_ids=False,
        create_attention_mask=False,
        reset_attention_mask=False,
        eod_mask_loss=eod_mask,
        sbhd_validation=sbhd,
        data_parallel_size=dp,
        context_parallel_size=cp,
        sequence_parallel_size=sp,
        hybrid_context_parallel=False,
    )


def _make_varlen(items, config):
    dataset = VarlenDataset.__new__(VarlenDataset)
    dataset.config = config
    dataset.dataset = items
    dataset.indices = np.arange(len(items))
    return dataset


def _make_mock_varlen(token_arrays, config):
    dataset = MockVarlenDataset.__new__(MockVarlenDataset)
    dataset.config = config
    dataset.dataset = token_arrays
    dataset.indices = np.arange(len(token_arrays))
    return dataset


def test_getitem_thd_raw_text_keys_and_shapes():
    dataset = _make_varlen(
        ["hello world"], _make_config(_FakeTokenizer(eod=0, pad=7))
    )

    sample = dataset[0]

    assert set(sample) == {
        "tokens",
        "labels",
        "loss_mask",
        "position_ids",
        "original_seq_len",
        "padded_seq_len",
    }
    size = sample["tokens"].numel()
    assert sample["labels"].numel() == size
    assert sample["loss_mask"].numel() == size
    assert sample["position_ids"].numel() == size
    assert sample["padded_seq_len"].item() == size


def test_getitem_thd_pad_mask_keeps_real_eod():
    tokenizer = _FakeTokenizer(eod=0, pad=None)
    dataset = _make_varlen(["abc"], _make_config(tokenizer, cp=2))

    sample = dataset[0]

    assert sample["labels"].tolist()[2] == tokenizer.eod
    assert sample["loss_mask"].tolist() == [1.0, 1.0, 1.0, 0.0]


def test_getitem_thd_pads_to_cp_divisor():
    dataset = _make_varlen(
        ["abcde"], _make_config(_FakeTokenizer(eod=0, pad=7), cp=2)
    )

    sample = dataset[0]

    assert sample["padded_seq_len"].item() % 4 == 0


def test_getitem_empty_text_is_nonempty():
    dataset = _make_varlen([""], _make_config(_FakeTokenizer(eod=0, pad=7)))

    sample = dataset[0]

    assert sample["tokens"].numel() >= 1
    assert sample["labels"].numel() == sample["tokens"].numel()


def test_getitem_exact_capacity_reserves_eod():
    tokenizer = _FakeTokenizer(eod=0, pad=7)
    dataset = _make_varlen(["abcde"], _make_config(tokenizer, seq_length=4))

    sample = dataset[0]

    assert sample["tokens"].numel() == 4
    assert sample["labels"][-1].item() == tokenizer.eod
    assert sample["loss_mask"][-1].item() == 1.0


@pytest.mark.parametrize("sbhd", [False, True])
def test_getitem_honors_eod_mask_loss(sbhd):
    tokenizer = _FakeTokenizer(eod=0, pad=7)
    tokenizer.tokenize = lambda _: [1, tokenizer.eod, 2]
    dataset = _make_varlen(
        ["abc"], _make_config(tokenizer, seq_length=8, sbhd=sbhd, eod_mask=True)
    )

    sample = dataset[0]

    assert sample["tokens"][1].item() == tokenizer.eod
    assert sample["loss_mask"][1].item() == 0.0
    assert sample["labels"][0].item() == tokenizer.eod
    assert sample["loss_mask"][0].item() == 1.0


def test_getitem_sbhd_pads_to_sequence_length():
    dataset = _make_varlen(
        ["abc"], _make_config(_FakeTokenizer(eod=0, pad=None), seq_length=8, sbhd=True)
    )

    sample = dataset[0]

    assert set(sample) == {"tokens", "labels", "loss_mask", "position_ids"}
    assert sample["tokens"].numel() == 8
    assert sample["loss_mask"].tolist() == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_mock_getitem_thd_keys_and_pad_fallback():
    dataset = _make_mock_varlen(
        [np.array([1, 2, 3, 4], dtype=np.int64)],
        _make_config(_FakeTokenizer(eod=0, pad=None), cp=2),
    )

    sample = dataset[0]

    assert set(sample) == {
        "tokens",
        "labels",
        "loss_mask",
        "position_ids",
        "original_seq_len",
        "padded_seq_len",
    }
    assert sample["padded_seq_len"].item() % 4 == 0


def test_mock_getitem_truncates_to_sequence_length():
    dataset = _make_mock_varlen(
        [np.arange(1, 10, dtype=np.int64)],
        _make_config(_FakeTokenizer(eod=0, pad=7), seq_length=4),
    )

    sample = dataset[0]

    assert sample["original_seq_len"].item() == 4
    assert sample["tokens"].numel() == 4


def test_mock_getitem_length_one_is_nonempty():
    dataset = _make_mock_varlen(
        [np.array([], dtype=np.int64)], _make_config(_FakeTokenizer(eod=0, pad=7))
    )

    sample = dataset[0]

    assert sample["original_seq_len"].item() == 1
    assert sample["tokens"].numel() == 1


def test_unpack_batch_normalizes_varlen_samples():
    from megatron.core.datasets.data_schedule_utils import _unpack_batch

    batch = [
        {
            "tokens": torch.arange(4, dtype=torch.int64).view(1, 4),
            "labels": torch.arange(4, dtype=torch.int64).view(1, 4),
            "loss_mask": torch.ones(1, 4),
            "position_ids": torch.arange(4, dtype=torch.int64).view(1, 4),
            "padded_seq_len": torch.tensor([4], dtype=torch.int32),
        }
    ]

    output = _unpack_batch(batch)

    assert output[0]["tokens"].shape == (4,)
    assert output[0]["original_seq_len"].item() == 4


def test_unpack_batch_requires_varlen_lengths():
    from megatron.core.datasets.data_schedule_utils import _unpack_batch

    with pytest.raises(KeyError, match="padded_seq_len"):
        _unpack_batch([{"cu_seqlens": torch.tensor([0, 4], dtype=torch.int32)}])


def _build_varlen_for_loader(items, config, num_samples):
    from megatron.core.datasets.utils import Split

    dataset = VarlenDataset.__new__(VarlenDataset)
    dataset.config = config
    dataset.dataset = items
    dataset.indices = np.arange(len(items))
    dataset.num_samples = num_samples
    dataset.index_split = Split.train
    return dataset


def _loader_args(*, use_varlen, sbhd, scheduler, mbs, gbs=None):
    return SimpleNamespace(
        dataloader_type="single",
        micro_batch_size=mbs,
        global_batch_size=mbs if gbs is None else gbs,
        full_validation=False,
        num_workers=0,
        use_varlen_dataset=use_varlen,
        varlen_sbhd_validation=sbhd,
        sequence_packing_scheduler=scheduler,
    )


def test_sbhd_validation_dataloader_uses_default_collate():
    from megatron.core import parallel_state
    from megatron.training.datasets.data_samplers import build_pretraining_data_loader
    from megatron.training.global_vars import destroy_global_vars, set_args
    from tests.unit_tests.test_utilities import Utils

    Utils.initialize_model_parallel(1, 1)
    try:
        tokenizer = _FakeTokenizer(eod=0, pad=7)
        seq_len, micro_batch_size = 16, 2
        data_parallel_size = parallel_state.get_data_parallel_world_size()
        num_samples = micro_batch_size * data_parallel_size * 4
        dataset = _build_varlen_for_loader(
            ["hello world"] * num_samples,
            _make_config(tokenizer, seq_length=seq_len, sbhd=True),
            num_samples,
        )
        set_args(
            _loader_args(
                use_varlen=True,
                sbhd=True,
                scheduler=None,
                mbs=micro_batch_size,
            )
        )

        batch = next(iter(build_pretraining_data_loader(dataset, consumed_samples=0)))

        assert isinstance(batch, dict)
        assert batch["tokens"].shape == (micro_batch_size, seq_len)
    finally:
        destroy_global_vars()
        Utils.destroy_model_parallel()


def test_thd_dataloader_uses_identity_collate():
    from megatron.core import parallel_state
    from megatron.training.datasets.data_samplers import build_pretraining_data_loader
    from megatron.training.global_vars import destroy_global_vars, set_args
    from tests.unit_tests.test_utilities import Utils

    Utils.initialize_model_parallel(1, 1)
    try:
        tokenizer = _FakeTokenizer(eod=0, pad=7)
        micro_batch_size = 2
        data_parallel_size = parallel_state.get_data_parallel_world_size()
        num_samples = micro_batch_size * data_parallel_size * 4
        values = ["a", "abcdef", "xy", "qwerty"]
        dataset = _build_varlen_for_loader(
            [values[index % len(values)] for index in range(num_samples)],
            _make_config(tokenizer),
            num_samples,
        )
        set_args(
            _loader_args(
                use_varlen=True,
                sbhd=False,
                scheduler="dp_balanced",
                mbs=micro_batch_size,
            )
        )

        batch = next(iter(build_pretraining_data_loader(dataset, consumed_samples=0)))

        assert isinstance(batch, list)
        assert len(batch) == micro_batch_size
        assert "padded_seq_len" in batch[0]
    finally:
        destroy_global_vars()
        Utils.destroy_model_parallel()


def test_packing_scheduler_dataloader_yields_microbatches():
    from megatron.core import parallel_state
    from megatron.training.datasets.data_samplers import build_pretraining_data_loader
    from megatron.training.global_vars import destroy_global_vars, set_args
    from tests.unit_tests.test_utilities import Utils

    Utils.initialize_model_parallel(1, 1)
    try:
        tokenizer = _FakeTokenizer(eod=0, pad=7)
        micro_batch_size = 2
        num_microbatches = 3
        data_parallel_size = parallel_state.get_data_parallel_world_size()
        global_batch_size = micro_batch_size * data_parallel_size * num_microbatches
        num_samples = global_batch_size * 2
        values = ["a", "abcdef", "xy", "qwerty"]
        dataset = _build_varlen_for_loader(
            [values[index % len(values)] for index in range(num_samples)],
            _make_config(tokenizer, dp=data_parallel_size),
            num_samples,
        )
        set_args(
            _loader_args(
                use_varlen=True,
                sbhd=False,
                scheduler="dp_balanced",
                mbs=micro_batch_size,
                gbs=global_batch_size,
            )
        )

        batch = next(iter(build_pretraining_data_loader(dataset, consumed_samples=0)))

        assert isinstance(batch, list)
        assert len(batch) == micro_batch_size
        assert "padded_seq_len" in batch[0]
    finally:
        destroy_global_vars()
        Utils.destroy_model_parallel()
