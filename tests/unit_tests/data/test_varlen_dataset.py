# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU-safe regression tests for training-local SFT and varlen datasets."""

from dataclasses import fields

import numpy as np
import pytest
import torch

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.utils import Split
from megatron.training.datasets import sft_dataset as sft_dataset_module
from megatron.training.datasets.sft_dataset import (
    IGNORE_INDEX,
    MockSFTDataset,
    MockSFTLowLevelDataset,
    SFTDataset,
    SFTDatasetConfig,
    _calculate_padding_divisor,
)
from megatron.training.datasets.varlen_dataset import (
    MockVarlenDataset,
    VarlenDataset,
    VarlenDatasetConfig,
    VarlenLowLevelDataset,
    _alpaca_to_messages,
    _looks_like_hf_id,
    _messages_passthrough,
    _raw_text_loader,
    _select_converter,
    _sharegpt_to_messages,
)


class _FakeTokenizer:
    """Minimal tokenizer with the interfaces exercised by the datasets."""

    vocab_size = 512
    unique_identifiers = {"class": "_FakeTokenizer"}

    def __init__(self, eod: int = 0, pad: int | None = 511) -> None:
        self.eod = eod
        self.eos = eod
        self.pad = pad
        self.special_tokens_dict = {"eos_token": eod, "pad_token": pad}

    @staticmethod
    def tokenize(text: str) -> list[int]:
        return [ord(character) % 200 + 1 for character in text]

    def tokenize_conversation(
        self, messages, return_target: bool = True, add_generation_prompt: bool = False
    ):
        assert return_target and not add_generation_prompt
        tokens = []
        targets = []
        for message in messages:
            token_ids = self.tokenize(message["content"])
            tokens.extend(token_ids)
            targets.extend(
                token_ids if message["role"] == "assistant" else [IGNORE_INDEX] * len(token_ids)
            )
        return torch.tensor(tokens, dtype=torch.int64), torch.tensor(targets, dtype=torch.int64)


def _config(config_type=SFTDatasetConfig, **overrides):
    values = {
        "random_seed": 123,
        "sequence_length": 16,
        "tokenizer": _FakeTokenizer(),
        "reset_position_ids": False,
        "reset_attention_mask": False,
        "eod_mask_loss": False,
        "create_attention_mask": False,
        "context_parallel_size": 1,
    }
    values.update(overrides)
    return config_type(**values)


def _mock_source(**overrides):
    values = {
        "mode": "distribution",
        "path": None,
        "distribution": "lognormal",
        "min_sequence_length": 4,
        "max_sequence_length": 12,
        "mean_sequence_length": 8,
        "lognormal_sigma": 0.5,
        "seed": 17,
        "size": 8,
    }
    values.update(overrides)
    return MockSFTLowLevelDataset(**values)


def _mid_level(dataset_type, low_level_dataset, config):
    return dataset_type(
        low_level_dataset,
        None,
        np.arange(len(low_level_dataset), dtype=np.int64),
        len(low_level_dataset),
        Split.train,
        config,
    )


def test_mock_config_is_flat_and_does_not_modify_core_config():
    field_names = {field.name for field in fields(SFTDatasetConfig)}
    assert "mock_data_mode" in field_names
    assert "mock_data_path" in field_names
    assert all("json" not in field_name for field_name in field_names)
    assert "mock_data_mode" not in {field.name for field in fields(GPTDatasetConfig)}


def test_mock_config_derives_typed_defaults_from_sequence_length():
    config = _config(sequence_length=32)
    assert config.mock_data_min_sequence_length == 16
    assert config.mock_data_max_sequence_length == 32
    assert config.mock_data_mean_sequence_length == 24


def test_mock_config_requires_path_for_file_mode():
    with pytest.raises(ValueError, match="mock_data_path is required"):
        _config(mock_data_mode="file")


def test_varlen_config_validates_packing_divisibility():
    with pytest.raises(ValueError, match="must be divisible"):
        _config(
            VarlenDatasetConfig,
            sequence_length=10,
            context_parallel_size=2,
            sequence_parallel_size=2,
        )


def test_varlen_sbhd_config_rejects_hybrid_cp():
    with pytest.raises(ValueError, match="incompatible"):
        _config(VarlenDatasetConfig, varlen_sbhd_validation=True, hybrid_context_parallel=True)


def test_padding_divisor_covers_standard_and_hybrid_parallelism():
    standard = _config(context_parallel_size=2, sequence_parallel_size=2)
    hybrid = _config(
        sequence_length=32,
        context_parallel_size=2,
        sequence_parallel_size=2,
        data_parallel_size=2,
        hybrid_context_parallel=True,
    )
    assert _calculate_padding_divisor(standard) == 8
    assert _calculate_padding_divisor(hybrid) == 16


def test_mock_file_mode_reads_first_headerless_value(tmp_path):
    path = tmp_path / "lengths.csv"
    path.write_text("3\n5\n8\n", encoding="utf-8")
    dataset = _mock_source(mode="file", path=str(path))
    assert dataset.sequence_lengths.tolist() == [3, 5, 8]
    assert dataset[0].tolist() == [1, 2]


def test_mock_distribution_is_deterministic_and_bounded():
    first = _mock_source()
    second = _mock_source()
    assert np.array_equal(first.sequence_lengths, second.sequence_lengths)
    assert first.sequence_lengths.min() >= 4
    assert first.sequence_lengths.max() <= 12


def test_mock_verification_mode_concatenates_indexed_documents(monkeypatch):
    class _FakeIndexedDataset:
        def __init__(self, path):
            assert path == "indexed-prefix"
            self.documents = [np.array([11, 12]), np.array([21, 22, 23])]

        def __len__(self):
            return len(self.documents)

        def __getitem__(self, idx):
            return self.documents[idx]

    monkeypatch.setattr(sft_dataset_module, "IndexedDataset", _FakeIndexedDataset)
    dataset = _mock_source(
        mode="verification",
        path="indexed-prefix",
        min_sequence_length=5,
        max_sequence_length=5,
        mean_sequence_length=5,
        lognormal_sigma=0,
    )
    assert dataset[0].tolist() == [11, 12, 21, 22]


def test_mock_sft_keeps_fixed_width_cu_seqlens_and_real_eod(tmp_path):
    path = tmp_path / "lengths.txt"
    path.write_text("4\n6\n", encoding="utf-8")
    config = _config(
        tokenizer=_FakeTokenizer(eod=0, pad=None),
        sequence_length=8,
        mock_data_mode="file",
        mock_data_path=str(path),
        emit_packing_metadata=True,
    )
    dataset = _mid_level(
        MockSFTDataset, MockSFTDataset.build_low_level_dataset("unused", config), config
    )
    sample = dataset[0]
    assert sample["tokens"].shape == (8,)
    assert sample["cu_seqlens"].shape == (9,)
    assert sample["cu_seqlens"].tolist() == [0] + [8] * 8
    assert sample["cu_seqlens_original"].tolist() == [0] + [3] * 8
    assert sample["labels"][2].item() == config.tokenizer.eod
    assert sample["loss_mask"][2].item() == 1.0
    assert sample["loss_mask"][3:].sum().item() == 0.0


def test_sft_keeps_current_main_fixed_width_cu_seqlens():
    config = _config(sequence_length=8)
    conversations = [
        {"role": "system", "content": ""},
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "system", "content": ""},
        {"role": "user", "content": "cd"},
        {"role": "assistant", "content": "e"},
    ]
    sample = _mid_level(SFTDataset, [conversations], config)[0]
    assert sample["cu_seqlens"].shape == (9,)
    assert sample["cu_seqlens"][-1].item() == 8
    assert sample["cu_seqlens"][-2].item() == 8
    assert "cu_seqlens_original" not in sample


def test_sft_emits_fixed_width_real_boundaries_for_packing_scheduler():
    config = _config(sequence_length=8, emit_packing_metadata=True)
    conversations = [
        {"role": "system", "content": ""},
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "system", "content": ""},
        {"role": "user", "content": "cd"},
        {"role": "assistant", "content": "e"},
    ]
    sample = _mid_level(SFTDataset, [conversations], config)[0]

    assert sample["cu_seqlens_original"].shape == sample["cu_seqlens"].shape
    assert sample["cu_seqlens_original"].tolist() == [0, 1] + [3] * 7
    assert torch.all(sample["cu_seqlens_original"] <= sample["cu_seqlens"])


def test_sft_no_pad_token_keeps_real_eod_supervision():
    class _EodTokenizer(_FakeTokenizer):
        def tokenize_conversation(
            self, messages, return_target: bool = True, add_generation_prompt: bool = False
        ):
            del messages
            assert return_target and not add_generation_prompt
            return torch.tensor([7, self.eod]), torch.tensor([IGNORE_INDEX, self.eod])

    config = _config(tokenizer=_EodTokenizer(eod=0, pad=None), sequence_length=8)
    conversations = [[{"role": "system", "content": ""}]]
    sample = _mid_level(SFTDataset, conversations, config)[0]

    assert sample["labels"][0].item() == config.tokenizer.eod
    assert sample["loss_mask"][0].item() == 1.0
    assert sample["loss_mask"][1:].sum().item() == 0.0


def test_sft_truncation_metadata_excludes_final_padding_target():
    config = _config(sequence_length=8, emit_packing_metadata=True)
    conversations = [
        {"role": "system", "content": ""},
        {"role": "user", "content": "abcdefghij"},
        {"role": "assistant", "content": "answer"},
    ]
    sample = _mid_level(SFTDataset, [conversations], config)[0]

    assert sample["cu_seqlens"][:2].tolist() == [0, 8]
    assert sample["cu_seqlens_original"][:2].tolist() == [0, 7]


def test_sft_packing_metadata_folds_zero_real_token_tail():
    config = _config(sequence_length=8, emit_packing_metadata=True)
    conversations = [
        {"role": "system", "content": ""},
        {"role": "user", "content": "abcdefg"},
        {"role": "system", "content": ""},
        {"role": "user", "content": "xy"},
    ]
    sample = _mid_level(SFTDataset, [conversations], config)[0]

    assert sample["cu_seqlens"].tolist() == [0] + [8] * 8
    assert sample["cu_seqlens_original"].tolist() == [0] + [6] * 8


def test_sft_packing_metadata_rejects_record_without_prediction_steps():
    class _SingleTokenTokenizer(_FakeTokenizer):
        def tokenize_conversation(
            self, messages, return_target: bool = True, add_generation_prompt: bool = False
        ):
            del messages
            assert return_target and not add_generation_prompt
            return torch.tensor([7]), torch.tensor([7])

    config = _config(tokenizer=_SingleTokenTokenizer(), emit_packing_metadata=True)
    conversations = [[{"role": "system", "content": ""}]]
    dataset = _mid_level(SFTDataset, conversations, config)

    with pytest.raises(ValueError, match="two or more tokens"):
        dataset[0]


@pytest.mark.parametrize(
    "path,expected",
    [
        ("owner/repository", True),
        ("/tmp/data.jsonl", False),
        ("./data.jsonl", False),
        ("../data.jsonl", False),
        ("data.jsonl", False),
        ("missing/data.jsonl", False),
        ("wikitext", True),
        (None, False),
    ],
)
def test_looks_like_hf_id(path, expected):
    assert _looks_like_hf_id(path) is expected


def test_alpaca_converter_supports_context_and_field_synonyms():
    messages = _alpaca_to_messages(
        {"prompt": "Question", "context": "Details", "response": "Answer"}
    )
    assert [message["role"] for message in messages] == ["system", "user", "assistant"]
    assert messages[1]["content"] == "Question\n\nDetails"
    assert messages[2]["content"] == "Answer"


def test_sharegpt_converter_maps_roles_and_prepends_system():
    messages = _sharegpt_to_messages(
        {
            "conversations": [
                {"from": "human", "value": "Question"},
                {"from": "gpt", "value": "Answer"},
            ]
        }
    )
    assert [message["role"] for message in messages] == ["system", "user", "assistant"]


def test_messages_converter_strips_extra_keys():
    messages = _messages_passthrough(
        {
            "messages": [
                {"role": "user", "content": "Question", "name": "Alice"},
                {"role": "assistant", "content": "Answer", "tool_calls": []},
            ]
        }
    )
    assert [set(message) for message in messages] == [
        {"role", "content"},
        {"role", "content"},
        {"role", "content"},
    ]


def test_converters_reject_multimodal_content():
    with pytest.raises(ValueError, match="must be a string"):
        _messages_passthrough({"messages": [{"role": "user", "content": [{"type": "image"}]}]})


def test_converter_selection_priority_and_pretrain_text():
    converter, schema = _select_converter(["text", "messages", "instruction", "output"])
    assert converter is _messages_passthrough
    assert schema == "openai-messages"
    converter, schema = _select_converter(["text", "metadata"])
    assert converter is _raw_text_loader
    assert schema == "pretrain-text"
    assert converter({"text": "document"}) == "document"


def test_converter_selection_rejects_unknown_schema():
    with pytest.raises(ValueError, match="cannot infer"):
        _select_converter(["id", "metadata"])


def test_local_jsonl_loader_handles_heterogeneous_columns(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        '{"instruction":"Q1","output":"A1"}\n'
        '{"instruction":"Q2","output":"A2","source":"extra"}\n',
        encoding="utf-8",
    )
    dataset = VarlenLowLevelDataset(str(path))
    assert dataset.schema_name == "alpaca"
    assert len(dataset) == 2
    assert dataset[1][2] == {"role": "assistant", "content": "A2"}


def test_local_json_array_loader_supports_pretrain_text(tmp_path):
    path = tmp_path / "data.json"
    path.write_text('[{"text":"first"},{"text":"second","id":2}]', encoding="utf-8")
    dataset = VarlenLowLevelDataset(str(path))
    assert dataset.schema_name == "pretrain-text"
    assert [dataset[index] for index in range(len(dataset))] == ["first", "second"]


def test_local_loader_selects_schema_per_record(tmp_path):
    path = tmp_path / "mixed.json"
    path.write_text(
        '[{"messages":[{"role":"user","content":"question"}]},' '{"text":"document"}]',
        encoding="utf-8",
    )
    dataset = VarlenLowLevelDataset(str(path))

    assert dataset[0][0]["role"] == "system"
    assert dataset[1] == "document"


def test_varlen_thd_pads_to_divisor_and_keeps_real_eod():
    config = _config(
        VarlenDatasetConfig, tokenizer=_FakeTokenizer(eod=0, pad=None), context_parallel_size=2
    )
    dataset = _mid_level(VarlenDataset, ["abc"], config)
    sample = dataset[0]
    assert sample["original_seq_len"].item() == 3
    assert sample["padded_seq_len"].item() == 4
    assert sample["labels"][2].item() == config.tokenizer.eod
    assert sample["loss_mask"].tolist() == [1.0, 1.0, 1.0, 0.0]


def test_varlen_sft_masks_prompt_targets():
    config = _config(VarlenDatasetConfig)
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Answer"},
    ]
    sample = _mid_level(VarlenDataset, [messages], config)[0]
    assert torch.all(sample["loss_mask"][sample["labels"] == IGNORE_INDEX] == 0)
    assert sample["loss_mask"].sum() > 0


def test_varlen_empty_text_produces_nonempty_sample():
    config = _config(VarlenDatasetConfig)
    sample = _mid_level(VarlenDataset, [""], config)[0]
    assert sample["tokens"].numel() >= 1
    assert sample["labels"].shape == sample["tokens"].shape


def test_varlen_sbhd_validation_returns_fixed_width_sample():
    config = _config(
        VarlenDatasetConfig,
        tokenizer=_FakeTokenizer(eod=0, pad=None),
        sequence_length=8,
        varlen_sbhd_validation=True,
    )
    sample = _mid_level(VarlenDataset, ["abc"], config)[0]
    assert set(sample) == {"tokens", "labels", "loss_mask", "position_ids"}
    assert sample["tokens"].shape == (8,)
    assert sample["loss_mask"].tolist() == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_mock_varlen_matches_real_varlen_metadata_contract():
    config = _config(
        VarlenDatasetConfig,
        tokenizer=_FakeTokenizer(eod=0, pad=None),
        context_parallel_size=2,
        mock_data_min_sequence_length=5,
        mock_data_max_sequence_length=5,
        mock_data_mean_sequence_length=5,
        mock_data_lognormal_sigma=0,
        mock_data_size=2,
    )
    low_level = MockVarlenDataset.build_low_level_dataset("unused", config)
    sample = _mid_level(MockVarlenDataset, low_level, config)[0]
    assert set(sample) == {
        "tokens",
        "labels",
        "loss_mask",
        "position_ids",
        "original_seq_len",
        "padded_seq_len",
    }
    assert sample["original_seq_len"].item() == 4
    assert sample["padded_seq_len"].item() == 4


def test_mock_varlen_supports_null_tokenizer_interface_and_bounds_ids():
    class _NullTokenizerLike:
        vocab_size = 4
        eod = 3
        pad_id = -1
        unique_identifiers = {"class": "_NullTokenizerLike"}

    config = _config(
        VarlenDatasetConfig,
        tokenizer=_NullTokenizerLike(),
        sequence_length=8,
        mock_data_min_sequence_length=8,
        mock_data_max_sequence_length=8,
        mock_data_mean_sequence_length=8,
        mock_data_lognormal_sigma=0,
        mock_data_size=1,
    )
    low_level = MockVarlenDataset.build_low_level_dataset("unused", config)
    sample = _mid_level(MockVarlenDataset, low_level, config)[0]

    assert sample["tokens"].min().item() >= 0
    assert sample["tokens"].max().item() < config.tokenizer.vocab_size


def test_varlen_truncation_does_not_invent_supervised_eod():
    config = _config(VarlenDatasetConfig, sequence_length=4)
    sample = _mid_level(VarlenDataset, ["abcdef"], config)[0]

    assert sample["labels"][-1].item() != config.tokenizer.eod
