# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for ModelOpt Hugging Face finetuning dataset helpers."""

import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("datasets")
pytest.importorskip("modelopt")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
transformers = pytest.importorskip("transformers")

_MODEL_OPT_EXAMPLE_DIR = Path(__file__).parents[3] / "examples" / "post_training" / "modelopt"
sys.path.insert(0, str(_MODEL_OPT_EXAMPLE_DIR))

import finetune
from finetune import SFTDataset, validate_eagle_streaming_args
from utils import get_eos_token_id

from megatron.post_training.arguments import add_modelopt_args


class _Tokenizer:
    eos_token = "<|return|>"
    eos_token_id = 3

    def __init__(self, converted_id):
        self.converted_id = converted_id

    def __len__(self):
        return 10

    def convert_tokens_to_ids(self, _token):
        return self.converted_id


def test_finetune_data_files_argument_accepts_multiple_files():
    parser = ArgumentParser()
    add_modelopt_args(parser)

    args = parser.parse_args(
        [
            "--finetune-data-split",
            "chat",
            "--finetune-data-files",
            "data/chat-00000.parquet",
            "data/chat-00001.parquet",
        ]
    )

    assert args.finetune_data_split == "chat"
    assert args.finetune_data_files == ["data/chat-00000.parquet", "data/chat-00001.parquet"]


def test_streaming_finetune_loads_explicit_splits(tmp_path, monkeypatch):
    tokenizers = pytest.importorskip("tokenizers")
    streaming = pytest.importorskip("modelopt.torch.speculative.plugins.hf_streaming_dataset")
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizers.Tokenizer(tokenizers.models.WordLevel({"[EOS]": 0})),
        eos_token="[EOS]",
    )
    train_path = tmp_path / "train.jsonl"
    valid_path = tmp_path / "valid.jsonl"
    train_path.write_text('{"conversation_id":"train","messages":[]}\n')
    valid_path.write_text('{"conversation_id":"valid","messages":[]}\n')
    args = SimpleNamespace(
        micro_batch_size=1,
        seq_length=16,
        export_offline_model=True,
        eagle_streaming_server_urls="http://127.0.0.1:8765",
        eagle_streaming_model="target",
        eagle_streaming_answer_only_loss=False,
        eagle_streaming_train_data=str(train_path),
        eagle_streaming_valid_data=str(valid_path),
        eagle_streaming_test_data=None,
    )
    monkeypatch.setattr("finetune.get_args", lambda: args)
    monkeypatch.setattr("finetune.get_hf_tokenizer", lambda: tokenizer)

    train, valid, test = finetune.train_valid_test_sft_datasets_provider(
        [1, 1, 0], sequence_length=16
    )

    assert isinstance(train, streaming.MegatronEagleVllmStreamingDataset)
    assert isinstance(valid, streaming.MegatronEagleVllmStreamingDataset)
    assert train.entries[0]["conversation_id"] == "train"
    assert valid.entries[0]["conversation_id"] == "valid"
    assert test is None


@pytest.fixture
def streaming_args():
    return SimpleNamespace(
        eagle_streaming_server_urls="http://127.0.0.1:8765",
        eagle_streaming_model="teacher",
        eagle_streaming_train_data="train.jsonl",
        eagle_streaming_valid_data=None,
        eagle_streaming_test_data=None,
        eagle_streaming_answer_only_loss=False,
        export_offline_model=True,
        load="checkpoint",
        export_model_type="GPTModel",
        offline_distillation_data=None,
        micro_batch_size=1,
        bf16=True,
        context_parallel_size=1,
        pipeline_model_parallel_size=1,
        dynamic_context_parallel=False,
        dataloader_type="single",
        full_validation=False,
        eval_iters=0,
    )


def test_eagle_streaming_args_accept_validation_with_a_corpus(streaming_args):
    streaming_args.eval_iters = 1
    streaming_args.eagle_streaming_valid_data = "validation.jsonl"
    validate_eagle_streaming_args(streaming_args)


@pytest.mark.parametrize(
    ("options", "error"),
    [
        ({"eagle_streaming_server_urls": None}, "--eagle-streaming-server-urls"),
        ({"eagle_streaming_train_data": None}, "--eagle-streaming-train-data"),
        ({"export_model_type": "HybridModel"}, "offline EAGLE GPTModel checkpoint"),
        ({"offline_distillation_data": "dump"}, "--offline-distillation-data"),
        ({"micro_batch_size": 2}, "--micro-batch-size 1"),
        ({"context_parallel_size": 2}, "BF16 and CP=PP=1"),
        ({"dataloader_type": "external"}, "built-in sampler"),
        ({"eval_iters": 1}, "--eagle-streaming-valid-data"),
    ],
)
def test_eagle_streaming_args_reject_unsupported_configuration(streaming_args, options, error):
    vars(streaming_args).update(options)
    with pytest.raises(ValueError, match=error):
        validate_eagle_streaming_args(streaming_args)


@pytest.mark.parametrize("source", ["streaming", "disk", "sft"])
def test_eagle_finetune_forward_batch_alignment(monkeypatch, streaming_args, source):
    offline = source != "sft"
    if source != "streaming":
        streaming_args.eagle_streaming_server_urls = None
    streaming_args.reset_position_ids = False
    streaming_args.reset_attention_mask = False
    streaming_args.eod_mask_loss = False
    monkeypatch.setattr(finetune, "get_args", lambda: streaming_args)
    monkeypatch.setattr(finetune, "_is_offline_model", lambda: offline)
    monkeypatch.setattr(
        finetune,
        "get_run_config",
        lambda: SimpleNamespace(
            model=SimpleNamespace(transformer=SimpleNamespace(max_seq_length_per_sample=3))
        ),
    )
    monkeypatch.setattr(finetune, "get_eos_token_id", lambda: 99)
    monkeypatch.setattr(finetune, "get_context_parallel_group", lambda: None)
    monkeypatch.setattr(finetune.mpu, "is_pipeline_first_stage", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        finetune,
        "get_timers",
        lambda: lambda *a, **kw: SimpleNamespace(start=lambda: None, stop=lambda: None),
    )

    sample = {
        "input_ids": torch.tensor([[11, 23, 37, 41]]),
        "loss_mask": torch.tensor([[0, 1, 0, 1]]),
        "aux_hidden_states": torch.tensor([[[2, 3], [5, 7], [11, 13]]], dtype=torch.bfloat16),
        "hidden_states": torch.tensor([[[17], [19], [29]]], dtype=torch.bfloat16),
    }
    broadcasts = []

    def broadcast(keys, data, dtype):
        broadcasts.append(keys)
        result = {key: data[key] for key in keys}
        assert all(value.dtype == dtype for value in result.values())
        return result

    monkeypatch.setattr(finetune.tensor_parallel, "broadcast_data", broadcast)

    class Model:
        eagle_offline = offline

        def __call__(self, tokens, position_ids, attention_mask, *, labels, **features):
            assert tokens.tolist() == [[11, 23, 37]]
            assert labels.tolist() == [[23, 37, 41]]
            assert position_ids.tolist() == [[0, 1, 2]]
            if offline:
                assert features.pop("aux_hidden_states").tolist() == [
                    [[2, 3]],
                    [[5, 7]],
                    [[11, 13]],
                ]
                assert features.pop("hidden_states").tolist() == [[[17]], [[19]], [[29]]]
            assert features == ({"base_hidden_prenorm": True} if source == "streaming" else {})
            return labels

    output, loss_callback = finetune.forward_step(iter((sample,)), Model())
    assert output.tolist() == [[23, 37, 41]]
    assert loss_callback.args[0].tolist() == ([[1, 1, 1]] if source == "disk" else [[1, 0, 1]])
    assert broadcasts[0] == (["input_ids"] if source == "disk" else ["input_ids", "loss_mask"])
    assert broadcasts[1:] == ([["aux_hidden_states", "hidden_states"]] if offline else [])


def test_materialize_data_files_resolves_local_dataset_directory(tmp_path):
    data_file = tmp_path / "train.jsonl"
    data_file.write_text('{"messages": []}\n')

    resolved = SFTDataset._materialize_data_files(str(tmp_path), ["train.jsonl"])

    assert resolved == [str(data_file)]


def test_materialize_data_files_downloads_explicit_hub_file(monkeypatch):
    calls = []

    def _download(**kwargs):
        calls.append(kwargs)
        return "/cache/chat.parquet"

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _download)

    resolved = SFTDataset._materialize_data_files(
        "nvidia/Nemotron-Post-Training-Dataset-v2", ["data/chat.parquet"]
    )

    assert resolved == ["/cache/chat.parquet"]
    assert calls == [
        {
            "repo_id": "nvidia/Nemotron-Post-Training-Dataset-v2",
            "filename": "data/chat.parquet",
            "repo_type": "dataset",
            "token": None,
        }
    ]


def test_materialize_data_files_rejects_wildcards():
    with pytest.raises(ValueError, match="explicit files"):
        SFTDataset._materialize_data_files("org/dataset", ["data/chat-*.parquet"])


def test_infer_data_files_builder_rejects_mixed_formats():
    with pytest.raises(ValueError, match="single format"):
        SFTDataset._infer_data_files_builder(["train.jsonl", "train.parquet"])


def test_load_parquet_data_files_ignores_huggingface_schema_metadata(tmp_path):
    data_file = tmp_path / "train.parquet"
    table = pa.table({"text": ["sample"]}).replace_schema_metadata(
        {b"huggingface": b"unsupported feature metadata"}
    )
    pq.write_table(table, data_file)

    dataset = SFTDataset._load_parquet_data_files([str(data_file)])

    assert dataset["text"] == ["sample"]


def test_explicit_dataset_loading_does_not_use_global_barriers(monkeypatch):
    monkeypatch.setattr(
        torch.distributed,
        "barrier",
        lambda: pytest.fail("dataset loading must not enter a global barrier"),
    )

    dataset = object()

    assert SFTDataset._load_dataset_synchronized(lambda: dataset) is dataset


@pytest.mark.parametrize(
    ("input_ids", "expected"),
    [
        ([1, 2], [1, 2]),
        ([[1, 2]], [1, 2]),
        (torch.tensor([[1, 2]]), [1, 2]),
        (transformers.BatchEncoding({"input_ids": [[1, 2]]}), [1, 2]),
    ],
)
def test_normalize_input_ids(input_ids, expected):
    assert SFTDataset._normalize_input_ids(input_ids) == expected


def test_normalize_input_ids_rejects_batches():
    with pytest.raises(ValueError, match="one tokenized chat sample"):
        SFTDataset._normalize_input_ids([[1, 2], [3, 4]])


def test_get_eos_token_id_prefers_valid_converted_id():
    assert get_eos_token_id(_Tokenizer(converted_id=7)) == 7


def test_get_eos_token_id_falls_back_from_out_of_vocab_chat_id():
    assert get_eos_token_id(_Tokenizer(converted_id=199999)) == 3


def test_get_eos_token_id_rejects_out_of_vocab_fallback():
    tokenizer = _Tokenizer(converted_id=199999)
    tokenizer.eos_token_id = 10

    with pytest.raises(ValueError, match="outside the tokenizer vocabulary"):
        get_eos_token_id(tokenizer)


@pytest.mark.parametrize("original_num_layers, expected", [(None, False), (32, True)])
def test_offline_mode_follows_the_model_config(original_num_layers, expected):
    from megatron.training.models.transforms import ModelTransformConfig

    cfg = Namespace(
        model=Namespace(
            transformations=ModelTransformConfig(original_num_layers=original_num_layers)
        )
    )
    with patch.object(finetune, "get_run_config", return_value=cfg):
        assert finetune._is_offline_model() is expected
