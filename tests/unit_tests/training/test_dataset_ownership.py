# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The data section is owned by PretrainConfigContainer: construction from the arguments, the
fields the container writes, the lifecycle and the serialization."""

import copy
import json
import re
from argparse import ArgumentParser, Namespace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import yaml

from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.datasets.gpt_fim_dataset import GPTFIMDataset, GPTFIMDatasetConfig
from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.core.transformer import TransformerConfig
from megatron.training import argument_utils, arguments, global_vars
from megatron.training.argument_utils import (
    _DATASET_FIELD_SOURCES,
    dataloader_config_from_args,
    gpt_data_config_from_args,
    model_config_from_args,
    pretrain_cfg_container_from_args,
    requested_dataset_family,
)
from megatron.training.arguments import parse_args, validate_args
from megatron.training.config import DataConfig, DataloaderConfig, PretrainConfigContainer
from megatron.training.config import container as container_module
from megatron.training.config.dataset_config import CONTAINER_WRITTEN_FIELDS
from megatron.training.datasets.sft_dataset import MockSFTDataset, SFTDataset
from megatron.training.datasets.varlen_dataset import MockVarlenDataset, VarlenDataset
from megatron.training.models import GPTModelConfig, LLaVAModelConfig, T5ModelConfig
from tests.unit_tests.test_utilities import Utils

MODEL_ARGV = (
    "--num-layers 2 --hidden-size 128 --num-attention-heads 8 --micro-batch-size 1 "
    "--global-batch-size 8 --seq-length 32 --max-position-embeddings 32 "
    "--tokenizer-type NullTokenizer --vocab-size 2048"
).split()
SOURCE = ["--data-path", "prefix-a", "--split", "9,1,0"]
BLEND = (["prefix-a", "prefix-b"], [0.3, 0.7])
# Megatron Core's defaults for the fields the container writes: what an unset field holds.
CORE_DEFAULTS = {
    "data_parallel_size": 1, "context_parallel_size": None, "sequence_parallel_size": 0,
    "dynamic_context_parallel": False,
    "multiple_validation_sets": None, "full_validation": None,
}  # fmt: skip


def _written_fields(dataset):
    return {name: getattr(dataset, name) for name in CORE_DEFAULTS}


@pytest.fixture(scope="module")
def tokenizer():
    return build_tokenizer(
        Namespace(
            vocab_size=2048,
            tokenizer_type="NullTokenizer",
            rank=0,
            make_vocab_size_divisible_by=128,
            tensor_model_parallel_size=1,
            pad_vocab_size=True,
        )
    )


@pytest.fixture
def files(tmp_path):
    data_args = tmp_path / "data_args.txt"
    data_args.write_text("0.3 prefix-a 0.7 prefix-b\n")
    per_split = tmp_path / "per_split.json"
    per_split.write_text(
        json.dumps({"train": "prefix-a prefix-b", "valid": ["prefix-c"], "test": "prefix-d"})
    )
    sequences = tmp_path / "sequences.json"
    sequences.write_text(json.dumps({"prefix-a": [10, 2], "prefix-b": [20, 4]}))
    return {"data_args": str(data_args), "per_split": str(per_split), "sequences": str(sequences)}


def _data_argv(files):
    return [
        "--seq-length", "64",
        "--data-path", "0.3", "prefix-a", "0.7", "prefix-b",
        "--split", "9,1,0",
        "--num-workers", "0",
        "--dataloader-type", "cyclic",
        "--no-data-sharding",
        "--reset-position-ids",
        "--reset-attention-mask",
        "--eod-mask-loss",
        "--dataloader-inter-document-masking",
        "--no-create-attention-mask-in-dataloader",
        "--data-cache-path", "/cache",
        "--dataloader-defer-npy-index-mmap",
        "--no-mmap-bin-files",
        "--per-dataset-sequences-path", files["sequences"],
        "--phase-transition-iterations", "20,10",
        "--sft-mock-dataset-config-json", "{}",
        "--varlen-mock-dataset-config-json", "{\"mode\": \"file\"}",
        "--allow-ambiguous-pad-tokens",
        "--mid-level-dataset-surplus", "0.1",
        "--num-dataset-builder-threads", "3",
        "--object-storage-cache-path", "/object-cache",
        "--fim-rate", "0.9",
        "--fim-spm-rate", "0.4",
        "--fim-split-sample", "<split>",
        "--fim-fragment-rate", "0.2",
        "--fim-no-prefix", "nofim",
        "--fim-prefix-token", "<p>",
        "--fim-middle-token", "<m>",
        "--fim-suffix-token", "<s>",
        "--fim-pad-token", "<x>",
        "--fim-eod-token", "<e>",
    ]  # fmt: skip


def _parse(argv):
    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    return parser.parse_args(list(argv))


def _validated(extra):
    with patch("sys.argv", ["test_dataset_ownership.py", *MODEL_ARGV, *extra]):
        return validate_args(parse_args())


@pytest.fixture
def container_factory(monkeypatch):
    """pretrain_cfg_container_from_args on parsed arguments, without the optimizer / DDP setup."""
    from megatron.training import argument_utils

    monkeypatch.setattr(
        argument_utils, "get_megatron_ddp_config", lambda args, **_: DistributedDataParallelConfig()
    )

    def build(argv, *, data=False, **kwargs):
        # These tests exercise data translation; supply the required model section separately.
        args = _parse(["--micro-batch-size", "1", *argv])
        args.batch_parallel_size = 1
        monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", args)
        if data:
            kwargs["data_cfg"] = gpt_data_config_from_args(args)
        kwargs.setdefault("model_cfg", _layout())
        return args, pretrain_cfg_container_from_args(args, **kwargs)

    return build


@pytest.fixture
def warnings_log(monkeypatch):
    log = []
    monkeypatch.setattr(argument_utils, "warn_rank_0", log.append)
    return log


def _layout(**overrides):
    layout = dict(
        fp16=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        dynamic_context_parallel=False,
        sequence_packing_scheduler=None,
        calculate_per_token_loss=True,
        max_seqlen_per_dp_cp_rank=None,
        gtp_weight_remat_size=1,
        tensor_parallel_num_sequence_shards=None,
        cuda_graph_impl="none",
    )
    layout.update(overrides)
    # What Megatron Core derives at construction: a scheduler sets variable_seq_lengths.
    layout.setdefault("variable_seq_lengths", layout["sequence_packing_scheduler"] is not None)
    transformer = TransformerConfig(num_layers=2, hidden_size=128, num_attention_heads=8)
    for name, value in layout.items():
        setattr(transformer, name, value)
    return GPTModelConfig(transformer=transformer, seq_length=4096, vocab_size=2048)


# Construction from the arguments.


def test_container_from_args_builds_the_core_fields_from_every_data_argument(
    container_factory, files
):
    args, cfg = container_factory(_data_argv(files), data=True)
    data = cfg.data
    dataset = data.dataset
    assert type(dataset) is GPTDatasetConfig and data.dataset_type is GPTDataset
    for field_name, source in _DATASET_FIELD_SOURCES.items():
        assert getattr(dataset, field_name) == getattr(args, source), field_name
    assert dataset.sequence_length == 64 and dataset.split == "9,1,0"
    assert dataset.blend == BLEND and dataset.blend_per_split is None
    assert dataset.sequences_per_dataset == {"prefix-a": [10, 2], "prefix-b": [20, 4]}
    assert dataset.create_attention_mask is False and dataset.inter_document_masking is True
    assert dataset.path_to_cache == "/cache" and dataset.defer_npy_index_mmap is True
    assert dataset.mmap_bin_files is False and dataset.allow_ambiguous_pad_tokens is True
    # The FIM flags are parsed but belong to GPTFIMDatasetConfig; a plain config has no FIM field.
    assert not hasattr(dataset, "fim_rate")
    # Phase transitions stay a training argument.
    assert args.phase_transition_iterations == "20,10" and not hasattr(
        dataset, "phase_transition_iterations"
    )
    # Runtime state and the fields the container writes are not taken from the arguments by the
    # container factory.
    assert dataset.tokenizer is None
    assert _written_fields(dataset) == CORE_DEFAULTS
    assert data.dataloader == DataloaderConfig(
        dataloader_type="cyclic", num_workers=0, data_sharding=False
    )
    assert dataloader_config_from_args(args) == data.dataloader
    cfg.validate()


def test_fim_flags_build_the_fim_config_only_with_fim_data(container_factory, files):
    args, cfg = container_factory([*_data_argv(files), "--fim-data"], data=True)
    dataset = cfg.data.dataset
    assert type(dataset) is GPTFIMDatasetConfig and cfg.data.dataset_type is GPTFIMDataset
    assert dataset.fim_rate == 0.9 and dataset.fim_spm_rate == 0.4
    assert dataset.fim_split_sample == "<split>" and dataset.fim_fragment_rate == 0.2
    assert dataset.fim_no_prefix == "nofim"
    assert dataset.fim_extra_tokens == {
        "prefix": "<p>",
        "middle": "<m>",
        "suffix": "<s>",
        "pad": "<x>",
        "eod": "<e>",
    }
    assert dataset.sequence_length == 64 and dataset.blend == BLEND  # the shared fields, as before
    cfg.validate()
    _, plain = container_factory(_data_argv(files), data=True)
    assert type(plain.data.dataset) is GPTDatasetConfig


@pytest.mark.parametrize(
    ("extra", "dataset_type", "fim_config", "warning"),
    [
        (SOURCE, GPTDataset, False, None),
        (["--mock-data"], MockGPTDataset, False, None),
        ([], MockGPTDataset, False, "No data source is declared"),
        (["--sft", *SOURCE], SFTDataset, False, None),
        (["--sft", "--mock-data"], MockSFTDataset, False, None),
        (["--use-varlen-dataset", *SOURCE], None, False, None),
        (["--use-varlen-dataset", "--mock-data"], MockVarlenDataset, False, None),
        (["--fim-data", *SOURCE], GPTFIMDataset, True, None),
        (["--sft", "--fim-data", *SOURCE], SFTDataset, False, "'sft' wins"),
        (["--use-varlen-dataset", "--fim-data", *SOURCE], None, False, "'varlen' wins"),
    ],
)  # fmt: skip
def test_dataset_class_follows_the_legacy_flag_precedence(
    warnings_log, extra, dataset_type, fim_config, warning
):
    """One rule for every GPT entry point: --sft, else --use-varlen-dataset, else --fim-data, else
    gpt, with --mock-data (or no data source) selecting the family's mock class."""
    from megatron.training.datasets.varlen_dataset import VarlenDataset

    data = gpt_data_config_from_args(
        _parse(["--seq-length", "32", "--no-create-attention-mask-in-dataloader", *extra])
    )
    expected = VarlenDataset if dataset_type is None else dataset_type
    assert data.dataset_type is expected
    assert isinstance(data.dataset, GPTFIMDatasetConfig) is fim_config
    data.validate()
    data.finalize()
    assert data.dataset_type is expected
    if warning is None:
        assert warnings_log == []
    else:
        assert len(warnings_log) == 1 and warning in warnings_log[0]


def test_fim_without_a_source_is_rejected_at_validation(warnings_log):
    """The fim family has no mock dataset: the real class stays and validation reports it."""
    data = gpt_data_config_from_args(_parse(["--seq-length", "32", "--fim-data"]))
    assert data.dataset_type is GPTFIMDataset and isinstance(data.dataset, GPTFIMDatasetConfig)
    with pytest.raises(ValueError, match="no mock dataset for the fim family"):
        data.validate()
    assert warnings_log == []


@pytest.mark.parametrize("flag", ["data_args", "per_split"])
def test_mock_data_next_to_a_file_backed_source_is_rejected(files, flag):
    option = {"data_args": "--data-args-path", "per_split": "--per-split-data-args-path"}[flag]
    argv = ["--seq-length", "32", "--mock-data", option, files[flag]]
    if flag == "data_args":
        argv += ["--split", "9,1,0"]
    with pytest.raises(ValueError, match="single data source"):
        gpt_data_config_from_args(_parse(argv))


def test_file_backed_sources_are_resolved_by_the_factory(files):
    data = gpt_data_config_from_args(
        _parse(["--seq-length", "32", "--data-args-path", files["data_args"], "--split", "9,1,0"])
    )
    assert data.dataset.blend == BLEND and data.dataset_type is GPTDataset
    data = gpt_data_config_from_args(
        _parse(["--seq-length", "32", "--per-split-data-args-path", files["per_split"]])
    )
    assert data.dataset.blend is None and data.dataset.split is None
    assert data.dataset.blend_per_split == [
        (["prefix-a", "prefix-b"], None),
        (["prefix-c"], None),
        (["prefix-d"], None),
    ]
    data.validate()
    data = gpt_data_config_from_args(
        _parse(
            ["--seq-length", "32", "--train-data-path", "prefix-a", "--valid-data-path", "prefix-b"]
        )
    )
    assert data.dataset.blend_per_split == [(["prefix-a"], None), (["prefix-b"], None), None]
    data.validate()


def test_conflicting_dataset_families_are_rejected():
    with pytest.raises(ValueError, match="mutually exclusive"):
        requested_dataset_family(Namespace(sft=True, use_varlen_dataset=True))


# The loader-only default of every other entry point.


def test_loader_only_is_the_default_for_other_entry_points(container_factory):
    # A T5 / BERT style command line declares data paths and a tokenizer; the container factory
    # does not infer a GPT dataset from them.
    args, cfg = container_factory(
        ["--seq-length", "64", *SOURCE, "--num-workers", "4", "--dataloader-type", "cyclic"]
    )
    data = cfg.data
    assert data.dataset is None and data.dataset_type is None
    assert data.dataloader == DataloaderConfig(dataloader_type="cyclic", num_workers=4)
    cfg.validate()
    cfg.finalize()
    assert data.dataset is None and data.dataset_type is None
    with pytest.raises(ValueError, match="no dataset"):
        data.builder_inputs()
    serialized = cfg.to_dict()
    assert serialized["data"]["dataset"] is None and serialized["data"]["dataset_type"] is None
    restored = PretrainConfigContainer.from_dict(serialized)
    assert restored.data.dataset is None and restored.data.dataloader == data.dataloader
    assert restored.to_dict() == serialized


def test_container_keeps_a_caller_supplied_data_section(container_factory):
    preset = DataConfig(
        dataset=GPTDatasetConfig(
            random_seed=7,  # Core's required argument: a Python caller passes its own seed
            sequence_length=8,
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
        )
    )
    _, cfg = container_factory(["--seq-length", "64"], data_cfg=preset)
    assert cfg.data is preset
    cfg.validate()
    cfg.finalize()
    assert (
        cfg.data.dataset_type is MockGPTDataset and cfg.data.dataloader.dataloader_type == "single"
    )
    assert cfg.data.dataset.random_seed == 7 and cfg.rng.seed == 1234  # independent seeds


# The fields the container writes and the cross-section checks.


def test_container_writes_the_owned_fields_at_finalize(container_factory, monkeypatch):
    args, cfg = container_factory(
        ["--seq-length", "64", "--mock-data", "--sft", "--seed", "99", "--full-validation"],
        data=True,
    )
    dataset = cfg.data.dataset
    assert args.seed == 99 and cfg.rng.seed == 99 and dataset.random_seed == 99
    cfg.model = _layout(
        tensor_model_parallel_size=2,
        context_parallel_size=2,
        sequence_parallel=True,
        dynamic_context_parallel=True,
    )
    monkeypatch.setattr(container_module, "safe_get_world_size", lambda: 8)
    # The owners stay editable after construction: no stale copy on the dataset config. The dataset
    # seed is the section's own setting and does not follow rng.seed.
    cfg.rng.seed = 42
    cfg.validation.multiple_validation_sets = True
    cfg.validate()
    assert dataset.random_seed == 99  # validation writes nothing
    # data_parallel_size needs the world size: an explicit wrong value passes validate() and is
    # rejected by finalize() before anything is written.
    dataset.data_parallel_size = 99
    with pytest.raises(ValueError, match="data.dataset.data_parallel_size=99 disagrees"):
        cfg.finalize()
    assert dataset.data_parallel_size == 99 and dataset.random_seed == 99
    assert not cfg._finalized
    dataset.data_parallel_size = 2  # an agreeing explicit value is accepted
    cfg.finalize()
    assert dataset.random_seed == 99 and cfg.rng.seed == 42
    assert dataset.full_validation is True and dataset.multiple_validation_sets is True
    assert (dataset.context_parallel_size, dataset.sequence_parallel_size) == (2, 2)
    assert dataset.dynamic_context_parallel is True and dataset.data_parallel_size == 2
    assert cfg.data.dataset_type is MockSFTDataset
    assert not (set(CONTAINER_WRITTEN_FIELDS) & set(cfg.to_dict()["data"]["dataset"]))
    assert cfg.to_dict()["data"]["dataset"]["random_seed"] == 99
    cfg.validate()  # the written values agree with their owners
    dataset.random_seed = 7
    cfg.validate()  # the dataset seed has no owner to agree with


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("full_validation", False, "data.dataset.full_validation=False disagrees with validation.full_validation=True"),
        ("multiple_validation_sets", True, "data.dataset.multiple_validation_sets=True disagrees with validation.multiple_validation_sets=False"),
        ("context_parallel_size", 4, "data.dataset.context_parallel_size=4 disagrees with model.transformer.context_parallel_size=2"),
        ("sequence_parallel_size", 1, "data.dataset.sequence_parallel_size=1 disagrees with the model layout"),
        ("dynamic_context_parallel", True, "data.dataset.dynamic_context_parallel=True disagrees with model.transformer.dynamic_context_parallel=False"),
    ],
)  # fmt: skip
def test_each_owned_field_is_checked_against_its_owner(container_factory, field, value, message):
    """One explicit check per field: a value given on the dataset config must agree with the
    section that owns it (data_parallel_size needs the world size and is checked at finalize())."""
    _, cfg = container_factory(
        ["--seq-length", "64", "--mock-data", "--sft", "--full-validation"], data=True
    )
    cfg.model = _layout(tensor_model_parallel_size=2, context_parallel_size=2)
    cfg.validate()
    assert hasattr(cfg.data.dataset, field)
    setattr(cfg.data.dataset, field, value)
    with pytest.raises(ValueError, match=re.escape(message)):
        cfg.validate()


@pytest.mark.parametrize("position_embedding_type", ["learned_absolute", "rope", "yarn"])
def test_model_capacity_must_cover_the_sample_length(container_factory, position_embedding_type):
    _, cfg = container_factory(["--seq-length", "64", "--mock-data"], data=True)
    cfg.model = _layout()
    cfg.model.position_embedding_type = position_embedding_type
    cfg.model.seq_length = 32
    if position_embedding_type == "learned_absolute":
        with pytest.raises(ValueError, match="model.seq_length \\(32"):
            cfg.validate()
    else:
        cfg.validate()
    cfg.model.seq_length = 64
    cfg.validate()


@pytest.mark.parametrize("position_embedding_type", ["learned_absolute", "rope", "yarn"])
def test_cli_position_capacity_only_limits_learned_embeddings(position_embedding_type, monkeypatch):
    warnings = []
    monkeypatch.setattr(arguments, "warn_rank_0", warnings.append)
    extra = ["--seq-length", "64", "--position-embedding-type", position_embedding_type]
    if position_embedding_type == "learned_absolute":
        with pytest.raises(AssertionError, match="max_position_embeddings"):
            _validated(extra)
        # The legacy flag overrides the learned-absolute selection.
        extra.append("--use-rotary-position-embeddings")
    args = _validated(extra)
    assert args.max_position_embeddings == 32
    assert args.seq_length == 64
    assert any("checkpoint/export context-length" in warning for warning in warnings)


@pytest.mark.parametrize("family_flag", ["--sft", "--use-varlen-dataset"])
@pytest.mark.parametrize(
    "unsupported_flag", [None, "--reset-position-ids", "--reset-attention-mask"]
)
def test_online_dataset_cli_reports_unsupported_masks(family_flag, unsupported_flag):
    extra = [family_flag, *SOURCE, "--calculate-per-token-loss"]
    if unsupported_flag is not None:
        extra += ["--no-create-attention-mask-in-dataloader", unsupported_flag]
    data = gpt_data_config_from_args(_validated(extra))
    with pytest.raises(
        ValueError,
        match="--no-create-attention-mask-in-dataloader.*--reset-position-ids.*--reset-attention-mask",
    ):
        data.validate()  # the real SFT / varlen datasets assert these per sample


def test_the_run_execution_length_matches_fixed_dataset_rows(container_factory, monkeypatch):
    """Finalize an unset run length; reject either mismatch without mutating the config."""
    monkeypatch.setattr(container_module, "safe_get_world_size", lambda: 1)
    _, cfg = container_factory(["--seq-length", "64", "--mock-data"], data=True)
    cfg.model = _layout()
    assert cfg.model.transformer.max_seq_length_per_sample is None
    cfg.validate()
    cfg.finalize()
    assert cfg.model.transformer.max_seq_length_per_sample == 64
    _, cfg = container_factory(["--seq-length", "64", "--mock-data"], data=True)
    for width in (128, 32):
        cfg.model = _layout(max_seq_length_per_sample=width)
        with pytest.raises(ValueError, match="must equal"):
            cfg.validate()
        assert cfg.model.transformer.max_seq_length_per_sample == width
    args = _validated(["--seq-length", "48", "--max-position-embeddings", "48", "--mock-data"])
    cfg = pretrain_cfg_container_from_args(args, data_cfg=gpt_data_config_from_args(args))
    assert cfg.model.transformer.max_seq_length_per_sample == 48
    assert cfg.data.dataset.sequence_length == 48
    cfg.validate()


# Serialization and loading.


def test_full_container_round_trips_before_and_after_binding_the_tokenizer(
    container_factory, files, monkeypatch, tmp_path, tokenizer
):
    _, cfg = container_factory(_data_argv(files), data=True)
    serialized = cfg.to_dict()
    section = serialized["data"]
    assert section["dataset"]["_target_"].endswith(".GPTDatasetConfig")
    assert section["dataset_type"] is GPTDataset
    assert set(section) == {"_target_", "dataset", "dataloader", "dataset_type"}
    assert "tokenizer" not in section["dataset"] and not (
        set(CONTAINER_WRITTEN_FIELDS) & set(section["dataset"])
    )
    restored = PretrainConfigContainer.from_dict(serialized)
    assert (
        restored.data.dataset_type is GPTDataset and type(restored.data.dataset) is GPTDatasetConfig
    )
    assert restored.data.to_cfg_dict() == cfg.data.to_cfg_dict()
    assert restored.to_dict() == serialized
    path = tmp_path / "run_config.yaml"
    cfg.to_yaml(str(path))
    text = path.read_text()
    assert "_call_: false" in text
    assert "tokenizer" not in yaml.safe_load(text)["data"]["dataset"]
    from_file = PretrainConfigContainer.from_yaml(str(path))
    assert from_file.data.dataset_type is GPTDataset
    assert from_file.data.to_cfg_dict() == cfg.data.to_cfg_dict()
    # After finalize() and binding the tokenizer the file still carries no runtime state.
    monkeypatch.setattr(container_module, "safe_get_world_size", lambda: 2)
    cfg.finalize()
    cfg.data.bind_tokenizer(tokenizer)
    assert cfg.data.dataset.tokenizer is tokenizer
    after = cfg.to_dict()
    assert (
        "tokenizer" not in after["data"]["dataset"]
        and after["data"]["dataset"] == section["dataset"]
    )
    reloaded = PretrainConfigContainer.from_dict(after)
    assert reloaded.data.dataset.tokenizer is None
    assert reloaded.data.dataset.random_seed == cfg.data.dataset.random_seed  # saved with it
    reloaded.validate()
    reloaded.finalize()
    assert reloaded.data.dataset_type is GPTDataset
    assert reloaded.data.dataset.random_seed == cfg.data.dataset.random_seed
    assert _written_fields(reloaded.data.dataset) == _written_fields(cfg.data.dataset)


def test_pre_data_run_config_loads_with_the_default_section(run_config):
    """A run_config.yaml saved before the data section existed loads with the loader-only
    default: the field's dataclass default, no upgrade hook."""
    serialized = run_config.to_dict()
    legacy = copy.deepcopy(serialized)
    legacy.pop("data")
    restored = PretrainConfigContainer.from_dict(legacy)
    assert restored.data == DataConfig()
    assert "data" not in legacy  # the caller's dictionary is not modified


# The runtime services bind the tokenizer.


def test_runtime_services_bind_the_tokenizer_to_a_finalized_dataset_config(
    container_factory, monkeypatch, tokenizer
):
    _, cfg = container_factory(["--seq-length", "64", "--mock-data"], data=True)
    monkeypatch.setattr(container_module, "safe_get_world_size", lambda: 1)
    monkeypatch.setattr(global_vars, "_GLOBAL_TOKENIZER", tokenizer)
    with pytest.raises(RuntimeError, match="before the container is finalized"):
        global_vars._bind_dataset_tokenizer(cfg)  # the owned fields are not written yet
    assert cfg.data.dataset.tokenizer is None
    cfg.finalize()
    global_vars._bind_dataset_tokenizer(cfg)
    assert cfg.data.dataset.tokenizer is tokenizer
    global_vars._bind_dataset_tokenizer(cfg)  # idempotent for the run's tokenizer
    # A config bound to another tokenizer is not silently kept: bind_tokenizer() decides.
    monkeypatch.setattr(global_vars, "_GLOBAL_TOKENIZER", SimpleNamespace(vocab_size=2048))
    with pytest.raises(RuntimeError, match="another tokenizer"):
        global_vars._bind_dataset_tokenizer(cfg)
    assert cfg.data.dataset.tokenizer is tokenizer
    monkeypatch.setattr(global_vars, "_GLOBAL_TOKENIZER", None)
    _, loader_only = container_factory(["--seq-length", "64"])
    loader_only.finalize()
    global_vars._bind_dataset_tokenizer(loader_only)  # nothing to bind
    _, unbound = container_factory(["--seq-length", "64", "--mock-data"], data=True)
    unbound.finalize()
    global_vars._bind_dataset_tokenizer(unbound)  # no tokenizer (build_tokenizer=False)
    assert unbound.data.dataset.tokenizer is None
    with pytest.raises(RuntimeError, match="no tokenizer bound"):
        unbound.data.builder_inputs()


def test_set_global_variables_finalizes_a_dataset_section(
    container_factory, monkeypatch, tokenizer
):
    """The compatibility bootstrap finalizes the container before the runtime services bind the
    tokenizer (every PretrainConfigContainer, loader-only ones included, since upstream #8149)."""
    globals_ = ("_GLOBAL_ARGS", "_GLOBAL_RUN_CONFIG", "_GLOBAL_TOKENIZER", "_GLOBAL_TRAIN_STATE")
    for name in globals_:
        monkeypatch.setattr(global_vars, name, None)
    monkeypatch.setattr(container_module, "safe_get_world_size", lambda: 1)

    def build(_cfg, _args):  # main's _build_tokenizer(cfg, args) since #8391
        global_vars._GLOBAL_TOKENIZER = tokenizer
        return tokenizer

    monkeypatch.setattr(global_vars, "_build_tokenizer", build)
    monkeypatch.setattr(global_vars, "_set_wandb_writer", lambda args: None)
    monkeypatch.setattr(global_vars, "_set_telemetry", lambda args, **kwargs: None)
    monkeypatch.setattr(global_vars, "initialize_training_runtime_services", lambda args: None)
    args, cfg = container_factory(["--seq-length", "64", "--mock-data"], data=True)
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", None)  # the bootstrap registers them
    global_vars.set_global_variables(args, cfg)
    assert global_vars.get_run_config() is cfg and cfg._finalized
    assert cfg.data.dataset.tokenizer is tokenizer
    dataset_type, config, _ = cfg.data.builder_inputs()
    assert dataset_type is MockGPTDataset and config is cfg.data.dataset
    for name in globals_:
        monkeypatch.setattr(global_vars, name, None)
    args, loader_only = container_factory(["--seq-length", "64"])
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", None)
    global_vars.set_global_variables(args, loader_only, build_tokenizer=False)
    assert global_vars.get_run_config() is loader_only and loader_only._finalized


def test_dummy_train_step_reads_the_section_and_the_model_seams(monkeypatch):
    """dummy_train_step (the skip-iteration path) runs for any container: the batch flags come
    from the data section, the model-parallel values from the model section, the sample length
    from the model's resolved run maximum; the poisoned namespace shows which source won."""
    from megatron.training import training

    calls = []
    machine = Mock()
    machine.should_run_forward_backward.side_effect = [True, False, True, False]
    monkeypatch.setattr(training, "get_num_microbatches", lambda: 1)
    monkeypatch.setattr(training, "get_rerun_state_machine", lambda: machine)
    monkeypatch.setattr(
        training, "get_batch_on_this_tp_rank", lambda batch, **kw: calls.append(kw) or {}
    )
    monkeypatch.setattr(training, "get_batch_on_this_cp_rank", lambda batch, **kw: batch)
    monkeypatch.setattr(training, "get_context_parallel_group", lambda: None)
    monkeypatch.setattr(training.mpu, "get_tensor_model_parallel_rank", lambda: 1)  # no iterator
    monkeypatch.setattr(training.mpu, "get_tensor_model_parallel_src_rank", lambda: 0)
    monkeypatch.setattr(training.mpu, "get_tensor_model_parallel_group", lambda: None)
    monkeypatch.setattr(training.mpu, "is_pipeline_first_stage", lambda: True)
    monkeypatch.setattr(training.mpu, "is_pipeline_last_stage", lambda: True)
    args = SimpleNamespace(  # poisoned wherever the data section or the model section rules
        seq_length=999,
        context_parallel_size=3,
        pipeline_model_parallel_size=5,
        dynamic_context_parallel=True,
        sequence_packing_scheduler="dp_balanced",
        sft=True,
        dataloader_inter_document_masking=True,
        create_attention_mask_in_dataloader=False,
    )
    monkeypatch.setattr(training, "get_args", lambda: args)
    cfg = SimpleNamespace(
        model=_layout(
            context_parallel_size=2, pipeline_model_parallel_size=4, max_seq_length_per_sample=64
        ),
        data=DataConfig(),
        train=SimpleNamespace(micro_batch_size=3),
    )
    monkeypatch.setattr(training, "get_run_config", lambda: cfg)
    monkeypatch.setattr(torch.distributed, "broadcast", lambda tensor, **kw: tensor.fill_(False))

    # A loader-only section (an entry point whose datasets come from its own callbacks): the
    # length is the model's per-sample maximum, resolved before startup,
    # the flags are the plain-GPT ones whatever args.sft says, the model-parallel values are the
    # model section's.
    training.dummy_train_step(None)
    kw = calls.pop()
    assert (kw["cp_size"], kw["pipeline_model_parallel_size"], kw["dynamic_cp"]) == (2, 4, False)
    assert kw["seq_length"] == 64 and kw["micro_batch_size"] == 3
    assert kw["has_cu_seqlens"] is False and kw["create_attention_mask_in_dataloader"] is False

    # A dataset section and a model section: the section's flags, the model's values and length.
    dataset = GPTDatasetConfig(
        sequence_length=16,
        random_seed=1,
        inter_document_masking=True,
        create_attention_mask=False,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
    )
    cfg.model = _layout(dynamic_context_parallel=True, max_seq_length_per_sample=16)
    cfg.data = DataConfig(dataset=dataset, dataset_type=MockGPTDataset)
    training.dummy_train_step(None)
    kw = calls.pop()
    assert (kw["cp_size"], kw["pipeline_model_parallel_size"], kw["dynamic_cp"]) == (1, 1, True)
    assert kw["seq_length"] == 16
    assert kw["has_cu_seqlens"] is True and kw["create_attention_mask_in_dataloader"] is False


@pytest.mark.parametrize("with_mask", [False, True])
def test_callback_dummy_step_agrees_on_mask_presence(monkeypatch, with_mask):
    from megatron.training import training

    if Utils.world_size < 2:
        pytest.skip("Requires two TP ranks")
    Utils.initialize_model_parallel(tensor_model_parallel_size=2)
    try:
        cfg = SimpleNamespace(
            model=_layout(tensor_model_parallel_size=2, max_seq_length_per_sample=4),
            data=DataConfig(),
            train=SimpleNamespace(micro_batch_size=1),
        )
        monkeypatch.setattr(training, "get_run_config", lambda: cfg)
        monkeypatch.setattr(training, "get_num_microbatches", lambda: 1)
        machine = Mock()
        machine.should_run_forward_backward.side_effect = [True, False]
        monkeypatch.setattr(training, "get_rerun_state_machine", lambda: machine)
        received = []
        real_get_batch = training.get_batch_on_this_tp_rank

        def capture_batch(batch, **kwargs):
            batch = real_get_batch(batch, **kwargs)
            received.append(batch)
            return batch

        monkeypatch.setattr(training, "get_batch_on_this_tp_rank", capture_batch)
        batch = {
            "tokens": torch.ones((1, 4), dtype=torch.int64),
            "labels": torch.ones((1, 4), dtype=torch.int64),
            "position_ids": torch.arange(4).view(1, 4),
            "loss_mask": torch.ones((1, 4)),
        }
        if with_mask:
            batch["attention_mask"] = torch.ones((1, 1, 4, 4), dtype=torch.bool)
        training.dummy_train_step(iter([batch]))
        assert torch.equal(received[0]["tokens"], batch["tokens"].cuda())
        assert (received[0]["attention_mask"] is not None) is with_mask
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("mode", ["token-weighted", "microbatch"])
def test_sft_loss_mode_is_config_owned(container_factory, mode):
    _, cfg = container_factory(
        ["--seq-length", "64", "--sft", "--mock-data", "--sft-loss-log-mode", mode], data=True
    )
    cfg.validate()
    restored = PretrainConfigContainer.from_dict(cfg.to_dict())
    assert restored.data.dataset.sft_loss_log_mode == mode
    restored.data.dataset.sft_loss_log_mode = "invalid"
    with pytest.raises(ValueError, match="sft_loss_log_mode"):
        restored.validate()


def test_microbatch_loss_reduction_is_keyed_on_the_sft_family(monkeypatch):
    """The reporting reductions of train_step, in _reduce_microbatch_losses: SFT averages over the
    trainable tokens of the global batch (or over the valid per-microbatch means in 'microbatch'
    mode, an all-masked microbatch excluded), the others keep the per-microbatch mean; the family
    and reporting policy are the data section's."""
    import torch

    from megatron.training import training

    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, group=None: None)
    pairs = [torch.tensor([3.0, 3.0]), torch.tensor([10.0, 1.0]), torch.tensor([0.0, 0.0])]
    losses = [{"lm loss": p} for p in pairs]  # (sum, token count) per microbatch
    token_weighted = training._reduce_microbatch_losses(
        losses, is_sft=True, sft_loss_log_mode="token", group=None
    )
    assert token_weighted["lm loss"].item() == pytest.approx(13.0 / 4.0)
    per_microbatch = training._reduce_microbatch_losses(
        losses, is_sft=True, sft_loss_log_mode="microbatch", group=None
    )
    assert per_microbatch["lm loss"].item() == pytest.approx((1.0 + 10.0) / 2)  # masked mb skipped
    plain = training._reduce_microbatch_losses(
        losses, is_sft=False, sft_loss_log_mode="microbatch", group=None
    )
    assert plain["lm loss"].item() == pytest.approx(13.0 / 4.0)  # sum / sum, as before for non-SFT
    masked = training._reduce_microbatch_losses(
        [{"lm loss": torch.tensor([0.0, 0.0])}], is_sft=True, sft_loss_log_mode="token", group=None
    )
    assert masked["lm loss"].item() == 0.0
    legacy = training._reduce_microbatch_losses(
        [{"lm loss": torch.tensor([2.0])}, {"lm loss": torch.tensor([4.0])}],
        is_sft=False,
        sft_loss_log_mode="token",
        group=None,
    )
    assert legacy["lm loss"].item() == pytest.approx(3.0)
    with pytest.raises(ValueError, match="Invalid value shape"):
        training._reduce_microbatch_losses(
            [{"lm loss": torch.zeros(3)}], is_sft=False, sft_loss_log_mode="token", group=None
        )


# The data-path rules of validate_args, on the container (placed by their inputs).


def test_dynamic_cp_needs_the_single_loader_even_without_a_dataset(container_factory):
    _, cfg = container_factory(["--seq-length", "64"])  # a loader-only section
    cfg.model = _layout(dynamic_context_parallel=True)
    cfg.validate()  # unset selects single
    cfg.data.dataloader.dataloader_type = "cyclic"
    with pytest.raises(ValueError, match="needs data.dataloader.dataloader_type='single'"):
        cfg.validate()
    cfg.data.dataloader.dataloader_type = "single"
    cfg.validate()
    cfg.model = _layout(dynamic_context_parallel=False)
    cfg.data.dataloader.dataloader_type = "cyclic"
    cfg.validate()


def test_a_packing_scheduler_needs_per_token_loss(container_factory):
    _, cfg = container_factory(["--seq-length", "64", "--mock-data"], data=True)
    cfg.model = _layout(sequence_packing_scheduler="dp_balanced", calculate_per_token_loss=False)
    with pytest.raises(ValueError, match="needs model.transformer.calculate_per_token_loss=True"):
        cfg.validate()
    cfg.model = _layout(sequence_packing_scheduler="dp_balanced", calculate_per_token_loss=True)
    cfg.validate()


def test_varlen_datasets_and_the_packing_scheduler(container_factory, monkeypatch):
    """A packed varlen dataset needs a scheduler (the CLI selects one; a Python container names
    it); SBHD validation excludes one and emits fixed-width rows."""
    monkeypatch.setattr(container_module, "safe_get_world_size", lambda: 2)
    _, cfg = container_factory(["--seq-length", "64"])
    plain = GPTDatasetConfig(
        sequence_length=64,
        random_seed=1,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
    )
    cfg.data = DataConfig(
        dataset=plain, dataset_type=MockVarlenDataset, dataloader=cfg.data.dataloader
    )
    cfg.model = _layout()
    with pytest.raises(ValueError, match="needs model.transformer.sequence_packing_scheduler"):
        cfg.validate()
    cfg.model = _layout(sequence_packing_scheduler="dp_balanced", max_seq_length_per_sample=128)
    cfg.validate()
    sbhd = GPTDatasetConfig(
        sequence_length=64,
        random_seed=1,
        blend=(["prefix-a"], None),
        split="9,1,0",
        varlen_sbhd_validation=True,
        reset_position_ids=False,
        reset_attention_mask=False,
        create_attention_mask=False,
        eod_mask_loss=False,
    )
    cfg.data = DataConfig(dataset=sbhd, dataset_type=VarlenDataset, dataloader=cfg.data.dataloader)
    cfg.model = _layout(sequence_packing_scheduler="dp_balanced")
    with pytest.raises(ValueError, match="does not use a packing scheduler"):
        cfg.validate()
    for width in (128, 32):
        cfg.model = _layout(pipeline_model_parallel_size=2, max_seq_length_per_sample=width)
        before = cfg.to_dict()
        with pytest.raises(ValueError, match="must equal"):
            cfg.validate()
        assert cfg.to_dict() == before
    for width in (64, None):
        cfg.model = _layout(pipeline_model_parallel_size=2, max_seq_length_per_sample=width)
        before = cfg.to_dict()
        cfg.validate()
        assert cfg.to_dict() == before
    cfg.finalize()
    assert cfg.model.transformer.max_seq_length_per_sample == 64


def test_sample_length_is_a_multiple_of_twice_the_cp_size(container_factory):
    _, cfg = container_factory(["--seq-length", "60", "--mock-data"], data=True)
    cfg.model = _layout(context_parallel_size=4)
    with pytest.raises(
        ValueError, match="multiple of 2 \\* model.transformer.context_parallel_size"
    ):
        cfg.validate()
    cfg.model = _layout(context_parallel_size=2)  # 60 % 4 == 0
    cfg.validate()


@pytest.mark.parametrize("pack_length", [1025, 1032])
def test_sft_dynamic_cp_pack_alignment(container_factory, monkeypatch, pack_length):
    monkeypatch.setattr(container_module, "safe_get_world_size", lambda: 4)
    _, cfg = container_factory(["--seq-length", "1024", "--sft", "--mock-data"], data=True)
    cfg.model = _layout(
        dynamic_context_parallel=True,
        min_dynamic_context_parallel_size=4,
        sequence_packing_scheduler="default_dynamic_cp",
        max_seq_length_per_sample=pack_length,
        max_seqlen_per_dp_cp_rank=2048,
    )
    cfg.validate()
    if pack_length == 1025:
        before = cfg.to_dict()
        with pytest.raises(ValueError, match="dynamic CP/SP padding alignment \\(8\\)"):
            cfg.finalize()
        assert cfg.to_dict() == before
    else:
        cfg.finalize()
        assert cfg.data.dataset.packed_sequence_length == 1032
        assert cfg.data.dataset.data_parallel_size == 4


def test_packed_buffer_budget_is_checked_at_finalize(container_factory, monkeypatch):
    """cp ranks (times the data-parallel replicas under default_dynamic_cp) times the per-rank
    budget must cover one sample; it needs the world size, so it runs at finalize(), before any
    derived field is written."""
    monkeypatch.setattr(container_module, "safe_get_world_size", lambda: 4)
    _, cfg = container_factory(["--seq-length", "64", "--mock-data", "--sft"], data=True)
    cfg.model = _layout(
        context_parallel_size=2,
        sequence_packing_scheduler="dp_balanced",
        max_seqlen_per_dp_cp_rank=16,
    )
    cfg.validate()
    with pytest.raises(
        ValueError,
        match="packed-sequence buffer \\(2 ranks \\* max_seqlen_per_dp_cp_rank 16 = 32\\)",
    ):
        cfg.finalize()
    assert not cfg._finalized and cfg.data.dataset.data_parallel_size == 1  # nothing written
    cfg.model = _layout(
        context_parallel_size=2,
        sequence_packing_scheduler="dp_balanced",
        max_seqlen_per_dp_cp_rank=32,
    )
    cfg.finalize()
    assert cfg.data.dataset.data_parallel_size == 2
    _, cfg = container_factory(["--seq-length", "64", "--mock-data", "--sft"], data=True)
    cfg.model = _layout(  # 2 cp ranks * 2 replicas * 16 = 64 covers the sample
        context_parallel_size=2,
        dynamic_context_parallel=True,
        sequence_packing_scheduler="default_dynamic_cp",
        max_seqlen_per_dp_cp_rank=16,
    )
    cfg.finalize()


def test_runtime_services_read_config_lengths_not_legacy_args(container_factory, monkeypatch):
    """Runtime reads the resolved run length, not position capacity or stale CLI values.

    Dataset token accounting retains T5 encoder rows and excludes VLM image expansion.
    """
    monkeypatch.setattr(container_module, "safe_get_world_size", lambda: 1)
    args, cfg = container_factory(["--seq-length", "16", "--mock-data"], data=True)
    args.rank = 0  # the calculator takes the rank from the namespace
    cfg.model = _layout(max_seq_length_per_sample=16)
    cfg.model.seq_length = 128
    cfg.validate()
    cfg.finalize()
    assert cfg.model.transformer.max_seq_length_per_sample == 16
    args.seq_length = 64  # stale, must not be read
    monkeypatch.setattr(global_vars, "_GLOBAL_RUN_CONFIG", cfg)
    assert global_vars.build_telemetry_resource_attrs(args)["dl.sequence_length"] == 16
    calls = []
    monkeypatch.setattr(
        global_vars, "init_num_microbatches_calculator", lambda **kw: calls.append(kw)
    )
    for name in (
        "_set_tensorboard_writer",
        "_set_one_logger",
        "_set_adlr_autoresume",
        "_set_timers",
        "_set_energy_monitor",
    ):
        monkeypatch.setattr(global_vars, name, lambda *a, **k: None)
    monkeypatch.setattr(global_vars, "_set_train_state", lambda: None)
    global_vars.initialize_training_runtime_services(args)
    assert calls.pop()["seq_length"] == 16
    del args.seq_length  # a GPT run never needs it
    global_vars.initialize_training_runtime_services(args)
    assert calls.pop()["seq_length"] == 16
    # A loader-only section (the callback entry points): the model section the builder derives
    # from the arguments carries --seq-length as the per-sample maximum; the namespace is not
    # read afterwards.
    args = _validated(["--seq-length", "64", "--max-position-embeddings", "64"])
    args.rank = 0
    loader_only = pretrain_cfg_container_from_args(args)
    assert loader_only.data.dataset is None
    assert loader_only.model.transformer.max_seq_length_per_sample == 64
    args.seq_length = 999  # stale
    monkeypatch.setattr(global_vars, "_GLOBAL_RUN_CONFIG", loader_only)
    assert global_vars.build_telemetry_resource_attrs(args)["dl.sequence_length"] == 64
    global_vars.initialize_training_runtime_services(args)
    assert calls.pop()["seq_length"] == 64
    loader_only.model.transformer.max_seq_length_per_sample = None
    assert "dl.sequence_length" not in global_vars.build_telemetry_resource_attrs(args)

    # T5 dataset token accounting uses encoder rows, not the run's maximum across both sides.
    loader_only.model = T5ModelConfig(
        transformer=_layout(max_seq_length_per_sample=64).transformer,
        seq_length=128,
        encoder_num_layers=2,
        encoder_seq_length=16,
        decoder_seq_length=32,
    )
    global_vars.initialize_training_runtime_services(args)
    assert calls.pop()["seq_length"] == 16
    assert global_vars.build_telemetry_resource_attrs(args)["dl.sequence_length"] == 16


@pytest.mark.parametrize("decoder_length", [None, 96])
def test_vlm_resolves_expanded_width_before_startup(monkeypatch, decoder_length):
    """Text rows and expanded model inputs stay distinct without a second VLM length field."""
    import pretrain_vlm
    from megatron.core.pipeline_parallel.schedules import get_tensor_shapes

    monkeypatch.setattr(container_module, "safe_get_world_size", lambda: 1)
    monkeypatch.setattr(pretrain_vlm, "get_num_image_embeddings", lambda *a, **kw: 4)
    monkeypatch.setattr(pretrain_vlm.context_parallel, "get_padding", lambda *a: 0)
    argv = MODEL_ARGV + ["--seq-length", "64", "--max-position-embeddings", "128", "--mock-data"]
    if decoder_length is not None:
        argv += ["--decoder-seq-length", str(decoder_length)]
    with patch("sys.argv", ["test"] + argv):
        args = validate_args(parse_args(extra_args_provider=pretrain_vlm.add_vlm_extra_args))
    text_length, padding = pretrain_vlm.resolve_vlm_sequence_lengths(args)
    width = decoder_length or 68
    assert (text_length, padding) == (64, width - 68)
    model = model_config_from_args(args, model_config_cls=LLaVAModelConfig)
    cfg = pretrain_cfg_container_from_args(
        args, model_cfg=model, data_cfg=gpt_data_config_from_args(args)
    )
    assert not hasattr(model, "decoder_seq_length")
    assert model.transformer.max_seq_length_per_sample == width
    cfg.validate()
    cfg.finalize()
    assert model.transformer.max_seq_length_per_sample == width
    assert cfg.data.dataset.sequence_length == 64
    # Core still accepts decoder_seq_length. Passing the resolved VLM width as
    # seq_length must yield the same pipeline buffers as the old decoder override.
    groups = {"tp_group": Mock(size=lambda: 1), "cp_group": Mock(size=lambda: 1)}
    assert get_tensor_shapes(
        seq_length=model.transformer.max_seq_length_per_sample,
        decoder_seq_length=None,
        micro_batch_size=1,
        config=model.transformer,
        **groups,
    ) == get_tensor_shapes(
        seq_length=64,
        decoder_seq_length=width,
        micro_batch_size=1,
        config=model.transformer,
        **groups,
    )
    monkeypatch.setattr(global_vars, "_GLOBAL_RUN_CONFIG", cfg)
    assert global_vars.build_telemetry_resource_attrs(args)["dl.sequence_length"] == 64
    saved = cfg.to_dict()
    assert "decoder_seq_length" not in saved["model"]
    restored = type(cfg).from_dict(saved)
    restored.finalize()
    assert restored.model.transformer.max_seq_length_per_sample == width
    assert restored.to_dict() == saved


@pytest.mark.parametrize("entry_point", ["gpt", "hybrid", "hybrid_flex"])
def test_get_batch_ignores_the_poisoned_data_arguments(monkeypatch, entry_point):
    """The three scripts' get_batch read the batch flags from the section, the model-parallel
    values from the model section and the sample length from the container; every data-owned
    value on args is poisoned and must not surface in the captured call."""
    if entry_point == "gpt":
        import pretrain_gpt as script
    elif entry_point == "hybrid":
        import pretrain_hybrid as script
    else:
        from megatron.elastification import pretrain_hybrid_flex as script

    dataset = GPTDatasetConfig(
        sequence_length=16,
        random_seed=1,
        inter_document_masking=True,
        create_attention_mask=False,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
    )
    cfg = SimpleNamespace(
        model=_layout(pipeline_model_parallel_size=3, max_seq_length_per_sample=16),
        data=DataConfig(dataset=dataset, dataset_type=MockGPTDataset),
        train=SimpleNamespace(micro_batch_size=5),
    )
    args = SimpleNamespace(  # poisoned: none of these may reach the batch path
        sft=True,
        dataloader_inter_document_masking=False,
        create_attention_mask_in_dataloader=True,
        micro_batch_size=99,
        seq_length=4096,
        sequence_packing_scheduler="dp_balanced",
        dynamic_context_parallel=True,
        context_parallel_size=8,
        pipeline_model_parallel_size=8,
    )
    monkeypatch.setattr(script, "get_args", lambda: args)
    monkeypatch.setattr(script, "get_run_config", lambda: cfg)
    monkeypatch.setattr(script, "is_first_or_last_pipeline_stage", lambda vp_stage: True)
    monkeypatch.setattr(script, "mtp_on_this_rank_func", lambda **kw: False)
    monkeypatch.setattr(script.mpu, "get_tensor_model_parallel_rank", lambda: 1)  # no iterator
    monkeypatch.setattr(script.mpu, "get_tensor_model_parallel_src_rank", lambda: 0)
    monkeypatch.setattr(script.mpu, "get_tensor_model_parallel_group", lambda: None)
    monkeypatch.setattr(script.mpu, "get_pipeline_model_parallel_group", lambda: None)
    monkeypatch.setattr(script.mpu, "is_pipeline_first_stage", lambda: True)
    monkeypatch.setattr(script.mpu, "is_pipeline_last_stage", lambda: True)
    monkeypatch.setattr(script, "get_context_parallel_group", lambda: None)
    tp_calls, cp_calls = [], []
    monkeypatch.setattr(
        script,
        "get_batch_on_this_tp_rank",
        lambda batch, **kw: tp_calls.append(kw) or dict.fromkeys(script.BATCH_KEYS),
    )
    monkeypatch.setattr(script, "flatten_batch_for_packed_sequences", lambda batch: batch)
    if entry_point == "hybrid":
        monkeypatch.setattr(
            script, "get_batches_on_this_cp_rank", lambda batch, **kw: cp_calls.append(kw) or batch
        )
    else:
        monkeypatch.setattr(
            script, "get_batch_on_this_cp_rank", lambda batch, **kw: cp_calls.append(kw) or batch
        )
    script.get_batch(None)
    tp = tp_calls.pop()
    assert tp["has_cu_seqlens"] is True  # inter-document masking from the section, not args.sft
    assert tp["create_attention_mask_in_dataloader"] is False
    assert tp["micro_batch_size"] == 5 and tp["seq_length"] == 16
    assert (tp["cp_size"], tp["pipeline_model_parallel_size"]) == (1, 3)
    assert tp["dynamic_cp"] is False
    assert cp_calls.pop()["use_per_sequence_balancing"] is True  # masking without SFT


@pytest.mark.parametrize("scheduler_at_construction", [True, False])
def test_a_scheduler_selected_after_construction_needs_variable_seq_lengths(
    container_factory, scheduler_at_construction
):
    """Megatron Core derives variable_seq_lengths from the scheduler in ModelParallelConfig's
    __post_init__; a scheduler assigned to a constructed config leaves it False, and the pipeline
    would keep fixed shapes while the batch path packs. The container rejects that pair on a real
    TransformerConfig; the constructor-time selection is the positive control."""
    import torch

    transformer = TransformerConfig(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=8,
        pipeline_model_parallel_size=2,
        pipeline_dtype=torch.float32,
        calculate_per_token_loss=True,
        max_seqlen_per_dp_cp_rank=64,
        sequence_packing_scheduler="dp_balanced" if scheduler_at_construction else None,
    )
    if not scheduler_at_construction:
        transformer.sequence_packing_scheduler = "dp_balanced"  # after construction
    _, cfg = container_factory(["--seq-length", "16", "--mock-data", "--sft"], data=True)
    cfg.model = GPTModelConfig(transformer=transformer, seq_length=64, vocab_size=16)
    if scheduler_at_construction:
        assert transformer.variable_seq_lengths is True
        cfg.validate()
    else:
        assert transformer.variable_seq_lengths is False
        with pytest.raises(ValueError, match="needs model.transformer.variable_seq_lengths=True"):
            cfg.validate()
        transformer.variable_seq_lengths = True  # the pair a Python caller must set together
        cfg.validate()


def test_gpt_forward_step_uses_dataset_length_for_packed_metadata(monkeypatch):
    """Packed metadata must describe the actual samples even with stale args."""
    from unittest.mock import MagicMock

    import torch

    import pretrain_gpt
    from megatron.training.distillation import logits_saver

    args = Namespace(seq_length=32)
    dataset = GPTDatasetConfig(
        sequence_length=16,
        random_seed=1,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
    )
    cfg = SimpleNamespace(data=DataConfig(dataset=dataset), model=SimpleNamespace(seq_length=64))
    cu_seqlens = torch.tensor([[0, 8, 16]], dtype=torch.int32)
    tokens = torch.zeros((1, 16), dtype=torch.long)
    batch = (
        None,
        cu_seqlens,
        cu_seqlens,
        None,
        tokens,
        None,
        tokens,
        torch.tensor(8),
        tokens,
        tokens,
    )
    model = MagicMock(vp_stage=None, pg_collection=None)
    monkeypatch.setattr(pretrain_gpt, "get_args", lambda: args)
    monkeypatch.setattr(pretrain_gpt, "get_run_config", lambda: cfg)
    monkeypatch.setattr(pretrain_gpt, "get_batch", lambda *a, **kw: batch)
    monkeypatch.setattr(pretrain_gpt, "get_timers", MagicMock())
    monkeypatch.setattr(pretrain_gpt, "stimer", MagicMock())
    monkeypatch.setattr(pretrain_gpt, "update_seqlen_stats_from_cu_seqlens", lambda *a: None)
    monkeypatch.setattr(logits_saver, "get_logits_saver", lambda: None)

    pretrain_gpt.forward_step(None, model)

    packed = model.call_args.kwargs["packed_seq_params"]
    assert packed.tokens_per_sample == 16
    assert torch.equal(packed.cu_seqlens_q, cu_seqlens.squeeze(0))
