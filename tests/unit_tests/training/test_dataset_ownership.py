# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The data section is owned by PretrainConfigContainer: construction from the arguments, the
fields the container writes, the lifecycle and the serialization."""

import copy
import json
import re
from argparse import ArgumentParser, Namespace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.datasets.gpt_fim_dataset import GPTFIMDataset, GPTFIMDatasetConfig
from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.training import argument_utils, arguments, global_vars
from megatron.training.argument_utils import (
    _DATASET_FIELD_SOURCES,
    dataloader_config_from_args,
    gpt_data_config_from_args,
    pretrain_cfg_container_from_args,
    requested_dataset_family,
)
from megatron.training.arguments import parse_args, validate_args
from megatron.training.config import (
    CheckpointConfig,
    DataConfig,
    DataloaderConfig,
    InferenceConfigContainer,
    InferenceSetupConfig,
    PretrainConfigContainer,
)
from megatron.training.config import container as container_module
from megatron.training.config.dataset_config import CONTAINER_WRITTEN_FIELDS
from megatron.training.datasets.sft_dataset import MockSFTDataset, SFTDataset
from megatron.training.datasets.varlen_dataset import MockVarlenDataset

MODEL_ARGV = (
    "--num-layers 2 --hidden-size 128 --num-attention-heads 8 --micro-batch-size 1 "
    "--global-batch-size 8 --seq-length 32 --max-position-embeddings 32 "
    "--tokenizer-type NullTokenizer --vocab-size 2048"
).split()
SOURCE = ["--data-path", "prefix-a", "--split", "9,1,0"]
BLEND = (["prefix-a", "prefix-b"], [0.3, 0.7])
# The layout fields the container writes from the model config; a run without one keeps them.
LAYOUT_FIELDS = (
    "data_parallel_size", "context_parallel_size", "sequence_parallel_size",
    "dynamic_context_parallel", "hybrid_context_parallel",
)  # fmt: skip
# Megatron Core's defaults for the fields the container writes: what an unset field holds.
CORE_DEFAULTS = {
    "data_parallel_size": 1, "context_parallel_size": None, "sequence_parallel_size": 0,
    "dynamic_context_parallel": False, "hybrid_context_parallel": False,
    "multiple_validation_sets": None, "full_validation": None,
}  # fmt: skip


def _written_fields(dataset):
    return {name: getattr(dataset, name) for name in CONTAINER_WRITTEN_FIELDS}


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
    from megatron.training import training

    monkeypatch.setattr(
        training, "get_megatron_optimizer_config", lambda args, **_: (OptimizerConfig(), None)
    )
    monkeypatch.setattr(
        training, "get_megatron_ddp_config", lambda args, **_: DistributedDataParallelConfig()
    )

    def build(argv, *, data=False, **kwargs):
        # finalize() resolves the batch sizes from the registered arguments (upstream #8149):
        # a micro batch and the batch-parallel size validate_args derives from the world size.
        args = _parse(["--micro-batch-size", "1", *argv])
        args.batch_parallel_size = 1
        monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", args)
        if data:
            kwargs["data_cfg"] = gpt_data_config_from_args(args)
        return args, pretrain_cfg_container_from_args(args, **kwargs)

    return build


@pytest.fixture
def warnings_log(monkeypatch):
    log = []
    monkeypatch.setattr(argument_utils, "warn_rank_0", log.append)
    return log


def _layout(**overrides):
    layout = dict(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        dynamic_context_parallel=False,
        gtp_weight_remat_size=1,
        gtp_remat_num_sequence_shards=1,
        cuda_graph_impl="none",
    )
    layout.update(overrides)
    return SimpleNamespace(transformer=SimpleNamespace(**layout), seq_length=4096)


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

    data = gpt_data_config_from_args(_parse(["--seq-length", "32", *extra]))
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


def test_every_data_argument_is_forwarded(files, tmp_path):
    """No entry-point exceptions: the fields the legacy hybrid, hybrid-flex and cache factories left
    at Megatron Core's defaults are forwarded like every other data argument, and the sequences file
    is always read."""
    argv = [
        "--seq-length", "32", *SOURCE, "--allow-ambiguous-pad-tokens",
        "--dataloader-inter-document-masking", "--sft-mock-dataset-config-json", "{}",
        "--varlen-mock-dataset-config-json", "{}",
    ]  # fmt: skip
    dataset = gpt_data_config_from_args(
        _parse([*argv, "--per-dataset-sequences-path", files["sequences"]])
    ).dataset
    assert dataset.allow_ambiguous_pad_tokens is True and dataset.inter_document_masking is True
    assert dataset.sft_mock_dataset_config_json == "{}"
    assert dataset.varlen_mock_dataset_config_json == "{}"
    assert dataset.sequences_per_dataset == {"prefix-a": [10, 2], "prefix-b": [20, 4]}
    with pytest.raises(FileNotFoundError):
        gpt_data_config_from_args(
            _parse([*argv, "--per-dataset-sequences-path", str(tmp_path / "missing.json")])
        )


def test_standalone_config_takes_the_container_written_fields_from_args():
    args = _validated(
        [
            "--mock-data",
            "--seed",
            "99",
            "--full-validation",
            "--sequence-parallel",
            "--tensor-model-parallel-size",
            "1",
        ]
    )
    standalone = gpt_data_config_from_args(args, standalone=True).dataset
    assert standalone.random_seed == 99 and standalone.full_validation is True
    assert standalone.multiple_validation_sets is False
    assert standalone.data_parallel_size == args.data_parallel_size
    assert standalone.context_parallel_size == args.context_parallel_size
    assert standalone.sequence_parallel_size == args.tensor_model_parallel_size * int(
        args.sequence_parallel
    )
    assert standalone.dynamic_context_parallel is False
    assert standalone.hybrid_context_parallel is False
    owned = gpt_data_config_from_args(args).dataset
    assert _written_fields(owned) == CORE_DEFAULTS
    assert owned.random_seed == 99  # the seed is the section's own setting
    assert set(argument_utils._container_written_fields_from_args(args)) == set(
        CONTAINER_WRITTEN_FIELDS
    )


def test_requested_family_precedence(warnings_log):
    assert requested_dataset_family(Namespace()) == "gpt"
    assert requested_dataset_family(Namespace(sft=True, fim_data=True)) == "sft"
    assert requested_dataset_family(Namespace(use_varlen_dataset=True, fim_data=True)) == "varlen"
    assert len(warnings_log) == 2
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


def test_run_config_fixture_is_loader_only(run_config, monkeypatch):
    assert run_config.data.dataset is None
    run_config.validate()
    # finalize() resolves the batch sizes from the registered arguments (upstream #8149).
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", SimpleNamespace(batch_parallel_size=1))
    run_config.train.micro_batch_size = 1
    run_config.finalize()


# The fields the container writes and the cross-section checks.


def test_container_writes_the_owned_fields_at_finalize(container_factory, monkeypatch):
    args, cfg = container_factory(
        ["--seq-length", "64", "--mock-data", "--seed", "99", "--full-validation"], data=True
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
    assert dataset.hybrid_context_parallel is False  # the deprecated alias is never written True
    assert cfg.data.dataset_type is MockGPTDataset
    assert not (set(CONTAINER_WRITTEN_FIELDS) & set(cfg.to_dict()["data"]["dataset"]))
    assert cfg.to_dict()["data"]["dataset"]["random_seed"] == 99
    cfg.validate()  # the written values agree with their owners
    dataset.random_seed = 7
    cfg.validate()  # the dataset seed has no owner to agree with


def test_a_layout_field_at_core_default_carries_no_explicit_value(container_factory, monkeypatch):
    """The layout fields keep Megatron Core's fields and defaults (1, 0, False): a value equal to
    the default is not an explicit value, so it agrees with any layout and is written over; a
    different value must agree with the layout."""
    _, cfg = container_factory(["--seq-length", "64", "--mock-data"], data=True)
    dataset = cfg.data.dataset
    cfg.model = _layout(
        tensor_model_parallel_size=2,
        context_parallel_size=2,
        sequence_parallel=True,
        dynamic_context_parallel=True,
    )
    monkeypatch.setattr(container_module, "safe_get_world_size", lambda: 8)
    dataset.data_parallel_size, dataset.sequence_parallel_size = 1, 0
    dataset.dynamic_context_parallel = False
    cfg.validate()
    dataset.sequence_parallel_size = 1
    with pytest.raises(ValueError, match="data.dataset.sequence_parallel_size=1 disagrees"):
        cfg.validate()
    dataset.sequence_parallel_size = 0
    cfg.finalize()
    assert (dataset.data_parallel_size, dataset.sequence_parallel_size) == (2, 2)
    assert dataset.dynamic_context_parallel is True


def test_the_deprecated_hybrid_alias_is_folded_into_dynamic_context_parallel(
    container_factory, monkeypatch
):
    """A section carrying Megatron Core's deprecated hybrid_context_parallel alias (#7047) counts
    as the canonical flag for the cross-section check and is folded at finalize(): Core warns about
    the alias in validate_without_tokenizer() and does not rewrite it; the container writes
    dynamic_context_parallel from the model config and clears the alias."""
    _, cfg = container_factory(["--seq-length", "64", "--mock-data"], data=True)
    dataset = cfg.data.dataset
    dataset.hybrid_context_parallel = True
    cfg.model = _layout(dynamic_context_parallel=False)
    monkeypatch.setattr(container_module, "safe_get_world_size", lambda: 1)
    with pytest.warns(DeprecationWarning, match="hybrid_context_parallel is deprecated"):
        with pytest.raises(
            ValueError, match="data.dataset.dynamic_context_parallel=True disagrees"
        ):
            cfg.validate()
    cfg.model = _layout(dynamic_context_parallel=True)
    with pytest.warns(DeprecationWarning):
        cfg.validate()
        cfg.finalize()
    assert dataset.dynamic_context_parallel is True and dataset.hybrid_context_parallel is False
    cfg.validate()  # nothing deprecated is left on the section
    assert not (set(CONTAINER_WRITTEN_FIELDS) & set(cfg.to_dict()["data"]["dataset"]))


def test_layout_without_a_model_section_survives_a_reload(
    container_factory, monkeypatch, tokenizer
):
    """What a run without a model section (pretrain_hybrid_flex.py) saves reproduces its layout
    fields: Megatron Core's defaults."""
    _, cfg = container_factory(["--seq-length", "64", "--mock-data"], data=True)
    cfg.model = None
    monkeypatch.setattr(container_module, "safe_get_world_size", lambda: 8)
    cfg.finalize()
    cfg.data.bind_tokenizer(tokenizer)
    layout = tuple(getattr(cfg.data.dataset, name) for name in LAYOUT_FIELDS)
    assert layout == (1, None, 0, False, False)
    cfg.validate()  # Core's defaults are not explicit values
    reloaded = PretrainConfigContainer.from_dict(cfg.to_dict())
    reloaded.finalize()
    reloaded.data.bind_tokenizer(tokenizer)
    assert tuple(getattr(reloaded.data.dataset, name) for name in LAYOUT_FIELDS) == layout


def test_owned_fields_without_a_model_section(container_factory, tokenizer):
    """pretrain_hybrid_flex.py builds its container without a model config: the validation flags
    are written, the layout fields have no owner and keep what they hold (Megatron Core's defaults
    from the factory; a Python caller's value is used by the run and not saved)."""
    _, cfg = container_factory(
        ["--seq-length", "64", "--mock-data", "--full-validation"], data=True
    )
    cfg.model = None
    dataset = cfg.data.dataset
    dataset.data_parallel_size = 4
    cfg.validate()
    cfg.finalize()
    assert dataset.random_seed == 1234  # --seed, from construction
    assert dataset.full_validation is True and dataset.multiple_validation_sets is False
    assert dataset.data_parallel_size == 4 and dataset.context_parallel_size is None
    assert "data_parallel_size" not in cfg.to_dict()["data"]["dataset"]
    cfg.data.bind_tokenizer(tokenizer)  # Core's defaults stay on the rest of the layout
    assert (dataset.sequence_parallel_size, dataset.hybrid_context_parallel) == (0, False)
    cfg.validate()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("full_validation", False, "data.dataset.full_validation=False disagrees with validation.full_validation=True"),
        ("multiple_validation_sets", True, "data.dataset.multiple_validation_sets=True disagrees with validation.multiple_validation_sets=False"),
        ("context_parallel_size", 4, "data.dataset.context_parallel_size=4 disagrees with model.transformer.context_parallel_size=2"),
        ("sequence_parallel_size", 1, "data.dataset.sequence_parallel_size=1 disagrees with the model layout"),
        ("dynamic_context_parallel", True, "data.dataset.dynamic_context_parallel=True disagrees with model.transformer.dynamic_context_parallel=False"),
        # the deprecated alias counts as the canonical flag (Core warns about it)
        ("hybrid_context_parallel", True, "data.dataset.dynamic_context_parallel=True disagrees with model.transformer.dynamic_context_parallel=False"),
    ],
)  # fmt: skip
def test_each_owned_field_is_checked_against_its_owner(container_factory, field, value, message):
    """One explicit check per field: a value given on the dataset config must agree with the
    section that owns it (data_parallel_size needs the world size and is checked at finalize())."""
    _, cfg = container_factory(
        ["--seq-length", "64", "--mock-data", "--full-validation"], data=True
    )
    cfg.model = _layout(tensor_model_parallel_size=2, context_parallel_size=2)
    cfg.validate()
    setattr(cfg.data.dataset, field, value)
    with pytest.raises(ValueError, match=re.escape(message)):
        cfg.validate()


@pytest.mark.parametrize(
    ("world_size", "layout", "configured_cp", "expected"),
    [
        (8, dict(tensor_model_parallel_size=2, pipeline_model_parallel_size=2), 1, 2),
        (8, dict(tensor_model_parallel_size=2, context_parallel_size=2), 2, 2),
        (16, dict(tensor_model_parallel_size=2, context_parallel_size=2, gtp_weight_remat_size=2, gtp_remat_num_sequence_shards=2), 1, 4),
        (16, dict(tensor_model_parallel_size=2, context_parallel_size=2, gtp_weight_remat_size=2), 2, 2),
        (4, dict(), 1, 4),
    ],
)  # fmt: skip
def test_get_data_parallel_size_matches_validate_args(
    container_factory, world_size, layout, configured_cp, expected
):
    """validate_args: world_size // (tp * pp * configured cp * weight remat); the model config holds
    the resolved cp (configured cp times the GTP sequence shards)."""
    _, cfg = container_factory(["--seq-length", "64"])
    cfg.model = _layout(**layout)
    transformer = cfg.model.transformer
    assert (
        transformer.context_parallel_size
        == configured_cp * transformer.gtp_remat_num_sequence_shards
    )
    legacy = world_size // (
        transformer.tensor_model_parallel_size
        * transformer.pipeline_model_parallel_size
        * configured_cp
        * transformer.gtp_weight_remat_size
    )
    assert expected == legacy
    assert cfg.get_data_parallel_size(world_size) == expected
    cfg.model = None
    assert cfg.get_data_parallel_size(world_size) is None


def test_get_data_parallel_size_rejects_an_indivisible_world(container_factory):
    _, cfg = container_factory(["--seq-length", "64"])
    cfg.model = _layout(tensor_model_parallel_size=2, pipeline_model_parallel_size=2)
    with pytest.raises(ValueError, match="divisible"):
        cfg.get_data_parallel_size(6)
    cfg.model = _layout(context_parallel_size=3, gtp_remat_num_sequence_shards=2)
    with pytest.raises(ValueError, match="multiple of the GTP sequence shards"):
        cfg.get_data_parallel_size(12)


def test_model_capacity_must_cover_the_sample_length(container_factory):
    _, cfg = container_factory(["--seq-length", "64", "--mock-data"], data=True)
    cfg.model = _layout()
    cfg.model.seq_length = 32
    with pytest.raises(ValueError, match="model.seq_length \\(32"):
        cfg.validate()
    cfg.model.seq_length = 64
    cfg.validate()
    cfg.model = None
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
    cfg.model = None
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


def test_pre_data_run_config_loads_with_the_default_section(run_config):
    """A run_config.yaml saved before the data section existed loads with the loader-only
    default: the field's dataclass default, no upgrade hook."""
    serialized = run_config.to_dict()
    legacy = copy.deepcopy(serialized)
    legacy.pop("data")
    restored = PretrainConfigContainer.from_dict(legacy)
    assert restored.data == DataConfig()
    assert "data" not in legacy  # the caller's dictionary is not modified
    # A snapshot of the unmerged earlier schema is not migrated.
    old_schema = copy.deepcopy(legacy)
    old_schema["dataset"] = {"_target_": "megatron.training.config.dataset_config.GPTDatasetConfig"}
    with pytest.raises(ValueError, match="extra keys"):
        PretrainConfigContainer.from_dict(old_schema)


def test_inference_container_is_unaffected():
    cfg = InferenceConfigContainer(
        model=None, checkpoint=CheckpointConfig(), inference=InferenceSetupConfig()
    )
    assert not hasattr(cfg, "data")
    serialized = cfg.to_dict()
    assert InferenceConfigContainer.from_dict(serialized).to_dict() == serialized


# The runtime services bind the tokenizer.


def test_runtime_services_bind_the_tokenizer_to_a_finalized_dataset_config(
    container_factory, monkeypatch, tokenizer
):
    _, cfg = container_factory(["--seq-length", "64", "--mock-data"], data=True)
    cfg.model = None
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
    unbound.model = None
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
    cfg.model = None
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
