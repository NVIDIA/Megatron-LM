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
from megatron.training.datasets.varlen_dataset import MockVarlenDataset, VarlenDataset

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
        sequence_packing_scheduler=None,
        calculate_per_token_loss=True,
        max_seqlen_per_dp_cp_rank=None,
        gtp_weight_remat_size=1,
        gtp_remat_num_sequence_shards=1,
        cuda_graph_impl="none",
    )
    layout.update(overrides)
    # What Megatron Core derives at construction: a scheduler sets variable_seq_lengths.
    layout.setdefault("variable_seq_lengths", layout["sequence_packing_scheduler"] is not None)
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


def test_dummy_train_step_reads_the_section_and_the_model_seams(monkeypatch):
    """dummy_train_step (the skip-iteration path) runs for any container: the batch flags and the
    data length come from the data section, the model-parallel values from the model section when
    there is one and from the arguments otherwise; the poisoned namespace shows which source won."""
    from megatron.training import training
    from megatron.training.config import DataConfig

    calls = []

    class _Machine:
        def __init__(self):
            self.runs = 0

        def should_run_forward_backward(self, data_iterator):
            self.runs += 1
            return self.runs == 1

    monkeypatch.setattr(training, "get_num_microbatches", lambda: 1)
    monkeypatch.setattr(training, "get_rerun_state_machine", lambda: _Machine())
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
    args = SimpleNamespace(  # poisoned wherever the section or the model section rules
        seq_length=64,
        context_parallel_size=2,
        pipeline_model_parallel_size=4,
        dynamic_context_parallel=False,
        sequence_packing_scheduler=None,
        sft=True,
        dataloader_inter_document_masking=True,
        create_attention_mask_in_dataloader=False,
    )
    monkeypatch.setattr(training, "get_args", lambda: args)
    train = SimpleNamespace(micro_batch_size=3)

    # No model section (pretrain_hybrid_flex.py) and a loader-only section: the topology and the
    # length come from the arguments, the flags are the plain-GPT ones whatever args.sft says.
    monkeypatch.setattr(
        training,
        "get_run_config",
        lambda: SimpleNamespace(model=None, data=DataConfig(), train=train),
    )
    training.dummy_train_step(None)
    kw = calls.pop()
    assert (kw["cp_size"], kw["pipeline_model_parallel_size"], kw["dynamic_cp"]) == (2, 4, False)
    assert kw["seq_length"] == 64 and kw["micro_batch_size"] == 3
    assert kw["has_cu_seqlens"] is False and kw["create_attention_mask_in_dataloader"] is True

    # A dataset section and a model section: the section's flags and length, the model's values.
    dataset = GPTDatasetConfig(
        sequence_length=16,
        random_seed=1,
        inter_document_masking=True,
        create_attention_mask=False,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
    )
    transformer = SimpleNamespace(
        context_parallel_size=1,
        pipeline_model_parallel_size=1,
        dynamic_context_parallel=True,
        sequence_packing_scheduler=None,
    )
    monkeypatch.setattr(
        training,
        "get_run_config",
        lambda: SimpleNamespace(
            model=SimpleNamespace(transformer=transformer),
            data=DataConfig(dataset=dataset, dataset_type=MockGPTDataset),
            train=train,
        ),
    )
    training.dummy_train_step(None)
    kw = calls.pop()
    assert (kw["cp_size"], kw["pipeline_model_parallel_size"], kw["dynamic_cp"]) == (1, 1, True)
    assert kw["seq_length"] == 16
    assert kw["has_cu_seqlens"] is True and kw["create_attention_mask_in_dataloader"] is False


def test_microbatch_loss_reduction_is_keyed_on_the_sft_family(monkeypatch):
    """The reporting reductions of train_step, in _reduce_microbatch_losses: SFT averages over the
    trainable tokens of the global batch (or over the valid per-microbatch means in 'microbatch'
    mode, an all-masked microbatch excluded), the others keep the per-microbatch mean; the family
    is the data section's, --sft-loss-log-mode stays a logging argument."""
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


def test_varlen_datasets_and_the_packing_scheduler(container_factory):
    """A packed varlen dataset needs a scheduler (the CLI selects one; a Python container names
    it); SBHD validation excludes one."""
    from megatron.training.config import DataConfig

    bools = dict(reset_position_ids=False, reset_attention_mask=False, eod_mask_loss=False)
    _, cfg = container_factory(["--seq-length", "64"])
    plain = GPTDatasetConfig(sequence_length=64, random_seed=1, **bools)
    cfg.data = DataConfig(
        dataset=plain, dataset_type=MockVarlenDataset, dataloader=cfg.data.dataloader
    )
    cfg.model = _layout()
    with pytest.raises(ValueError, match="needs model.transformer.sequence_packing_scheduler"):
        cfg.validate()
    cfg.model = _layout(sequence_packing_scheduler="dp_balanced")
    cfg.validate()
    sbhd = GPTDatasetConfig(
        sequence_length=64,
        random_seed=1,
        blend=(["prefix-a"], None),
        split="9,1,0",
        varlen_sbhd_validation=True,
        **bools,
    )
    cfg.data = DataConfig(dataset=sbhd, dataset_type=VarlenDataset, dataloader=cfg.data.dataloader)
    with pytest.raises(ValueError, match="does not use a packing scheduler"):
        cfg.validate()
    cfg.model = _layout()
    cfg.validate()
    cfg.model = None  # no model section: the model-dependent rules do not apply
    cfg.validate()


def test_sample_length_is_a_multiple_of_twice_the_cp_size(container_factory):
    _, cfg = container_factory(["--seq-length", "60", "--mock-data"], data=True)
    cfg.model = _layout(context_parallel_size=4)
    with pytest.raises(
        ValueError, match="multiple of 2 \\* model.transformer.context_parallel_size"
    ):
        cfg.validate()
    cfg.model = _layout(context_parallel_size=2)  # 60 % 4 == 0
    cfg.validate()


def test_packed_buffer_budget_is_checked_at_finalize(container_factory, monkeypatch):
    """cp ranks (times the data-parallel replicas under default_dynamic_cp) times the per-rank
    budget must cover one sample; it needs the world size, so it runs at finalize(), before any
    derived field is written."""
    monkeypatch.setattr(container_module, "safe_get_world_size", lambda: 4)
    _, cfg = container_factory(["--seq-length", "64", "--mock-data"], data=True)
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
    _, cfg = container_factory(["--seq-length", "64", "--mock-data"], data=True)
    cfg.model = _layout(  # 2 cp ranks * 2 replicas * 16 = 64 covers the sample
        context_parallel_size=2,
        dynamic_context_parallel=True,
        sequence_packing_scheduler="default_dynamic_cp",
        max_seqlen_per_dp_cp_rank=16,
    )
    cfg.finalize()


def test_the_loop_reads_the_data_length_from_the_section(container_factory, monkeypatch):
    """Three different numbers: the dataset's sequence_length (what the loop uses), a stale
    args.seq_length (never read when the section has a dataset) and the model capacity. The
    calculator and the OTel resource attributes receive the dataset's; a loader-only section
    falls back to the namespace."""
    from megatron.training.config.compat import data_sequence_length

    args, cfg = container_factory(["--seq-length", "16", "--mock-data"], data=True)
    args.rank = 0  # the calculator takes the rank from the namespace
    cfg.model = _layout()
    cfg.model.seq_length = 128
    args.seq_length = 64  # stale, must not be read
    assert data_sequence_length(cfg, args) == 16
    monkeypatch.setattr(global_vars, "_GLOBAL_RUN_CONFIG", cfg)
    assert global_vars.build_telemetry_resource_attrs(args)["dl.sequence_length"] == 16
    calls = []
    monkeypatch.setattr(
        global_vars, "init_num_microbatches_calculator", lambda **kw: calls.append(kw)
    )
    monkeypatch.setattr(global_vars, "get_batch_parallel_size", lambda ns: 1)
    for name in (
        "_set_tensorboard_writer",
        "_set_one_logger",
        "_set_adlr_autoresume",
        "_set_timers",
        "_set_energy_monitor",
    ):
        monkeypatch.setattr(global_vars, name, lambda ns: None)
    monkeypatch.setattr(global_vars, "_set_train_state", lambda: None)
    global_vars.initialize_training_runtime_services(args)
    assert calls.pop()["seq_length"] == 16
    del args.seq_length  # a GPT run never needs it
    assert data_sequence_length(cfg, args) == 16
    global_vars.initialize_training_runtime_services(args)
    assert calls.pop()["seq_length"] == 16
    # A loader-only section: the namespace is the source (the callback entry points).
    args, loader_only = container_factory(["--seq-length", "64"])
    args.rank = 0
    monkeypatch.setattr(global_vars, "_GLOBAL_RUN_CONFIG", loader_only)
    assert data_sequence_length(loader_only, args) == 64
    assert global_vars.build_telemetry_resource_attrs(args)["dl.sequence_length"] == 64
    global_vars.initialize_training_runtime_services(args)
    assert calls.pop()["seq_length"] == 64
    args.seq_length = None
    assert "dl.sequence_length" not in global_vars.build_telemetry_resource_attrs(args)


@pytest.mark.parametrize("entry_point", ["gpt", "hybrid", "hybrid_flex"])
def test_get_batch_ignores_the_poisoned_data_arguments(monkeypatch, entry_point):
    """The three scripts' get_batch read the batch flags and the data length from the section and
    the model-parallel values from the model section (hybrid-flex: from the transformer config the
    arguments describe); every data-owned value on args is poisoned and must not surface in the
    captured call."""
    from megatron.training import arguments
    from megatron.training.config import DataConfig

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
    transformer = SimpleNamespace(
        sequence_packing_scheduler=None,
        dynamic_context_parallel=False,
        context_parallel_size=1,
        pipeline_model_parallel_size=3,
        pipeline_model_parallel_layout=None,
        mtp_num_layers=None,
        virtual_pipeline_model_parallel_size=None,
        linear_cp_layout="zigzag",
        attention_cp_layout="zigzag",
        sequence_parallel=False,
        tensor_model_parallel_size=1,
    )
    model = None if entry_point == "hybrid_flex" else SimpleNamespace(transformer=transformer)
    cfg = SimpleNamespace(
        model=model,
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
    # hybrid-flex has no model section: its transformer config is built from the arguments.
    monkeypatch.setattr(arguments, "core_transformer_config_from_args", lambda ns: transformer)
    monkeypatch.setattr(script, "is_first_or_last_pipeline_stage", lambda vp_stage: True)
    monkeypatch.setattr(script, "mtp_on_this_rank_func", lambda **kw: False)
    monkeypatch.setattr(script.mpu, "get_tensor_model_parallel_rank", lambda: 1)  # no iterator
    monkeypatch.setattr(script.mpu, "get_tensor_model_parallel_src_rank", lambda: 0)
    monkeypatch.setattr(script.mpu, "get_tensor_model_parallel_group", lambda: None)
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

    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.training.models.gpt import GPTModelConfig

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
    _, cfg = container_factory(["--seq-length", "16", "--mock-data"], data=True)
    cfg.data.dataset_type = MockVarlenDataset
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
