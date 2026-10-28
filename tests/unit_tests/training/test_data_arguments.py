# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The data arguments are unchanged, and the data section accepts what validate_args accepts.

``legacy_data_arguments.json`` is a snapshot of the data-related argparse actions of the parser at
Megatron-LM commit 438ef0025: option strings, dest, default, nargs, type, choices, action class,
required and const.
"""

import json
from argparse import ArgumentParser
from pathlib import Path
from unittest.mock import patch

import pytest

from megatron.core.datasets.gpt_fim_dataset import GPTFIMDatasetConfig
from megatron.training import argument_utils, arguments
from megatron.training.argument_utils import gpt_data_config_from_args
from megatron.training.arguments import parse_args, validate_args

LEGACY = json.loads((Path(__file__).parent / "legacy_data_arguments.json").read_text())

MODEL_ARGV = (
    "--num-layers 2 --hidden-size 128 --num-attention-heads 8 --micro-batch-size 1 "
    "--global-batch-size 8 --seq-length 32 --max-position-embeddings 32"
).split()


def _full_parser():
    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    return parser


def _actions_by_dest(parser):
    actions = {}
    for action in parser._actions:
        if action.dest == "help":
            continue
        if action.dest in LEGACY:
            assert action.dest not in actions, f"{action.dest} is registered twice"
        actions.setdefault(action.dest, action)
    return actions


def _describe(action):
    return dict(
        option_strings=list(action.option_strings),
        default=action.default,
        nargs=action.nargs,
        type=None if action.type is None else getattr(action.type, "__name__", repr(action.type)),
        choices=None if action.choices is None else list(action.choices),
        action=type(action).__name__,
        required=action.required,
        const=action.const,
    )


def test_every_legacy_data_argument_is_unchanged():
    actions = _actions_by_dest(_full_parser())
    for dest, legacy in LEGACY.items():
        assert dest in actions, dest
        assert _describe(actions[dest]) == legacy, dest


def test_full_parser_has_no_duplicate_option_strings():
    seen = {}
    for action in _full_parser()._actions:
        for option in action.option_strings:
            assert option not in seen, option
            seen[option] = action.dest


def _validated(extra):
    argv = ["test_data_arguments.py", *MODEL_ARGV, *extra]
    with patch("sys.argv", argv):
        return validate_args(parse_args())


@pytest.fixture
def warnings_log(monkeypatch):
    log = []
    monkeypatch.setattr(argument_utils, "warn_rank_0", log.append)
    return log


def test_validate_args_keeps_the_legacy_data_normalizations(warnings_log):
    args = _validated(["--sft", "--mock-data", "--phase-transition-iterations", "20,10"])
    assert args.sft is True and args.phase_transition_iterations == [10, 20]
    assert json.loads(args.sft_mock_dataset_config_json)["max_seq_len"] == 32
    assert args.dataloader_type == "single"
    data = gpt_data_config_from_args(args)
    data.validate()
    # A section built from validated arguments carries the defaults validate_args materialized
    # (here the SFT mock distribution) explicitly, and serializes them as declared values.
    dataset = data.dataset
    assert dataset.sft_mock_dataset_config_json == args.sft_mock_dataset_config_json
    section = data.to_cfg_dict()["dataset"]
    assert section["sft_mock_dataset_config_json"] == args.sft_mock_dataset_config_json
    # The arguments carry no split for a mock run and neither does the section: Megatron Core
    # derives the even split (split_matrix) from the absence of a source.
    assert args.split is None and dataset.split is None and dataset.split_matrix is not None
    # An explicit --split with mock data is kept on the section but, as before, the mock dataset
    # uses the even split.
    explicit = gpt_data_config_from_args(_validated(["--mock-data", "--split", "949,50,1"]))
    assert explicit.dataset.split == "949,50,1"
    assert explicit.dataset.split_matrix == dataset.split_matrix
    assert data.dataloader.dataloader_type == "single"
    assert not hasattr(dataset, "phase_transition_iterations")
    varlen = _validated(["--use-varlen-dataset", "--mock-data", "--calculate-per-token-loss"])
    assert (
        varlen.sequence_packing_scheduler == "dp_balanced" and varlen.variable_seq_lengths is True
    )
    assert warnings_log == []


def test_legacy_mixed_fim_flags_still_validate(warnings_log):
    for extra in (
        ["--sft", "--fim-data"],
        ["--use-varlen-dataset", "--fim-data", "--calculate-per-token-loss"],
    ):
        args = _validated([*extra, "--data-path", "a"])
        assert args.fim_data is True
        data = gpt_data_config_from_args(args)
        data.validate()
        assert type(data.dataset) is not GPTFIMDatasetConfig  # the packed family wins, as before
    assert len(warnings_log) == 2 and all("wins" in message for message in warnings_log)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--sft", "--use-varlen-dataset"], "mutually exclusive"),
        (["--fim-data", "--mock-data"], "Mock dataset is not supported"),
        (["--mock-data", "--data-path", "a"], "single data source"),
    ],
)
def test_validate_args_rejects_contradictory_data_flags(extra, message):
    with pytest.raises((ValueError, AssertionError), match=message):
        _validated(extra)


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["--mock-data"],
        ["--data-path", "0.3", "a", "0.7", "b", "--split", "9,1,0"],
        ["--data-path", "a"],
        ["--train-data-path", "a", "--valid-data-path", "b", "--test-data-path", "c"],
        ["--sft", "--mock-data", "--sft-mock-dataset-config-json", "{}"],
        ["--use-varlen-dataset", "--mock-data", "--calculate-per-token-loss"],
        ["--use-varlen-dataset", "--data-path", "a", "--varlen-sbhd-validation"],
        ["--fim-data", "--data-path", "a", "--fim-rate", "0.9"],
        ["--train-data-path", "a", "--data-cache-path", "/c", "--dataloader-fast-cache-load"],
        ["--data-path", "a", "--dataloader-inter-document-masking", "--reset-position-ids"],
        ["--data-path", "a", "--phase-transition-iterations", "5,1", "--dataloader-type", "cyclic"],
        ["--data-path", "a", "--no-data-sharding", "--no-mmap-bin-files", "--num-workers", "0"],
        ["--mock-data", "--no-create-attention-mask-in-dataloader", "--eod-mask-loss"],
        ["--data-path", "a", "--varlen-sbhd-validation"],
    ],
)
def test_data_section_accepts_everything_validate_args_accepts(warnings_log, extra):
    args = _validated(extra)
    data = gpt_data_config_from_args(args)
    data.validate()
    data.finalize()
    assert isinstance(data.dataset, GPTFIMDatasetConfig) is bool(args.fim_data)
    assert data.dataset.create_attention_mask == args.create_attention_mask_in_dataloader
    assert data.dataloader.dataloader_type == args.dataloader_type
    assert data.dataset_type is not None


def test_loader_flags_feed_the_section():
    """The two loader flags of the data section: store_false on pin_memory / persistent_workers,
    read by dataloader_config_from_args like every other dest; the defaults are today's built-in
    loader values, and zero workers keeps persistence off whatever the flag says."""
    from megatron.training.argument_utils import dataloader_config_from_args

    args = _validated([])
    assert args.pin_memory is True and args.persistent_workers is True
    loader = dataloader_config_from_args(args)
    assert loader.pin_memory is True and loader.persistent_workers is True
    loader = dataloader_config_from_args(
        _validated(["--no-pin-memory", "--no-persistent-workers", "--num-workers", "2"])
    )
    assert loader.pin_memory is False and loader.persistent_workers is False
    loader = dataloader_config_from_args(_validated(["--num-workers", "0"]))
    loader.finalize()
    assert loader.persistent_workers is False
