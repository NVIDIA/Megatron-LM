# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The checkpointing arguments are unchanged while CheckpointConfig becomes their owner.

``legacy_checkpoint_arguments.json`` is a snapshot of the argparse actions of the "checkpointing"
group at Megatron-LM commit ed9c2c8c1: option strings, dest, default, nargs, type, choices, action
class, required and const.
"""

import json
from argparse import ArgumentParser
from dataclasses import fields
from pathlib import Path

import pytest

from megatron.training import arguments
from megatron.training.config import CheckpointConfig

LEGACY = json.loads((Path(__file__).parent / "legacy_checkpoint_arguments.json").read_text())

PROMOTED = ("dist_ckpt_workers", "override_ckpt_iteration", "ckpt_drop_redundant_extra_state")
DEPRECATED = (
    "use_dist_ckpt_deprecated",
    "dist_ckpt_format_deprecated",
    "ckpt_fully_parallel_save_deprecated",
)


def _full_parser():
    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    return parser


def _checkpointing_actions(parser):
    (group,) = [group for group in parser._action_groups if group.title == "checkpointing"]
    actions = {}
    for action in group._group_actions:
        assert action.dest not in actions, f"{action.dest} is registered twice"
        actions[action.dest] = action
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


def test_every_legacy_checkpoint_argument_is_unchanged():
    actions = _checkpointing_actions(_full_parser())
    assert set(actions) == set(LEGACY)
    for dest, legacy in LEGACY.items():
        assert _describe(actions[dest]) == legacy, dest


def test_full_parser_has_no_duplicate_option_strings():
    seen = {}
    for action in _full_parser()._actions:
        for option in action.option_strings:
            assert option not in seen, option
            seen[option] = action.dest


def test_promoted_flags_are_fields_with_their_legacy_defaults():
    config_fields = {item.name: item for item in fields(CheckpointConfig)}
    for dest in PROMOTED:
        assert config_fields[dest].default == LEGACY[dest]["default"], dest
    for dest in DEPRECATED:
        assert dest not in config_fields, dest
    assert "use_dist_ckpt" not in config_fields


def test_use_dist_ckpt_is_derived_from_the_format():
    assert CheckpointConfig().use_dist_ckpt is True
    assert CheckpointConfig(ckpt_format="torch").use_dist_ckpt is False
    for ckpt_format in ("torch_dist", "torch_dcp", "fsdp_dtensor"):
        assert CheckpointConfig(ckpt_format=ckpt_format).use_dist_ckpt is True
    with pytest.raises(AttributeError):
        CheckpointConfig().use_dist_ckpt = False
