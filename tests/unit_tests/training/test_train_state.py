# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the Bridge-compatible Megatron-LM training state."""

from types import SimpleNamespace

import torch

from megatron.training.train_state import TrainState, load_train_state, save_train_state


def test_train_state_matches_bridge_schema():
    args = SimpleNamespace(
        consumed_train_samples=11,
        skipped_train_samples=12,
        consumed_valid_samples=13,
        do_train=True,
        do_valid=False,
        do_test=True,
    )

    train_state = TrainState.from_args(args, step=14, floating_point_operations_so_far=15)
    state_dict = train_state.state_dict()

    assert set(state_dict) == {
        "step",
        "consumed_train_samples",
        "skipped_train_samples",
        "consumed_valid_samples",
        "floating_point_operations_so_far",
        "do_train",
        "do_valid",
        "do_test",
    }
    assert state_dict["step"].dtype == torch.int64
    assert state_dict["consumed_train_samples"].dtype == torch.int64
    assert state_dict["skipped_train_samples"].dtype == torch.int64
    assert state_dict["consumed_valid_samples"].dtype == torch.int64
    assert state_dict["floating_point_operations_so_far"].dtype == torch.float64
    assert state_dict["do_train"].dtype == torch.bool
    assert state_dict["do_valid"].dtype == torch.bool
    assert state_dict["do_test"].dtype == torch.bool


def test_train_state_save_load_and_apply(tmp_path):
    expected = TrainState(
        step=21,
        consumed_train_samples=22,
        skipped_train_samples=23,
        consumed_valid_samples=24,
        floating_point_operations_so_far=25,
        do_train=True,
        do_valid=True,
        do_test=False,
    )
    filename = tmp_path / "train_state.pt"

    save_train_state(expected, filename)
    actual = load_train_state(filename)
    args = SimpleNamespace()
    actual.apply_to_args(args)

    assert actual == expected
    assert args.consumed_train_samples == 22
    assert args.skipped_train_samples == 23
    assert args.consumed_valid_samples == 24
    assert args.do_train is True
    assert args.do_valid is True
    assert args.do_test is False
