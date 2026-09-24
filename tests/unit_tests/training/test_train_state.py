# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the Bridge-compatible Megatron-LM training state."""

from types import SimpleNamespace

import pytest
import torch

from megatron.training.state import TrainState, load_train_state, save_train_state


def test_train_state_matches_bridge_schema():
    args = SimpleNamespace(
        consumed_train_samples=11,
        skipped_train_samples=12,
        consumed_valid_samples=13,
        do_train=True,
        do_valid=False,
        do_test=True,
    )

    train_state = TrainState()
    train_state.update_from_args(args, iteration=14, num_floating_point_operations_so_far=15)
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


@pytest.mark.parametrize("flops", [25, 2**64])
def test_train_state_save_load_and_apply(tmp_path, flops):
    expected = TrainState(
        iteration=21,
        consumed_train_samples=22,
        skipped_train_samples=23,
        consumed_valid_samples=24,
        num_floating_point_operations_so_far=flops,
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
    assert isinstance(actual.num_floating_point_operations_so_far, int)
    assert args.consumed_train_samples == 22
    assert args.skipped_train_samples == 23
    assert args.consumed_valid_samples == 24
    assert args.do_train is True
    assert args.do_valid is True
    assert args.do_test is False


def test_missing_train_state_falls_back_only_when_allowed(tmp_path):
    filename = tmp_path / "train_state.pt"
    assert load_train_state(filename, missing_ok=True) is None
    with pytest.raises(RuntimeError, match="Unable to load train state file"):
        load_train_state(filename)


def test_corrupt_train_state_does_not_fall_back(tmp_path):
    filename = tmp_path / "train_state.pt"
    filename.write_bytes(b"invalid checkpoint")
    with pytest.raises(RuntimeError, match="Unable to load train state file"):
        load_train_state(filename, missing_ok=True)


def test_nonzero_rank_uses_broadcast_presence_decision(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)

    def missing_sidecar(objects, src):
        assert src == 0
        objects[0] = {"state_dict": None}

    monkeypatch.setattr(torch.distributed, "broadcast_object_list", missing_sidecar)
    assert load_train_state(tmp_path / "train_state.pt", missing_ok=True) is None
