# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest

from megatron.training.distillation import utils
from megatron.training.distillation.cached_logits_loss import (
    LegacyTeacherTarDataset,
    TeacherTarDataset,
)


def _args(**overrides):
    values = dict(
        data_path=None,
        data_args_path=None,
        train_data_path=["1", "/dataset/source"],
        valid_data_path=None,
        test_data_path=None,
        per_split_data_args_path=None,
        train_data_down_sampling_prob_override=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "overrides",
    [
        {"train_data_path": ["1", "/dataset/source", "0.5"]},
        {"train_data_down_sampling_prob_override": 0.5},
    ],
)
def test_cached_logits_rejects_effective_downsampling(overrides):
    with pytest.raises(ValueError, match="does not support training-data downsampling"):
        utils._blend_identifiers(_args(**overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        {"train_data_path": ["1", "/dataset/source", "1.0"]},
        {"train_data_down_sampling_prob_override": 1.0},
        {
            "train_data_path": ["1", "/dataset/source", "0.5"],
            "train_data_down_sampling_prob_override": 1.0,
        },
    ],
)
def test_probability_one_preserves_legacy_cached_logits_identity(overrides):
    assert utils._blend_identifiers(_args(**overrides)) == utils._blend_identifiers(_args())


def test_cached_logits_rejects_downsampling_from_python_configuration(monkeypatch):
    cfg = SimpleNamespace(
        data=SimpleNamespace(
            dataset=SimpleNamespace(blend_per_split=[(["source"], [1.0], [0.5]), None, None])
        ),
        train=SimpleNamespace(train_samples=64),
        rng=SimpleNamespace(seed=1234),
    )
    monkeypatch.setattr(utils, "get_run_config", lambda: cfg)
    monkeypatch.setattr(utils, "get_args", _args)

    with pytest.raises(ValueError, match="does not support training-data downsampling"):
        utils.compute_dataset_hash()


@pytest.mark.parametrize("dataset_class", [TeacherTarDataset, LegacyTeacherTarDataset])
def test_ignore_hash_does_not_bypass_effective_dataset_guard(monkeypatch, dataset_class):
    cfg = SimpleNamespace(
        data=SimpleNamespace(
            dataset=SimpleNamespace(blend_per_split=[(["source"], [1.0], [0.5]), None, None])
        )
    )
    monkeypatch.setattr(utils, "is_run_config_initialized", lambda: True)
    monkeypatch.setattr(utils, "get_run_config", lambda: cfg)

    with pytest.raises(ValueError, match="does not support training-data downsampling"):
        dataset_class("/unused", cp_rank=0, cp_size=1, dp_rank=0, dp_size=1, ignore_hash=True)


def test_effective_probability_one_is_supported(monkeypatch):
    cfg = SimpleNamespace(
        data=SimpleNamespace(
            dataset=SimpleNamespace(blend_per_split=[(["source"], [1.0], [1.0]), None, None])
        )
    )
    monkeypatch.setattr(utils, "is_run_config_initialized", lambda: True)
    monkeypatch.setattr(utils, "get_run_config", lambda: cfg)
    utils.validate_cached_logits_dataset()
