# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Range checks for the intervals that training uses as `iteration % interval`."""

import pytest

from megatron.training.config import CheckpointConfig, LoggerConfig, TrainingConfig


@pytest.mark.parametrize("name", ["log_interval", "tensorboard_log_interval"])
@pytest.mark.parametrize("value", [0, -1])
def test_logger_intervals_must_be_positive(name, value):
    with pytest.raises(ValueError, match=f"{name} must be positive"):
        LoggerConfig(**{name: value}).validate()


@pytest.mark.parametrize(
    "name", ["check_weight_hash_across_dp_replicas_interval", "gpu_sniff_test_interval"]
)
@pytest.mark.parametrize("value", [0, -1])
def test_training_intervals_must_be_positive(name, value):
    with pytest.raises(ValueError, match=f"{name} must be positive"):
        TrainingConfig(**{name: value}).validate()


@pytest.mark.parametrize(
    "name",
    [
        "save_params_interval",
        "save_activations_interval",
        "save_tokens_per_expert_interval",
        "save_wgrads_interval",
        "save_dgrads_interval",
    ],
)
@pytest.mark.parametrize("value", [0, -1])
def test_dump_intervals_must_be_positive(name, value):
    with pytest.raises(ValueError, match=f"{name} must be positive"):
        CheckpointConfig(**{name: value}).validate()


def test_default_intervals_are_valid():
    LoggerConfig().validate()
    TrainingConfig().validate()
    CheckpointConfig().validate()
