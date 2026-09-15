# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Catch missing experiment metrics across TensorBoard and W&B."""

import copy

from examples.engram.check_logs import compare_records


def _records() -> dict:
    values = {
        "lm loss": 3.0,
        "learning-rate": 8e-7,
        "grad-norm": 1.0,
        "recipe/update_lr": 0.0,
        "recipe/mtp_weight": 0.3,
        "recipe/phase": 0,
        "recipe/completed_samples_before_update": 0,
        "recipe/completed_tokens_before_update": 0,
        "mtp_1 loss": 3.2,
        "seq_load_balancing_loss": 1.01,
        "batch-size": 64,
        "mem-reserved-bytes": 4096,
        "mem-allocated-bytes": 2048,
        "mem-max-allocated-bytes": 3072,
        "mem-allocated-count": 10,
    }
    return {(key, 1): [value] for key, value in values.items()}


def test_gpu_memory_requires_both_loggers() -> None:
    """A TensorBoard-only GPU memory record cannot satisfy the two-sink contract."""
    tensorboard = _records()
    wandb = copy.deepcopy(tensorboard)
    assert compare_records(tensorboard, wandb, 1, 1, require_experiment_metrics=True)["passed"]
    del wandb[("mem-allocated-bytes", 1)]
    result = compare_records(tensorboard, wandb, 1, 1, require_experiment_metrics=True)
    assert not result["passed"]
    assert any("mem-allocated-bytes@1" in message for message in result["errors"])


def test_engram_requires_matching_table_update_lr() -> None:
    """The Engram arm requires its optimizer LR in both sinks without extra diagnostics."""
    tensorboard = _records()
    wandb = copy.deepcopy(tensorboard)
    result = compare_records(tensorboard, wandb, 1, 1, engram=True)
    assert not result["passed"]
    assert any("recipe/engram_update_lr@1" in message for message in result["errors"])

    tensorboard[("recipe/engram_update_lr", 1)] = [4e-3]
    wandb[("recipe/engram_update_lr", 1)] = [4e-3]
    assert compare_records(tensorboard, wandb, 1, 1, engram=True, require_experiment_metrics=True)[
        "passed"
    ]

    wandb[("recipe/engram_update_lr", 1)] = [8e-4]
    result = compare_records(tensorboard, wandb, 1, 1, engram=True)
    assert not result["passed"]
    assert any("recipe/engram_update_lr@1" in message for message in result["errors"])
