# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import logging
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from megatron.training import initialize


def test_set_random_seed_offsets_pipeline_and_data_parallel_ranks(monkeypatch):
    manual_seed_calls = []
    monkeypatch.setattr(initialize.mpu, "get_pipeline_model_parallel_rank", lambda: 2)
    monkeypatch.setattr(initialize.mpu, "get_data_parallel_rank", lambda: 3)
    monkeypatch.setattr(initialize.torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(initialize.torch, "manual_seed", lambda seed: manual_seed_calls.append(seed))

    initialize._set_random_seed(11, data_parallel_random_init=True)

    assert manual_seed_calls == [241]
    assert random.randint(0, 10) >= 0
    assert np.random.randint(0, 10) >= 0

    with pytest.raises(ValueError, match="positive integer"):
        initialize._set_random_seed(0)


def test_write_args_to_tensorboard_writes_all_namespace_fields(monkeypatch):
    calls = []
    args = SimpleNamespace(iteration=7, alpha=1, beta="two")
    writer = SimpleNamespace(add_text=lambda name, value, global_step: calls.append((name, value, global_step)))
    monkeypatch.setattr(initialize, "get_args", lambda: args)
    monkeypatch.setattr(initialize, "get_tensorboard_writer", lambda: writer)

    initialize.write_args_to_tensorboard()

    assert ("alpha", "1", 7) in calls
    assert ("beta", "two", 7) in calls


def test_init_autoresume_runs_barriers_when_available(monkeypatch):
    calls = []
    autoresume = SimpleNamespace(init=lambda: calls.append("init"))
    monkeypatch.setattr(initialize, "get_adlr_autoresume", lambda: autoresume)
    monkeypatch.setattr(initialize.torch.distributed, "barrier", lambda: calls.append("barrier"))

    initialize._init_autoresume()

    assert calls == ["barrier", "init", "barrier"]


def test_setup_logging_uses_env_and_argument_precedence(monkeypatch):
    root_logger = logging.getLogger()
    original_level = root_logger.level
    try:
        monkeypatch.setenv("MEGATRON_LOGGING_LEVEL", str(logging.WARNING))
        monkeypatch.setattr(initialize, "is_rank0", lambda: True)
        monkeypatch.setattr(initialize, "get_args", lambda: SimpleNamespace(logging_level=None))

        initialize.setup_logging()
        assert root_logger.level == logging.WARNING

        monkeypatch.setattr(initialize, "get_args", lambda: SimpleNamespace(logging_level=logging.ERROR))
        initialize.setup_logging()
        assert root_logger.level == logging.ERROR
    finally:
        root_logger.setLevel(original_level)
