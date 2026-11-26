# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Optimizer setup must use the same process-group domain as model construction."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.training import training


@pytest.mark.parametrize("source", ["explicit", "model", "wrapped_model", "legacy"])
@pytest.mark.parametrize("use_gloo", [False, True])
def test_optimizer_uses_model_process_groups(monkeypatch, run_config, source, use_gloo):
    """Exercise the setup entrypoint without constructing a distributed model."""
    groups = ProcessGroupCollection()
    model = torch.nn.Linear(2, 2)
    if source in ("model", "wrapped_model"):
        model.pg_collection = groups
    if source == "wrapped_model":
        wrapped = torch.nn.Module()
        wrapped.add_module("module", model)
        model = wrapped
    fallback = Mock(
        return_value=groups,
        side_effect=None if source == "legacy" else AssertionError("unexpected MPU fallback"),
    )
    monkeypatch.setattr(ProcessGroupCollection, "use_mpu_process_groups", fallback)
    args = SimpleNamespace(
        logits_save_dir=None, logits_load_dir=None, dump_param_to_param_group_map=None
    )
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "get_train_state", Mock())
    monkeypatch.setattr(training, "get_timers", Mock())
    monkeypatch.setattr(training, "get_one_logger", lambda: None)
    monkeypatch.setattr(training, "unwrap_model", lambda value: value)
    build = Mock(return_value=[model])
    monkeypatch.setattr(training, "get_model", build)
    run_config.dist.use_gloo_process_groups = use_gloo

    class ReachedOptimizer(Exception):
        pass

    optimizer_factory = Mock(side_effect=ReachedOptimizer)
    monkeypatch.setattr(training, "get_megatron_optimizer", optimizer_factory)
    supplied_groups = groups if source == "explicit" else None
    with pytest.raises(ReachedOptimizer):
        training.setup_model_and_optimizer(
            None, lambda: None, cfg_container=run_config, pg_collection=supplied_groups
        )
    assert build.call_args.kwargs["pg_collection"] is supplied_groups
    assert optimizer_factory.call_args.kwargs["pg_collection"] is groups
    assert optimizer_factory.call_args.kwargs["use_gloo_process_groups"] is use_gloo
    assert fallback.call_count == (1 if source == "legacy" else 0)
