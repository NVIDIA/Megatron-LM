# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Recipe overrides are built from runtime context and reach the Core factory."""

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core.optimizer import _get_param_groups, get_standard_config_overrides
from megatron.core.optimizer.optimizer_config import ParamKey
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.training import training
from megatron.training.config import (
    OptimizerConfigOverrideContext,
    OptimizerConfigOverrideProvider,
    PretrainConfigContainer,
    target_allowlist,
)
from tests.unit_tests.test_utilities import Utils


@dataclass
class WeightLRProvider(OptimizerConfigOverrideProvider):
    """Example application configuration that extends the standard overrides."""

    multiplier: float = 4.0

    def build_config_overrides(self, context):
        overrides = get_standard_config_overrides(context.optimizer_config) or {}
        overrides[ParamKey(name="weight")] = {
            "max_lr": context.optimizer_config.lr * self.multiplier
        }
        return overrides


@pytest.mark.parametrize("provided", ["default", "none", "empty", "custom"])
@pytest.mark.parametrize("use_mup", [False, True])
def test_setup_uses_provider_and_keeps_mup_precedence(monkeypatch, run_config, provided, use_mup):
    model = [torch.nn.Linear(2, 2)]
    model[0].pg_collection = ProcessGroupCollection()
    key = ParamKey(name="weight")
    custom = {key: {"max_lr": 0.2}}
    standard = {ParamKey(name="bias"): {"wd_mult": 0.0}}
    mup = {key: {"max_lr": 0.4}}
    value = {"none": None, "empty": {}, "custom": custom}.get(provided)
    provider = Mock(spec=OptimizerConfigOverrideProvider)
    provider.build_config_overrides.return_value = value
    run_config.optimizer_config_override_provider = None if provided == "default" else provider
    run_config.model.transformer.use_mup = use_mup
    monkeypatch.setattr(
        training,
        "get_args",
        lambda: SimpleNamespace(
            logits_save_dir=None, logits_load_dir=None, dump_param_to_param_group_map=None
        ),
    )
    monkeypatch.setattr(training, "get_train_state", Mock())
    monkeypatch.setattr(training, "get_timers", Mock())
    monkeypatch.setattr(training, "get_one_logger", lambda: None)
    monkeypatch.setattr(training, "get_model", Mock(return_value=model))
    monkeypatch.setattr(
        training, "get_model_config", lambda model: SimpleNamespace(mup_width_mult=4)
    )
    standard_factory = Mock(return_value=standard)
    monkeypatch.setattr(training, "get_standard_config_overrides", standard_factory)
    monkeypatch.setattr(training, "get_mup_config_overrides", lambda **kwargs: mup)

    class ReachedOptimizer(Exception):
        pass

    factory = Mock(side_effect=ReachedOptimizer)
    monkeypatch.setattr(training, "get_megatron_optimizer", factory)
    with pytest.raises(ReachedOptimizer):
        training.setup_model_and_optimizer(None, lambda: None, cfg_container=run_config)
    expected = standard if value is None else value
    assert factory.call_args.kwargs["config_overrides"] == {**expected, **(mup if use_mup else {})}
    assert standard_factory.call_count == (1 if value is None else 0)
    if provided != "default":
        context = provider.build_config_overrides.call_args.args[0]
        assert context.optimizer_config is run_config.optimizer
        assert context.scheduler_config is run_config.scheduler
        assert context.model is model
    assert custom[key]["max_lr"] == 0.2


def test_recipe_provider_round_trip_and_real_parameter_groups(run_config):
    Utils.initialize_model_parallel(1, 1)
    provider = WeightLRProvider(multiplier=8.0)
    run_config.optimizer_config_override_provider = provider
    target = f"{WeightLRProvider.__module__}.{WeightLRProvider.__qualname__}"
    target_allowlist.add_exact(target)
    try:
        restored = PretrainConfigContainer.from_dict(run_config.to_dict())
        assert restored.optimizer_config_override_provider == provider
        model = [torch.nn.Linear(2, 2)]
        restored.optimizer.lr = 0.001
        context = OptimizerConfigOverrideContext(restored.optimizer, restored.scheduler, model)
        overrides = restored.optimizer_config_override_provider.build_config_overrides(context)
        groups = _get_param_groups(model, restored.optimizer, overrides)
        weight_group = next(g for g in groups if any(p is model[0].weight for p in g["params"]))
        bias_group = next(g for g in groups if any(p is model[0].bias for p in g["params"]))
        assert weight_group["max_lr"] == 0.008
        assert bias_group["wd_mult"] == 0.0
    finally:
        target_allowlist.remove_exact(target)
        Utils.destroy_model_parallel()
