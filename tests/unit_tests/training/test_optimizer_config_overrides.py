# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Recipe mappings/callables compose standard and model-owned optimizer policy."""

from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core.optimizer import OptimizerConfig, _get_param_groups, merge_config_overrides
from megatron.core.optimizer.optimizer import FP32Optimizer, get_param_group_identifier_tuple
from megatron.core.optimizer.optimizer_config import ParamKey
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from megatron.training import training
from megatron.training.config import PretrainConfigContainer, target_allowlist
from megatron.training.config.optimizer_config import build_optimizer_config_overrides
from tests.unit_tests.test_utilities import Utils


@dataclass
class WeightLRHook:
    """Serializable callable with no provider superclass."""

    multiplier: float = 4.0

    def __call__(self, *, optimizer_config, model):
        assert model[0].weight.requires_grad
        return {ParamKey(name="weight"): {"max_lr": optimizer_config.lr * self.multiplier}}


def _model(*, use_mup=False):
    model = torch.nn.Linear(4, 4)
    model.config = TransformerConfig(num_layers=1, hidden_size=4, num_attention_heads=1)
    model.config.use_mup = use_mup
    model.config.mup_width_mult = 2.0
    return [model]


@pytest.fixture
def distributed_groups():
    Utils.initialize_model_parallel(1, 1)
    yield
    Utils.destroy_model_parallel()


@pytest.mark.parametrize("mode", ["extend", "replace"])
@pytest.mark.parametrize("custom", [None, {}, WeightLRHook(8.0)])
def test_resolved_policy_reaches_actual_parameter_groups(distributed_groups, mode, custom):
    config = OptimizerConfig(lr=0.001, min_lr=0.0)
    model = _model()
    overrides = build_optimizer_config_overrides(config, model, custom, mode=mode)
    groups = _get_param_groups(model, config, overrides)
    by_param = {id(param): group for group in groups for param in group["params"]}
    assert by_param[id(model[0].weight)]["max_lr"] == (0.008 if custom else 0.001)
    assert by_param[id(model[0].bias)]["wd_mult"] == (0.0 if mode == "extend" else 1.0)


def test_identical_selectors_merge_fields_without_mutation_or_silent_precedence():
    key = ParamKey(name="weight")
    first, second = {key: {"max_lr": 0.1}}, {key: {"wd_mult": 0.0}}
    original = deepcopy(first)
    assert merge_config_overrides(first, second)[key] == {"max_lr": 0.1, "wd_mult": 0.0}
    assert first == original and second[key] == {"wd_mult": 0.0}
    assert merge_config_overrides(first, first) == first
    with pytest.raises(ValueError, match="Conflicting optimizer overrides"):
        merge_config_overrides(first, {key: {"max_lr": 0.2}})


def test_mup_custom_conflicts_are_detected_after_selector_matching(distributed_groups):
    config = OptimizerConfig(lr=0.001, min_lr=0.0)
    model = _model(use_mup=True)
    with pytest.raises(ValueError, match="Conflicting overrides for max_lr"):
        build_optimizer_config_overrides(config, model, WeightLRHook(8.0))
    # Replacing standard WD policy does not silently disable the model's muP scaling.
    overrides = build_optimizer_config_overrides(config, model, {}, mode="replace")
    groups = _get_param_groups(model, config, overrides)
    weight_group = next(g for g in groups if any(p is model[0].weight for p in g["params"]))
    assert weight_group["max_lr"] == 0.0005


def test_frozen_and_rank_local_parameters_preserve_group_alignment(distributed_groups):
    model = _model()
    model[0].bias.requires_grad_(False)
    # Only rank zero has this custom trainable parameter; both ranks still align groups.
    if Utils.rank == 0:
        model[0].extra = torch.nn.Parameter(torch.ones(4, 4))
    config = OptimizerConfig(lr=0.001)
    custom = {ParamKey(name="extra"): {"max_lr": 0.007}}
    groups = _get_param_groups(
        model, config, build_optimizer_config_overrides(config, model, custom)
    )
    assert not any(p is model[0].bias for g in groups for p in g["params"])
    extra = next(g for g in groups if g["max_lr"] == 0.007)
    assert len(extra["params"]) == (1 if Utils.rank == 0 else 0)


@pytest.mark.parametrize("yaml_round_trip", [False, True])
@pytest.mark.parametrize("static", [False, True])
def test_callable_configuration_round_trip(run_config, tmp_path, yaml_round_trip, static):
    hook = (
        {ParamKey(name=("weight", "*.weight")): {"max_lr": 0.008}}
        if static
        else WeightLRHook(multiplier=8.0)
    )
    run_config.optimizer_config_overrides = hook
    run_config.optimizer.lr = 0.001
    target = f"{WeightLRHook.__module__}.{WeightLRHook.__qualname__}"
    target_allowlist.add_exact(target)
    try:
        if yaml_round_trip:
            path = tmp_path / "recipe.yaml"
            run_config.to_yaml(str(path))
            restored = PretrainConfigContainer.from_yaml(str(path))
        else:
            restored = PretrainConfigContainer.from_dict(run_config.to_dict())
        resolved = build_optimizer_config_overrides(
            restored.optimizer, _model(), restored.optimizer_config_overrides
        )
        key = ParamKey(name=("weight", "*.weight")) if static else ParamKey(name="weight")
        assert resolved[key]["max_lr"] == 0.008
    finally:
        target_allowlist.remove_exact(target)


def test_setup_calls_hook_with_the_built_model(monkeypatch, run_config):
    model = _model()
    model[0].pg_collection = ProcessGroupCollection()
    hook = Mock(return_value={ParamKey(name="weight"): {"max_lr": 0.008}})
    run_config.optimizer_config_overrides = hook
    run_config.optimizer.lr = 0.001
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

    class ReachedOptimizer(Exception):
        pass

    factory = Mock(side_effect=ReachedOptimizer)
    monkeypatch.setattr(training, "get_megatron_optimizer", factory)
    with pytest.raises(ReachedOptimizer):
        training.setup_model_and_optimizer(None, lambda: None, cfg_container=run_config)
    hook.assert_called_once_with(optimizer_config=run_config.optimizer, model=model)
    overrides = factory.call_args.kwargs["config_overrides"]
    assert overrides[ParamKey(name="weight")]["max_lr"] == 0.008
    assert any(value.get("wd_mult") == 0.0 for value in overrides.values())


def test_invalid_hook_result_and_mode_fail_before_optimizer_construction():
    config, model = OptimizerConfig(lr=0.001), _model()
    with pytest.raises(TypeError, match="must return"):
        build_optimizer_config_overrides(config, model, lambda **kwargs: None)
    with pytest.raises(ValueError, match="Unknown optimizer override mode"):
        build_optimizer_config_overrides(config, model, {}, mode="last_wins")


def test_mimo_does_not_silently_discard_custom_policy():
    from megatron.core.models.mimo.model.base import MimoModel

    model = MimoModel.__new__(MimoModel)
    torch.nn.Module.__init__(model)
    with pytest.raises(NotImplementedError, match="per module"):
        build_optimizer_config_overrides(
            OptimizerConfig(), [model], {ParamKey(name="weight"): {"max_lr": 0.1}}
        )


def test_plain_torch_module_needs_no_model_specific_configuration():
    model = [torch.nn.Linear(4, 4)]
    overrides = build_optimizer_config_overrides(OptimizerConfig(lr=0.001), model, WeightLRHook())
    assert overrides[ParamKey(name="weight")]["max_lr"] == 0.004
    assert any(value.get("wd_mult") == 0.0 for value in overrides.values())


@pytest.mark.parametrize(
    "contribution", [{}, {"max_lr": 0.001}, {"wd_mult": 1.0}, {"max_lr": 0.008}]
)
def test_redundant_overrides_keep_unique_parameter_state_on_resume(
    distributed_groups, contribution
):
    model = torch.nn.Module()
    model.a = torch.nn.Parameter(torch.ones(4, 4))
    model.b = torch.nn.Parameter(torch.full((4, 4), 2.0))
    config = OptimizerConfig(lr=0.001, min_lr=0.0, clip_grad=0.0)
    policy = build_optimizer_config_overrides(config, [model], {ParamKey(name="a"): contribution})
    groups = _get_param_groups([model], config, policy)
    assert len(groups) == len({get_param_group_identifier_tuple(group) for group in groups})
    assert len(groups) == (2 if contribution.get("max_lr") == 0.008 else 1)
    inner = torch.optim.AdamW(groups, lr=config.lr)
    model.a.grad = torch.ones_like(model.a)
    model.b.grad = torch.full_like(model.b, 2.0)
    inner.step()
    saved = deepcopy(inner.state_dict())
    expected = {param: deepcopy(inner.state[param]) for param in model.parameters()}
    wrapper = FP32Optimizer.__new__(FP32Optimizer)
    wrapper.optimizer = inner
    wrapper.load_state_dict(saved)
    members = [param for group in inner.param_groups for param in group["params"]]
    assert len(members) == len({id(param) for param in members}) == 2
    for param in model.parameters():
        for key, value in expected[param].items():
            torch.testing.assert_close(inner.state[param][key], value)
