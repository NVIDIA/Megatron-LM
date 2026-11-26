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
from megatron.core.optimizer.optimizer_config import ParamKey, ParamPredicate
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from megatron.training import training
from megatron.training.config import PretrainConfigContainer, target_allowlist
from megatron.training.config.optimizer_config import (
    LoRAPlusOptimizerOverrides,
    build_optimizer_config_overrides,
    get_lora_plus_lr_overrides,
    validate_lora_plus_warmup,
)
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


@pytest.mark.parametrize("reject_nonzero_warmup", [False, True])
def test_setup_calls_hook_with_the_built_model(monkeypatch, run_config, reject_nonzero_warmup):
    model = _model()
    model[0].pg_collection = ProcessGroupCollection()
    hook = Mock(return_value={ParamKey(name="weight"): {"max_lr": 0.008}})
    run_config.optimizer_config_overrides = hook
    run_config.optimizer.lr = 0.001
    if reject_nonzero_warmup:
        hook.return_value = {ParamKey(name="*.linear_out.weight"): {"max_lr": 0.016}}
        run_config.scheduler.lr_warmup_init = 1e-5
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
    if reject_nonzero_warmup:
        with pytest.raises(ValueError, match="lr_warmup_init=0"):
            training.setup_model_and_optimizer(None, lambda: None, cfg_container=run_config)
        factory.assert_not_called()
        return
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


@pytest.mark.parametrize("ratio", [1.0, 16.0, 0.25])
def test_lora_plus_groups_schedule_and_resume(distributed_groups, ratio):
    config = OptimizerConfig(lr=1e-3, min_lr=1e-4, weight_decay=0.1)

    def build():
        model = _model()[0]
        model.adapter = torch.nn.Module()
        model.adapter.linear_in = torch.nn.Linear(4, 2, bias=False)
        model.adapter.linear_out = torch.nn.Linear(2, 4, bias=False)
        policy = build_optimizer_config_overrides(
            config, [model], LoRAPlusOptimizerOverrides(ratio)
        )
        groups = _get_param_groups([model], config, policy)
        # A retains the base group, including other trainable weights. No duplicate identities.
        identities = [get_param_group_identifier_tuple(group) for group in groups]
        assert len(identities) == len(set(identities))
        opt = torch.optim.AdamW(groups, lr=config.lr)
        sched = OptimizerParamScheduler(
            opt,
            init_lr=0,
            max_lr=config.lr,
            min_lr=config.min_lr,
            lr_warmup_steps=2,
            lr_decay_steps=8,
            lr_decay_style="linear",
            start_wd=0.1,
            end_wd=0.1,
            wd_incr_steps=8,
            wd_incr_style="constant",
        )
        return model, opt, sched

    model, optimizer, scheduler = build()
    by_param = {id(p): g for g in optimizer.param_groups for p in g["params"]}
    a, b = (
        by_param[id(getattr(model.adapter, name).weight)] for name in ("linear_in", "linear_out")
    )
    assert by_param[id(model.bias)]["wd_mult"] == 0

    def step(module, opt, sched):
        sched.step(1)
        for i, param in enumerate(module.parameters()):
            param.grad = torch.full_like(param, 0.1 * (i + 1))
        opt.step()

    for _ in range(3):
        step(model, optimizer, scheduler)
        assert b["lr"] == pytest.approx(ratio * a["lr"])
    restored, opt2, sched2 = build()
    restored.load_state_dict(deepcopy(model.state_dict()))
    opt2.load_state_dict(deepcopy(optimizer.state_dict()))
    sched2.load_state_dict(scheduler.state_dict())
    for _ in range(3):
        step(model, optimizer, scheduler)
        step(restored, opt2, sched2)
        assert b["lr"] == pytest.approx(ratio * a["lr"])
    for expected, actual in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                optimizer.state[expected][key], opt2.state[actual][key], rtol=0, atol=0
            )
    scheduler.step(8 - scheduler.num_steps)
    assert b["lr"] == pytest.approx(ratio * config.min_lr)


@pytest.mark.parametrize("ratio", [0, -1, float("nan"), float("inf")])
def test_lora_plus_rejects_invalid_ratio(ratio):
    with pytest.raises(ValueError, match="finite and positive"):
        LoRAPlusOptimizerOverrides(ratio)(optimizer_config=OptimizerConfig(lr=1e-3), model=[])


@pytest.mark.parametrize("initial_lr,ratio", [(0, 16.0), (1e-5, 1.0), (1e-5, 16.0)])
def test_lora_plus_warmup_policy_is_independent_of_local_parameters(initial_lr, ratio):
    config = OptimizerConfig(lr=1e-3, min_lr=1e-4)
    policy = LoRAPlusOptimizerOverrides(ratio)(optimizer_config=config, model=[])
    if initial_lr and ratio != 1:
        with pytest.raises(ValueError, match="lr_warmup_init=0"):
            validate_lora_plus_warmup(policy, config, initial_lr)
    else:
        validate_lora_plus_warmup(policy, config, initial_lr)


def test_manual_lora_mapping_obeys_the_same_warmup_guard():
    with pytest.raises(ValueError, match="lr_warmup_init=0"):
        validate_lora_plus_warmup(
            {ParamKey(name=("*.linear_in.weight",)): {"min_lr": 0.0}},
            OptimizerConfig(lr=1e-3, min_lr=1e-4),
            1e-5,
        )


def test_lora_plus_callable_roundtrips_with_recipe(run_config, tmp_path):
    run_config.optimizer.lr = 1e-3
    run_config.optimizer.min_lr = 1e-4
    run_config.optimizer_config_overrides = LoRAPlusOptimizerOverrides(16.0)
    path = tmp_path / "lora.yaml"
    run_config.to_yaml(str(path))
    restored = PretrainConfigContainer.from_yaml(str(path))
    policy = restored.optimizer_config_overrides(optimizer_config=restored.optimizer, model=[])
    assert policy[ParamKey(name="*.linear_out.weight")] == {"max_lr": 0.016, "min_lr": 0.0016}


def test_lora_plus_does_not_silently_change_ratio_under_mup():
    with pytest.raises(ValueError, match="explicit combined"):
        build_optimizer_config_overrides(
            OptimizerConfig(lr=1e-3), _model(use_mup=True), LoRAPlusOptimizerOverrides(0.5)
        )


@pytest.mark.parametrize(
    "lr,minimum",
    [(None, 0.0), (float("inf"), 0.0), (0.001, -0.1), (0.001, 0.01), (0.001, float("nan"))],
)
def test_lora_plus_shared_builder_rejects_invalid_bounds(lr, minimum):
    with pytest.raises(ValueError, match="finite base LR bounds"):
        get_lora_plus_lr_overrides(OptimizerConfig(lr=lr, min_lr=minimum), 16.0)


def test_lora_plus_shared_builder_preserves_explicit_legacy_groups():
    overrides = get_lora_plus_lr_overrides(OptimizerConfig(lr=0.001, min_lr=None), 1.0)
    assert overrides == {
        ParamKey(name="*.linear_in.weight"): {"max_lr": 0.001, "min_lr": 0.0},
        ParamKey(name="*.linear_out.weight"): {"max_lr": 0.001, "min_lr": 0.0},
    }
    with pytest.raises(ValueError, match="scaled learning rate must be finite"):
        get_lora_plus_lr_overrides(OptimizerConfig(lr=1e308), 16.0)


@pytest.mark.parametrize("overrides", [None, {}])
def test_lora_plus_warmup_accepts_absent_policies(overrides):
    validate_lora_plus_warmup(overrides, OptimizerConfig(lr=0.001), 0.0001)


@pytest.mark.parametrize("name", ["*linear_out.weight", ("*.linear_out.weight", "*.other")])
def test_warmup_composes_all_matching_name_selectors(name):
    config = OptimizerConfig(lr=0.001, min_lr=0.0001)
    raised = {"max_lr": 0.004}
    mapping = {ParamKey(name=name): raised}
    with pytest.raises(ValueError, match="lr_warmup_init=0"):
        validate_lora_plus_warmup(mapping, config, 1e-5)
    mapping[ParamKey(name="*.weight")] = raised
    validate_lora_plus_warmup(mapping, config, 1e-5)


@pytest.mark.parametrize("mixed", [False, True])
def test_warmup_does_not_infer_lora_from_general_selectors(mixed):
    selector = (
        ParamKey(
            name="*.linear_out.weight", predicate=ParamPredicate("matrix", lambda p: p.ndim == 2)
        )
        if mixed
        else ParamKey(name="*out*.weight")
    )
    validate_lora_plus_warmup({selector: {"max_lr": 0.016}}, OptimizerConfig(lr=0.001), 1e-5)


def test_lora_b_and_decoupled_embedding_share_a_safe_group(distributed_groups):
    config = OptimizerConfig(lr=0.001, min_lr=0.0, decoupled_lr=0.016)
    model = torch.nn.Module()
    model.embedding = torch.nn.Parameter(torch.ones(4, 4))
    model.embedding.is_embedding_or_output_parameter = True
    model.adapter = torch.nn.Module()
    model.adapter.linear_in = torch.nn.Linear(4, 4, bias=False)
    model.adapter.linear_out = torch.nn.Linear(4, 4, bias=False)
    policy = build_optimizer_config_overrides(config, [model], LoRAPlusOptimizerOverrides(16.0))
    groups = _get_param_groups([model], config, policy)
    assert len(groups) == len({get_param_group_identifier_tuple(group) for group in groups})
    inner = torch.optim.AdamW(groups, lr=config.lr)
    for index, param in enumerate(model.parameters()):
        param.grad = torch.full_like(param, index + 1)
    inner.step()
    expected = {param: deepcopy(inner.state[param]) for param in model.parameters()}
    wrapper = FP32Optimizer.__new__(FP32Optimizer)
    wrapper.optimizer = inner
    wrapper.load_state_dict(deepcopy(inner.state_dict()))
    members = [param for group in inner.param_groups for param in group["params"]]
    assert len(members) == len({id(param) for param in members}) == 3
    for param in model.parameters():
        torch.testing.assert_close(inner.state[param]["exp_avg"], expected[param]["exp_avg"])
