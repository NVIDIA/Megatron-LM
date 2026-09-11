# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from megatron.core import fp4_utils, fp8_utils
from megatron.core.extensions import transformer_engine as te_ext
from megatron.core.transformer.transformer_config import TransformerConfig

pytestmark = pytest.mark.skipif(
    not fp8_utils.HAVE_TE or not hasattr(te_ext.te.pytorch, "QuantizationCalibrationConfig"),
    reason="Transformer Engine calibration API is required",
)


def _disable_scale_collectives(monkeypatch):
    monkeypatch.setattr(te_ext, "_get_ptq_scale_reduction_groups", lambda _: ())
    monkeypatch.setattr(te_ext, "get_pipeline_model_parallel_world_size", lambda: 1)


def test_calibration_options_default_to_disabled():
    fields = {field.name: field for field in dataclasses.fields(TransformerConfig)}

    assert fields["buffer_transformer_engine_calibration_metadata"].default is False
    assert fields["transformer_engine_calibration_decay"].default == 0.0


def test_disabled_megatron_option_returns_no_te_calibration_config():
    config = SimpleNamespace(
        buffer_transformer_engine_calibration_metadata=False,
        transformer_engine_calibration_decay=0.5,
    )

    assert fp8_utils.get_te_calibration_config(config) is None


def test_calibration_option_requires_te_config_support(monkeypatch):
    config = SimpleNamespace(
        buffer_transformer_engine_calibration_metadata=True,
        transformer_engine_calibration_decay=0.123,
    )
    monkeypatch.setattr(
        fp8_utils.transformer_engine.pytorch,
        "QuantizationCalibrationConfig",
        None,
    )
    fp8_utils._make_te_calibration_config.cache_clear()

    with pytest.raises(RuntimeError, match="QuantizationCalibrationConfig support"):
        fp8_utils.get_te_calibration_config(config)


def test_megatron_options_create_te_calibration_config():
    config = SimpleNamespace(
        buffer_transformer_engine_calibration_metadata=True,
        transformer_engine_calibration_decay=0.5,
    )

    calibration_config = fp8_utils.get_te_calibration_config(config)

    assert isinstance(calibration_config, te_ext.te.pytorch.QuantizationCalibrationConfig)
    assert calibration_config.transformer_engine_calibration_decay == 0.5
    assert fp8_utils.get_te_calibration_config(config) is calibration_config


def test_fp8_context_receives_calibration_config(monkeypatch):
    recipe = object()
    context = object()
    autocast_kwargs = {}
    config = SimpleNamespace(
        fp8=True,
        fp8_param=False,
        first_last_layers_bf16=False,
        buffer_transformer_engine_calibration_metadata=True,
        transformer_engine_calibration_decay=0.5,
    )
    monkeypatch.setattr(fp8_utils, "is_first_last_bf16_layer", lambda *_: False)
    monkeypatch.setattr(fp8_utils, "get_fp8_recipe", lambda _: recipe)
    monkeypatch.setattr(fp8_utils.parallel_state, "model_parallel_is_initialized", lambda: False)
    monkeypatch.setattr(
        fp8_utils.transformer_engine.pytorch,
        "fp8_autocast",
        lambda **kwargs: (autocast_kwargs.update(kwargs), context)[1],
    )

    assert fp8_utils.get_fp8_context(config) is context
    assert autocast_kwargs == {
        "enabled": True,
        "fp8_recipe": recipe,
        "fp8_group": None,
        "calibration_config": fp8_utils.get_te_calibration_config(config),
    }


def test_fp8_context_reports_unsupported_te_calibration_autocast(monkeypatch):
    config = SimpleNamespace(
        fp8=True,
        fp8_param=False,
        first_last_layers_bf16=False,
        buffer_transformer_engine_calibration_metadata=True,
        transformer_engine_calibration_decay=0.5,
    )
    monkeypatch.setattr(fp8_utils, "is_first_last_bf16_layer", lambda *_: False)
    monkeypatch.setattr(fp8_utils, "get_fp8_recipe", lambda _: object())
    monkeypatch.setattr(fp8_utils.parallel_state, "model_parallel_is_initialized", lambda: False)

    def legacy_fp8_autocast(*, enabled, fp8_recipe, fp8_group):
        return enabled, fp8_recipe, fp8_group

    monkeypatch.setattr(
        fp8_utils.transformer_engine.pytorch,
        "fp8_autocast",
        legacy_fp8_autocast,
    )

    with pytest.raises(
        RuntimeError,
        match="--buffer-transformer-engine-calibration-metadata.*accepts calibration_config",
    ):
        fp8_utils.get_fp8_context(config)


def test_fp8_context_ignores_calibration_when_fp8_is_disabled(monkeypatch):
    config = SimpleNamespace(
        fp8=False,
        fp8_param=False,
        first_last_layers_bf16=False,
        buffer_transformer_engine_calibration_metadata=True,
        transformer_engine_calibration_decay=0.5,
    )
    monkeypatch.setattr(fp8_utils, "is_first_last_bf16_layer", lambda *_: False)
    monkeypatch.setattr(
        fp8_utils.transformer_engine.pytorch,
        "fp8_autocast",
        lambda **_: pytest.fail("non-quantized layer should not enter TE autocast"),
    )

    with fp8_utils.get_fp8_context(config):
        pass


def test_fp4_context_receives_calibration_config(monkeypatch):
    recipe = object()
    context = object()
    autocast_kwargs = {}
    config = SimpleNamespace(
        fp4=True,
        fp4_param=False,
        first_last_layers_bf16=False,
        num_layers=1,
        buffer_transformer_engine_calibration_metadata=True,
        transformer_engine_calibration_decay=0.5,
    )
    monkeypatch.setattr(fp4_utils, "get_fp4_recipe", lambda _: recipe)
    monkeypatch.setattr(fp4_utils.parallel_state, "model_parallel_is_initialized", lambda: False)
    monkeypatch.setattr(
        fp4_utils.transformer_engine.pytorch,
        "fp8_autocast",
        lambda **kwargs: (autocast_kwargs.update(kwargs), context)[1],
    )

    assert fp4_utils.get_fp4_context(config) is context
    assert autocast_kwargs == {
        "enabled": True,
        "fp8_recipe": recipe,
        "fp8_group": None,
        "calibration_config": fp8_utils.get_te_calibration_config(config),
    }


def test_fp4_context_ignores_calibration_when_fp4_is_disabled(monkeypatch):
    config = SimpleNamespace(
        fp4=False,
        fp4_param=False,
        first_last_layers_bf16=False,
        num_layers=1,
        buffer_transformer_engine_calibration_metadata=True,
        transformer_engine_calibration_decay=0.5,
    )
    monkeypatch.setattr(
        fp4_utils.transformer_engine.pytorch,
        "fp8_autocast",
        lambda **_: pytest.fail("non-quantized layer should not enter TE autocast"),
    )

    with fp4_utils.get_fp4_context(config):
        pass


def test_disabled_fp8_context_ignores_calibration(monkeypatch):
    context = object()
    autocast_kwargs = {}
    config = SimpleNamespace(
        fp8=True,
        fp4=None,
        buffer_transformer_engine_calibration_metadata=True,
        transformer_engine_calibration_decay=0.5,
    )
    monkeypatch.setattr(
        fp8_utils.transformer_engine.pytorch,
        "fp8_autocast",
        lambda **kwargs: (autocast_kwargs.update(kwargs), context)[1],
    )

    assert fp8_utils.get_fp8_disabled_context(config) is context
    assert autocast_kwargs == {"enabled": False}


def test_per_module_disabled_autocast_ignores_calibration_config(monkeypatch):
    calibration_config = object()
    autocast_kwargs = {}
    monkeypatch.setattr(te_ext.FP8GlobalStateManager, "is_fp8_enabled", lambda: False)
    monkeypatch.setattr(te_ext, "fp8_autocast", lambda **kwargs: autocast_kwargs.update(kwargs))
    recipe = te_ext.TEQuantizationRecipe(override_nonquantized_autocast=True)

    te_ext._get_fp8_autocast_for_quant_recipe(
        recipe,
        calibration_config=calibration_config,
    )

    assert autocast_kwargs == {"enabled": False}


def test_per_module_quantized_autocast_receives_calibration_config(monkeypatch):
    calibration_config = object()
    autocast_kwargs = {}
    monkeypatch.setattr(te_ext.FP8GlobalStateManager, "is_fp8_enabled", lambda: False)
    monkeypatch.setattr(te_ext, "model_parallel_is_initialized", lambda: False)
    monkeypatch.setattr(te_ext, "fp8_autocast", lambda **kwargs: autocast_kwargs.update(kwargs))
    recipe = te_ext.TEQuantizationRecipe(
        fp8_quantization_recipe=te_ext.Fp8Recipe.tensorwise,
        override_nonquantized_autocast=True,
    )

    te_ext._get_fp8_autocast_for_quant_recipe(
        recipe,
        calibration_config=calibration_config,
    )

    assert autocast_kwargs["enabled"]
    assert autocast_kwargs["calibration_config"] is calibration_config


def test_calibration_amax_conversion_and_scale_inv_passthrough(monkeypatch):
    _disable_scale_collectives(monkeypatch)
    linear = torch.nn.Module()
    linear.register_buffer(
        "input_tensor_amax_fp8_delayed_scaling_te_ptq_calibrated", torch.tensor([896.0])
    )
    linear.register_buffer(
        "weight_tensor_scale_inv_fp8_current_scaling_te_ptq_calibrated", torch.tensor([0.25])
    )
    model = torch.nn.Module()
    model.add_module("linear", linear)
    state_dict = {"model": {}}

    te_ext.add_ptq_calibration_metadata_to_state_dict(state_dict, [model])

    torch.testing.assert_close(state_dict["model"]["linear.input_scale"], torch.tensor([2.0]))
    torch.testing.assert_close(state_dict["model"]["linear.weight_scale"], torch.tensor([0.25]))
    assert state_dict["model"]["linear.input_scale"].device.type == "cpu"
    assert state_dict["model"]["linear.weight_scale"].device.type == "cpu"


def test_nvfp4_weight_amax_exports_secondary_weight_scale(monkeypatch):
    _disable_scale_collectives(monkeypatch)
    linear = torch.nn.Module()
    linear.register_buffer("weight_tensor_amax_nvfp4_te_ptq_calibrated", torch.tensor([2688.0]))
    model = torch.nn.Module()
    model.add_module("linear", linear)
    state_dict = {"model": {}}

    te_ext.add_ptq_calibration_metadata_to_state_dict(state_dict, [model])

    torch.testing.assert_close(state_dict["model"]["linear.weight_scale_2"], torch.ones(1))


def test_global_layer_fqn_uses_layer_number():
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.layers[0].layer_number = 5
    named_modules = dict(model.named_modules())

    assert (
        te_ext._get_global_layer_fqn("layers.0.mlp.linear_fc1", named_modules)
        == "layers.4.mlp.linear_fc1"
    )


def test_expert_parallel_calibration_scales_are_gathered(monkeypatch):
    _disable_scale_collectives(monkeypatch)
    monkeypatch.setattr(te_ext, "get_pg_size", lambda _: 2)

    ep_group = object()
    grouped_linear = torch.nn.Module()
    grouped_linear.num_gemms = 2
    grouped_linear._pg_collection = SimpleNamespace(ep=ep_group)
    grouped_linear.register_buffer(
        "input_gemm0_tensor_scale_inv_fp8_current_scaling_te_ptq_calibrated", torch.tensor([0.25])
    )
    grouped_linear.register_buffer(
        "input_gemm1_tensor_scale_inv_fp8_current_scaling_te_ptq_calibrated", torch.tensor([0.5])
    )
    mlp = torch.nn.Module()
    mlp.add_module("linear_fc1", grouped_linear)
    model = torch.nn.Module()
    model.add_module("mlp", mlp)

    def fake_all_gather_into_tensor(output, local_scales, group):
        assert group is ep_group
        torch.testing.assert_close(local_scales, torch.tensor([0.25, 0.5]))
        output.copy_(torch.tensor([0.25, 0.5, 0.75, 1.0]))

    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", fake_all_gather_into_tensor)
    state_dict = {"model": {}}

    te_ext.add_ptq_calibration_metadata_to_state_dict(state_dict, [model])

    for expert_idx, expected_scale in enumerate((0.25, 0.5, 0.75, 1.0)):
        torch.testing.assert_close(
            state_dict["model"][f"mlp.experts.{expert_idx}.linear_fc1.input_scale"],
            torch.tensor(expected_scale),
        )


def test_pipeline_parallel_calibration_states_are_merged(monkeypatch):
    monkeypatch.setattr(te_ext, "_get_ptq_scale_reduction_groups", lambda _: ())
    monkeypatch.setattr(te_ext, "get_pipeline_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(te_ext, "get_pg_size", lambda _: 2)

    pp_group = object()
    model = torch.nn.Module()
    model.pg_collection = SimpleNamespace(pp=pp_group)
    linear = torch.nn.Module()
    linear.register_buffer(
        "input_tensor_scale_inv_fp8_current_scaling_te_ptq_calibrated", torch.tensor([0.25])
    )
    model.add_module("linear", linear)

    remote_state = {"remote.linear.input_scale": torch.tensor([0.5])}

    def fake_all_gather_object(output, local_state, group):
        assert group is pp_group
        output[:] = [local_state, remote_state]

    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather_object)
    state_dict = {"model": {}}

    te_ext.add_ptq_calibration_metadata_to_state_dict(state_dict, [model])

    torch.testing.assert_close(state_dict["model"]["linear.input_scale"], torch.tensor([0.25]))
    torch.testing.assert_close(
        state_dict["model"]["remote.linear.input_scale"], torch.tensor([0.5])
    )


def test_pipeline_parallel_conflicting_calibration_states_are_rejected(monkeypatch):
    monkeypatch.setattr(te_ext, "_get_ptq_scale_reduction_groups", lambda _: ())
    monkeypatch.setattr(te_ext, "get_pipeline_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(te_ext, "get_pg_size", lambda _: 2)

    pp_group = object()
    model = torch.nn.Module()
    model.pg_collection = SimpleNamespace(pp=pp_group)
    linear = torch.nn.Module()
    linear.register_buffer(
        "input_tensor_scale_inv_fp8_current_scaling_te_ptq_calibrated", torch.tensor([0.25])
    )
    model.add_module("linear", linear)

    def fake_all_gather_object(output, local_state, group):
        assert group is pp_group
        output[:] = [local_state, {"linear.input_scale": torch.tensor([0.5])}]

    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather_object)

    with pytest.raises(RuntimeError, match="Conflicting PTQ calibration values"):
        te_ext.add_ptq_calibration_metadata_to_state_dict({"model": {}}, [model])
