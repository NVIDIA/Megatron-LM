# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import torch

from megatron.core.dist_checkpointing import load, save
from megatron.core.extensions import transformer_engine as te_ext
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


def _disable_scale_collectives(monkeypatch):
    monkeypatch.setattr(te_ext, "_get_ptq_scale_reduction_groups", lambda _: ())
    monkeypatch.setattr(te_ext, "get_pipeline_model_parallel_world_size", lambda: 1)


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
