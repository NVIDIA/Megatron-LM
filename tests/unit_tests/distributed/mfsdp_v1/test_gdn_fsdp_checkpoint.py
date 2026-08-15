# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Distributed regression tests for GDN-family FSDP DTensor checkpoint splitting."""

import pytest
import torch
from torch.distributed._tensor import DeviceMesh, Shard, distribute_tensor

from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
    uneven_dtensor_to_full_tensor,
)
from megatron.core.transformer.fsdp_dtensor_checkpoint import handle_gdn_in_state_dict


class _FakeGDN(torch.nn.Module):
    def __init__(self, split_names, split_sections):
        super().__init__()
        self.qk_dim = 16
        self.v_dim = 24
        self.tp_size = 1
        self.in_proj_dim = sum(split_sections)
        self.in_proj_split_names = split_names
        self.in_proj_split_sections = split_sections
        self.in_proj = torch.nn.Linear(4, self.in_proj_dim, bias=False)
        self.conv1d = torch.nn.Conv1d(4, 2 * self.qk_dim + self.v_dim, 1, bias=True)


def _make_wrapped_model(split_names, split_sections):
    model = torch.nn.Module()
    model.module = torch.nn.Module()
    model.module.gdn = _FakeGDN(split_names, split_sections)
    return model


def _make_sharded_tensor(shape, device_mesh, device, offset=0):
    tensor = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32, device=device)
    tensor = tensor.reshape(shape) + offset
    return tensor, distribute_tensor(tensor, device_mesh, [Shard(0)])


def _assert_dtensor_splits(state_dict, key, names, reference):
    reference_splits = torch.split(
        reference, [state_dict[f"{key}.{name}"].shape[0] for name in names]
    )
    for name, expected in zip(names, reference_splits):
        actual = uneven_dtensor_to_full_tensor(state_dict[f"{key}.{name}"])
        torch.testing.assert_close(actual, expected)


def _optimizer_entry(exp_avg_dtensor, exp_avg_sq_dtensor):
    return {"exp_avg": exp_avg_dtensor, "exp_avg_sq": exp_avg_sq_dtensor, "step": torch.tensor(1.0)}


@pytest.mark.parametrize(
    ("split_names", "split_sections"),
    [
        (("query", "key", "value", "z", "beta", "alpha"), (16, 16, 24, 24, 16, 24)),
        (("query", "key", "value", "z", "f", "b", "w"), (16, 16, 24, 24, 16, 16, 24)),
        (("query", "key", "value", "g", "gate"), (16, 16, 24, 16, 24)),
    ],
)
def test_gdn_model_and_optimizer_dtensors_follow_variant_metadata(
    distributed_setup, split_names, split_sections
):
    mesh = DeviceMesh(distributed_setup.device.type, list(range(distributed_setup.world_size)))
    model = _make_wrapped_model(split_names, split_sections)
    gdn = model.module.gdn

    weight, weight_dtensor = _make_sharded_tensor(
        tuple(gdn.in_proj.weight.shape), mesh, distributed_setup.device
    )
    conv_weight, conv_weight_dtensor = _make_sharded_tensor(
        tuple(gdn.conv1d.weight.shape), mesh, distributed_setup.device, offset=1_000
    )
    conv_bias, conv_bias_dtensor = _make_sharded_tensor(
        tuple(gdn.conv1d.bias.shape), mesh, distributed_setup.device, offset=2_000
    )
    exp_avg, exp_avg_dtensor = _make_sharded_tensor(
        tuple(gdn.in_proj.weight.shape), mesh, distributed_setup.device, offset=3_000
    )
    exp_avg_sq, exp_avg_sq_dtensor = _make_sharded_tensor(
        tuple(gdn.in_proj.weight.shape), mesh, distributed_setup.device, offset=4_000
    )
    conv_weight_exp_avg, conv_weight_exp_avg_dtensor = _make_sharded_tensor(
        tuple(gdn.conv1d.weight.shape), mesh, distributed_setup.device, offset=5_000
    )
    conv_weight_exp_avg_sq, conv_weight_exp_avg_sq_dtensor = _make_sharded_tensor(
        tuple(gdn.conv1d.weight.shape), mesh, distributed_setup.device, offset=6_000
    )
    conv_bias_exp_avg, conv_bias_exp_avg_dtensor = _make_sharded_tensor(
        tuple(gdn.conv1d.bias.shape), mesh, distributed_setup.device, offset=7_000
    )
    conv_bias_exp_avg_sq, conv_bias_exp_avg_sq_dtensor = _make_sharded_tensor(
        tuple(gdn.conv1d.bias.shape), mesh, distributed_setup.device, offset=8_000
    )
    untouched = torch.tensor([9_000.0], device=distributed_setup.device)

    model_state = {
        "gdn.in_proj.weight": weight_dtensor,
        "gdn.conv1d.weight": conv_weight_dtensor,
        "gdn.conv1d.bias": conv_bias_dtensor,
    }
    optimizer_state = {
        "state": {
            "module.gdn.in_proj.weight": _optimizer_entry(exp_avg_dtensor, exp_avg_sq_dtensor),
            "module.gdn.conv1d.weight": _optimizer_entry(
                conv_weight_exp_avg_dtensor, conv_weight_exp_avg_sq_dtensor
            ),
            "module.gdn.conv1d.bias": _optimizer_entry(
                conv_bias_exp_avg_dtensor, conv_bias_exp_avg_sq_dtensor
            ),
            "module.unrelated.weight": {"step": untouched},
        }
    }

    split_model_state, split_optimizer_state = handle_gdn_in_state_dict(
        model, model_state, optimizer_state
    )

    assert "gdn.in_proj.weight" not in split_model_state
    assert "gdn.conv1d.weight" not in split_model_state
    assert "gdn.conv1d.bias" not in split_model_state
    _assert_dtensor_splits(split_model_state, "gdn.in_proj.weight", split_names, weight)
    _assert_dtensor_splits(
        split_model_state, "gdn.conv1d.weight", ("query", "key", "value"), conv_weight
    )
    _assert_dtensor_splits(
        split_model_state, "gdn.conv1d.bias", ("query", "key", "value"), conv_bias
    )

    split_optimizer = split_optimizer_state["state"]
    assert "module.gdn.in_proj.weight" not in split_optimizer
    _assert_dtensor_splits(
        {key: value["exp_avg"] for key, value in split_optimizer.items() if "exp_avg" in value},
        "module.gdn.in_proj.weight",
        split_names,
        exp_avg,
    )
    _assert_dtensor_splits(
        {
            key: value["exp_avg_sq"]
            for key, value in split_optimizer.items()
            if "exp_avg_sq" in value
        },
        "module.gdn.in_proj.weight",
        split_names,
        exp_avg_sq,
    )
    optimizer_references = {
        "module.gdn.conv1d.weight": (conv_weight_exp_avg, conv_weight_exp_avg_sq),
        "module.gdn.conv1d.bias": (conv_bias_exp_avg, conv_bias_exp_avg_sq),
    }
    for key, (expected_exp_avg, expected_exp_avg_sq) in optimizer_references.items():
        _assert_dtensor_splits(
            {
                name: value["exp_avg"]
                for name, value in split_optimizer.items()
                if "exp_avg" in value
            },
            key,
            ("query", "key", "value"),
            expected_exp_avg,
        )
        _assert_dtensor_splits(
            {
                name: value["exp_avg_sq"]
                for name, value in split_optimizer.items()
                if "exp_avg_sq" in value
            },
            key,
            ("query", "key", "value"),
            expected_exp_avg_sq,
        )

    gdn_states = [value for key, value in split_optimizer.items() if key.startswith("module.gdn.")]
    assert all(value["step"].item() == 1.0 for value in gdn_states)
    assert split_optimizer["module.unrelated.weight"]["step"] is untouched
