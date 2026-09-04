# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.moe.megakernel import parameter_bridge
from megatron.core.transformer.moe.megakernel.mok import backend as mok_backend
from megatron.core.transformer.moe.megakernel.mok import weights as mok_weights


def test_parameter_bridge_uses_main_grad_without_allocating_dummy_storage():
    param = torch.nn.Parameter(torch.zeros((4, 8), dtype=torch.bfloat16))
    param.main_grad = torch.full((4, 8), 0.25, dtype=torch.float32)
    param.grad_added_to_main_grad = False

    assert parameter_bridge.main_grad_buffer(param) is param.main_grad
    dummy = parameter_bridge.finish_weight_gradient(param)

    assert dummy.data_ptr() == param.data_ptr()
    assert not dummy.requires_grad
    assert param.grad_added_to_main_grad
    torch.testing.assert_close(param.main_grad, torch.full_like(param.main_grad, 0.25))


def test_mxfp8_scale_layout_and_single_grouped_storage_contract(monkeypatch):
    from megatron.core import fp8_utils

    rows = columns = 128
    logical = torch.arange(rows * (columns // 32), dtype=torch.int32).to(torch.uint8)
    logical = logical.reshape(1, rows, columns // 32)
    swizzled = mok_weights._swizzle_mxfp8_scale(logical, rows=rows, columns=columns)

    assert swizzled.shape == (1, 1, 32, 16)
    for lane, row_group, column_scale in ((0, 0, 0), (7, 2, 1), (31, 3, 3)):
        assert swizzled[0, 0, lane, row_group * 4 + column_scale] == logical[
            0, row_group * 32 + lane, column_scale
        ]

    num_experts = 3
    member_shape = (4, 2)
    backing = torch.empty((num_experts, *member_shape), device="cuda", dtype=torch.uint8)
    members = [SimpleNamespace(_rowwise_scale_inv=backing[index]) for index in range(num_experts)]
    monkeypatch.setattr(fp8_utils, "get_grouped_quantized_members", lambda _: members)

    view = mok_weights._single_grouped_mxfp8_scale_view(
        object(), "_rowwise_scale_inv", (num_experts, *member_shape), name="test rowwise"
    )
    assert view.data_ptr() == backing.data_ptr()

    separate_members = [
        SimpleNamespace(
            _rowwise_scale_inv=torch.empty(member_shape, device="cuda", dtype=torch.uint8)
        )
        for _ in range(2)
    ]
    monkeypatch.setattr(
        fp8_utils, "get_grouped_quantized_members", lambda _: separate_members
    )
    with pytest.raises(RuntimeError, match="not packed expert-major"):
        mok_weights._single_grouped_mxfp8_scale_view(
            object(), "_rowwise_scale_inv", (2, *member_shape), name="test rowwise"
        )


def _shared_module(fc1_weight, fc2_weight):
    shared = torch.nn.Module()
    shared.linear_fc1 = torch.nn.Module()
    shared.linear_fc2 = torch.nn.Module()
    shared.linear_fc1.register_parameter("weight", fc1_weight)
    shared.linear_fc2.register_parameter("weight", fc2_weight)
    return shared


@pytest.mark.parametrize("single_grouped", [False, True])
def test_checkpoint_uses_only_canonical_mcore_parameters(single_grouped):
    experts = torch.nn.Module()
    experts.linear_fc1 = torch.nn.Module()
    experts.linear_fc2 = torch.nn.Module()
    if single_grouped:
        routed_fc1 = torch.nn.Parameter(torch.zeros((2, 8, 8)))
        routed_fc2 = torch.nn.Parameter(torch.zeros((2, 8, 4)))
        experts.linear_fc1.register_parameter("weight", routed_fc1)
        experts.linear_fc2.register_parameter("weight", routed_fc2)
        routed_checkpoint = {
            "experts.linear_fc1.weight": torch.full_like(routed_fc1, 3.0),
            "experts.linear_fc2.weight": torch.full_like(routed_fc2, 5.0),
        }
    else:
        routed_fc1 = torch.nn.Parameter(torch.zeros((8, 8)))
        routed_fc2 = torch.nn.Parameter(torch.zeros((8, 4)))
        experts.linear_fc1.register_parameter("weight0", routed_fc1)
        experts.linear_fc2.register_parameter("weight0", routed_fc2)
        routed_checkpoint = {
            "experts.linear_fc1.weight0": torch.full_like(routed_fc1, 3.0),
            "experts.linear_fc2.weight0": torch.full_like(routed_fc2, 5.0),
        }

    shared_fc1 = torch.nn.Parameter(torch.zeros((8, 8)))
    shared_fc2 = torch.nn.Parameter(torch.zeros((8, 4)))
    shared = _shared_module(shared_fc1, shared_fc2)

    mok = mok_backend.MoKMegakernel.__new__(mok_backend.MoKMegakernel)
    torch.nn.Module.__init__(mok)
    mok.native_single_grouped_weights = single_grouped
    mok._routed_weight_view_cache = object()
    mok._split_main_grad_descriptor_cache = object()
    mok.is_first_microbatch = False
    if single_grouped:
        mok.register_parameter("routed_fc1_weight", routed_fc1)
        mok.register_parameter("routed_fc2_weight", routed_fc2)
    else:
        mok._routed_fc1_parameter_names = ("routed_fc1_weight0",)
        mok._routed_fc2_parameter_names = ("routed_fc2_weight0",)
        mok.register_parameter("routed_fc1_weight0", routed_fc1)
        mok.register_parameter("routed_fc2_weight0", routed_fc2)
    mok.register_parameter("shared_fc1_weight", shared_fc1)
    mok.register_parameter("shared_fc2_weight", shared_fc2)

    parent = torch.nn.Module()
    parent.add_module("experts", experts)
    parent.add_module("shared_experts", shared)
    parent.add_module("megakernel_experts", mok)
    checkpoint = {
        **routed_checkpoint,
        "shared_experts.linear_fc1.weight": torch.full_like(shared_fc1, 7.0),
        "shared_experts.linear_fc2.weight": torch.full_like(shared_fc2, 11.0),
    }

    assert set(parent.state_dict()) == set(checkpoint)
    assert mok.sharded_state_dict(prefix="megakernel_experts.") == {}
    parent.load_state_dict(checkpoint, strict=True)

    torch.testing.assert_close(routed_fc1, next(iter(routed_checkpoint.values())))
    torch.testing.assert_close(routed_fc2, tuple(routed_checkpoint.values())[1])
    torch.testing.assert_close(shared_fc1, checkpoint["shared_experts.linear_fc1.weight"])
    torch.testing.assert_close(shared_fc2, checkpoint["shared_experts.linear_fc2.weight"])
    assert mok._routed_weight_view_cache is None
    assert mok._split_main_grad_descriptor_cache is None
    assert mok.is_first_microbatch
