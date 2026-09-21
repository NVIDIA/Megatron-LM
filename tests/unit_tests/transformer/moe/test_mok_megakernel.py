# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from megatron.core.transformer.moe.megakernel import parameter_bridge
from megatron.core.transformer.moe.megakernel.mok import backend as mok_backend
from megatron.core.transformer.moe.megakernel.mok import runtime as mok_runtime
from megatron.core.transformer.moe.megakernel.mok import weights as mok_weights


def test_gate_up_weight_arguments_split_only_shared_fc1():
    shared_fc1 = torch.arange(64, dtype=torch.bfloat16).reshape(8, 8)
    routed_fc1 = object()

    shared_gate, shared_up, routed_gate, routed_up = mok_runtime._gate_up_weight_arguments(
        shared_fc1, routed_fc1, intermediate_size=4
    )

    torch.testing.assert_close(shared_gate, shared_fc1[:4])
    torch.testing.assert_close(shared_up, shared_fc1[4:])
    assert shared_gate.untyped_storage().data_ptr() == shared_fc1.untyped_storage().data_ptr()
    assert shared_up.untyped_storage().data_ptr() == shared_fc1.untyped_storage().data_ptr()
    assert routed_gate is routed_fc1
    assert routed_up is routed_fc1


def test_gate_up_main_grad_arguments_preserve_fc1_aliases():
    shared_fc1 = torch.zeros((8, 8), dtype=torch.float32)
    routed_fc1 = torch.zeros((2, 8, 8), dtype=torch.float32)
    shared_fc2 = torch.zeros((8, 4), dtype=torch.float32)
    routed_fc2 = torch.zeros((2, 8, 4), dtype=torch.float32)
    fc1_table = object()
    fc2_table = object()

    actual, tables = mok_runtime._gate_up_main_grad_arguments(
        (shared_fc1, routed_fc1, shared_fc2, routed_fc2),
        (fc1_table, fc2_table),
        intermediate_size=4,
    )

    shared_gate, routed_gate, shared_up, routed_up, actual_shared_fc2, actual_routed_fc2 = actual
    assert shared_gate.untyped_storage().data_ptr() == shared_fc1.untyped_storage().data_ptr()
    assert shared_up.untyped_storage().data_ptr() == shared_fc1.untyped_storage().data_ptr()
    assert routed_gate is routed_fc1
    assert routed_up is routed_fc1
    assert actual_shared_fc2 is shared_fc2
    assert actual_routed_fc2 is routed_fc2
    assert tables == (fc1_table, fc1_table, fc2_table)


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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_main_grad_buffer_accepts_supported_dtype(dtype):
    param = torch.nn.Parameter(torch.zeros((4, 8), dtype=torch.bfloat16))
    param.main_grad = torch.zeros((4, 8), dtype=dtype)

    assert parameter_bridge.main_grad_buffer(param) is param.main_grad


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_main_grad_buffer_rejects_unsupported_dtype(dtype):
    param = torch.nn.Parameter(torch.zeros((4, 8), dtype=torch.bfloat16))
    param.main_grad = torch.zeros((4, 8), dtype=dtype)

    with pytest.raises(RuntimeError, match="FP32 or BF16"):
        parameter_bridge.main_grad_buffer(param)


def test_mxfp8_scale_layout_and_single_grouped_storage_contract(monkeypatch):
    from megatron.core import fp8_utils

    rows = columns = 128
    logical = torch.arange(rows * (columns // 32), dtype=torch.int32).to(torch.uint8)
    logical = logical.reshape(1, rows, columns // 32)
    swizzled = mok_weights._swizzle_mxfp8_scale(logical, rows=rows, columns=columns)

    assert swizzled.shape == (1, 1, 32, 16)
    for lane, row_group, column_scale in ((0, 0, 0), (7, 2, 1), (31, 3, 3)):
        assert (
            swizzled[0, 0, lane, row_group * 4 + column_scale]
            == logical[0, row_group * 32 + lane, column_scale]
        )

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
    monkeypatch.setattr(fp8_utils, "get_grouped_quantized_members", lambda _: separate_members)
    with pytest.raises(RuntimeError, match="not packed expert-major"):
        mok_weights._single_grouped_mxfp8_scale_view(
            object(), "_rowwise_scale_inv", (2, *member_shape), name="test rowwise"
        )


def test_swizzle_mxfp8_scale_refreshes_existing_output_in_place():
    rows = columns = 128
    logical = torch.arange(rows * (columns // 32), dtype=torch.int32).to(torch.uint8)
    logical = logical.reshape(1, rows, columns // 32)
    expected = mok_weights._swizzle_mxfp8_scale(logical, rows=rows, columns=columns)
    output = torch.empty_like(expected)
    output_ptr = output.data_ptr()

    actual = mok_weights._swizzle_mxfp8_scale(logical, rows=rows, columns=columns, out=output)

    assert actual is output
    assert actual.data_ptr() == output_ptr
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_single_grouped_mxfp8_view_builds_and_refreshes_scales(monkeypatch):
    from megatron.core import fp8_utils

    class GroupedWeight:
        def __init__(self, shape):
            self.shape = shape
            self.rowwise_data = object()
            self.columnwise_data = object()

    num_experts, rows, columns = 1, 256, 128
    weight = GroupedWeight((num_experts, rows, columns))
    scales = {}
    swizzle_calls = []

    monkeypatch.setattr(fp8_utils, "is_grouped_mxfp8tensor", lambda param: True)
    monkeypatch.setattr(mok_weights, "_storage_view", lambda storage, shape, **kwargs: storage)

    def fake_scale_view(param, member_attr, shape, *, name):
        del name
        return scales.setdefault((id(param), member_attr), torch.empty(shape))

    def fake_swizzle(scale, *, rows, columns, out=None):
        result = object() if out is None else out
        swizzle_calls.append((scale, rows, columns, out, result))
        return result

    monkeypatch.setattr(mok_weights, "_single_grouped_mxfp8_scale_view", fake_scale_view)
    monkeypatch.setattr(mok_weights, "_swizzle_mxfp8_scale", fake_swizzle)

    first = mok_weights._native_single_grouped_weight_view(
        weight, num_experts=num_experts, rows=rows, columns=columns, use_mxfp8=True
    )
    assert first[0] is weight.rowwise_data
    assert first[2] is weight.columnwise_data
    assert [call[3] for call in swizzle_calls] == [None, None]

    swizzle_calls.clear()
    refreshed = mok_weights._native_single_grouped_weight_view(
        weight,
        num_experts=num_experts,
        rows=rows,
        columns=columns,
        use_mxfp8=True,
        cached_view=first,
    )
    expected_outputs = [first[1], first[3]]
    assert [call[3] for call in swizzle_calls] == expected_outputs
    assert [refreshed[1], refreshed[3]] == expected_outputs


def test_mxfp8_scale_layout_cache_refreshes_once_per_optimizer_iteration(monkeypatch):
    module = mok_backend.MoKMegakernel.__new__(mok_backend.MoKMegakernel)
    torch.nn.Module.__init__(module)
    module.native_single_grouped_weights = True
    module.use_mxfp8_weights = True
    module.is_first_microbatch = True
    module._routed_weight_view_cache = None
    module.routed_fc1_weight = object()
    module.routed_fc2_weight = object()
    module.num_local_experts = 2
    module.intermediate_size = 128
    module.hidden_size = 256

    view_calls = []

    def fake_view(*args, cached_view=None, **kwargs):
        del args, kwargs
        view_calls.append(cached_view)
        return object()

    monkeypatch.setattr(mok_backend, "_native_single_grouped_weight_view", fake_view)

    first = module.quantized_routed_weights()
    second = module.quantized_routed_weights()

    assert second is first
    assert view_calls == [None, None]

    module.is_first_microbatch = True
    third = module.quantized_routed_weights()

    assert third is not first
    assert view_calls == [None, None, first[0], first[1]]


def _split_module(*, use_mxfp8_weights):
    module = mok_backend.MoKMegakernel.__new__(mok_backend.MoKMegakernel)
    torch.nn.Module.__init__(module)
    module.native_single_grouped_weights = False
    module.use_mxfp8_weights = use_mxfp8_weights
    module.is_first_microbatch = True
    module._routed_weight_view_cache = None
    module.intermediate_size = 4
    module.hidden_size = 8
    module._routed_fc1_parameter_names = ("routed_fc1_weight0", "routed_fc1_weight1")
    module._routed_fc2_parameter_names = ("routed_fc2_weight0", "routed_fc2_weight1")
    for name in module._routed_fc1_parameter_names:
        module.register_parameter(
            name, torch.nn.Parameter(torch.randn((8, 8), dtype=torch.bfloat16))
        )
    for name in module._routed_fc2_parameter_names:
        module.register_parameter(
            name, torch.nn.Parameter(torch.randn((8, 4), dtype=torch.bfloat16))
        )
    return module


def test_parameter_storage_attr_supports_public_and_private_te_fields():
    public = SimpleNamespace(rowwise_data=torch.tensor([1]))
    private = SimpleNamespace(_rowwise_data=torch.tensor([2]))
    wrapped = SimpleNamespace(data=private)

    assert mok_weights._parameter_storage_attr(public, "rowwise_data") is public.rowwise_data
    assert mok_weights._parameter_storage_attr(private, "rowwise_data") is private._rowwise_data
    assert mok_weights._parameter_storage_attr(wrapped, "rowwise_data") is private._rowwise_data


def test_bf16_split_descriptors_are_cached(monkeypatch):
    module = _split_module(use_mxfp8_weights=False)
    calls = []

    def fake_split(params, *, rows, columns, use_mxfp8, cached_view=None):
        calls.append((params, rows, columns, use_mxfp8, cached_view))
        return object()

    monkeypatch.setattr(mok_backend, "_native_split_weight_view", fake_split)

    first = module.quantized_routed_weights()
    second = module.quantized_routed_weights()

    assert second is first
    assert [(rows, columns, use_mxfp8) for _, rows, columns, use_mxfp8, _ in calls] == [
        (8, 8, False),
        (8, 4, False),
    ]
    assert not module.is_first_microbatch


def test_mxfp8_split_scale_and_descriptor_cache_refreshes_per_iteration(monkeypatch):
    module = _split_module(use_mxfp8_weights=True)
    view_calls = []

    def fake_split(params, *, rows, columns, use_mxfp8, cached_view=None):
        view_calls.append((params, rows, columns, use_mxfp8, cached_view))
        return object() if cached_view is None else cached_view

    monkeypatch.setattr(mok_backend, "_native_split_weight_view", fake_split)

    first = module.quantized_routed_weights()
    second = module.quantized_routed_weights()

    assert second is first
    assert [call[-1] for call in view_calls] == [None, None]

    module.is_first_microbatch = True
    third = module.quantized_routed_weights()

    assert third[0] is first[0]
    assert third[1] is first[1]
    assert [call[-1] for call in view_calls] == [None, None, first[0], first[1]]


def _shared_module(fc1_weight, fc2_weight, gate_weight=None):
    shared = torch.nn.Module()
    shared.linear_fc1 = torch.nn.Module()
    shared.linear_fc2 = torch.nn.Module()
    shared.linear_fc1.register_parameter("weight", fc1_weight)
    shared.linear_fc2.register_parameter("weight", fc2_weight)
    shared.register_parameter("gate_weight", gate_weight)
    return shared


def test_register_shared_weights_rejects_non_bf16_parameters():
    fc1 = torch.nn.Parameter(torch.randn((4, 3), dtype=torch.float32))
    fc2 = torch.nn.Parameter(torch.randn((3, 2), dtype=torch.float32))
    shared = _shared_module(fc1, fc2)
    module = mok_backend.MoKMegakernel.__new__(mok_backend.MoKMegakernel)
    torch.nn.Module.__init__(module)
    module.intermediate_size = 2
    module.hidden_size = 3

    with pytest.raises(RuntimeError, match="native BF16"):
        module._register_shared_weights(shared)


@pytest.mark.parametrize("single_grouped, gated", [(False, False), (True, False), (False, True)])
def test_checkpoint_uses_only_canonical_mcore_parameters(single_grouped, gated):
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

    shared_fc1 = torch.nn.Parameter(torch.zeros((8, 8), dtype=torch.bfloat16))
    shared_fc2 = torch.nn.Parameter(torch.zeros((8, 4), dtype=torch.bfloat16))
    gate_weight = torch.nn.Parameter(torch.zeros((1, 8), dtype=torch.bfloat16)) if gated else None
    shared = _shared_module(shared_fc1, shared_fc2, gate_weight)

    mok = mok_backend.MoKMegakernel.__new__(mok_backend.MoKMegakernel)
    torch.nn.Module.__init__(mok)
    mok.native_single_grouped_weights = single_grouped
    mok.intermediate_size = 4
    mok.hidden_size = 8
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
    mok._register_shared_weights(shared, use_output_gate=gated)

    parent = torch.nn.Module()
    parent.add_module("experts", experts)
    parent.add_module("shared_experts", shared)
    parent.add_module("megakernel_experts", mok)
    checkpoint = {
        **routed_checkpoint,
        "shared_experts.linear_fc1.weight": torch.full_like(shared_fc1, 7.0),
        "shared_experts.linear_fc2.weight": torch.full_like(shared_fc2, 11.0),
    }
    if gated:
        checkpoint["shared_experts.gate_weight"] = torch.full_like(gate_weight, 13.0)

    assert set(parent.state_dict()) == set(checkpoint)
    assert set(dict(parent.named_parameters())) == set(checkpoint)
    assert mok.shared_output_gate_weight is gate_weight
    if gated:
        assert dict(mok.named_parameters(recurse=False))["shared_output_gate_weight"] is gate_weight
    assert mok.sharded_state_dict(prefix="megakernel_experts.") == {}
    parent.load_state_dict(checkpoint, strict=True)

    torch.testing.assert_close(routed_fc1, next(iter(routed_checkpoint.values())))
    torch.testing.assert_close(routed_fc2, tuple(routed_checkpoint.values())[1])
    torch.testing.assert_close(shared_fc1, checkpoint["shared_experts.linear_fc1.weight"])
    torch.testing.assert_close(shared_fc2, checkpoint["shared_experts.linear_fc2.weight"])
    if gated:
        torch.testing.assert_close(gate_weight, checkpoint["shared_experts.gate_weight"])
        assert mok.shared_output_gate_weight is shared.gate_weight
    assert mok._routed_weight_view_cache is None
    assert mok._split_main_grad_descriptor_cache is None
    assert mok.is_first_microbatch


def _mock_backend(monkeypatch, *, single_grouped=False, gated=True, mxfp8=False):
    """Build the actual adapter while keeping these contract tests independent of MOK."""
    if mxfp8:
        from megatron.core import fp8_utils

        monkeypatch.setattr(fp8_utils, "te_post_all_gather_processing", object())
    package = ModuleType("mok")
    functional = ModuleType("mok.functional")
    functional.MoKConfig = lambda **kwargs: SimpleNamespace(**kwargs)
    package.functional = functional
    package.ops = SimpleNamespace(make_routed_d_weight_storage_table=lambda _: object())
    monkeypatch.setitem(sys.modules, "mok", package)
    monkeypatch.setitem(sys.modules, "mok.functional", functional)
    experts = torch.nn.Module()
    experts.linear_fc1 = torch.nn.Module()
    experts.linear_fc2 = torch.nn.Module()
    for linear, shape in ((experts.linear_fc1, (8, 8)), (experts.linear_fc2, (8, 4))):
        linear.single_grouped_weight = single_grouped
        if single_grouped:
            linear.register_parameter(
                "weight", torch.nn.Parameter(torch.zeros((2, *shape), dtype=torch.bfloat16))
            )
        else:
            for index in range(2):
                linear.register_parameter(
                    f"weight{index}", torch.nn.Parameter(torch.zeros(shape, dtype=torch.bfloat16))
                )
    gate = torch.nn.Parameter(torch.zeros((1, 8), dtype=torch.bfloat16)) if gated else None
    shared = _shared_module(
        torch.nn.Parameter(torch.zeros((8, 8), dtype=torch.bfloat16)),
        torch.nn.Parameter(torch.zeros((8, 4), dtype=torch.bfloat16)),
        gate,
    )
    config = SimpleNamespace(
        gradient_accumulation_fusion=True,
        moe_mlp_glu_interleave_size=None,
        moe_shared_expert_glu_interleave_size=None,
        moe_pad_expert_input_to_capacity=False,
        moe_shared_expert_gate=gated,
        hidden_size=8,
        moe_ffn_hidden_size=4,
        moe_shared_expert_intermediate_size=4,
        moe_router_topk=2,
        activation_func_clamp_value=None,
        fp8="hybrid" if mxfp8 else None,
        fp8_recipe="mxfp8",
        fp8_param=mxfp8,
        moe_single_grouped_weight=single_grouped,
        moe_megakernel_backend_config=None,
    )
    module = mok_backend.MoKMegakernel(config, object(), experts, shared, 2)
    return module, shared, functional


def test_register_shared_output_gate_rejects_non_bf16_parameter(monkeypatch):
    module, shared, _ = _mock_backend(monkeypatch)
    shared.gate_weight = torch.nn.Parameter(torch.zeros((1, 8), dtype=torch.float32))
    with pytest.raises(RuntimeError, match="native contiguous BF16 Parameter"):
        module._register_shared_weights(shared, use_output_gate=True)


@pytest.mark.parametrize(
    "mxfp8, single_grouped, gated, main_grad_dtype",
    [
        pytest.param(False, False, True, torch.float32, id="bf16-split-g-fp32"),
        pytest.param(True, False, True, torch.bfloat16, id="mxfp8-split-g-bf16"),
        pytest.param(True, True, True, torch.float32, id="mxfp8-single-g-fp32"),
        pytest.param(False, True, False, torch.bfloat16, id="bf16-single-u-bf16"),
    ],
)
def test_gate_autograd_accumulation_and_ddp_hooks(
    monkeypatch, single_grouped, gated, mxfp8, main_grad_dtype
):
    from megatron.core.distributed import distributed_data_parallel as ddp_module

    module, shared, functional = _mock_backend(
        monkeypatch, single_grouped=single_grouped, gated=gated, mxfp8=mxfp8
    )
    gate = shared.gate_weight
    assert module.shared_output_gate_weight is gate
    parameters = tuple(module.parameters())
    for parameter in parameters:
        parameter.main_grad = torch.zeros_like(parameter, dtype=main_grad_dtype)
    expected = torch.full((1, 8), 0.25, dtype=main_grad_dtype)
    if gated:
        assert gate.dtype == torch.bfloat16
        with torch.no_grad():
            gate.fill_(3.0)  # Expose accidental accumulation of the hook-only sentinel.
        gate_main_grad = gate.main_grad
        gate_main_grad.copy_(expected)
    gate_contribution = torch.full((1, 8), 0.1234, dtype=torch.float32)
    # Placeholder views support MXFP8 slicing; this test checks gate wiring, not routed math.
    view = (None,) * 4
    monkeypatch.setattr(module, "quantized_routed_weights", lambda: (view, view))
    functional.get_workspace = functional.build_schedule = lambda *args, **kwargs: object()
    gate_keys = {"shared_output_gate_weight"} if gated else set()

    def forward(*args, **kwargs):
        assert set(kwargs) == {"swiglu_limit"} | gate_keys
        assert kwargs.get("shared_output_gate_weight") is gate
        return args[3].clone(), object()

    def backward(*args, **kwargs):
        keys = {"swiglu_limit", "main_grads", "main_grad_storage_tables"} | gate_keys
        if gated:
            keys.add("shared_output_gate_main_grad")
        assert set(kwargs) == keys
        assert kwargs.get("shared_output_gate_weight") is gate
        gate_grad = kwargs.get("shared_output_gate_main_grad")
        if gated:
            assert gate_grad is gate_main_grad and gate_grad.dtype == main_grad_dtype
            gate_grad.add_(gate_contribution)
        grads = (torch.ones_like(args[5]), torch.ones_like(args[6]), *kwargs["main_grads"])
        # Preserve the legacy ungated eight-item API.
        return (*grads, gate_grad) if gated else grads

    functional.forward = forward
    functional.backward = backward
    x = torch.zeros((2, 2, 8), dtype=torch.bfloat16, requires_grad=True)
    probs = torch.zeros((4, 2), dtype=torch.float32, requires_grad=True)
    experts = torch.zeros((4, 2), dtype=torch.int32)
    monkeypatch.setattr(mok_backend, "routing_map_to_mok_inputs", lambda *_: (probs, experts))
    monkeypatch.setattr(ddp_module, "is_graph_capturing", lambda: False)
    waited, ready = [], []
    bucket = SimpleNamespace(register_grad_ready=lambda param, force: ready.append((param, force)))
    ddp = SimpleNamespace(
        use_forward_hook=True,
        param_to_bucket_group={gate: bucket} if gated else {},
        _finish_param_sync_for_bucket_group=lambda group: waited.append(group),
        ddp_config=SimpleNamespace(overlap_grad_reduce=True),
        force_all_reduce=False,
    )
    handles = [
        module.register_forward_pre_hook(
            ddp_module.DistributedDataParallel._make_forward_pre_hook(ddp)
        )
    ]
    if gated:
        accumulator = gate.expand_as(gate).grad_fn.next_functions[0][0]
        handles.append(
            accumulator.register_hook(
                ddp_module.DistributedDataParallel._make_backward_post_hook(ddp, gate)
            )
        )
    try:
        for iteration in range(3):
            x.grad = probs.grad = None
            for parameter in parameters:
                parameter.grad = None
                parameter.grad_added_to_main_grad = False
            if gated and iteration == 2:
                gate_main_grad.zero_()  # The same buffer must also work after optimizer zeroing.
                expected.zero_()
            output = module(x, probs, experts)
            assert output.shape == x.shape
            bridge = output.grad_fn.next_functions[0][0]
            output.backward(torch.ones_like(output))
            torch.testing.assert_close(x.grad, torch.ones_like(x))
            torch.testing.assert_close(probs.grad, torch.ones_like(probs))
            assert all(param.grad_added_to_main_grad for param in parameters)
            if gated:
                assert gate.grad is None and gate.main_grad is gate_main_grad
                expected = (expected.float() + gate_contribution).to(main_grad_dtype)
                torch.testing.assert_close(gate_main_grad, expected, rtol=0, atol=0)
                torch.testing.assert_close(gate, torch.full_like(gate, 3.0))
            assert bridge.module is None and bridge.forward_context is None
    finally:
        for handle in handles:
            handle.remove()
    assert waited == ([bucket] * 3 if gated else [])
    assert ready == ([(gate, False)] * 3 if gated else [])


def test_runtime_requires_output_gate_main_grad(monkeypatch):
    module, shared, functional = _mock_backend(monkeypatch)
    parameters = module.autograd_routed_parameters + (
        shared.linear_fc1.weight,
        shared.linear_fc2.weight,
    )
    for parameter in parameters:
        parameter.main_grad = torch.zeros_like(parameter, dtype=torch.float32)
    x = torch.zeros((4, 8), dtype=torch.bfloat16)
    probs = torch.zeros((4, 2), dtype=torch.float32)
    ctx = SimpleNamespace(
        module=module,
        routed_weight_views=(module.routed_fc1_parameters[0], module.routed_fc2_parameters[0]),
        saved_tensors=(x, probs, shared.gate_weight, *parameters),
    )
    functional.backward = lambda *args, **kwargs: pytest.fail("invalid main_grad reached MOK")
    with pytest.raises(RuntimeError, match="DDP to assign param.main_grad"):
        mok_runtime._MoKAutograd.backward(ctx, torch.ones_like(x))
