# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.transformer.moe.megakernel import parameter_bridge
from megatron.core.transformer.moe.megakernel.mok import backend as mok_megakernel
from megatron.core.transformer.moe.megakernel.mok import runtime as mok_runtime
from megatron.core.transformer.moe.megakernel.mok import weights as mok_weights


def _parameter_with_main_grad(shape=(4, 8)):
    param = torch.nn.Parameter(torch.zeros(shape, dtype=torch.bfloat16))
    param.main_grad = torch.zeros(shape, dtype=torch.float32)
    param.grad_added_to_main_grad = False
    return param


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


def test_dummy_weight_gradient_reuses_parameter_storage():
    param = _parameter_with_main_grad()

    dummy = parameter_bridge.dummy_weight_gradient(param)

    assert dummy.shape == param.shape
    assert dummy.dtype == param.dtype
    assert dummy.data_ptr() == param.data_ptr()
    assert not dummy.requires_grad


def test_finish_weight_gradient_marks_ready_without_accumulating(monkeypatch):
    param = _parameter_with_main_grad()
    param.main_grad.fill_(0.25)
    dummy = torch.empty_like(param)
    monkeypatch.setattr(parameter_bridge, "dummy_weight_gradient", lambda _: dummy)

    actual = parameter_bridge.finish_weight_gradient(param)

    assert actual is dummy
    torch.testing.assert_close(param.main_grad, torch.full_like(param.main_grad, 0.25))
    assert param.grad_added_to_main_grad


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


def test_swizzle_mxfp8_scale_matches_tcgen05_lane_layout():
    rows = columns = 128
    logical = torch.arange(rows * (columns // 32), dtype=torch.int32).to(torch.uint8)
    logical = logical.reshape(1, rows, columns // 32)

    actual = mok_weights._swizzle_mxfp8_scale(logical, rows=rows, columns=columns)

    assert actual.shape == (1, 1, 32, 16)
    expected = torch.empty_like(actual)
    for lane in range(32):
        for row_group in range(4):
            for column_scale in range(4):
                expected[0, 0, lane, row_group * 4 + column_scale] = logical[
                    0, row_group * 32 + lane, column_scale
                ]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


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


def test_mxfp8_backward_views_keep_native_columnwise_payload_zero_copy():
    num_experts, rows, columns = 1, 256, 128
    row_data = torch.empty((num_experts, rows, columns))
    row_scale = torch.zeros((num_experts, rows, columns // 32), dtype=torch.uint8)
    column_data = torch.empty_like(row_data)
    column_scale = torch.zeros((num_experts, columns, rows // 32), dtype=torch.uint8)
    source = mok_weights._NativeExpertWeightSource(
        storage_layout=mok_weights._ExpertWeightStorageLayout.SINGLE_GROUPED,
        num_experts=num_experts,
        rows=rows,
        columns=columns,
        row_data=(row_data,),
        row_scales=(row_scale,),
        column_data=(column_data,),
        column_scales=(column_scale,),
    )

    actual = mok_weights._prepare_mok_weight(source)

    assert actual[0].data_ptr() == row_data.data_ptr()
    assert actual[2].data_ptr() == column_data.data_ptr()
    assert actual[4] is True


def test_mxfp8_scale_layout_cache_refreshes_once_per_optimizer_iteration(monkeypatch):
    module = mok_megakernel.MoKMegakernel.__new__(mok_megakernel.MoKMegakernel)
    torch.nn.Module.__init__(module)
    module.native_single_grouped_weights = True
    module.use_mxfp8_weights = True
    module.is_first_microbatch = True
    module._prepared_routed_weight_cache = None
    module.routed_fc1_weight = object()
    module.routed_fc2_weight = object()
    module.num_local_experts = 2
    module.intermediate_size = 128
    module.hidden_size = 256

    extract_calls = []
    prepare_calls = []

    def fake_extract(params, **kwargs):
        extract_calls.append((params, kwargs))
        return kwargs["name"]

    def fake_prepare(source):
        prepare_calls.append(source)
        return object()

    monkeypatch.setattr(
        mok_megakernel, "_extract_native_expert_weight_source", fake_extract
    )
    monkeypatch.setattr(mok_megakernel, "_prepare_mok_weight", fake_prepare)

    first = module.quantized_routed_weights()
    second = module.quantized_routed_weights()

    assert second is first
    assert [kwargs["name"] for _, kwargs in extract_calls] == [
        "routed FC1",
        "routed FC2",
    ]
    assert prepare_calls == ["routed FC1", "routed FC2"]

    module.is_first_microbatch = True
    third = module.quantized_routed_weights()

    assert third is not first
    assert [kwargs["name"] for _, kwargs in extract_calls] == [
        "routed FC1",
        "routed FC2",
    ] * 2
    assert prepare_calls == ["routed FC1", "routed FC2"] * 2


def _split_module(*, use_mxfp8_weights):
    module = mok_megakernel.MoKMegakernel.__new__(mok_megakernel.MoKMegakernel)
    torch.nn.Module.__init__(module)
    module.native_single_grouped_weights = False
    module.use_mxfp8_weights = use_mxfp8_weights
    module.is_first_microbatch = True
    module._prepared_routed_weight_cache = None
    module.num_local_experts = 2
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
    class Storage:
        pass

    public = Storage()
    public.rowwise_data = torch.tensor([1])
    assert mok_weights._parameter_storage_attr(public, "rowwise_data") is public.rowwise_data

    private = Storage()
    private._rowwise_data = torch.tensor([2])
    assert mok_weights._parameter_storage_attr(private, "rowwise_data") is private._rowwise_data

    wrapped = Storage()
    wrapped.data = private
    assert mok_weights._parameter_storage_attr(wrapped, "rowwise_data") is private._rowwise_data


def test_bf16_split_descriptors_are_cached(monkeypatch):
    module = _split_module(use_mxfp8_weights=False)
    extract_calls = []
    prepare_calls = []

    def fake_extract(params, **kwargs):
        extract_calls.append((params, kwargs))
        return kwargs["name"]

    def fake_prepare(source):
        prepare_calls.append(source)
        return object()

    monkeypatch.setattr(
        mok_megakernel, "_extract_native_expert_weight_source", fake_extract
    )
    monkeypatch.setattr(mok_megakernel, "_prepare_mok_weight", fake_prepare)

    first = module.quantized_routed_weights()
    second = module.quantized_routed_weights()

    assert second is first
    assert [
        (kwargs["rows"], kwargs["columns"], kwargs["use_mxfp8"])
        for _, kwargs in extract_calls
    ] == [(8, 8, False), (8, 4, False)]
    assert prepare_calls == ["routed FC1", "routed FC2"]
    assert not module.is_first_microbatch


def test_mxfp8_split_scale_and_descriptor_cache_refreshes_per_iteration(monkeypatch):
    module = _split_module(use_mxfp8_weights=True)
    extract_calls = []
    prepare_calls = []
    refresh_calls = []

    def fake_extract(params, **kwargs):
        extract_calls.append((params, kwargs))
        return kwargs["name"]

    def fake_prepare(source):
        prepare_calls.append(source)
        return object()

    def fake_refresh(prepared, source):
        refresh_calls.append((prepared, source))

    monkeypatch.setattr(
        mok_megakernel, "_extract_native_expert_weight_source", fake_extract
    )
    monkeypatch.setattr(mok_megakernel, "_prepare_mok_weight", fake_prepare)
    monkeypatch.setattr(mok_megakernel, "_refresh_prepared_mok_weight", fake_refresh)

    first = module.quantized_routed_weights()
    second = module.quantized_routed_weights()

    assert second is first
    assert prepare_calls == ["routed FC1", "routed FC2"]
    assert not refresh_calls

    module.is_first_microbatch = True
    third = module.quantized_routed_weights()

    assert third is first
    assert [kwargs["name"] for _, kwargs in extract_calls] == [
        "routed FC1",
        "routed FC2",
    ] * 2
    assert refresh_calls == [(first[0], "routed FC1"), (first[1], "routed FC2")]


def test_mxfp8_split_scales_use_per_expert_descriptor_tables(monkeypatch):
    from mok import ops

    from megatron.core import fp8_utils

    rows, columns = 256, 128

    class SplitParam:
        pass

    params = []
    for expert in range(2):
        param = SplitParam()
        param.rowwise_data = torch.empty((rows, columns), dtype=torch.float8_e4m3fn)
        param.columnwise_data = torch.empty_like(param.rowwise_data)
        param._rowwise_scale_inv = torch.full(
            (rows, columns // 32), expert + 1, dtype=torch.uint8
        )
        param._columnwise_scale_inv = torch.full(
            (rows // 32, columns), expert + 3, dtype=torch.uint8
        )
        params.append(param)

    monkeypatch.setattr(fp8_utils, "is_mxfp8tensor", lambda param: True)
    monkeypatch.setattr(
        mok_weights,
        "_storage_view",
        lambda storage, shape, **kwargs: storage.view(shape),
    )
    monkeypatch.setattr(
        torch,
        "stack",
        lambda *args, **kwargs: pytest.fail("split MXFP8 scales must not be stacked"),
    )

    scale_table_inputs = []
    monkeypatch.setattr(
        ops,
        "make_routed_weight_storage_table_mxfp8",
        lambda tensors: torch.tensor([len(tensors)], dtype=torch.uint8),
    )

    def fake_scale_table(tensors):
        scale_table_inputs.append(tuple(tensors))
        return torch.tensor([len(tensors)], dtype=torch.uint8)

    monkeypatch.setattr(ops, "make_routed_scale_storage_table", fake_scale_table)

    source = mok_weights._extract_native_expert_weight_source(
        tuple(params),
        storage_layout=mok_weights._ExpertWeightStorageLayout.PER_EXPERT,
        num_experts=2,
        rows=rows,
        columns=columns,
        use_mxfp8=True,
        name="routed FC1",
    )
    prepared = mok_weights._prepare_mok_weight(source)

    assert len(scale_table_inputs) == 2
    assert all(
        actual is expected
        for actual, expected in zip(
            prepared.scale_tensors, scale_table_inputs[0], strict=True
        )
    )
    assert all(
        actual is expected
        for actual, expected in zip(
            prepared.transposed_scale_tensors, scale_table_inputs[1], strict=True
        )
    )
    assert tuple(prepared.scale.shape) == (rows // 128, columns // 128, 32, 16)
    assert tuple(prepared.transposed_scale.shape) == (
        columns // 128,
        rows // 128,
        32,
        16,
    )
    assert prepared.scale_tensors[0].data_ptr() != prepared.scale_tensors[1].data_ptr()
    assert (
        prepared.transposed_scale_tensors[0].data_ptr()
        != prepared.transposed_scale_tensors[1].data_ptr()
    )

    row_ptrs = tuple(tensor.data_ptr() for tensor in prepared.scale_tensors)
    column_ptrs = tuple(
        tensor.data_ptr() for tensor in prepared.transposed_scale_tensors
    )
    row_table = prepared.scale_storage_table
    column_table = prepared.transposed_scale_storage_table
    params[0]._rowwise_scale_inv.fill_(9)
    params[1]._columnwise_scale_inv.fill_(11)

    refreshed_source = mok_weights._extract_native_expert_weight_source(
        tuple(params),
        storage_layout=mok_weights._ExpertWeightStorageLayout.PER_EXPERT,
        num_experts=2,
        rows=rows,
        columns=columns,
        use_mxfp8=True,
        name="routed FC1",
    )
    mok_weights._refresh_prepared_mok_weight(prepared, refreshed_source)

    assert tuple(tensor.data_ptr() for tensor in prepared.scale_tensors) == row_ptrs
    assert (
        tuple(tensor.data_ptr() for tensor in prepared.transposed_scale_tensors)
        == column_ptrs
    )
    assert prepared.scale_storage_table is row_table
    assert prepared.transposed_scale_storage_table is column_table
    for expert, param in enumerate(params):
        expected_row = mok_weights._swizzle_mxfp8_scale(
            param._rowwise_scale_inv.unsqueeze(0), rows=rows, columns=columns
        )
        expected_column = mok_weights._swizzle_mxfp8_scale(
            param._columnwise_scale_inv.transpose(-2, -1).unsqueeze(0),
            rows=columns,
            columns=rows,
        )
        torch.testing.assert_close(
            prepared.scale_tensors[expert], expected_row, rtol=0, atol=0
        )
        torch.testing.assert_close(
            prepared.transposed_scale_tensors[expert], expected_column, rtol=0, atol=0
        )


def test_native_single_grouped_bf16_views_alias_authoritative_parameters():
    num_experts, intermediate_size, hidden_size = 2, 4, 8
    fc1 = torch.nn.Parameter(
        torch.randn(
            num_experts, 2 * intermediate_size, hidden_size, dtype=torch.bfloat16
        )
    )
    fc2 = torch.nn.Parameter(
        torch.randn(num_experts, hidden_size, intermediate_size, dtype=torch.bfloat16)
    )

    storage_layout = mok_weights._ExpertWeightStorageLayout.SINGLE_GROUPED
    fc1_source = mok_weights._extract_native_expert_weight_source(
        (fc1,),
        storage_layout=storage_layout,
        num_experts=num_experts,
        rows=2 * intermediate_size,
        columns=hidden_size,
        use_mxfp8=False,
        name="routed FC1",
    )
    fc2_source = mok_weights._extract_native_expert_weight_source(
        (fc2,),
        storage_layout=storage_layout,
        num_experts=num_experts,
        rows=hidden_size,
        columns=intermediate_size,
        use_mxfp8=False,
        name="routed FC2",
    )

    assert mok_weights._prepare_mok_weight(fc1_source) is fc1
    assert mok_weights._prepare_mok_weight(fc2_source) is fc2


def test_native_single_grouped_bf16_views_use_rowwise_storage(monkeypatch):
    num_experts, intermediate_size, hidden_size = 2, 4, 8
    fc1 = torch.nn.Parameter(
        torch.randn(
            num_experts, 2 * intermediate_size, hidden_size, dtype=torch.bfloat16
        )
    )
    fc2 = torch.nn.Parameter(
        torch.randn(num_experts, hidden_size, intermediate_size, dtype=torch.bfloat16)
    )
    fc1.rowwise_data = torch.empty_like(fc1)
    fc2.rowwise_data = torch.empty_like(fc2)
    storage_calls = []

    def fake_storage_view(storage, shape, *, dtype, name):
        storage_calls.append((storage, shape, dtype, name))
        return storage.view(shape)

    monkeypatch.setattr(mok_weights, "_storage_view", fake_storage_view)

    storage_layout = mok_weights._ExpertWeightStorageLayout.SINGLE_GROUPED
    fc1_source = mok_weights._extract_native_expert_weight_source(
        (fc1,),
        storage_layout=storage_layout,
        num_experts=num_experts,
        rows=2 * intermediate_size,
        columns=hidden_size,
        use_mxfp8=False,
        name="routed FC1",
    )
    fc2_source = mok_weights._extract_native_expert_weight_source(
        (fc2,),
        storage_layout=storage_layout,
        num_experts=num_experts,
        rows=hidden_size,
        columns=intermediate_size,
        use_mxfp8=False,
        name="routed FC2",
    )
    fc1_view = mok_weights._prepare_mok_weight(fc1_source)
    fc2_view = mok_weights._prepare_mok_weight(fc2_source)

    assert fc1_view.data_ptr() == fc1.rowwise_data.data_ptr()
    assert fc2_view.data_ptr() == fc2.rowwise_data.data_ptr()
    assert [call[1] for call in storage_calls] == [
        (num_experts, 2 * intermediate_size, hidden_size),
        (num_experts, hidden_size, intermediate_size),
    ]


def _shared_module(fc1_weight, fc2_weight):
    shared = torch.nn.Module()
    shared.linear_fc1 = torch.nn.Module()
    shared.linear_fc2 = torch.nn.Module()
    shared.linear_fc1.register_parameter("weight", fc1_weight)
    shared.linear_fc2.register_parameter("weight", fc2_weight)
    return shared


def _shared_adapter(intermediate_size=2, hidden_size=3):
    module = mok_megakernel.MoKMegakernel.__new__(mok_megakernel.MoKMegakernel)
    torch.nn.Module.__init__(module)
    module.intermediate_size = intermediate_size
    module.hidden_size = hidden_size
    return module


def test_register_shared_weights_reuses_native_bf16_combined_parameters():
    fc1 = torch.nn.Parameter(torch.randn((4, 3), dtype=torch.bfloat16))
    fc2 = torch.nn.Parameter(torch.randn((3, 2), dtype=torch.bfloat16))
    shared = _shared_module(fc1, fc2)
    module = _shared_adapter()

    module._register_shared_weights(shared)

    assert module.shared_fc1_weight is fc1
    assert module.shared_fc2_weight is fc2
    assert shared.linear_fc1.weight is fc1
    assert shared.linear_fc2.weight is fc2


def test_register_shared_weights_rejects_non_bf16_parameters():
    fc1 = torch.nn.Parameter(torch.randn((4, 3), dtype=torch.float32))
    fc2 = torch.nn.Parameter(torch.randn((3, 2), dtype=torch.float32))
    shared = _shared_module(fc1, fc2)
    module = _shared_adapter()

    with pytest.raises(RuntimeError, match="native BF16"):
        module._register_shared_weights(shared)


def test_combined_shared_main_grad_uses_canonical_fc1_fc2_buffers(monkeypatch):
    from mok import ops

    module = _split_module(use_mxfp8_weights=False)
    module._split_main_grad_descriptor_cache = None
    for param in module.autograd_routed_parameters:
        param.main_grad = torch.zeros_like(param, dtype=torch.float32)
    module.shared_fc1_weight = _parameter_with_main_grad((8, 8))
    module.shared_fc2_weight = _parameter_with_main_grad((8, 4))

    monkeypatch.setattr(ops, "make_routed_d_weight_storage_table", lambda grads: tuple(grads))
    main_grads, _ = module.main_grad_arguments()

    shared_fc1_grad = module.shared_fc1_weight.main_grad
    assert main_grads[0].untyped_storage().data_ptr() == (
        shared_fc1_grad.untyped_storage().data_ptr()
    )
    assert main_grads[0] is shared_fc1_grad
    assert main_grads[2] is module.shared_fc2_weight.main_grad


@pytest.mark.parametrize("single_grouped", [False, True])
def test_mok_sharded_state_dict_emits_no_parameter_aliases(single_grouped):
    module = _split_module(use_mxfp8_weights=False)
    module.native_single_grouped_weights = single_grouped

    assert module.sharded_state_dict(prefix="layers.0.mlp.megakernel_experts.") == {}
    assert module.state_dict(prefix="layers.0.mlp.megakernel_experts.") == {}


@pytest.mark.parametrize("single_grouped", [False, True])
def test_native_checkpoint_load_uses_only_canonical_weights(single_grouped):
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

    mok = mok_megakernel.MoKMegakernel.__new__(mok_megakernel.MoKMegakernel)
    torch.nn.Module.__init__(mok)
    mok.native_single_grouped_weights = single_grouped
    mok._prepared_routed_weight_cache = object()
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

    parent.load_state_dict(checkpoint, strict=True)

    torch.testing.assert_close(routed_fc1, next(iter(routed_checkpoint.values())))
    torch.testing.assert_close(routed_fc2, tuple(routed_checkpoint.values())[1])
    torch.testing.assert_close(shared_fc1, checkpoint["shared_experts.linear_fc1.weight"])
    torch.testing.assert_close(shared_fc2, checkpoint["shared_experts.linear_fc2.weight"])
    assert mok._prepared_routed_weight_cache is None
    assert mok._split_main_grad_descriptor_cache is None
    assert mok.is_first_microbatch
