# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.transformer.moe import mok_megakernel


def _parameter_with_main_grad(shape=(4, 8)):
    param = torch.nn.Parameter(torch.zeros(shape, dtype=torch.bfloat16))
    param.main_grad = torch.zeros(shape, dtype=torch.float32)
    param.grad_added_to_main_grad = False
    return param


def test_mxfp8_compatibility_warning_is_emitted_once(monkeypatch, capsys):
    monkeypatch.setattr(
        mok_megakernel, "_MOK_MXFP8_COMPAT_WARNING_EMITTED", False
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    mok_megakernel._warn_mxfp8_compatibility_fallback()
    mok_megakernel._warn_mxfp8_compatibility_fallback()

    output = capsys.readouterr().out
    assert output.count("WARNING: MOK MXFP8") == 1
    assert "other eligible TE parameters retain" in output
    assert "substantially more GPU memory" in output


def test_dummy_weight_gradient_reuses_parameter_storage():
    param = _parameter_with_main_grad()

    dummy = mok_megakernel._dummy_weight_gradient(param)

    assert dummy.shape == param.shape
    assert dummy.dtype == param.dtype
    assert dummy.data_ptr() == param.data_ptr()
    assert not dummy.requires_grad


def test_accumulate_weight_gradient_adds_to_main_grad_and_returns_dummy(monkeypatch):
    param = _parameter_with_main_grad()
    grad = torch.full_like(param, 0.5)
    dummy = torch.empty_like(param)
    monkeypatch.setattr(mok_megakernel, "_dummy_weight_gradient", lambda _: dummy)

    actual = mok_megakernel._accumulate_weight_gradient(param, grad)
    mok_megakernel._accumulate_weight_gradient(param, grad)

    assert actual is dummy
    torch.testing.assert_close(param.main_grad, torch.ones_like(param.main_grad))
    assert param.grad_added_to_main_grad


def test_accumulate_weight_gradient_requires_main_grad():
    param = torch.nn.Parameter(torch.zeros((4, 8), dtype=torch.bfloat16))

    with pytest.raises(RuntimeError, match="param.main_grad"):
        mok_megakernel._accumulate_weight_gradient(param, torch.zeros_like(param))


def test_accumulate_weight_gradient_rejects_shape_mismatch():
    param = _parameter_with_main_grad()

    with pytest.raises(RuntimeError, match="shape mismatch"):
        mok_megakernel._accumulate_weight_gradient(
            param, torch.zeros((2, 16), dtype=torch.bfloat16)
        )


def test_finish_weight_gradient_marks_ready_without_accumulating(monkeypatch):
    param = _parameter_with_main_grad()
    param.main_grad.fill_(0.25)
    dummy = torch.empty_like(param)
    monkeypatch.setattr(mok_megakernel, "_dummy_weight_gradient", lambda _: dummy)

    actual = mok_megakernel._finish_weight_gradient(param)

    assert actual is dummy
    torch.testing.assert_close(param.main_grad, torch.full_like(param.main_grad, 0.25))
    assert param.grad_added_to_main_grad


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_main_grad_buffer_accepts_supported_dtype(dtype):
    param = torch.nn.Parameter(torch.zeros((4, 8), dtype=torch.bfloat16))
    param.main_grad = torch.zeros((4, 8), dtype=dtype)

    assert mok_megakernel._main_grad_buffer(param) is param.main_grad


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_main_grad_buffer_rejects_unsupported_dtype(dtype):
    param = torch.nn.Parameter(torch.zeros((4, 8), dtype=torch.bfloat16))
    param.main_grad = torch.zeros((4, 8), dtype=dtype)

    with pytest.raises(RuntimeError, match="FP32 or BF16"):
        mok_megakernel._main_grad_buffer(param)


def test_swizzle_mxfp8_scale_matches_tcgen05_lane_layout():
    rows = columns = 128
    logical = torch.arange(rows * (columns // 32), dtype=torch.int32).to(torch.uint8)
    logical = logical.reshape(1, rows, columns // 32)

    actual = mok_megakernel._swizzle_mxfp8_scale(
        logical, rows=rows, columns=columns
    )

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
    expected = mok_megakernel._swizzle_mxfp8_scale(
        logical, rows=rows, columns=columns
    )
    output = torch.empty_like(expected)
    output_ptr = output.data_ptr()

    actual = mok_megakernel._swizzle_mxfp8_scale(
        logical, rows=rows, columns=columns, out=output
    )

    assert actual is output
    assert actual.data_ptr() == output_ptr
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_mxfp8_backward_views_keep_native_columnwise_payload_zero_copy():
    num_experts, rows, columns = 1, 256, 128
    row_data = torch.empty((num_experts, rows, columns))
    row_scale = torch.zeros((num_experts, rows, columns // 32), dtype=torch.uint8)
    column_data = torch.empty_like(row_data)
    column_scale = torch.zeros(
        (num_experts, columns, rows // 32), dtype=torch.uint8
    )
    native = (row_data, row_scale, column_data, column_scale, True)

    actual = mok_megakernel._mok_mxfp8_backward_weight_views(
        native,
        rows=rows,
        columns=columns,
    )

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
    module.routed_down_weight = object()
    module.num_local_experts = 2
    module.intermediate_size = 128
    module.hidden_size = 256

    native_calls = []
    prepare_calls = []

    def fake_native_views(*args, **kwargs):
        del args, kwargs
        native_calls.append(True)
        gate = (object(),)
        down = (object(),)
        return gate, gate, down

    def fake_prepare(native, *, rows, columns):
        prepare_calls.append((rows, columns))
        return (native[0], object(), object(), object(), True)

    monkeypatch.setattr(
        mok_megakernel, "_native_single_grouped_weight_views", fake_native_views
    )
    monkeypatch.setattr(
        mok_megakernel, "_mok_mxfp8_backward_weight_views", fake_prepare
    )

    first = module.quantized_routed_weights()
    second = module.quantized_routed_weights()

    assert second is first
    assert first[0] is first[1]
    assert len(native_calls) == 1
    assert prepare_calls == [(256, 256), (256, 128)]

    module.is_first_microbatch = True
    third = module.quantized_routed_weights()

    assert third is not first
    assert third[0] is third[1]
    assert len(native_calls) == 2
    assert prepare_calls == [(256, 256), (256, 128)] * 2


def _split_module(*, use_mxfp8_weights):
    module = mok_megakernel.MoKMegakernel.__new__(mok_megakernel.MoKMegakernel)
    torch.nn.Module.__init__(module)
    module.native_single_grouped_weights = False
    module.use_mxfp8_weights = use_mxfp8_weights
    module.is_first_microbatch = True
    module._prepared_routed_weight_cache = None
    module.intermediate_size = 4
    module.hidden_size = 8
    module._routed_fc1_parameter_names = ("routed_fc1_weight0", "routed_fc1_weight1")
    module._routed_down_parameter_names = ("routed_down_weight0", "routed_down_weight1")
    for name in module._routed_fc1_parameter_names:
        module.register_parameter(
            name, torch.nn.Parameter(torch.randn((8, 8), dtype=torch.bfloat16))
        )
    for name in module._routed_down_parameter_names:
        module.register_parameter(
            name, torch.nn.Parameter(torch.randn((8, 4), dtype=torch.bfloat16))
        )
    return module


def test_parameter_storage_attr_supports_public_and_private_te_fields():
    class Storage:
        pass

    public = Storage()
    public.rowwise_data = torch.tensor([1])
    assert (
        mok_megakernel._parameter_storage_attr(public, "rowwise_data")
        is public.rowwise_data
    )

    private = Storage()
    private._rowwise_data = torch.tensor([2])
    assert (
        mok_megakernel._parameter_storage_attr(private, "rowwise_data")
        is private._rowwise_data
    )

    wrapped = Storage()
    wrapped.data = private
    assert (
        mok_megakernel._parameter_storage_attr(wrapped, "rowwise_data")
        is private._rowwise_data
    )


def test_bf16_split_descriptors_are_cached(monkeypatch):
    module = _split_module(use_mxfp8_weights=False)
    calls = []

    def fake_split(params, *, rows, columns, use_mxfp8):
        calls.append((params, rows, columns, use_mxfp8))
        return object()

    monkeypatch.setattr(mok_megakernel, "_native_split_weight_view", fake_split)

    first = module.quantized_routed_weights()
    second = module.quantized_routed_weights()

    assert second is first
    assert first[0] is first[1]
    assert [(rows, columns, use_mxfp8) for _, rows, columns, use_mxfp8 in calls] == [
        (8, 8, False),
        (8, 4, False),
    ]
    assert not module.is_first_microbatch


def test_mxfp8_split_scale_and_descriptor_cache_refreshes_per_iteration(monkeypatch):
    module = _split_module(use_mxfp8_weights=True)
    build_calls = []
    refresh_calls = []

    def fake_split(params, *, rows, columns, use_mxfp8):
        build_calls.append((params, rows, columns, use_mxfp8))
        return object()

    def fake_refresh(prepared, params, *, rows, columns):
        refresh_calls.append((prepared, params, rows, columns))

    monkeypatch.setattr(mok_megakernel, "_native_split_weight_view", fake_split)
    monkeypatch.setattr(
        mok_megakernel, "_refresh_native_split_weight_scales", fake_refresh
    )

    first = module.quantized_routed_weights()
    second = module.quantized_routed_weights()

    assert second is first
    assert len(build_calls) == 2
    assert not refresh_calls

    module.is_first_microbatch = True
    third = module.quantized_routed_weights()

    assert third is first
    assert third[0] is third[1]
    assert len(build_calls) == 2
    assert [
        (prepared, rows, columns) for prepared, _, rows, columns in refresh_calls
    ] == [(first[0], 8, 8), (first[2], 8, 4)]


def test_mxfp8_split_scales_use_per_expert_descriptor_tables(monkeypatch):
    from megatron.core import fp8_utils
    from mok import ops

    rows, columns = 256, 128

    class SplitParam:
        pass

    params = []
    for expert in range(2):
        param = SplitParam()
        param.rowwise_data = torch.empty(
            (rows, columns), dtype=torch.float8_e4m3fn
        )
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
        mok_megakernel,
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

    prepared = mok_megakernel._native_split_weight_view(
        tuple(params), rows=rows, columns=columns, use_mxfp8=True
    )

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

    mok_megakernel._refresh_native_split_weight_scales(
        prepared, tuple(params), rows=rows, columns=columns
    )

    assert tuple(tensor.data_ptr() for tensor in prepared.scale_tensors) == row_ptrs
    assert (
        tuple(tensor.data_ptr() for tensor in prepared.transposed_scale_tensors)
        == column_ptrs
    )
    assert prepared.scale_storage_table is row_table
    assert prepared.transposed_scale_storage_table is column_table
    for expert, param in enumerate(params):
        expected_row = mok_megakernel._swizzle_mxfp8_scale(
            param._rowwise_scale_inv.unsqueeze(0), rows=rows, columns=columns
        )
        expected_column = mok_megakernel._swizzle_mxfp8_scale(
            param._columnwise_scale_inv.transpose(-2, -1).unsqueeze(0),
            rows=columns,
            columns=rows,
        )
        torch.testing.assert_close(
            prepared.scale_tensors[expert], expected_row, rtol=0, atol=0
        )
        torch.testing.assert_close(
            prepared.transposed_scale_tensors[expert],
            expected_column,
            rtol=0,
            atol=0,
        )


def test_native_single_grouped_bf16_views_alias_authoritative_parameters():
    num_experts, intermediate_size, hidden_size = 2, 4, 8
    fc1 = torch.nn.Parameter(
        torch.randn(
            num_experts, 2 * intermediate_size, hidden_size, dtype=torch.bfloat16
        )
    )
    fc2 = torch.nn.Parameter(
        torch.randn(
            num_experts, hidden_size, intermediate_size, dtype=torch.bfloat16
        )
    )

    gate, up, down = mok_megakernel._native_single_grouped_weight_views(
        fc1,
        fc2,
        num_experts=num_experts,
        intermediate_size=intermediate_size,
        hidden_size=hidden_size,
        use_mxfp8=False,
    )

    assert gate is fc1
    assert up is fc1
    assert down is fc2


def test_native_single_grouped_bf16_views_use_rowwise_storage(monkeypatch):
    num_experts, intermediate_size, hidden_size = 2, 4, 8
    fc1 = torch.nn.Parameter(
        torch.randn(
            num_experts, 2 * intermediate_size, hidden_size, dtype=torch.bfloat16
        )
    )
    fc2 = torch.nn.Parameter(
        torch.randn(
            num_experts, hidden_size, intermediate_size, dtype=torch.bfloat16
        )
    )
    fc1.rowwise_data = torch.empty_like(fc1)
    fc2.rowwise_data = torch.empty_like(fc2)
    storage_calls = []

    def fake_storage_view(storage, shape, *, dtype, name):
        storage_calls.append((storage, shape, dtype, name))
        return storage.view(shape)

    monkeypatch.setattr(mok_megakernel, "_storage_view", fake_storage_view)

    gate, up, down = mok_megakernel._native_single_grouped_weight_views(
        fc1,
        fc2,
        num_experts=num_experts,
        intermediate_size=intermediate_size,
        hidden_size=hidden_size,
        use_mxfp8=False,
    )

    assert gate.data_ptr() == fc1.rowwise_data.data_ptr()
    assert up.data_ptr() == gate.data_ptr()
    assert down.data_ptr() == fc2.rowwise_data.data_ptr()
    assert [call[1] for call in storage_calls] == [
        (num_experts, 2 * intermediate_size, hidden_size),
        (num_experts, hidden_size, intermediate_size),
    ]


def _parameter_with_preserved_init(init_val):
    param = torch.nn.Parameter(torch.zeros(init_val.shape, dtype=torch.bfloat16))
    cleared = []
    param.get_high_precision_init_val = lambda: init_val
    param.clear_high_precision_init_val = lambda: cleared.append(True)
    return param, cleared


def test_import_weights_preserves_reordered_init_for_optimizer(monkeypatch):
    class Stub:
        pass

    monkeypatch.setattr(mok_megakernel, "_debug_tag", lambda *_: None)

    hidden_size = 3
    routed_intermediate = 2
    shared_intermediate = 1
    num_experts = 2

    routed = Stub()
    routed.linear_fc1 = Stub()
    routed.linear_fc2 = Stub()
    routed.linear_fc1.single_grouped_weight = False
    routed.linear_fc2.single_grouped_weight = False

    routed_fc1_init = []
    routed_fc2_init = []
    cleared = []
    for expert_idx in range(num_experts):
        fc1_init = (
            torch.arange(2 * routed_intermediate * hidden_size, dtype=torch.float32)
            .reshape(2 * routed_intermediate, hidden_size)
            .add_(100 * expert_idx + 0.125)
        )
        fc2_init = (
            torch.arange(hidden_size * routed_intermediate, dtype=torch.float32)
            .reshape(hidden_size, routed_intermediate)
            .add_(100 * expert_idx + 0.375)
        )
        fc1_param, fc1_cleared = _parameter_with_preserved_init(fc1_init)
        fc2_param, fc2_cleared = _parameter_with_preserved_init(fc2_init)
        setattr(routed.linear_fc1, f"weight{expert_idx}", fc1_param)
        setattr(routed.linear_fc2, f"weight{expert_idx}", fc2_param)
        routed_fc1_init.append(fc1_init)
        routed_fc2_init.append(fc2_init)
        cleared.extend((fc1_cleared, fc2_cleared))

    shared = Stub()
    shared.linear_fc1 = Stub()
    shared.linear_fc2 = Stub()
    shared_fc1_init = torch.arange(
        2 * shared_intermediate * hidden_size, dtype=torch.float32
    ).reshape(2 * shared_intermediate, hidden_size)
    shared_fc2_init = torch.arange(hidden_size * shared_intermediate, dtype=torch.float32).reshape(
        hidden_size, shared_intermediate
    )
    shared.linear_fc1.weight, shared_fc1_cleared = _parameter_with_preserved_init(shared_fc1_init)
    shared.linear_fc2.weight, shared_fc2_cleared = _parameter_with_preserved_init(shared_fc2_init)
    cleared.extend((shared_fc1_cleared, shared_fc2_cleared))

    module = mok_megakernel.MoKMegakernel.__new__(mok_megakernel.MoKMegakernel)
    torch.nn.Module.__init__(module)
    module.hidden_size = hidden_size
    module.intermediate_size = routed_intermediate
    module.shared_intermediate_size = shared_intermediate
    module.num_local_experts = num_experts
    module._debug_module_index = 0

    module._import_routed_weights(routed)
    module._import_shared_weights(shared)

    expected_routed_gate = torch.stack([value[:routed_intermediate] for value in routed_fc1_init])
    expected_routed_up = torch.stack([value[routed_intermediate:] for value in routed_fc1_init])
    expected_routed_down = torch.stack(routed_fc2_init)

    expected_shared_gate = torch.zeros((routed_intermediate, hidden_size))
    expected_shared_up = torch.zeros_like(expected_shared_gate)
    expected_shared_down = torch.zeros((hidden_size, routed_intermediate))
    expected_shared_gate[:shared_intermediate].copy_(shared_fc1_init[:shared_intermediate])
    expected_shared_up[:shared_intermediate].copy_(shared_fc1_init[shared_intermediate:])
    expected_shared_down[:, :shared_intermediate].copy_(shared_fc2_init)

    expected_by_param = {
        module.routed_gate_weight: expected_routed_gate,
        module.routed_up_weight: expected_routed_up,
        module.routed_down_weight: expected_routed_down,
        module.shared_gate_weight: expected_shared_gate,
        module.shared_up_weight: expected_shared_up,
        module.shared_down_weight: expected_shared_down,
    }
    from megatron.core.optimizer.optimizer import _pop_high_precision_init_val

    for param, expected in expected_by_param.items():
        torch.testing.assert_close(
            param.float(), expected.to(torch.bfloat16).float(), rtol=0, atol=0
        )
        preserved = _pop_high_precision_init_val(param)
        torch.testing.assert_close(preserved, expected, rtol=0, atol=0)
        assert _pop_high_precision_init_val(param) is None

    assert all(item == [True] for item in cleared)
