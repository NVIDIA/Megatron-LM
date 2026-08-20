# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import sys
import types

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


def _compatibility_module(*, use_mxfp8_weights):
    module = mok_megakernel.MoKMegakernel.__new__(mok_megakernel.MoKMegakernel)
    torch.nn.Module.__init__(module)
    module.native_single_grouped_weights = False
    module.use_mxfp8_weights = use_mxfp8_weights
    module.is_first_microbatch = True
    module._quantized_cache = None
    module._quantized_versions = None
    module.routed_gate_weight = torch.nn.Parameter(
        torch.randn((2, 4, 8), dtype=torch.bfloat16)
    )
    module.routed_up_weight = torch.nn.Parameter(
        torch.randn((2, 4, 8), dtype=torch.bfloat16)
    )
    module.routed_down_weight = torch.nn.Parameter(
        torch.randn((2, 8, 4), dtype=torch.bfloat16)
    )
    return module


def test_bf16_compatibility_weights_are_zero_copy_and_separate():
    module = _compatibility_module(use_mxfp8_weights=False)

    gate, up, down = module.quantized_routed_weights()

    assert gate is module.routed_gate_weight
    assert up is module.routed_up_weight
    assert down is module.routed_down_weight
    assert gate.data_ptr() != up.data_ptr()
    assert not module.is_first_microbatch


def test_mxfp8_compatibility_cache_refreshes_per_optimizer_iteration(monkeypatch):
    module = _compatibility_module(use_mxfp8_weights=True)
    quantize_calls = []

    def fake_quantize(weight, rowwise, columnwise):
        assert rowwise and columnwise
        quantize_calls.append(weight)
        return (weight, object(), object(), object())

    fake_mok = types.ModuleType("mok")
    fake_mok.__path__ = []
    fake_ops = types.ModuleType("mok.ops")
    fake_ops.mxfp8_quantize = fake_quantize
    monkeypatch.setitem(sys.modules, "mok", fake_mok)
    monkeypatch.setitem(sys.modules, "mok.ops", fake_ops)

    first = module.quantized_routed_weights()
    second = module.quantized_routed_weights()

    assert second is first
    assert len(quantize_calls) == 3

    module.is_first_microbatch = True
    third = module.quantized_routed_weights()

    assert third is not first
    assert len(quantize_calls) == 6

    with torch.no_grad():
        module.routed_gate_weight.add_(1)
    fourth = module.quantized_routed_weights()

    assert fourth is not third
    assert len(quantize_calls) == 9


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
