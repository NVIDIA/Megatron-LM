# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from unittest.mock import Mock, patch
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from megatron.core import fp8_utils
from megatron.core.enums import Fp8Recipe
from tests.unit_tests.test_utilities import Utils


def test_resolve_callable_from_python_import_path_success():
    fn = fp8_utils._resolve_callable_from_python_import_path("math.sqrt")

    assert fn(9) == 3


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("", "non-empty string"),
        ("math", "Expected 'pkg.mod.func'"),
        ("missing_module.sqrt", "Failed to import module"),
        ("math.missing_attr", "Attribute 'missing_attr' not found"),
        ("math.pi", "is not callable"),
    ],
)
def test_resolve_callable_from_python_import_path_errors(path, message):
    with pytest.raises(ValueError, match=message):
        fp8_utils._resolve_callable_from_python_import_path(path)


def test_get_fp8_align_size():
    assert fp8_utils.get_fp8_align_size(Fp8Recipe.mxfp8) == 32
    assert fp8_utils.get_fp8_align_size(Fp8Recipe.delayed) == 16


def test_fp8_tensor_type_helpers(monkeypatch):
    class FakeFloat8Tensor:
        pass

    class FakeMXFP8Tensor:
        pass

    float8_tensor = FakeFloat8Tensor()
    mxfp8_tensor = FakeMXFP8Tensor()

    monkeypatch.setattr(fp8_utils, "HAVE_TE_FP8_TENSOR_CLASS", True)
    monkeypatch.setattr(fp8_utils, "FP8_TENSOR_CLASS", FakeFloat8Tensor)
    monkeypatch.setattr(fp8_utils, "HAVE_TE_MXFP8TENSOR", True)
    monkeypatch.setattr(fp8_utils, "MXFP8Tensor", FakeMXFP8Tensor, raising=False)

    assert fp8_utils.is_float8tensor(float8_tensor)
    assert not fp8_utils.is_float8tensor(object())
    assert fp8_utils.is_mxfp8tensor(mxfp8_tensor)
    assert not fp8_utils.is_mxfp8tensor(object())


def test_dequantize_fp8_tensor_dispatches_by_te_version(monkeypatch):
    tensor = Mock()

    monkeypatch.setattr(fp8_utils, "is_te_min_version", lambda version: True)
    fp8_utils.dequantize_fp8_tensor(tensor)
    tensor.dequantize.assert_called_once_with()

    tensor = Mock()
    monkeypatch.setattr(fp8_utils, "is_te_min_version", lambda version: False)
    fp8_utils.dequantize_fp8_tensor(tensor)
    tensor.from_float8.assert_called_once_with()


def test_parallel_linear_type_helpers(monkeypatch):
    class FakeColumn:
        pass

    class FakeRow:
        pass

    class FakeTEColumn:
        pass

    class FakeTELayerNormColumn:
        pass

    class FakeTERow:
        pass

    monkeypatch.setattr(fp8_utils, "ColumnParallelLinear", FakeColumn)
    monkeypatch.setattr(fp8_utils, "RowParallelLinear", FakeRow)
    monkeypatch.setattr(fp8_utils, "TEColumnParallelLinear", FakeTEColumn, raising=False)
    monkeypatch.setattr(
        fp8_utils, "TELayerNormColumnParallelLinear", FakeTELayerNormColumn, raising=False
    )
    monkeypatch.setattr(fp8_utils, "TERowParallelLinear", FakeTERow, raising=False)

    monkeypatch.setattr(fp8_utils, "HAVE_TE", False)
    assert fp8_utils.is_column_parallel_linear(FakeColumn())
    assert fp8_utils.is_row_parallel_linear(FakeRow())
    assert not fp8_utils.is_column_parallel_linear(FakeTEColumn())
    assert not fp8_utils.is_row_parallel_linear(FakeTERow())

    monkeypatch.setattr(fp8_utils, "HAVE_TE", True)
    assert fp8_utils.is_column_parallel_linear(FakeTEColumn())
    assert fp8_utils.is_column_parallel_linear(FakeTELayerNormColumn())
    assert fp8_utils.is_row_parallel_linear(FakeTERow())


def test_interface_functions_delegate_to_selected_implementations(monkeypatch):
    modify_impl = Mock()
    quantize_impl = Mock()
    correct_impl = Mock()
    post_all_gather = Mock()

    monkeypatch.setattr(fp8_utils, "_modify_underlying_storage_impl", modify_impl)
    monkeypatch.setattr(fp8_utils, "_quantize_param_shard_impl", quantize_impl)
    monkeypatch.setattr(fp8_utils, "_correct_amax_history_if_needed_impl", correct_impl)
    monkeypatch.setattr(fp8_utils, "te_post_all_gather_processing", post_all_gather)

    fp8_utils.modify_underlying_storage("tensor", "raw")
    fp8_utils.quantize_param_shard(["model"], ["main"], [0], "group", ["fsdp"])
    fp8_utils.correct_amax_history_if_needed(["module"])
    fp8_utils.post_all_gather_processing(["param"])

    modify_impl.assert_called_once_with("tensor", "raw")
    quantize_impl.assert_called_once_with(["model"], ["main"], [0], "group", ["fsdp"])
    correct_impl.assert_called_once_with(["module"])
    post_all_gather.assert_called_once_with(["param"])


def test_post_all_gather_processing_noops_without_te_helper(monkeypatch):
    monkeypatch.setattr(fp8_utils, "te_post_all_gather_processing", None)

    assert fp8_utils.post_all_gather_processing(["param"]) is None


def test_is_first_last_bf16_layer():
    config = SimpleNamespace(
        first_last_layers_bf16=True,
        num_layers_at_start_in_bf16=2,
        num_layers_at_end_in_bf16=1,
        num_layers=6,
    )

    assert fp8_utils.is_first_last_bf16_layer(config, 0)
    assert fp8_utils.is_first_last_bf16_layer(config, 1)
    assert not fp8_utils.is_first_last_bf16_layer(config, 2)
    assert fp8_utils.is_first_last_bf16_layer(config, 5)
    assert not fp8_utils.is_first_last_bf16_layer(config, -1)

    config.first_last_layers_bf16 = False
    assert not fp8_utils.is_first_last_bf16_layer(config, 0)


class MockTELinear(nn.Module):
    """Mock TE Linear module for testing."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        return x @ self.weight.t()


class TestFP8Padding:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        # Clear the wrapped modules set before each test
        fp8_utils._fp8_inference_wrapped_modules.clear()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        fp8_utils._fp8_inference_wrapped_modules.clear()

    def test_prepare_model_for_fp8_inference_basic(self):
        """Test prepare_model_for_fp8_inference wraps TE modules."""

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.te_layer = MockTELinear(128, 128)
                self.regular_layer = nn.Linear(128, 128)

        with (
            patch.object(fp8_utils, 'HAVE_TE', True),
            patch.object(fp8_utils, 'Fp8Padding'),
            patch.object(fp8_utils, 'Fp8Unpadding'),
            patch.object(fp8_utils, 'TE_LINEAR_TYPES', (MockTELinear,)),
        ):

            model = SimpleModel()
            original_te_forward = model.te_layer.forward
            original_regular_forward = model.regular_layer.forward

            # Prepare model
            prepared_model = fp8_utils.prepare_model_for_fp8_inference(model)

            # Check same model returned
            assert prepared_model is model

            # Check TE layer was wrapped
            assert model.te_layer.forward != original_te_forward
            assert model.te_layer in fp8_utils._fp8_inference_wrapped_modules

            # Check regular layer was not wrapped
            assert model.regular_layer.forward == original_regular_forward

    def test_padding_mechanism_works(self):
        """Test that the padding mechanism actually pads and unpads correctly."""

        with (
            patch.object(fp8_utils, 'HAVE_TE', True),
            patch.object(fp8_utils, 'Fp8Padding') as mock_pad_class,
            patch.object(fp8_utils, 'Fp8Unpadding') as mock_unpad_class,
        ):

            # Setup padding mock to pad from 6 to 16
            mock_pad_instance = Mock()
            mock_pad_instance.return_value = (torch.zeros(16, 8192), [16])
            mock_pad_class.return_value = mock_pad_instance

            # Setup unpadding mock to unpad from 16 to 6
            mock_unpad_instance = Mock()
            mock_unpad_instance.return_value = torch.zeros(6, 8192)
            mock_unpad_class.return_value = mock_unpad_instance

            # Create module and get access to padded_forward directly
            module = MockTELinear(4096, 4096)
            module.cuda()

            # Store original forward to track what it receives
            original_forward_input = None

            def track_forward(x):
                nonlocal original_forward_input
                original_forward_input = x
                return torch.randn(x.shape[0], x.shape[1], 4096).cuda()

            module.forward = track_forward

            # Manually create the wrapped forward function
            fp8_utils._wrap_te_linear_for_padding(module)
            padded_forward = module.forward

            # Mock FP8GlobalStateManager.is_fp8_enabled to return True
            with patch(
                'transformer_engine.pytorch.fp8.FP8GlobalStateManager.is_fp8_enabled',
                return_value=True,
            ):
                # Create input: (seq_len=6, batch=2, hidden=4096)
                input_tensor = torch.randn(6, 2, 4096).cuda()

                # Call padded_forward directly
                output = padded_forward(input_tensor)

            # Verify padding was called with correct reshaped input
            mock_pad_instance.assert_called_once()
            call_args = mock_pad_instance.call_args[0]
            assert call_args[0].shape == (6, 8192)  # Reshaped to 2D
            assert call_args[1] == [6]  # Split info

            # Verify the original forward received padded input with correct shape
            assert original_forward_input.shape == (16, 2, 4096)  # Padded to 16

            # Verify unpadding was called
            mock_unpad_instance.assert_called_once()
            unpad_args = mock_unpad_instance.call_args[0]
            assert unpad_args[0].shape == (16, 8192)  # Padded 2D tensor
            assert unpad_args[1] == [6]  # Original split

            # Verify output has original shape
            assert output.shape == (6, 2, 4096)  # Back to original seq_len
