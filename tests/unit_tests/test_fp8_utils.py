# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from megatron.core import fp8_utils
from megatron.core.enums import Fp8Recipe
from megatron.core.extensions import transformer_engine as te_ext
from megatron.training.utils import get_device_arch_version
from tests.unit_tests.test_utilities import Utils

try:
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    HAVE_MXFP8_TENSOR = True
except ImportError:
    HAVE_MXFP8_TENSOR = False

# MXFP8 needs Blackwell or newer.
mxfp8_available = HAVE_MXFP8_TENSOR and get_device_arch_version() >= 10
reason_for_no_mxfp8 = "MXFP8 requires Transformer Engine and device arch >= 10"


@pytest.mark.skipif(not fp8_utils.HAVE_TE, reason="Transformer Engine is not installed")
@pytest.mark.parametrize(
    ("is_init", "config_values", "te_helper"),
    [
        (
            False,
            {"fp8": "hybrid", "fp4": None, "fp8_param": False, "fp4_param": False},
            "fp8_autocast",
        ),
        (True, {"fp8": None, "fp4": None, "fp8_param": True, "fp4_param": False}, "fp8_model_init"),
    ],
)
def test_get_fp8_disabled_context_uses_disabled_te_context(is_init, config_values, te_helper):
    config = Mock(**config_values)
    disabled_context = Mock()

    with patch.object(
        fp8_utils.transformer_engine.pytorch, te_helper, return_value=disabled_context
    ) as te_context:
        result = fp8_utils.get_fp8_disabled_context(config, is_init=is_init)

    assert result is disabled_context
    te_context.assert_called_once_with(enabled=False)


@pytest.mark.skipif(not fp8_utils.HAVE_TE, reason="Transformer Engine is not available")
class TestMXFP82DRecipe:
    """Test MCore propagation of the MXFP8 2D quantization option."""

    @pytest.mark.parametrize("mxfp8_2d_quantization", [False, True])
    def test_get_fp8_recipe_forwards_2d_quantization_option(self, mxfp8_2d_quantization):
        recipe = object()
        recipe_constructor = Mock(return_value=recipe)
        config = Mock(
            fp8="e4m3",
            fp8_recipe=Fp8Recipe.mxfp8,
            fp8_dot_product_attention=False,
            mxfp8_2d_quantization=mxfp8_2d_quantization,
        )

        with (
            patch.object(fp8_utils, "is_te_min_version", return_value=True),
            patch(
                "megatron.core.extensions.transformer_engine.te.common.recipe.MXFP8BlockScaling",
                recipe_constructor,
            ),
        ):
            assert fp8_utils.get_fp8_recipe(config) is recipe

        expected_kwargs = {
            "fp8_format": fp8_utils.transformer_engine.common.recipe.Format.E4M3,
            "fp8_dpa": False,
        }
        if mxfp8_2d_quantization:
            expected_kwargs["enable_2d_quantization"] = True
        recipe_constructor.assert_called_once_with(**expected_kwargs)

    def test_get_fp8_recipe_rejects_te_without_2d_quantization_support(self):
        config = Mock(
            fp8="e4m3",
            fp8_recipe=Fp8Recipe.mxfp8,
            fp8_dot_product_attention=False,
            mxfp8_2d_quantization=True,
        )

        def old_mxfp8_recipe(fp8_format, fp8_dpa=False):
            return Mock(fp8_format=fp8_format, fp8_dpa=fp8_dpa)

        with (
            patch.object(fp8_utils, "is_te_min_version", return_value=True),
            patch(
                "megatron.core.extensions.transformer_engine.te.common.recipe.MXFP8BlockScaling",
                old_mxfp8_recipe,
            ),
            pytest.raises(RuntimeError, match="enable_2d_quantization"),
        ):
            fp8_utils.get_fp8_recipe(config)

    def test_per_module_recipe_forwards_2d_quantization_option(self):
        recipe = object()
        model_init_context = object()
        quantization_recipe = te_ext.TEQuantizationRecipe(
            fp8_quantization_recipe=Fp8Recipe.mxfp8, mxfp8_2d_quantization=True, fp8_param=True
        )

        with (
            patch.object(te_ext, "get_mxfp8_block_scaling_recipe", return_value=recipe) as factory,
            patch.object(te_ext, "fp8_model_init", return_value=model_init_context),
        ):
            assert (
                te_ext._get_fp8_model_init_for_quant_recipe(quantization_recipe)
                is model_init_context
            )

        factory.assert_called_once_with(
            mxfp8_2d_quantization=True, fp8_format=te_ext.te.common.recipe.Format.E4M3
        )

    def test_per_module_recipe_rejects_2d_quantization_without_mxfp8(self):
        with pytest.raises(ValueError, match="requires fp8_quantization_recipe='mxfp8'"):
            te_ext.TEQuantizationRecipe.parse_from_config(
                {"fp8_quantization_recipe": Fp8Recipe.tensorwise, "mxfp8_2d_quantization": True}
            )


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


@pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
class TestCopyTensorsToQuantizedParams:
    """Cover the batched MXFP8 param copy-back used by _post_param_sync.

    ``copy_tensors_to_quantized_params`` bypasses ``copy_`` and calls the destination quantizer
    directly, so the contract to protect is that it still writes exactly what the per-param
    ``copy_tensor_to_quantized_param`` would have written.
    """

    SHAPES = [(1024, 512), (2048, 256), (512, 1024)]

    def _make_param(self, shape):
        quantizer = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)
        tensor = quantizer.make_empty(shape, dtype=torch.bfloat16, device="cuda")
        return torch.nn.Parameter(tensor, requires_grad=False)

    def _raw_buffers(self, param):
        """The four buffers MXFP8 storage is made of, i.e. everything a cast writes."""
        data = param.data
        return (
            data._rowwise_data,
            data._rowwise_scale_inv,
            data._columnwise_data,
            data._columnwise_scale_inv,
        )

    def test_matches_per_param_copy(self):
        """Batched copy-back is bitwise identical to copying one param at a time."""
        torch.manual_seed(0)
        reference_params = [self._make_param(shape) for shape in self.SHAPES]
        batched_params = [self._make_param(shape) for shape in self.SHAPES]
        # Sources are flat slices, matching how _post_param_sync views the param buffer.
        srcs = [
            torch.randn(shape, dtype=torch.bfloat16, device="cuda").view(-1)
            for shape in self.SHAPES
        ]

        for param, src in zip(reference_params, srcs):
            fp8_utils.copy_tensor_to_quantized_param(param, src)
        fp8_utils.copy_tensors_to_quantized_params(batched_params, srcs)
        torch.cuda.synchronize()

        for reference, batched in zip(reference_params, batched_params):
            for expected, actual in zip(self._raw_buffers(reference), self._raw_buffers(batched)):
                assert torch.equal(expected, actual)

    def test_falls_back_without_quantizer(self):
        """A destination with no quantizer of its own still gets written."""
        param = self._make_param(self.SHAPES[0])
        param.data._quantizer = None
        src = torch.randn(self.SHAPES[0], dtype=torch.bfloat16, device="cuda").view(-1)

        fp8_utils.copy_tensors_to_quantized_params([param], [src])
        torch.cuda.synchronize()

        # A quantized copy of a non-zero source cannot be all zeros.
        assert param.data._rowwise_data.any()

    def test_empty_input(self):
        """No params is a no-op rather than an error."""
        fp8_utils.copy_tensors_to_quantized_params([], [])
