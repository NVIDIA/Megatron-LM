import pytest
import torch

from megatron.core.fusions.fused_bias_swiglu import (
    BiasAlphaSwiGLUFunction,
    BiasSwiGLUFunction,
    bias_alphaswiglu,
    bias_alphaswiglu_back,
    bias_swiglu_impl,
    weighted_bias_swiglu_impl,
)


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
def test_weighted_bias_swiglu(input_dtype):
    if input_dtype == torch.float32:
        tols = dict(rtol=1.0e-6, atol=1.0e-6)
    elif input_dtype == torch.bfloat16:
        tols = dict(rtol=2.0e-2, atol=1.0e-3)
    else:
        raise ValueError(f"Invalid input dtype: {input_dtype}")

    x = torch.randn(16, 64, dtype=input_dtype, device="cuda")
    x.requires_grad = True
    weights = torch.randn(16, 1, dtype=torch.float32, device="cuda")
    weights.requires_grad = True
    bwd_input = torch.randn(16, 32, dtype=input_dtype, device="cuda")

    y = bias_swiglu_impl(x, None) * weights
    y = y.to(input_dtype)
    y.backward(bwd_input)

    x_2 = x.detach()
    x_2.requires_grad = True
    weights_2 = weights.detach()
    weights_2.requires_grad = True
    bwd_input_2 = bwd_input.detach()

    y_2 = weighted_bias_swiglu_impl(x_2, None, weights_2)
    y_2.backward(bwd_input_2)

    assert y_2.dtype == y.dtype
    assert torch.allclose(y, y_2, **tols)
    assert x_2.grad.dtype == x.grad.dtype
    assert torch.allclose(x.grad, x_2.grad, **tols)
    assert weights_2.grad.dtype == weights.grad.dtype
    if input_dtype == torch.float32:
        assert torch.allclose(weights.grad, weights_2.grad, **tols)


class TestBiasAlphaSwiGLUFunction:
    """Comprehensive tests for BiasAlphaSwiGLUFunction class."""

    @pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0, 5.0])
    @pytest.mark.parametrize("fp8_input_store", [False, True])
    def test_bias_alpha_swiglu_forward(self, input_dtype, alpha, fp8_input_store):
        """Test BiasAlphaSwiGLUFunction forward pass with different configurations."""
        if input_dtype == torch.float32:
            tols = dict(rtol=1e-5, atol=1e-6)
        elif input_dtype == torch.bfloat16:
            tols = dict(rtol=2e-2, atol=1e-2)

        batch_size, hidden_size = 8, 32
        input_tensor = torch.randn(batch_size, hidden_size, dtype=input_dtype, device="cuda")
        bias = torch.randn(hidden_size, dtype=input_dtype, device="cuda")

        # Manual computation for verification
        y_manual = input_tensor + bias
        y1_manual, y2_manual = torch.chunk(y_manual, 2, -1)
        expected = y1_manual * torch.sigmoid(alpha * y1_manual) * (y2_manual + 1)

        # Function computation
        result = BiasAlphaSwiGLUFunction.apply(input_tensor, bias, fp8_input_store, alpha)

        assert result.shape == expected.shape
        assert result.dtype == expected.dtype
        assert torch.allclose(result, expected, **tols)

    @pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("alpha", [1.0, 2.0])
    def test_bias_alpha_swiglu_backward(self, input_dtype, alpha):
        """Test BiasAlphaSwiGLUFunction backward pass gradient computation."""
        if input_dtype == torch.float32:
            tols = dict(rtol=1e-4, atol=1e-5)
        else:  # bfloat16
            tols = dict(rtol=2e-2, atol=1e-2)

        batch_size, hidden_size = 8, 32
        input_tensor = torch.randn(
            batch_size, hidden_size, dtype=input_dtype, device="cuda", requires_grad=True
        )
        bias = torch.randn(hidden_size, dtype=input_dtype, device="cuda", requires_grad=True)

        # Using autograd function
        output = BiasAlphaSwiGLUFunction.apply(input_tensor, bias, False, alpha)
        grad_output = torch.randn_like(output)
        output.backward(grad_output)

        input_grad_autograd = input_tensor.grad.clone()
        bias_grad_autograd = bias.grad.clone()

        # Manual gradient computation for verification
        input_tensor.grad = None
        bias.grad = None

        with torch.enable_grad():
            input_tensor.requires_grad_(True)
            bias.requires_grad_(True)
            y_manual = input_tensor + bias
            y1_manual, y2_manual = torch.chunk(y_manual, 2, -1)
            output_manual = y1_manual * torch.sigmoid(alpha * y1_manual) * (y2_manual + 1)
            output_manual.backward(grad_output)

        # Compare gradients
        assert torch.allclose(input_grad_autograd, input_tensor.grad, **tols)
        assert torch.allclose(bias_grad_autograd, bias.grad, **tols)

    def test_bias_alpha_swiglu_different_shapes(self):
        """Test BiasAlphaSwiGLUFunction with different input shapes."""
        alpha = 1.0
        test_shapes = [(16, 64), (8, 32, 128), (4, 16, 32)]  # 3D case handled by bias_swiglu_impl

        for shape in test_shapes:
            if len(shape) == 3:
                # Test through bias_swiglu_impl which handles 3D
                input_tensor = torch.randn(*shape, device="cuda")
                bias = torch.randn(shape[-1], device="cuda")
                result = bias_swiglu_impl(input_tensor, bias, alpha=alpha)
                assert result.shape == (*shape[:-1], shape[-1] // 2)
            else:
                # Test direct function call for 2D
                input_tensor = torch.randn(*shape, device="cuda")
                bias = torch.randn(shape[-1], device="cuda")
                result = BiasAlphaSwiGLUFunction.apply(input_tensor, bias, False, alpha)
                assert result.shape == (shape[0], shape[1] // 2)

    def test_alpha_parameter_impact(self):
        """Test that alpha parameter affects the output as expected."""
        batch_size, hidden_size = 4, 16
        input_tensor = torch.randn(batch_size, hidden_size, device="cuda")
        bias = torch.zeros(hidden_size, device="cuda")  # Use zero bias for clarity

        alphas = [0.5, 1.0, 2.0]
        outputs = {}

        for alpha in alphas:
            output = BiasAlphaSwiGLUFunction.apply(input_tensor, bias, False, alpha)
            outputs[alpha] = output

        # Different alpha values should produce different outputs
        assert not torch.allclose(outputs[0.5], outputs[1.0])
        assert not torch.allclose(outputs[1.0], outputs[2.0])

    def test_fp8_storage(self):
        """Test FP8 storage functionality."""
        batch_size, hidden_size = 8, 32
        input_tensor = torch.randn(batch_size, hidden_size, device="cuda")
        bias = torch.randn(hidden_size, device="cuda")
        alpha = 1.0

        # Test with FP8 storage enabled and disabled
        output_fp8 = BiasAlphaSwiGLUFunction.apply(input_tensor, bias, True, alpha)
        output_normal = BiasAlphaSwiGLUFunction.apply(input_tensor, bias, False, alpha)

        # Outputs should be the same (storage format shouldn't affect forward pass)
        assert torch.allclose(output_fp8, output_normal, rtol=1e-3, atol=1e-4)

    def test_bias_alpha_swiglu_vs_regular_swiglu(self):
        """Test difference between regular SwiGLU and AlphaSwiGLU."""
        batch_size, hidden_size = 8, 32
        input_tensor = torch.randn(batch_size, hidden_size, device="cuda")
        bias = torch.randn(hidden_size, device="cuda")
        alpha = 2.0  # Non-unity alpha to see difference

        # Regular SwiGLU
        regular_output = BiasSwiGLUFunction.apply(input_tensor, bias, False)

        # Alpha SwiGLU
        alpha_output = BiasAlphaSwiGLUFunction.apply(input_tensor, bias, False, alpha)

        # They should be different
        assert not torch.allclose(regular_output, alpha_output)

        # When alpha = 1.0, they should be more similar but still different due to the +1 bias
        alpha_one_output = BiasAlphaSwiGLUFunction.apply(input_tensor, bias, False, 1.0)
        assert not torch.allclose(regular_output, alpha_one_output)


class TestBiasAlphaSwiGLUHelperFunctions:
    """Test the helper functions used by BiasAlphaSwiGLUFunction."""

    def test_bias_alphaswiglu_function(self):
        """Test the bias_alphaswiglu helper function."""
        batch_size, hidden_size = 4, 16
        input_tensor = torch.randn(batch_size, hidden_size, device="cuda")
        bias = torch.randn(hidden_size, device="cuda")
        alpha = 1.5

        result = bias_alphaswiglu(input_tensor, bias, alpha)

        # Manual computation
        y = input_tensor + bias
        y1, y2 = torch.chunk(y, 2, -1)
        expected = y1 * torch.sigmoid(alpha * y1) * (y2 + 1)

        assert torch.allclose(result, expected, rtol=1e-5)

    def test_bias_alphaswiglu_back_function(self):
        """Test the bias_alphaswiglu_back helper function."""
        batch_size, hidden_size = 4, 16
        input_tensor = torch.randn(batch_size, hidden_size, device="cuda")
        bias = torch.randn(hidden_size, device="cuda")
        alpha = 1.5
        grad_output = torch.randn(batch_size, hidden_size // 2, device="cuda")

        # Get gradient using helper function
        grad_result = bias_alphaswiglu_back(grad_output, input_tensor, bias, alpha)

        # Compare with autograd
        input_tensor_auto = input_tensor.clone().requires_grad_(True)
        bias_auto = bias.clone().requires_grad_(True)

        y = input_tensor_auto + bias_auto
        y1, y2 = torch.chunk(y, 2, -1)
        output_auto = y1 * torch.sigmoid(alpha * y1) * (y2 + 1)
        output_auto.backward(grad_output)

        expected_grad = input_tensor_auto.grad

        assert torch.allclose(grad_result, expected_grad, rtol=1e-4, atol=1e-5)

    def test_edge_cases(self):
        """Test edge cases for BiasAlphaSwiGLU functions."""
        # Test with zero input
        input_tensor = torch.zeros(4, 16, device="cuda")
        bias = torch.zeros(16, device="cuda")
        alpha = 1.0

        result = bias_alphaswiglu(input_tensor, bias, alpha)
        # With zero input and bias, y1=0, y2=0, so output should be 0 * sigmoid(0) * (0 + 1) = 0
        expected = torch.zeros(4, 8, device="cuda")
        assert torch.allclose(result, expected)

        # Test with very large alpha
        alpha_large = 100.0
        input_tensor = torch.randn(4, 16, device="cuda") * 0.01  # Small input
        bias = torch.zeros(16, device="cuda")

        result = bias_alphaswiglu(input_tensor, bias, alpha_large)
        assert torch.all(torch.isfinite(result))  # Should not have NaN or inf

        # Test with very small alpha
        alpha_small = 0.001
        result_small = bias_alphaswiglu(input_tensor, bias, alpha_small)
        assert torch.all(torch.isfinite(result_small))
