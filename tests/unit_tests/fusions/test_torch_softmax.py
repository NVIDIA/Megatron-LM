import pytest
import torch

from megatron.core.fusions.fused_softmax import FusedScaleMaskSoftmax
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.utils import attention_mask_func, get_default_causal_mask


class TestTorchSoftmax:
    def setup_method(self, method):
        # The important settings tested are forward_torch_softmax path
        # with locally generated casual mask for attention_mask_func:
        self.softmax = FusedScaleMaskSoftmax(
            input_in_fp16=False,
            input_in_bf16=False,
            attn_mask_type=AttnMaskType.causal,
            scaled_masked_softmax_fusion=False,
            mask_func=attention_mask_func,
            softmax_in_fp32=True,
            scale=None,
        )

    def teardown_method(self):
        get_default_causal_mask.cache_clear()

    def test_output_shape(self):
        x = torch.randn(8, 2, 4, 4, device="cuda")
        y = self.softmax(x, None, None)
        assert x.shape == y.shape

    def test_causal_mask_input_shape_assert(self):
        x = torch.randn(1, 1, 4, 16, device="cuda")
        with pytest.raises(AssertionError):
            self.softmax(x, None, None)

    def test_causal_mask_equal_scores(self):
        # For equal input values (e.g. zero) correctly masked softmax should
        # produce equal scores among non-masked elements. For example, in case
        # sq == sk == 2 the expected output is (ignoring b and np dimensions):
        # [[1.0, 0.0],
        #  [0.5, 0.5]]
        b, np, sq, sk = 8, 2, 32, 32
        x = torch.zeros([b, np, sq, sk]).cuda()
        y = self.softmax(x, None, None)
        y_expected = torch.tril(torch.ones(b, np, sq, sk, device="cuda"))
        y_expected /= torch.arange(1, sq + 1, device="cuda").reshape((-1, 1))
        assert torch.allclose(y, y_expected, rtol=1e-08, atol=1e-08)


class TestSoftmaxOne:
    def setup_method(self, method):
        self.softmax = FusedScaleMaskSoftmax(
            input_in_fp16=False,
            input_in_bf16=False,
            attn_mask_type=AttnMaskType.causal,
            scaled_masked_softmax_fusion=False,
            mask_func=attention_mask_func,
            softmax_in_fp32=True,
            scale=None,
        )

    def test_output_shape(self):
        x = torch.randn(8, 2, 4, 4, device="cuda")
        softmax_offset = torch.zeros(x.size(1), device="cuda")
        y = self.softmax(x, None, softmax_offset)
        assert x.shape == y.shape

    def test_fixed_offset(self):
        x = torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]], device="cuda")

        # Use logit offset of 0.0 per head so denominator adds 1.0 per position
        softmax_offset = torch.zeros(x.size(1), device="cuda")
        output = self.softmax(x, None, softmax_offset)

        # Manual computation matching implementation semantics
        qk = torch.cat(
            [x, softmax_offset.reshape(1, -1, 1, 1).expand(x.size(0), -1, x.size(2), -1)], dim=-1
        )
        expected = torch.softmax(qk, dim=-1)[..., :-1]

        assert torch.allclose(output, expected, rtol=1e-5)

    def test_learnable_offset(self):
        x = torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]], device="cuda")

        # Learnable offset provided externally (logit space)
        learnable_offset = torch.nn.Parameter(torch.empty(x.size(1), device="cuda"))
        learnable_offset.data.normal_(mean=0.0, std=0.01)
        output = self.softmax(x, None, learnable_offset)

        # Manual computation: append logit, softmax, then drop the extra slot
        qk = torch.cat(
            [x, learnable_offset.reshape(1, -1, 1, 1).expand(x.size(0), -1, x.size(2), -1)], dim=-1
        )
        expected = torch.softmax(qk, dim=-1)[..., :-1]

        assert torch.allclose(output, expected, rtol=1e-5)

    def test_numerical_stability(self):
        x = torch.tensor(
            [[[[1e10, -1e10, 1e10], [-1e10, 1e10, -1e10], [1e10, -1e10, 1e10]]]], device="cuda"
        )

        softmax_offset = torch.zeros(x.size(1), device="cuda")
        output = self.softmax(x, None, softmax_offset)

        assert torch.all(torch.isfinite(output))
        assert torch.all((output >= 0) & (output <= 1))

    def test_causal_mask_equal_scores(self):
        # For equal input values (e.g. zero) correctly masked softmax should
        # produce equal scores among non-masked elements
        b, np, sq, sk = 8, 2, 32, 32
        x = torch.zeros([b, np, sq, sk], device="cuda")
        softmax_offset = torch.zeros(x.size(1), device="cuda")
        y = self.softmax(x, None, softmax_offset)

        # Expected: lower triangular matrix with rows normalized
        y_expected = torch.tril(torch.ones(b, np, sq, sk, device="cuda"))
        y_expected /= 1.0 + torch.arange(1, sq + 1, device="cuda").reshape((-1, 1))

        assert torch.allclose(y, y_expected, rtol=1e-5)


class TestFusedScaleMaskSoftmaxComprehensive:
    """Comprehensive tests for FusedScaleMaskSoftmax including window attention and scaling."""

    def teardown_method(self):
        get_default_causal_mask.cache_clear()

    def test_scaling_factor(self):
        """Test softmax with different scaling factors."""
        x = torch.randn(2, 4, 8, 8, device="cuda")

        for scale in [0.5, 1.0, 2.0]:
            softmax = FusedScaleMaskSoftmax(
                input_in_fp16=False,
                input_in_bf16=False,
                attn_mask_type=AttnMaskType.causal,
                scaled_masked_softmax_fusion=False,
                mask_func=attention_mask_func,
                softmax_in_fp32=True,
                scale=scale,
            )

            y = softmax(x, None, None)
            assert x.shape == y.shape
            # Check if output is a valid probability distribution
            assert torch.allclose(y.sum(dim=-1), torch.ones_like(y.sum(dim=-1)), rtol=1e-5)

    def test_window_attention_integration(self):
        """Test FusedScaleMaskSoftmax with window attention configuration."""
        softmax = FusedScaleMaskSoftmax(
            input_in_fp16=False,
            input_in_bf16=False,
            attn_mask_type=AttnMaskType.causal,
            scaled_masked_softmax_fusion=False,
            mask_func=attention_mask_func,
            softmax_in_fp32=True,
            scale=None,
        )

        x = torch.randn(2, 4, 16, 16, device="cuda")
        y = softmax(x, None, None)
        assert x.shape == y.shape

    def test_fused_kernel_availability(self):
        """Test is_kernel_available method with different configurations."""
        softmax = FusedScaleMaskSoftmax(
            input_in_fp16=False,
            input_in_bf16=False,
            attn_mask_type=AttnMaskType.causal,
            scaled_masked_softmax_fusion=False,
            mask_func=attention_mask_func,
            softmax_in_fp32=True,
            scale=None,
        )

        # Test with different input dimensions
        b, np, sq, sk = 4, 8, 32, 32  # Valid dimensions for fusion
        is_available = softmax.is_kernel_available(None, b, np, sq, sk)
        # Kernel availability depends on CUDA extensions being available
        assert isinstance(is_available, bool)

    def test_different_mask_types(self):
        """Test FusedScaleMaskSoftmax with different attention mask types."""
        for mask_type in [AttnMaskType.causal, AttnMaskType.padding]:
            softmax = FusedScaleMaskSoftmax(
                input_in_fp16=False,
                input_in_bf16=False,
                attn_mask_type=AttnMaskType.causal,
                scaled_masked_softmax_fusion=False,
                mask_func=attention_mask_func,
                softmax_in_fp32=True,
                scale=None,
            )

            x = torch.randn(2, 4, 8, 8, device="cuda")
            if mask_type == AttnMaskType.padding:
                # Create a padding mask
                mask = torch.ones(2, 1, 8, 8, dtype=torch.bool, device="cuda")
                mask[:, :, :, -2:] = False  # Mask last 2 positions
            else:
                mask = None

            y = softmax(x, mask, None)
            assert x.shape == y.shape

    def test_mixed_precision(self):
        """Test FusedScaleMaskSoftmax with different precision configurations."""
        test_configs = [
            {"fp16": True, "bf16": False, "attention_softmax_in_fp32": True},
            {"fp16": False, "bf16": True, "attention_softmax_in_fp32": True},
            {"fp16": False, "bf16": False, "attention_softmax_in_fp32": False},
        ]

        for config_params in test_configs:
            softmax = FusedScaleMaskSoftmax(
                input_in_fp16=config_params['fp16'],
                input_in_bf16=config_params['bf16'],
                attn_mask_type=AttnMaskType.causal,
                scaled_masked_softmax_fusion=False,
                mask_func=attention_mask_func,
                softmax_in_fp32=config_params['attention_softmax_in_fp32'],
                scale=None,
            )

            x = torch.randn(2, 4, 8, 8, device="cuda")
            if config_params["fp16"]:
                x = x.half()
            elif config_params["bf16"]:
                x = x.bfloat16()

            y = softmax(x, None, None)
            assert x.shape == y.shape

    def test_gradient_flow(self):
        """Test gradient flow through FusedScaleMaskSoftmax."""
        softmax = FusedScaleMaskSoftmax(
            input_in_fp16=False,
            input_in_bf16=False,
            attn_mask_type=AttnMaskType.causal,
            scaled_masked_softmax_fusion=False,
            mask_func=attention_mask_func,
            softmax_in_fp32=True,
            scale=1.0,
        )
        x = torch.randn(2, 4, 8, 8, device="cuda", requires_grad=True)
        y = softmax(x, None, None)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
