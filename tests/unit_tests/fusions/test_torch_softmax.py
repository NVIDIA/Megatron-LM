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
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            fp16=False,
            bf16=False,
            masked_softmax_fusion=False,
            attention_softmax_in_fp32=True,
        )
        self.softmax = FusedScaleMaskSoftmax(
            config=transformer_config,
            attn_mask_type=AttnMaskType.causal,
            mask_func=attention_mask_func,
            scale=None,
        )

    def teardown_method(self):
        get_default_causal_mask.cache_clear()

    def test_output_shape(self):
        x = torch.randn(8, 2, 4, 4, device="cuda")
        y = self.softmax(x, None)
        assert x.shape == y.shape

    def test_causal_mask_input_shape_assert(self):
        x = torch.randn(1, 1, 4, 16, device="cuda")
        with pytest.raises(AssertionError):
            self.softmax(x, None)

    def test_causal_mask_equal_scores(self):
        # For equal input values (e.g. zero) correctly masked softmax should
        # produce equal scores among non-masked elements. For example, in case
        # sq == sk == 2 the expected output is (ignoring b and np dimensions):
        # [[1.0, 0.0],
        #  [0.5, 0.5]]
        b, np, sq, sk = 8, 2, 32, 32
        x = torch.zeros([b, np, sq, sk]).cuda()
        y = self.softmax(x, None)
        y_expected = torch.tril(torch.ones(b, np, sq, sk, device="cuda"))
        y_expected /= torch.arange(1, sq + 1, device="cuda").reshape((-1, 1))
        assert torch.allclose(y, y_expected, rtol=1e-08, atol=1e-08)


class TestSoftmaxOne:
    def setup_method(self, method):
        self.transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            fp16=False,
            bf16=False,
            attention_softmax_denominator_offset=1.0,
            attention_softmax_in_fp32=True,
        )
        self.softmax = FusedScaleMaskSoftmax(
            config=self.transformer_config,
            attn_mask_type=AttnMaskType.causal,
            mask_func=attention_mask_func,
            scale=None,
        )

    def test_output_shape(self):
        x = torch.randn(8, 2, 4, 4, device="cuda")
        y = self.softmax(x, None)
        assert x.shape == y.shape

    def test_fixed_offset(self):
        x = torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]], device="cuda")

        output = self.softmax(x, None)

        # Manual computation for verification
        exp_x = torch.exp(x - x.max(dim=-1, keepdim=True).values)
        expected = exp_x / (1.0 + exp_x.sum(dim=-1, keepdim=True))

        assert torch.allclose(output, expected, rtol=1e-5)

    def test_learnable_offset(self):
        x = torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]], device="cuda")

        # Configure with learnable offset
        config_learnable = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            attention_softmax_denominator_offset='learnable',
        )
        softmax_learnable = FusedScaleMaskSoftmax(
            config=config_learnable,
            attn_mask_type=AttnMaskType.causal,
            mask_func=attention_mask_func,
            scale=None,
        ).to("cuda")

        # Initialize learnable weight
        softmax_learnable.softmax_denominator_weight.data.normal_(mean=0.0, std=0.01)
        output = softmax_learnable(x, None)

        exp_x = torch.exp(x - x.max(dim=-1, keepdim=True).values)
        expected = exp_x / (
            softmax_learnable.softmax_denominator_weight + exp_x.sum(dim=-1, keepdim=True)
        )

        assert torch.allclose(output, expected, rtol=1e-5)

    def test_numerical_stability(self):
        x = torch.tensor(
            [[[[1e10, -1e10, 1e10], [-1e10, 1e10, -1e10], [1e10, -1e10, 1e10]]]], device="cuda"
        )

        output = self.softmax(x, None)

        assert torch.all(torch.isfinite(output))
        assert torch.all((output >= 0) & (output <= 1))

    def test_causal_mask_equal_scores(self):
        # For equal input values (e.g. zero) correctly masked softmax should
        # produce equal scores among non-masked elements
        b, np, sq, sk = 8, 2, 32, 32
        x = torch.zeros([b, np, sq, sk], device="cuda")
        y = self.softmax(x, None)

        # Expected: lower triangular matrix with rows normalized
        y_expected = torch.tril(torch.ones(b, np, sq, sk, device="cuda"))
        y_expected /= 1.0 + torch.arange(1, sq + 1, device="cuda").reshape((-1, 1))

        assert torch.allclose(y, y_expected, rtol=1e-5)
