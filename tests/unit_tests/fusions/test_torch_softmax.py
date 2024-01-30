import pytest
import torch

from megatron.core.fusions.fused_softmax import FusedScaleMaskSoftmax
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.utils import attention_mask_func


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
