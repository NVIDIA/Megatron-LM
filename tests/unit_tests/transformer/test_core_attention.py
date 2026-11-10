# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.


import pytest
import torch

from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention import CrossAttention
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from tests.unit_tests.test_utilities import Utils

""" 

@pytest.fixture
def core_attention(transformer_config):
    return CrossAttention(transformer_config)


class TestCoreAttention:
    def test_constructor(self, core_attention):
        assert isinstance(core_attention, CrossAttention)
        assert core_attention.layer_number == 1

        num_weights = sum([p.numel() for p in core_attention.parameters()])
        assert num_weights == 0

    def test_cpu_forward(self, core_attention):
        # we can't currently do this because the global memory buffer is on GPU
        pass

    def test_gpu_forward(self, core_attention):

        # destroy_global_memory_buffer()
        # _set_global_memory_buffer()
        # model_parallel_cuda_manual_seed(123)

        core_attention.cuda()
        config = core_attention.config
        sequence_length = 32
        micro_batch_size = 2
        # query_layer (float): [sequence_length, micro_batch_size, num_attention_heads, hidden_size / num_attention_heads]
        query_layer = torch.ones(
            (
                sequence_length,
                micro_batch_size,
                config.num_attention_heads,
                config.hidden_size // config.num_attention_heads,
            )
        ).cuda()

        key_layer = torch.ones_like(query_layer).cuda()

        value_layer = torch.ones_like(query_layer).cuda()

        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()

        context_layer = core_attention(
            query_layer=query_layer, key_layer=key_layer, value_layer=value_layer, attention_mask=attention_mask
        )

        assert context_layer.shape[0] == sequence_length
        assert context_layer.shape[1] == micro_batch_size
        assert context_layer.shape[2] == config.hidden_size
        assert context_layer.device.type == 'cuda'
        assert context_layer.dtype == torch.float32

"""


class TestDotProductAttentionSoftcap:
    """The cap must reach the softmax at the configured magnitude.

    apply_query_key_layer_scaling divides softmax_scale by layer_number and defers the
    matching multiply to scale_mask_softmax, so a cap applied between the two has to be
    divided as well. Otherwise the softmax sees an effective cap of cap * layer_number,
    which grows with depth and quietly stops capping in the layers that need it most.
    """

    _CAP = 0.5
    _LAYER_NUMBER = 8

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _attention(self, qk_layer_scaling, cap=_CAP):
        config = TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            use_cpu_initialization=True,
            fp16=True,
            params_dtype=torch.float16,
            attn_logit_softcapping=cap,
            apply_query_key_layer_scaling=qk_layer_scaling,
            # Dropout would give each forward a different mask, which both hides the cap
            # difference and satisfies the separation check below on noise alone.
            attention_dropout=0.0,
        )
        return DotProductAttention(
            config=config,
            layer_number=self._LAYER_NUMBER,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        ).cuda()

    def _qkv(self, seed=7):
        torch.manual_seed(seed)
        # Large values so the logits saturate the cap; that is where a wrong magnitude shows.
        shape = (16, 2, 4, 32)
        return tuple(
            (8.0 * torch.randn(shape, dtype=torch.float16, device="cuda")) for _ in range(3)
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cap_magnitude_is_independent_of_qk_layer_scaling(self):
        """The same configured cap must produce the same attention either way."""
        q, k, v = self._qkv()
        without = self._attention(qk_layer_scaling=False)(q, k, v, None).float()
        with_scaling = self._attention(qk_layer_scaling=True)(q, k, v, None).float()

        # Anti-vacuity: a cap that is wrong by exactly layer_number must be distinguishable,
        # otherwise this comparison would pass no matter what the code did.
        wrong = self._attention(qk_layer_scaling=False, cap=self._CAP * self._LAYER_NUMBER)(
            q, k, v, None
        ).float()
        assert (without - wrong).abs().max() > 1e-3, "inputs do not separate the two caps"

        torch.testing.assert_close(with_scaling, without, rtol=1e-2, atol=1e-2)
