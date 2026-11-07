# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.


import pytest
import torch

from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.optimizer.clip_grads import count_zeros_fp32, get_grad_norm_fp32
from megatron.core.tensor_parallel import param_is_not_tensor_parallel_duplicate
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention import CrossAttention
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.utils import is_te_min_version
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


class TestLearnableSoftmaxOffsetTensorParallel:
    """Every head's learnable attention sink must count once toward the gradient statistics.

    softmax_offset holds one learnable logit per local attention head, so with TP>1 each TP
    rank owns a different slice of the heads. Unless the parameter carries the TP attributes,
    param_is_not_tensor_parallel_duplicate takes it for a replica and keeps it only on TP rank
    0, so the norm used for gradient clipping and the zero count both drop the sink gradients
    of the heads that live on the other TP ranks.
    """

    _TP = 2
    _NUM_HEADS = 4
    # Sink gradient of each global head. Every TP rank holds one zero and one non-zero entry,
    # so both the norm and the zero count change if any rank's slice is dropped.
    _FULL_SINK_GRAD = (0.0, 2.0, 0.0, 4.0)

    def setup_method(self, method):
        if Utils.world_size % self._TP != 0:
            pytest.skip(f"requires a world size divisible by TP={self._TP}, got {Utils.world_size}")
        Utils.initialize_model_parallel(tensor_model_parallel_size=self._TP)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _config(self):
        return TransformerConfig(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=self._NUM_HEADS,
            tensor_model_parallel_size=self._TP,
            softmax_type="learnable",
            use_cpu_initialization=True,
            params_dtype=torch.float32,
        )

    def _assert_sink_counted_once_per_head(self, softmax_offset):
        tp_group = parallel_state.get_tensor_model_parallel_group()
        heads_per_rank = self._NUM_HEADS // self._TP
        assert softmax_offset.shape == (heads_per_rank,)
        assert softmax_offset.tensor_model_parallel is True
        assert softmax_offset.partition_dim == 0
        assert softmax_offset.partition_stride == 1

        full_grad = torch.tensor(self._FULL_SINK_GRAD, device="cuda")
        first_head = tp_group.rank() * heads_per_rank
        softmax_offset.grad = full_grad[first_head : first_head + heads_per_rank].clone()

        # The duplicate filter that MegatronOptimizer._filter_grads_for_norm applies.
        grads_for_norm = [
            param.grad
            for param in [softmax_offset]
            if param_is_not_tensor_parallel_duplicate(param, tp_group=tp_group)
        ]
        grad_norm = get_grad_norm_fp32(grads_for_norm, grad_stats_parallel_group=tp_group)
        assert float(grad_norm) == pytest.approx(full_grad.norm().item(), rel=1e-6)

        num_zeros = count_zeros_fp32(
            [softmax_offset], grad_stats_parallel_group=tp_group, tp_group=tp_group
        )
        assert num_zeros == (full_grad == 0).sum().item()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_dot_product_attention_sink_counted_on_every_tp_rank(self):
        attention = DotProductAttention(
            config=self._config(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        self._assert_sink_counted_once_per_head(attention.softmax_offset)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    @pytest.mark.skipif(not is_te_min_version("2.8.0"), reason="softmax_type requires TE >= 2.8.0")
    def test_te_dot_product_attention_sink_counted_on_every_tp_rank(self):
        attention = TEDotProductAttention(
            config=self._config(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        self._assert_sink_counted_once_per_head(attention.softmax_offset)
