# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""TE attention must not receive Megatron-only ``PackedSeqParams`` fields.

``PackedSeqParams.padding_mask`` marks THD padding slots for Megatron's own token aggregates
(the QSA indexer loss, the Engram fixed-capacity buckets). ``TEDotProductAttention`` forwards
every field it keeps to Transformer Engine's ``DotProductAttention.forward``, which has no such
argument, so any packed (THD) model with TE attention -- a vision encoder, a GQA decoder layer --
failed with ``unexpected keyword argument 'padding_mask'``.
"""

import pytest
import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

try:
    from megatron.core.extensions.transformer_engine import TEDotProductAttention

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


@pytest.mark.skipif(not HAVE_TE, reason="Transformer Engine is required")
class TestTEAttentionPackedPaddingMask:
    def setup_method(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self):
        Utils.destroy_model_parallel()

    def _module(self):
        config = TransformerConfig(
            num_layers=1,
            hidden_size=256,
            num_attention_heads=4,
            kv_channels=64,
            bf16=True,
            params_dtype=torch.bfloat16,
            attention_dropout=0.0,
            hidden_dropout=0.0,
        )
        return TEDotProductAttention(
            config, layer_number=1, attn_mask_type=AttnMaskType.causal, attention_type="self"
        ).cuda()

    def test_padding_mask_is_not_forwarded(self):
        assert "padding_mask" not in self._module().kept_packed_seq_params

    def test_thd_forward_with_padding_mask(self):
        module = self._module()
        # Two documents of 48 and 64 tokens plus a 16-token padding tail, laid out as THD.
        cu_seqlens = torch.tensor([0, 48, 112, 128], dtype=torch.int32, device="cuda")
        padding_mask = torch.zeros(1, 128, dtype=torch.bool, device="cuda")
        padding_mask[:, 112:] = True
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens,
            cu_seqlens_kv_padded=cu_seqlens,
            max_seqlen_q=64,
            max_seqlen_kv=64,
            padding_mask=padding_mask,
        )
        q, k, v = (torch.randn(128, 4, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3))
        out = module(q, k, v, None, AttnMaskType.causal, packed_seq_params=packed_seq_params)
        assert out.shape[0] == 128
        assert torch.isfinite(out).all()
