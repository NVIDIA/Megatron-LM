# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

import megatron.core.transformer.attention as attention_module
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb_with_cos_sin
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.transformer.attention import SelfAttention


class TestRotaryEmbeddingWithPrecomputedCosSin:

    def setup_method(self):
        self.batch_size = 3
        self.seq_len = 4
        self.d_rot = 6
        self.rotary_embedding = RotaryEmbedding(kv_channels=4, rotary_percent=1.0)

    def test_output_shapes_match(self):

        # Create input tensors
        t = torch.randn(self.seq_len, self.batch_size, 2, self.d_rot * 2, device="cuda")
        rotary_pos_cos, rotary_pos_sin = self.rotary_embedding.get_cos_sin(self.seq_len)

        # Test using Flash Decoding optimized kernel which requires precomputed cos & sin tensors
        expected_shape = torch.Size(
            [self.seq_len, self.batch_size, self.seq_len // 2, self.seq_len * self.batch_size]
        )
        output_flash_rotary = apply_rotary_pos_emb_with_cos_sin(
            t, rotary_pos_cos, rotary_pos_sin, rotary_interleaved=True
        )

        assert (
            output_flash_rotary.shape == expected_shape
        ), f"Outputs do not match: {output_flash_rotary.shape} != {expected_shape}"


@pytest.mark.parametrize(
    ("batch_invariant_mode", "num_requests", "tokens_per_request", "padded_token_count"),
    [
        (False, 728, 1, 728),
        (True, 728, 1, 728),
        (True, 728, 1, 768),
        (False, 20, 3, 60),
        (True, 20, 3, 64),
    ],
)
def test_decode_attention_preserves_batch_invariant_token_padding(
    monkeypatch, batch_invariant_mode, num_requests, tokens_per_request, padded_token_count
):
    """Only batch-invariant token-only rows bypass the attention kernel."""
    attention = object.__new__(SelfAttention)
    torch.nn.Module.__init__(attention)
    attention.config = SimpleNamespace(window_size=None, window_attn_skip_freq=None)
    attention.layer_number = 1
    attention.batch_invariant_mode = batch_invariant_mode
    attention.flash_attention_version = 4
    attention.train(False)

    kernel_queries = []

    def fake_fa4_varlen(q, _k, _v, **_kwargs):
        kernel_queries.append(q.clone())
        return q.clone(), None

    monkeypatch.setattr(attention_module, "HAVE_FA4", True)
    monkeypatch.setattr(attention_module, "flash_attn4_varlen_func", fake_fa4_varlen, raising=False)

    metadata_token_count = num_requests * tokens_per_request
    num_heads = 2
    head_dim = 4
    q = torch.arange(padded_token_count * num_heads * head_dim, dtype=torch.float32).reshape(
        padded_token_count, 1, num_heads, head_dim
    )
    cu_seqlens = torch.arange(0, metadata_token_count + 1, tokens_per_request, dtype=torch.int32)
    seqlens_k = torch.ones(num_requests, dtype=torch.int32)

    output = attention.flash_decode_and_prefill(
        q=q,
        k=torch.empty(0),
        v=torch.empty(0),
        max_seqlen_q=tokens_per_request,
        max_seqlen_k=1,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        seqlens_k=seqlens_k,
        block_table=torch.zeros((num_requests, 1), dtype=torch.int32),
        is_decode_only=True,
    )

    assert kernel_queries[0].shape == (metadata_token_count, num_heads, head_dim)
    assert torch.equal(kernel_queries[0], q[:metadata_token_count, 0])
    assert output.shape == q.shape
    assert torch.equal(output[:metadata_token_count], q[:metadata_token_count])
    assert torch.count_nonzero(output[metadata_token_count:]) == 0
