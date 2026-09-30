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
    attention.config = SimpleNamespace(
        window_size=None, window_attn_skip_freq=None, attn_logit_softcapping=None
    )
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
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (10, 0))

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


@pytest.mark.parametrize("is_decode_only", [False, True])
@pytest.mark.parametrize("batch_invariant_mode", [False, True])
@pytest.mark.parametrize("device_capability", [(9, 0), (10, 0), (11, 0)])
def test_fa4_split_kv_respects_device_support(
    monkeypatch, is_decode_only, batch_invariant_mode, device_capability
):
    """Hopper cannot use SplitKV; supported devices retain automatic splitting."""
    attention = object.__new__(SelfAttention)
    torch.nn.Module.__init__(attention)
    attention.config = SimpleNamespace(
        window_size=None, window_attn_skip_freq=None, attn_logit_softcapping=None
    )
    attention.layer_number = 1
    attention.batch_invariant_mode = batch_invariant_mode
    attention.flash_attention_version = 4
    attention.train(False)

    q = torch.ones(2, 1, 4, 8)
    split_counts = []

    def get_device_capability(device):
        assert device == q.device
        return device_capability

    def fake_fa4_varlen(query, _k, _v, *, num_splits, **_kwargs):
        if device_capability == (9, 0):
            assert num_splits == 1, "SplitKV not supported on SM 9.0"
        split_counts.append(num_splits)
        return query.clone(), None

    monkeypatch.setattr(torch.cuda, "get_device_capability", get_device_capability)
    monkeypatch.setattr(attention_module, "HAVE_FA4", True)
    monkeypatch.setattr(attention_module, "flash_attn4_varlen_func", fake_fa4_varlen)

    output = attention.flash_decode_and_prefill(
        q=q,
        k=torch.empty(0),
        v=torch.empty(0),
        max_seqlen_q=1,
        max_seqlen_k=1024,
        cu_seqlens_q=torch.tensor([0, 1, 2], dtype=torch.int32),
        cu_seqlens_k=None,
        seqlens_k=torch.tensor([1024, 1024], dtype=torch.int32),
        block_table=torch.zeros((2, 16), dtype=torch.int32),
        is_decode_only=is_decode_only,
    )

    assert split_counts == [1 if batch_invariant_mode or device_capability == (9, 0) else 0]
    assert torch.equal(output, q)


@pytest.mark.parametrize("is_decode_only", [False, True])
@pytest.mark.parametrize("has_sink", [False, True])
def test_fa4_requests_lse_for_sink_correction(monkeypatch, is_decode_only, has_sink):
    """FA4 omits LSE during inference unless the caller explicitly requests it."""
    attention = object.__new__(SelfAttention)
    torch.nn.Module.__init__(attention)
    attention.config = SimpleNamespace(
        window_size=None, window_attn_skip_freq=None, attn_logit_softcapping=None
    )
    attention.layer_number = 1
    attention.batch_invariant_mode = False
    attention.flash_attention_version = 4
    attention.train(False)

    q = torch.ones(2, 1, 4, 8)
    offset = torch.arange(4, dtype=torch.float32) if has_sink else None
    requested_lse = []

    def fake_fa4_varlen(query, _k, _v, *, return_lse=False, **_kwargs):
        requested_lse.append(return_lse)
        lse = torch.zeros(4, 2) if return_lse else None
        return query.clone(), lse

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (10, 0))
    monkeypatch.setattr(attention_module, "HAVE_FA4", True)
    monkeypatch.setattr(attention_module, "flash_attn4_varlen_func", fake_fa4_varlen)

    output = attention.flash_decode_and_prefill(
        q=q,
        k=torch.empty(0),
        v=torch.empty(0),
        max_seqlen_q=1,
        max_seqlen_k=2,
        cu_seqlens_q=torch.tensor([0, 1, 2], dtype=torch.int32),
        cu_seqlens_k=None,
        seqlens_k=torch.tensor([2, 2], dtype=torch.int32),
        block_table=torch.zeros((2, 1), dtype=torch.int32),
        is_decode_only=is_decode_only,
        softmax_offset=offset,
    )

    assert requested_lse == [has_sink]
    expected = q * torch.sigmoid(-offset).view(1, 1, 4, 1) if has_sink else q
    torch.testing.assert_close(output, expected)
