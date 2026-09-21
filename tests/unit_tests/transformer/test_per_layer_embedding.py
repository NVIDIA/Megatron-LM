# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for the Qwen4-Exp per-layer (hashed n-gram) embedding.

Reference implementations are ports of the HuggingFace ``Qwen4ExpTextNGramEmbedding`` /
``Qwen4ExpTextPLELayer`` modules (single, unpacked sequences, no cache).
"""

import math

import pytest
import torch
import torch.nn.functional as F

from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.hyper_connection import gated_residual_group_rmsnorm
from megatron.core.transformer.per_layer_embedding import (
    NGramEmbedding,
    PerLayerEmbedding,
    build_ngram_layer_multipliers,
    build_ngram_vocab_layout,
    find_nth_prime_after,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

EOS = 7


def _reference_shift_right_ignore_eos(token_ids, shift, eos):
    """HF Qwen4ExpTextNGramEmbedding._shift_right_ignore_eos."""
    if shift == 0:
        return token_ids
    batch_size, seq_len = token_ids.shape
    positions = torch.arange(seq_len, device=token_ids.device, dtype=torch.long)
    eos_positions = torch.where(token_ids == eos, positions, -1)
    previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
    previous_eos = torch.cat(
        [eos_positions.new_full((batch_size, 1), -1), previous_eos_inclusive[:, :-1]], dim=1
    )
    segment_start = previous_eos + 1
    position_in_segment = positions.unsqueeze(0) - segment_start
    source_positions = positions - shift
    gather_positions = source_positions.clamp_min(0).unsqueeze(0).expand(batch_size, -1)
    shifted = token_ids.gather(dim=1, index=gather_positions)
    valid = (position_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)
    return torch.where(valid, shifted, token_ids.new_full((), eos))


def _reference_ngram_ids(module: NGramEmbedding, input_ids):
    """HF forward without cache: EOS-filled context prepended, then hashed per head."""
    ctx = input_ids.new_full((input_ids.shape[0], module.ngram_size - 1), module.eos_token_id)
    history = torch.cat([ctx, input_ids], dim=-1)
    shifted = [
        _reference_shift_right_ignore_eos(history, k, module.eos_token_id)
        for k in range(module.ngram_size)
    ]
    blocks = []
    for ngram in range(2, module.ngram_size + 1):
        start = (ngram - 2) * module.heads_per_ngram
        end = start + module.heads_per_ngram
        mixed = shifted[0] * module.layer_multipliers[0]
        for pos in range(1, ngram):
            mixed = torch.bitwise_xor(mixed, shifted[pos] * module.layer_multipliers[pos])
        sizes = module.ngram_heads_vocab_sizes[start:end]
        offsets = module.ngram_heads_offsets[start:end]
        blocks.append(
            torch.remainder(mixed.unsqueeze(-1), sizes.view(1, 1, -1)) + offsets.view(1, 1, -1)
        )
    return torch.cat(blocks, dim=-1)[:, -input_ids.shape[1] :]


def _reference_ple(module: PerLayerEmbedding, hidden_states_bsd, input_ids):
    """HF Qwen4ExpTextPLELayer.forward for [b, s, n*C] inputs without cache/mask."""
    n, C = module.n, module.hidden_size
    ids = _reference_ngram_ids(module.ple_embedding, input_ids)
    emb = F.embedding(ids, module.ple_embedding.ngram_embedding.weight).flatten(-2)
    key = gated_residual_group_rmsnorm(
        F.linear(emb, module.key_proj.weight), module.norm_key.weight, n, module.norm_eps
    )
    key = key.unflatten(-1, (n, C))
    value = F.linear(emb, module.value_proj.weight)
    query = gated_residual_group_rmsnorm(
        hidden_states_bsd, module.norm_query.weight, n, module.norm_eps
    )
    query = query.unflatten(-1, (n, C))
    gate = (key * query).sum(dim=-1, keepdim=True) / math.sqrt(C)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    gated_value = torch.sigmoid(gate) * value.unsqueeze(-2)
    gated_value_normed = gated_residual_group_rmsnorm(
        gated_value.flatten(-2), module.norm_conv.weight, n, module.norm_eps
    )
    gated_value = gated_value.flatten(-2)
    # dilated depthwise causal conv over the sequence
    x = gated_value_normed.transpose(1, 2)
    x = F.pad(x, (module.halo, 0))
    conv = F.conv1d(x, module.conv1d.weight, groups=x.shape[1], dilation=module.conv_dilation)
    conv = F.silu(conv).transpose(1, 2)
    return gated_value + conv


def _make_config(hidden_size=16, n=2, vocab=64):
    return TransformerConfig(
        num_layers=2,
        hidden_size=hidden_size,
        num_attention_heads=4,
        use_cpu_initialization=False,
        enable_mhc_connections=True,
        mhc_variant="gated_residual",
        mhc_num_residual_streams=n,
        mhc_gated_residual_rank=4,
        ple_layer_ids=[2],
        ple_embed_dim=hidden_size,
        ple_ngram_size=3,
        ple_heads_per_ngram=2,
        ple_ngram_vocab_size_base=97,
        ple_ngram_vocab_divisible_by=8,
        ple_eos_token_id=EOS,
        ple_unigram_vocab_size=vocab,
        params_dtype=torch.float32,
        layernorm_epsilon=1e-6,
    )


class TestNGramHashing:
    def test_primes_and_layout(self):
        assert find_nth_prime_after(96, 1) == 97
        assert find_nth_prime_after(96, 2) == 101
        sizes, offsets, total = build_ngram_vocab_layout(97, 4, ple_layer_index=0)
        assert sizes == [97, 101, 103, 107]
        assert offsets == [0, 97, 198, 301]
        assert total == 408
        # A second PLE module continues the prime sequence.
        sizes2, _, _ = build_ngram_vocab_layout(97, 4, ple_layer_index=1)
        assert sizes2[0] == 109

    def test_multipliers_are_odd_and_deterministic(self):
        m1 = build_ngram_layer_multipliers(248320, 3, 0, 1234)
        m2 = build_ngram_layer_multipliers(248320, 3, 0, 1234)
        assert m1 == m2 and all(m % 2 == 1 for m in m1)
        assert max(m1) * 248320 < (1 << 63)
        assert m1 != build_ngram_layer_multipliers(248320, 3, 1, 1234)


class TestPerLayerEmbedding:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_ngram_ids_match_reference(self):
        config = _make_config()
        module = PerLayerEmbedding(config, layer_number=2).cuda()
        ids = torch.randint(0, 64, (2, 11), device="cuda")
        ids[0, 4] = EOS
        ids[1, 0] = EOS
        got = module.ple_embedding.compute_ngram_ids(ids)
        ref = _reference_ngram_ids(module.ple_embedding, ids)
        torch.testing.assert_close(got, ref)

    def test_packed_sequences_reset_context(self):
        config = _make_config()
        module = PerLayerEmbedding(config, layer_number=2).cuda()
        a = torch.randint(0, 64, (1, 6), device="cuda")
        b = torch.randint(0, 64, (1, 9), device="cuda")
        packed = torch.cat([a, b], dim=1)
        cu_seqlens = torch.tensor([0, 6, 15], device="cuda", dtype=torch.int32)
        got = module.ple_embedding.compute_ngram_ids(packed, cu_seqlens=cu_seqlens)
        ref = torch.cat(
            [
                _reference_ngram_ids(module.ple_embedding, a),
                _reference_ngram_ids(module.ple_embedding, b),
            ],
            dim=1,
        )
        torch.testing.assert_close(got, ref)

    def test_forward_matches_reference(self):
        config = _make_config()
        module = PerLayerEmbedding(config, layer_number=2).cuda()
        with torch.no_grad():
            module.conv1d.weight.normal_(0, 0.3)
            for norm in (module.norm_key, module.norm_query, module.norm_conv):
                norm.weight.normal_(0, 0.1)
        s, b = 13, 2
        ids = torch.randint(0, 64, (b, s), device="cuda")
        ids[1, 5] = EOS
        hidden = torch.randn(
            s, b, config.mhc_num_residual_streams * config.hidden_size, device="cuda"
        )

        module.prepare(ids)
        out = module(hidden)

        ref = _reference_ple(module, hidden.transpose(0, 1), ids).transpose(0, 1)
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    def test_forward_packed_matches_per_document_reference(self):
        config = _make_config()
        module = PerLayerEmbedding(config, layer_number=2).cuda()
        with torch.no_grad():
            module.conv1d.weight.normal_(0, 0.3)
        lens = [5, 12, 3]
        ids = torch.randint(0, 64, (1, sum(lens)), device="cuda")
        hidden = torch.randn(
            sum(lens), 1, config.mhc_num_residual_streams * config.hidden_size, device="cuda"
        )
        cu = torch.tensor(
            [0] + list(torch.tensor(lens).cumsum(0).tolist()), device="cuda", dtype=torch.int32
        )

        module.prepare(ids, cu_seqlens=cu)
        out = module(hidden)

        refs, start = [], 0
        for length in lens:
            h = hidden[start : start + length].transpose(0, 1)
            refs.append(_reference_ple(module, h, ids[:, start : start + length]).transpose(0, 1))
            start += length
        torch.testing.assert_close(out, torch.cat(refs, dim=0), atol=1e-5, rtol=1e-5)

    def test_requires_prepare(self):
        config = _make_config()
        module = PerLayerEmbedding(config, layer_number=2).cuda()
        with pytest.raises(RuntimeError):
            module(
                torch.randn(
                    3, 1, config.mhc_num_residual_streams * config.hidden_size, device="cuda"
                )
            )
