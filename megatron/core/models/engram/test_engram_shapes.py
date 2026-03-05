# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""
Unit tests for Engram module shapes.

Verifies that all Engram components can be instantiated and produce correct
output shapes. Tests 1-3 use synthetic data (no tokenizer needed). The
end-to-end test requires the HuggingFace tokenizer from EngramConfig.

Usage:
    pytest megatron/core/models/engram/test_engram_shapes.py -v
"""

import numpy as np
import pytest
import torch

from megatron.core.models.engram.engram_module import (
    EngramConfig,
    EngramModule,
    MultiHeadEmbedding,
    ShortConv,
)

B, S = 2, 16
HIDDEN_SIZE = 128
HC_MULT = 4


def _make_engram_config(**overrides) -> EngramConfig:
    defaults = dict(
        engram_vocab_size=[1000, 1000],
        max_ngram_size=3,
        n_embed_per_ngram=512,
        n_head_per_ngram=8,
        engram_layer_ids=[1],
        pad_id=2,
        seed=0,
        kernel_size=4,
        hc_mult=HC_MULT,
    )
    defaults.update(overrides)
    return EngramConfig(**defaults)


def _make_vocab_sizes(cfg: EngramConfig):
    """Build synthetic per-head vocab sizes (no tokenizer needed)."""
    num_ngram_levels = cfg.max_ngram_size - 1
    sizes = []
    base = 1009
    for _ in range(num_ngram_levels):
        head_sizes = [base + h * 10 for h in range(cfg.n_head_per_ngram)]
        sizes.append(head_sizes)
        base += cfg.n_head_per_ngram * 10
    return sizes


class TestShortConv:
    def test_output_shape(self):
        conv = ShortConv(
            hidden_size=HIDDEN_SIZE, kernel_size=4, dilation=3, hc_mult=HC_MULT,
        )
        x = torch.randn(B, S, HC_MULT, HIDDEN_SIZE)
        y = conv(x)
        assert y.shape == (B, S, HC_MULT, HIDDEN_SIZE)

    def test_causal_padding(self):
        """Output length must equal input length regardless of kernel/dilation."""
        for kernel, dilation in [(3, 1), (4, 3), (7, 2)]:
            conv = ShortConv(
                hidden_size=HIDDEN_SIZE,
                kernel_size=kernel,
                dilation=dilation,
                hc_mult=HC_MULT,
            )
            x = torch.randn(B, S, HC_MULT, HIDDEN_SIZE)
            y = conv(x)
            assert y.shape[1] == S, f"kernel={kernel}, dilation={dilation}"


class TestMultiHeadEmbedding:
    def test_output_shape(self):
        list_of_N = [1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049,
                     1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097]
        D = 64
        num_heads = len(list_of_N)
        emb = MultiHeadEmbedding(list_of_N=list_of_N, D=D)

        ids = torch.stack(
            [torch.randint(0, n, (B, S)) for n in list_of_N], dim=2
        )
        out = emb(ids)
        assert out.shape == (B, S, num_heads, D)

    def test_offset_isolation(self):
        """Each head's IDs should index a non-overlapping region."""
        list_of_N = [100, 200, 300]
        emb = MultiHeadEmbedding(list_of_N=list_of_N, D=8)
        assert emb.offsets.tolist() == [0, 100, 300]


class TestEngramModule:
    def _build_module(self, cfg=None):
        cfg = cfg or _make_engram_config()
        vocab_sizes = _make_vocab_sizes(cfg)
        return EngramModule(
            layer_id=cfg.engram_layer_ids[0],
            hidden_size=HIDDEN_SIZE,
            engram_config=cfg,
            vocab_size_for_layer=vocab_sizes,
        ), cfg

    def test_output_shape(self):
        module, cfg = self._build_module()
        num_heads = (cfg.max_ngram_size - 1) * cfg.n_head_per_ngram
        hash_ids = np.random.randint(
            0, 100, size=(B, S, num_heads),
        ).astype(np.int64)

        module.precompute_embeddings(hash_ids, device=torch.device("cpu"))
        hidden = torch.randn(S, B, HIDDEN_SIZE)
        out = module(hidden)
        assert out.shape == (S, B, HIDDEN_SIZE)

    def test_cache_cleared_after_forward(self):
        module, cfg = self._build_module()
        num_heads = (cfg.max_ngram_size - 1) * cfg.n_head_per_ngram
        hash_ids = np.random.randint(
            0, 100, size=(B, S, num_heads),
        ).astype(np.int64)

        module.precompute_embeddings(hash_ids, device=torch.device("cpu"))
        module(torch.randn(S, B, HIDDEN_SIZE))
        assert module._cached_embeddings is None

    def test_forward_without_precompute_raises(self):
        module, _ = self._build_module()
        with pytest.raises(AssertionError, match="precompute_embeddings"):
            module(torch.randn(S, B, HIDDEN_SIZE))


class TestEndToEnd:
    """End-to-end test through NgramHashMapping. Requires tokenizer."""

    @pytest.fixture()
    def mapping_and_cfg(self):
        from megatron.core.models.engram.engram_module import NgramHashMapping

        cfg = _make_engram_config(engram_layer_ids=[1, 15])
        try:
            mapping = NgramHashMapping(
                engram_vocab_size=cfg.engram_vocab_size,
                max_ngram_size=cfg.max_ngram_size,
                n_embed_per_ngram=cfg.n_embed_per_ngram,
                n_head_per_ngram=cfg.n_head_per_ngram,
                layer_ids=cfg.engram_layer_ids,
                tokenizer_name_or_path=cfg.tokenizer_name_or_path,
                pad_id=cfg.pad_id,
                seed=cfg.seed,
            )
        except Exception as e:
            pytest.skip(f"tokenizer not available: {e}")
        return mapping, cfg

    def test_hash_shapes(self, mapping_and_cfg):
        mapping, cfg = mapping_and_cfg
        num_heads = (cfg.max_ngram_size - 1) * cfg.n_head_per_ngram
        fake_ids = np.random.randint(0, 1000, size=(B, S))
        result = mapping.hash(fake_ids)
        for layer_id in cfg.engram_layer_ids:
            assert result[layer_id].shape == (B, S, num_heads)

    def test_full_forward(self, mapping_and_cfg):
        mapping, cfg = mapping_and_cfg
        fake_ids = np.random.randint(0, 1000, size=(B, S))
        hash_all = mapping.hash(fake_ids)

        hidden = torch.randn(S, B, HIDDEN_SIZE)
        for layer_id in cfg.engram_layer_ids:
            mod = EngramModule(
                layer_id=layer_id,
                hidden_size=HIDDEN_SIZE,
                engram_config=cfg,
                vocab_size_for_layer=mapping.vocab_size_across_layers[layer_id],
            )
            mod.precompute_embeddings(hash_all[layer_id], device=torch.device("cpu"))
            out = mod(hidden)
            assert out.shape == (S, B, HIDDEN_SIZE)
            hidden = out + hidden

        assert hidden.shape == (S, B, HIDDEN_SIZE)
