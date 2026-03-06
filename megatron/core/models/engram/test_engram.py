# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""
End-to-end inference test for the Engram module.

Mirrors the reference implementation (engram_demo_v1.py) to verify the
Megatron port produces a correct forward pass through a mock LLM backbone
with Engram-augmented layers, and that the engram computation matches the
reference step-by-step.

Usage:
    pytest megatron/core/models/engram/test_engram_shapes.py -v
"""

import math

import numpy as np
import torch
import torch.nn as nn

from megatron.core.models.engram.engram_module import EngramConfig, EngramModule, _find_next_prime

HIDDEN_SIZE = 128
HC_MULT = 4
VOCAB_SIZE = 1000
NUM_LAYERS = 5
ENGRAM_LAYER_IDS = [1, 3]
B, S = 2, 16


def _make_engram_config():
    return EngramConfig(
        engram_vocab_size=[1000, 1000],
        max_ngram_size=3,
        n_embed_per_ngram=512,
        n_head_per_ngram=8,
        engram_layer_ids=ENGRAM_LAYER_IDS,
        pad_id=2,
        seed=0,
        kernel_size=4,
        hc_mult=HC_MULT,
    )


def _make_vocab_sizes_across_layers(cfg):
    """Build per-head vocab sizes using the same prime logic as the reference."""
    seen_primes: set = set()
    vocab_sizes = {}
    for layer_id in cfg.engram_layer_ids:
        per_layer = []
        for ngram in range(2, cfg.max_ngram_size + 1):
            head_sizes = []
            start = cfg.engram_vocab_size[ngram - 2] - 1
            for _ in range(cfg.n_head_per_ngram):
                p = _find_next_prime(start, seen_primes)
                seen_primes.add(p)
                head_sizes.append(p)
                start = p
            per_layer.append(head_sizes)
        vocab_sizes[layer_id] = per_layer
    return vocab_sizes


class TestEngramE2EInference:
    """End-to-end inference mirroring engram_demo_v1.py's __main__ block.

    Builds a mock LLM (Embedding -> TransformerBlocks with Engram -> Linear),
    runs a full forward pass, and verifies:
      1. Output shape is [B, S, vocab_size] and finite.
      2. Engram caches are cleared after consumption.
      3. The engram computation matches the reference gate/value/conv formula
         from engram_demo_v1.py exactly (torch.testing.assert_close).
    """

    @torch.no_grad()
    def test_e2e_inference(self):
        torch.manual_seed(42)
        cfg = _make_engram_config()
        vocab_sizes = _make_vocab_sizes_across_layers(cfg)
        num_heads = (cfg.max_ngram_size - 1) * cfg.n_head_per_ngram

        # -- Build mock LLM (mirrors reference __main__) --
        embedding = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
        output_proj = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE)

        engram_modules = {}
        for layer_id in cfg.engram_layer_ids:
            engram_modules[layer_id] = EngramModule(
                layer_id=layer_id,
                hidden_size=HIDDEN_SIZE,
                engram_config=cfg,
                vocab_size_for_layer=vocab_sizes[layer_id],
            )
            engram_modules[layer_id].eval()

        # -- Synthetic inputs (no tokenizer needed) --
        input_ids = torch.randint(0, VOCAB_SIZE, (B, S))

        hash_ids_all = {}
        for layer_id in cfg.engram_layer_ids:
            flat_vocab = [v for ngram in vocab_sizes[layer_id] for v in ngram]
            hash_ids_all[layer_id] = np.stack(
                [np.random.randint(0, flat_vocab[h], (B, S)) for h in range(num_heads)], axis=2
            ).astype(np.int64)

        # -- Precompute embeddings (model-level, once per forward) --
        for layer_id, mod in engram_modules.items():
            mod.precompute_embeddings(hash_ids_all[layer_id], device=torch.device("cpu"))

        # -- Forward pass: Embed -> Layers -> Output --
        hidden = embedding(input_ids)  # [B, S, H]
        hidden = hidden.transpose(0, 1).contiguous()  # [S, B, H]

        for layer_id in range(NUM_LAYERS):
            if layer_id in engram_modules:
                engram_out = engram_modules[layer_id](hidden)
                hidden = engram_out + hidden
            hidden = hidden + hidden  # mock attn residual
            hidden = hidden + hidden  # mock moe residual

        hidden = hidden.transpose(0, 1).contiguous()  # [B, S, H]
        output = output_proj(hidden)  # [B, S, V]

        # -- Check e2e output --
        assert output.shape == (B, S, VOCAB_SIZE)
        assert torch.isfinite(output).all(), "Output contains NaN or Inf"

        for mod in engram_modules.values():
            assert mod._cached_embeddings is None, "Cache not cleared"

        # -- Verify engram forward matches reference step-by-step --
        # Reproduce the gate/value/conv computation from engram_demo_v1.py's
        # Engram.forward, then compare against EngramModule's output.
        layer_id = cfg.engram_layer_ids[0]
        mod = engram_modules[layer_id]

        hash_t = torch.from_numpy(hash_ids_all[layer_id])
        embeddings = mod.multi_head_embedding(hash_t).flatten(start_dim=-2)

        test_hidden_sbh = torch.randn(S, B, HIDDEN_SIZE)
        test_hidden_bsh = test_hidden_sbh.transpose(0, 1).contiguous()
        test_hidden_hc = test_hidden_bsh.unsqueeze(2).expand(-1, -1, HC_MULT, -1)

        # Reference gate computation (engram_demo_v1.py Engram.forward lines 365-375)
        ref_gates = []
        for hc_idx in range(HC_MULT):
            key = mod.key_projs[hc_idx](embeddings)
            normed_key = mod.norm1[hc_idx](key)
            query = test_hidden_hc[:, :, hc_idx, :]
            normed_query = mod.norm2[hc_idx](query)
            gate = (normed_key * normed_query).sum(dim=-1) / math.sqrt(HIDDEN_SIZE)
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gate = gate.sigmoid().unsqueeze(-1)
            ref_gates.append(gate)
        ref_gates = torch.stack(ref_gates, dim=2)

        ref_value = ref_gates * mod.value_proj(embeddings).unsqueeze(2)
        ref_output = ref_value + mod.short_conv(ref_value)
        ref_final = ref_output.mean(dim=2).transpose(0, 1).contiguous()

        mod.precompute_embeddings(hash_ids_all[layer_id], device=torch.device("cpu"))
        actual = mod(test_hidden_sbh)

        torch.testing.assert_close(actual, ref_final)
