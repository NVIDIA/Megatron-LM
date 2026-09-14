# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU tests for the Engram bucket layout, hashing and the frozen memory module."""

import pytest
import torch

from megatron.core.models.deepseek_v41.engram import (
    EngramMemory,
    EngramTableLayout,
    NgramHasher,
    hash_multipliers,
    is_prime,
    next_unused_prime,
)
from megatron.core.transformer.transformer_config import TransformerConfig


def _layout(max_ngram=4, n_heads=3, bucket=97, layers=(1, 4), rows=None):
    n_cols = (max_ngram - 1) * n_heads
    # generous: the primes drawn for later layers sit further above ``bucket``
    rows = rows or [n_cols * (bucket + 200)] * len(layers)
    return EngramTableLayout.build(
        layer_ids=layers,
        num_embeddings=rows,
        max_ngram_size=max_ngram,
        n_heads=n_heads,
        head_dim=8,
        bucket_size=bucket,
    )


class TestPrimes:
    def test_is_prime(self):
        primes = [n for n in range(2, 60) if is_prime(n)]
        assert primes == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]
        assert is_prime(16_000_057)
        assert not is_prime(16_000_000) and not is_prime(1) and not is_prime(0)

    def test_next_unused_prime(self):
        used = set()
        first = next_unused_prime(96, used)
        used.add(first)
        second = next_unused_prime(96, used)
        assert first == 97 and second == 101

    def test_layout_disjoint_and_ordered(self):
        layout = _layout()
        seen = set()
        for layer in layout.primes:
            for per_ngram in layer:
                for p in per_ngram:
                    assert is_prime(p) and p >= 97
                    assert p not in seen
                    seen.add(p)
        assert layout.n_hash_cols == 9
        offsets = layout.offsets(0)
        assert offsets[0] == 0
        assert offsets == [sum(layout.flat_primes(0)[:i]) for i in range(9)]
        assert layout.rows_required(0) <= layout.num_embeddings[0]

    def test_layout_rejects_small_table(self):
        with pytest.raises(ValueError, match="rows"):
            _layout(rows=[10, 10])


class TestMultipliers:
    def test_odd_bounded_deterministic(self):
        m1 = hash_multipliers([1, 4], 4, 1000)
        m2 = hash_multipliers([1, 4], 4, 1000)
        assert torch.equal(m1, m2)
        assert m1.shape == (2, 4) and m1.dtype == torch.int64
        assert (m1 % 2 == 1).all()
        bound = (torch.iinfo(torch.int64).max // 1000) // 2
        assert (m1 <= 2 * bound + 1).all()
        assert not torch.equal(m1[0], m1[1])


class TestHasher:
    def _hasher(self, vocab=50):
        layout = _layout()
        token_map = torch.arange(vocab)
        return layout, NgramHasher(layout, token_map, compressed_vocab_size=vocab, pad_token_id=2)

    def test_shapes_and_ranges(self):
        layout, hasher = self._hasher()
        ids = torch.randint(0, 50, (2, 11))
        rows = hasher(ids)
        assert rows.shape == (11, 2, 2, layout.n_hash_cols)
        for layer_index in range(2):
            flat = layout.flat_primes(layer_index)
            offsets = layout.offsets(layer_index)
            for col in range(layout.n_hash_cols):
                column = rows[:, :, layer_index, col]
                assert (column >= offsets[col]).all()
                assert (column < offsets[col] + flat[col]).all()

    def test_same_ngram_same_row(self):
        _, hasher = self._hasher()
        # identical 4-token histories (7, 3, 9, 4) at two positions and in two batch rows
        ids = torch.tensor([[7, 3, 9, 4, 7, 3, 9, 4], [5, 7, 3, 9, 4, 8, 8, 8]])
        rows = hasher(ids)
        assert torch.equal(rows[3, 0], rows[7, 0])
        assert torch.equal(rows[3, 0], rows[4, 1])
        # (1, 3, 9, 4) shares the 2- and 3-gram columns with (7, 3, 9, 4) but not the 4-gram
        ids2 = torch.tensor([[1, 3, 9, 4]])
        rows2 = hasher(ids2)
        n_heads = 3
        assert torch.equal(rows2[3, 0, :, : 2 * n_heads], rows[3, 0, :, : 2 * n_heads])
        assert not torch.equal(rows2[3, 0, :, 2 * n_heads :], rows[3, 0, :, 2 * n_heads :])
        # a different history gives different rows
        assert not torch.equal(rows[3, 0], rows[2, 0])

    def test_start_of_sequence_uses_pad(self):
        _, hasher = self._hasher()
        ids = torch.tensor([[7, 7, 7, 7, 7]])
        rows = hasher(ids)
        # From position 3 on, every 4-gram is (7,7,7,7) and rows repeat exactly.
        assert torch.equal(rows[3, 0], rows[4, 0])
        # Position 0 hashes (7, pad, pad, pad): differs from the full history.
        assert not torch.equal(rows[0, 0], rows[3, 0])
        # The 2-gram columns at position 1 (7,7) equal those at position 3 (7,7).
        n_heads = 3
        assert torch.equal(rows[1, 0, :, :n_heads], rows[3, 0, :, :n_heads])

    def test_token_map_applied(self):
        layout = _layout()
        token_map = torch.zeros(50, dtype=torch.long)  # every token collapses to id 0
        hasher = NgramHasher(layout, token_map, compressed_vocab_size=50, pad_token_id=2)
        ids = torch.randint(0, 50, (1, 6))
        rows = hasher(ids)
        assert torch.equal(rows[3, 0], rows[5, 0])

    def test_rejects_map_out_of_range(self):
        layout = _layout()
        with pytest.raises(ValueError, match="compressed_vocab_size"):
            NgramHasher(layout, torch.arange(50), compressed_vocab_size=10, pad_token_id=2)


class TestEngramMemory:
    def _config(self):
        return TransformerConfig(
            num_layers=2,
            hidden_size=16,
            num_attention_heads=2,
            use_cpu_initialization=True,
            params_dtype=torch.float32,
            enable_hyper_connections=True,
            num_residual_streams=3,
            hidden_dropout=0.0,
        )

    def test_forward_frozen(self):
        config = self._config()
        layout = _layout(max_ngram=3, n_heads=2, bucket=31, layers=(1,))
        memory = EngramMemory(config, layout, layer_index=0)
        assert all(not p.requires_grad for p in memory.parameters())
        s, b = 5, 2
        hidden = torch.randn(s, b, 3 * 16, requires_grad=True)
        rows = torch.randint(0, layout.rows_required(0), (s, b, layout.n_hash_cols))
        out = memory(hidden, rows)
        assert out.shape == hidden.shape
        assert torch.isfinite(out).all()
        # residual streams still carry gradient through the gate
        out.sum().backward()
        assert hidden.grad is not None and torch.isfinite(hidden.grad).all()

    def test_gate_bounds_change(self):
        config = self._config()
        layout = _layout(max_ngram=3, n_heads=2, bucket=31, layers=(1,))
        memory = EngramMemory(config, layout, layer_index=0)
        hidden = torch.randn(4, 1, 3 * 16)
        rows = torch.randint(0, layout.rows_required(0), (4, 1, layout.n_hash_cols))
        with torch.no_grad():
            out = memory(hidden, rows)
            value = memory.linear_wkv(memory.lookup(rows).flatten(-2))[..., -16:]
        delta = (out - hidden).view(4, 1, 3, 16)
        # each stream receives gate * value with gate in (0, 1)
        ratio = delta / value.unsqueeze(2)
        ratio = ratio[torch.isfinite(ratio) & (value.unsqueeze(2).abs() > 1e-3)]
        assert (ratio > 0).all() and (ratio < 1).all()

    def test_token_mask_disables_gate(self):
        config = self._config()
        layout = _layout(max_ngram=3, n_heads=2, bucket=31, layers=(1,))
        memory = EngramMemory(config, layout, layer_index=0)
        hidden = torch.randn(4, 1, 3 * 16)
        rows = torch.randint(0, layout.rows_required(0), (4, 1, layout.n_hash_cols))
        mask = torch.zeros(4, 1, dtype=torch.bool)
        out = memory(hidden, rows, token_mask=mask)
        torch.testing.assert_close(out, hidden)


class TestHasherShortSequences:
    @pytest.mark.parametrize("length", [1, 2, 3, 5])
    def test_short_sequences(self, length):
        layout = _layout()
        hasher = NgramHasher(layout, torch.arange(50), compressed_vocab_size=50, pad_token_id=2)
        ids = torch.randint(0, 50, (2, length))
        rows = hasher(ids)
        assert rows.shape == (length, 2, 2, layout.n_hash_cols)
        assert (rows >= 0).all()
        # a one-token sequence hashes (t, pad, pad, pad) like position 0 of a longer sequence
        longer = hasher(torch.cat([ids, torch.randint(0, 50, (2, 4))], dim=1))
        assert torch.equal(rows, longer[:length])


class TestHasherPacked:
    def test_segments_block_lookback(self):
        layout = _layout()
        hasher = NgramHasher(layout, torch.arange(50), compressed_vocab_size=50, pad_token_id=2)
        a = torch.tensor([[7, 3, 9, 4]])
        b = torch.tensor([[5, 7, 3, 9, 4, 8]])
        packed = torch.cat([a, b], dim=1)  # two segments back to back
        cu = torch.tensor([0, 4, 10])
        rows = hasher(packed, cu_seqlens=cu)
        assert torch.equal(rows[:4, 0], hasher(a)[:, 0])
        assert torch.equal(rows[4:, 0], hasher(b)[:, 0])
        # without cu_seqlens the second segment's start would see the first segment
        assert not torch.equal(hasher(packed)[4, 0], rows[4, 0])
