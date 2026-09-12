# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for :mod:`megatron.diffusion.two_tower.time_conditioning`.

CPU-only tests that verify the shape contracts and mathematical properties
of :class:`TimestepEmbedder`, :func:`modulate`, and
:func:`get_modulation_params`.
"""

import torch

from megatron.diffusion.two_tower.time_conditioning import (
    TimestepEmbedder,
    get_modulation_params,
    modulate,
)


class TestTimestepEmbedder:
    """Shape and value checks for :class:`TimestepEmbedder`."""

    def test_output_shape(self):
        """Output must be ``(B, hidden_size)``."""
        hidden_size = 64
        embedder = TimestepEmbedder(hidden_size)
        t = torch.rand(4)
        out = embedder(t)
        assert out.shape == (4, hidden_size)

    def test_output_finite(self):
        """Embeddings must be finite for typical timestep values."""
        embedder = TimestepEmbedder(128)
        t = torch.tensor([0.0, 0.5, 1.0])
        out = embedder(t)
        assert torch.isfinite(out).all()

    def test_different_timesteps_different_outputs(self):
        """Distinct timesteps must produce distinct embeddings."""
        embedder = TimestepEmbedder(64)
        t = torch.tensor([0.1, 0.9])
        out = embedder(t)
        assert not torch.allclose(out[0], out[1])


class TestModulate:
    """Shape and correctness checks for :func:`modulate`."""

    def test_sequence_first_shape(self):
        """Sequence-first ``(S, B, D)`` input preserves shape."""
        x = torch.randn(8, 2, 32)
        shift = torch.randn(2, 32)
        scale = torch.randn(2, 32)
        out = modulate(x, shift, scale)
        assert out.shape == x.shape

    def test_broadcast_on_sequence_dim(self):
        """Shift and scale ``(B, D)`` broadcast across the sequence dimension."""
        S, B, D = 8, 2, 32
        x = torch.ones(S, B, D)
        shift = torch.full((B, D), 1.0)
        scale = torch.full((B, D), 1.0)
        out = modulate(x, shift, scale)
        assert torch.allclose(out, torch.full_like(x, 3.0))

    def test_identity_at_zero(self):
        """With ``shift=0, scale=0``, modulation is the identity."""
        x = torch.randn(4, 2, 16)
        zero = torch.zeros(2, 16)
        out = modulate(x, zero, zero)
        assert torch.allclose(out, x)


class TestGetModulationParams:
    """Shape and decomposition tests for :func:`get_modulation_params`."""

    def test_output_shapes(self):
        """Each of shift, scale, gate must be ``(B, D)``."""
        B, D = 3, 64
        t_emb = torch.randn(B, 3 * D)
        table = torch.randn(3, D)
        shift, scale, gate = get_modulation_params(t_emb, table)
        assert shift.shape == (B, D)
        assert scale.shape == (B, D)
        assert gate.shape == (B, D)

    def test_bias_additive(self):
        """With zero t_emb, outputs equal the table rows (bias only)."""
        D = 32
        t_emb = torch.zeros(1, 3 * D)
        table = torch.randn(3, D)
        shift, scale, gate = get_modulation_params(t_emb, table)
        assert torch.allclose(shift, table[0].unsqueeze(0))
        assert torch.allclose(scale, table[1].unsqueeze(0))
        assert torch.allclose(gate, table[2].unsqueeze(0))
