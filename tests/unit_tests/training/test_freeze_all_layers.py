# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the --freeze-all-layers helper in megatron.training.training.

These exercise ``_freeze_all_model_chunks`` in isolation. The helper only calls
``requires_grad_(False)`` and flips a plain python attribute, so it runs on CPU
and needs neither CUDA nor a real Megatron model.
"""

import torch

from megatron.training.training import _freeze_all_model_chunks


class _FakeRouter(torch.nn.Module):
    """Stand-in for an MoE router that carries ``frozen_expert_bias`` (see
    ``megatron/core/transformer/moe/router.py``)."""

    def __init__(self):
        super().__init__()
        self.gate = torch.nn.Linear(4, 2)
        self.frozen_expert_bias = False


class _FakeModelChunk(torch.nn.Module):
    """Minimal module tree: some trainable params plus a router submodule."""

    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Linear(4, 8)
        self.router = _FakeRouter()
        self.output_layer = torch.nn.Linear(8, 4)


def _all_require_grad(module, value):
    return all(p.requires_grad is value for p in module.parameters())


def test_freezes_every_parameter():
    """All parameters across all chunks end up with requires_grad=False."""
    chunks = [_FakeModelChunk(), _FakeModelChunk()]
    assert all(_all_require_grad(c, True) for c in chunks), "params start trainable"

    _freeze_all_model_chunks(chunks)

    assert all(_all_require_grad(c, False) for c in chunks)


def test_sets_frozen_expert_bias():
    """Modules exposing ``frozen_expert_bias`` are flipped to True; others are
    left alone."""
    chunk = _FakeModelChunk()
    assert chunk.router.frozen_expert_bias is False

    _freeze_all_model_chunks([chunk])

    assert chunk.router.frozen_expert_bias is True
    # A module without the attribute must not gain one.
    assert not hasattr(chunk.embedding, "frozen_expert_bias")


def test_returns_same_list_object():
    """The helper freezes in place and returns the list it was given."""
    chunks = [_FakeModelChunk()]

    returned = _freeze_all_model_chunks(chunks)

    assert returned is chunks


def test_handles_multiple_routers_and_pp_style_chunks():
    """A VPP/PP-style list with several chunks, each with its own router, is
    fully handled."""
    chunks = [_FakeModelChunk() for _ in range(3)]

    _freeze_all_model_chunks(chunks)

    for chunk in chunks:
        assert _all_require_grad(chunk, False)
        assert chunk.router.frozen_expert_bias is True


def test_empty_list_is_noop():
    """An empty chunk list is accepted and returned unchanged."""
    assert _freeze_all_model_chunks([]) == []


def test_idempotent():
    """Applying the freeze twice keeps everything frozen."""
    chunk = _FakeModelChunk()

    _freeze_all_model_chunks([chunk])
    _freeze_all_model_chunks([chunk])

    assert _all_require_grad(chunk, False)
    assert chunk.router.frozen_expert_bias is True
