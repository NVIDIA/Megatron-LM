# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the --freeze-all-layers helpers in megatron.training.training.

These exercise ``_freeze_all_model_chunks`` and ``_forward_backward_grad_context``
in isolation. Both operate on plain python objects (``requires_grad_``, an
attribute flip, and a grad context), so they run on CPU and need neither CUDA nor
a real Megatron model. The grad-context tests reproduce the PP>1 case that
motivates the fix: a recv_prev input activation with ``requires_grad=True``.
"""

from contextlib import nullcontext
from types import SimpleNamespace

import torch

from megatron.training.training import _forward_backward_grad_context, _freeze_all_model_chunks


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


# ---------------------------------------------------------------------------
# _forward_backward_grad_context
#
# The helper returns a ``(grad_context, forward_only)`` tuple that a frozen train
# step uses to mirror the eval forward pass:
#   * grad_context is ``torch.no_grad()`` when frozen, else a no-op context;
#   * forward_only is True when frozen, so the schedule skips the backward and
#     finalize-grads collectives entirely.
#
# Why the grad_context matters for PP>1: on a non-first pipeline stage the input
# activation is received via ``create_tensor_recv_prev()``
# (``megatron/core/pipeline_parallel/p2p_communication.py``), which allocates it
# with ``requires_grad=True`` so gradients can flow back to the prior stage.
# That means a fully *frozen* model still builds an autograd graph during forward
# -- purely because its input requires grad -- retaining activations for a
# backward that is never useful. ``forward_only`` alone does not prevent that
# graph (it only skips the backward call); ``torch.no_grad()`` is what suppresses
# it on frozen (e.g. teacher logits dump) runs.
# ---------------------------------------------------------------------------


def _recv_prev_activation():
    """A stand-in for the PP>1 stage input from ``create_tensor_recv_prev()``:
    an activation tensor allocated with ``requires_grad=True``."""
    return torch.ones(2, 4, requires_grad=True)


def _forward_through_frozen_stage(chunk, recv_prev):
    """Run a recv_prev activation through a frozen model chunk (Linear stack)."""
    return chunk.output_layer(chunk.embedding(recv_prev))


def test_frozen_returns_no_grad_and_forward_only():
    """With --freeze-all-layers: no_grad context and forward_only=True."""
    args = SimpleNamespace(freeze_all_layers=True)

    grad_context, forward_only = _forward_backward_grad_context(args)

    assert isinstance(grad_context, torch.no_grad)
    assert forward_only is True


def test_unfrozen_returns_nullcontext_and_not_forward_only():
    """Without --freeze-all-layers: no-op context and forward_only=False."""
    args = SimpleNamespace(freeze_all_layers=False)

    grad_context, forward_only = _forward_backward_grad_context(args)

    assert isinstance(grad_context, nullcontext)
    assert forward_only is False


def test_missing_flag_defaults_to_not_frozen():
    """A minimal args mock (no freeze_all_layers attr) is treated as unfrozen."""
    grad_context, forward_only = _forward_backward_grad_context(SimpleNamespace())

    assert isinstance(grad_context, nullcontext)
    assert forward_only is False


def test_pp_gt1_frozen_forward_without_context_still_builds_graph():
    """Regression: a frozen PP>1 stage builds a graph anyway, because the
    recv_prev input requires grad. This is the situation the fix addresses."""
    chunk = _FakeModelChunk()
    _freeze_all_model_chunks([chunk])
    assert _all_require_grad(chunk, False)  # every parameter is frozen

    out = _forward_through_frozen_stage(chunk, _recv_prev_activation())

    # Graph built despite all params frozen -- solely due to the recv_prev input.
    assert out.requires_grad
    assert out.grad_fn is not None


def test_pp_gt1_frozen_forward_under_context_skips_graph():
    """The fix: running the same frozen PP>1 forward under the freeze context
    suppresses the autograd graph even though recv_prev requires grad."""
    chunk = _FakeModelChunk()
    _freeze_all_model_chunks([chunk])
    args = SimpleNamespace(freeze_all_layers=True)

    grad_context, _ = _forward_backward_grad_context(args)
    with grad_context:
        assert not torch.is_grad_enabled()
        out = _forward_through_frozen_stage(chunk, _recv_prev_activation())

    assert out.grad_fn is None  # no graph, no retained activations
    assert torch.is_grad_enabled()  # grad state restored on exit


def test_unfrozen_forward_builds_graph_normally():
    """The no-op context leaves normal (trainable) training untouched: a forward
    over a recv_prev input still builds its graph."""
    chunk = _FakeModelChunk()  # params trainable
    args = SimpleNamespace(freeze_all_layers=False)

    grad_context, _ = _forward_backward_grad_context(args)
    with grad_context:
        assert torch.is_grad_enabled()
        out = _forward_through_frozen_stage(chunk, _recv_prev_activation())

    assert out.grad_fn is not None
