# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""A dense multi-row batch must be refused where the caller can still act on it."""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.mlite]


class _FakeCSA(torch.nn.Module):
    pass


def _model_with(attention: torch.nn.Module) -> torch.nn.Module:
    model = torch.nn.Module()
    model.attn = attention
    return model


def test_dense_multi_row_batch_is_refused_at_the_boundary() -> None:
    """``[B, S]`` with ``B > 1`` must raise here, naming the constraint."""
    from megatron.lite.model.deepseek_v4.lite import protocol
    from megatron.lite.primitive.modules.attention.csa import CompressedSparseAttention

    model = _model_with(CompressedSparseAttention.__new__(CompressedSparseAttention))
    torch.nn.Module.__init__(model.attn)

    batch = type("B", (), {"input_ids": torch.zeros(4, 8, dtype=torch.long)})()
    with pytest.raises(NotImplementedError) as excinfo:
        protocol._prepare_model_forward_kwargs(model, batch)

    message = str(excinfo.value)
    assert "B=4" in message, "the message must name the batch that was rejected"
    assert "pack" in message.lower(), "the message must name the way out"


@pytest.mark.parametrize("shape", [(8,), (1, 8)])
def test_packed_shapes_still_take_the_packed_route(shape: tuple[int, ...]) -> None:
    """Guard the guard: the refusal must not swallow the shapes that do work."""
    from megatron.lite.model.deepseek_v4.lite import protocol

    called = {}

    def _fake_packed(model, batch):
        called["yes"] = True
        return {}

    original = protocol._prepare_packed_batch_kwargs
    protocol._prepare_packed_batch_kwargs = _fake_packed
    try:
        batch = type("B", (), {"input_ids": torch.zeros(*shape, dtype=torch.long)})()
        protocol._prepare_model_forward_kwargs(_model_with(_FakeCSA()), batch)
    finally:
        protocol._prepare_packed_batch_kwargs = original

    assert called.get("yes"), f"shape {shape} should have taken the packed route"


def test_models_without_csa_keep_the_dense_route() -> None:
    """The refusal is CSA's constraint, not the batch builder's."""
    from megatron.lite.model.deepseek_v4.lite import protocol

    reached = {}

    def _fake_dense(model, kwargs):
        reached["yes"] = True
        return kwargs

    original_dense = protocol._prepare_contiguous_cp_kwargs
    original_base = protocol._base_model_forward_kwargs
    protocol._prepare_contiguous_cp_kwargs = _fake_dense
    protocol._base_model_forward_kwargs = lambda batch: {}
    try:
        batch = type("B", (), {"input_ids": torch.zeros(4, 8, dtype=torch.long)})()
        protocol._prepare_model_forward_kwargs(_model_with(_FakeCSA()), batch)
    finally:
        protocol._prepare_contiguous_cp_kwargs = original_dense
        protocol._base_model_forward_kwargs = original_base

    assert reached.get("yes"), "a non-CSA model must still reach the dense builder"
