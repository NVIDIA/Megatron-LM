# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DeepSeek-V4's half of the cross-entropy fusion wiring."""

from __future__ import annotations

import dataclasses

import pytest

from megatron.lite.model.deepseek_v4.lite import protocol as ds4_protocol
from megatron.lite.model.deepseek_v4.lite.protocol import ImplConfig

pytestmark = [pytest.mark.mlite]


def test_impl_config_exposes_cross_entropy_fusion() -> None:
    """The switch exists and is reachable; see the config for why it stays off."""
    names = {f.name for f in dataclasses.fields(ImplConfig)}
    assert "cross_entropy_fusion" in names
    assert ImplConfig().cross_entropy_fusion is False


@pytest.mark.parametrize("enabled", [True, False])
def test_forward_step_passes_use_fused_kernels(monkeypatch, enabled: bool) -> None:
    """``_forward_step`` must consult the flag and forward it to the model."""

    class _Model:
        def __init__(self):
            # An instance attribute, as ``set_cross_entropy_fusion`` writes it:
            # the resolver reads ``vars(module)`` and would miss a class attribute.
            self.cross_entropy_fusion = enabled
            self.seen: dict = {}

        def __call__(self, **kwargs):
            self.seen = kwargs
            return {"loss": None}

    model = _Model()
    monkeypatch.setattr(
        ds4_protocol, "_prepare_model_forward_kwargs", lambda _model, _batch: {"input_ids": None}
    )
    ds4_protocol._forward_step(model, object())

    assert model.seen["use_fused_kernels"] is enabled
    # The rest of the kwargs must survive untouched.
    assert "input_ids" in model.seen
