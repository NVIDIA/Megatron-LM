# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DeepSeek-V4's half of the cross-entropy fusion wiring.

``DeepseekV4ForCausalLM`` has accepted ``use_fused_kernels`` since it gained the
``linear_cross_entropy`` path, which avoids materialising the full
``[seq, batch, vocab]`` logits. The DeepSeek-V4 protocol never set it: no
``cross_entropy_fusion`` field on its ``ImplConfig``, and neither
``set_cross_entropy_fusion`` at build nor ``add_cross_entropy_fusion`` at
forward. GLM-5 and Kimi-K2 wire all three, so the model-side capability existed
and was unreachable from configuration -- every DeepSeek-V4 run took the unfused
branch no matter what was asked for.

``tests/unit/model/test_cross_entropy_fusion_unit.py`` already covers the generic
resolution through data-parallel wrappers; this file covers only what is specific
to DeepSeek-V4, namely that the switch exists and that its forward step consults
it.
"""

from __future__ import annotations

import dataclasses

import pytest

from megatron.lite.model.deepseek_v4.lite import protocol as ds4_protocol
from megatron.lite.model.deepseek_v4.lite.protocol import ImplConfig

pytestmark = [pytest.mark.mlite]


def test_impl_config_exposes_cross_entropy_fusion() -> None:
    """The switch exists and is reachable; see the config for why it stays off.

    Megatron Core runs DeepSeek-V4 with ``--cross-entropy-loss-fusion``, so the
    default here is deliberately *not* matched to it: measured on this backend
    the fused path is both slower and larger, which is the opposite of its
    purpose. The default moves when ``linear_cross_entropy`` is fixed.
    """
    names = {f.name for f in dataclasses.fields(ImplConfig)}
    assert "cross_entropy_fusion" in names
    assert ImplConfig().cross_entropy_fusion is False


@pytest.mark.parametrize("enabled", [True, False])
def test_forward_step_passes_use_fused_kernels(monkeypatch, enabled: bool) -> None:
    """``_forward_step`` must consult the flag and forward it to the model.

    This is the read side of the gap: without it the model keeps its
    ``use_fused_kernels: bool = False`` default forever, and enabling
    ``cross_entropy_fusion`` in ``ImplConfig`` would change nothing at all while
    still reporting as enabled.
    """

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
