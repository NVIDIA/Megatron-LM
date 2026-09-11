# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""cross_entropy_fusion must survive the data-parallel wrapper.

``set_cross_entropy_fusion`` runs at build time on the bare model chunks. By the
time ``_forward_step`` runs, the runtime has wrapped each chunk in a
data-parallel container whose only reference to the wrapped model is
``self.module`` and which does not proxy unknown attributes to it. A plain
``getattr(wrapper, "cross_entropy_fusion")`` therefore misses, and the model
silently takes the unfused branch.

That failure is invisible from configuration: the key resolves to ``True``, it
appears as ``True`` in the config dump, and the only symptom is memory: the
unfused path materialises the full ``[tokens, vocab/tp]`` logits in fp32.

**These tests assert which code path runs, not what the config says.** Asserting
``use_fused_kernels is True`` would have passed even before the fix, because the
configuration was never the broken part.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from megatron.lite.model.protocol_utils import (
    add_cross_entropy_fusion,
    set_cross_entropy_fusion,
)

pytestmark = pytest.mark.mlite


class _CountingCE:
    """Stand-in for vocab_parallel_cross_entropy that records whether it ran."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, logits, labels):
        self.calls += 1
        return logits.float().mean()


class _Model(nn.Module):
    """Minimal mirror of qwen3_moe's loss branch.

    The unfused branch materialises logits and calls the (counted) cross-entropy;
    the fused branch does neither. Which one runs is the observable under test.
    """

    def __init__(self, ce: _CountingCE):
        super().__init__()
        self.ce = ce
        self.head = nn.Linear(4, 8, bias=False)
        self.fused_calls = 0

    def forward(self, x, use_fused_kernels: bool = False):
        if use_fused_kernels:
            self.fused_calls += 1
            return {"loss": x.sum()}
        logits = self.head(x)
        return {"loss": self.ce(logits, None)}


class _BaseDataParallelLike(nn.Module):
    """Mirrors megatron.core.distributed.data_parallel_base._BaseDataParallel.

    Holds the wrapped model in ``self.module`` but does not proxy unknown
    attributes to it.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def _forward_step(model, x):
    """Mirrors qwen3_moe/lite/protocol.py::_forward_step."""
    kwargs: dict = {}
    add_cross_entropy_fusion(kwargs, model)
    return model(x, **kwargs)


def test_wrapper_does_not_forward_attributes():
    """The premise of the bug, pinned so it cannot silently change."""
    inner = _Model(_CountingCE())
    inner.cross_entropy_fusion = True
    wrapped = _BaseDataParallelLike(inner)
    assert "cross_entropy_fusion" not in vars(wrapped)
    assert not hasattr(wrapped, "cross_entropy_fusion")


def test_fusion_enabled_through_wrapper_skips_cross_entropy():
    """THE regression test: behaviour, not configuration."""
    ce = _CountingCE()
    inner = _Model(ce)
    set_cross_entropy_fusion([inner], True)
    wrapped = _BaseDataParallelLike(inner)

    _forward_step(wrapped, torch.randn(3, 4))

    assert ce.calls == 0, (
        "cross-entropy ran despite cross_entropy_fusion=True -- the flag did not "
        "survive the data-parallel wrapper"
    )
    assert inner.fused_calls == 1, "the fused branch did not run"


def test_fusion_disabled_through_wrapper_still_uses_cross_entropy():
    """Negative control: with the flag off, the unfused branch must run."""
    ce = _CountingCE()
    inner = _Model(ce)
    set_cross_entropy_fusion([inner], False)
    wrapped = _BaseDataParallelLike(inner)

    _forward_step(wrapped, torch.randn(3, 4))

    assert ce.calls == 1
    assert inner.fused_calls == 0


def test_fusion_works_unwrapped_too():
    """The pre-existing (unwrapped) path must keep working."""
    ce = _CountingCE()
    inner = _Model(ce)
    set_cross_entropy_fusion([inner], True)
    _forward_step(inner, torch.randn(3, 4))
    assert ce.calls == 0 and inner.fused_calls == 1


def test_fusion_survives_nested_wrappers():
    """More than one wrapper is legal; the walk must not stop at the first."""
    ce = _CountingCE()
    inner = _Model(ce)
    set_cross_entropy_fusion([inner], True)
    doubly = _BaseDataParallelLike(_BaseDataParallelLike(inner))
    _forward_step(doubly, torch.randn(3, 4))
    assert ce.calls == 0 and inner.fused_calls == 1


def test_unset_defaults_to_unfused():
    """No attribute anywhere -> False, not a crash."""
    ce = _CountingCE()
    wrapped = _BaseDataParallelLike(_Model(ce))
    _forward_step(wrapped, torch.randn(3, 4))
    assert ce.calls == 1


def test_resolution_terminates_on_self_referential_wrapper():
    """A cycle must not hang the forward path."""
    from megatron.lite.model.protocol_utils import _resolve_cross_entropy_fusion

    class _Cyclic:
        pass

    a = _Cyclic()
    a.module = a
    assert _resolve_cross_entropy_fusion(a) is False


def test_old_implementation_would_fail_this_suite():
    """Proof the suite discriminates: the pre-fix lookup returns False here.

    If this ever starts returning True, the tests above have stopped testing the
    thing that was broken.
    """
    inner = _Model(_CountingCE())
    set_cross_entropy_fusion([inner], True)
    wrapped = _BaseDataParallelLike(inner)
    old_result = bool(getattr(wrapped, "cross_entropy_fusion", False))
    assert old_result is False, "the old lookup no longer reproduces the bug"
