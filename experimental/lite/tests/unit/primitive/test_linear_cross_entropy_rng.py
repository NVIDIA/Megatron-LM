# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Resolving VERL's fused cross entropy leaves the global RNG streams unchanged.

VERL's import chain draws from Python's RNG at import time (rich's style ids via
transformers). The first import happens inside a training step, after a resumed
run restored its RNG; it must not advance the restored streams.
"""

import importlib.abc
import importlib.machinery
import random
import sys
import types

import numpy as np
import torch


class _DrawingFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Serve verl.utils.kernel.linear_cross_entropy; its import draws from the RNGs."""

    names = ('verl', 'verl.utils', 'verl.utils.kernel', 'verl.utils.kernel.linear_cross_entropy')

    def find_spec(self, name, path=None, target=None):
        if name in self.names:
            return importlib.machinery.ModuleSpec(name, self, is_package=name != self.names[-1])
        return None

    def create_module(self, spec):
        return types.ModuleType(spec.name)

    def exec_module(self, module):
        random.getrandbits(24)
        np.random.randint(1 << 20)
        if module.__name__ == self.names[-1]:
            module.linear_cross_entropy = lambda *args: None


def test_resolving_the_fused_kernel_keeps_rng_streams(monkeypatch):
    from megatron.lite.primitive.ops import linear_cross_entropy as module

    for name in _DrawingFinder.names:
        # Absent first; setitem restores the absence (or the real module) afterwards.
        monkeypatch.setitem(sys.modules, name, None)
        del sys.modules[name]
    monkeypatch.setattr(sys, 'meta_path', [_DrawingFinder(), *sys.meta_path])
    monkeypatch.setattr(module, '_VERL_LCE', None)
    random.seed(915)
    np.random.seed(915)
    torch.manual_seed(915)
    before = random.getstate(), np.random.get_state()[1].copy(), torch.get_rng_state()

    fused = module._verl_linear_cross_entropy()

    assert fused is not None, 'the fake fused kernel resolves'
    assert random.getstate() == before[0]
    assert np.array_equal(np.random.get_state()[1], before[1])
    assert torch.equal(torch.get_rng_state(), before[2])
    # Resolved once: later steps neither import nor draw.
    assert module._verl_linear_cross_entropy() is fused


def test_unavailable_kernel_resolves_to_the_fallback(monkeypatch):
    from megatron.lite.primitive.ops import linear_cross_entropy as module

    for name in _DrawingFinder.names:
        monkeypatch.setitem(sys.modules, name, None)  # import raises ImportError
    monkeypatch.setattr(module, '_VERL_LCE', None)
    random.seed(3)
    before = random.getstate()
    assert module._verl_linear_cross_entropy() is None
    assert random.getstate() == before
    hidden = torch.randn(4, 8, dtype=torch.float64)
    weight = torch.randn(16, 8, dtype=torch.float64)
    labels = torch.randint(16, (4,))
    log_probs, entropy = module.linear_cross_entropy(hidden, weight, labels)
    reference = torch.log_softmax(hidden @ weight.t(), -1).gather(1, labels[:, None])[:, 0]
    assert torch.allclose(log_probs.double(), reference, atol=1e-5)
    assert entropy.shape == labels.shape
