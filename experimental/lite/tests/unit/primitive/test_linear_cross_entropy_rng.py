# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Lazy fused-kernel import must preserve every global RNG stream across resume."""

import importlib.abc
import importlib.machinery
import random
import sys
import types

import numpy as np
import torch


class _DrawingFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):

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
    assert module._verl_linear_cross_entropy() is fused
