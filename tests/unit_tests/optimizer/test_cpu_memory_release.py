# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from megatron.core.optimizer import _release_unused_cpu_memory_after_optimizer_init
from megatron.core.optimizer.optimizer import ChainedOptimizer


def _optimizer(high_precision_init_value_count):
    return SimpleNamespace(_high_precision_init_value_count=high_precision_init_value_count)


def test_release_unused_cpu_memory_once_after_all_optimizers(monkeypatch):
    empty_cache = Mock()
    monkeypatch.setattr(torch.cpu, 'empty_cache', empty_cache, raising=False)
    optimizer = ChainedOptimizer([ChainedOptimizer([_optimizer(3)]), _optimizer(5)])

    released = _release_unused_cpu_memory_after_optimizer_init(optimizer)

    assert released
    empty_cache.assert_called_once_with()


def test_release_unused_cpu_memory_skips_when_no_values_were_observed(monkeypatch):
    empty_cache = Mock()
    monkeypatch.setattr(torch.cpu, 'empty_cache', empty_cache, raising=False)

    released = _release_unused_cpu_memory_after_optimizer_init(_optimizer(0))

    assert not released
    empty_cache.assert_not_called()


def test_release_unused_cpu_memory_supports_older_pytorch(monkeypatch, caplog):
    monkeypatch.delattr(torch.cpu, 'empty_cache', raising=False)

    released = _release_unused_cpu_memory_after_optimizer_init(_optimizer(1))

    assert not released
    assert "does not provide torch.cpu.empty_cache" in caplog.text
