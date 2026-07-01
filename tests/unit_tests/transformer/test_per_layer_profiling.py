# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.transformer.per_layer_profiling import (
    PerLayerProfiler,
    log_per_layer_resource_usage,
)


class _ToyLayer(torch.nn.Linear):
    def __init__(self, layer_number: int) -> None:
        super().__init__(4, 4)
        self.layer_number = layer_number


class _ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_ToyLayer(1), _ToyLayer(2)])
        self.per_layer_profiler = PerLayerProfiler(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def test_per_layer_profiler_accumulates_forward_and_backward_stats():
    model = _ToyModel()
    x = torch.ones(2, 4, requires_grad=True)

    loss = model(x).sum()
    loss.backward()

    summary = model.per_layer_profiler.summary()
    assert [stats.layer_number for stats in summary] == [1, 2]
    assert [stats.forward_calls for stats in summary] == [1, 1]
    assert [stats.backward_calls for stats in summary] == [1, 1]
    assert all(stats.forward_time_ms >= 0.0 for stats in summary)
    assert all(stats.backward_time_ms >= 0.0 for stats in summary)


def test_per_layer_profiler_summary_can_reset_counters():
    model = _ToyModel()
    model(torch.ones(2, 4)).sum().backward()

    summary = model.per_layer_profiler.summary(reset=True)
    assert all(stats.forward_calls == 1 for stats in summary)
    assert all(stats.backward_calls == 1 for stats in summary)

    reset_summary = model.per_layer_profiler.summary()
    assert all(stats.forward_calls == 0 for stats in reset_summary)
    assert all(stats.backward_calls == 0 for stats in reset_summary)


def test_log_per_layer_resource_usage_prints_and_resets(capsys):
    model = _ToyModel()
    model(torch.ones(2, 4)).sum().backward()

    log_per_layer_resource_usage(model, iteration=10)

    captured = capsys.readouterr()
    assert "per-layer resource usage after 10 iterations" in captured.out
    assert "layer    1" in captured.out
    assert "layer    2" in captured.out
    assert all(stats.forward_calls == 0 for stats in model.per_layer_profiler.summary())


def test_log_per_layer_resource_usage_accepts_missing_model():
    log_per_layer_resource_usage(None, iteration=10)
