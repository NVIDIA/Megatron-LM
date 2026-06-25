# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import torch
import torch.nn as nn

from megatron.training.raw_moment_logging import RawMomentLogger


class _LinearBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(2))

    def forward(self, x):
        return self.linear(x)


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList([_LinearBlock()])

    def forward(self, x):
        return self.decoder.layers[0](x)


class _EmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(4, 2)

    def forward(self, x):
        return self.embedding(x)


class _OutputLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_layer = nn.Linear(2, 3, bias=False)

    def forward(self, x):
        return self.output_layer(x)


def _values_dict(values):
    return {name: moments for name, moments in values}


def test_activation_raw_moments_accumulate_by_module_site():
    model = [_ToyModel()]
    logger = RawMomentLogger()
    logger.register_activation_hooks(model)

    model[0](torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    model[0](torch.tensor([[5.0, 6.0]]))

    logger.finalize_activation_raw_moments_by_layer()
    logger.remove_activation_hooks()

    values = _values_dict(logger.consume_activation_raw_moments_by_layer())
    moments = values["decoder.layers.0.linear/input0"]
    assert moments == {"count": 6.0, "sum_1": 21.0, "sum_2": 91.0, "sum_3": 441.0, "sum_4": 2275.0}
    assert values["decoder.layers.0.linear/output0"] == moments


def test_activation_raw_moments_skip_no_grad_forward():
    model = [_ToyModel()]
    logger = RawMomentLogger()
    logger.register_activation_hooks(model)

    with torch.no_grad():
        model[0](torch.tensor([[10.0, 20.0]]))
    model[0](torch.tensor([[1.0, 2.0]]))

    logger.finalize_activation_raw_moments_by_layer()
    logger.remove_activation_hooks()

    values = _values_dict(logger.consume_activation_raw_moments_by_layer())
    assert values["decoder.layers.0.linear/input0"]["count"] == 2.0
    assert values["decoder.layers.0.linear/input0"]["sum_1"] == 3.0


def test_activation_raw_moments_skip_integer_inputs():
    model = [_EmbeddingModel()]
    logger = RawMomentLogger()
    logger.register_activation_hooks(model)

    model[0](torch.tensor([0, 1, 2], dtype=torch.long))

    logger.finalize_activation_raw_moments_by_layer()
    logger.remove_activation_hooks()

    values = _values_dict(logger.consume_activation_raw_moments_by_layer())
    assert "embedding/input0" not in values
    assert "embedding/output0" in values


def test_raw_moments_skip_output_layer_logits_site():
    model = [_OutputLayerModel()]
    logger = RawMomentLogger()
    logger.register_activation_hooks(model)

    model[0](torch.tensor([[1.0, 2.0]], requires_grad=True))

    logger.finalize_activation_raw_moments_by_layer()
    logger.remove_activation_hooks()

    values = _values_dict(logger.consume_activation_raw_moments_by_layer())
    assert "output_layer/input0" in values
    assert "output_layer/output0" not in values


def test_dgrad_raw_moments_skip_output_layer_logits_site():
    model = [_OutputLayerModel()]
    logger = RawMomentLogger()
    logger.register_dgrad_hooks(model, loss_scale=None)

    x = torch.tensor([[1.0, 2.0]], requires_grad=True)
    model[0](x).sum().backward()

    logger.finalize_dgrad_raw_moments_by_layer()
    logger.remove_dgrad_hooks()

    values, loss_scale = logger.consume_dgrad_raw_moments_by_layer()
    values = _values_dict(values)
    assert loss_scale is None
    assert "output_layer/input0" in values
    assert "output_layer/output0" not in values


def test_dgrad_raw_moments_accumulate_by_module_site_with_loss_scale():
    model = [_ToyModel()]
    logger = RawMomentLogger()
    logger.register_dgrad_hooks(model, loss_scale=128.0)

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    model[0](x).sum().backward()

    logger.finalize_dgrad_raw_moments_by_layer()
    logger.remove_dgrad_hooks()

    values, loss_scale = logger.consume_dgrad_raw_moments_by_layer()
    values = _values_dict(values)
    assert loss_scale == 128.0
    assert values["decoder.layers.0.linear/output0"] == {
        "count": 4.0,
        "sum_1": 4.0,
        "sum_2": 4.0,
        "sum_3": 4.0,
        "sum_4": 4.0,
    }
    assert values["decoder.layers.0.linear/input0"] == {
        "count": 4.0,
        "sum_1": 4.0,
        "sum_2": 4.0,
        "sum_3": 4.0,
        "sum_4": 4.0,
    }
