# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from megatron.core.transformer.layer_boundary_observer import (
    observe_transformer_layer_boundaries,
    observe_transformer_layer_input,
    observe_transformer_layer_output,
)
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.training.arguments import _validate_raw_moment_logging_args
from megatron.training.raw_moment_logging import RawMomentLogger

_STATS_INTERVAL_FLAGS = (
    'log_param_stats_interval',
    'log_wgrad_stats_interval',
    'log_activation_stats_interval',
    'log_dgrad_stats_interval',
    'log_residual_stats_interval',
    'log_residual_grad_stats_interval',
)


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


class _ResidualLayer(nn.Module):
    def __init__(self, layer_number):
        super().__init__()
        self.layer_number = layer_number
        self.config = SimpleNamespace(sequence_parallel=False)


class _ResidualModel(nn.Module):
    def __init__(self):
        super().__init__()
        decoder = TransformerBlock.__new__(TransformerBlock)
        nn.Module.__init__(decoder)
        decoder.layers = nn.ModuleList([_ResidualLayer(1), _ResidualLayer(2)])
        self.decoder = decoder


def _values_dict(values):
    return {name: moments for name, moments in values}


def _raw_moment_logging_args():
    return SimpleNamespace(
        use_megatron_fsdp=False,
        use_torch_fsdp2=False,
        log_param_stats_interval=0,
        log_wgrad_stats_interval=0,
        log_activation_stats_interval=0,
        log_dgrad_stats_interval=0,
        log_residual_stats_interval=0,
        log_residual_grad_stats_interval=0,
        overlap_moe_expert_parallel_comm=False,
    )


@pytest.mark.parametrize('logging_flag', _STATS_INTERVAL_FLAGS)
@pytest.mark.parametrize('fsdp_flag', ('use_megatron_fsdp', 'use_torch_fsdp2'))
def test_raw_moment_logging_rejects_fsdp(logging_flag, fsdp_flag):
    args = _raw_moment_logging_args()
    setattr(args, logging_flag, 1)
    setattr(args, fsdp_flag, True)

    with pytest.raises(ValueError, match='Raw-moment statistics logging is not supported'):
        _validate_raw_moment_logging_args(args)


@pytest.mark.parametrize('logging_flag', _STATS_INTERVAL_FLAGS)
@pytest.mark.parametrize('disabled_interval', (0, -1))
@pytest.mark.parametrize('fsdp_flag', ('use_megatron_fsdp', 'use_torch_fsdp2'))
def test_disabled_raw_moment_logging_allows_fsdp(logging_flag, disabled_interval, fsdp_flag):
    args = _raw_moment_logging_args()
    setattr(args, logging_flag, disabled_interval)
    setattr(args, fsdp_flag, True)

    _validate_raw_moment_logging_args(args)


@pytest.mark.parametrize(
    "logging_flag",
    ("log_residual_stats_interval", "log_residual_grad_stats_interval"),
)
def test_residual_raw_moment_logging_rejects_fine_grained_ep_overlap(logging_flag):
    args = _raw_moment_logging_args()
    setattr(args, logging_flag, 1)
    args.overlap_moe_expert_parallel_comm = True

    with pytest.raises(ValueError, match='--overlap-moe-expert-parallel-comm'):
        _validate_raw_moment_logging_args(args)


def test_residual_raw_moments_capture_distinct_layer_boundaries():
    model = _ResidualModel()
    logger = RawMomentLogger()
    logger.prepare_residual_logging(model)
    first_layer, second_layer = model.decoder.layers

    with observe_transformer_layer_boundaries(logger.record_residual_boundary):
        observe_transformer_layer_input(model.decoder, first_layer, torch.tensor([1.0, 2.0]))
        observe_transformer_layer_output(model.decoder, first_layer, torch.tensor([3.0, 4.0]))
        observe_transformer_layer_input(model.decoder, second_layer, torch.tensor([30.0, 40.0]))
        observe_transformer_layer_output(model.decoder, second_layer, torch.tensor([5.0, 6.0]))
        observe_transformer_layer_output(model.decoder, first_layer, torch.tensor([7.0, 8.0]))

    # The observer is scoped and does not retain events after the context exits.
    observe_transformer_layer_output(model.decoder, second_layer, torch.tensor([50.0, 60.0]))
    logger.finalize_residual_raw_moments_by_layer()

    values = _values_dict(logger.consume_residual_raw_moments_by_layer())
    assert set(values) == {
        "decoder/input0",
        "decoder.layers.0/output0",
        "decoder.layers.1/output0",
    }
    assert values["decoder/input0"] == {
        "count": 2.0,
        "sum_1": 3.0,
        "sum_2": 5.0,
        "sum_3": 9.0,
        "sum_4": 17.0,
    }
    assert values["decoder.layers.0/output0"] == {
        "count": 4.0,
        "sum_1": 22.0,
        "sum_2": 138.0,
        "sum_3": 946.0,
        "sum_4": 6834.0,
    }
    assert values["decoder.layers.1/output0"]["sum_1"] == 11.0


def test_residual_raw_moments_skip_no_grad_forward():
    model = _ResidualModel()
    logger = RawMomentLogger()
    logger.prepare_residual_logging(model)
    first_layer = model.decoder.layers[0]

    with observe_transformer_layer_boundaries(logger.record_residual_boundary):
        with torch.no_grad():
            observe_transformer_layer_output(
                model.decoder, first_layer, torch.tensor([10.0, 20.0])
            )
        observe_transformer_layer_output(model.decoder, first_layer, torch.tensor([1.0, 2.0]))

    logger.finalize_residual_raw_moments_by_layer()
    values = _values_dict(logger.consume_residual_raw_moments_by_layer())
    assert values["decoder.layers.0/output0"]["count"] == 2.0
    assert values["decoder.layers.0/output0"]["sum_1"] == 3.0


def test_residual_dgrad_raw_moments_capture_boundary_gradients():
    model = _ResidualModel()
    logger = RawMomentLogger()
    logger.prepare_residual_logging(model, capture_residuals=False, capture_dgrads=True)
    first_layer, second_layer = model.decoder.layers

    residual_input = torch.tensor([1.0, 2.0], requires_grad=True)
    first_output = residual_input * torch.tensor([2.0, 3.0])
    second_output = first_output * torch.tensor([5.0, 7.0])
    with observe_transformer_layer_boundaries(logger.record_residual_boundary):
        observe_transformer_layer_input(model.decoder, first_layer, residual_input)
        observe_transformer_layer_output(model.decoder, first_layer, first_output)
        observe_transformer_layer_output(model.decoder, second_layer, second_output)
        second_output.sum().backward()

    logger.finalize_residual_dgrad_raw_moments_by_layer()
    values = _values_dict(logger.consume_residual_dgrad_raw_moments_by_layer())

    assert values["decoder/input0"] == {
        "count": 2.0,
        "sum_1": 31.0,
        "sum_2": 541.0,
        "sum_3": 10261.0,
        "sum_4": 204481.0,
    }
    assert values["decoder.layers.0/output0"]["sum_1"] == 12.0
    assert values["decoder.layers.1/output0"]["sum_1"] == 2.0
    assert residual_input.grad.tolist() == [10.0, 21.0]
    assert not logger._residual_dgrad_hooks


def test_residual_dgrad_raw_moments_capture_checkpoint_recomputation_once():
    model = _ResidualModel()
    logger = RawMomentLogger()
    logger.prepare_residual_logging(model, capture_residuals=False, capture_dgrads=True)
    first_layer = model.decoder.layers[0]

    def checkpointed_layer(hidden_states):
        observe_transformer_layer_input(model.decoder, first_layer, hidden_states)
        output = hidden_states * 2
        observe_transformer_layer_output(model.decoder, first_layer, output)
        return output

    hidden_states = torch.tensor([1.0, 2.0], requires_grad=True)
    with observe_transformer_layer_boundaries(logger.record_residual_boundary):
        checkpoint(checkpointed_layer, hidden_states, use_reentrant=True).sum().backward()

    logger.finalize_residual_dgrad_raw_moments_by_layer()
    values = _values_dict(logger.consume_residual_dgrad_raw_moments_by_layer())

    assert values["decoder/input0"]["count"] == 2.0
    assert values["decoder/input0"]["sum_1"] == 4.0
    assert values["decoder.layers.0/output0"]["count"] == 2.0
    assert values["decoder.layers.0/output0"]["sum_1"] == 2.0


def test_residual_dgrad_raw_moments_support_reused_autograd_output_tensor():
    class ReuseOutput(torch.autograd.Function):
        output = torch.empty(2)

        @staticmethod
        def forward(ctx, tensor):
            ReuseOutput.output.copy_(tensor)
            return ReuseOutput.output

        @staticmethod
        def backward(ctx, grad):
            return grad

    model = _ResidualModel()
    logger = RawMomentLogger()
    first_layer = model.decoder.layers[0]
    output_tensor = None

    for input_values in ([1.0, 2.0], [3.0, 4.0]):
        logger.prepare_residual_logging(model, capture_residuals=False, capture_dgrads=True)
        tensor = ReuseOutput.apply(torch.tensor(input_values, requires_grad=True))
        if output_tensor is None:
            output_tensor = tensor
        else:
            assert tensor is output_tensor

        with observe_transformer_layer_boundaries(logger.record_residual_boundary):
            observe_transformer_layer_output(model.decoder, first_layer, tensor)
            tensor.sum().backward()

        logger.finalize_residual_dgrad_raw_moments_by_layer()
        values = _values_dict(logger.consume_residual_dgrad_raw_moments_by_layer())
        assert values["decoder.layers.0/output0"]["count"] == 2.0
        assert values["decoder.layers.0/output0"]["sum_1"] == 2.0


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
    logger.register_dgrad_hooks(model)

    x = torch.tensor([[1.0, 2.0]], requires_grad=True)
    model[0](x).sum().backward()

    logger.finalize_dgrad_raw_moments_by_layer()
    logger.remove_dgrad_hooks()

    values = _values_dict(logger.consume_dgrad_raw_moments_by_layer())
    assert "output_layer/input0" in values
    assert "output_layer/output0" not in values


def test_dgrad_raw_moments_accumulate_by_module_site():
    model = [_ToyModel()]
    logger = RawMomentLogger()
    logger.register_dgrad_hooks(model)

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    model[0](x).sum().backward()

    logger.finalize_dgrad_raw_moments_by_layer()
    logger.remove_dgrad_hooks()

    values = _values_dict(logger.consume_dgrad_raw_moments_by_layer())
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
