# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Local state plumbing through real Mamba schedules with a small deterministic mixer."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.ssm.wide_residual_mamba_layer import WideResidualMambaLayer
from megatron.core.transformer.residual_connection import ResidualBranchInput
from megatron.core.transformer.wide_residual import (
    StreamwiseSigmoidWideResidualConnection,
    load_residual_state,
)
from megatron.core.transformer.wide_residual_config import WideResidualConfig


class _Mixer(nn.Module):
    def __init__(self, dtype) -> None:
        super().__init__()
        self.projection = nn.Linear(8, 8, dtype=dtype)

    def forward_pre_attn_and_core_attn(self, value, **kwargs):
        return value * 0.5

    def forward_post_core_attn(self, value):
        return self.projection(value), None

    def forward(self, value, **kwargs):
        return self.forward_post_core_attn(self.forward_pre_attn_and_core_attn(value))


class _OperatorTrace(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.calls = []

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        self.calls.append(str(function))
        return function(*args, **(kwargs or {}))


def _layer(wide, fp32, dropout=0.0):
    # Isolate the real layer schedule from GPU-only mixer construction. The registered
    # norm, mixer and WR reader/writer remain real modules with real autograd execution.
    cls = WideResidualMambaLayer if wide else MambaLayer
    layer = cls.__new__(cls)
    nn.Module.__init__(layer)
    layer.config = SimpleNamespace(
        hidden_size=8,
        params_dtype=torch.bfloat16,
        fp32_residual_connection=fp32,
        bias_dropout_fusion=False,
        sequence_parallel=False,
        wide_residual=WideResidualConfig(num_streams=3, learned_retention=True) if wide else None,
    )
    layer.layer_number = 1
    layer.hidden_dropout = dropout
    layer.norm = nn.LayerNorm(8, dtype=torch.bfloat16)
    layer.mixer = _Mixer(torch.bfloat16)
    layer.mamba_bda = get_bias_dropout_add
    layer.bias_dropout_add_exec_handler = torch.enable_grad
    if wide:
        layer.residual_read = StreamwiseSigmoidWideResidualConnection(
            layer.config, 1, "mamba", operation="read"
        )
        layer.residual_write = StreamwiseSigmoidWideResidualConnection(
            layer.config, 1, "mamba", operation="write"
        )
        layer.register_load_state_dict_pre_hook(load_residual_state)
    return layer


def _parent_forward(layer, value, wide):
    """The eager schedule before opaque write-state plumbing (main d56953fb08)."""
    if wide:
        branch, state = layer.residual_read(
            value,
            fp32_residual_connection=layer.config.fp32_residual_connection,
            branch_input_dtype=layer.config.params_dtype,
        )
        layer.residual_read.residual_stream(state)
        branch = layer.norm(branch.to(layer.config.params_dtype))
    else:
        residual = layer._prepare_residual(value)
        branch = layer._prepare_mixer_input(value)
    output = layer.mixer(branch)
    if wide:
        with layer.bias_dropout_add_exec_handler():
            return layer.residual_write(
                output,
                state=state,
                dropout_probability=layer.hidden_dropout,
                training=layer.training,
            )
    return layer._apply_mixer_bda(output, residual)


@pytest.mark.parametrize("wide", [False, True], ids=["ordinary", "wr"])
@pytest.mark.parametrize("fp32", [False, True], ids=["bf16", "fp32"])
@pytest.mark.parametrize("mode", ["train", "eval", "no_grad", "inference"])
@pytest.mark.parametrize("backward_hooks", [False, True])
def test_parent_eager_parity(wide, fp32, mode, backward_hooks):
    torch.manual_seed(71)
    layer = _layer(wide, fp32, dropout=0.25)
    reference = copy.deepcopy(layer)
    layer.train(mode == "train")
    reference.train(mode == "train")
    dtype = torch.float32 if fp32 else torch.bfloat16
    value = torch.randn(4, 2, 24 if wide else 8, dtype=dtype, requires_grad=True)
    reference_value = value.detach().clone().requires_grad_(True)
    events = []
    reference_events = []
    for model, log in ((layer, events), (reference, reference_events)):
        for name in ("residual_read", "norm", "mixer", "residual_write"):
            if hasattr(model, name):
                getattr(model, name).register_forward_hook(
                    lambda module, args, output, name=name, log=log: log.append(name)
                )
                if backward_hooks:
                    getattr(model, name).register_full_backward_hook(
                        lambda module, grad_input, grad_output, name=name, log=log: log.append(
                            name + "_backward"
                        )
                    )
    context = {
        "train": torch.enable_grad,
        "eval": torch.enable_grad,
        "no_grad": torch.no_grad,
        "inference": torch.inference_mode,
    }[mode]
    rng = torch.get_rng_state()
    expected_trace, actual_trace = _OperatorTrace(), _OperatorTrace()
    with context(), expected_trace:
        expected = _parent_forward(reference, reference_value, wide)
    expected_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    with context(), actual_trace:
        actual = layer.forward(value)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == expected.dtype
    assert actual.requires_grad == expected.requires_grad
    assert actual_trace.calls == expected_trace.calls
    assert events == reference_events
    assert torch.equal(torch.get_rng_state(), expected_rng)
    if mode in ("train", "eval"):
        expected.float().square().sum().backward()
        actual.float().square().sum().backward()
        assert events == reference_events
        torch.testing.assert_close(value.grad, reference_value.grad, rtol=0, atol=0)
        parameters = dict(layer.named_parameters())
        reference_parameters = dict(reference.named_parameters())
        assert parameters.keys() == reference_parameters.keys()
        for name, param in parameters.items():
            ref_param = reference_parameters[name]
            assert (param.grad is None) == (ref_param.grad is None)
            if param.grad is not None:
                torch.testing.assert_close(param.grad, ref_param.grad, rtol=0, atol=0, msg=name)


def test_prepared_result_is_immutable_and_does_not_impose_state_layout():
    branch = torch.randn(2, 8, requires_grad=True)
    for state in (None, branch, (branch,), SimpleNamespace(residual=None)):
        result = ResidualBranchInput(branch_input=branch, state=state)
        assert result.branch_input is branch
        assert result.state is state
        with pytest.raises(AttributeError):
            result.state = None


@pytest.mark.parametrize("two_stage", [False, True], ids=["normal", "two_stage"])
def test_shared_schedule_keeps_state_opaque_and_call_local(monkeypatch, two_stage):
    layer = _layer(False, False)
    contexts = []
    updates = []

    def prepare(value, residual_stream_recompute_context=None):
        # Neither a tensor nor a residual-first tuple: only the matching update knows it.
        state = SimpleNamespace(residual=value)
        contexts.append(state)
        return ResidualBranchInput(value, state), None

    def update(output, state, recompute_context=None):
        assert recompute_context is None
        assert any(state is context for context in contexts)
        updates.append(state)
        return output[0] + state.residual

    monkeypatch.setattr(layer, "_prepare_mixer_state", prepare)
    monkeypatch.setattr(layer, "_apply_mixer_update", update)
    first, second = (torch.randn(2, 1, 8, dtype=torch.bfloat16) for _ in range(2))
    if two_stage:
        first_stage = layer.forward_pre_attn_and_core_attn(
            first, residual_stream_recompute_context=object()
        )
        second_stage = layer.forward_pre_attn_and_core_attn(
            second, residual_stream_recompute_context=object()
        )
        # Finish in reverse order so one invocation cannot overwrite the other's state.
        layer.forward_post_core_attn(*second_stage)
        layer.forward_post_core_attn(*first_stage)
    else:
        layer.forward(first, residual_stream_recompute_context=object())
        layer.forward(second, residual_stream_recompute_context=object())
    expected_updates = contexts[::-1] if two_stage else contexts
    assert len(updates) == len(expected_updates) == 2
    assert all(actual is expected for actual, expected in zip(updates, expected_updates))
    assert contexts[0] is not contexts[1]
    assert contexts[0].residual is first
    assert contexts[1].residual is second
    assert not any(value is contexts[0] or value is contexts[1] for value in vars(layer).values())


class _AliasWithScaledGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.view_as(value)

    @staticmethod
    def backward(ctx, grad):
        return 7 * grad


def test_wide_two_stage_preserves_read_selected_alias_and_state():
    layer = _layer(True, True)
    seen = []

    def choose_alias(module, args, output):
        branch, state = output
        state = (_AliasWithScaledGradient.apply(state[0]), *state[1:])
        seen.append(state)
        return branch, state

    layer.residual_read.register_forward_hook(choose_alias)
    value = torch.randn(2, 1, 24, requires_grad=True)
    stage = layer.forward_pre_attn_and_core_attn(value)
    assert len(stage) == 3
    assert stage[1] is seen[0][0]
    assert stage[2] is seen[0]
    assert stage[1] is not value
    stage[1].sum().backward()
    torch.testing.assert_close(value.grad, torch.full_like(value, 7), rtol=0, atol=0)
    output = layer.forward_post_core_attn(*stage)
    assert output.shape == value.shape


@pytest.mark.parametrize("wide", [False, True], ids=["ordinary", "wr"])
def test_two_stage_outputs_and_gradients_match_normal_forward(wide):
    layer = _layer(wide, True)
    split_layer = copy.deepcopy(layer)
    value = torch.randn(2, 1, 24 if wide else 8, requires_grad=True)
    split_value = value.detach().clone().requires_grad_(True)
    expected = layer.forward(value)
    state = split_layer.forward_pre_attn_and_core_attn(split_value)
    assert len(state) == (3 if wide else 2)
    if not wide:
        assert state[1] is split_value
    actual = split_layer.forward_post_core_attn(*state)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.float().square().sum().backward()
    actual.float().square().sum().backward()
    torch.testing.assert_close(split_value.grad, value.grad, rtol=0, atol=0)
    for name, param in layer.named_parameters():
        other = dict(split_layer.named_parameters())[name]
        assert (param.grad is None) == (other.grad is None)
        if param.grad is not None:
            torch.testing.assert_close(param.grad, other.grad, rtol=0, atol=0, msg=name)


def test_wide_operations_use_current_registered_owners():
    layer = _layer(True, True)
    layer.residual_read = copy.deepcopy(layer.residual_read)
    layer.residual_write = copy.deepcopy(layer.residual_write)
    events = []
    for name in ("residual_read", "residual_write"):
        getattr(layer, name).register_forward_hook(
            lambda module, args, output, name=name: events.append(name)
        )
    layer.forward(torch.randn(2, 1, 24))
    assert events == ["residual_read", "residual_write"]


class _RecordingContext:
    """Record checkpoint selection, not actual discard/replay execution."""

    def __init__(self, is_block_end) -> None:
        self.is_block_end = is_block_end
        self.calls = []

    def checkpoint(self, function, *args):
        self.calls.append(getattr(function, "__name__", type(function).__name__))
        return function(*args)


@pytest.mark.parametrize("wide", [False, True], ids=["ordinary", "wr"])
@pytest.mark.parametrize("is_block_end", [False, True])
def test_replay_selection_stays_in_layer_binding(wide, is_block_end):
    layer = _layer(wide, True)
    context = _RecordingContext(is_block_end)
    value = torch.randn(2, 1, 24 if wide else 8, requires_grad=True)
    prepared, active_context = layer._prepare_mixer_state(value, context)
    assert active_context is (context if wide else None)
    layer._apply_mixer_update(layer.mixer(prepared.branch_input), prepared.state, active_context)
    assert len(context.calls) == (2 + (not is_block_end) if wide else 0)
    if wide:
        assert context.calls[0] == "run_read"
        if not is_block_end:
            assert context.calls[-1] == "run_write"


def test_wr_registration_and_legacy_load_are_unchanged():
    layer = _layer(True, True)
    before = [(name, id(param)) for name, param in layer.named_parameters()]
    expected = {name: value.detach().clone() for name, value in layer.state_dict().items()}
    prepared, _ = layer._prepare_mixer_state(torch.randn(2, 1, 24))
    layer._apply_mixer_update(layer.mixer(prepared.branch_input), prepared.state)
    assert [(name, id(param)) for name, param in layer.named_parameters()] == before
    assert not isinstance(prepared, nn.Module)
    assert not set(map(id, layer.residual_read.parameters())) & set(
        map(id, layer.residual_write.parameters())
    )
    assert layer.residual_read.read_map.logit.is_residual_map_parameter
    assert layer.residual_write.retention.retention_logit.is_wide_residual_retention_parameter
    legacy = {
        name.replace("residual_read.", "residual_connection.").replace(
            "residual_write.", "residual_connection."
        ): value
        for name, value in expected.items()
    }
    destination = _layer(True, True)
    destination.load_state_dict(legacy, strict=True)
    for name, value in destination.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
