# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Shared Transformer scheduling with small, architecture-independent connections.

These tests use the real scheduling and checkpoint methods. Minimal construction
and an inert offload context exclude attention, MLP, and offload-engine integration.
"""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from megatron.core.tensor_parallel.random import (
    CheckpointWithoutOutputManager,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.residual_connection import ResidualConnection
from megatron.core.transformer.residual_recompute import ResidualStreamRecomputeContext
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.test_utilities import Utils


class _OffloadContext:
    def __init__(self, value) -> None:
        self.value = value

    def __enter__(self):
        return self.value

    def __exit__(self, *args):
        pass

    def group_offload(self, value, **kwargs):
        return value


class _Connection(ResidualConnection):
    """Replay numerical operations without re-entering the module's hooks."""

    def __init__(self, operation) -> None:
        super().__init__(8, 4, operation=operation)
        self.scale = nn.Parameter(torch.tensor(0.75))

    def _read_math(self, value, dtype):
        return (value[..., :4].sin() * self.scale).to(dtype=dtype)

    def _read_state(self, value):
        self.selected_state = (value * 2,)
        return self.selected_state

    def _eager_read(self, value, **kwargs):
        return self._read_math(value, kwargs["branch_input_dtype"]), self._read_state(value)

    def _checkpoint_read(self, value, context, **kwargs):
        branch = context.checkpoint(self._read_math, value, kwargs["branch_input_dtype"])
        return branch, self._read_state(value)

    def _write_math(self, value, bias, residual):
        update = value if bias is None else value + bias
        return residual + torch.cat((update, update), dim=-1) * self.scale

    def _eager_write(self, branch_output, state, **kwargs):
        value, bias = branch_output if isinstance(branch_output, tuple) else (branch_output, None)
        return self._write_math(value, bias, *state)

    def _checkpoint_write(self, branch_output, state, context, **kwargs):
        value, bias = branch_output if isinstance(branch_output, tuple) else (branch_output, None)
        return context.checkpoint(self._write_math, value, bias, *state)


class _TupleNorm(nn.Module):
    returns_residual = True

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(4))
        self.selected = []

    def forward(self, value):
        residual = value.view_as(value)
        self.selected.append(residual)
        return value.square() * self.weight, residual


class _DispatchLayer(TransformerLayer):
    def __init__(self, connections=True) -> None:
        # Exercise inherited scheduling without constructing GPU attention/MLP modules.
        nn.Module.__init__(self)
        self.config = SimpleNamespace(
            params_dtype=torch.float32,
            fp32_residual_connection=False,
            fine_grained_activation_offloading=False,
            inference_fuse_tp_communication=False,
        )
        self.hidden_dropout = 0.0
        self.input_layernorm = nn.LayerNorm(4)
        self.pre_mlp_layernorm = nn.LayerNorm(4)
        self._input_layernorm_returns_residual = False
        self._pre_mlp_layernorm_returns_residual = False
        self.recompute_input_layernorm = self.recompute_pre_mlp_layernorm = False
        self.mhc_checkpoint_input_layernorm = self.mhc_checkpoint_pre_mlp_layernorm = False
        self.offload_attn_norm = self.offload_mlp_norm = False
        self.off_interface = lambda enabled, value, name: _OffloadContext(value)
        self.bias_dropout_add_exec_handler = torch.enable_grad
        self._legacy_forward_post_mlp = None
        self.events = []
        for stage in ("self_attn", "mlp"):
            for operation in ("read", "write"):
                setattr(
                    self,
                    f"residual_{operation}_{stage}",
                    _Connection(operation) if connections else None,
                )

    def _read_residual_connection(self, connection, value, *, is_mlp, recompute_context):
        self.events.append(("read", connection, is_mlp, recompute_context))
        return super()._read_residual_connection(
            connection, value, is_mlp=is_mlp, recompute_context=recompute_context
        )

    def _write_residual_connection(self, connection, value, state, *, is_mlp, recompute_context):
        self.events.append(("write", connection, is_mlp, recompute_context))
        return super()._write_residual_connection(
            connection, value, state, is_mlp=is_mlp, recompute_context=recompute_context
        )


def _layer(connections=True):
    return _DispatchLayer(connections).to("cuda" if torch.cuda.is_available() else "cpu")


def _read(layer, value, is_mlp, context=None):
    method = layer._pre_mlp_layernorm_and_residual if is_mlp else layer._run_input_layernorm
    return method(value, residual_stream_recompute_context=context)


def _write(layer, value, residual, state, is_mlp, context=None):
    method = layer._apply_mlp_bda_step if is_mlp else layer._apply_self_attn_bda_step
    return method(value, residual, state, residual_stream_recompute_context=context)


def _checkpoint(layer, is_mlp):
    return layer.pre_mlp_norm_checkpoint if is_mlp else layer.input_layernorm_checkpoint


def _assert_gradients(actual, reference):
    for name, parameter in actual.named_parameters():
        expected = dict(reference.named_parameters())[name]
        assert (parameter.grad is None) == (expected.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, expected.grad, msg=name)


@pytest.fixture
def gpu_model_parallel():
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    yield
    Utils.destroy_model_parallel()


@pytest.mark.parametrize("is_mlp", [False, True], ids=["attention", "mlp"])
@pytest.mark.parametrize("operation", ["read", "write"])
def test_getters_resolve_registered_connections(is_mlp, operation):
    layer = _layer()
    stage = "mlp" if is_mlp else "self_attn"
    getter = (
        layer._get_residual_connection_for_mlp
        if is_mlp
        else layer._get_residual_connection_for_self_attention
    )
    name = f"residual_{operation}_{stage}"
    assert getter(operation) is getattr(layer, name)
    parameter_names = {
        f"residual_{op}_{branch}.scale"
        for op in ("read", "write")
        for branch in ("self_attn", "mlp")
    }
    assert {name for name, _ in layer.named_parameters() if name.startswith("residual_")} == (
        parameter_names
    )
    assert {name for name in layer.state_dict() if name.startswith("residual_")} == parameter_names
    replacement = _Connection(operation).to(next(layer.parameters()).device)
    setattr(layer, name, replacement)
    assert getter(operation) is replacement
    setattr(layer, name, None)
    assert getter(operation) is None
    delattr(layer, name)
    assert getter(operation) is None


@pytest.mark.parametrize("is_mlp", [False, True], ids=["attention", "mlp"])
def test_getters_reject_invalid_operation(is_mlp):
    layer = _layer(connections=False)
    getter = (
        layer._get_residual_connection_for_mlp
        if is_mlp
        else layer._get_residual_connection_for_self_attention
    )
    with pytest.raises(ValueError, match="Unsupported residual connection operation: invalid"):
        getter("invalid")


@pytest.mark.parametrize("is_mlp", [False, True], ids=["attention", "mlp"])
def test_helper_forwards_call_settings(is_mlp):
    layer = _layer().eval()
    layer.config.fp32_residual_connection = True
    layer.config.params_dtype = torch.bfloat16
    layer.hidden_dropout = 0.25
    stage = "mlp" if is_mlp else "self_attn"
    reader, writer = getattr(layer, f"residual_read_{stage}"), getattr(
        layer, f"residual_write_{stage}"
    )
    calls = []
    for connection in (reader, writer):
        connection.register_forward_pre_hook(
            lambda module, args, kwargs: calls.append((module, args, kwargs.copy())),
            with_kwargs=True,
        )
    value = torch.randn(2, 1, 8, device=next(layer.parameters()).device)
    branch, state = layer._read_residual_connection(
        reader, value, is_mlp=is_mlp, recompute_context=None
    )
    branch_output = (branch, None)
    layer._write_residual_connection(
        writer, branch_output, state, is_mlp=is_mlp, recompute_context=None
    )
    assert branch.dtype == torch.bfloat16 and len(calls) == 2
    assert calls[0][0] is reader and calls[0][1][0] is value
    assert calls[0][2] == {
        "fp32_residual_connection": True,
        "branch_input_dtype": torch.bfloat16,
        "recompute_context": None,
    }
    assert calls[1][0] is writer and calls[1][1][0] is branch_output
    assert calls[1][2]["state"] is state
    assert calls[1][2]["dropout_probability"] == 0.25
    assert calls[1][2]["training"] is False and calls[1][2]["recompute_context"] is None
    assert layer.events == [("read", reader, is_mlp, None), ("write", writer, is_mlp, None)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Output-discard replay needs CUDA")
@pytest.mark.parametrize("is_mlp", [False, True], ids=["attention", "mlp"])
@pytest.mark.parametrize("mode", ["eager", "nonterminal", "terminal"])
def test_stage_dispatch_preserves_reader_state_and_module_hooks(is_mlp, mode, gpu_model_parallel):
    actual = _layer()
    # A norm replaced after construction can leave stale metadata on the layer.
    actual._input_layernorm_returns_residual = actual._pre_mlp_layernorm_returns_residual = True
    reference = copy.deepcopy(actual)
    context = (
        None
        if mode == "eager"
        else ResidualStreamRecomputeContext(
            CheckpointWithoutOutputManager(), is_block_end=mode == "terminal"
        )
    )
    stage = "mlp" if is_mlp else "self_attn"
    reader, writer = getattr(actual, f"residual_read_{stage}"), getattr(
        actual, f"residual_write_{stage}"
    )
    hooks = []
    for connection in (reader, writer):
        connection.register_forward_pre_hook(lambda module, args: hooks.append(("pre", module)))
        connection.register_forward_hook(
            lambda module, args, output: hooks.append(("post", module))
        )
    value = torch.randn(2, 1, 8, device=next(actual.parameters()).device, requires_grad=True)
    reference_value = value.detach().clone().requires_grad_()
    branch, residual, state = _read(actual, value, is_mlp, context)
    expected_branch, expected_residual, expected_state = _read(reference, reference_value, is_mlp)
    assert state is reader.selected_state and residual is state[0]
    torch.testing.assert_close(residual, value * 2)
    output = _write(actual, (branch.sin(), None), residual, state, is_mlp, context)
    expected = _write(
        reference, (expected_branch.sin(), None), expected_residual, expected_state, is_mlp
    )
    torch.testing.assert_close(output, expected)
    write_context = None if is_mlp and mode == "terminal" else context
    assert actual.events == [
        ("read", reader, is_mlp, context),
        ("write", writer, is_mlp, write_context),
    ]
    boundary = output.square().sum()
    managed_outputs = []
    if context is not None:
        assert not _checkpoint(actual, is_mlp).retain_input_tensors
        assert len(context.manager.checkpoints) == (2 if write_context is None else 3)
        managed_outputs = [
            tensor for checkpoint in context.manager.checkpoints for tensor in checkpoint.outputs
        ]
        context.manager.discard_all_outputs_and_register_unified_recompute(boundary)
        assert all(tensor.untyped_storage().nbytes() == 0 for tensor in managed_outputs)
        assert bool(output.untyped_storage().nbytes()) == (write_context is None)
        assert value.untyped_storage().nbytes() > 0 and residual.untyped_storage().nbytes() > 0
    boundary.backward()
    expected.square().sum().backward()
    assert all(tensor.untyped_storage().nbytes() > 0 for tensor in managed_outputs)
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(value.grad, reference_value.grad)
    _assert_gradients(actual, reference)
    assert hooks == [("pre", reader), ("post", reader), ("pre", writer), ("post", writer)]
    assert len(actual.events) == 2


@pytest.mark.parametrize("is_mlp", [False, True], ids=["attention", "mlp"])
@pytest.mark.parametrize("advertised", [False, True])
@pytest.mark.parametrize("source", ["replacement", "hook"])
def test_reader_rejects_actual_tuple_norm_output(is_mlp, advertised, source):
    layer = _layer()
    name = "pre_mlp_layernorm" if is_mlp else "input_layernorm"
    setattr(layer, f"_{name}_returns_residual", advertised)
    if source == "replacement":
        setattr(layer, name, _TupleNorm().to(next(layer.parameters()).device))
    else:
        getattr(layer, name).register_forward_hook(lambda module, args, output: (output, args[0]))
    value = torch.randn(2, 1, 8, device=next(layer.parameters()).device)
    with pytest.raises(ValueError, match=f"{name} cannot return a residual"):
        _read(layer, value, is_mlp)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Output-discard replay needs CUDA")
@pytest.mark.parametrize("is_mlp", [False, True], ids=["attention", "mlp"])
@pytest.mark.parametrize("recompute", [False, True])
def test_ordinary_tuple_norm_preserves_selected_residual_and_gradients(
    is_mlp, recompute, gpu_model_parallel
):
    actual = _layer(connections=False)
    name = "pre_mlp_layernorm" if is_mlp else "input_layernorm"
    norm = _TupleNorm().to(next(actual.parameters()).device)
    setattr(actual, name, norm)
    setattr(actual, f"_{name}_returns_residual", True)
    reference = copy.deepcopy(actual)
    setattr(actual, f"recompute_{name}", recompute)
    value = torch.randn(2, 1, 4, device=norm.weight.device, requires_grad=True)
    reference_value = value.detach().clone().requires_grad_()
    branch, residual, state = _read(actual, value, is_mlp)
    expected_branch, expected_residual, expected_state = _read(reference, reference_value, is_mlp)
    assert state == expected_state == () and residual is norm.selected[0]
    assert residual is not value and residual.data_ptr() == value.data_ptr()
    torch.testing.assert_close(residual, value)
    boundary = (branch + residual).square().sum()
    expected = (expected_branch + expected_residual).square().sum()
    torch.testing.assert_close(boundary, expected)
    if recompute:
        checkpoint = _checkpoint(actual, is_mlp)
        assert checkpoint.retain_input_tensors
        input_bytes = value.untyped_storage().nbytes()
        checkpoint.discard_output_and_register_recompute(boundary)
        assert branch.untyped_storage().nbytes() == 0
        assert (
            value.untyped_storage().nbytes() == residual.untyped_storage().nbytes() == input_bytes
        )
    boundary.backward()
    expected.backward()
    assert branch.untyped_storage().nbytes() > 0
    torch.testing.assert_close(branch, expected_branch)
    torch.testing.assert_close(residual, expected_residual)
    assert len(norm.selected) == 1 + int(recompute)
    torch.testing.assert_close(value.grad, reference_value.grad)
    _assert_gradients(actual, reference)
    assert actual.events == []
