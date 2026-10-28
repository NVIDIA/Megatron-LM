# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Shared Transformer caller contracts with small recording connections.

Scheduling uses real Transformer methods; ordinary tuple-norm tests also replay
real checkpoints. Attention, MLP, residual arithmetic, and offload are excluded.
"""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from megatron.core.tensor_parallel.random import (
    CheckpointWithoutOutputManager,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.residual_recompute import ResidualStreamRecomputeContext
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.test_utilities import Utils


class _OffloadContext(nullcontext):
    def group_offload(self, value, **kwargs):
        return value


class _Connection(nn.Module):
    """Supply reader-owned state without implementing residual arithmetic or replay."""

    def __init__(self, operation) -> None:
        super().__init__()
        self.operation = operation

    def forward(self, value, **kwargs):
        if self.operation == "read":
            self.selected_state = (value * 2,)
            return value[..., :4].to(kwargs["branch_input_dtype"]), self.selected_state
        return kwargs["state"][0]

    @staticmethod
    def residual_stream(state):
        return state[0]


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


@pytest.fixture
def gpu_model_parallel():
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    yield
    Utils.destroy_model_parallel()


@pytest.mark.parametrize("is_mlp", [False, True], ids=["attention", "mlp"])
def test_getters_handle_optional_connections_and_invalid_operations(is_mlp):
    layer = _layer(connections=False)
    stage = "mlp" if is_mlp else "self_attn"
    getter = (
        layer._get_residual_connection_for_mlp
        if is_mlp
        else layer._get_residual_connection_for_self_attention
    )
    for operation in ("read", "write"):
        assert getter(operation) is None
        delattr(layer, f"residual_{operation}_{stage}")
        assert getter(operation) is None
    with pytest.raises(ValueError, match="Unsupported residual connection operation: invalid"):
        getter("invalid")


@pytest.mark.parametrize(
    ("is_mlp", "mode"),
    [
        (False, "eager"),
        (True, "eager"),
        (False, "nonterminal"),
        (True, "nonterminal"),
        (False, "terminal"),
        (True, "terminal"),
        (False, "terminal_attention_only"),
    ],
)
def test_stage_dispatch_preserves_call_settings_and_reader_state(is_mlp, mode):
    layer = _layer().eval()
    layer.config.fp32_residual_connection = True
    layer.config.params_dtype = torch.bfloat16
    layer.hidden_dropout = 0.25
    # A replaced norm can leave stale metadata; the reader still owns the residual.
    layer._input_layernorm_returns_residual = layer._pre_mlp_layernorm_returns_residual = True
    if mode == "terminal_attention_only":
        layer.residual_write_mlp = None
    context = (
        None
        if mode == "eager"
        else ResidualStreamRecomputeContext(
            CheckpointWithoutOutputManager(), is_block_end=mode.startswith("terminal")
        )
    )
    stage = "mlp" if is_mlp else "self_attn"
    reader, writer = getattr(layer, f"residual_read_{stage}"), getattr(
        layer, f"residual_write_{stage}"
    )
    calls = []
    for connection in (reader, writer):
        connection.register_forward_pre_hook(
            lambda module, args, kwargs: calls.append(
                (module, args, kwargs.copy(), torch.is_grad_enabled())
            ),
            with_kwargs=True,
        )
    value = torch.randn(2, 1, 8, device=next(layer.parameters()).device)
    branch, residual, state = _read(layer, value, is_mlp, context)
    assert state is reader.selected_state and residual is state[0]
    torch.testing.assert_close(residual, value * 2)
    branch_output = (branch, None)
    with torch.no_grad():
        _write(layer, branch_output, residual, state, is_mlp, context)
    write_context = (
        None if mode == "terminal_attention_only" or (is_mlp and mode == "terminal") else context
    )
    assert branch.dtype == torch.bfloat16 and len(calls) == 2
    assert calls[0][0] is reader and calls[0][1][0] is value
    assert calls[0][2] == {
        "fp32_residual_connection": True,
        "branch_input_dtype": torch.bfloat16,
        "recompute_context": context,
    }
    assert calls[1][0] is writer and calls[1][1][0] is branch_output
    assert calls[1][2]["state"] is state
    assert calls[1][2]["dropout_probability"] == 0.25
    assert calls[1][2]["training"] is False
    assert calls[1][2]["recompute_context"] is write_context
    assert calls[1][3] is (write_context is None)
    assert layer.events == [
        ("read", reader, is_mlp, context),
        ("write", writer, is_mlp, write_context),
    ]
    if context is not None:
        assert not _checkpoint(layer, is_mlp).retain_input_tensors


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
    setattr(actual, f"recompute_{name}", recompute)
    value = torch.randn(2, 1, 4, device=norm.weight.device, requires_grad=True)
    saved_value = value.detach().clone()
    expected_branch = saved_value.square()
    expected_sum = expected_branch + saved_value
    branch, residual, state = _read(actual, value, is_mlp)
    assert state == () and residual is norm.selected[0]
    assert residual is not value and residual.data_ptr() == value.data_ptr()
    torch.testing.assert_close(residual, value)
    boundary = (branch + residual).square().sum()
    torch.testing.assert_close(boundary, expected_sum.square().sum())
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
    assert branch.untyped_storage().nbytes() > 0
    torch.testing.assert_close(branch, expected_branch)
    torch.testing.assert_close(residual, saved_value)
    assert len(norm.selected) == 1 + int(recompute)
    torch.testing.assert_close(value.grad, 2 * expected_sum * (2 * saved_value + 1))
    torch.testing.assert_close(norm.weight.grad, (2 * expected_sum * expected_branch).sum((0, 1)))
    assert actual.events == []
