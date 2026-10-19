# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared-prefix Mamba runs through ``MambaLayer.__call__`` and ``MambaMixer.__call__``.

Module hooks, such as Megatron FSDP's unshard, prefetch and reshard hooks and its per-module
parameter gathers, fire only on module calls, so the shared-prefix forward must reach the mixer
through the layer as the dense forward does. The CPU tests use bare modules and a stubbed mixer
half; the GPU test runs each differentiable backend on a real layer.
"""

import pytest
import torch

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.shared_prefix_layout import SharedPrefixLayout
from megatron.core.ssm import mamba_shared_prefix
from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.ssm.mamba_mixer import MambaMixer, causal_conv1d_fn, mamba_chunk_scan_combined
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.shared_prefix.ssm.shared_prefix_test_utils import build_local_mamba_layer
from tests.unit_tests.test_utilities import Utils

_LAYOUT = SharedPrefixLayout(prefix_len=3, completion_lens=(2, 5, 1))
_HIDDEN = 8

# The request flags each NRL_SP_MAMBA_IMPL value (None: the default) sets for a training layer.
_FLAGS = ("replay_prefix", "packed_recurrence", "ragged_state_fork")
_SET_FLAGS = {
    None: {"ragged_state_fork"},
    "state_fork": set(),
    "replay_prefix": {"replay_prefix"},
    "packed_recurrence": {"packed_recurrence"},
}


def _bare(cls, config):
    """An instance with only ``config`` set: no parameters, process groups or kernels."""
    module = cls.__new__(cls)
    torch.nn.Module.__init__(module)
    module.config = config
    return module


def _layer(mixer_cls=MambaMixer):
    """A training MambaLayer with an identity norm, no dropout and a bare mixer."""
    config = TransformerConfig(
        num_layers=1,
        hidden_size=_HIDDEN,
        num_attention_heads=2,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bias_dropout_fusion=False,
    )
    layer = _bare(MambaLayer, config)
    layer.layer_number = 1
    layer.hidden_dropout = 0.0
    layer.norm = torch.nn.Identity()
    layer.mamba_bda = get_bias_dropout_add
    layer.bias_dropout_add_exec_handler = torch.enable_grad
    layer.mixer = _bare(mixer_cls, config)
    return layer


def _select_impl(monkeypatch, impl):
    if impl is None:
        monkeypatch.delenv("NRL_SP_MAMBA_IMPL", raising=False)
    else:
        monkeypatch.setenv("NRL_SP_MAMBA_IMPL", impl)


def _record_module_calls(layer, events):
    for name, module in (("layer", layer), ("mixer", layer.mixer)):
        module.register_forward_pre_hook(lambda m, args, name=name: events.append(f"{name}_pre"))
        module.register_forward_hook(lambda m, args, out, name=name: events.append(f"{name}_post"))


def _assert_request_removed(layer):
    assert "_shared_prefix_call" not in vars(layer.mixer)
    assert layer.mixer._shared_prefix_call is None


@pytest.mark.parametrize("impl", list(_SET_FLAGS))
def test_shared_prefix_forward_runs_layer_and_mixer_module_calls(monkeypatch, impl):
    events, requests = [], []

    def mixer_half(mixer, hidden_states, call):
        call.consumed = True
        requests.append(call)
        events.append("scan")
        return 2 * hidden_states, None

    monkeypatch.setattr(mamba_shared_prefix, "forward_mamba_mixer_shared_prefix", mixer_half)
    _select_impl(monkeypatch, impl)
    layer = _layer()
    _record_module_calls(layer, events)
    hidden_states = torch.randn(_LAYOUT.total_len, 1, _HIDDEN)

    output = mamba_shared_prefix.forward_mamba_layer_shared_prefix(layer, hidden_states, _LAYOUT)

    assert events == ["layer_pre", "mixer_pre", "scan", "mixer_post", "layer_post"]
    (request,) = requests
    assert request.layout is _LAYOUT
    assert {flag for flag in _FLAGS if getattr(request, flag)} == _SET_FLAGS[impl]
    # MambaLayer.forward adds the residual to the mixer output.
    torch.testing.assert_close(output, 3 * hidden_states)
    _assert_request_removed(layer)


def test_request_is_removed_when_the_layer_raises(monkeypatch):
    def mixer_half(mixer, hidden_states, call):
        raise RuntimeError("boom")

    monkeypatch.setattr(mamba_shared_prefix, "forward_mamba_mixer_shared_prefix", mixer_half)
    monkeypatch.delenv("NRL_SP_MAMBA_IMPL", raising=False)
    layer = _layer()
    with pytest.raises(RuntimeError, match="boom"):
        mamba_shared_prefix.forward_mamba_layer_shared_prefix(
            layer, torch.zeros(_LAYOUT.total_len, 1, _HIDDEN), _LAYOUT
        )
    _assert_request_removed(layer)


class _DenseOnlyMixer(MambaMixer):
    """A mixer whose forward never reads the shared-prefix request."""

    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states, None


def test_forward_that_ignores_the_request_fails_closed(monkeypatch):
    monkeypatch.delenv("NRL_SP_MAMBA_IMPL", raising=False)
    layer = _layer(_DenseOnlyMixer)
    with pytest.raises(RuntimeError, match="ignored the shared-prefix request"):
        mamba_shared_prefix.forward_mamba_layer_shared_prefix(
            layer, torch.zeros(_LAYOUT.total_len, 1, _HIDDEN), _LAYOUT
        )
    _assert_request_removed(layer)


def test_nested_dispatch_is_rejected(monkeypatch):
    layer = _layer()

    def mixer_half(mixer, hidden_states, call):
        call.consumed = True
        inner = mamba_shared_prefix.forward_mamba_layer_shared_prefix(layer, hidden_states, _LAYOUT)
        return inner, None

    monkeypatch.setattr(mamba_shared_prefix, "forward_mamba_mixer_shared_prefix", mixer_half)
    monkeypatch.delenv("NRL_SP_MAMBA_IMPL", raising=False)
    with pytest.raises(RuntimeError, match="nested shared-prefix Mamba dispatch"):
        mamba_shared_prefix.forward_mamba_layer_shared_prefix(
            layer, torch.zeros(_LAYOUT.total_len, 1, _HIDDEN), _LAYOUT
        )
    _assert_request_removed(layer)


@pytest.mark.parametrize(
    "kwargs", [{"packed_seq_params": object()}, {"inference_context": object()}]
)
def test_mixer_rejects_packed_or_inference_inputs_with_a_request(kwargs):
    mixer = _layer().mixer
    mixer._shared_prefix_call = mamba_shared_prefix.MambaSharedPrefixCall(_LAYOUT)
    with pytest.raises(NotImplementedError, match="static, unpacked training"):
        mixer(torch.zeros(_LAYOUT.total_len, 1, _HIDDEN), **kwargs)


@pytest.mark.skipif(
    causal_conv1d_fn is None or mamba_chunk_scan_combined is None or not torch.cuda.is_available(),
    reason="requires CUDA, causal-conv1d and mamba-ssm",
)
class TestSharedPrefixMambaModuleCalls:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        "impl", ["ragged_state_fork", "state_fork", "replay_prefix", "packed_recurrence"]
    )
    def test_real_layer_calls_its_submodules(self, monkeypatch, impl):
        monkeypatch.setenv("NRL_SP_MAMBA_IMPL", impl)
        layer = build_local_mamba_layer(torch.bfloat16)
        mixer = layer.mixer
        events = []
        submodules = {
            "layer": layer,
            "layer.norm": layer.norm,
            "mixer": mixer,
            "in_proj": mixer.in_proj,
            "mixer.norm": mixer.norm,
            "out_proj": mixer.out_proj,
        }
        for name, module in submodules.items():
            module.register_forward_pre_hook(lambda m, args, name=name: events.append(name))
        # The prompt crosses the 128-token scan chunk, so state forking splits it.
        layout = SharedPrefixLayout(prefix_len=150, completion_lens=(17, 5, 30))
        mamba_shared_prefix.validate_mamba_layer_shared_prefix(layer, tp_size=1, cp_size=1)
        hidden_states = torch.randn(
            layout.total_len, 1, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )

        output = mamba_shared_prefix.forward_mamba_layer_shared_prefix(layer, hidden_states, layout)
        output.float().square().sum().backward()

        assert events == list(submodules)
        assert output.shape == hidden_states.shape
        assert hidden_states.grad is not None
        assert all(parameter.grad is not None for parameter in layer.parameters())
        _assert_request_removed(layer)
