# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core import tensor_parallel
from megatron.core.extensions import transformer_engine
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_layer import MambaLayer, MambaLayerSubmodules
from megatron.core.transformer import TransformerConfig, cuda_graphs
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp

_TP_GROUP = object()
_INFERENCE_CONTEXT = object()
_PACKED_SEQ_PARAMS = object()


class _RecordingMixer(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.calls = []

    def forward(
        self,
        hidden_states,
        inference_context=None,
        packed_seq_params=None,
        packed_sequence_cp_metadata=None,
    ):
        self.calls.append(
            (hidden_states, inference_context, packed_seq_params, packed_sequence_cp_metadata)
        )
        return hidden_states + 1, None


def _build_test_layer(recompute_granularity, recompute_modules, *, fp8=False, fp4=False):
    config = TransformerConfig(
        hidden_size=8,
        num_layers=1,
        num_attention_heads=1,
        recompute_granularity=recompute_granularity,
        recompute_modules=recompute_modules,
        use_cpu_initialization=True,
    )
    config.fp8 = fp8
    config.fp4 = fp4
    return MambaLayer(
        config,
        MambaLayerSubmodules(norm=IdentityOp, mixer=_RecordingMixer, mamba_bda=IdentityFuncOp),
        pg_collection=ProcessGroupCollection(tp=_TP_GROUP),
    )


@pytest.mark.parametrize(
    ("granularity", "modules", "training", "inference_context", "fp8"),
    [
        pytest.param(None, ["mamba"], True, None, False, id="recompute-disabled"),
        pytest.param("selective", ["core_attn"], True, None, False, id="mamba-not-selected"),
        pytest.param("selective", ["mamba"], False, None, False, id="eval"),
        pytest.param("selective", ["mamba"], True, _INFERENCE_CONTEXT, True, id="inference"),
    ],
)
def test_mamba_mixer_bypasses_checkpoint(
    monkeypatch, granularity, modules, training, inference_context, fp8
):
    def unexpected_checkpoint(*args, **kwargs):
        pytest.fail("checkpoint should not run")

    monkeypatch.setattr(tensor_parallel, "checkpoint", unexpected_checkpoint)
    monkeypatch.setattr(transformer_engine, "te_checkpoint", unexpected_checkpoint)
    layer = _build_test_layer(granularity, modules, fp8=fp8)
    layer.train(training)
    hidden_states = torch.ones(2, 1, layer.config.hidden_size)

    result = layer._run_mamba_mixer(hidden_states, inference_context, _PACKED_SEQ_PARAMS)

    assert torch.equal(result[0], hidden_states + 1)
    call = layer.mixer.calls[0]
    assert call[0] is hidden_states
    assert call[1:] == (inference_context, _PACKED_SEQ_PARAMS, None)


def test_mamba_mixer_uses_tensor_parallel_checkpoint(monkeypatch):
    checkpoint_call = {}

    def checkpoint(forward_func, distribute_saved_activations, *args):
        checkpoint_call.update(distribute=distribute_saved_activations, args=args)
        return forward_func(*args)

    monkeypatch.setattr(tensor_parallel, "checkpoint", checkpoint)
    layer = _build_test_layer("selective", ["mamba"])
    hidden_states = torch.ones(2, 1, layer.config.hidden_size)

    result = layer._run_mamba_mixer(hidden_states, None, _PACKED_SEQ_PARAMS)

    assert torch.equal(result[0], hidden_states + 1)
    assert checkpoint_call["distribute"] is False
    assert checkpoint_call["args"] == (hidden_states,)


@pytest.mark.parametrize(
    ("fp8", "fp4"), [pytest.param(True, False, id="fp8"), pytest.param(False, True, id="fp4")]
)
def test_mamba_mixer_uses_te_checkpoint_for_quantized_recompute(monkeypatch, fp8, fp4):
    checkpoint_call = {}

    def te_checkpoint(forward_func, distribute, get_rng_state_tracker, tp_group, *args, **kwargs):
        checkpoint_call.update(
            distribute=distribute,
            get_rng_state_tracker=get_rng_state_tracker,
            tp_group=tp_group,
            args=args,
            kwargs=kwargs,
        )
        return forward_func(*args, **kwargs)

    monkeypatch.setattr(transformer_engine, "te_checkpoint", te_checkpoint)
    layer = _build_test_layer("selective", ["mamba"], fp8=fp8, fp4=fp4)
    hidden_states = torch.ones(2, 1, layer.config.hidden_size)

    result = layer._run_mamba_mixer(hidden_states, None, _PACKED_SEQ_PARAMS)

    assert torch.equal(result[0], hidden_states + 1)
    assert checkpoint_call["distribute"] is False
    assert checkpoint_call["get_rng_state_tracker"] is tensor_parallel.random.get_cuda_rng_tracker
    assert checkpoint_call["tp_group"] is _TP_GROUP
    assert checkpoint_call["args"] == (hidden_states,)
    assert checkpoint_call["kwargs"] == {
        "inference_context": None,
        "packed_seq_params": _PACKED_SEQ_PARAMS,
    }


@pytest.mark.parametrize(
    ("fp8", "fp4"), [pytest.param(True, False, id="fp8"), pytest.param(False, True, id="fp4")]
)
@pytest.mark.parametrize("graph_state", ["warmup", "capture"])
def test_quantized_mamba_mixer_bypasses_checkpoint_during_cuda_graph(
    monkeypatch, fp8, fp4, graph_state
):
    def unexpected_checkpoint(*args, **kwargs):
        pytest.fail("checkpoint should not run during CUDA graph warmup or capture")

    monkeypatch.setattr(tensor_parallel, "checkpoint", unexpected_checkpoint)
    monkeypatch.setattr(transformer_engine, "te_checkpoint", unexpected_checkpoint)
    monkeypatch.setattr(cuda_graphs, "is_graph_warmup", lambda: graph_state == "warmup")
    monkeypatch.setattr(cuda_graphs, "is_graph_capturing", lambda: graph_state == "capture")
    layer = _build_test_layer("selective", ["mamba"], fp8=fp8, fp4=fp4)
    hidden_states = torch.ones(2, 1, layer.config.hidden_size)

    result = layer._run_mamba_mixer(hidden_states, None, _PACKED_SEQ_PARAMS)

    assert torch.equal(result[0], hidden_states + 1)
    call = layer.mixer.calls[0]
    assert call[0] is hidden_states
    assert call[1:] == (None, _PACKED_SEQ_PARAMS, None)


def test_mamba_mixer_forwards_packed_sequence_cp_metadata(monkeypatch):
    def checkpoint(forward_func, distribute_saved_activations, *args):
        return forward_func(*args)

    monkeypatch.setattr(tensor_parallel, "checkpoint", checkpoint)
    layer = _build_test_layer("selective", ["mamba"])
    hidden_states = torch.ones(2, 1, layer.config.hidden_size)
    packed_sequence_cp_metadata = object()

    result = layer._run_mamba_mixer(
        hidden_states, None, _PACKED_SEQ_PARAMS, packed_sequence_cp_metadata
    )

    assert torch.equal(result[0], hidden_states + 1)
    call = layer.mixer.calls[0]
    assert call[0] is hidden_states
    assert call[1:] == (None, _PACKED_SEQ_PARAMS, packed_sequence_cp_metadata)
