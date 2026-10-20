# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import logging

import pytest
import torch

from megatron.core import _rank_utils, tensor_parallel
from megatron.core.extensions import transformer_engine
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_layer import MambaLayer, MambaLayerSubmodules
from megatron.core.transformer import TransformerConfig, cuda_graphs
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp

_TP_GROUP = object()
_INFERENCE_CONTEXT = object()
_PACKED_SEQ_PARAMS = object()


@pytest.mark.internal
class TestMambaModelRecompute:
    """Compare real Mamba kernels and parameter gradients with checkpointing enabled."""

    @pytest.mark.parametrize("use_mem_eff_path", [True, False])
    @pytest.mark.parametrize("tp_size,sequence_parallel", [(1, False), (2, False), (2, True)])
    def test_forward_backward_parity(self, tp_size, sequence_parallel, use_mem_eff_path):
        from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
        from megatron.core.models.hybrid.hybrid_model import HybridModel
        from megatron.core.ssm.mamba_mixer import MambaMixer
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        from tests.unit_tests.test_utilities import Utils

        Utils.initialize_model_parallel(tp_size, 1)
        try:
            model_parallel_cuda_manual_seed(123)
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()

            def build_model(recompute):
                config = TransformerConfig(
                    hidden_size=256,
                    num_layers=1,
                    num_attention_heads=4,
                    tensor_model_parallel_size=tp_size,
                    sequence_parallel=sequence_parallel,
                    use_cpu_initialization=True,
                    use_mamba_mem_eff_path=use_mem_eff_path,
                    hidden_dropout=0.0,
                    attention_dropout=0.0,
                    recompute_granularity="selective" if recompute else None,
                    recompute_modules=["mamba"],
                )
                return (
                    HybridModel(
                        config,
                        hybrid_stack_spec,
                        vocab_size=128,
                        max_sequence_length=32,
                        hybrid_layer_pattern="M",
                        parallel_output=False,
                        pg_collection=pg_collection,
                    )
                    .cuda()
                    .train()
                )

            baseline = build_model(False)
            recomputed = build_model(True)
            recomputed.load_state_dict(baseline.state_dict())
            mixer = recomputed.decoder.layers[0].mixer
            assert isinstance(mixer, MambaMixer)
            mixer_grad_modes = []
            hook = mixer.register_forward_hook(
                lambda _module, _inputs, _output: mixer_grad_modes.append(torch.is_grad_enabled())
            )
            input_ids = torch.arange(64, device="cuda").reshape(2, 32)
            position_ids = torch.arange(32, device="cuda").expand(2, -1)
            baseline_logits = baseline(input_ids, position_ids, attention_mask=None)
            recomputed_logits = recomputed(input_ids, position_ids, attention_mask=None)
            torch.testing.assert_close(recomputed_logits, baseline_logits, rtol=0, atol=0)
            baseline_logits.float().square().mean().backward()
            recomputed_logits.float().square().mean().backward()
            hook.remove()
            assert mixer_grad_modes == [False, True], "The real mixer must run again in backward."

            baseline_parameters = dict(baseline.named_parameters())
            for name, parameter in recomputed.named_parameters():
                reference_gradient = baseline_parameters[name].grad
                assert parameter.grad is not None, f"Missing recomputed gradient: {name}"
                assert reference_gradient is not None, f"Missing baseline gradient: {name}"
                assert torch.isfinite(parameter.grad).all(), f"Non-finite gradient: {name}"
                torch.testing.assert_close(parameter.grad, reference_gradient, rtol=1e-5, atol=1e-6)
        finally:
            Utils.destroy_model_parallel()


@pytest.mark.parametrize("rank", [0, 1])
def test_mamba_recompute_warns_with_shortcut_moe(caplog, monkeypatch, rank):
    monkeypatch.setattr(_rank_utils, "safe_get_rank", lambda: rank)
    with caplog.at_level(logging.WARNING, logger="megatron.core.transformer.transformer_config"):
        TransformerConfig(
            hidden_size=256,
            num_layers=2,
            num_attention_heads=4,
            num_moe_experts=8,
            moe_shortcut_connection=True,
            recompute_granularity="selective",
            recompute_modules=["mamba"],
        )
    warnings = [r for r in caplog.records if "Mamba mixer recomputation" in r.message]
    assert len(warnings) == int(rank in _rank_utils.get_default_log_ranks())


@pytest.mark.parametrize(("shortcut_moe", "modules"), [(False, ["mamba"]), (True, ["layernorm"])])
def test_mamba_recompute_warning_only_for_shortcut_moe(caplog, shortcut_moe, modules):
    with caplog.at_level(logging.WARNING, logger="megatron.core.transformer.transformer_config"):
        TransformerConfig(
            hidden_size=256,
            num_layers=2,
            num_attention_heads=4,
            num_moe_experts=8,
            moe_shortcut_connection=shortcut_moe,
            recompute_granularity="selective",
            recompute_modules=modules,
        )
    assert not any("Mamba mixer recomputation" in r.message for r in caplog.records)


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
