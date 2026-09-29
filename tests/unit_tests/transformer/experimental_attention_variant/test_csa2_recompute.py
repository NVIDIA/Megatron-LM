# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Selective and full Hybrid recompute with shared CSA2 state and outstanding microbatches.

Native CPU adapters exercise the production stack, state, attention and checkpoint
engine. Pipeline cases simulate chunk handoff without claiming NCCL/TE coverage.
"""

from dataclasses import replace

import pytest
import torch

from megatron.core.tensor_parallel import random as checkpoint_runtime
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant import (
    deepseek_v4_hybrid_attention as dsv4_attention,
)
from megatron.core.transformer.experimental_attention_variant.csa import (
    CompressedSparseAttentionSubmodules,
    CompressorSubmodules,
)
from megatron.core.transformer.experimental_attention_variant.csa2 import (
    CompressedSparseAttention2,
    CSA2Compressor,
    CSA2Indexer,
    CSA2IndexerSubmodules,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_pipeline import (
    build_csa2_pipeline_plan,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from tests.unit_tests.transformer.experimental_attention_variant.test_csa2 import (
    _CPUFrequencyTable,
    _Linear,
    _record_losses,
    _RMSNorm,
)
from tests.unit_tests.transformer.experimental_attention_variant.test_csa2_pipeline import (
    _Attention,
    _config,
    _pattern,
    _stack,
)


@pytest.fixture
def cpu_checkpoint_rng(monkeypatch):
    """Keep real CPU RNG snapshot/restore on hosts without a CUDA generator."""
    if not torch.cuda.is_available():
        monkeypatch.setattr(checkpoint_runtime, "_get_cuda_rng_state", lambda **kwargs: None)
        monkeypatch.setattr(checkpoint_runtime, "_set_cuda_rng_state", lambda *args, **kwargs: None)


@pytest.fixture
def checkpoints(monkeypatch, cpu_checkpoint_rng):
    """Observe the real manager's storage discard and replay, without replacing either."""
    managers = []
    original_init = checkpoint_runtime.MHCCheckpointManager.__init__

    def record_init(manager):
        original_init(manager)
        managers.append(manager)

    monkeypatch.setattr(checkpoint_runtime.MHCCheckpointManager, "__init__", record_init)
    return managers


def _recompute_config(config, group_size):
    return replace(
        config,
        recompute_granularity="selective",
        recompute_modules=["mhc"],
        mhc_recompute_layer_num=group_size,
    )


def _stacks(reference, config, cuts, layout, attention=_Attention):
    plan = build_csa2_pipeline_plan(config, _pattern(cuts), qkv_format=layout)
    stacks = [_stack(config, chunk, attention=attention) for chunk in plan]
    for stack, chunk in zip(stacks, plan):
        for local, layer in enumerate(stack.layers):
            layer.load_state_dict(reference.layers[chunk.layer_offset + local].state_dict())
    return stacks, plan


class _NativeDSv4Attention(dsv4_attention.DSv4HybridSelfAttention):
    """Execute production attention/QKV/recompute with ordinary PyTorch linear modules."""

    def __init__(self, config, layer_number, pg_collection, **kwargs):
        super().__init__(
            config,
            dsv4_attention.DSv4HybridSelfAttentionSubmodules(
                q_layernorm=_RMSNorm,
                kv_layernorm=_RMSNorm,
                linear_q_down_proj=_Linear,
                linear_q_up_proj=_Linear,
                linear_kv_proj=_Linear,
                linear_proj=_Linear,
                core_attention=ModuleSpec(
                    CompressedSparseAttention2,
                    submodules=CompressedSparseAttentionSubmodules(
                        compressor=ModuleSpec(
                            CSA2Compressor,
                            submodules=CompressorSubmodules(_Linear, _Linear, _RMSNorm),
                        ),
                        indexer=ModuleSpec(
                            CSA2Indexer,
                            submodules=CSA2IndexerSubmodules(_Linear, _Linear, _RMSNorm, _Linear),
                        ),
                    ),
                ),
            ),
            layer_number,
            attn_mask_type=AttnMaskType.causal,
            pg_collection=pg_collection,
        )


@pytest.fixture
def native_attention(monkeypatch):
    """Adapt GPU frequency-table allocation and the TE constructor gate, preserving math."""

    class RotaryTable(_CPUFrequencyTable):
        def __init__(self, dim, **kwargs):
            super().__init__(dim)

        def get_rotary_seq_len(self, inference_context, transformer, hidden, config, params):
            return hidden.shape[0] if params is None else params.max_seqlen_q

    class YarnTable(RotaryTable):
        def forward(self, length, packed_seq=False):
            return super().forward(length, packed_seq=packed_seq), 1.0

    monkeypatch.setattr(dsv4_attention, "TELinear", _Linear)
    monkeypatch.setattr(dsv4_attention, "RotaryEmbedding", RotaryTable)
    monkeypatch.setattr(dsv4_attention, "YarnRotaryEmbedding", YarnTable)
    return _NativeDSv4Attention


def _full_config(config, method, count):
    return replace(
        config, recompute_granularity="full", recompute_method=method, recompute_num_layers=count
    )


def _assert_gradient(actual, expected, *, dtype=torch.float32):
    # Reentrant checkpoints may materialize zero gradients on unused side outputs.
    if actual is None and expected is not None:
        actual = torch.zeros_like(expected)
    elif expected is None and actual is not None:
        expected = torch.zeros_like(actual)
    tolerance = dict(atol=2e-7, rtol=2e-5) if dtype == torch.float32 else dict(atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(actual, expected, **tolerance)


def _assert_parameter_gradients(reference, stacks, plan, dtype=torch.float32):
    for stack, chunk in zip(stacks, plan):
        for local, layer in enumerate(stack.layers):
            expected = dict(reference.layers[chunk.layer_offset + local].named_parameters())
            for name, parameter in layer.named_parameters():
                try:
                    _assert_gradient(parameter.grad, expected[name].grad, dtype=dtype)
                except AssertionError as error:
                    raise AssertionError(f"Layer {chunk.layer_offset + local}: {name}") from error


def _observe_layers(stacks):
    calls = []
    for stack in stacks:
        for layer in stack.layers:
            layer.register_forward_pre_hook(
                lambda module, inputs: calls.append((module.layer_number, torch.is_grad_enabled()))
            )
    return calls


@pytest.mark.usefixtures("cpu_checkpoint_rng")
@pytest.mark.parametrize("field", ["pre_mix", "global_kv"])
@pytest.mark.parametrize("method,count", [("uniform", 3), ("block", 5)])
def test_full_recompute_side_output_only_backward(monkeypatch, field, method, count):
    """A PP side-output objective must replay the producer without a hidden-state hook."""
    torch.manual_seed(182)
    _record_losses(monkeypatch)
    config = _config(coefficient=0)
    reference = _stack(config)
    stacks, _ = _stacks(reference, _full_config(config, method, count), (7,), "sbhd")
    reference_stacks, _ = _stacks(reference, config, (7,), "sbhd")
    x = torch.randn(9, 2, config.hidden_size, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    actual = stacks[0](x, None)
    expected = reference_stacks[0](reference_x, None)
    index = tuple(s.name for s in actual.tensor_specs).index(field)
    actual.tensors[index].square().sum().backward()
    expected.tensors[index].square().sum().backward()
    _assert_gradient(x.grad, reference_x.grad)
    for parameter, ref_parameter in zip(stacks[0].parameters(), reference_stacks[0].parameters()):
        _assert_gradient(parameter.grad, ref_parameter.grad)


@pytest.mark.usefixtures("cpu_checkpoint_rng")
@pytest.mark.parametrize("method,count", [("uniform", 3), ("block", 5)])
def test_full_recompute_indexer_k_boundary_gradient(monkeypatch, method, count):
    """Exercise indexer K gradients with the auxiliary objective explicitly enabled."""
    torch.manual_seed(192)
    _record_losses(monkeypatch)
    config = _config()
    reference = _stack(config)
    stacks, _ = _stacks(reference, _full_config(config, method, count), (7,), "sbhd")
    reference_stacks, _ = _stacks(reference, config, (7,), "sbhd")
    x = torch.randn(9, 2, config.hidden_size, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    actual = stacks[0](x, None)
    expected = reference_stacks[0](reference_x, None)
    index = tuple(s.name for s in actual.tensor_specs).index("indexer_k")
    # The zero hidden objective invokes the same auxiliary autoscaler in both
    # paths: reentrant checkpoint backward also materializes unused-output zeros.
    (actual.tensors[index].square().sum() + actual.tensors[0].sum() * 0).backward()
    (expected.tensors[index].square().sum() + expected.tensors[0].sum() * 0).backward()
    _assert_gradient(x.grad, reference_x.grad)
    for parameter, ref_parameter in zip(stacks[0].parameters(), reference_stacks[0].parameters()):
        _assert_gradient(parameter.grad, ref_parameter.grad)


@pytest.mark.usefixtures("cpu_checkpoint_rng")
@pytest.mark.parametrize("method,count", [("uniform", 3), ("block", 7)])
@pytest.mark.parametrize("mhc", [False, True])
def test_full_recompute_restores_dropout_rng(monkeypatch, method, count, mhc):
    torch.manual_seed(816)
    _record_losses(monkeypatch)
    config = replace(_config(enable_hyper_connections=mhc), hidden_dropout=0.3)
    reference = _stack(config)
    actual = _stack(_full_config(config, method, count))
    actual.load_state_dict(reference.state_dict())
    x = torch.randn(9, 2, config.hidden_size, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    rng = torch.get_rng_state()
    expected = reference(reference_x, None)
    torch.set_rng_state(rng)
    output = actual(x, None)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    after_forward = torch.get_rng_state()
    output.square().sum().backward()
    assert torch.equal(torch.get_rng_state(), after_forward)
    expected.square().sum().backward()
    _assert_gradient(x.grad, reference_x.grad)
    for parameter, ref_parameter in zip(actual.parameters(), reference.parameters()):
        _assert_gradient(parameter.grad, ref_parameter.grad)


@pytest.mark.usefixtures("cpu_checkpoint_rng")
def test_full_recompute_reduces_saved_activation_storage(monkeypatch):
    """Checkpoints must not retain the interior CSA2 graph through mutable shared state."""
    torch.manual_seed(219)
    _record_losses(monkeypatch)
    reference = _stack(_config())
    actual = _stack(_full_config(reference.config, "uniform", 3))
    actual.load_state_dict(reference.state_dict())
    saved_bytes = []
    for stack in (reference, actual):
        parameter_ptrs = {
            parameter.untyped_storage().data_ptr() for parameter in stack.parameters()
        }
        saved = {}

        def observe(tensor):
            storage = tensor.untyped_storage()
            if storage.data_ptr() not in parameter_ptrs:
                saved[storage.data_ptr()] = storage.nbytes()
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(observe, lambda tensor: tensor):
            output = stack(torch.randn(9, 2, stack.config.hidden_size, requires_grad=True), None)
        saved_bytes.append(sum(saved.values()))
        output.square().sum().backward()
    assert 0 < saved_bytes[1] < saved_bytes[0]


@pytest.mark.usefixtures("cpu_checkpoint_rng")
def test_full_recompute_frozen_hidden_with_live_pipeline_state(monkeypatch):
    """Live side inputs suffice for checkpointing even when received hidden is frozen."""
    torch.manual_seed(481)
    _record_losses(monkeypatch)
    config = _config(coefficient=0)
    reference = _stack(config)
    stacks, _ = _stacks(reference, _full_config(config, "uniform", 3), (7,), "sbhd")
    reference_stacks, _ = _stacks(reference, config, (7,), "sbhd")
    calls = _observe_layers(stacks[1:])
    x = torch.randn(9, 2, config.hidden_size, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    outputs = []
    for chunk_stacks, inputs in ((stacks, x), (reference_stacks, reference_x)):
        payload = chunk_stacks[0](inputs, None)
        payload = replace(payload, tensors=(payload.tensors[0].detach(), *payload.tensors[1:]))
        chunk_stacks[1].set_input_tensor(payload)
        outputs.append(chunk_stacks[1](None, None))
    torch.testing.assert_close(*outputs, atol=0, rtol=0)
    assert all(not enabled for _, enabled in calls)
    for output in outputs:
        output.square().sum().backward()
    assert len(calls) == 2 * len(stacks[1].layers)
    _assert_gradient(x.grad, reference_x.grad)
    for stack, ref_stack in zip(stacks, reference_stacks):
        for parameter, ref_parameter in zip(stack.parameters(), ref_stack.parameters()):
            _assert_gradient(parameter.grad, ref_parameter.grad)


@pytest.mark.usefixtures("cpu_checkpoint_rng")
def test_recompute_restore_rebuilds_differentiable_fused_k_views(monkeypatch):
    """Replay flat buffers must remain connected to explicit checkpoint state inputs."""
    _record_losses(monkeypatch)
    config = _config()
    stacks, _ = _stacks(_stack(config), config, (7,), "sbhd")
    payload = stacks[0](torch.randn(9, 2, config.hidden_size, requires_grad=True), None)
    hidden, context, _ = payload.restore()
    state = context.cross_layer_state
    state.global_kv_flat = state.indexer_k_flat = None
    with torch.no_grad():
        state.prepare_fused_kv()
    assert not state.global_kv_flat.requires_grad and not state.indexer_k_flat.requires_grad
    # Only declare fused views here; the CPU forward above used native attention.
    config.dsa_kernel_backend = "cudnn"
    region = stacks[1].forward_adapter.checkpoint_region(0, 2, hidden, context)
    tensors = region.codec.export(context, region.schema.inputs)
    positional = tuple(
        tensor.detach().requires_grad_(tensor.requires_grad) if tensor is not None else None
        for tensor in tensors
    )
    replay = region.codec.restore(region.schema.inputs, positional, region.input_metadata)
    replay_state = replay.cross_layer_state
    objective = (
        replay.mhc_state.pre_mix.square().sum()
        + replay_state.global_kv_flat.square().sum()
        + replay_state.indexer_k_flat.square().sum()
    )
    differentiable = tuple(
        tensor for tensor in positional if tensor is not None and tensor.requires_grad
    )
    gradients = torch.autograd.grad(objective, differentiable)
    for tensor, gradient in zip(differentiable, gradients):
        torch.testing.assert_close(gradient, 2 * tensor)
