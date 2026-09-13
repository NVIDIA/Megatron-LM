# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Full Hybrid recompute across shared CSA2 and single-pass mHC boundaries.

Native CPU adapters exercise the production stack, state, attention and checkpoint
engine. Pipeline cases simulate chunk handoff without claiming NCCL/TE coverage.
"""

from contextlib import contextmanager
from dataclasses import replace

import pytest
import torch

from megatron.core.enums import Fp8Recipe
from megatron.core.models.hybrid import hybrid_block as hybrid_runtime
from megatron.core.models.hybrid.hybrid_stack_adapter import HybridStackForwardContext
from megatron.core.tensor_parallel import random as checkpoint_runtime
from megatron.core.transformer.experimental_attention_variant.csa_utils import (
    csa2_hybrid_adapter as adapter_runtime,
)
from tests.unit_tests.transformer.experimental_attention_variant import (
    test_csa2_recompute as recompute_helpers,
)
from tests.unit_tests.transformer.experimental_attention_variant.test_csa2 import (
    _packed,
    _record_losses,
)
from tests.unit_tests.transformer.experimental_attention_variant.test_csa2_pipeline import (
    _config,
    _run_chunks,
    _stack,
)

_stacks = recompute_helpers._stacks
cpu_checkpoint_rng = recompute_helpers.cpu_checkpoint_rng

pytestmark = pytest.mark.usefixtures("cpu_checkpoint_rng")


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


def _assert_losses(actual, expected):
    assert len(actual) == len(expected)
    assert {item["layer_number"] for item in actual} == {item["layer_number"] for item in expected}
    for layer_number in {item["layer_number"] for item in expected}:
        actual_losses = (
            torch.stack([item["loss"] for item in actual if item["layer_number"] == layer_number])
            .sort()
            .values
        )
        expected_losses = (
            torch.stack([item["loss"] for item in expected if item["layer_number"] == layer_number])
            .sort()
            .values
        )
        torch.testing.assert_close(actual_losses, expected_losses)


@pytest.mark.parametrize("layout", ["sbhd", "thd"])
@pytest.mark.parametrize(
    "method,count,cuts,mhc,coefficient,dtype",
    [
        ("uniform", 1, (), True, 0.3, torch.float32),
        ("uniform", 3, (), True, 0.3, torch.float32),
        ("uniform", 5, (), False, 0.3, torch.float32),
        ("block", 3, (), True, 0.3, torch.float32),
        ("block", 7, (), False, 0.3, torch.float32),
        ("uniform", 3, (1, 5, 7, 8, 10), True, 0.3, torch.float32),
        ("uniform", 5, (4, 7, 8, 10), False, 0.3, torch.float32),
        ("block", 3, (1, 5, 7, 8, 10), True, 0.3, torch.float32),
        ("uniform", 3, (1, 5, 7, 8, 10), True, 0.0, torch.float32),
        ("block", 3, (4, 7, 8, 10), False, 0.0, torch.float32),
        ("uniform", 3, (1, 5, 7, 8, 10), True, 0.3, torch.bfloat16),
    ],
)
def test_full_recompute_shared_state_and_outstanding_microbatches(
    monkeypatch, layout, method, count, cuts, mhc, coefficient, dtype
):
    """Group cuts cross Full/Reindex/Reuse and an FFN-only chunk relays shared state."""
    torch.manual_seed(391)
    records = _record_losses(monkeypatch)
    reference = _stack(_config(dtype, coefficient, mhc))
    config = _full_config(reference.config, method, count)
    stacks, plan = _stacks(reference, config, cuts, layout)
    calls = _observe_layers(stacks)
    params, shape = None, (9, 2)
    if layout == "thd":
        params, _, valid = _packed([1, 4, 0, 2], [2, 5, 0, 4], tail=3)
        shape = (valid.numel(), 1)
    expected_records, actual_records, pairs = [], [], []
    for _ in range(2):
        x = torch.randn(*shape, config.hidden_size, dtype=dtype, requires_grad=True)
        reference_x = x.detach().clone().requires_grad_()
        expected = reference(reference_x, None, packed_seq_params=params)
        expected_records.extend(records)
        records.clear()
        actual, payloads = _run_chunks(stacks, plan, x, params)
        actual_records.extend(records)
        records.clear()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for payload in payloads:
            assert (
                payload.tensor_specs
                == payload.boundary.payload_spec(config, *shape, params).tensor_specs
            )
        pairs.append((actual, expected, x, reference_x))
    assert len(calls) == 2 * config.num_layers
    assert any(not grad_enabled for _, grad_enabled in calls)
    for actual, expected, x, reference_x in reversed(pairs):
        probe = torch.randn_like(actual) * 0.1
        (actual * probe).sum().backward()
        actual_records.extend(records)
        records.clear()
        (expected * probe).sum().backward()
        assert not records
        _assert_gradient(x.grad, reference_x.grad, dtype=dtype)
    replayed_layers = sum(
        len(stack.layers) if method == "uniform" else min(count, len(stack.layers))
        for stack in stacks
    )
    assert len(calls) == 2 * (config.num_layers + replayed_layers)
    _assert_parameter_gradients(reference, stacks, plan, dtype)
    assert len(expected_records) == (6 if coefficient else 0)
    _assert_losses(actual_records, expected_records)


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
    index = actual.boundary.field_names.index(field)
    actual.tensors[index].square().sum().backward()
    expected.tensors[index].square().sum().backward()
    _assert_gradient(x.grad, reference_x.grad)
    for parameter, ref_parameter in zip(stacks[0].parameters(), reference_stacks[0].parameters()):
        _assert_gradient(parameter.grad, ref_parameter.grad)


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
    index = actual.boundary.field_names.index("indexer_k")
    # The zero hidden objective invokes the same auxiliary autoscaler in both
    # paths: reentrant checkpoint backward also materializes unused-output zeros.
    (actual.tensors[index].square().sum() + actual.tensors[0].sum() * 0).backward()
    (expected.tensors[index].square().sum() + expected.tensors[0].sum() * 0).backward()
    _assert_gradient(x.grad, reference_x.grad)
    for parameter, ref_parameter in zip(stacks[0].parameters(), reference_stacks[0].parameters()):
        _assert_gradient(parameter.grad, ref_parameter.grad)


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


@pytest.mark.parametrize("method", ["uniform", "block"])
def test_full_recompute_no_grad_bypasses_checkpoint(monkeypatch, method):
    _record_losses(monkeypatch)
    stack = _stack(_full_config(_config(), method, 3))

    def unexpected_checkpoint(*args, **kwargs):
        raise AssertionError("A no-grad forward must not allocate activation checkpoints")

    monkeypatch.setattr(checkpoint_runtime.CheckpointFunction, "forward", unexpected_checkpoint)
    with torch.no_grad():
        output = stack(torch.randn(9, 2, stack.config.hidden_size), None)
    assert torch.isfinite(output).all()


def test_full_recompute_short_sequence_has_empty_compressed_state(monkeypatch):
    torch.manual_seed(642)
    _record_losses(monkeypatch)
    reference = _stack(_config())
    stacks, plan = _stacks(reference, _full_config(reference.config, "uniform", 3), (5, 7), "sbhd")
    x = torch.randn(1, 1, reference.config.hidden_size, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    actual, _ = _run_chunks(stacks, plan, x)
    expected = reference(reference_x, None)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.square().sum().backward()
    expected.square().sum().backward()
    _assert_gradient(x.grad, reference_x.grad)
    _assert_parameter_gradients(reference, stacks, plan)


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


@pytest.mark.parametrize("method", ["uniform", "block"])
@pytest.mark.parametrize("mhc", [False, True])
def test_full_recompute_frozen_hidden_input_preserves_parameter_gradients(monkeypatch, method, mhc):
    """A frozen embedding must not disconnect trainable layers from reentrant backward."""
    torch.manual_seed(924)
    _record_losses(monkeypatch)
    reference = _stack(_config(enable_hyper_connections=mhc))
    actual = _stack(_full_config(reference.config, method, 3))
    actual.load_state_dict(reference.state_dict())
    x = torch.randn(9, 2, reference.config.hidden_size)
    expected = reference(x, None)
    output = actual(x, None)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    output.square().sum().backward()
    expected.square().sum().backward()
    assert any(parameter.grad is not None for parameter in actual.parameters())
    for parameter, ref_parameter in zip(actual.parameters(), reference.parameters()):
        _assert_gradient(parameter.grad, ref_parameter.grad)


@pytest.mark.parametrize("quantization", ["fp8", "fp4", "quant_recipe"])
def test_full_recompute_quantization_dispatch_and_layer_context(monkeypatch, quantization):
    """Validate TE checkpoint dispatch and global layer contexts without quantized kernels."""
    torch.manual_seed(437)
    _record_losses(monkeypatch)
    reference = _stack(_config())
    config = _full_config(reference.config, "uniform", 3)
    stacks, plan = _stacks(reference, config, (5,), "sbhd")
    # These adapters contain ordinary CPU linears. Enabling only dispatch after
    # construction isolates checkpoint/context wiring from TE recipe dependencies.
    if quantization == "fp8":
        config.fp8 = "e4m3"
        config.fp8_recipe = Fp8Recipe.tensorwise
    elif quantization == "fp4":
        config.fp4 = "e2m1"
    else:
        config.quant_recipe = object()
    contexts, checkpoint_calls = [], []

    @contextmanager
    def quantization_context(config, layer_number=None):
        contexts.append((layer_number, torch.is_grad_enabled()))
        yield

    def te_checkpoint(function, distribute, rng_tracker, tp_group, *inputs):
        assert rng_tracker is checkpoint_runtime.get_cuda_rng_tracker
        assert any(tp_group is stack.pg_collection.tp for stack in stacks)
        checkpoint_calls.append(len(inputs))
        return checkpoint_runtime.checkpoint(function, distribute, *inputs)

    monkeypatch.setattr(adapter_runtime, "te_checkpoint", te_checkpoint)
    monkeypatch.setattr(hybrid_runtime, "get_fp8_context", quantization_context)
    monkeypatch.setattr(hybrid_runtime, "get_fp4_context", quantization_context)
    x = torch.randn(9, 2, reference.config.hidden_size, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    actual, _ = _run_chunks(stacks, plan, x)
    expected = reference(reference_x, None)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.square().sum().backward()
    expected.square().sum().backward()
    _assert_gradient(x.grad, reference_x.grad)
    _assert_parameter_gradients(reference, stacks, plan)
    assert len(checkpoint_calls) == 5
    if quantization == "quant_recipe":
        assert not contexts
    else:
        assert sorted(contexts) == [
            (layer, grad_enabled)
            for layer in range(config.num_layers)
            for grad_enabled in (False, True)
        ]


@pytest.mark.parametrize("method", ["uniform", "block"])
def test_full_recompute_legacy_mhc(monkeypatch, method):
    """The existing multi-pass mHC wrapper shares the same full-recompute state path."""
    if not torch.cuda.is_available():
        monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda message: None)
        monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: None)
    torch.manual_seed(845)
    _record_losses(monkeypatch)
    config = replace(_config(), mhc_single_pass=False)
    reference = _stack(config)
    actual_stack = _stack(_full_config(config, method, 3))
    actual_stack.load_state_dict(reference.state_dict())
    x = torch.randn(9, 2, config.hidden_size, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    actual = actual_stack(x, None)
    expected = reference(reference_x, None)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.square().sum().backward()
    expected.square().sum().backward()
    _assert_gradient(x.grad, reference_x.grad)
    for parameter, ref_parameter in zip(actual_stack.parameters(), reference.parameters()):
        _assert_gradient(parameter.grad, ref_parameter.grad)


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


def test_recompute_restore_rebuilds_differentiable_fused_k_views(monkeypatch):
    """Replay flat buffers must remain connected to explicit checkpoint state inputs."""
    _record_losses(monkeypatch)
    config = _config()
    stacks, _ = _stacks(_stack(config), config, (7,), "sbhd")
    payload = stacks[0](torch.randn(9, 2, config.hidden_size, requires_grad=True), None)
    _, state, mhc_state, _ = payload.restore()
    state.global_kv_flat = state.indexer_k_flat = None
    with torch.no_grad():
        state.prepare_fused_kv()
    assert not state.global_kv_flat.requires_grad and not state.indexer_k_flat.requires_grad
    context = HybridStackForwardContext(layer_kwargs={"csa2_state": state}, mhc_state=mhc_state)
    adapter = stacks[0].forward_adapter
    tensors, metadata = adapter._export_recompute_context(context)
    assert all(
        getattr(metadata[0], name) is None
        for name in ("global_kv", "indexer_k", "global_kv_flat", "indexer_k_flat")
    )
    positional = tuple(tensor.detach().requires_grad_() for tensor in tensors)
    monkeypatch.setattr(
        "megatron.core.transformer.experimental_attention_variant.csa_utils."
        "csa2_hybrid_adapter.use_fused_dsa_kernels",
        lambda config: True,
    )
    replay = adapter._restore_recompute_context(positional, metadata)
    replay_state = replay.layer_kwargs["csa2_state"]
    objective = (
        replay.mhc_state.pre_mix.square().sum()
        + replay_state.global_kv_flat.square().sum()
        + replay_state.indexer_k_flat.square().sum()
    )
    gradients = torch.autograd.grad(objective, positional)
    for tensor, gradient in zip(positional, gradients):
        torch.testing.assert_close(gradient, 2 * tensor)
