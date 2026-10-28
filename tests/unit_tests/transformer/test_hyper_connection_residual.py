# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""The mHC adapter's ownership, selected state, and checkpoint callback contract."""

import copy
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn

from megatron.core.tensor_parallel.random import (
    CheckpointWithoutOutputManager,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.hyper_connection import (
    HyperConnectionExecutionContext,
    HyperConnectionModule,
    HyperConnectionResidualConnection,
)
from megatron.core.transformer.residual_recompute import ResidualStreamRecomputeContext
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _model():
    config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        use_cpu_initialization=True,
        enable_mhc_connections=True,
        mhc_num_residual_streams=4,
        mhc_sinkhorn_iterations=3,
        use_fused_mhc=False,
    )
    model = nn.Module()
    model.owner = HyperConnectionModule(config, layer_number=1)
    model.read = HyperConnectionResidualConnection(config, operation="read")
    model.write = HyperConnectionResidualConnection(config, operation="write")
    return model.to(device="cuda" if torch.cuda.is_available() else "cpu")


def _step(model, value, bias=None, context=None, dropout=0.25):
    execution = HyperConnectionExecutionContext(model.owner)
    branch, state = model.read(value, execution_context=execution, recompute_context=context)
    return model.write(
        (branch.sin(), bias),
        state=state,
        dropout_probability=dropout,
        training=True,
        execution_context=execution,
        recompute_context=context,
    )


def _rng():
    return [torch.get_rng_state()] + (
        [torch.cuda.get_rng_state()] if torch.cuda.is_available() else []
    )


@pytest.mark.parametrize("deepcopy", [False, True])
@torch.compiler.set_stance("force_eager")
def test_adapter_owns_no_parameters_and_resolves_replaced_owner(deepcopy):
    original = _model()
    value = torch.randn(2, 1, 32, device=next(original.parameters()).device, requires_grad=True)
    _step(original, value)  # Populate any accidental cached dependency.
    model = copy.deepcopy(original) if deepcopy else original
    old_owner = original.owner
    if not deepcopy:
        model.owner = copy.deepcopy(old_owner)
    old_owner.register_forward_hook(lambda *args: pytest.fail("The stale owner was invoked"))
    for adapter in (model.read, model.write):
        assert not list(adapter.parameters()) and not list(adapter.children())
    assert list(model.state_dict()) == ["owner." + name for name in model.owner.state_dict()]
    _step(model, value).square().sum().backward()
    assert all(parameter.grad is None for parameter in old_owner.parameters())
    assert all(parameter.grad is not None for parameter in model.owner.parameters())


@torch.compiler.set_stance("force_eager")
def test_adapter_preserves_owner_selected_state_across_interleaved_reads():
    model = _model()
    selected, calls = [], []

    def select_alias(module, args, output):
        branch, h_res, h_post, residual = output
        selected.append((branch, h_res, h_post, residual.view_as(residual)))
        return selected[-1]

    model.owner.register_forward_hook(select_alias)
    for adapter in (model.read, model.write):
        adapter.register_forward_hook(lambda module, args, output: calls.append(module.operation))
    execution = HyperConnectionExecutionContext(model.owner)
    values = [
        torch.randn(
            2,
            1,
            32,
            device=next(model.parameters()).device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        for _ in range(2)
    ]
    prepared = [
        model.read(value, fp32_residual_connection=True, execution_context=execution)
        for value in values
    ]
    for value, (branch, state), chosen in zip(values, prepared, selected):
        assert branch is chosen[0]
        assert all(a is b for a, b in zip(state, (chosen[3], chosen[1], chosen[2])))
        assert state[0] is not value and state[0].data_ptr() == value.data_ptr()
        assert all(t.dtype == value.dtype for t in state)
        assert all(t.grad_fn is not None for t in state[1:])
    with patch.object(
        model.owner, "fused_h_res_h_post_bda", wraps=model.owner.fused_h_res_h_post_bda
    ) as write:
        for branch, state in reversed(prepared):
            model.write(
                branch,
                state=state,
                dropout_probability=0.0,
                training=True,
                execution_context=execution,
            )
        for call, (_, state) in zip(write.call_args_list, reversed(prepared)):
            assert all(a is b for a, b in zip(call.args[:3], (state[1], state[0], state[2])))
    assert calls == ["read", "read", "write", "write"]


@pytest.mark.parametrize("has_bias", [False, True])
@torch.compiler.set_stance("force_eager")
def test_owner_callback_receives_flat_operands_and_bound_write_options(has_bias):
    owner = _model().owner
    value = torch.randn(2, 1, 32, device=next(owner.parameters()).device, requires_grad=True)
    callback = Mock(side_effect=lambda function, *args: function(*args))
    branch, h_res, h_post, residual = owner(value, return_residual=True, checkpoint_fn=callback)
    bias = torch.randn(8, device=value.device) if has_bias else None
    options = dict(dropout_prob=0.25, training=True, fused=False)
    owner.fused_h_res_h_post_bda(
        h_res, residual, h_post, (branch, bias), checkpoint_fn=callback, **options
    )
    assert callback.call_count == 2
    assert callback.call_args_list[0].args[0] == owner.aggregate
    function, *operands = callback.call_args.args
    assert len(operands) == 5
    assert all(a is b for a, b in zip(operands, (h_res, residual, h_post, branch, bias)))
    assert function.func == owner._compute_write and function.keywords == options


@pytest.mark.parametrize("operation", ["read", "write"])
def test_owner_rejects_manager_and_callback_before_math(operation):
    owner = _model().owner
    value = torch.zeros(2, 1, 32, device=next(owner.parameters()).device)
    manager = CheckpointWithoutOutputManager()
    callback = Mock(side_effect=AssertionError("Conflicting callback invoked"))
    with (
        patch.object(owner, "compute_mappings", side_effect=AssertionError("prediction")),
        patch.object(owner, "_compute_write", side_effect=AssertionError("write")),
        pytest.raises(ValueError),
    ):
        if operation == "read":
            owner(value, mhc_recompute_manager=manager, checkpoint_fn=callback)
        else:
            owner.fused_h_res_h_post_bda(
                value,
                value,
                value,
                (value, None),
                0.0,
                True,
                False,
                manager=manager,
                checkpoint_fn=callback,
            )
    callback.assert_not_called()


@pytest.fixture
def gpu_model_parallel():
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    yield
    Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Output-discard replay needs CUDA")
@pytest.mark.parametrize("mode", ["eager", "nonterminal", "terminal"])
@pytest.mark.parametrize("dropout", [0.0, 0.25])
@torch.compiler.set_stance("force_eager")
def test_adapter_replay_matches_direct_owner(mode, dropout, gpu_model_parallel):
    actual = _model()
    reference = copy.deepcopy(actual)
    template = torch.randn(2, 1, 32, device=next(actual.parameters()).device)
    bias_template = torch.randn(8, device=template.device)
    results = []
    for model in (reference, actual):
        manager = None if mode == "eager" else CheckpointWithoutOutputManager()
        context = (
            None
            if manager is None
            else ResidualStreamRecomputeContext(manager, is_block_end=mode == "terminal")
        )
        value = template.detach().clone().requires_grad_()
        bias = bias_template.detach().clone().requires_grad_()
        owner = model.owner
        hooks = []
        for adapter in (model.read, model.write):
            adapter.register_forward_hook(
                lambda module, args, output: hooks.append(module.operation)
            )
        with (
            patch.object(owner, "compute_mappings", wraps=owner.compute_mappings) as mappings,
            patch.object(owner, "aggregate", wraps=owner.aggregate) as aggregate,
            patch.object(owner, "_compute_write", wraps=owner._compute_write) as write,
        ):
            torch.manual_seed(93)
            if model is reference:
                # Match the legacy replay schedule; eager/replay gradient accumulation may differ.
                branch, h_res, h_post, residual = owner(
                    value, return_residual=True, mhc_recompute_manager=manager
                )
                output = owner.fused_h_res_h_post_bda(
                    h_res,
                    residual,
                    h_post,
                    (branch.sin(), bias),
                    dropout,
                    True,
                    False,
                    manager=manager if mode == "nonterminal" else None,
                )
            else:
                output = _step(model, value, bias, context, dropout)
            forward_rng = _rng()
            boundary = output.square()
            replay = context is not None
            if replay:
                assert len(context.manager.checkpoints) == 2 - int(context.is_block_end)
                # Use a separate live boundary to exercise both selected replay regions.
                context.manager.discard_all_outputs_and_register_unified_recompute(boundary)
                assert bool(output.untyped_storage().nbytes()) == context.is_block_end
            boundary.sum().backward()
            # Projection has its own checkpoint; the mapping API is not replayed by the adapter.
            assert mappings.call_count == 1
            assert aggregate.call_count == 1 + int(replay)
            assert write.call_count == 1 + int(replay and not context.is_block_end)
        assert hooks == (["read", "write"] if model is actual else [])
        gradients = [value.grad, bias.grad] + [p.grad for p in model.parameters()]
        assert all(gradient is not None for gradient in gradients)
        torch.testing.assert_close(_rng(), forward_rng, rtol=0, atol=0)
        results.append((output.detach().clone(), gradients, forward_rng))
    torch.testing.assert_close(results[1], results[0], rtol=0, atol=0)
