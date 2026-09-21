# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Real reentrant checkpoint execution with explicit side tensors and active roots."""

import sys
from types import ModuleType

import pytest
import torch
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

from megatron.core.extensions import transformer_engine as te_runtime
from megatron.core.tensor_parallel import random as runtime
from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexerLossAutoScaler
from megatron.core.transformer.state_boundary import (
    CheckpointBoundaryPolicy,
    TensorField,
    TensorSchema,
)


@pytest.fixture(autouse=True)
def cpu_rng(monkeypatch):
    if not torch.cuda.is_available():
        monkeypatch.setattr(runtime, "_get_cuda_rng_state", lambda **kwargs: None)
        monkeypatch.setattr(runtime, "_set_cuda_rng_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(DSAIndexerLossAutoScaler, "main_loss_backward_scale", None)


@pytest.fixture
def cpu_te_checkpoint(monkeypatch):
    """Exercise the production TE bridge with PyTorch's two CPU checkpoint backends."""
    # Finish optional-TE imports before overriding availability on the bridge.
    from megatron.core.models.gpt import experimental_attention_variant_module_specs  # noqa: F401

    calls = []
    backend = ModuleType("transformer_engine.pytorch.distributed")

    def te_checkpoint(function, *args, use_reentrant=True, **kwargs):
        calls.append(use_reentrant)
        # TE's non-reentrant implementation always recomputes the complete region.
        with set_checkpoint_early_stop(False):
            return checkpoint(function, *args, use_reentrant=use_reentrant)

    backend.checkpoint = te_checkpoint
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    monkeypatch.setattr(te_runtime, "HAVE_TE", True)
    monkeypatch.setattr(te_runtime, "is_te_min_version", lambda version: True)
    return calls


def _policy(*fields, strict=False):
    return CheckpointBoundaryPolicy(TensorSchema(tuple(fields)), strict=strict)


def _field(key, shape=(3,), dtype=torch.float32, differentiable=True, present=True):
    return TensorField(key, shape, dtype, "strided", differentiable, present)


def test_te_boundary_honors_output_qualification_and_default_backend(cpu_te_checkpoint):
    x = torch.ones(3, requires_grad=True)
    policy = _policy(_field("hidden"), _field("stopped", differentiable=False))
    hidden, stopped = te_runtime.te_checkpoint(
        lambda x: (x.square(), x.sin()), False, None, None, x, boundary_policy=policy
    )
    assert hidden.requires_grad and not stopped.requires_grad
    hidden.sum().backward()
    torch.testing.assert_close(x.grad, 2 * x.detach())
    assert cpu_te_checkpoint == [True]
    te_runtime.te_checkpoint(lambda x: x.square(), False, None, None, x).sum().backward()
    assert cpu_te_checkpoint == [True, True]


def test_te_boundary_rejects_incompatible_backend_and_wrong_schema(monkeypatch, cpu_te_checkpoint):
    x = torch.ones(3, requires_grad=True)
    policy = _policy(_field("hidden"))
    for options in ({"use_reentrant": False}, {"distribute_saved_activations": True}):
        distribute = options.pop("distribute_saved_activations", False)
        with pytest.raises(ValueError, match="reentrant"):
            te_runtime.te_checkpoint(
                lambda x: x, distribute, None, None, x, boundary_policy=policy, **options
            )
    with pytest.raises(ValueError, match="shape/dtype"):
        te_runtime.te_checkpoint(lambda x: x[:1], False, None, None, x, boundary_policy=policy)
    monkeypatch.setattr(te_runtime, "is_te_min_version", lambda version: False)
    with pytest.raises(ValueError, match=">= 1.5"):
        te_runtime.te_checkpoint(lambda x: x, False, None, None, x, boundary_policy=policy)


def test_te_boundary_frozen_inputs_no_grad_and_constant_outputs(cpu_te_checkpoint):
    x = torch.ones(3)
    weight = torch.nn.Parameter(torch.tensor(2.0))
    policy = _policy(_field("hidden"))
    te_runtime.te_checkpoint(
        lambda x: x * weight, False, None, None, x, boundary_policy=policy
    ).sum().backward()
    assert weight.grad == 3 and not cpu_te_checkpoint
    with pytest.raises(ValueError, match="differentiable input"):
        te_runtime.te_checkpoint(
            lambda x: x,
            False,
            None,
            None,
            x,
            boundary_policy=_policy(_field("hidden"), strict=True),
        )
    x.requires_grad_()
    with torch.no_grad():
        hidden = te_runtime.te_checkpoint(
            lambda x: x * weight, False, None, None, x, boundary_policy=policy
        )
    assert not hidden.requires_grad and not cpu_te_checkpoint
    constant = te_runtime.te_checkpoint(
        lambda x: torch.ones_like(x), False, None, None, x, boundary_policy=policy
    )
    constant.sum().backward()
    assert x.grad is None


def test_te_boundary_retained_backward_uses_fresh_input_gradients(cpu_te_checkpoint):
    x = torch.ones(3, requires_grad=True)
    left, right = te_runtime.te_checkpoint(
        lambda x: (x * 2, x * 3),
        False,
        None,
        None,
        x,
        boundary_policy=_policy(_field("left"), _field("right")),
    )
    left.sum().backward(retain_graph=True)
    torch.testing.assert_close(x.grad, torch.full_like(x, 2))
    right.sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 5))


@pytest.mark.parametrize("hidden_active", [False, True])
def test_unused_hidden_does_not_inject_aux_loss_but_an_explicit_zero_does(hidden_active):
    """Exercise the actual DSA autoscaler, not just gradients of a synthetic sum."""
    grads = []
    for checkpointed in (False, True):
        x = torch.arange(1.0, 4.0, requires_grad=True)
        weight = torch.nn.Parameter(torch.tensor(2.0))
        indexer = torch.nn.Parameter(torch.tensor(3.0))
        calls = []

        def region(x):
            calls.append(torch.is_grad_enabled())
            kv = x * weight
            hidden = DSAIndexerLossAutoScaler.apply(x.square(), indexer.square())
            return hidden, kv, indexer.sum()

        if checkpointed:
            outputs = runtime.checkpoint(
                region,
                False,
                x,
                boundary_policy=_policy(
                    _field("host/hidden"), _field("csa2/kv:L0"), _field("host/unused_scalar", ())
                ),
            )
        else:
            outputs = region(x)
        loss = outputs[1].sum()
        if hidden_active:
            loss = loss + outputs[0].sum() * 0
        loss.backward()
        grads.append((x.grad, weight.grad, indexer.grad))
        assert calls == ([False, True] if checkpointed else [True])
    for actual, expected in zip(grads[1], grads[0]):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected)
    assert (grads[1][2] is not None) is hidden_active


def test_prepared_lookup_and_integer_mask_are_explicit_and_replayed_without_prepare():
    table = torch.nn.Parameter(torch.arange(6.0).view(3, 2))
    ids = torch.tensor([2, 0])
    prepared = table[ids]
    masks = torch.tensor([1, 0], dtype=torch.int64)
    calls = []

    def region(lookup, mask):
        calls.append(mask.clone())
        return lookup * mask[:, None], mask + 1

    result, integer_output = runtime.checkpoint(
        region,
        False,
        prepared,
        masks,
        boundary_policy=_policy(
            _field("host/hidden", (2, 2)),
            _field("host/mask", (2,), torch.int64, False),
            _field("host/optional", present=False),
        ),
    )
    result.sum().backward()
    torch.testing.assert_close(table.grad, torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]]))
    assert len(calls) == 2 and torch.equal(calls[0], calls[1])
    assert not integer_output.requires_grad


def test_frozen_inputs_preserve_closure_parameter_grads_or_fail_strict_recompute():
    weight = torch.nn.Parameter(torch.tensor(2.0))
    x = torch.ones(3)
    calls = []

    def region(x):
        calls.append(torch.is_grad_enabled())
        return x * weight

    runtime.checkpoint(
        region, False, x, boundary_policy=_policy(_field("host/out"))
    ).sum().backward()
    assert calls == [True]
    assert weight.grad == 3
    with pytest.raises(ValueError, match="differentiable input"):
        runtime.checkpoint(
            region, False, x, boundary_policy=_policy(_field("host/out"), strict=True)
        )


@pytest.mark.parametrize("backend", ["mcore", "te"])
@pytest.mark.parametrize("execution", ["checkpoint", "frozen-input", "no-grad"])
@pytest.mark.parametrize("single_output", [False, True])
def test_output_qualification_including_eager_fallback(
    cpu_te_checkpoint, backend, execution, single_output
):
    """Frozen inputs must not reconnect an ineligible closed-over parameter."""
    x = torch.ones(3, requires_grad=execution != "frozen-input")
    trainable = torch.nn.Parameter(torch.tensor(2.0))
    stopped = torch.nn.Parameter(torch.tensor(4.0))

    def forward(x):
        side = x * stopped
        return side if single_output else (x * trainable, side)

    fields = (_field("stopped", differentiable=False),)
    if not single_output:
        fields = (_field("hidden"), _field("absent", present=False), *fields)
    with torch.set_grad_enabled(execution != "no-grad"):
        if backend == "te":
            output = te_runtime.te_checkpoint(
                forward, False, None, None, x, boundary_policy=_policy(*fields)
            )
        else:
            output = runtime.checkpoint(forward, False, x, boundary_policy=_policy(*fields))
    tensors = (output,) if single_output else output
    # Keep a legal root even when every boundary output is detached.
    (sum(t.sum() for t in tensors) + trainable * 0).backward()
    assert stopped.grad is None
    assert not tensors[-1].requires_grad
    expected = 3.0 if not single_output and execution != "no-grad" else 0.0
    torch.testing.assert_close(trainable.grad, torch.tensor(expected))
    assert cpu_te_checkpoint == ([True] if backend == "te" and execution == "checkpoint" else [])


@pytest.mark.parametrize("backend", ["eager", "mcore", "te"])
@pytest.mark.parametrize("frozen_input", [False, True])
@pytest.mark.parametrize("eligibility", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize("used", [(0,), (1,), (0, 1)])
def test_aliased_outputs_preserve_independent_slot_gradients(
    cpu_te_checkpoint, backend, frozen_input, eligibility, used
):
    x = torch.tensor([2.0, 3.0], requires_grad=not frozen_input)
    weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(x):
        y = (x * weight).square()
        return y, y

    policy = _policy(
        *(_field(str(i), (2,), differentiable=flag) for i, flag in enumerate(eligibility))
    )
    if backend == "eager":
        outputs = policy.apply(forward(x))
    elif backend == "mcore":
        outputs = runtime.checkpoint(forward, False, x, boundary_policy=policy)
    else:
        outputs = te_runtime.te_checkpoint(forward, False, None, None, x, boundary_policy=policy)
    assert tuple(t.requires_grad for t in outputs) == eligibility
    loss = sum(outputs[i].sum() for i in used)
    if loss.requires_grad:
        loss.backward()
    count = sum(eligibility[i] for i in used)
    if count:
        torch.testing.assert_close(weight.grad, 2 * x.detach().square().sum() * count)
        if not frozen_input:
            torch.testing.assert_close(x.grad, 2 * x.detach() * count)
    else:
        assert weight.grad is None and x.grad is None


@pytest.mark.parametrize("backend", ["mcore", "te"])
def test_aliased_outputs_retained_backward(cpu_te_checkpoint, backend):
    x = torch.tensor([2.0, 3.0], requires_grad=True)

    def forward(x):
        y = x.square()
        return y, y

    policy = _policy(_field("left", (2,)), _field("right", (2,)))
    if backend == "mcore":
        left, right = runtime.checkpoint(forward, False, x, boundary_policy=policy)
    else:
        left, right = te_runtime.te_checkpoint(
            forward, False, None, None, x, boundary_policy=policy
        )
    left.sum().backward(retain_graph=True)
    torch.testing.assert_close(x.grad, 2 * x.detach())
    right.sum().backward()
    torch.testing.assert_close(x.grad, 4 * x.detach())


def test_boundary_checkpoint_restores_rng_and_checkpoint_flag_on_errors():
    x = torch.ones(3, requires_grad=True)
    calls = []

    def region(x):
        calls.append(torch.rand(3))
        return x * calls[-1]

    output = runtime.checkpoint(region, False, x, boundary_policy=_policy(_field("host/out")))
    after_forward = torch.get_rng_state()
    output.sum().backward()
    assert torch.equal(calls[0], calls[1])
    assert torch.equal(after_forward, torch.get_rng_state())
    assert not runtime.is_checkpointing()

    def fail(x):
        raise RuntimeError("region failed")

    with pytest.raises(RuntimeError, match="region failed"):
        runtime.checkpoint(fail, False, x, boundary_policy=_policy(_field("host/out")))
    assert not runtime.is_checkpointing()


def test_boundary_checkpoint_rejects_unsupported_backends_and_wrong_output_schema(monkeypatch):
    x = torch.ones(3, requires_grad=True)
    policy = _policy(_field("host/out"))
    with pytest.raises(ValueError, match="distributed saved"):
        runtime.checkpoint(lambda x: x * 2, True, x, boundary_policy=policy)
    with pytest.raises(ValueError, match="shape/dtype"):
        runtime.checkpoint(lambda x: x[:1], False, x, boundary_policy=policy)
    monkeypatch.setattr("megatron.core.transformer.cuda_graphs.is_graph_capturing", lambda: True)
    with pytest.raises(ValueError, match="bridge"):
        runtime.checkpoint(lambda x: x * 2, False, x, boundary_policy=policy)


def test_original_host_arguments_include_nested_route_tensors():
    from megatron.core.context_parallel_layout import ThdCpRoute
    from megatron.core.models.hybrid.hybrid_stack_adapter import _explicit_layer_arguments
    from megatron.core.packed_seq_params import PackedSeqParams

    route = ThdCpRoute(torch.tensor([1, 0]), [1, 1], torch.tensor([0, 1]), [1, 1])
    params = PackedSeqParams(cp_partition_route=route)
    original = {"packed_seq_params": params, "mask": torch.tensor([True, False])}
    tensors, restore = _explicit_layer_arguments(original)
    replay = restore(tuple(t.clone() for t in tensors))
    assert replay["packed_seq_params"] is not params
    replay_route = replay["packed_seq_params"].cp_partition_route
    assert replay_route is not route
    assert replay_route.zigzag_index is not route.zigzag_index
    torch.testing.assert_close(replay_route.zigzag_index, route.zigzag_index)
    with pytest.raises(TypeError, match="mutable checkpoint argument"):
        _explicit_layer_arguments({"cache": object()})


def test_checkpoint_restores_third_component_without_host_feature_dispatch():
    from types import SimpleNamespace

    from megatron.core.transformer.state_boundary import (
        BoundarySchema,
        StateRegion,
        TensorMappingCodec,
        compose_state_regions,
    )

    fields = tuple(_field(f"{name}/value:L0") for name in ("kv", "mix", "third"))
    regions = tuple(
        (name, StateRegion(BoundarySchema(name, (field,), (field,)), TensorMappingCodec()))
        for name, field in zip(("kv", "mix", "third"), fields)
    )
    region = compose_state_regions("checkpoint/three-states", regions, SimpleNamespace)
    x = torch.arange(1.0, 4.0, requires_grad=True)
    context = SimpleNamespace(
        **{
            name: {field.key: (i + 1) * x}
            for i, (name, field) in enumerate(zip(("kv", "mix", "third"), fields))
        }
    )
    calls = []

    def forward(*inputs):
        working = region.codec.restore(region.schema.inputs, inputs, region.input_metadata)
        calls.append(torch.is_grad_enabled())
        for name, field in zip(("kv", "mix", "third"), fields):
            state = getattr(working, name)
            state[field.key] = state[field.key].square()
        return region.codec.export(working, region.schema.outputs)

    outputs = runtime.checkpoint(
        forward,
        False,
        *region.codec.export(context, region.schema.inputs),
        boundary_policy=_policy(*region.schema.outputs),
    )
    context.third.clear()
    restored = region.codec.restore(region.schema.outputs, outputs, region.output_metadata)
    sum(value.sum() for value in region.codec.export(restored, region.schema.outputs)).backward()
    assert calls == [False, True]
    torch.testing.assert_close(x.grad, 28 * x.detach())
