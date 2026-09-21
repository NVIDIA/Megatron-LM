# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Real reentrant checkpoint execution with explicit side tensors and active roots."""

import pytest
import torch

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


def _policy(*fields, strict=False):
    return CheckpointBoundaryPolicy(TensorSchema(tuple(fields)), strict=strict)


def _field(key, shape=(3,), dtype=torch.float32, differentiable=True, present=True):
    return TensorField(key, shape, dtype, "strided", differentiable, present)


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
