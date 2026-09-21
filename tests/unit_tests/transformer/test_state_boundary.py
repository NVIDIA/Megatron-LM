# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Common state contracts: placement, prepared inputs, versions and packed indices."""

from dataclasses import replace

import pytest
import torch

from megatron.core.transformer.state_boundary import (
    StateBoundary,
    StateDependency,
    StatePlacement,
    StaticTensorSchema,
    TensorField,
    TensorMappingCodec,
    TensorSchema,
    prepare_boundaries,
    prepare_state_region,
    validate_metadata,
)


def _field(key, *, dtype=torch.float32, grad=True, present=True, shape=(2,)):
    return TensorField(key, shape, dtype, "tokens", grad, present)


def _dependencies():
    return (
        StateDependency(_field("feature/k:L1"), "activation", 3, 0, (5, 11), "pipeline"),
        StateDependency(
            _field("feature/ids:L1", dtype=torch.int64, grad=False),
            "activation",
            3,
            0,
            (11,),
            "pipeline",
        ),
        StateDependency(_field("feature/k:L3"), "activation", 7, 0, (9,), "pipeline"),
    )


def test_input_eligibility_separates_aliased_slots_without_detaching_the_producer():
    source = torch.ones(2, requires_grad=True)
    schema = TensorSchema(
        (
            _field("input/active"),
            _field("input/stopped", grad=False),
            _field("input/absent", present=False),
        )
    )
    active, stopped = schema.detach_ineligible((source, source))
    assert active is source and not stopped.requires_grad
    (active.square() + stopped * 7).sum().backward()
    torch.testing.assert_close(source.grad, 2 * source.detach())


def test_regions_use_actual_producers_and_leave_unread_relay_in_parent():
    dependencies = _dependencies()
    local = (StatePlacement(0, 12, 0),)
    checkpoint, capture = prepare_boundaries(
        dependencies,
        (
            StateBoundary("checkpoint:3:4", "checkpoint", 6, 10),
            StateBoundary("capture:4", "capture", 8, 10),
        ),
        local,
    )
    assert checkpoint.inputs == checkpoint.outputs == ()
    assert capture.inputs == (dependencies[2].field,)
    assert capture.outputs == ()
    placement = (StatePlacement(0, 8, 0), StatePlacement(8, 12, 1))
    wire, producer, consumer = prepare_boundaries(
        dependencies,
        (
            StateBoundary("pp:3", "pipeline", 8, 8),
            StateBoundary("checkpoint:1:2", "checkpoint", 2, 6),
            StateBoundary("checkpoint:4:5", "checkpoint", 8, 12),
        ),
        placement,
    )
    assert wire.inputs == wire.outputs == tuple(d.field for d in dependencies)
    assert producer.inputs == ()
    assert producer.outputs == tuple(d.field for d in dependencies[:2])
    assert consumer.inputs == tuple(d.field for d in dependencies)


def test_default_region_retains_unread_and_entry_prepared_edges():
    dependencies = (
        *_dependencies(),
        StateDependency(_field("lookup/local"), "prepared", 6, 0, (7, 11), "local"),
    )
    region = prepare_state_region(
        dependencies,
        StateBoundary("middle", "checkpoint", 6, 10),
        (StatePlacement(0, 12, 0),),
        TensorMappingCodec(),
    )
    assert region.schema.inputs == (dependencies[-1].field,)
    assert region.schema.outputs == ()
    assert region.retained_fields == tuple(dependencies[i].field for i in (0, 1, 3))


@pytest.mark.parametrize("origin", ["prepared", "activation"])
def test_prepared_values_are_inputs_even_when_their_prepare_position_is_at_region_entry(origin):
    dependency = StateDependency(
        _field("host.engram/lookup:L14:pp1"), origin, 28, 1, (29,), "local"
    )
    (schema,) = prepare_boundaries(
        (dependency,), (StateBoundary("L14", "checkpoint", 28, 30),), (StatePlacement(28, 32, 1),)
    )
    assert schema.inputs == ((dependency.field,) if origin == "prepared" else ())
    assert schema.outputs == ()


def test_local_prepared_inputs_do_not_enter_pipeline_wire():
    placement = (StatePlacement(0, 4, 0), StatePlacement(4, 8, 1), StatePlacement(8, 12, 2))
    local = StateDependency(
        _field("host.engram/hash:L2:pp1", dtype=torch.int64, grad=False),
        "prepared",
        4,
        1,
        (5,),
        "local",
    )
    remote = StateDependency(
        _field("csa2.decoder/global_kv:L0"), "activation", 1, 0, (5, 9), "pipeline"
    )
    schemas = prepare_boundaries(
        (local, remote),
        (
            StateBoundary("pp0", "pipeline", 4, 4),
            StateBoundary("pp1", "pipeline", 8, 8),
            StateBoundary("local", "capture", 4, 8),
        ),
        placement,
    )
    assert schemas[0].inputs == schemas[1].inputs == (remote.field,)
    assert schemas[2].inputs == (local.field, remote.field)


@pytest.mark.parametrize(
    "fault", ["rank", "local", "late", "cross_rank", "cross_chunk", "duplicate", "future"]
)
def test_invalid_state_boundaries_are_rejected_before_execution(fault):
    dependency = StateDependency(
        _field("host/hash", dtype=torch.int64, grad=False), "prepared", 0, 0, (5,), "pipeline"
    )
    placement = (StatePlacement(0, 4, 0), StatePlacement(4, 8, 1))
    boundary = StateBoundary("body", "checkpoint", 4, 8)
    if fault == "rank":
        dependency = replace(dependency, source_pp_rank=1)
    elif fault == "local":
        dependency = replace(dependency, delivery="local")
    elif fault == "late":
        dependency = replace(dependency, available_at=5, source_pp_rank=1)
    elif fault == "future":
        dependency = replace(dependency, available_at=6, source_pp_rank=1)
    elif fault in ("cross_rank", "cross_chunk"):
        boundary = replace(boundary, start=2)
        if fault == "cross_chunk":
            placement = (placement[0], replace(placement[1], pp_rank=0, chunk_id=1))
    dependencies = (dependency, dependency) if fault == "duplicate" else (dependency,)
    with pytest.raises(ValueError):
        prepare_boundaries(dependencies, (boundary,), placement)


def test_generic_pipeline_delivery_follows_logical_interleaved_placement():
    placement = (
        StatePlacement(0, 4, 0, 0),
        StatePlacement(4, 8, 1, 0),
        StatePlacement(8, 12, 0, 1),
    )
    dependencies = _dependencies()[:2]
    schemas = prepare_boundaries(
        dependencies,
        (StateBoundary("pp0", "pipeline", 4, 4), StateBoundary("pp1", "pipeline", 8, 8)),
        placement,
    )
    assert all(schema.inputs == tuple(d.field for d in dependencies) for schema in schemas)


@pytest.mark.parametrize("ranks", [(0, 1), (1, 0), (0, 0)])
def test_last_consumer_at_pipeline_cut_is_also_a_checkpoint_input(ranks):
    placement = (StatePlacement(0, 4, ranks[0], 0), StatePlacement(4, 8, ranks[1], 1))
    dependency = StateDependency(
        _field("memory/value:L0"), "activation", 1, ranks[0], (4,), "pipeline"
    )
    pipeline, checkpoint = prepare_boundaries(
        (dependency,),
        (StateBoundary("pp", "pipeline", 4, 4), StateBoundary("body", "checkpoint", 4, 8)),
        placement,
    )
    assert pipeline.inputs == checkpoint.inputs == (dependency.field,)


def test_absent_gradient_qualification_and_packed_maps_are_independent():
    schema = TensorSchema(
        (
            _field("host/hidden"),
            _field("host/optional", present=False),
            _field("host/token", dtype=torch.int64, grad=False),
            _field("glm.mtp/kv:L32:d0"),
        )
    )
    assert schema.present_spec_indices == (0, 2, 3)
    assert schema.grad_tensor_indices == (0, 2)
    codec = TensorMappingCodec()
    table = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    values = {
        "host/hidden": torch.ones(2),
        "host/optional": None,
        "host/token": torch.tensor([1, 0]),
        "glm.mtp/kv:L32:d0": table * 2,
    }
    packed = codec.export(values, schema.fields)
    restored = codec.restore(schema.fields, packed)
    assert restored is not values and restored["host/optional"] is None
    restored["glm.mtp/kv:L32:d0"].sum().backward()
    torch.testing.assert_close(table.grad, torch.full((2,), 2.0))
    with pytest.raises(ValueError, match="peer schemas"):
        schema.validate_peer(
            TensorSchema((*schema.fields[:-1], replace(schema.fields[-1], layout="different")))
        )
    assert (
        schema.fingerprint
        != TensorSchema(tuple(replace(f, key="other/" + f.key) for f in schema.fields)).fingerprint
    )


def test_metadata_cannot_hide_dynamic_tensors_or_native_payloads():
    validate_metadata((None, 3, "thd", (True, 2.0)))
    for metadata in ((torch.ones(1),), {"mask": torch.ones(1)}, [1, 2]):
        with pytest.raises(TypeError, match="immutable"):
            validate_metadata(metadata)


def test_graph_profile_includes_strides():
    tensor = torch.ones(2, 2)
    field = _field("host/hash", shape=(2, 2), grad=False)
    profile = StaticTensorSchema.from_tensors((field,), (tensor,))
    profile.validate((tensor * 3,))
    with pytest.raises(ValueError, match="strides"):
        profile.validate((tensor.t(),))


def test_three_state_codecs_share_prepared_fields_and_restore_fresh_contexts():
    from types import SimpleNamespace

    from megatron.core.transformer.state_boundary import (
        BoundarySchema,
        StateRegion,
        compose_state_regions,
    )

    prefix = _field("host/prefix:batch", dtype=torch.int32, grad=False)
    features = tuple(_field(f"{name}/value:L0") for name in ("attention", "residual", "third"))
    regions = tuple(
        (
            name,
            StateRegion(
                BoundarySchema(name, (field, prefix), (field,)),
                TensorMappingCodec(),
                retained_fields=(prefix,),
            ),
        )
        for name, field in zip(("attention", "residual", "third"), features)
    )
    region = compose_state_regions(
        "three-modules", regions, SimpleNamespace, shared_inputs=(prefix.key,)
    )
    assert len(region.schema.inputs) == 4
    x = torch.arange(1.0, 3.0, requires_grad=True)
    shared = torch.tensor([0, 2], dtype=torch.int32)
    original = SimpleNamespace(
        **{
            name: {field.key: x * (i + 1), prefix.key: shared}
            for i, (name, field) in enumerate(zip(("attention", "residual", "third"), features))
        }
    )
    tensors = region.codec.export(original, region.schema.inputs)
    restored = region.codec.restore(region.schema.inputs, tensors, region.input_metadata)
    again = region.codec.restore(region.schema.inputs, tensors, region.input_metadata)
    assert restored.third is not again.third
    assert restored.attention[prefix.key] is restored.third[prefix.key] is shared
    sum(t.square().sum() for t in region.codec.export(restored, region.schema.outputs)).backward()
    torch.testing.assert_close(x.grad, 28 * x.detach())
    restored.third.clear()
    assert again.third[features[2].key] is original.third[features[2].key]
    original.third[prefix.key] = shared.clone()
    region.codec.export(original, region.schema.inputs)
    original.third[prefix.key][0] = 1
    with pytest.raises(ValueError, match="conflicting snapshots"):
        region.codec.export(original, region.schema.inputs)
    with pytest.raises(ValueError, match="conflicting declarations"):
        compose_state_regions(
            "bad",
            (
                *regions,
                (
                    "other",
                    StateRegion(
                        BoundarySchema("other", (replace(prefix, shape=(3,)),), ()),
                        TensorMappingCodec(),
                    ),
                ),
            ),
            SimpleNamespace,
            shared_inputs=(prefix.key,),
        )


@pytest.mark.parametrize("direction", ["inputs", "outputs", "retained_fields"])
def test_composed_states_reject_duplicate_owners(direction):
    from types import SimpleNamespace

    from megatron.core.transformer.state_boundary import (
        BoundarySchema,
        StateRegion,
        compose_state_regions,
    )

    field = _field("memory/value:L0")
    region = StateRegion(
        BoundarySchema(
            "memory",
            (field,) if direction == "inputs" else (),
            (field,) if direction == "outputs" else (),
        ),
        TensorMappingCodec(),
        retained_fields=(field,) if direction == "retained_fields" else (),
    )
    with pytest.raises(ValueError, match="Duplicate state field memory/value:L0"):
        compose_state_regions("two-instances", (("a", region), ("b", region)), SimpleNamespace)


def test_explicit_shared_inputs_preserve_identity_and_joint_gradient():
    from types import SimpleNamespace

    from megatron.core.transformer.state_boundary import (
        BoundarySchema,
        StateRegion,
        compose_state_regions,
    )

    field = _field("prepared/lookup:batch")
    relay = StateRegion(BoundarySchema("relay", (field,), (field,)), TensorMappingCodec())
    combined = compose_state_regions(
        "shared", (("a", relay), ("b", relay)), SimpleNamespace, shared_inputs=(field.key,)
    )
    x, y = torch.full((2,), 2.0, requires_grad=True), torch.full((2,), 7.0, requires_grad=True)
    context = SimpleNamespace(a={field.key: x}, b={field.key: y})
    with pytest.raises(ValueError, match="must refer to the same tensor"):
        combined.codec.export(context, combined.schema.outputs)
    # Equal values and shared storage still do not establish the same autograd edge.
    for other in (x.clone(), x.view_as(x)):
        context.b[field.key] = other
        with pytest.raises(ValueError, match="must refer to the same tensor"):
            combined.codec.export(context, combined.schema.inputs)
    context.b[field.key] = x
    tensors = combined.codec.export(context, combined.schema.inputs)
    restored = combined.codec.restore(combined.schema.inputs, tensors, combined.input_metadata)
    (restored.a[field.key].sum() + restored.b[field.key].sum()).backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 2))
    assert y.grad is None
    producer = StateRegion(BoundarySchema("producer", (), (field,)), TensorMappingCodec())
    with pytest.raises(ValueError, match="cannot be produced"):
        compose_state_regions(
            "bad-shared",
            (("a", relay), ("b", producer)),
            SimpleNamespace,
            shared_inputs=(field.key,),
        )


@pytest.mark.parametrize("single_output", [False, True])
@pytest.mark.parametrize("invalid_output", [None, "missing", "extra", "shape"])
def test_graph_boundary_executes_once_and_publishes_three_independent_states(
    single_output, invalid_output
):
    from megatron.core.models.hybrid.hybrid_stack_adapter import HybridStateGraphAdapter

    class Declaration:
        def __init__(self, name):
            self.name = name

        def get_static_inputs(self, inputs):
            return {self.name + "_tensor": torch.ones_like(inputs["hidden_states"])}

        def restore_inputs(self, hidden, kwargs):
            native = [kwargs[self.name + "_tensor"]]
            return self.region(hidden), native

        def region(self, hidden):
            from megatron.core.transformer.state_boundary import BoundarySchema, StateRegion

            field = TensorField(self.name, tuple(hidden.shape), hidden.dtype, "strided", True)
            return StateRegion(BoundarySchema(self.name, (field,), (field,)), self)

        def export(self, state, fields):
            return (state[0],)

        def restore(self, fields, tensors, metadata):
            return [tensors[0]]

        def prepare_replay(self, hidden, kwargs):
            native = kwargs[self.name]
            owned = {self.name + "_tensor": native[0]}

            def publish(result, side):
                native[0] = side[0]
                return result

            return self.region(hidden), owned, publish

    names = ("kv", "mix", "third")
    adapter = HybridStateGraphAdapter(
        tuple(Declaration(name) for name in names), lambda states: dict(zip(names, states))
    )
    samples = adapter.get_static_inputs({"hidden_states": torch.ones(3, 1, 4)})
    adapter.finalize_sample_inputs((samples.pop("hidden_states"),), samples)
    x = torch.arange(1.0, 13.0).reshape(3, 1, 4).requires_grad_()
    states = {name: [x * (i + 1)] for i, name in enumerate(("kv", "mix", "third"))}
    calls = []

    def body(hidden, **native):
        calls.append(True)
        for state in native.values():
            state[0] = state[0].square()
        return hidden.sin() if single_output else (hidden.sin(),)

    def runner(hidden, **kwargs):
        outputs = adapter.capture(body, hidden, **kwargs)
        if invalid_output == "missing":
            outputs = outputs[:-1]
        elif invalid_output == "extra":
            outputs += (outputs[-1],)
        elif invalid_output == "shape":
            outputs = (outputs[0][0], *outputs[1:])
        return outputs

    if invalid_output is not None:
        with pytest.raises(ValueError, match="tensor count|incompatible shape"):
            adapter.replay(runner, x, **states)
        return
    result = adapter.replay(runner, x, **states)
    assert all(torch.equal(v[0], ((i + 1) * x).square()) for i, v in enumerate(states.values()))
    assert isinstance(result, torch.Tensor) == single_output
    hidden = result if single_output else result[0]
    assert hidden.shape == x.shape
    (hidden.sum() + sum(v[0].sum() for v in states.values())).backward()
    assert len(calls) == 1
    torch.testing.assert_close(x.grad, x.detach().cos() + 28 * x.detach())
