# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from megatron.core.transformer.forward_sharing import (
    ForwardSharingState,
    forward_sharing_lifetime,
    get_forward_sharing_state,
    is_forward_sharing_enabled,
    preserve_forward_sharing_for_checkpoint,
)


@dataclass
class _FirstPayload:
    value: int = 1


@dataclass
class _SecondPayload:
    value: int = 2


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({}, False),
        ({"dsa_indexer_topk_freq": None}, False),
        ({"dsa_indexer_topk_freq": 0}, False),
        ({"dsa_indexer_topk_freq": 1}, False),
        ({"dsa_indexer_topk_freq": 2}, True),
        ({"dsa_indexer_topk_freq": 4}, True),
        ({"mtp_repeated_layer_shared_components": None}, False),
        ({"mtp_repeated_layer_shared_components": []}, False),
        ({"mtp_repeated_layer_shared_components": ["sparse_attention_index"]}, True),
        ({"mtp_repeated_layer_shared_components": ["latent_kv"]}, True),
        ({"mtp_repeated_layer_shared_components": ["latent_kv", "sparse_attention_index"]}, True),
        ({"dsa_indexer_topk_freq": 1, "mtp_repeated_layer_shared_components": ["latent_kv"]}, True),
    ],
)
def test_forward_sharing_enablement(fields, expected):
    assert is_forward_sharing_enabled(SimpleNamespace(**fields)) is expected


def test_carrier_priority_and_reuse():
    packed_seq_params = SimpleNamespace()
    attention_mask = SimpleNamespace()
    config = SimpleNamespace()

    packed_state = get_forward_sharing_state(packed_seq_params, attention_mask, config)
    assert packed_state is get_forward_sharing_state(packed_seq_params, attention_mask, config)
    assert packed_state is packed_seq_params._forward_sharing_state
    assert not hasattr(attention_mask, "_forward_sharing_state")
    assert not hasattr(config, "_forward_sharing_state")

    mask_state = get_forward_sharing_state(None, attention_mask, config)
    assert mask_state is attention_mask._forward_sharing_state
    assert mask_state is not packed_state
    assert not hasattr(config, "_forward_sharing_state")

    config_state = get_forward_sharing_state(config=config)
    assert config_state is config._forward_sharing_state
    assert config_state is not mask_state


def test_carrier_validation():
    with pytest.raises(ValueError, match="forward sharing requires"):
        get_forward_sharing_state()

    carrier = SimpleNamespace(_forward_sharing_state=object())
    with pytest.raises(TypeError, match="must contain a ForwardSharingState"):
        get_forward_sharing_state(config=carrier)


def test_global_typed_registry():
    state = ForwardSharingState()

    assert state.get(_FirstPayload) is None
    first = state.get_or_create(_FirstPayload)
    second = state.get_or_create(_SecondPayload)
    assert first is state.get_or_create(_FirstPayload)
    assert second is state.get(_SecondPayload)

    state.clear(_FirstPayload)
    assert state.get(_FirstPayload) is None
    assert state.get(_SecondPayload) is second

    state.clear()
    assert state.get(_SecondPayload) is None
    with pytest.raises(TypeError, match="payload_type must be a type"):
        state.get("not-a-type")


def test_snapshot_copies_payloads_without_changing_type_keys():
    state = ForwardSharingState()
    first = state.get_or_create(_FirstPayload)
    second = state.get_or_create(_SecondPayload)
    snapshot = state.snapshot()
    assert set(state._entries) == {_FirstPayload, _SecondPayload}
    assert snapshot.get(_FirstPayload) is not first
    assert snapshot.get(_SecondPayload) is not second
    first.value = 9
    state.clear(_FirstPayload)
    assert snapshot.get(_FirstPayload).value == 1
    assert state.get(_SecondPayload) is second


def test_generic_helpers_treat_config_as_an_opaque_carrier():
    class Carrier:
        def __getattr__(self, name):
            if name == "_forward_sharing_state":
                raise AttributeError(name)
            raise AssertionError(f"Generic sharing code inspected config field {name}")

    carrier = Carrier()

    def consumer(attention_mask):
        state = get_forward_sharing_state(attention_mask=attention_mask, config=carrier)
        return state.get(_FirstPayload).value

    wrapped = preserve_forward_sharing_for_checkpoint(consumer, None, carrier, attention_mask_arg=0)
    with forward_sharing_lifetime(config=carrier) as state:
        state.get_or_create(_FirstPayload).value = 7
        assert wrapped(None) == 7
    assert state._entries == {}
    assert wrapped(None) == 7
    assert state._entries == {}


@pytest.mark.parametrize("carrier_kind", ["mask", "packed", "config"])
@pytest.mark.parametrize("fail", [False, True])
def test_forward_lifetime_clears_reused_carriers_and_exceptions(carrier_kind, fail):
    config = SimpleNamespace()
    packed = SimpleNamespace() if carrier_kind == "packed" else None
    mask = SimpleNamespace() if carrier_kind == "mask" else None
    state = get_forward_sharing_state(packed, mask, config)
    state.get_or_create(_FirstPayload).value = -1
    for _ in range(2):
        try:
            with forward_sharing_lifetime(packed, mask, config) as current:
                assert current is state
                assert state.get(_FirstPayload) is None
                payload = state.get_or_create(_FirstPayload)
                with forward_sharing_lifetime(packed, mask, config):
                    assert state.get(_FirstPayload) is payload
                assert state.get(_FirstPayload) is payload
                if fail:
                    raise RuntimeError("forward failed")
        except RuntimeError as error:
            assert fail and str(error) == "forward failed"
        assert not state.active
        assert state._entries == {}


def test_model_boundary_preserves_decoder_payload_until_mtp():
    class Model:
        config = SimpleNamespace(dsa_indexer_topk_freq=4)

        def decoder(self, attention_mask, packed_seq_params=None):
            with forward_sharing_lifetime(packed_seq_params, attention_mask, self.config):
                state = get_forward_sharing_state(packed_seq_params, attention_mask, self.config)
                state.get_or_create(_FirstPayload).value = 42

        def forward(self, attention_mask, packed_seq_params=None):
            with forward_sharing_lifetime(packed_seq_params, attention_mask, self.config):
                self.decoder(attention_mask, packed_seq_params)
                state = get_forward_sharing_state(packed_seq_params, attention_mask, self.config)
                assert state.get(_FirstPayload).value == 42

    model = Model()
    mask = SimpleNamespace()
    model.forward(mask)
    assert get_forward_sharing_state(attention_mask=mask)._entries == {}


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("recompute", [None, "full"])
def test_block_lifetime_wraps_layer_execution(monkeypatch, nested, fail, recompute):
    from contextlib import nullcontext

    from megatron.core.transformer import transformer_block

    monkeypatch.setattr(transformer_block, "get_transformer_layer_offset", lambda *_: 0)
    monkeypatch.setattr(transformer_block, "get_pg_rank", lambda *_: 0)
    config = SimpleNamespace(
        dsa_indexer_topk_freq=4,
        sequence_parallel=False,
        fp8=None,
        fp4=None,
        enable_hyper_connections=False,
        recompute_granularity=recompute,
        cpu_offloading=False,
    )
    state = get_forward_sharing_state(config=config)
    events = []

    def preprocess(hidden):
        assert state.active is nested
        events.append("preprocess")
        return hidden

    def layer(hidden_states, **_kwargs):
        assert state.active
        events.append("layer")
        state.get_or_create(_FirstPayload).value = 42
        if fail:
            raise RuntimeError("layer failed")
        return hidden_states, None

    def postprocess(hidden, **_kwargs):
        assert state.active is nested
        assert (state.get(_FirstPayload) is not None) is nested
        events.append("postprocess")
        return hidden, None

    block = SimpleNamespace(
        config=config,
        pg_collection=SimpleNamespace(pp=None),
        vp_stage=None,
        training=True,
        layers=[layer],
        offload_context=nullcontext(),
        preprocess_for_layer_schedule=preprocess,
        postprocess_for_layer_schedule=postprocess,
        _build_mhc_recompute_layer_plan=lambda _: ([None], [False]),
        _finalize_mhc_recompute_layer=lambda **_: None,
        _checkpointed_forward=lambda **kwargs: layer(**kwargs)[0],
    )
    outer_lifetime = forward_sharing_lifetime(config=config) if nested else nullcontext()
    with outer_lifetime:
        if fail:
            with pytest.raises(RuntimeError, match="layer failed"):
                transformer_block.TransformerBlock.forward(block, 7, None)
        else:
            assert transformer_block.TransformerBlock.forward(block, 7, None) == 7
        assert state.active is nested
        if nested:
            assert state.get(_FirstPayload).value == 42
        else:
            assert state._entries == {}
    assert not state.active
    assert state._entries == {}
    assert events == (["preprocess", "layer"] if fail else ["preprocess", "layer", "postprocess"])


@pytest.mark.parametrize("carrier_kind", ["mask", "packed", "config"])
@pytest.mark.parametrize("boundary", ["helper", "core_attn"])
def test_checkpoint_pins_indices_across_cleanup_and_interleaved_forwards(
    monkeypatch, carrier_kind, boundary
):
    import torch

    from megatron.core import tensor_parallel
    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
        AbsorbedMLASelfAttention,
    )
    from megatron.core.transformer.experimental_attention_variant.dsa import _DSAIndexSharingPayload

    monkeypatch.setattr(
        tensor_parallel.random, "_get_all_rng_states", lambda: (torch.get_rng_state(),)
    )
    monkeypatch.setattr(tensor_parallel.random, "_set_all_rng_states", torch.set_rng_state)
    config = SimpleNamespace(dsa_indexer_topk_freq=4)
    packed = SimpleNamespace() if carrier_kind == "packed" else None
    mask = torch.zeros(1) if carrier_kind == "mask" else None
    state = get_forward_sharing_state(packed, mask, config)
    calls = []

    def consumer(hidden, attention_mask):
        current = get_forward_sharing_state(packed, attention_mask, config)
        indices = current.get(_DSAIndexSharingPayload).topk_by_layer[1]
        calls.append(int(indices.item()))
        return hidden.index_select(0, indices).square()

    def core_attention(_query, _key, *, x, attention_mask, **_kwargs):
        return consumer(x, attention_mask)

    outputs, inputs = [], []
    for source_index in [0, 1]:
        with forward_sharing_lifetime(packed, mask, config):
            payload = state.get_or_create(_DSAIndexSharingPayload)
            payload.topk_by_layer[1] = torch.tensor([source_index])
            hidden = torch.tensor([2.0, 3.0], requires_grad=True)
            if boundary == "helper":
                wrapped = preserve_forward_sharing_for_checkpoint(
                    consumer, packed, config, attention_mask_arg=1
                )
                output = tensor_parallel.checkpoint(wrapped, False, hidden, mask)
            else:
                attention = SimpleNamespace(
                    config=config, core_attention=core_attention, attn_mask_type=AttnMaskType.causal
                )
                output = AbsorbedMLASelfAttention._checkpointed_attention_forward(
                    attention,
                    hidden.detach(),
                    hidden.detach(),
                    hidden,
                    hidden.detach(),
                    mask,
                    hidden.detach(),
                    packed_seq_params=packed,
                )
            outputs.append(output)
            inputs.append(hidden)
            # Updating the live dictionary must not change this checkpoint's inputs.
            payload.topk_by_layer[1] = torch.tensor([1 - source_index])
        assert state._entries == {}
    outputs[1].sum().backward()
    outputs[0].sum().backward()
    assert calls == [0, 1, 1, 0]
    torch.testing.assert_close(inputs[0].grad, torch.tensor([4.0, 0.0]))
    torch.testing.assert_close(inputs[1].grad, torch.tensor([0.0, 6.0]))
    assert state._entries == {}


@pytest.mark.parametrize("carrier_kind", ["mask", "packed", "config"])
@pytest.mark.parametrize(
    ("method", "chunk_size"), [("uniform", 1), ("uniform", 2), ("block", 1), ("block", 2)]
)
def test_block_checkpoint_preserves_cross_chunk_indices(
    monkeypatch, carrier_kind, method, chunk_size
):
    import torch

    from megatron.core import tensor_parallel
    from megatron.core.transformer.experimental_attention_variant.dsa import _DSAIndexSharingPayload
    from megatron.core.transformer.transformer_block import TransformerBlock

    monkeypatch.setattr(
        tensor_parallel.random, "_get_all_rng_states", lambda: (torch.get_rng_state(),)
    )
    monkeypatch.setattr(tensor_parallel.random, "_set_all_rng_states", torch.set_rng_state)
    config = SimpleNamespace(
        dsa_indexer_topk_freq=4,
        recompute_method=method,
        recompute_num_layers=chunk_size,
        distribute_saved_activations=False,
        fp8=None,
        fp4=None,
    )
    packed = SimpleNamespace() if carrier_kind == "packed" else None
    mask = torch.zeros(1) if carrier_kind == "mask" else None
    state = get_forward_sharing_state(packed, mask, config)

    def producer(hidden_states, attention_mask, context, **kwargs):
        current = get_forward_sharing_state(kwargs["packed_seq_params"], attention_mask, config)
        payload = current.get_or_create(_DSAIndexSharingPayload)
        payload.topk_by_layer[1] = torch.tensor([int(hidden_states[0].item()) % 2])
        return hidden_states * 2, context

    def consumer(hidden_states, attention_mask, context, **kwargs):
        current = get_forward_sharing_state(kwargs["packed_seq_params"], attention_mask, config)
        indices = current.get(_DSAIndexSharingPayload).topk_by_layer[1]
        return hidden_states.index_select(0, indices).square(), context

    block = SimpleNamespace(
        config=config,
        num_layers_per_pipeline_rank=2,
        _get_layer=lambda index: [producer, consumer][index],
    )
    inputs, outputs = [], []
    for values in ([2.0, 3.0], [3.0, 4.0]):
        hidden = torch.tensor(values, requires_grad=True)
        with forward_sharing_lifetime(packed, mask, config):
            output = TransformerBlock._checkpointed_forward(
                block,
                hidden,
                attention_mask=mask,
                context=None,
                context_mask=None,
                rotary_pos_emb=None,
                attention_bias=None,
                packed_seq_params=packed,
                use_inner_quantization_context=False,
            )
        assert state._entries == {}
        inputs.append(hidden)
        outputs.append(output)
    outputs[1].sum().backward()
    outputs[0].sum().backward()
    torch.testing.assert_close(inputs[0].grad, torch.tensor([16.0, 0.0]))
    torch.testing.assert_close(inputs[1].grad, torch.tensor([0.0, 32.0]))
    assert state._entries == {}


def test_overlap_nodes_bind_plan_owned_payloads_and_release_them():
    from megatron.core.models.common.model_chunk_schedule_plan import (
        TransformerModelChunkSchedulePlan,
    )
    from megatron.core.models.gpt.fine_grained_callables import TransformerLayerNode

    config = SimpleNamespace(dsa_indexer_topk_freq=4)
    chunks = [
        SimpleNamespace(
            forward_sharing_state=ForwardSharingState(),
            packed_seq_params=None,
            attention_mask=None,
            model=SimpleNamespace(config=config),
        )
        for _ in range(2)
    ]

    def component(node, value=None):
        state = get_forward_sharing_state(config=node.chunk_state.model.config)
        if value is not None:
            state.get_or_create(_FirstPayload).value = value
        return state.get(_FirstPayload).value

    nodes = [SimpleNamespace(chunk_state=chunk, submodule=component) for chunk in chunks]
    assert TransformerLayerNode.forward_impl(nodes[0], 11) == 11
    assert TransformerLayerNode.forward_impl(nodes[1], 22) == 22
    assert TransformerLayerNode.forward_impl(nodes[0]) == 11
    assert not hasattr(config, "_forward_sharing_state")
    plan = SimpleNamespace(
        _model_chunk_state=chunks[0],
        _recompute_segments=[],
        pre_process=SimpleNamespace(),
        post_process=None,
    )
    first_state = chunks[0].forward_sharing_state
    TransformerModelChunkSchedulePlan.release_state(plan)
    assert first_state._entries == {}
    assert TransformerLayerNode.forward_impl(nodes[1]) == 22

    def fail(_node):
        raise RuntimeError("failed schedule node")

    nodes[1].submodule = fail
    with pytest.raises(RuntimeError, match="failed schedule node"):
        TransformerLayerNode.forward_impl(nodes[1])
    assert chunks[1].forward_sharing_state._entries == {}
    assert not hasattr(config, "_forward_sharing_state")
