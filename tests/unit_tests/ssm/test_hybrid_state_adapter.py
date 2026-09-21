# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Independent single-pass mHC training with ordinary attention and Hybrid boundaries."""

from copy import copy
from dataclasses import replace
from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.hybrid.hybrid_block import HybridStack, HybridStackSubmodules
from megatron.core.models.hybrid.hybrid_model import HybridModel, get_hybrid_state_components
from megatron.core.models.hybrid.hybrid_stack_adapter import HybridStatePayload
from megatron.core.models.hybrid.hybrid_state import (
    HybridStateDeclaration,
    build_hybrid_state_pipeline_plan,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.pipeline_payload import (
    PipelineDataIterator,
    PipelinePayloadPlan,
    backward_pipeline_payload,
)
from megatron.core.pipeline_parallel.schedules import (
    forward_backward_pipelining_with_interleaving,
    forward_backward_pipelining_without_interleaving,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import random as checkpoint_runtime
from megatron.core.transformer.mhc_recompute import MHCRecomputeArenaSlot, MHCRecomputeSlotMetadata
from megatron.core.transformer.module import GraphableMegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.state_boundary import StatePlacement
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from tests.unit_tests.pipeline_parallel.test_typed_pipeline import (
    transport_groups as transport_groups,
)


class _Norm(nn.Module):
    def __init__(self, config, hidden_size, eps, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=config.params_dtype))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


class _Attention(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x, attention_mask=None, packed_seq_params=None, **kwargs):
        q, k, v = (t.transpose(0, 1).unsqueeze(1) for t in self.qkv(x).chunk(3, dim=-1))
        allowed = torch.ones(x.shape[0], x.shape[0], dtype=torch.bool, device=x.device).tril()
        if packed_seq_params is not None:
            boundaries = packed_seq_params.cu_seqlens_q_padded
            boundaries = packed_seq_params.cu_seqlens_q if boundaries is None else boundaries
            positions = torch.arange(x.shape[0], device=x.device)
            ids = torch.bucketize(positions, boundaries[1:], right=True)
            allowed = allowed & (ids[:, None] == ids[None, :])
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        return self.proj(output.squeeze(1).transpose(0, 1)), None


class _FFN(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x, **kwargs):
        return self.proj(x).tanh(), None


def _config(**kwargs):
    defaults = dict(
        num_layers=6,
        hidden_size=8,
        num_attention_heads=2,
        use_cpu_initialization=True,
        enable_hyper_connections=True,
        mhc_single_pass=True,
        num_residual_streams=4,
        use_fused_mhc=False,
        bias_dropout_fusion=False,
        hidden_dropout=0,
        attention_dropout=0,
    )
    defaults.update(kwargs)
    return TransformerConfig(**defaults)


def _submodules(config=None):
    return HybridStackSubmodules(
        state_components=() if config is None else get_hybrid_state_components(config),
        attention_layer=ModuleSpec(
            TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=_Norm, self_attention=_Attention, self_attn_bda=get_bias_dropout_add
            ),
        ),
        mlp_layer=ModuleSpec(
            TransformerLayer,
            submodules=TransformerLayerSubmodules(
                pre_mlp_layernorm=_Norm, mlp=_FFN, mlp_bda=get_bias_dropout_add
            ),
        ),
    )


def _stack(config, chunk=None):
    groups = ProcessGroupCollection()
    groups.tp = groups.cp = groups.pp = torch.distributed.ProcessGroup(0, 1)
    return HybridStack(
        config,
        _submodules(config),
        layer_type_list=list(
            "*-" * (config.num_layers // 2) if chunk is None else chunk.layer_pattern
        ),
        pp_layer_offset=0 if chunk is None else chunk.layer_offset,
        pre_process=chunk is None or chunk.incoming is None,
        post_process=chunk is None or chunk.outgoing is None,
        post_layer_norm=False,
        pg_collection=groups,
    )


@pytest.mark.parametrize(
    "mhc,variant,version,expected",
    [
        (False, None, "v4", ()),
        (True, None, "v4", ("mhc_state",)),
        (False, None, "v4.1", ()),
        (True, "dsv4_hybrid", "v4", ("mhc_state",)),
        (False, "dsv4_hybrid", "v4.1", ("cross_layer_state",)),
        (True, "dsv4_hybrid", "v4.1", ("mhc_state", "cross_layer_state")),
    ],
)
def test_model_state_registration_is_explicit_and_idempotent(mhc, variant, version, expected):
    from megatron.core.transformer.spec_utils import get_module

    config = SimpleNamespace(
        mhc_single_pass=mhc, experimental_attention_variant=variant, dsv4_version=version
    )
    assert HybridStackSubmodules().state_components == ()
    components = get_hybrid_state_components(config)
    assert tuple(c.context_attribute for c in components) == expected
    assert get_hybrid_state_components(config, components) == components

    # Explicit subclasses and import-path ModuleSpecs retain their configuration.
    explicit = tuple(
        ModuleSpec(
            (c.__module__, c.__name__) if i else type("CustomState", (c,), {}),
            params={"custom_option": i},
        )
        for i, c in enumerate(components)
    )
    resolved = get_hybrid_state_components(config, explicit)
    assert resolved == explicit
    assert tuple(get_module(c).context_attribute for c in resolved) == expected
    # Assembly must not hide duplicate declarations from the host's validation.
    assert get_hybrid_state_components(config, explicit + explicit) == explicit + explicit


@pytest.mark.parametrize("recompute", [False, True])
def test_model_assembles_mhc_without_mutating_shared_custom_spec(monkeypatch, recompute):
    monkeypatch.setattr(
        torch.distributed, "get_rank", lambda group=None: 0 if group is None else group.rank()
    )
    monkeypatch.setattr(
        torch.distributed, "get_world_size", lambda group=None: 1 if group is None else group.size()
    )
    monkeypatch.setattr(
        "megatron.core.models.hybrid.hybrid_layer_allocation.log_on_each_pipeline_stage",
        lambda *args, **kwargs: None,
    )
    groups = ProcessGroupCollection()
    groups.tp = groups.cp = groups.pp = groups.dp_cp = torch.distributed.ProcessGroup(0, 1)
    groups.embd = groups.pos_embd = None
    spec = ModuleSpec(
        HybridStack,
        params={"post_layer_norm": False},
        submodules=_submodules(),
        metainfo={"custom": True},
    )
    models = []
    for mhc in (False, True, False):
        config = _config(
            num_layers=2,
            enable_hyper_connections=mhc,
            mhc_single_pass=mhc,
            recompute_granularity="full" if recompute else None,
            recompute_method="uniform" if recompute else None,
            recompute_num_layers=1 if recompute else None,
        )
        model = HybridModel(
            config,
            spec,
            vocab_size=32,
            max_sequence_length=16,
            hybrid_layer_pattern="*-",
            position_embedding_type="none",
            post_process=False,
            pg_collection=groups,
        )
        assert spec.submodules.state_components == ()
        assert model.hybrid_stack_spec is not spec
        assert model.hybrid_stack_spec.params == spec.params
        assert model.hybrid_stack_spec.metainfo == spec.metainfo
        assert model.hybrid_stack_spec.submodules.attention_layer is spec.submodules.attention_layer
        adapter = model.decoder.forward_adapter
        assert (tuple(c.context_attribute for c in adapter.components) if adapter else ()) == (
            ("mhc_state",) if mhc else ()
        )
        models.append(model)

    # Reusing the spec cannot affect either an earlier or a later model's forward.
    for model in models:
        reference = HybridStack(
            model.config,
            _submodules(model.config),
            layer_type_list=["*", "-"],
            post_process=False,
            post_layer_norm=False,
            pg_collection=groups,
        )
        reference.load_state_dict(model.decoder.state_dict())
        hidden = torch.randn(3, 1, 8, requires_grad=True)
        ref_hidden = hidden.detach().clone().requires_grad_()
        output = model(None, None, None, decoder_input=hidden)
        expected = reference(ref_hidden, None)
        width = 32 if model.config.mhc_single_pass else 8
        assert output.shape == (3, 1, width)
        torch.testing.assert_close(output, expected)
        output.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(hidden.grad, ref_hidden.grad)
        for actual, ref in zip(model.decoder.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, ref.grad)


@pytest.mark.parametrize("pp_size,vp_size", [(1, 1), (2, 1), (2, 2)])
def test_model_binds_complete_placement_before_graph_queries(monkeypatch, pp_size, vp_size):
    from megatron.core.transformer.state_boundary import (
        BoundarySchema,
        StateRegion,
        TensorMappingCodec,
    )

    # No collectives run in this constructor test. Simulate the group coordinates
    # used by the real layer allocator for each physical/virtual stage.
    monkeypatch.setattr(
        torch.distributed, "get_rank", lambda group=None: 0 if group is None else group.rank()
    )
    monkeypatch.setattr(
        torch.distributed, "get_world_size", lambda group=None: 1 if group is None else group.size()
    )
    monkeypatch.setattr(
        "megatron.core.models.hybrid.hybrid_layer_allocation.log_on_each_pipeline_stage",
        lambda *args, **kwargs: None,
    )

    for stage in range(pp_size * vp_size):
        events = []

        class PlacementState(HybridStateDeclaration):
            context_attribute = "placement_state"

            def __init__(self, config, *, layer_type_list, **kwargs):
                self.local_layers = len(layer_type_list)

            def bind_placement(self, pattern, placement):
                assert events == ["layer"] * self.local_layers
                self.placement = placement
                assert len(pattern) == 4
                assert len(placement) == pp_size * vp_size
                events.append("bind")

            def graph_state(self, layer, symbol, region):
                assert events[-1] == "validate_graph"
                assert self.placement[stage].pp_rank == stage % pp_size
                assert self.placement[stage].chunk_id == stage // pp_size
                assert "attention" in region.branches
                events.append("graph")
                return None

            def validate_runtime(self, runtime):
                assert runtime.placement == self.placement
                assert runtime.graph_backend == "transformer_engine"
                assert runtime.tensor_parallel_size == runtime.context_parallel_size == 1
                assert not runtime.sequence_parallel and not runtime.is_mtp_layer
                assert not runtime.graph_output_activity
                if runtime.graph_region is None:
                    assert events[-1] == "bind"
                    events.append("validate_stack")
                else:
                    assert "attention" in runtime.graph_region.branches
                    events.append("validate_graph")

            def pipeline_region(self, boundary, hidden, params, **kwargs):
                assert self.placement
                return StateRegion(BoundarySchema("placement", (), ()), TensorMappingCodec())

        class Attention(_Attention):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                events.append("layer")

        config = _config(
            num_layers=4,
            pipeline_model_parallel_size=pp_size,
            virtual_pipeline_model_parallel_size=vp_size if vp_size > 1 else None,
            pipeline_dtype=torch.float32,
            cross_entropy_loss_fusion=False,
            enable_hyper_connections=False,
            mhc_single_pass=False,
        )
        config.cuda_graph_impl = "transformer_engine"
        submodules = _submodules()
        submodules.state_components = (PlacementState,)
        submodules.attention_layer.submodules.self_attention = Attention
        groups = ProcessGroupCollection()
        groups.tp = groups.cp = groups.dp_cp = torch.distributed.ProcessGroup(0, 1)
        groups.pp = torch.distributed.ProcessGroup(stage % pp_size, pp_size)
        groups.embd = groups.pos_embd = None
        pattern = "|".join(["*" * (4 // (pp_size * vp_size))] * (pp_size * vp_size))
        model = HybridModel(
            config,
            ModuleSpec(HybridStack, params={"post_layer_norm": False}, submodules=submodules),
            vocab_size=32,
            max_sequence_length=16,
            hybrid_layer_pattern=pattern,
            position_embedding_type="none",
            share_embeddings_and_output_weights=False,
            pre_process=stage == 0,
            post_process=stage == pp_size * vp_size - 1,
            pg_collection=groups,
            vp_stage=stage // pp_size if vp_size > 1 else None,
        )
        count = len(model.decoder.layers)
        assert (
            events
            == ["layer"] * count + ["bind", "validate_stack"] + ["validate_graph", "graph"] * count
        )
        assert all(hasattr(layer, "_te_cuda_graph_adapter") for layer in model.decoder.layers)
        # Repeating setup cannot rebind components or discard their graph signatures.
        adapter = model.decoder.forward_adapter
        adapter.configure_distributed_pipeline(pattern, groups.pp, model.vp_stage)
        adapter.configure_cuda_graphs(model.decoder.layers)
        assert events.count("bind") == 1 and events.count("graph") == count
        incoming, outgoing = model.pipeline_payload_spec(8, 1)
        assert (incoming is None) == (stage == 0)
        assert (outgoing is None) == (stage == pp_size * vp_size - 1)


@pytest.fixture(autouse=True)
def _cpu_rng(monkeypatch):
    if not torch.cuda.is_available():
        monkeypatch.setattr(checkpoint_runtime, "_get_cuda_rng_state", lambda **kwargs: None)
        monkeypatch.setattr(checkpoint_runtime, "_set_cuda_rng_state", lambda *args, **kwargs: None)
        monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda message: None)
        monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: None)


@pytest.mark.parametrize("receiving_stage", [False, True])
@pytest.mark.parametrize("recompute", [False, True])
def test_model_prepares_token_lookup_without_hash_moe(monkeypatch, receiving_stage, recompute):
    from megatron.core.models.hybrid.hybrid_stack_adapter import HybridStateAdapter
    from megatron.core.transformer.state_boundary import (
        BoundarySchema,
        StateRegion,
        TensorField,
        TensorMappingCodec,
    )

    monkeypatch.setattr(
        torch.distributed, "get_rank", lambda group=None: 0 if group is None else group.rank()
    )
    monkeypatch.setattr(
        torch.distributed, "get_world_size", lambda group=None: 1 if group is None else group.size()
    )
    monkeypatch.setattr(
        "megatron.core.models.hybrid.hybrid_layer_allocation.log_on_each_pipeline_stage",
        lambda *args, **kwargs: None,
    )
    calls = []
    lookup = nn.Embedding(13, 8)
    key = "lookup/tokens:batch"

    class LocalLookup(HybridStateDeclaration):
        context_attribute = "lookup_owner"

        def __init__(self, *args, **kwargs):
            pass

        def initial_state(self, hidden, params):
            return {}

        def prepare_local_state(self, hidden, state, inputs):
            calls.append(inputs["input_ids"])
            assert inputs["input_ids"] is not None
            return {key: lookup(inputs["input_ids"]).transpose(0, 1).contiguous()}

        def layer_kwargs(self, layer, state):
            return {"layer": {"token_lookup": state[key]}}

        def recompute_boundary_tensors(self, state):
            return tuple(state.values())

        def pipeline_region(self, *args, **kwargs):
            return StateRegion(BoundarySchema("lookup/pp", (), ()), TensorMappingCodec())

        def checkpoint_region(self, start, end, hidden, state):
            field = TensorField(key, tuple(hidden.shape), hidden.dtype, "sbhd", True)
            return StateRegion(
                BoundarySchema("lookup/checkpoint", (field,), (field,)), TensorMappingCodec()
            )

    class Consumer(nn.Module):
        def __init__(self, config, layer_number, **kwargs):
            super().__init__()
            self.layer_number = layer_number
            self.lookup = lookup  # The model owns parameters, the declaration owns forward state.

        def forward(self, hidden_states, *, token_lookup, **kwargs):
            return hidden_states + token_lookup

    pp_size = 3 if receiving_stage else 1
    config = _config(
        num_layers=3,
        enable_hyper_connections=False,
        mhc_single_pass=False,
        moe_n_hash_layers=0,
        pipeline_model_parallel_size=pp_size,
        pipeline_dtype=torch.float32,
        cross_entropy_loss_fusion=False,
        recompute_granularity="full" if recompute else None,
        recompute_method="uniform" if recompute else None,
        recompute_num_layers=1 if recompute else None,
    )
    groups = ProcessGroupCollection()
    groups.tp = groups.cp = groups.dp_cp = torch.distributed.ProcessGroup(0, 1)
    groups.pp = torch.distributed.ProcessGroup(1 if receiving_stage else 0, pp_size)
    groups.embd = groups.pos_embd = None
    pattern = "M|M|M" if receiving_stage else "MMM"
    model = HybridModel(
        config,
        ModuleSpec(
            HybridStack,
            params={"post_layer_norm": False},
            submodules=HybridStackSubmodules(mamba_layer=Consumer, state_components=(LocalLookup,)),
        ),
        vocab_size=32,
        max_sequence_length=16,
        hybrid_layer_pattern=pattern,
        position_embedding_type="none",
        share_embeddings_and_output_weights=False,
        pre_process=not receiving_stage,
        post_process=False,
        pg_collection=groups,
    )
    sender = None
    if receiving_stage:
        chunk = build_hybrid_state_pipeline_plan(config, pattern, pp_size=3)[0]
        sender = HybridStateAdapter(
            config,
            components=(),
            hidden_size=8,
            hidden_dtype=torch.float32,
            layer_type_list=list(chunk.layer_pattern),
            pp_layer_offset=0,
            pre_process=True,
            post_process=False,
            is_mtp_layer=False,
            pg_collection=groups,
        )
        sender.configure_pipeline(chunk)

    for ids in (torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5, 6]])):
        hidden = torch.randn(3, 1, 8, requires_grad=True)
        if sender is not None:
            _, _, context = sender.prepare_forward(hidden, None, None)
            model.set_input_tensor(sender.finalize_forward(hidden, None, context))
        output = model(
            input_ids=ids,
            position_ids=None,
            attention_mask=None,
            decoder_input=None if receiving_stage else hidden,
        )
        if isinstance(output, HybridStatePayload):
            output = output.tensors[0]
        count = len(model.decoder.layers)
        torch.testing.assert_close(output, hidden + count * lookup(ids).transpose(0, 1))
        output.sum().backward()
        torch.testing.assert_close(
            lookup.weight.grad[ids.flatten()], torch.full((3, 8), float(count))
        )
        torch.testing.assert_close(hidden.grad, torch.ones_like(hidden))
        assert calls[-1] is ids
        model.zero_grad()
    assert len(calls) == 2  # Preparation runs once per forward, never during recompute.


def test_default_tensor_graph_state_supports_independent_instances():
    from megatron.core.models.hybrid.hybrid_stack_adapter import HybridStateGraphAdapter
    from megatron.core.models.hybrid.hybrid_state import HybridTensorGraphState
    from megatron.core.transformer.state_boundary import (
        BoundarySchema,
        StateRegion,
        TensorField,
        TensorMappingCodec,
    )

    def region(namespace, hidden):
        def field(name):
            return TensorField(
                f"{namespace}/{name}:L0", tuple(hidden.shape), hidden.dtype, "sbhd", True
            )

        return StateRegion(
            BoundarySchema(namespace, (field("input"),), (field("output"),)), TensorMappingCodec()
        )

    components = tuple(
        HybridTensorGraphState(
            lambda hidden, context, ns=ns: region(ns, hidden),
            lambda context, ns=ns: context["states"][ns],
        )
        for ns in ("first", "second")
    )
    adapter = HybridStateGraphAdapter(
        components, lambda states: dict(zip(("first", "second"), states))
    )
    sample = torch.ones(3, 1, 4, requires_grad=True)
    inputs = adapter.get_static_inputs({"hidden_states": sample})
    adapter.finalize_sample_inputs((inputs.pop("hidden_states"),), inputs)

    def capture(hidden, first, second):
        first["first/output:L0"] = hidden * first["first/input:L0"]
        second["second/output:L0"] = hidden * second["second/input:L0"]
        return hidden.square()

    outstanding = []
    for _ in range(2):
        hidden = torch.randn_like(sample, requires_grad=True)
        x = torch.full_like(sample, 2.0, requires_grad=True)
        y = torch.full_like(sample, 7.0, requires_grad=True)
        states = {"first": {"first/input:L0": x}, "second": {"second/input:L0": y}}
        output = adapter.replay(
            lambda *args, **kwargs: adapter.capture(capture, *args, **kwargs), hidden, states=states
        )
        assert states["first"]["first/input:L0"] is x
        torch.testing.assert_close(states["first"]["first/output:L0"], hidden * 2)
        torch.testing.assert_close(states["second"]["second/output:L0"], hidden * 7)
        loss = (
            output.sum()
            + states["first"]["first/output:L0"].sum()
            + states["second"]["second/output:L0"].sum()
        )
        outstanding.append((hidden, x, y, loss))
    for hidden, x, y, loss in reversed(outstanding):
        loss.backward()
        torch.testing.assert_close(x.grad, hidden.detach())
        torch.testing.assert_close(y.grad, hidden.detach())
        torch.testing.assert_close(hidden.grad, 2 * hidden.detach() + 9)


@pytest.mark.parametrize("metadata_side", ["input", "output"])
def test_default_tensor_graph_state_rejects_changed_codec_metadata(metadata_side):
    from megatron.core.models.hybrid.hybrid_stack_adapter import HybridStateGraphAdapter
    from megatron.core.models.hybrid.hybrid_state import HybridTensorGraphState
    from megatron.core.transformer.state_boundary import (
        BoundarySchema,
        StateRegion,
        TensorField,
        TensorMappingCodec,
    )

    metadata = {"input": (2,), "output": (3,)}

    class Codec(TensorMappingCodec):
        def restore(self, fields, tensors, metadata):
            return dict(super().restore(fields, tensors, metadata), scale=metadata[0])

    def region(hidden, context):
        field = TensorField("memory/value:L0", tuple(hidden.shape), hidden.dtype, "sbhd", True)
        return StateRegion(
            BoundarySchema("memory", (field,), (field,)),
            Codec(),
            metadata["input"],
            metadata["output"],
        )

    adapter = HybridStateGraphAdapter(
        (HybridTensorGraphState(region, lambda context: context["state"]),),
        lambda states: {"state": states[0]},
    )
    hidden = torch.ones(3, 1, 4, requires_grad=True)
    inputs = adapter.get_static_inputs({"hidden_states": hidden})
    adapter.finalize_sample_inputs((inputs.pop("hidden_states"),), inputs)
    captured = adapter.capture(lambda hidden, state: hidden * state["scale"], hidden, **inputs)
    calls = []

    def graph(*args, **kwargs):
        calls.append(True)
        return captured

    state = {"memory/value:L0": torch.ones_like(hidden, requires_grad=True)}
    adapter.replay(graph, hidden, state=state)
    assert calls == [True]
    metadata[metadata_side] = (7,)
    with pytest.raises(ValueError, match="metadata|contract"):
        adapter.replay(graph, hidden, state=state)
    assert calls == [True]  # Reject before executing stale captured computation.


@pytest.mark.parametrize("shared_key", ["host/cu_seqlens:batch", "model/lookup:batch"])
@pytest.mark.parametrize("differentiable", [False, True])
@pytest.mark.parametrize("relay", [False, True])
def test_default_graph_shared_inputs_follow_boundary_rules(shared_key, differentiable, relay):
    from megatron.core.models.hybrid.hybrid_stack_adapter import HybridStateGraphAdapter
    from megatron.core.models.hybrid.hybrid_state import HybridTensorGraphState
    from megatron.core.transformer.state_boundary import (
        BoundarySchema,
        StateRegion,
        TensorField,
        TensorMappingCodec,
    )

    def region(namespace, hidden):
        shared = TensorField(shared_key, tuple(hidden.shape), hidden.dtype, "sbhd", differentiable)
        output = TensorField(
            namespace + "/value:L0", tuple(hidden.shape), hidden.dtype, "sbhd", True
        )
        outputs = (shared, output) if relay else (output,)
        return StateRegion(BoundarySchema(namespace, (shared,), outputs), TensorMappingCodec())

    components = tuple(
        HybridTensorGraphState(
            lambda hidden, context, name=name: region(name, hidden),
            lambda context, name=name: context["states"][name],
        )
        for name in ("first", "second")
    )
    adapter = HybridStateGraphAdapter(
        components,
        lambda states: dict(zip(("first", "second"), states)),
        shared_inputs=(shared_key,),
    )
    hidden = torch.ones(3, 1, 4, requires_grad=True)
    inputs = adapter.get_static_inputs({"hidden_states": hidden})
    assert tuple(inputs) == ("hidden_states", shared_key)
    adapter.finalize_sample_inputs((inputs.pop("hidden_states"),), inputs)

    def capture(hidden, first, second):
        first["first/value:L0"] = hidden * first[shared_key]
        second["second/value:L0"] = hidden * second[shared_key] * 2
        return hidden.square()

    captured = adapter.capture(capture, hidden, **inputs)
    assert len(captured) == 3 + int(relay)
    value = torch.full_like(hidden, 3.0, requires_grad=True)
    states = {"first": {shared_key: value}, "second": {shared_key: value}}
    calls = []

    def graph(*args, **kwargs):
        calls.append(True)
        return adapter.capture(capture, *args, **kwargs)

    output = adapter.replay(graph, hidden, states=states)
    (output.sum() + sum(states[name][name + "/value:L0"].sum() for name in states)).backward()
    torch.testing.assert_close(hidden.grad, 2 * hidden.detach() + 9)
    if differentiable:
        torch.testing.assert_close(value.grad, 3 * hidden.detach())
    else:
        assert value.grad is None
    # A second producer cannot hide behind the shared-input declaration.
    states["second"][shared_key] = value.clone() if differentiable else value + 1
    with pytest.raises(ValueError, match="same tensor|conflicting snapshots"):
        adapter.replay(graph, hidden, states=states)
    assert len(calls) == 1


@pytest.mark.parametrize("mode", ["eager", "full", "graph"])
@pytest.mark.parametrize("declared", [False, True])
def test_model_spec_shares_prepared_inputs_across_boundaries(mode, declared):
    """Two native consumers share a model-defined input through the real Hybrid loop."""
    from megatron.core.models.hybrid.hybrid_state import HybridTensorState
    from megatron.core.transformer.state_boundary import StateDependency, TensorField

    key = "model/lookup:batch"

    class Declaration(HybridTensorState):
        def __init__(self, config, name, **kwargs):
            super().__init__(config, **kwargs)
            self.context_attribute = self.name = name

        def initial_state(self, hidden, params):
            return {key: hidden}

        def state_bindings(self, layer):
            return {"attention": self.name + "_state"}

        def dependencies(self, hidden):
            return (
                StateDependency(
                    TensorField(key, tuple(hidden.shape), hidden.dtype, "sbhd", True),
                    "prepared",
                    0,
                    self.placement[0].pp_rank,
                    tuple(2 * i + 1 for i in range(len(self.pattern))),
                    "pipeline",
                ),
            )

    class Attention(_Attention):
        def forward(self, hidden, *, first_state, second_state, **kwargs):
            return super().forward(
                hidden + first_state[key] * 0.1 + second_state[key] * 0.2, **kwargs
            )

    torch.manual_seed(732)
    config = _config(num_layers=2, enable_hyper_connections=False, mhc_single_pass=False)
    plan = build_hybrid_state_pipeline_plan(config, "*|*", pp_size=2)
    groups = ProcessGroupCollection()
    groups.tp = groups.cp = groups.pp = torch.distributed.ProcessGroup(0, 1)

    def build(config, chunk=None, *, shared=True):
        submodules = _submodules()
        submodules.attention_layer.submodules.self_attention = Attention
        submodules.state_components = tuple(
            ModuleSpec(Declaration, params={"name": name}) for name in ("first", "second")
        )
        submodules.shared_state_inputs = (key,) if shared else ()
        stack = HybridStack(
            config,
            submodules,
            layer_type_list=list("**" if chunk is None else chunk.layer_pattern),
            pp_layer_offset=0 if chunk is None else chunk.layer_offset,
            pre_process=chunk is None or chunk.incoming is None,
            post_process=chunk is None or chunk.outgoing is None,
            post_layer_norm=False,
            pg_collection=groups,
        )
        if chunk is not None:
            stack.forward_adapter.bind_placement(
                "**", (StatePlacement(0, 2, 0), StatePlacement(2, 4, 1))
            )
            stack.forward_adapter.configure_pipeline(chunk)
            stack.forward_adapter.configure_cuda_graphs(stack.layers)
        if config.cuda_graph_impl == "transformer_engine":
            for layer in stack.layers:
                adapter = layer._te_cuda_graph_adapter
                inputs = adapter.get_static_inputs(
                    {"hidden_states": torch.ones(3, 1, config.hidden_size, requires_grad=True)}
                )
                assert sum(name == key for name in inputs) == 1
                adapter.finalize_sample_inputs((inputs.pop("hidden_states"),), inputs)
                layer._get_te_cuda_graph_replay_args = MethodType(
                    GraphableMegatronModule._get_te_cuda_graph_replay_args, layer
                )

                def graph(*args, layer=layer, adapter=adapter, **kwargs):
                    kwargs.pop("is_first_microbatch", None)
                    return adapter.capture(layer._te_cuda_graph_capture, *args, **kwargs)

                layer.cuda_graphs = [graph, graph]
        return stack

    reference = build(config)
    runtime = replace(config)
    if mode == "full":
        runtime.recompute_granularity = "full"
        runtime.recompute_method, runtime.recompute_num_layers = "uniform", 1
    elif mode == "graph":
        runtime.cuda_graph_impl = "transformer_engine"
    if not declared:
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            first = build(runtime, plan[0], shared=False)
            first(torch.ones(3, 1, config.hidden_size, requires_grad=True), None)
        return
    stacks = [build(runtime, chunk) for chunk in plan]
    for stack, layer in zip(stacks, reference.layers):
        stack.layers[0].load_state_dict(layer.state_dict())
    pairs = []
    for _ in range(2):
        x = torch.randn(3, 1, config.hidden_size, requires_grad=True)
        ref_x = x.detach().clone().requires_grad_()
        expected = reference(ref_x, None)
        payload = stacks[0](x, None)
        assert sum(spec.field.key == key for spec in payload.tensor_specs) == 1
        stacks[1].set_input_tensor(payload)
        actual = stacks[1](None, None)
        torch.testing.assert_close(actual, expected)
        pairs.append((x, ref_x, actual, expected))
    for x, ref_x, actual, expected in reversed(pairs):
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(x.grad, ref_x.grad)
    for stack, layer in zip(stacks, reference.layers):
        for actual, expected in zip(stack.layers[0].parameters(), layer.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad)


@pytest.mark.parametrize("mhc", [False, True])
@pytest.mark.parametrize("mode", ["eager", "full", "selective", "graph"])
def test_default_tensor_state_pipeline_relay_and_local_preparation(cpu_graph_slots, mhc, mode):
    """One declaration drives PP/VPP, retained edges, recompute and graph publication."""
    from megatron.core.models.hybrid.hybrid_state import HybridTensorState
    from megatron.core.transformer.state_boundary import StateDependency, TensorField

    class Memory(HybridTensorState):
        context_attribute = "tensor_memory"

        def dependencies(self, hidden):
            def dependency(key, origin, source, readers, delivery):
                rank = next(p.pp_rank for p in self.placement if p.start <= source < p.end)
                return StateDependency(
                    TensorField(
                        key,
                        (*hidden.shape[:2], self.config.hidden_size),
                        hidden.dtype,
                        "sbhd",
                        True,
                    ),
                    origin,
                    source,
                    rank,
                    readers,
                    delivery,
                )

            return (
                dependency("memory/value:L0", "activation", 1, (5, 13), "pipeline"),
                dependency("memory/value:L4", "activation", 9, (13,), "pipeline"),
                *(
                    dependency(
                        f"memory/lookup:P{p.start}",
                        "prepared",
                        p.start,
                        tuple(i for i in (1, 5, 9, 13) if p.start <= i < p.end),
                        "local",
                    )
                    for p in self.placement
                ),
            )

        def state_bindings(self, layer):
            return (
                {"attention": "memory_state"} if isinstance(layer.self_attention, Attention) else {}
            )

        def prepare_local_state(self, hidden, state, inputs):
            key = f"memory/lookup:P{2 * self.layer_offset}"
            state[key] = (
                inputs["input_ids"]
                .to(hidden.dtype)
                .unsqueeze(-1)
                .expand(*hidden.shape[:2], self.config.hidden_size)
                .contiguous()
            )
            return state

    class Attention(_Attention):
        def __init__(self, config, layer_number, **kwargs):
            super().__init__(config, **kwargs)
            self.index = layer_number - 1
            self.scale = nn.Parameter(torch.tensor(0.2))

        def forward(self, hidden, *, memory_state, **kwargs):
            lookup = next(v for k, v in memory_state.items() if "/lookup:" in k)
            if self.index in (0, 4):
                memory_state[f"memory/value:L{self.index}"] = self.scale * hidden.sin()
            if self.index in (2, 6):
                hidden = hidden + 0.1 * memory_state["memory/value:L0"].square()
            if self.index == 6:
                hidden = hidden + 0.1 * memory_state["memory/value:L4"].cos()
            return super().forward(hidden + lookup * 0.03, **kwargs)

    config = _config(num_layers=8, enable_hyper_connections=mhc, mhc_single_pass=mhc)
    runtime = replace(config)
    if mode == "full":
        runtime.recompute_granularity = "full"
        runtime.recompute_method, runtime.recompute_num_layers = "uniform", 1
    elif mode == "selective":
        runtime.recompute_granularity = "selective"
        runtime.recompute_modules = ["mhc"] if mhc else ["layernorm"]
        runtime.mhc_recompute_layer_num = 2
    elif mode == "graph":
        runtime.cuda_graph_impl = "transformer_engine"
    pattern = "*|-|*-*-|*-"
    plan = build_hybrid_state_pipeline_plan(runtime, pattern, pp_size=2)
    placement = tuple(
        StatePlacement(
            2 * c.layer_offset, 2 * (c.layer_offset + len(c.layer_pattern)), c.pp_rank, c.vp_stage
        )
        for c in plan
    )
    groups = ProcessGroupCollection()
    groups.tp = groups.cp = groups.pp = torch.distributed.ProcessGroup(0, 1)

    def build(config, chunk=None):
        modules = _submodules(config)
        modules.state_components = (*modules.state_components, Memory)
        modules.attention_layer.submodules.self_attention = Attention
        stack = HybridStack(
            config,
            modules,
            layer_type_list=list(
                pattern.replace("|", "") if chunk is None else chunk.layer_pattern
            ),
            pp_layer_offset=0 if chunk is None else chunk.layer_offset,
            pre_process=chunk is None or chunk.incoming is None,
            post_process=chunk is None or chunk.outgoing is None,
            post_layer_norm=False,
            pg_collection=groups,
        )
        if chunk is not None:
            stack.forward_adapter.bind_placement(pattern.replace("|", ""), placement)
            stack.forward_adapter.configure_pipeline(chunk)
            stack.forward_adapter.configure_cuda_graphs(stack.layers)
        if config.cuda_graph_impl == "transformer_engine":
            _install_cpu_graphs(stack, False)
        return stack

    torch.manual_seed(408)
    reference = build(config)
    stacks = [build(runtime, chunk) for chunk in plan]
    for stack, chunk in zip(stacks, plan):
        for local, layer in enumerate(stack.layers):
            layer.load_state_dict(reference.layers[chunk.layer_offset + local].state_dict())
    if mode == "graph":
        for stack in stacks:
            # Prepare wire schemas too, so any later dependency scan really is
            # repeated planning of an already known profile.
            stack.forward_adapter.pipeline_payload_spec(9, 2)
            component = next(c for c in stack.forward_adapter.components if isinstance(c, Memory))
            component.dependencies = lambda _: pytest.fail(
                "Prepared replay must not rescan dependencies"
            )
    pairs = []
    for microbatch in range(2):
        ids = torch.randint(0, 7, (9, 2))
        hidden = torch.randn(9, 2, config.hidden_size, requires_grad=True)
        expected_hidden = hidden.detach().clone().requires_grad_()
        expected = reference(expected_hidden, None, input_ids=ids)
        output = hidden
        for index, stack in enumerate(stacks):
            for layer in stack.layers:
                layer.current_microbatch = microbatch
            if index:
                stack.set_input_tensor(output)
            output = stack(output if index == 0 else None, None, input_ids=ids)
            if index < len(stacks) - 1:
                keys = [spec.field.key for spec in output.tensor_specs]
                assert "memory/value:L0" in keys  # Includes the MLP-only relay chunk.
                assert not any("/lookup:" in key for key in keys)
        torch.testing.assert_close(output, expected)
        pairs.append((hidden, expected_hidden, output, expected))
    for hidden, expected_hidden, output, expected in reversed(pairs):
        output.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(hidden.grad, expected_hidden.grad)
    for stack, chunk in zip(stacks, plan):
        for local, layer in enumerate(stack.layers):
            for actual, expected in zip(
                layer.parameters(), reference.layers[chunk.layer_offset + local].parameters()
            ):
                torch.testing.assert_close(actual.grad, expected.grad)
    if mode == "eager":
        specs = [s.forward_adapter.pipeline_payload_spec(9, 2, requires_grad=False) for s in stacks]
        with torch.no_grad():
            output = hidden
            for index, stack in enumerate(stacks):
                if index:
                    stack.set_input_tensor(output)
                output = stack(output if index == 0 else None, None, input_ids=ids)
                if index < len(stacks) - 1:
                    assert output.descriptor == specs[index][1]
                    assert all(not spec.requires_grad for spec in output.tensor_specs)
            torch.testing.assert_close(output, reference(hidden, None, input_ids=ids))


def test_default_tensor_state_rejects_split_consumers_during_setup():
    from megatron.core.models.hybrid.hybrid_state import HybridTensorState
    from megatron.core.transformer.enums import CudaGraphModule

    class BothBranches(HybridTensorState):
        context_attribute = "shared_memory"

        def state_bindings(self, layer):
            return {"attention": "memory_state", "mlp": "memory_state"}

        def dependencies(self, hidden):
            pytest.fail("Reject unsupported capture before querying fields or running a layer")

    config = _config(num_layers=1, enable_hyper_connections=False, mhc_single_pass=False)
    config.cuda_graph_impl = "transformer_engine"
    config.cuda_graph_modules = [CudaGraphModule.attn]
    modules = _submodules()
    modules.state_components = (BothBranches,)
    groups = ProcessGroupCollection()
    groups.tp = groups.cp = groups.pp = torch.distributed.ProcessGroup(0, 1)
    with pytest.raises(ValueError, match="shared_memory:.*complete consumer branches"):
        HybridStack(
            config, modules, layer_type_list=["*"], post_layer_norm=False, pg_collection=groups
        )


@pytest.fixture
def cpu_graph_slots(monkeypatch):
    """Use CPU storage while preserving production arena and microbatch-slot lifetimes."""
    original_init = MHCRecomputeArenaSlot.__init__

    def initialize(slot, key, tensor):
        if tensor.is_cuda:
            return original_init(slot, key, tensor)
        slot.key, slot.consumer = key, tensor
        slot.metadata = MHCRecomputeSlotMetadata(
            tensor.shape, tensor.dtype, tensor.device, tensor.layout, tensor.data_ptr()
        )

    def set_inputs(layer, inputs):
        layer._te_cuda_graph_static_hidden_inputs = tuple(inputs)
        layer._te_cuda_graph_static_hidden_input_ptrs = tuple(t.data_ptr() for t in inputs)

    monkeypatch.setattr(MHCRecomputeArenaSlot, "__init__", initialize)
    monkeypatch.setattr(
        GraphableMegatronModule, "set_te_cuda_graph_static_hidden_inputs", set_inputs
    )


def _params():
    logical = torch.tensor([0, 2, 5, 7], dtype=torch.int32)
    physical = torch.tensor([0, 3, 7, 9], dtype=torch.int32)
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=logical,
        cu_seqlens_kv=logical,
        cu_seqlens_q_padded=physical,
        cu_seqlens_kv_padded=physical,
        max_seqlen_q=9,
        max_seqlen_kv=9,
        pad_between_seqs=True,
    )


def _install_cpu_graphs(stack, packed):
    for symbol, layer in zip(stack.layer_type_list, stack.layers):
        adapter = layer._te_cuda_graph_adapter
        split = getattr(layer, "_uses_mhc_recompute_cuda_graph_split", lambda: False)()
        width = layer.config.hidden_size * (
            layer.config.num_residual_streams
            if layer.config.enable_hyper_connections and not split
            else 1
        )
        static = {"hidden_states": torch.ones(9, 1 if packed else 2, width, requires_grad=True)}
        if packed and symbol == "*":
            for suffix in ("q", "kv", "q_padded", "kv_padded"):
                static["cu_seqlens_" + suffix] = getattr(_params(), "cu_seqlens_" + suffix)
        samples = adapter.get_static_inputs(static)
        adapter.finalize_sample_inputs((samples.pop("hidden_states"),), samples)
        layer._get_te_cuda_graph_replay_args = MethodType(
            GraphableMegatronModule._get_te_cuda_graph_replay_args, layer
        )

        def make_graph(layer, adapter):
            def graph(*args, **kwargs):
                kwargs.pop("is_first_microbatch", None)
                assert all(
                    value is None or isinstance(value, torch.Tensor) for value in kwargs.values()
                )
                return adapter.capture(layer._te_cuda_graph_capture, *args, **kwargs)

            return graph

        layer.cuda_graphs = [make_graph(layer, adapter) for _ in range(2)]
        if split:
            layer.set_te_cuda_graph_static_hidden_inputs(
                [static["hidden_states"].detach().clone() for _ in range(2)]
            )


def test_mamba_tensor_return_survives_real_hybrid_graph_dispatch():
    from megatron.core.models.hybrid.hybrid_stack_adapter import HybridStateGraphAdapter
    from megatron.core.ssm.mamba_layer import MambaLayer, MambaLayerSubmodules

    config = _config(
        num_layers=1, hidden_size=4, enable_hyper_connections=False, mhc_single_pass=False
    )
    groups = ProcessGroupCollection()
    groups.tp = groups.pp = groups.cp = torch.distributed.ProcessGroup(0, 1)

    def build(config):
        return HybridStack(
            config,
            HybridStackSubmodules(
                state_components=(),
                mamba_layer=ModuleSpec(
                    MambaLayer,
                    submodules=MambaLayerSubmodules(
                        norm=_Norm, mixer=_FFN, mamba_bda=get_bias_dropout_add
                    ),
                ),
            ),
            layer_type_list=["M"],
            post_layer_norm=False,
            pg_collection=groups,
        )

    reference = build(config)
    runtime = replace(config)
    runtime.cuda_graph_impl = "transformer_engine"
    actual = build(runtime)
    actual.load_state_dict(reference.state_dict())
    layer = actual.layers[0]
    adapter = HybridStateGraphAdapter((), lambda states: {})
    layer._te_cuda_graph_adapter = adapter
    samples = adapter.get_static_inputs({"hidden_states": torch.ones(3, 1, 4)})
    adapter.finalize_sample_inputs((samples.pop("hidden_states"),), samples)

    def graph(*args, **kwargs):
        kwargs.pop("is_first_microbatch", None)
        return adapter.capture(layer._te_cuda_graph_capture, *args, **kwargs)

    layer.cuda_graphs = [graph]
    x = torch.randn(3, 1, 4, requires_grad=True)
    rx = x.detach().clone().requires_grad_()
    output, expected = actual(x, None), reference(rx, None)
    assert output.shape == (3, 1, 4)
    torch.testing.assert_close(output, expected)
    output.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, rx.grad)
    for p, q in zip(actual.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, q.grad)


@pytest.mark.parametrize("packed_sp", [False, True], ids=["mhc-rope", "receiving-thd-sp"])
def test_final_te_samples_define_state_graph_inputs(monkeypatch, packed_sp):
    """Run the real sample builder and layer replay, including late RoPE/mask changes.

    CPU modules test the receiving-stage input contract, not distributed TP math.
    """
    from megatron.core.models.hybrid.hybrid_stack_adapter import HybridStateGraphAdapter
    from megatron.core.transformer import cuda_graphs
    from megatron.core.transformer.hyper_connection import SinglePassMHCState

    class RopeAttention(_Attention):
        def forward(self, hidden, *, rotary_pos_emb, **kwargs):
            # Make omission of the sample builder's RoPE argument observable in
            # both the graph call signature and the layer's numerical result.
            return super().forward(hidden + rotary_pos_emb.squeeze(2), **kwargs)

    config = _config(
        num_layers=2,
        sequence_parallel=packed_sp,
        mhc_single_pass=not packed_sp,
        tensor_model_parallel_size=2 if packed_sp else 1,
    )
    runtime = replace(config)
    runtime.cuda_graph_impl = "transformer_engine"
    if packed_sp:
        runtime.sequence_packing_scheduler = "pack_by_seq"
        runtime.max_seqlen_per_dp_cp_rank = 8
        runtime.thd_max_packed_sequences = 2

    def build(config):
        submodules = _submodules(config)
        if not packed_sp:
            submodules.attention_layer.submodules.self_attention = RopeAttention
        groups = ProcessGroupCollection()
        groups.tp = groups.cp = groups.pp = torch.distributed.ProcessGroup(0, 1)
        return HybridStack(
            config, submodules, layer_type_list=["*"], post_layer_norm=False, pg_collection=groups
        )

    reference, actual = build(config), build(runtime)
    if actual.forward_adapter is not None:
        actual.forward_adapter.bind_placement("*-", (StatePlacement(0, 4, 0),))
        actual.forward_adapter.configure_cuda_graphs(actual.layers)
    actual.load_state_dict(reference.state_dict())
    layer, ref_layer = actual.layers[0], reference.layers[0]
    if packed_sp:
        # No mHC state crosses this graph: single-pass mHC intentionally rejects
        # TP. The generic graph adapter still owns the receiving-stage signature.
        layer._te_cuda_graph_adapter = HybridStateGraphAdapter(
            (),
            lambda states: {},
            restore_packed=layer._reconstruct_packed_seq_params_from_kwargs,
            decompose_packed=layer._decompose_packed_seq_params_to_kwargs,
            max_seqlen=8,
        )
    adapter = layer._te_cuda_graph_adapter
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(cuda_graphs, "is_te_min_version", lambda _: True)
    rotary = torch.randn(8, 1, 1, config.hidden_size)
    chunk = SimpleNamespace(
        decoder=actual,
        pre_process=not packed_sp,
        position_embedding_type="learned_absolute" if packed_sp else "rope",
        rotary_pos_emb=SimpleNamespace(),
    )

    class RotaryEmbedding:
        def get_rotary_seq_len(self, *args):
            return rotary.shape[0]

        def __call__(self, length):
            return rotary[:length]

    chunk.rotary_pos_emb = RotaryEmbedding()
    helper = object.__new__(cuda_graphs.TECudaGraphHelper)
    helper.config, helper.seq_length, helper.micro_batch_size = runtime, 8, 1
    helper.num_model_chunks, helper.num_microbatches = 1, 2
    helper.flattened_callables, helper.callables_per_chunk = [layer], [[layer]]
    helper.num_layers_per_chunk, helper.chunks_with_decoder = [1], [chunk]
    helper.tp_group = helper.dp_cp_group = None
    helper._uses_mhc_direct_write_arena = lambda: False
    base_inputs = layer.get_layer_static_inputs(8, 1)
    assert adapter._static_schema is None
    sample_args, sample_kwargs = helper._get_sample_arguments([1, 1, -1, -1])
    if packed_sp:
        assert base_inputs["padding_mask"].shape == (1, 4)
        assert sample_kwargs[0]["padding_mask"].shape == (1, 8)
    else:
        assert "rotary_pos_emb" not in base_inputs
        assert sample_kwargs[0]["rotary_pos_emb"].shape == rotary.shape

    calls = []

    def graph(*args, **kwargs):
        kwargs.pop("is_first_microbatch", None)
        assert kwargs.keys() == sample_kwargs[0].keys()
        for name, sample in sample_kwargs[0].items():
            assert kwargs[name].shape == sample.shape
        calls.append(True)
        return adapter.capture(layer._te_cuda_graph_capture, *args, **kwargs)

    layer.cuda_graphs = [graph, graph]
    layer._get_te_cuda_graph_replay_args = MethodType(
        GraphableMegatronModule._get_te_cuda_graph_replay_args, layer
    )
    prepared = dict(sample_kwargs[0])
    layer._reconstruct_packed_seq_params_from_kwargs(prepared)
    params = prepared.get("packed_seq_params")
    batches = []
    for microbatch in range(2):
        layer.current_microbatch = microbatch
        hidden = torch.randn_like(sample_args[0][0], requires_grad=True)
        expected_hidden = hidden.detach().clone().requires_grad_()
        kwargs = dict(attention_mask=None, padding_mask=None, packed_seq_params=params)
        if not packed_sp:
            kwargs["rotary_pos_emb"] = rotary * (microbatch + 1)
        state, expected_state = SinglePassMHCState(), SinglePassMHCState()
        output, _ = layer(hidden, **kwargs, **({"mhc_state": state} if not packed_sp else {}))
        expected, _ = ref_layer.forward(
            expected_hidden, **kwargs, **({"mhc_state": expected_state} if not packed_sp else {})
        )
        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(state.pre_mix, expected_state.pre_mix)
        batches.append((hidden, expected_hidden, output, expected))
    for hidden, expected_hidden, output, expected in reversed(batches):
        output.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(hidden.grad, expected_hidden.grad)
    for parameter, expected_parameter in zip(layer.parameters(), ref_layer.parameters()):
        torch.testing.assert_close(parameter.grad, expected_parameter.grad)
    assert len(calls) == 2
    bad = dict(sample_kwargs[0])
    name = "padding_mask" if packed_sp else "rotary_pos_emb"
    bad[name] = bad[name][..., :4] if packed_sp else bad[name][:4]
    with pytest.raises(ValueError, match="shape"):
        adapter.replay(graph, sample_args[0][0], mhc_state=SinglePassMHCState(), **bad)
    assert len(calls) == 2


@pytest.mark.parametrize("layout", ["sbhd", "thd"])
@pytest.mark.parametrize("pattern", ["*-*-*-", "*-*|-*-", "*|-*|-|*-"])
@pytest.mark.parametrize("mode", ["eager", "full", "graph", "selective", "selective_graph", "mlp"])
def test_mhc_without_csa2_matches_unsplit_training(cpu_graph_slots, layout, pattern, mode):
    """Two in-flight batches preserve output, input gradients and all parameter gradients."""
    config = _config()
    reference = _stack(config)
    runtime = (
        replace(
            config, recompute_granularity="full", recompute_method="uniform", recompute_num_layers=2
        )
        if mode == "full"
        else config
    )
    if mode.startswith("selective"):
        runtime = replace(
            config,
            recompute_granularity="selective",
            recompute_modules=["mhc"],
            mhc_recompute_layer_num=2,
        )
    if mode == "mlp":
        runtime = replace(config, recompute_granularity="selective", recompute_modules=["mlp"])
    if "graph" in mode:
        # CPU callables test the actual graph adapter contract, not CUDA replay.
        runtime = replace(runtime)
        runtime.cuda_graph_impl = "transformer_engine"
        if layout == "thd":
            runtime.sequence_packing_scheduler = "pack_by_seq"
            runtime.max_seqlen_per_dp_cp_rank = 9
    plan = build_hybrid_state_pipeline_plan(
        runtime, pattern, pp_size=2 if "|" in pattern else 1, qkv_format=layout
    )
    stacks = [_stack(runtime, chunk) for chunk in plan]
    placement = tuple(
        StatePlacement(
            2 * c.layer_offset, 2 * (c.layer_offset + len(c.layer_pattern)), c.pp_rank, c.vp_stage
        )
        for c in plan
    )
    for stack, chunk in zip(stacks, plan):
        assert [c.context_attribute for c in stack.forward_adapter.components] == ["mhc_state"]
        stack.forward_adapter.configure_pipeline(chunk)
        stack.forward_adapter.bind_placement(pattern.replace("|", ""), placement)
        stack.forward_adapter.configure_cuda_graphs(stack.layers)
        for local, layer in enumerate(stack.layers):
            layer.load_state_dict(reference.layers[chunk.layer_offset + local].state_dict())
        if "graph" in mode:
            _install_cpu_graphs(stack, layout == "thd")
    pairs = []
    for microbatch in range(2):
        params = _params() if layout == "thd" else None
        x = torch.randn(9, 1 if params is not None else 2, 8, requires_grad=True)
        rx = x.detach().clone().requires_grad_()
        expected = reference(rx, None, packed_seq_params=params)
        payload = None
        for stack, chunk in zip(stacks, plan):
            for layer in stack.layers:
                layer.current_microbatch = microbatch
            stack.set_input_tensor(payload)
            output = stack(
                x if chunk.incoming is None else None,
                None,
                packed_seq_params=params if chunk.incoming is None else None,
            )
            if chunk.outgoing is not None:
                assert isinstance(output, HybridStatePayload)
                assert all("csa2" not in spec.field.key for spec in output.tensor_specs)
                spec = stack.forward_adapter.pipeline_payload_spec(*x.shape[:2], params)[1]
                assert output.descriptor == spec
                payload = output
        torch.testing.assert_close(output, expected)
        pairs.append((output, expected, x, rx))
    for actual, expected, x, rx in reversed(pairs):
        probe = torch.randn_like(actual)
        (actual * probe).sum().backward()
        (expected * probe).sum().backward()
        torch.testing.assert_close(x.grad, rx.grad)
    for stack, chunk in zip(stacks, plan):
        for local, layer in enumerate(stack.layers):
            for p, q in zip(
                layer.parameters(), reference.layers[chunk.layer_offset + local].parameters()
            ):
                torch.testing.assert_close(p.grad, q.grad)


@pytest.mark.parametrize("forward_only", [False, True])
@pytest.mark.parametrize("frozen", [False, True])
def test_pipeline_execution_mode_keeps_planned_schema(cpu_graph_slots, frozen, forward_only):
    config = _config()
    plan = build_hybrid_state_pipeline_plan(config, "*-*|-*-", pp_size=2)
    first, second = (_stack(config, chunk) for chunk in plan)
    for stack, chunk in zip((first, second), plan):
        stack.forward_adapter.configure_pipeline(chunk)
    first.requires_grad_(not frozen)
    x = torch.randn(7, 2, config.hidden_size)
    with torch.set_grad_enabled(not forward_only):
        outgoing = first(x, None)
    descriptor = first.forward_adapter.pipeline_payload_spec(
        *x.shape[:2], requires_grad=not forward_only
    )[1]
    assert all(spec.requires_grad == (not forward_only) for spec in descriptor.tensor_specs)
    assert all(
        tensor.requires_grad == (not frozen and not forward_only) for tensor in outgoing.tensors
    )
    assert outgoing.descriptor == descriptor
    tensors = tuple(
        t.detach().clone().requires_grad_(s.requires_grad)
        for t, s in zip(outgoing.tensors, descriptor.tensor_specs)
    )
    incoming = second.forward_adapter.make_pipeline_payload(tensors, outgoing.descriptor)
    second.set_input_tensor(incoming)
    with torch.set_grad_enabled(not forward_only):
        loss = second(None, None).square().sum()
    if forward_only:
        assert not loss.requires_grad
        return
    gradients = backward_pipeline_payload(incoming, loss, None)
    assert all(gradient is not None for gradient in gradients)
    backward_pipeline_payload(None, outgoing, gradients)
    assert any(p.grad is not None for p in first.parameters()) == (not frozen)
    assert any(p.grad is not None for p in second.parameters())


def test_independent_mhc_cli_allows_pp2_vpp_without_overlap():
    from argparse import ArgumentParser

    from megatron.training.arguments import add_megatron_arguments, validate_args

    args = add_megatron_arguments(ArgumentParser()).parse_args(
        [
            "--hybrid-layer-pattern",
            "*-|*-|*-|*-",
            "--pipeline-model-parallel-size",
            "2",
            "--enable-hyper-connections",
            "--mhc-single-pass",
            "--no-overlap-p2p-communication",
            "--hidden-size",
            "32",
            "--num-attention-heads",
            "4",
            "--micro-batch-size",
            "1",
            "--global-batch-size",
            "8",
            "--seq-length",
            "16",
            "--max-position-embeddings",
            "16",
            "--train-iters",
            "2",
            "--lr",
            "0.001",
        ]
    )
    args.rank, args.world_size = 0, 2
    args = validate_args(args)
    assert args.experimental_attention_variant is None
    assert args.virtual_pipeline_model_parallel_size == 2


def test_mhc_pipeline_rejects_another_batchs_packed_metadata():
    """Packed prefixes cannot be silently replaced by another microbatch's metadata."""
    config = _config()
    plan = build_hybrid_state_pipeline_plan(config, "*-*|-*-", pp_size=2, qkv_format="thd")
    first, second = [_stack(config, chunk) for chunk in plan]
    first.forward_adapter.configure_pipeline(plan[0])
    second.forward_adapter.configure_pipeline(plan[1])
    params = replace(_params(), tokens_per_sample=3)
    payload = first(torch.randn(9, 1, 8, requires_grad=True), None, packed_seq_params=params)
    changed = replace(params, cu_seqlens_q=torch.tensor([0, 1, 5, 7], dtype=torch.int32))
    second.set_input_tensor(payload)
    with pytest.raises(ValueError, match="prefixes do not match"):
        second(None, None, packed_seq_params=changed)
    with pytest.raises(ValueError, match="prefixes do not match"):
        first(torch.randn(9, 1, 8), None, packed_seq_params=changed)
    changed = replace(params, tokens_per_sample=9)
    second.set_input_tensor(payload)
    with pytest.raises(ValueError, match="tokens_per_sample does not match"):
        second(None, None, packed_seq_params=changed)
    incoming, _ = second.forward_adapter.pipeline_payload_spec(9, 1, changed)
    assert incoming.fingerprint != payload.descriptor.fingerprint
    assert _stack(config).forward_adapter.pipeline_payload_spec(9, 1) == (None, None)


@pytest.mark.parametrize("cp_size,cp_rank", [(1, 0), (2, 0), (2, 1)])
def test_packed_sample_metadata_preserves_cp_coordinates(cp_size, cp_rank):
    from megatron.core.models.hybrid.hybrid_stack_adapter import _HiddenPayload
    from megatron.core.models.hybrid.hybrid_state import HybridPipelineBoundary

    group = torch.distributed.ProcessGroup(cp_rank, cp_size)
    prefix = torch.tensor([0, 1, 2, 4], dtype=torch.int32)
    payload = _HiddenPayload(
        HybridPipelineBoundary(1, "thd"),
        (torch.randn(4 // cp_size, 1, 8), prefix, prefix),
        max_seqlen=2,
        cp_size=cp_size,
        cp_rank=cp_rank,
        cp_partition_mode="zigzag",
        tokens_per_sample=2,
    )
    received = _HiddenPayload.from_metadata(
        payload.boundary, payload.tensors, payload.metadata, cp_partition_mode="zigzag"
    )
    _, packed = received.restore(cp_group=group)
    assert received.metadata == payload.metadata
    assert packed.tokens_per_sample == 2
    assert packed.local_cp_size == cp_size
    assert packed.cp_group is group


@pytest.mark.parametrize("planned", [False, True])
@pytest.mark.parametrize("receiver_has_params", [False, True])
def test_packed_moe_seq_aux_loss_survives_pipeline(planned, receiver_has_params, monkeypatch):
    """PP must preserve sample grouping, not just the packed document boundaries."""
    from megatron.core.models.hybrid.hybrid_stack_adapter import HybridStateAdapter
    from megatron.core.pipeline_parallel.typed_p2p_communication import _decode_header, _pack_header
    from megatron.core.transformer.moe.router import TopKRouter

    config = _config(
        num_layers=3,
        hidden_size=2,
        enable_hyper_connections=False,
        mhc_single_pass=False,
        num_moe_experts=2,
        moe_router_topk=1,
        moe_router_pre_softmax=True,
        moe_router_load_balancing_type="seq_aux_loss",
        moe_aux_loss_coeff=1.0,
        moe_router_fusion=False,
    )
    groups = ProcessGroupCollection()
    groups.tp = groups.cp = groups.pp = groups.expt_tp = groups.tp_cp = groups.tp_dp_cp = (
        torch.distributed.ProcessGroup(0, 1)
    )
    prefix = torch.tensor([0, 1, 2, 4], dtype=torch.int32)
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=prefix,
        cu_seqlens_kv=prefix,
        max_seqlen_q=2,
        max_seqlen_kv=2,
        tokens_per_sample=2,
    )
    logits = torch.tensor([[[0.9, 0.1]], [[0.9, 0.1]], [[0.1, 0.9]], [[0.1, 0.9]]]).log()
    router = TopKRouter(config, pg_collection=groups, layer_number=1)
    losses = []
    # Preserve the explicit size-one group without initializing global MPU state.
    monkeypatch.setattr(
        "megatron.core.tensor_parallel.mappings.get_tensor_model_parallel_group_if_none",
        lambda group: group,
    )

    def record_loss(activation, coefficient, loss, name, *args, **kwargs):
        # Keep real routing/aux-loss math; avoid the distributed metrics tracker.
        assert name == "seq_load_balancing_loss"
        losses.append(loss)
        return activation

    monkeypatch.setattr(router, "attach_and_log_load_balancing_loss", record_loss)

    def seq_aux_loss(hidden, packed):
        grouped, _, _ = TransformerLayer._maybe_unflatten_for_moe(
            SimpleNamespace(is_moe_layer=True), hidden, None, packed
        )
        router.routing(grouped, packed_seq_params=packed)
        return losses.pop()

    reference = logits.clone().requires_grad_()
    expected = seq_aux_loss(reference, params)
    torch.testing.assert_close(expected, torch.tensor(1.8))
    expected.backward()
    # Without sample grouping, the same document prefixes give a different objective.
    torch.testing.assert_close(
        seq_aux_loss(logits, replace(params, tokens_per_sample=None)), torch.tensor(1.0)
    )

    plan = build_hybrid_state_pipeline_plan(config, "*|*|*", pp_size=3, qkv_format="thd")
    adapters = []
    for chunk in plan:
        adapter = HybridStateAdapter(
            config,
            components=(),
            hidden_size=2,
            hidden_dtype=torch.float32,
            layer_type_list=list(chunk.layer_pattern),
            pp_layer_offset=chunk.layer_offset,
            pre_process=chunk.incoming is None,
            post_process=chunk.outgoing is None,
            is_mtp_layer=False,
            pg_collection=groups,
        )
        adapter.configure_pipeline(chunk)
        adapters.append(adapter)

    hidden = logits.clone().requires_grad_()
    packed, payload = params, None
    for index, adapter in enumerate(adapters):
        incoming, outgoing = adapter.pipeline_payload_spec(4, 1, params)
        if payload is not None:
            if planned:
                descriptor = incoming
                assert descriptor == payload.descriptor
            else:
                descriptor = _decode_header(
                    _pack_header(
                        payload.tensor_specs, payload.metadata, 0, boundary_id=payload.boundary_id
                    ),
                    0,
                )
            tensors = tuple(t.detach().requires_grad_(t.requires_grad) for t in payload.tensors)
            payload = adapter.make_pipeline_payload(tensors, descriptor)
            hidden, packed, context = adapter.prepare_forward(
                payload, params if receiver_has_params else None, None
            )
        else:
            hidden, packed, context = adapter.prepare_forward(hidden, packed, None)
        loss = seq_aux_loss(hidden, packed)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        torch.testing.assert_close(hidden.grad, reference.grad)
        assert packed.tokens_per_sample == 2
        result = adapter.finalize_forward(hidden, packed, context)
        if index < len(adapters) - 1:
            payload = result
            assert payload.descriptor == outgoing


@pytest.mark.parametrize("selective", [False, True])
def test_mhc_graph_rejects_changed_static_packed_maximum(cpu_graph_slots, selective):
    config = _config(
        recompute_granularity="selective" if selective else None,
        recompute_modules=["mhc"] if selective else None,
    )
    config.cuda_graph_impl = "transformer_engine"
    config.sequence_packing_scheduler = "pack_by_seq"
    config.max_seqlen_per_dp_cp_rank = 9
    stack = _stack(config)
    _install_cpu_graphs(stack, True)
    params = replace(_params(), max_seqlen_q=8, max_seqlen_kv=8)
    with pytest.raises(ValueError, match="static max_seqlen"):
        stack(torch.randn(9, 1, 8, requires_grad=True), None, packed_seq_params=params)


@pytest.mark.parametrize("vp_size", [1, 2])
@pytest.mark.parametrize("layout", ["sbhd", "thd"])
@pytest.mark.parametrize("full_recompute", [False, True])
@pytest.mark.parametrize("planned", [False, True])
@pytest.mark.parametrize(
    "frozen", [pytest.param(False, id="trainable"), pytest.param(True, id="frozen")]
)
def test_independent_mhc_actual_pipeline(
    transport_groups, vp_size, layout, full_recompute, planned, frozen
):
    """HybridModel's normal factory runs real 1F1B/VPP transport without CSA2."""
    groups, device = transport_groups
    if groups.pp.size() != 2:
        pytest.skip("This placement requires PP=2")
    groups.dp_cp = groups.tp
    groups.embd = groups.pos_embd = None
    rank, count = groups.pp.rank(), 4
    config = _config(
        num_layers=8,
        pipeline_model_parallel_size=2,
        pipeline_dtype=torch.float32,
        virtual_pipeline_model_parallel_size=vp_size if vp_size > 1 else None,
        microbatch_group_size_per_vp_stage=2,
        cross_entropy_loss_fusion=False,
        deallocate_pipeline_outputs=True,
        recompute_granularity="full" if full_recompute else None,
        recompute_method="uniform" if full_recompute else None,
        recompute_num_layers=2 if full_recompute else None,
    )
    pattern = "*-*-|*-*-" if vp_size == 1 else "*-|*-|*-|*-"

    def model(config, groups, pattern, pre_process, post_process, vp_stage=None):
        return HybridModel(
            config,
            ModuleSpec(HybridStack, params={"post_layer_norm": False}, submodules=_submodules()),
            vocab_size=32,
            max_sequence_length=16,
            hybrid_layer_pattern=pattern,
            position_embedding_type="none",
            share_embeddings_and_output_weights=False,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=groups,
            vp_stage=vp_stage,
        ).to(device)

    torch.manual_seed(2026)
    local_groups = copy(groups)
    local_groups.pp = groups.tp
    reference = model(
        replace(
            config,
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            recompute_granularity=None,
            recompute_method=None,
            recompute_num_layers=None,
        ),
        local_groups,
        "*-" * 4,
        True,
        True,
    )
    models = [
        model(
            config,
            groups,
            pattern,
            rank == 0 and vp == 0,
            rank == 1 and vp == vp_size - 1,
            vp if vp_size > 1 else None,
        )
        for vp in range(vp_size)
    ]
    plan = build_hybrid_state_pipeline_plan(config, pattern, pp_size=2)
    chunks = [plan[rank + 2 * vp] for vp in range(vp_size)]
    parameters = dict(reference.named_parameters())

    def reference_name(name, offset):
        parts = name.split(".")
        if parts[:2] == ["decoder", "layers"]:
            parts[2] = str(int(parts[2]) + offset)
        return ".".join(parts)

    with torch.no_grad():
        for stage, chunk in zip(models, chunks):
            assert [c.context_attribute for c in stage.decoder.forward_adapter.components] == [
                "mhc_state"
            ]
            for name, p in stage.named_parameters():
                p.copy_(parameters[reference_name(name, chunk.layer_offset)])
    if frozen:
        reference.embedding.requires_grad_(False)
        for layer in reference.decoder.layers[: len(plan[0].layer_pattern)]:
            layer.requires_grad_(False)
        if rank == 0:
            models[0].requires_grad_(False)
    params = replace(_params(), tokens_per_sample=3) if layout == "thd" else None
    if params is not None:
        for suffix in ("q", "kv", "q_padded", "kv_padded"):
            setattr(
                params, "cu_seqlens_" + suffix, getattr(params, "cu_seqlens_" + suffix).to(device)
            )
    shape = (1 if params is not None else 2, 9)
    generator = torch.Generator().manual_seed(2027)
    batches = [
        (
            torch.randint(32, shape, generator=generator).to(device),
            torch.randint(32, shape, generator=generator).to(device),
        )
        for _ in range(count)
    ]
    expected, actual = [], []
    for tokens, labels in batches:
        loss = reference(tokens, None, None, labels=labels, packed_seq_params=params).float().mean()
        expected.append(loss.detach())
        (loss / count).backward()

    def forward_step(iterator, stage):
        tokens, labels = next(iterator)
        output = stage(
            tokens if stage.pre_process else None,
            None,
            None,
            labels=labels if stage.post_process else None,
            packed_seq_params=params if stage.pre_process else None,
        )

        def loss_func(losses):
            loss = losses.float().mean()
            actual.append(loss.detach().clone())
            return loss, {"loss": loss.detach()}

        return output, loss_func

    if planned:

        def prepare(iterator, stage, count, *, forward_only):
            pairs = [stage.pipeline_payload_spec(9, shape[0], params) for _ in range(count)]
            return PipelineDataIterator(
                iterator,
                PipelinePayloadPlan(
                    tuple(pair[0] for pair in pairs), tuple(pair[1] for pair in pairs)
                ),
            )

        forward_step.prepare_pipeline_inputs = prepare
    schedule = (
        forward_backward_pipelining_with_interleaving
        if vp_size > 1
        else forward_backward_pipelining_without_interleaving
    )
    schedule(
        forward_step_func=forward_step,
        data_iterator=[iter(batches) for _ in models] if vp_size > 1 else iter(batches),
        model=models if vp_size > 1 else models[0],
        num_microbatches=count,
        seq_length=9,
        micro_batch_size=shape[0],
        forward_only=False,
        p2p_communicator=P2PCommunicator(groups.pp, config),
        pg_collection=groups,
    )
    if rank == 1:
        torch.testing.assert_close(torch.stack(actual), torch.stack(expected))
    for stage, chunk in zip(models, chunks):
        for name, p in stage.named_parameters():
            torch.testing.assert_close(
                p.grad,
                parameters[reference_name(name, chunk.layer_offset)].grad,
                atol=3e-6,
                rtol=5e-5,
            )
