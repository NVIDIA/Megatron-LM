# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Independent single-pass mHC training with ordinary attention and Hybrid boundaries."""

from copy import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.hybrid.hybrid_block import HybridStack, HybridStackSubmodules
from megatron.core.models.hybrid.hybrid_model import HybridModel, get_hybrid_state_components
from megatron.core.models.hybrid.hybrid_state import build_hybrid_state_pipeline_plan
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.pipeline_payload import (
    PipelineDataIterator,
    PipelinePayloadPlan,
)
from megatron.core.pipeline_parallel.schedules import (
    forward_backward_pipelining_with_interleaving,
    forward_backward_pipelining_without_interleaving,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import random as checkpoint_runtime
from megatron.core.transformer.spec_utils import ModuleSpec
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


@pytest.fixture(autouse=True)
def _cpu_rng(monkeypatch):
    if not torch.cuda.is_available():
        monkeypatch.setattr(checkpoint_runtime, "_get_cuda_rng_state", lambda **kwargs: None)
        monkeypatch.setattr(checkpoint_runtime, "_set_cuda_rng_state", lambda *args, **kwargs: None)
        monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda message: None)
        monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: None)


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
