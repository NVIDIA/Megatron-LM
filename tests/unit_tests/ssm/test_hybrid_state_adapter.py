# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Independent single-pass mHC training with ordinary attention and Hybrid boundaries."""

from copy import copy
from dataclasses import replace
from types import MethodType

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.hybrid.hybrid_block import HybridStack, HybridStackSubmodules
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.models.hybrid.hybrid_stack_adapter import (
    HybridStatePayload,
    build_hybrid_state_pipeline_plan,
)
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
from megatron.core.transformer.mhc_recompute import MHCRecomputeArenaSlot, MHCRecomputeSlotMetadata
from megatron.core.transformer.module import GraphableMegatronModule
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


def _submodules():
    return HybridStackSubmodules(
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
        _submodules(),
        layer_type_list=list(
            "*-" * (config.num_layers // 2) if chunk is None else chunk.layer_pattern
        ),
        pp_layer_offset=0 if chunk is None else chunk.layer_offset,
        pre_process=chunk is None or chunk.incoming is None,
        post_process=chunk is None or chunk.outgoing is None,
        post_layer_norm=False,
        pg_collection=groups,
    )


@pytest.fixture(autouse=True)
def _cpu_rng(monkeypatch):
    if not torch.cuda.is_available():
        monkeypatch.setattr(checkpoint_runtime, "_get_cuda_rng_state", lambda **kwargs: None)
        monkeypatch.setattr(checkpoint_runtime, "_set_cuda_rng_state", lambda *args, **kwargs: None)


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
        assert len(adapter.components) == 1
        split = layer._uses_mhc_recompute_cuda_graph_split()
        width = layer.config.hidden_size * (1 if split else layer.config.num_residual_streams)
        static = {"hidden_states": torch.ones(9, 1 if packed else 2, width, requires_grad=True)}
        if packed and symbol == "*":
            for suffix in ("q", "kv", "q_padded", "kv_padded"):
                static["cu_seqlens_" + suffix] = getattr(_params(), "cu_seqlens_" + suffix)
        adapter.get_static_inputs(static)
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
    for stack, chunk in zip(stacks, plan):
        assert [c.context_attribute for c in stack.forward_adapter.components] == ["mhc_state"]
        stack.forward_adapter.configure_pipeline(chunk)
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
    params = _params()
    payload = first(torch.randn(9, 1, 8, requires_grad=True), None, packed_seq_params=params)
    changed = replace(params, cu_seqlens_q=torch.tensor([0, 1, 5, 7], dtype=torch.int32))
    second.set_input_tensor(payload)
    with pytest.raises(ValueError, match="prefixes do not match"):
        second(None, None, packed_seq_params=changed)
    with pytest.raises(ValueError, match="prefixes do not match"):
        first(torch.randn(9, 1, 8), None, packed_seq_params=changed)
    assert _stack(config).forward_adapter.pipeline_payload_spec(9, 1) == (None, None)


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
def test_independent_mhc_actual_pipeline(
    transport_groups, vp_size, layout, full_recompute, planned
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
    params = _params() if layout == "thd" else None
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
