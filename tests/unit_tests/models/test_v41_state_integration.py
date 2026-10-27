# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Run main's HybridStack with real CSA2/mHC and CPU projection substitutes."""

import copy
from dataclasses import fields, replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.deepseek_v41 import hybrid_adapter as v41_adapter
from megatron.core.models.deepseek_v41.config import DeepSeekV41Config
from megatron.core.models.deepseek_v41.hybrid_adapter import (
    V41ForwardContext,
    deepseek_v41_stack_spec,
)
from megatron.core.models.deepseek_v41.state import BatchShape, V41StateCodec
from megatron.core.models.hybrid import hybrid_block
from megatron.core.models.hybrid.layers.utils import create_layer_config
from megatron.core.pipeline_parallel.pipeline_payload import (
    TensorStatePayload,
    backward_pipeline_payload,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.cuda_graphs import (
    _set_capture_end,
    _set_capture_start,
    is_graph_capturing,
)
from megatron.core.transformer.experimental_attention_variant import (
    deepseek_v4_hybrid_attention as attention,
)
from megatron.core.transformer.experimental_attention_variant.csa2_module_spec import (
    get_csa2_module_spec_for_backend,
)
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from tests.unit_tests.transformer.experimental_attention_variant.test_csa2 import (
    CpuRoPE,
    Group,
    Linear,
    Norm,
    _setup,
)


class Backend:
    def linear(self):
        return Linear

    column_parallel_linear = linear
    row_parallel_linear = linear

    def layer_norm(self, **kwargs):
        return Norm


class CpuYarn(CpuRoPE):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def get_rotary_seq_len(self, inference, transformer, hidden, config, packed=None):
        return hidden.shape[0] if packed is None else packed.max_seqlen_q

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs), 1.0


class Dense(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x, **kwargs):
        return self.proj(x).tanh(), None


def _config():
    cfg, _ = _setup()
    values = {f.name: getattr(cfg, f.name) for f in fields(cfg) if f.init}
    values.update(
        num_layers=6,
        csa_compress_ratios=[2, 0, 2, 0, 2, 0],
        csa2_kv_source_layers=[0],
        csa2_index_source_layers=[0, 2],
        enable_mhc_connections=True,
        mhc_single_pass=True,
        mhc_num_residual_streams=2,
        output_projection_groups=1,
        output_projection_lora_rank=4,
    )
    return DeepSeekV41Config(**values)


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setattr(attention, 'YarnRotaryEmbedding', CpuYarn)
    monkeypatch.setattr(hybrid_block, 'TENorm', Norm)
    monkeypatch.setattr(DSAIndexerLossLoggingHelper, 'save_loss_to_tracker', lambda **kw: None)
    monkeypatch.setattr(DSAIndexerLossAutoScaler, 'main_loss_backward_scale', torch.tensor(1.0))
    attn = ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=Norm,
            self_attention=get_csa2_module_spec_for_backend(Backend()),
            self_attn_bda=get_bias_dropout_add,
        ),
    )
    dense = ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            pre_mlp_layernorm=Norm, mlp=Dense, mlp_bda=get_bias_dropout_add
        ),
    )
    spec = replace(deepseek_v41_stack_spec.submodules, csa2_layer=attn, moe_layer=dense)
    g = Group()
    groups = ProcessGroupCollection(tp=g, cp=g, pp=g, dp=g, dp_cp=g, ep=g, expt_dp=g, tp_cp=g)
    return _config(), spec, groups


def _stack(cfg, spec, groups, start=0, end=6):
    return hybrid_block.HybridStack(
        cfg,
        spec,
        pre_process=start == 0,
        post_process=end == 6,
        pp_layer_offset=start,
        layer_config_list=[create_layer_config(cfg, s) for s in ('VE' * 3)[start:end]],
        pg_collection=groups,
    )


@pytest.mark.parametrize("fused", [False, True])
def test_hybrid_final_contraction_uses_configured_aggregate(setup, monkeypatch, fused):
    """Do not silently drop the fusion callback at the HybridStack boundary."""
    cfg, spec, groups = setup
    cfg.use_fused_mhc = fused
    cfg.mhc_fused_backend = "native"
    calls = []

    def aggregate(streams, mix):
        calls.append((streams.shape, mix.dtype))
        return (streams.float() * mix.unsqueeze(-1)).sum(-2)

    monkeypatch.setattr(
        hybrid_block,
        "get_mhc_aggregate",
        lambda config: aggregate if config.use_fused_mhc else None,
    )
    stack = _stack(cfg, spec, groups)
    x = torch.randn(8, 2, cfg.hidden_size, requires_grad=True)
    output, _ = _run(stack, x, cfg, groups)
    output.square().mean().backward()
    assert calls == ([(torch.Size([8, 2, 2, 16]), torch.float32)] if fused else [])
    assert torch.isfinite(x.grad).all()


def _run(stack, x, cfg, groups, values=None, *, capture=None, shape=None, hasher=None):
    codec = V41StateCodec(cfg, shape or BatchShape(x.shape[0], x.shape[1]), groups.cp)
    context = V41ForwardContext(
        codec=codec, values={} if values is None else values, capture=capture, hasher=hasher
    )
    _, context.mhc_state, _ = codec.restore(context.values, stack.forward_adapter.start)
    if not stack.pre_process:
        stack.set_input_tensor(x)
    out = stack(x, None, forward_context=context)
    return out, context


class RoutedDense(nn.Module):
    """Small differentiable MoE substitute for CPU orchestration checks."""

    def __init__(self, config, **kwargs):
        super().__init__()
        self.router = nn.Linear(config.hidden_size, 2, bias=False)
        self.experts = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def select_token_dispatcher(self):
        pass

    def compute_routing(self, hidden, padding_mask=None, input_ids=None, packed_seq_params=None):
        probs = self.router(hidden).reshape(-1, 2).softmax(-1)
        return probs, torch.ones_like(probs, dtype=torch.bool)

    def forward(self, hidden, router_output=None, **kwargs):
        assert not is_graph_capturing(), 'Experts must remain eager'
        probs, _ = self.compute_routing(hidden) if router_output is None else router_output
        return self.experts(hidden).tanh() * probs[:, :1].reshape(*hidden.shape[:-1], 1), None


class EagerGraphSlots:
    """Exercise tensor boundaries on CPU; this does not emulate CUDA graph numerics."""

    def __init__(self, region, hidden, values, **kwargs):
        self.region = region
        names = tuple(name for name, _ in region.named_parameters())
        assert not any('hyper_connection' in name or 'experts' in name for name in names)

    def run(self, hidden, values, **kwargs):
        _set_capture_start()
        try:
            return self.region.run(hidden, values)
        finally:
            _set_capture_end()


def test_cli_warmup_arms_public_stateful_regions_with_pipeline_slots(setup, monkeypatch):
    """The existing TE helper supplies peak in-flight slots to model-owned regions."""
    from megatron.core.pipeline_parallel import schedules
    from megatron.core.transformer import cuda_graphs
    from megatron.core.transformer.enums import CudaGraphModule

    cfg, spec, groups = setup
    cfg = replace(
        cfg,
        num_moe_experts=2,
        cuda_graph_impl='transformer_engine',
        cuda_graph_modules=[CudaGraphModule.attn, CudaGraphModule.moe_router],
    )
    adapters = [
        _stack(cfg, spec, groups, 0, 2).forward_adapter,
        _stack(cfg, spec, groups, 2, 6).forward_adapter,
    ]
    helper = object.__new__(cuda_graphs.TECudaGraphHelper)
    helper.stateful_graph_adapters = list(enumerate(adapters))
    helper.pp_group = SimpleNamespace(size=lambda: 2)
    helper.num_model_chunks = 2
    helper.config = SimpleNamespace(microbatch_group_size_per_vp_stage=2)
    helper.p2p_communicator = None
    monkeypatch.setattr(cuda_graphs, 'get_num_microbatches', lambda: 2)
    monkeypatch.setattr(schedules, 'get_pp_rank_microbatches', lambda *a, **kw: (0, 0, 2, 0))
    monkeypatch.setattr(
        schedules, 'get_schedule_table', lambda *a: [(0, 0), (1, 0), (0, 1), (1, 1)]
    )
    helper._prepare_stateful_graphs()
    assert adapters[0]._automatic_capture == dict(
        layers={0, 1}, slots=2, backend='transformer_engine'
    )
    assert adapters[1]._automatic_capture == dict(
        layers={2, 3, 4, 5}, slots=1, backend='transformer_engine'
    )


@pytest.mark.parametrize('selected', [(0, 2, 4), (1, 3, 5), tuple(range(6))])
@pytest.mark.parametrize('selective', [False, True])
def test_consumer_graph_preserves_shared_state_and_mhc_checkpoints(
    setup, monkeypatch, selected, selective
):
    """Two outstanding forwards retain owner gradients through consumer-only regions."""
    cfg, spec, groups = setup
    if selective:
        cfg.recompute_granularity = 'selective'
        cfg.recompute_modules = ['mhc']
        cfg.mhc_recompute_layer_num = 2
    spec = replace(
        spec,
        moe_layer=ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                pre_mlp_layernorm=Norm, mlp=RoutedDense, mlp_bda=get_bias_dropout_add
            ),
        ),
    )
    reference = _stack(cfg, spec, groups)
    actual = copy.deepcopy(reference)
    monkeypatch.setattr(v41_adapter, 'StatefulGraphs', EagerGraphSlots)
    aggregate_calls = []
    for layer in actual.layers:
        original = layer.hyper_connection._single_pass_aggregate

        def aggregate(*args, original=original):
            aggregate_calls.append(is_graph_capturing())
            return original(*args)

        monkeypatch.setattr(layer.hyper_connection, '_single_pass_aggregate', aggregate)
    expected, observed = [], []
    inputs = [torch.randn(8, 1, cfg.hidden_size) for _ in range(2)]
    for i, value in enumerate(inputs):
        values = None
        shape = None
        x, y = value.clone().requires_grad_(), value.clone().requires_grad_()
        expected.append((_run(reference, x, cfg, groups, values, shape=shape)[0], x))
        capture = dict(layers=set(selected), slots=2, backend='torch') if i == 0 else None
        observed.append((_run(actual, y, cfg, groups, values, shape=shape, capture=capture)[0], y))
    for (want, x), (got, y) in zip(expected, observed):
        torch.testing.assert_close(got, want)
        want.square().mean().backward()
        got.square().mean().backward()
        torch.testing.assert_close(y.grad, x.grad)
    for (name, a), (_, b) in zip(reference.named_parameters(), actual.named_parameters()):
        torch.testing.assert_close(b.grad, a.grad, msg=name)
    assert len(actual.forward_adapter.graphs) == len(selected)
    assert aggregate_calls and not any(aggregate_calls)
    assert len(aggregate_calls) >= (24 if selective else 12)


@pytest.fixture
def graph_process_groups():
    """Initialize real groups and graph-safe TE RNG for the CUDA integration cases."""
    if not torch.cuda.is_available():
        pytest.skip('CUDA integration requires a GPU')
    import os

    from transformer_engine.pytorch.distributed import get_all_rng_states, set_all_rng_states

    from megatron.core import parallel_state
    from megatron.core.tensor_parallel import random as rng

    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    owned = not torch.distributed.is_initialized()
    if owned:
        torch.distributed.init_process_group('nccl')
    parallel_state.initialize_model_parallel()
    previous = get_all_rng_states()
    old_tracker, old_initialized = (
        rng._CUDA_RNG_STATE_TRACKER,
        rng._CUDA_RNG_STATE_TRACKER_INITIALIZED,
    )
    rng.model_parallel_cuda_manual_seed(123, te_rng_tracker=True, force_reset_rng=True)
    try:
        yield ProcessGroupCollection.use_mpu_process_groups()
    finally:
        set_all_rng_states(previous)
        rng._CUDA_RNG_STATE_TRACKER, rng._CUDA_RNG_STATE_TRACKER_INITIALIZED = (
            old_tracker,
            old_initialized,
        )
        parallel_state.destroy_model_parallel()
        if owned:
            torch.distributed.destroy_process_group()


@pytest.mark.parametrize('backend', ['torch', 'transformer_engine'])
@pytest.mark.parametrize('selective', [False, True])
@pytest.mark.parametrize('ratio', [1, 2])
def test_real_v41_consumer_graph_training_step(
    graph_process_groups, record_property, backend, selective, ratio
):
    """Real attention/router graphs, Engram, two microbatches and parameter updates."""
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.models.deepseek_v41.config import EngramConfig
    from megatron.core.models.deepseek_v41.engram_hash import EngramLayout
    from megatron.core.models.deepseek_v41.model import DeepSeekV41Model
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer

    torch.manual_seed(123)
    layout = EngramLayout.from_args(
        SimpleNamespace(
            engram_layer_ids=[0, 1],
            engram_num_embeddings=[0, 0],
            engram_max_ngram_size=2,
            engram_vocab_size=17,
            engram_n_heads=2,
            engram_head_dim=4,
        )
    )
    cfg = DeepSeekV41Config(
        num_layers=6,
        hidden_size=128,
        num_attention_heads=64,
        params_dtype=torch.bfloat16,
        bf16=True,
        multi_latent_attention=True,
        experimental_attention_variant='dsv4_hybrid',
        dsv4_version='v4.1',
        q_lora_rank=128,
        v_head_dim=512,
        qk_pos_emb_head_dim=64,
        output_projection_groups=8,
        output_projection_lora_rank=16,
        qk_layernorm=True,
        normalization='RMSNorm',
        csa_compress_ratios=[ratio, 0, ratio, 0, ratio, 0],
        csa2_kv_source_layers=[0],
        csa2_index_source_layers=[0, 2],
        csa_window_size=128,
        dsa_indexer_n_heads=32,
        dsa_indexer_head_dim=128,
        dsa_indexer_topk=8,
        dsa_indexer_rotate_activation=False,
        dsa_kernel_backend='cudnn',
        dsa_indexer_loss_coeff=0.01,
        enable_mhc_connections=True,
        mhc_single_pass=True,
        mhc_num_residual_streams=2,
        use_fused_mhc=True,
        num_moe_experts=4,
        moe_router_topk=2,
        moe_ffn_hidden_size=128,
        moe_shared_expert_intermediate_size=128,
        moe_router_dtype='fp32',
        moe_router_score_function='sqrtsoftplus',
        moe_router_fusion=True,
        moe_router_enable_expert_bias=True,
        moe_router_load_balancing_type='none',
        moe_token_dispatcher_type='alltoall',
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        add_bias_linear=False,
        attention_dropout=0,
        hidden_dropout=0,
        gradient_accumulation_fusion=True,
        cuda_graph_impl='transformer_engine' if backend == 'transformer_engine' else 'none',
        cuda_graph_modules=['attn', 'moe_router'] if backend == 'transformer_engine' else [],
        recompute_granularity='selective' if selective else None,
        recompute_modules=['mhc'] if selective else [],
        mhc_recompute_layer_num=2,
        engram_config=EngramConfig(
            layer_ids=[0, 1],
            num_embeddings=[sum(sum(heads) for heads in layer) for layer in layout.primes],
            max_ngram_size=2,
            vocab_size=17,
            n_heads=2,
            head_dim=4,
            pad_token_id=2,
            compressed_vocab_size=128,
        ),
    )

    def build():
        return (
            DeepSeekV41Model(
                cfg,
                128,
                64,
                pg_collection=graph_process_groups,
                token_map=torch.arange(128),
                share_embeddings_and_output_weights=False,
                parallel_output=False,
            )
            .cuda()
            .train()
        )

    reference, actual = build(), build()
    actual.load_state_dict(reference.state_dict())
    # Exercise BF16 kernels with FP32 accumulated gradients and master weights.
    # SGD isolates graph parity from Adam's amplification of near-zero mHC
    # coefficient gradients; distributed Adam has separate public-graph coverage.
    wrapped = [
        DistributedDataParallel(
            cfg,
            DistributedDataParallelConfig(grad_reduce_in_fp32=True),
            model,
            pg_collection=graph_process_groups,
        )
        for model in (reference, actual)
    ]
    optimizers = [
        get_megatron_optimizer(
            OptimizerConfig(
                optimizer='sgd',
                lr=0.1,
                sgd_momentum=0.0,
                bf16=True,
                params_dtype=torch.bfloat16,
                weight_decay=0.0,
                clip_grad=0.0,
            ),
            [model],
            pg_collection=graph_process_groups,
            use_gloo_process_groups=False,
        )
        for model in wrapped
    ]
    if backend == 'transformer_engine':
        actual.decoder.forward_adapter.prepare_cuda_graphs(slots=2, backend=backend)
    tokens = [torch.randint(0, 128, (1, 32), device='cuda') for _ in range(2)]
    hidden = [torch.randn(32, 1, 128, device='cuda', dtype=torch.bfloat16) for _ in range(2)]
    metrics = {'logits': [], 'input_gradient': [], 'parameter_gradient': []}
    try:
        for step in range(2):
            if step:
                # Test a captured replay after a parameter update at identical
                # weights. Independent BF16 trajectories can change discrete
                # selections and obscure the graph boundary comparison.
                actual.load_state_dict(reference.state_dict())
                for optimizer in optimizers:
                    optimizer.reload_model_params()
            results = []
            for model, wrapper, optimizer in zip((reference, actual), wrapped, optimizers):
                optimizer.zero_grad()
                wrapper.zero_grad_buffer()
                forwards = []
                for i in range(2):
                    x = hidden[i].clone().requires_grad_()
                    packed = None
                    if ratio == 2:
                        from megatron.core.packed_seq_params import PackedSeqParams

                        packed = PackedSeqParams(
                            qkv_format='thd',
                            cu_seqlens_q=torch.tensor(
                                [0, 13 - i, 29 - 2 * i], device='cuda', dtype=torch.int32
                            ),
                            cu_seqlens_kv=torch.tensor(
                                [0, 13 - i, 29 - 2 * i], device='cuda', dtype=torch.int32
                            ),
                            cu_seqlens_q_padded=torch.tensor(
                                [0, 16, 32], device='cuda', dtype=torch.int32
                            ),
                            cu_seqlens_kv_padded=torch.tensor(
                                [0, 16, 32], device='cuda', dtype=torch.int32
                            ),
                            max_seqlen_q=16,
                            max_seqlen_kv=16,
                            cp_group=graph_process_groups.cp,
                        )
                    kwargs = (
                        dict(capture_layers=list(range(6)), graph_slots=2, graph_backend=backend)
                        if model is actual and step == 0 and i == 0 and backend == 'torch'
                        else {}
                    )
                    out = wrapper(
                        tokens[i],
                        torch.arange(32, device='cuda').expand_as(tokens[i]),
                        decoder_input=x,
                        packed_seq_params=packed,
                        **kwargs,
                    )
                    forwards.append((out, x))
                for i, (out, _) in enumerate(forwards):
                    torch.nn.functional.cross_entropy(
                        out.float().reshape(-1, 128), tokens[i].reshape(-1)
                    ).backward()
                wrapper.finish_grad_sync()
                results.append(forwards)

            def close(a, b, name, *, gradient=False, updated=False):
                a, b = a.detach().double().flatten(), b.detach().double().flatten()
                # An angle is ill-conditioned for the near-zero mHC gradients.
                # Keep an absolute L2 gate as well as relative/cosine gates.
                if gradient and (a - b).norm() <= 1e-5:
                    return
                if updated and (a - b).norm() <= 1e-6:
                    return
                if a.numel() == 1:
                    torch.testing.assert_close(
                        a, b, rtol=0.02, atol=1e-6, msg=lambda msg: f'{name}: {msg}'
                    )
                elif min(a.norm(), b.norm()) < 1e-7:
                    assert (a - b).norm() <= 1e-6, name
                else:
                    cosine = torch.nn.functional.cosine_similarity(a, b, dim=0)
                    relative_l2 = (a - b).norm() / b.norm()
                    category = 'logits'
                    if gradient:
                        category = (
                            'input_gradient' if name == 'input gradient' else 'parameter_gradient'
                        )
                    if not updated:
                        metrics[category].append((cosine.item(), relative_l2.item()))
                    assert cosine >= (0.999 if gradient else 0.99999), (name, cosine.item())
                    assert relative_l2 <= (0.03 if gradient else 0.01), (name, relative_l2.item())

            for (want, x), (got, y) in zip(*results):
                close(got, want, 'logits')
                close(y.grad, x.grad, 'input gradient', gradient=True)
            for (name, a), (_, b) in zip(reference.named_parameters(), actual.named_parameters()):
                close(b.main_grad, a.main_grad, name, gradient=True)
            for optimizer in optimizers:
                assert optimizer.step()[0]
            for (name, a), (_, b) in zip(reference.named_parameters(), actual.named_parameters()):
                close(b, a, 'updated ' + name, updated=True)
            for a, b in zip(reference.decoder.layers, actual.decoder.layers):
                if hasattr(a.inner_layer.mlp, 'router'):
                    torch.testing.assert_close(
                        a.inner_layer.mlp.router.local_tokens_per_expert,
                        b.inner_layer.mlp.router.local_tokens_per_expert,
                    )
        assert len(actual.decoder.forward_adapter.graphs) == 6
        for category, values in metrics.items():
            if not values:
                continue
            record_property(category + '_min_cosine', min(c for c, _ in values))
            record_property(category + '_max_relative_l2', max(e for _, e in values))
    finally:
        for graph in actual.decoder.forward_adapter.graphs.values():
            graph.abort()
            graph.close()


@pytest.mark.parametrize(
    'method,num_layers',
    [
        ('uniform', 1),
        ('uniform', 2),
        ('uniform', 4),
        ('uniform', 7),
        ('block', 1),
        ('block', 3),
        ('block', 6),
    ],
)
def test_hybrid_stack_checkpoint_and_pipeline_owner_gradients(setup, method, num_layers):
    cfg, spec, groups = setup
    full = _stack(cfg, spec, groups)
    checkpointed = copy.deepcopy(full)
    checkpointed.config.recompute_granularity = 'full'
    checkpointed.config.recompute_method = method
    checkpointed.config.recompute_num_layers = num_layers
    # Split between Full and its future Reindex/Reuse consumers, including the mHC edge.
    stages = [_stack(cfg, spec, groups, 0, 2), _stack(cfg, spec, groups, 2, 6)]
    for stage, offset in zip(stages, (0, 2)):
        for i, layer in enumerate(stage.layers):
            layer.load_state_dict(full.layers[offset + i].state_dict())
    stages[-1].final_norm.load_state_dict(full.final_norm.state_dict())
    x = torch.randn(8, 2, cfg.hidden_size)
    expected, _ = _run(full, x.clone().requires_grad_(), cfg, groups)
    actual, _ = _run(checkpointed, x.clone().requires_grad_(), checkpointed.config, groups)
    torch.testing.assert_close(actual, expected)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for reference, recomputed in zip(full.parameters(), checkpointed.parameters()):
        torch.testing.assert_close(recomputed.grad, reference.grad)
    hidden, context = _run(stages[0], x.clone().requires_grad_(), cfg, groups)
    sent = context.codec.payload(hidden, context.values, 2)
    assert sent.descriptor == context.codec.descriptor(2)
    received = TensorStatePayload(
        tuple(
            t.detach().clone().requires_grad_(f.requires_grad)
            for t, f in zip(sent.tensors, sent.tensor_specs)
        ),
        sent.spec,
    )
    hidden, values = received.restore()
    actual, _ = _run(stages[1], hidden, cfg, groups, values)
    torch.testing.assert_close(actual, expected)
    grads = backward_pipeline_payload(received, actual.square().mean(), None)
    backward_pipeline_payload(None, sent, grads)
    for stage, offset in zip(stages, (0, 2)):
        for i, layer in enumerate(stage.layers):
            for ref, split in zip(full.layers[offset + i].parameters(), layer.parameters()):
                torch.testing.assert_close(split.grad, ref.grad)
