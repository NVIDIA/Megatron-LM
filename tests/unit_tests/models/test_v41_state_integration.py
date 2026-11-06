# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Run main's HybridStack with real CSA2/mHC and CPU projection substitutes."""

import copy
import sys
from dataclasses import fields, replace
from types import ModuleType, SimpleNamespace

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
from megatron.core.transformer.state_boundary import TensorField
from megatron.core.transformer.stateful_module import StatefulGraphs, StatefulModule
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
    if not torch.cuda.is_available():
        from megatron.core.tensor_parallel import random as rng

        # These cases use CPU tensors and zero dropout; preserve the real CPU
        # RNG while making checkpoint's otherwise-unused CUDA snapshot optional.
        monkeypatch.setattr(rng, '_get_cuda_rng_state', lambda **kw: torch.empty(0))
        monkeypatch.setattr(rng, '_set_cuda_rng_state', lambda *a, **kw: None)
        if not rng.HAVE_TE:
            checkpoint_init = rng.CheckpointWithoutOutput.__init__

            def cpu_checkpoint_init(self, *args, **kwargs):
                checkpoint_init(self, *args, **kwargs)
                # These FP32 CPU cases do not use TE's FP8 bookkeeping contexts.
                self.fp8 = False

            monkeypatch.setattr(rng.CheckpointWithoutOutput, '__init__', cpu_checkpoint_init)
    monkeypatch.setattr(attention, 'YarnRotaryEmbedding', CpuYarn)
    monkeypatch.setattr(hybrid_block, 'TENorm', Norm)
    monkeypatch.setattr(DSAIndexerLossLoggingHelper, 'save_loss_to_tracker', lambda **kw: None)
    monkeypatch.setattr(DSAIndexerLossAutoScaler, 'main_loss_backward_scale', torch.tensor(1.0))
    cpu_attention = get_csa2_module_spec_for_backend(Backend())
    # Exercise CPU orchestration even when the installed TE exports its CUDA-only op.
    cpu_attention.submodules = replace(cpu_attention.submodules, linear_o_group_proj=None)
    attn = ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=Norm, self_attention=cpu_attention, self_attn_bda=get_bias_dropout_add
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


def test_restore_reuses_previous_packed_layouts(setup):
    """A later boundary of the same microbatch reuses its packed layouts, equal to a rebuild."""
    cfg, _, groups = setup
    codec = V41StateCodec(cfg, BatchShape(16, 1, max_seqlen=10, num_sequences=2), groups.cp)
    values = dict(
        cu=torch.tensor([0, 9, 14], dtype=torch.int32),
        cu_padded=torch.tensor([0, 10, 16], dtype=torch.int32),
    )
    first, _, _ = codec.restore(values, 1)
    reused, _, _ = codec.restore(values, 3, previous=first)
    rebuilt, _, _ = codec.restore(values, 3)
    assert first.thd_layout is not None and first.compressed_layout is not None
    assert reused.thd_layout is first.thd_layout
    assert reused.compressed_layout is first.compressed_layout
    for name in ('thd_layout', 'compressed_layout'):
        actual, expected = getattr(reused, name), getattr(rebuilt, name)
        assert actual is not expected
        for item in fields(expected):
            value, reference = getattr(actual, item.name), getattr(expected, item.name)
            if isinstance(reference, torch.Tensor):
                assert value.dtype == reference.dtype, (name, item.name)
                assert torch.equal(value, reference), (name, item.name)
            else:
                assert value == reference, (name, item.name)


def test_prepare_forward_keeps_the_model_restored_state(setup):
    """The stack restores its start boundary only when the model wrapper has not."""
    cfg, spec, groups = setup
    adapter = _stack(cfg, spec, groups).forward_adapter
    codec = V41StateCodec(cfg, BatchShape(8, 2), groups.cp)
    csa, mhc, _ = codec.restore({}, adapter.start)
    context = V41ForwardContext(cross_layer_state=csa, mhc_state=mhc, codec=codec, values={})
    assert adapter.prepare_forward(None, None, None, context) is context
    assert context.cross_layer_state is csa and context.mhc_state is mhc
    fresh = V41ForwardContext(codec=codec, values={})
    assert adapter.prepare_forward(None, None, None, fresh) is fresh
    assert fresh.cross_layer_state is not None and fresh.cross_layer_state is not csa


def test_full_iteration_graphs_reject_engram():
    """V4.1 accepts full-iteration graphs, which need no single-pass mHC state adapter, but not
    with Engram, whose EP lookup plans its exchange on the host per batch."""
    from megatron.core.models.deepseek_v41.config import EngramConfig

    cfg = _config()
    values = {f.name: getattr(cfg, f.name) for f in fields(cfg) if f.init}
    values.update(cuda_graph_impl='full_iteration', cuda_graph_modules=[])
    assert DeepSeekV41Config(**values).cuda_graph_impl == 'full_iteration'
    values['engram_config'] = EngramConfig(
        layer_ids=[0],
        num_embeddings=[96],
        max_ngram_size=2,
        vocab_size=17,
        n_heads=2,
        head_dim=4,
        pad_token_id=2,
        compressed_vocab_size=128,
    )
    with pytest.raises(NotImplementedError, match='full_iteration'):
        DeepSeekV41Config(**values)
    values['cuda_graph_impl'] = 'none'
    assert DeepSeekV41Config(**values).engram_config is values['engram_config']


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
        self.first_microbatches = []
        names = tuple(name for name, _ in region.named_parameters())
        assert not any('hyper_connection' in name or 'experts' in name for name in names)

    def run(self, hidden, values, **kwargs):
        self.first_microbatches.append(kwargs.get('is_first_microbatch'))
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


@pytest.mark.parametrize('selective', [False, True])
@pytest.mark.parametrize('scopes', [('attn',), ('attn', 'moe_router')])
@pytest.mark.parametrize('overlap_param_gather', [None, False, True])
def test_cli_captures_warmed_consumers_together_between_steps(
    setup, monkeypatch, selective, scopes, overlap_param_gather
):
    """Exercise warmup, batch installation and replay with real CPU CSA2/mHC math."""
    from megatron.core.distributed import DistributedDataParallelConfig
    from megatron.core.pipeline_parallel import schedules
    from megatron.core.transformer import cuda_graphs

    cfg, spec, groups = setup
    cfg = replace(
        cfg,
        num_moe_experts=2,
        cuda_graph_impl='transformer_engine',
        cuda_graph_modules=list(scopes),
    )
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
    adapter = actual.forward_adapter
    adapter.record_cuda_graphs()
    selected = 3 * len(scopes)

    def compare_step():
        reference.zero_grad(set_to_none=True)
        actual.zero_grad(set_to_none=True)
        for microbatch in range(3):
            for stack in (reference, actual):
                cuda_graphs.set_current_microbatch(SimpleNamespace(decoder=stack), microbatch)
            x = torch.randn(8, 1, cfg.hidden_size, requires_grad=True)
            y = x.detach().clone().requires_grad_()
            expected = _run(reference, x, cfg, groups)[0]
            observed = _run(actual, y, cfg, groups)[0]
            torch.testing.assert_close(observed, expected)
            expected.square().mean().backward()
            observed.square().mean().backward()
            torch.testing.assert_close(y.grad, x.grad)
        for (name, p), (_, q) in zip(reference.named_parameters(), actual.named_parameters()):
            torch.testing.assert_close(q.grad, p.grad, msg=name)

    compare_step()
    assert not adapter.graphs
    assert len(adapter._capture_samples) == selected
    for _, hidden, values, _ in adapter._capture_samples.values():
        assert hidden.is_leaf
        assert all(value.is_leaf for value in values.values() if value is not None)

    helper = object.__new__(cuda_graphs.TECudaGraphHelper)
    helper.stateful_graph_adapters = [(0, adapter)]
    helper.pp_group = SimpleNamespace(size=lambda: 1)
    helper.tp_group = groups.tp
    helper.num_model_chunks = 1
    helper.config = cfg
    helper.p2p_communicator = None
    helper.flattened_callables = []
    assert not hasattr(cfg, 'overlap_param_gather')

    def make_forward_pre_hook():
        return lambda module, *args: None

    helper.model = [
        (
            actual
            if overlap_param_gather is None
            else SimpleNamespace(
                ddp_config=DistributedDataParallelConfig(overlap_param_gather=overlap_param_gather),
                _make_forward_pre_hook=make_forward_pre_hook,
            )
        )
    ]
    # Exercise the public capture entry point, stubbing only its CUDA lifecycle.
    monkeypatch.setattr(helper, '_start_capturing', lambda: None)
    monkeypatch.setattr(helper, '_finish_capturing', lambda start: None)
    monkeypatch.setattr(cuda_graphs, 'get_num_microbatches', lambda: 2)
    monkeypatch.setattr(schedules, 'get_pp_rank_microbatches', lambda *a, **kw: (0, 0, 0, 0))
    monkeypatch.setattr(schedules, 'get_schedule_table', lambda *a: [(0, 0)])
    calls = []

    def capture_many(profiles, **kwargs):
        calls.append(kwargs)
        expected_hook = make_forward_pre_hook if overlap_param_gather else None
        assert all(hook is expected_hook for _, _, _, hook in profiles)
        return [EagerGraphSlots(region, hidden, values) for region, hidden, values, _ in profiles]

    monkeypatch.setattr(StatefulGraphs, 'capture_many', capture_many)
    helper.create_cudagraphs()
    assert len(calls) == 1
    assert calls[0]['order'] == [1, -1]
    assert calls[0]['num_layers_per_chunk'] == [selected]
    assert calls[0]['preserve_gradients'] is False
    assert calls[0]['clone_inputs'] is False
    assert helper.graphs_created()
    assert not adapter._capture_samples
    compare_step()
    compare_step()
    for graph in adapter.graphs.values():
        # PP1 reuses slot zero for every microbatch, including non-first ones.
        assert graph.first_microbatches == [True, False, False] * 2
    assert len(calls) == 1
    with pytest.raises(ValueError, match='warmed batch shape'):
        _run(actual, torch.randn(10, 1, cfg.hidden_size, requires_grad=True), cfg, groups)
    assert len(adapter.graphs) == selected


@pytest.mark.parametrize('backend', ['torch', 'transformer_engine'])
def test_stateful_replay_passes_te_microbatch_flag_without_changing_tensor_inputs(
    monkeypatch, backend
):
    """Check TE's replay keyword contract on CPU, including slot reuse across steps."""

    class Region(nn.Module):
        def forward(self, hidden, state):
            return hidden.square(), state

    region = StatefulModule(Region())
    sample = torch.ones(4, requires_grad=True)
    owner = object.__new__(StatefulGraphs)
    owner.region, owner.device = region, sample.device
    owner._backend = backend
    owner._output_wrapper = None
    owner.debug_checks = False
    owner._signature = (owner._tensor_signature(sample),)
    owner._busy, owner._generation, owner._recorded = [False], [0], [False]
    owner._manual_forward_hooks, owner._fused_grad_params = [], [()]
    flags = []

    def graphed(hidden, **kwargs):
        if backend == 'transformer_engine':
            assert isinstance(kwargs.get('is_first_microbatch'), bool)
            flags.append(kwargs['is_first_microbatch'])
        else:
            assert not kwargs, 'PyTorch graph wrappers accept only the captured tensor inputs'
        return region(hidden)

    owner._callables = [graphed]
    monkeypatch.setattr(owner, '_release', lambda slot: owner._busy.__setitem__(slot, False))
    for _ in range(2):
        for microbatch in range(3):
            hidden = sample.detach().clone().requires_grad_()
            output, state = owner.run(hidden, {}, is_first_microbatch=microbatch == 0)
            assert state == {}
            output.sum().backward()
            torch.testing.assert_close(hidden.grad, 2 * hidden)
            assert not owner._busy[0]
    assert flags == ([True, False, False] * 2 if backend == 'transformer_engine' else [])


@pytest.mark.parametrize('layout', ['extra', 'reordered', 'frozen', 'failure'])
def test_shared_graph_preserves_output_slots_with_legacy_te_buffer_reuse(monkeypatch, layout):
    """Exercise capture_many/run with TE's old cache key and real CPU autograd."""
    keys = ['ids', 'grad_a', 'readonly', 'grad_b', 'counts', 'candidates']
    other_keys = keys.copy()
    if layout == 'reordered':
        other_keys[0], other_keys[1] = other_keys[1], other_keys[0]
    else:
        other_keys.insert(4, 'extra_ids')

    class Region(nn.Module):
        def __init__(self, names):
            super().__init__()
            self.names = names
            self.weight = nn.Parameter(torch.tensor(0.75))

        def forward(self, hidden, state):
            output = hidden * self.weight
            values = {}
            for name in self.names:
                if name.startswith('grad_'):
                    value = output * (2 if name == 'grad_a' else 3)
                    if layout == 'frozen' and name == 'grad_b':
                        value = value.detach()
                elif name == 'readonly' and layout == 'frozen':
                    value = hidden.detach() + 17
                else:
                    value = torch.arange(hidden.numel(), device=hidden.device, dtype=torch.int32)
                values[name] = value
            return output, values

    def region(names):
        fields = tuple(
            TensorField(
                name,
                (4,),
                (
                    torch.float32
                    if name.startswith('grad_') or (name == 'readonly' and layout == 'frozen')
                    else torch.int32
                ),
                'contiguous',
                name.startswith('grad_'),
            )
            for name in names
        )
        return StatefulModule(Region(names), output_fields=fields)

    regions = [region(names) for names in (other_keys, keys)]
    references = copy.deepcopy(regions)
    sample = torch.ones(4, requires_grad=True)
    cache_entries = []
    original_calls = {}

    class LegacyTEReplay(torch.autograd.Function):
        @staticmethod
        def forward(ctx, grad_slots, *outputs):
            ctx.grad_slots = grad_slots
            ctx.active = tuple(t is not None and t.requires_grad for t in outputs)
            return tuple(t.detach() if t is not None else None for t in outputs)

        @staticmethod
        def backward(ctx, *grads):
            if len(grads) != len(ctx.grad_slots):
                raise ValueError(
                    f'Backward graph grad dimension mismatch: received {len(grads)} grads, '
                    f'but expected {len(ctx.grad_slots)} static_grad_outputs'
                )
            if ctx.active != ctx.grad_slots:
                raise ValueError('Backward graph gradient slot positions mismatch')
            return (None, *grads)

    def capture(wrappers, args, **kwargs):
        assert kwargs['_reuse_graph_input_output_buffers'] is True
        if layout == 'failure':
            # TE temporarily replaces the capture module class's __call__.
            # An exception must not leave that replacement installed.
            cls = type(wrappers[0])
            original_calls[cls] = cls.__dict__.get('__call__')
            cls.__call__ = lambda *a, **kw: None
            raise RuntimeError('intentional shared capture failure')
        outputs = [wrapper(*inputs) for wrapper, inputs in zip(wrappers, args)]
        cache, slots = {}, [None] * len(wrappers)
        # TE captures each chunk's backwards in reverse layer order. Its old
        # key drops nondifferentiable slots; the cached value retains them.
        for i in reversed(range(len(wrappers))):
            key = tuple(
                (t.shape, t.dtype, t.layout)
                for t in outputs[i]
                if t is not None and t.requires_grad
            )
            slots[i] = cache.setdefault(
                key, tuple(t is not None and t.requires_grad for t in outputs[i])
            )
        cache_entries.append(len(cache))
        return [
            lambda *inputs, wrapper=wrapper, slots=slots, **kw: LegacyTEReplay.apply(
                slots, *wrapper(*inputs)
            )
            for wrapper, slots in zip(wrappers, slots)
        ]

    te = ModuleType('transformer_engine')
    te.pytorch = ModuleType('transformer_engine.pytorch')
    te.pytorch.graph = SimpleNamespace(
        is_graph_capturing=lambda: False, set_capture_end=lambda: None
    )
    te.pytorch.make_graphed_callables = capture
    monkeypatch.setitem(sys.modules, 'transformer_engine', te)
    monkeypatch.setitem(sys.modules, 'transformer_engine.pytorch', te.pytorch)

    def cpu_owner(self, region, hidden, state, *, slots, backend, **kwargs):
        assert kwargs['_defer_capture']
        self.region, self.device, self._backend = region, hidden.device, backend
        self._output_wrapper = None
        self.debug_checks = False
        self._signature = (self._tensor_signature(hidden),)
        self._busy, self._generation, self._recorded = [False] * slots, [0] * slots, [False] * slots
        self._wrappers, self._manual_forward_hooks = [], []

    monkeypatch.setattr(StatefulGraphs, '__init__', cpu_owner)
    if layout == 'failure':
        try:
            with pytest.raises(RuntimeError, match='intentional shared capture failure'):
                StatefulGraphs.capture_many(
                    [(module, sample, {}, None) for module in regions],
                    order=[1, -1],
                    num_layers_per_chunk=[2],
                )
            for cls, original in original_calls.items():
                assert cls.__dict__.get('__call__') is original
        finally:
            for cls, original in original_calls.items():
                if original is None:
                    if '__call__' in cls.__dict__:
                        delattr(cls, '__call__')
                else:
                    cls.__call__ = original
        return
    owners = StatefulGraphs.capture_many(
        [(module, sample, {}, None) for module in regions], order=[1, -1], num_layers_per_chunk=[2]
    )
    assert cache_entries == [1], 'Compatible gradient buffers must still be shared'
    for owner in owners:
        monkeypatch.setattr(
            owner, '_release', lambda slot, owner=owner: owner._busy.__setitem__(slot, False)
        )
    for _ in range(2):
        for owner, reference in zip(owners, references):
            owner.region.zero_grad(set_to_none=True)
            reference.zero_grad(set_to_none=True)
            x = torch.randn(4, requires_grad=True)
            y = x.detach().clone().requires_grad_()
            actual, state = owner.run(x, {})
            expected, ref_state = reference.run(y, {})
            assert tuple(state) == tuple(ref_state)
            for key in state:
                torch.testing.assert_close(state[key], ref_state[key])
                assert state[key].requires_grad == ref_state[key].requires_grad
            torch.testing.assert_close(actual, expected)
            (actual.sum() + state['grad_a'].sum() + state['grad_b'].sum()).backward()
            (expected.sum() + ref_state['grad_a'].sum() + ref_state['grad_b'].sum()).backward()
            torch.testing.assert_close(x.grad, y.grad)
            torch.testing.assert_close(
                owner.region.module.weight.grad, reference.module.weight.grad
            )


def test_between_step_capture_discards_gradients_without_main_grad_backups(monkeypatch):
    """A completed-step capture must not duplicate model-sized FP32 grad buffers."""
    module = nn.Linear(4, 4, bias=False)
    module.register_buffer('calls', torch.tensor(17))
    parameter = module.weight
    parameter.main_grad = torch.ones_like(parameter)
    parameter.grad = torch.full_like(parameter, 3)
    previous_grad = parameter.grad
    clone = torch.Tensor.clone

    def reject_main_grad_copy(tensor, *args, **kwargs):
        assert tensor is not parameter.main_grad, 'Completed-step main_grad must not be backed up'
        return clone(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'clone', reject_main_grad_copy)
    owner = object.__new__(StatefulGraphs)
    owner.region = module
    with owner._capture_grad_state(None, preserve_gradients=False):
        parameter.main_grad.add_(7)
        module.calls.add_(3)
    assert parameter.grad is previous_grad
    assert torch.count_nonzero(parameter.main_grad) == 0
    assert module.calls.item() == 17


@pytest.mark.parametrize('counts,order', [([3], [1, -1]), ([1, 1], [1, 1, 2, -1, 2, -1, -2, -2])])
@pytest.mark.parametrize('clone_inputs', [False, True])
def test_scheduled_static_inputs_survive_all_outstanding_forwards(counts, order, clone_inputs):
    """Overwrite each incoming microbatch to detect any premature buffer reuse."""
    samples = [(torch.zeros(4, 8, requires_grad=True),) for _ in range(sum(counts))]
    args = StatefulGraphs._scheduled_sample_args(samples, order, counts, clone_inputs=clone_inputs)
    source_pointers = {sample[0].data_ptr() for sample in samples}
    captured_pointers = {sample[0].data_ptr() for sample in args}
    assert bool(source_pointers & captured_pointers) is not clone_inputs
    microbatches = len(order) // (2 * len(counts))
    offsets = [sum(counts[:i]) for i in range(len(counts))]
    forward_index = [0] * len(counts)
    pending = [[] for _ in counts]
    live_pointers, all_pointers = set(), set()
    max_live = 0
    for event, operation in enumerate(order):
        chunk = abs(operation) - 1
        if operation > 0:
            current = []
            for layer in range(counts[chunk]):
                index = offsets[chunk] * microbatches + forward_index[chunk] * counts[chunk] + layer
                tensor = args[index][0]
                pointer = tensor.data_ptr()
                assert pointer not in live_pointers
                with torch.no_grad():
                    tensor.fill_(event)
                current.append((tensor, event))
                live_pointers.add(pointer)
                all_pointers.add(pointer)
            pending[chunk].append(current)
            forward_index[chunk] += 1
            max_live = max(max_live, len(live_pointers))
        else:
            for tensor, expected in pending[chunk].pop(0):
                torch.testing.assert_close(tensor, torch.full_like(tensor, expected))
                live_pointers.remove(tensor.data_ptr())
    assert not live_pointers
    assert len(all_pointers) == max_live


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
        actual.decoder.forward_adapter.record_cuda_graphs()
    tokens = [torch.randint(0, 128, (1, 32), device='cuda') for _ in range(2)]
    hidden = [torch.randn(32, 1, 128, device='cuda', dtype=torch.bfloat16) for _ in range(2)]
    metrics = {'logits': [], 'input_gradient': [], 'parameter_gradient': []}
    try:
        for step in range(3 if backend == 'transformer_engine' else 2):
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
            if step == 0 and backend == 'transformer_engine':
                adapter = actual.decoder.forward_adapter
                profiles = adapter.prepare_cuda_graphs(slots=2, backend=backend)
                graphs = StatefulGraphs.capture_many(
                    [sample for _, sample in profiles],
                    order=[1, 1, -1, -1],
                    num_layers_per_chunk=[6],
                    capture_kwargs=dict(num_warmup_iters=1),
                )
                adapter.install_cuda_graphs(
                    [(key, graph) for (key, _), graph in zip(profiles, graphs)], slots=2
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


@pytest.mark.parametrize('method,num_layers', [('uniform', 4), ('block', 3)])
@pytest.mark.parametrize('defer', [False, True])
def test_engram_lookup_is_not_repeated_by_full_recompute(
    setup, monkeypatch, method, num_layers, defer
):
    from megatron.core.models.deepseek_v41.config import EngramConfig
    from megatron.core.models.deepseek_v41.engram import Engram
    from megatron.core.models.engram.distributed_embedding import EngramRuntime
    from tests.unit_tests.models.test_v41_engram import _hasher

    cfg, spec, groups = setup
    hasher = _hasher()
    cfg.engram_config = EngramConfig([0], list(hasher.layout.num_embeddings), 4, 11, 2, 3, 0, 20)
    full = _stack(cfg, spec, groups)
    full.layers[0].engram = Engram(cfg, hasher.layout, 0, groups)
    checkpointed = copy.deepcopy(full)
    checkpointed.config.recompute_granularity = 'full'
    checkpointed.config.recompute_method = method
    checkpointed.config.recompute_num_layers = num_layers
    runtime = EngramRuntime(1, defer_grad_transfer=defer)
    runtime.attach(checkpointed.layers[0].engram.embed)
    calls = []
    original = runtime.prefetch

    def tracked(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime, 'prefetch', tracked)
    tokens = torch.randint(0, 20, (2, 8))
    x = torch.randn(8, 2, cfg.hidden_size)
    expected, _ = _run(
        full, x.clone().requires_grad_(), cfg, groups, {'tokens': tokens}, hasher=hasher
    )
    actual, _ = _run(
        checkpointed,
        x.clone().requires_grad_(),
        checkpointed.config,
        groups,
        {'tokens': tokens},
        hasher=hasher,
    )
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    expected.square().mean().backward()
    runtime.finish_deferred_gradients()
    assert calls == [1]
    for ref, param in zip(full.parameters(), checkpointed.parameters()):
        torch.testing.assert_close(param.grad, ref.grad, rtol=2e-5, atol=2e-6)
