# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core.enums import Fp8Recipe
from megatron.core.fp8_utils import get_fp8_context, get_fp8_recipe
from megatron.core.models.hybrid.layers.hybrid_hyper_connection import HyperConnectionHybridLayer
from megatron.core.pipeline_parallel.parallel_prewarm import (
    _build_layer_inputs,
    _build_packed_seq_params,
    prewarm_pipeline_model_parallel,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import (
    get_cuda_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.enums import AttnMaskType, CudaGraphModule
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
)
from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.test_utilities import Utils


def _make_static_layer(**overrides):
    # These tests exercise input construction without requiring CUDA or process groups.
    layer = object.__new__(TransformerLayer)
    torch.nn.Module.__init__(layer)
    config = dict(
        bf16=True,
        fp16=False,
        hidden_size=4,
        context_parallel_size=1,
        cp_partition_mode='zigzag',
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        create_attention_mask_in_dataloader=False,
        cuda_graph_impl='none',
        cuda_graph_modules=[],
        max_seqlen_per_dp_cp_rank=8,
        sequence_packing_scheduler=object(),
        thd_max_packed_sequences=2,
    )
    config.update(overrides)
    layer.config = SimpleNamespace(**config)
    layer.self_attention = SimpleNamespace(attn_mask_type=AttnMaskType.causal)
    layer.is_moe_layer = False
    layer._padding_mask_shape_seen = None
    return layer


@pytest.mark.parametrize('cuda_graph_impl', ['none', 'local', 'transformer_engine'])
def test_prewarm_thd_metadata_aliases_qk(monkeypatch, cuda_graph_impl):
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: torch.device('cpu'))
    layer = _make_static_layer(cuda_graph_impl=cuda_graph_impl, context_parallel_size=2)
    inputs = layer.get_layer_static_inputs(16, 2, for_pipeline_prewarm=True)
    assert inputs['hidden_states'].shape == (8, 1, 4)
    assert inputs['cu_seqlens_q'].tolist() == [0, 16, 16]
    assert inputs['cu_seqlens_q'] is inputs['cu_seqlens_kv']
    assert inputs['cu_seqlens_q_padded'] is inputs['cu_seqlens_kv_padded']
    assert inputs['cu_seqlens_q'] is not inputs['cu_seqlens_q_padded']

    _build_packed_seq_params(layer.config, inputs)
    packed = inputs['packed_seq_params']
    assert packed.cu_seqlens_q is packed.cu_seqlens_kv
    assert packed.cu_seqlens_q_padded is packed.cu_seqlens_kv_padded
    assert packed.max_seqlen_q == packed.max_seqlen_kv == 16
    assert packed.qkv_format == 'thd'
    assert not any(key.startswith('cu_seqlens') for key in inputs)


@pytest.mark.parametrize('cuda_graph_impl', ['local', 'transformer_engine'])
def test_graph_thd_metadata_remains_independently_writable(monkeypatch, cuda_graph_impl):
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: torch.device('cpu'))
    inputs = _make_static_layer(cuda_graph_impl=cuda_graph_impl).get_layer_static_inputs(8, 1)
    metadata = [value for key, value in inputs.items() if key.startswith('cu_seqlens')]
    assert len(metadata) == len({tensor.data_ptr() for tensor in metadata}) == 4
    metadata[0][1] = 4
    assert all(tensor[1].item() == 8 for tensor in metadata[1:])


@pytest.mark.parametrize('prewarm', [False, True])
@pytest.mark.parametrize('cp_size', [1, 2])
def test_sbhd_mask_allocated_as_bool_on_target_device(monkeypatch, prewarm, cp_size):
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: torch.device('cpu'))
    ones = Mock(wraps=torch.ones)
    monkeypatch.setattr(torch, 'ones', ones)
    layer = _make_static_layer(
        context_parallel_size=cp_size,
        sequence_packing_scheduler=None,
        create_attention_mask_in_dataloader=True,
    )
    inputs = layer.get_layer_static_inputs(8, 2, for_pipeline_prewarm=prewarm)
    mask = inputs['attention_mask']
    expected = torch.arange(8).view(1, -1) > torch.arange(8 // cp_size).view(-1, 1)
    assert mask.dtype == torch.bool
    assert mask.shape == (2, 1, 8 // cp_size, 8)
    assert mask.is_contiguous()
    assert torch.equal(mask[0, 0], expected)
    assert torch.equal(mask[1, 0], expected)
    # No temporary FP32 mask on the host, and no separate matrix to tile over the batch.
    mask_allocations = [
        call for call in ones.call_args_list if not call.kwargs.get('requires_grad')
    ]
    assert len(mask_allocations) == 1
    assert mask_allocations[0].kwargs['dtype'] == torch.bool
    assert mask_allocations[0].kwargs['device'] == torch.device('cpu')


def test_sbhd_prewarm_omits_mask_when_disabled(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: torch.device('cpu'))
    layer = _make_static_layer(sequence_packing_scheduler=None)
    inputs = layer.get_layer_static_inputs(8, 2, for_pipeline_prewarm=True)
    assert inputs['hidden_states'].shape == (8, 2, 4)
    assert 'attention_mask' not in inputs
    assert 'cu_seqlens_q' not in inputs


@pytest.mark.parametrize('prewarm', [False, True])
@pytest.mark.parametrize('packed', [False, True])
def test_linear_attention_omits_mask_but_keeps_thd_metadata(monkeypatch, prewarm, packed):
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: torch.device('cpu'))
    layer = _make_static_layer(
        cuda_graph_impl='local',
        sequence_packing_scheduler=object() if packed else None,
        create_attention_mask_in_dataloader=True,
    )
    # Linear attention has no attn_mask_type, but still consumes THD sequence boundaries.
    layer.self_attention = SimpleNamespace(uses_attention_mask=False)
    inputs = layer.get_layer_static_inputs(8, 2, for_pipeline_prewarm=prewarm)
    assert 'attention_mask' not in inputs
    assert ('cu_seqlens_q' in inputs) == packed


@pytest.mark.parametrize('prewarm', [False, True])
def test_hash_router_ids_follow_local_sbhd_tokens(monkeypatch, prewarm):
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: torch.device('cpu'))
    layer = _make_static_layer(
        context_parallel_size=2,
        tensor_model_parallel_size=2,
        sequence_parallel=True,
        sequence_packing_scheduler=None,
    )
    layer.is_moe_layer = True
    layer.mlp = SimpleNamespace(router=SimpleNamespace(is_hash_layer=True))
    inputs = layer.get_layer_static_inputs(8, 2, for_pipeline_prewarm=prewarm)
    assert inputs['hidden_states'].shape == (2, 2, 4)
    assert inputs['input_ids'].shape == (2, 2)


@pytest.mark.parametrize('cp_rank', [0, 1])
def test_dynamic_dsa_route_is_preserved_in_eager_prewarm_metadata(monkeypatch, cp_rank):
    from megatron.core.transformer.experimental_attention_variant import cp_balanced_indexer

    monkeypatch.setattr(torch.cuda, 'current_device', lambda: torch.device('cpu'))
    layer = _make_static_layer(
        cuda_graph_impl='transformer_engine',
        context_parallel_size=2,
        dsa_cp_balance_indexer_graph_dynamic_packs=True,
    )
    layer.self_attention.indexer = object()
    layer.pg_collection = SimpleNamespace(cp=SimpleNamespace(size=lambda: 2, rank=lambda: cp_rank))
    graph_inputs = layer.get_layer_static_inputs(16, 1)
    _, prewarm_inputs = _build_layer_inputs(layer, SimpleNamespace(), layer.config, 16, 1, {})
    for key in ('dsa_cp_graph_layout_buffer', 'dsa_cp_graph_route_buffer'):
        assert torch.is_tensor(graph_inputs[key])
        assert key not in prewarm_inputs
    packed = prewarm_inputs['packed_seq_params']
    assert packed.cu_seqlens_q.tolist() == [0, 16, 16]
    assert packed.cu_seqlens_q is packed.cu_seqlens_kv
    assert packed.cu_seqlens_q_padded is packed.cu_seqlens_kv_padded
    plan = cp_balanced_indexer.get_graph_dynamic_plan(packed)
    assert plan is not None
    assert (plan['cp_size'], plan['l_local']) == (2, 8)
    # Tensor-only reconstruction omits the host cp_rank tag. The route consumed by
    # eager dispatch must still match the plan built for this layer's explicit CP rank.
    expected = cp_balanced_indexer.build_graph_dynamic_plan(
        packed.cu_seqlens_q_padded, layer.pg_collection.cp, 16
    )
    for key in ('validated_cu', 'pos_head', 'pos_tail', 'src_slot', 'relay_perm', 'dst_slot'):
        assert torch.equal(plan[key], expected[key])


@pytest.mark.parametrize('split_attention', [False, True])
def test_hybrid_mhc_prewarm_uses_full_residual_streams(monkeypatch, split_attention):
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: torch.device('cpu'))
    inner = _make_static_layer(
        cuda_graph_impl='transformer_engine',
        cuda_graph_modules=[CudaGraphModule.attn],
        mhc_recompute_attn_cuda_graph_split=split_attention,
        recompute_granularity='selective',
        recompute_modules=['mhc'],
        num_residual_streams=4,
    )
    layer = object.__new__(HyperConnectionHybridLayer)
    torch.nn.Module.__init__(layer)
    layer.inner_layer = inner
    layer.config = inner.config
    graph_inputs = layer.get_layer_static_inputs(8, 2)
    prewarm_inputs = layer.get_layer_static_inputs(8, 2, for_pipeline_prewarm=True)
    assert graph_inputs['hidden_states'].shape == (8, 1, 4 if split_attention else 16)
    assert prewarm_inputs['hidden_states'].shape == (8, 1, 16)
    assert prewarm_inputs['cu_seqlens_q'] is prewarm_inputs['cu_seqlens_kv']


def test_prewarm_requires_multiple_pipeline_stages():
    with pytest.raises(AssertionError, match='pipeline_model_parallel_size'):
        TransformerConfig(
            num_layers=2,
            hidden_size=16,
            num_attention_heads=2,
            pipeline_model_parallel_prewarm=True,
        )


@pytest.fixture
def pp_groups():
    if Utils.world_size < 2 or torch.cuda.device_count() < 2:
        pytest.skip('State restoration tests require at least two GPUs for PP=2')
    Utils.initialize_model_parallel(1, 2)
    model_parallel_cuda_manual_seed(123)
    yield ProcessGroupCollection.use_mpu_process_groups()
    Utils.destroy_model_parallel()


class _StatefulLayer(torch.nn.Module):
    """Use real TE kernels while making prewarm's restorable state observable."""

    def __init__(self, config):
        super().__init__()
        import transformer_engine.pytorch as te

        self.config = config
        self.layer_number = 1
        self.is_first_microbatch = False
        self.linear = te.Linear(16, 16, params_dtype=torch.bfloat16)
        self.linear.is_first_microbatch = False
        self.register_buffer('counter', torch.tensor(3.0, device='cuda'))
        self.mutate_state = False
        self.fail_at = None

    def get_layer_static_inputs(self, seq_length, micro_batch_size, *, for_pipeline_prewarm=False):
        assert for_pipeline_prewarm
        return {
            'hidden_states': torch.ones(
                seq_length,
                micro_batch_size,
                16,
                device='cuda',
                dtype=torch.bfloat16,
                requires_grad=True,
            )
        }

    def forward(self, hidden_states):
        output = self.linear(hidden_states)
        if self.mutate_state:
            assert self.is_first_microbatch and self.linear.is_first_microbatch
            self.counter.add_(1)
            self.is_first_microbatch = False
            self.linear.is_first_microbatch = False
            torch.rand(4)
            output = torch.nn.functional.dropout(output, p=0.25, training=True)
            with get_cuda_rng_tracker().fork():
                output = torch.nn.functional.dropout(output, p=0.25, training=True)
            for scaler in (DSAIndexerLossAutoScaler, MoEAuxLossAutoScaler):
                scaler.set_loss_scale(torch.tensor(9.0, device='cuda'))
            DSAIndexerLossLoggingHelper.tracker['values'].add_(1)
            DSAIndexerLossLoggingHelper.tracker['temporary'] = torch.tensor(1.0, device='cuda')
            for param in self.parameters():
                param.main_grad.add_(1)
            if self.fail_at == 'forward':
                raise RuntimeError('synthetic forward failure')
            if self.fail_at == 'backward':

                def fail_backward(grad):
                    raise RuntimeError('synthetic backward failure')

                output.register_hook(fail_backward)
        return {'output': output, 'ignored': (None, output.detach())}


class _ModelChunk(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.decoder = torch.nn.Module()
        self.decoder.layers = torch.nn.ModuleList(layers)
        self.inside_no_sync = False
        for param in self.parameters():
            param.main_grad = torch.zeros_like(param, dtype=torch.float32)

    @contextmanager
    def no_sync(self):
        self.inside_no_sync = True
        try:
            yield
        finally:
            self.inside_no_sync = False

    def zero_grad_buffer(self):
        for param in self.parameters():
            param.main_grad.zero_()


def _assert_same_tensors(actual, expected):
    if torch.is_tensor(expected):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_same_tensors(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for value, reference in zip(actual, expected):
            _assert_same_tensors(value, reference)
    else:
        assert actual == expected


@pytest.mark.parametrize('fp8', [False, True])
@pytest.mark.parametrize('initial_loss_scale', [None, 7.0])
@pytest.mark.parametrize('fail_at', [None, 'forward', 'backward'])
def test_prewarm_restores_state(pp_groups, monkeypatch, fp8, initial_loss_scale, fail_at):
    from transformer_engine.pytorch.graph import save_fp8_tensors

    config = TransformerConfig(
        num_layers=2,
        hidden_size=16,
        num_attention_heads=2,
        pipeline_model_parallel_size=2,
        pipeline_dtype=torch.bfloat16,
        pipeline_model_parallel_prewarm=True,
        bf16=True,
        fp8='hybrid' if fp8 else None,
        fp8_recipe=Fp8Recipe.delayed,
    )
    layer = _StatefulLayer(config)
    chunk = _ModelChunk([layer])
    optimizer = torch.optim.SGD(chunk.parameters(), lr=0.1)
    optimizer.zero_grad = Mock(wraps=optimizer.zero_grad)
    inputs = layer.get_layer_static_inputs(16, 1, for_pipeline_prewarm=True)
    # Initialize TE's delayed-scaling state before taking the snapshot.
    with get_fp8_context(config):
        layer(**inputs)['output'].sum().backward()
    optimizer.zero_grad()
    optimizer.zero_grad.reset_mock()
    recipe = get_fp8_recipe(config) if fp8 else None
    quantization_before = save_fp8_tensors([layer], recipe)
    if fp8:
        assert quantization_before is not None
        assert any(state is not None for state in quantization_before)

    scales = []
    for scaler in (DSAIndexerLossAutoScaler, MoEAuxLossAutoScaler):
        scale = (
            torch.tensor(initial_loss_scale, device='cuda')
            if initial_loss_scale is not None
            else None
        )
        monkeypatch.setattr(scaler, 'main_loss_backward_scale', scale)
        scales.append((scaler, scale))
    tracker = {'values': torch.tensor([2.0], device='cuda'), 'reduce_group': pp_groups.tp}
    monkeypatch.setattr(DSAIndexerLossLoggingHelper, 'tracker', tracker)
    parameters_before = {name: param.detach().clone() for name, param in chunk.named_parameters()}
    buffers_before = {name: buffer.clone() for name, buffer in chunk.named_buffers()}
    cpu_rng_before = torch.get_rng_state()
    cuda_rng_before = torch.cuda.get_rng_state()
    tracker_rng_before = {
        name: state.clone() for name, state in get_cuda_rng_tracker().get_states().items()
    }
    layer.mutate_state = True
    layer.fail_at = fail_at
    expectation = (
        pytest.raises(RuntimeError, match=f'synthetic {fail_at} failure')
        if fail_at is not None
        else nullcontext()
    )
    with expectation:
        prewarm_pipeline_model_parallel(
            [chunk], config, 16, 1, optimizers=[optimizer], pg_collection=pp_groups
        )

    _assert_same_tensors(dict(chunk.named_parameters()), parameters_before)
    _assert_same_tensors(dict(chunk.named_buffers()), buffers_before)
    _assert_same_tensors(save_fp8_tensors([layer], recipe), quantization_before)
    assert torch.equal(torch.get_rng_state(), cpu_rng_before)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_rng_before)
    _assert_same_tensors(get_cuda_rng_tracker().get_states(), tracker_rng_before)
    assert layer.is_first_microbatch is False
    assert layer.linear.is_first_microbatch is False
    assert not chunk.inside_no_sync
    for param in chunk.parameters():
        assert param.grad is None
        assert torch.count_nonzero(param.main_grad) == 0
    optimizer.zero_grad.assert_called_once_with()
    assert DSAIndexerLossLoggingHelper.tracker is tracker
    assert set(tracker) == {'values', 'reduce_group'}
    assert tracker['values'].item() == 2.0
    assert tracker['reduce_group'] is pp_groups.tp
    for scaler, scale in scales:
        assert scaler.main_loss_backward_scale is scale
        if scale is not None:
            assert scale.item() == initial_loss_scale


def test_empty_stage_warns_and_participates_in_barriers(pp_groups, caplog, monkeypatch):
    config = TransformerConfig(
        num_layers=2,
        hidden_size=16,
        num_attention_heads=2,
        pipeline_model_parallel_size=2,
        pipeline_dtype=torch.bfloat16,
        pipeline_model_parallel_prewarm=True,
    )
    barrier = Mock(wraps=torch.distributed.barrier)
    monkeypatch.setattr(torch.distributed, 'barrier', barrier)
    prewarm_pipeline_model_parallel([_ModelChunk([])], config, 16, 1, pg_collection=pp_groups)
    assert 'found no local decoder or MTP layers' in caplog.text
    assert barrier.call_count == 2
