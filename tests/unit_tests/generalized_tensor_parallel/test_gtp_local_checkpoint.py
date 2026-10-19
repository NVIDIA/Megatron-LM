# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Local fused checkpoint fragments, optimizer state and topology changes."""

import math
from dataclasses import replace
from unittest import mock

import pytest
import torch
import torch.distributed as dist

from megatron.core import dist_checkpointing
from megatron.core import parallel_state as ps
from megatron.core.dist_checkpointing.gtp import _gtp_split_tensor_factory
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.dist_checkpointing.optimizer import (
    get_param_id_to_sharded_param_map,
    make_sharded_optimizer_tensor,
)
from megatron.core.ssm.utils import _split_tensor_factory
from megatron.core.transformer.mlp import apply_swiglu_sharded_factory
from megatron.core.utils import _make_gtp_logical_sharded_tensor
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (
    _torchrun_dist_init,
    reset_gtp_globals,
)


def _factory(kind, tp, gtp, rank, *, device='cuda', dtype=torch.float32, scribble=False):
    # Globally ordered section values, distinct across TP ranks and all sections.
    global_sizes = [12, 8, 12] if kind == 'kda' else [12, 12]
    full_parts = []
    cursor = 0
    for size in global_sizes:
        full_parts.append(
            torch.arange(cursor + 1, cursor + size * 3 + 1, device=device).reshape(size, 3)
        )
        cursor += size * 3
    tp_rank, gtp_rank = rank % (tp * gtp) // gtp, rank % gtp
    dp_rank = rank // (tp * gtp)
    logical = torch.cat([part.chunk(tp)[tp_rank] for part in full_parts]).to(dtype)
    rows = math.ceil(logical.shape[0] / (gtp * 8)) * 8 if gtp > 1 else logical.shape[0]
    start = rows * gtp_rank
    valid = max(0, min(rows, logical.shape[0] - start))
    expected = logical.new_zeros(rows, 3)
    expected[:valid] = logical[start : start + valid]
    weight = torch.nn.Parameter(expected.clone())
    if scribble:
        with torch.no_grad():
            weight.fill_(-999)
    # Actual EP2 ownership is covered by the module integration test.
    offsets = ((0, 0, 1),) if kind == 'expert' else ()
    key = 'layers.0.in_proj.weight' if kind == 'kda' else 'layers.0.fc1.weight'
    if gtp > 1:
        entry = _make_gtp_logical_sharded_tensor(
            weight,
            key,
            tp_axis=0,
            tp_rank=tp_rank,
            tp_size=tp,
            gtp_rank=gtp_rank,
            gtp_remat_size=gtp,
            pad_length=rows * gtp - logical.shape[0],
            prepend_offsets=offsets,
            prepend_axis_num=len(offsets),
            other_offsets=(),
            replica_id=(0, 0, dp_rank),
        )
        factory = _gtp_split_tensor_factory(
            entry,
            [size // tp for size in global_sizes],
            ['query', 'key', 'value'] if kind == 'kda' else ['w', 'v'],
            gtp_rank,
            tp_rank,
            tp,
            weight=weight,
            glu=kind != 'kda',
        )
    else:
        from megatron.core.dist_checkpointing.gtp import attach_optimizer_factory

        entry = ShardedTensor.from_rank_offsets(
            key,
            weight,
            *offsets,
            (len(offsets), tp_rank, tp),
            prepend_axis_num=len(offsets),
            replica_id=(0, 0, dp_rank),
        )
        if kind == 'kda':
            factory = _split_tensor_factory(
                entry, [size // tp for size in global_sizes], ['query', 'key', 'value'], 0
            )
        else:
            factory = apply_swiglu_sharded_factory(entry, offsets)
        factory = attach_optimizer_factory(factory, weight)
    return factory, weight, expected, valid


@pytest.mark.parametrize('kind', ['kda', 'glu', 'expert'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_local_fragments_and_flat_optimizer_slices(kind, dtype):
    """All/partial padding, TP offsets, optimizer identity and non-row-aligned DP slices."""
    for rank in range(8):
        with mock.patch.object(
            dist, 'all_gather_into_tensor', side_effect=AssertionError('gather')
        ):
            factory, weight, expected, valid = _factory(kind, 2, 4, rank, device='cpu', dtype=dtype)
            chunks = factory.build()
        tensors = [chunk for chunk in chunks if isinstance(chunk, ShardedTensor)]
        assert sum(chunk.data.numel() for chunk in tensors) == valid * 3
        for chunk in tensors:
            assert chunk.data.untyped_storage().data_ptr() == weight.untyped_storage().data_ptr()
        values = [
            chunk.data if isinstance(chunk, ShardedTensor) else chunk.unwrap() for chunk in chunks
        ]
        torch.testing.assert_close(factory.merge_fn(values), expected, rtol=0, atol=0)
        metadata = get_param_id_to_sharded_param_map({'model': factory}, [weight])[0]
        assert metadata.data is weight
        for start, stop in [(0, weight.numel()), (1, 5), (3, weight.numel() - 1)]:
            state = torch.arange(start, stop, dtype=torch.float32) + 100
            optimizer = replace(
                metadata,
                key='optimizer.momentum.' + factory.key,
                data=state,
                flattened_range=slice(start, stop),
            )
            built = optimizer.build()
            loaded = {
                'layout': built['layout'].unwrap(),
                'chunks': [part.data for part in built['chunks']],
            }
            reference = state.clone()
            reference[max(0, valid * 3 - start) :].zero_()
            torch.testing.assert_close(optimizer.merge_fn(loaded), reference, rtol=0, atol=0)
            for part in built['chunks']:
                assert part.key.startswith('optimizer.momentum.')
                assert part.flattened_range is None


@pytest.mark.parametrize('kind', ['kda', 'glu', 'expert'])
@pytest.mark.parametrize('source,target', [((2, 1), (1, 4)), ((1, 4), (2, 1)), ((2, 2), (1, 4))])
def test_cross_layout_model_and_optimizer(kind, source, target, tmp_path_dist_ckpt):
    """Real DCP cross-loads model, FP32 master and optimizer state, then continues an update."""
    assert dist.get_world_size() % 4 == 0
    rank = dist.get_rank()
    factory, weight, _, _ = _factory(kind, *source, rank)
    optimizer = torch.optim.SGD([weight], lr=0.01, momentum=0.9)
    weight.grad = torch.full_like(weight, 2)
    optimizer.step()
    # Master and momentum differ from model values: identity-only/zero-state tests cannot pass.
    metadata = get_param_id_to_sharded_param_map({'weight': factory}, [weight])[0]
    state = optimizer.state[weight]['momentum_buffer']
    checkpoint = {
        'weight': factory,
        'master': make_sharded_optimizer_tensor(metadata, weight.detach().float() + 1000, 'master'),
        'momentum': make_sharded_optimizer_tensor(metadata, state, 'momentum'),
    }
    path = tmp_path_dist_ckpt / f'{kind}-{source}-{target}'
    path.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    dist_checkpointing.save(checkpoint, path)
    factory, weight, expected, valid = _factory(kind, *target, rank, scribble=True)
    metadata = get_param_id_to_sharded_param_map({'weight': factory}, [weight])[0]
    loaded = dist_checkpointing.load(
        {
            'weight': factory,
            'master': make_sharded_optimizer_tensor(metadata, torch.empty_like(weight), 'master'),
            'momentum': make_sharded_optimizer_tensor(
                metadata, torch.empty_like(weight), 'momentum'
            ),
        },
        path,
    )
    for name, delta in [('weight', -0.02), ('master', 999.98), ('momentum', None)]:
        reference = expected + delta if delta is not None else torch.full_like(expected, 2)
        reference[valid:].zero_()
        torch.testing.assert_close(loaded[name], reference, rtol=1e-6, atol=1e-5)
    with torch.no_grad():
        weight.copy_(loaded['weight'])
    optimizer = torch.optim.SGD([weight], lr=0.01, momentum=0.9)
    optimizer.load_state_dict(
        {
            'state': {0: {'momentum_buffer': loaded['momentum']}},
            'param_groups': optimizer.state_dict()['param_groups'],
        }
    )
    weight.grad = torch.full_like(weight, 3)
    optimizer.step()
    torch.testing.assert_close(
        weight[:valid], expected[:valid] - 0.02 - 0.048, rtol=1e-6, atol=1e-5
    )


@pytest.mark.parametrize('kind', ['kda', 'dense', 'shared', 'expert'])
def test_real_kimi_module_uses_local_fragments(kind):
    """KDA, dense/shared GLU and EP2 x EGTP2 call the no-gather production factories."""
    import torch.nn.functional as F

    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelGroupedLinear,
        TELayerNormColumnParallelLinear,
        TERowParallelGroupedLinear,
        TERowParallelLinear,
    )
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
        GTP_CONFIG,
        GTPShardedParam,
        update_gtp_config,
    )
    from megatron.core.tensor_parallel.gtp_api import HAVE_GTP, is_gtp_param
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.mlp import MLP, MLPSubmodules
    from megatron.core.transformer.moe.experts import TEGroupedMLP
    from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
    from megatron.core.transformer.spec_utils import build_module
    from megatron.core.transformer.transformer_config import TransformerConfig

    assert HAVE_GTP, 'Run with the GTP TransformerEngine overlay'
    ps.destroy_model_parallel()
    expert = kind == 'expert'
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        gtp_remat_size=1 if expert else 2,
        expert_model_parallel_size=2 if expert else 1,
        expert_gtp_remat_size=2 if expert else 1,
    )
    old_pad = GTP_CONFIG.pad_for_alignment
    update_gtp_config(pad_for_alignment=16)
    model_parallel_cuda_manual_seed(42)
    try:
        pg = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=[
                'tp',
                'cp',
                'gtp_remat',
                'dp_cp',
                'ep',
                'expt_tp',
                'expt_dp',
                'expt_gtp_remat',
            ]
        )
        config = TransformerConfig(
            num_layers=1,
            hidden_size=256,
            num_attention_heads=8,
            normalization='RMSNorm',
            bf16=True,
            params_dtype=torch.bfloat16,
            transformer_impl='transformer_engine',
            ffn_hidden_size=388,
            gated_linear_unit=True,
            activation_func=F.silu,
            add_bias_linear=False,
            moe_ffn_hidden_size=388,
            num_moe_experts=2,
            expert_model_parallel_size=2 if expert else 1,
            moe_grouped_gemm=True,
            moe_shared_expert_intermediate_size=388,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=64,
            linear_value_head_dim=64,
            linear_num_key_heads=4,
            linear_num_value_heads=4,
            kda_f_lora_rank=64,
            is_hybrid_model=True,
            experimental_attention_variant='kda' if kind == 'kda' else None,
        )
        submodules = MLPSubmodules(
            linear_fc1=TELayerNormColumnParallelLinear, linear_fc2=TERowParallelLinear
        )
        if kind == 'kda':
            model = build_module(
                hybrid_stack_spec.submodules.kda_layer.submodules.self_attention,
                config=config,
                layer_number=1,
                pg_collection=pg,
            )
            key, weight = 'in_proj.weight', model.in_proj.weight
        elif expert:
            model = TEGroupedMLP(
                1,
                config,
                MLPSubmodules(
                    linear_fc1=TEColumnParallelGroupedLinear, linear_fc2=TERowParallelGroupedLinear
                ),
                pg_collection=pg,
            )
            key, weight = 'linear_fc1.weight0', model.linear_fc1.weight0
        elif kind == 'shared':
            model = SharedExpertMLP(config, submodules, gate=False, pg_collection=pg)
            key, weight = 'linear_fc1.weight', model.linear_fc1.weight
        else:
            model = MLP(config, submodules, tp_group=pg.tp, pg_collection=pg)
            key, weight = 'linear_fc1.weight', model.linear_fc1.weight
        with mock.patch.object(
            dist, 'all_gather_into_tensor', side_effect=AssertionError('checkpoint gathered')
        ):
            sd = model.sharded_state_dict(metadata={'dp_cp_group': pg.dp_cp})
        assert is_gtp_param(weight)
        factory = sd[key]
        assert factory.for_optimizer().data is weight
        chunks = factory.build()
        assert (
            sum(part.data.numel() for part in chunks if isinstance(part, ShardedTensor))
            <= weight.numel()
        )
        for part in chunks:
            if isinstance(part, ShardedTensor):
                assert part.data.untyped_storage().data_ptr() == weight.untyped_storage().data_ptr()
        indexed = get_param_id_to_sharded_param_map(sd, [weight])
        assert indexed[0].key == factory.key
    finally:
        update_gtp_config(pad_for_alignment=old_pad)
        ps.destroy_model_parallel()
        ps.initialize_model_parallel()
        GTPShardedParam._chain_state = {}


@pytest.mark.parametrize("master_layout", ["factories", "legacy_lists", "indexed_lists"])
@pytest.mark.parametrize("layerwise", [False, True])
def test_mixed_master_factories_preserve_parameter_indices_through_load(master_layout, layerwise):
    """Only the middle master has local factory metadata; neighbors must not shift."""
    from types import SimpleNamespace

    from megatron.core.dist_checkpointing.dict_utils import dict_list_map_outplace, merge
    from megatron.core.dist_checkpointing.mapping import apply_factory_merges
    from megatron.core.dist_checkpointing.state_dict_utils import load_preprocess
    from megatron.core.optimizer.optimizer import Float16OptimizerWithFloat16Params

    model_params = [torch.nn.Parameter(torch.zeros(4, 3, dtype=torch.bfloat16)) for _ in range(3)]
    masters = [torch.nn.Parameter(torch.full((4, 3), float(index + 11))) for index in range(3)]
    wrapper = Float16OptimizerWithFloat16Params.__new__(Float16OptimizerWithFloat16Params)
    wrapper.optimizer = torch.optim.SGD(masters, lr=0.1, momentum=0.9)
    wrapper.config = SimpleNamespace(fp16=False)
    wrapper.grad_scaler = None
    wrapper._optimizer_state_offloader = None
    wrapper.float16_groups = [model_params]
    wrapper.fp32_from_float16_groups = [masters]
    wrapper.fp32_from_fp32_groups = [[]]
    wrapper.synchronize_optimizer_state_for_checkpoint = lambda: None
    model_state = {}
    for index, param in enumerate(model_params):
        key = "layer%d.weight" % index
        model_state[key] = ShardedTensor.from_rank_offsets(key, param)
        wrapper.optimizer.state[masters[index]]["momentum_buffer"] = torch.full(
            (4, 3), float(index + 101)
        )
    live = model_params[1]
    live.is_gtp_weight_remat = True
    physical = ShardedTensor.from_rank_offsets("layer1.weight", live, (0, 0, 2))
    factory = _gtp_split_tensor_factory(
        physical, [4, 4], ['w', 'v'], 0, 0, 1, weight=live, glu=True
    )
    model_state["layer1.weight"] = factory

    owner = wrapper
    if layerwise:
        from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer

        owner = LayerWiseDistributedOptimizer.__new__(LayerWiseDistributedOptimizer)
        owner.chained_optimizers = [wrapper]
        owner.pg_collection = SimpleNamespace()
        owner.model_chunks = []
    if master_layout != "factories":
        restored = wrapper.state_dict()
        restored["fp32_from_fp16_params"] = [[param.detach().clone() for param in masters]]
    else:
        with mock.patch(
            "megatron.core.optimizer.layer_wise_optimizer._all_gather_param_group_metadata",
            side_effect=lambda group, pg: [group],
        ):
            sharded = owner.sharded_state_dict(model_state)
        remaining, nonpersistent, factories = load_preprocess(sharded)
        loaded = dict_list_map_outplace(
            lambda value: value.data.clone() if isinstance(value, ShardedTensor) else value,
            remaining,
        )
        merge(nonpersistent, loaded)
        apply_factory_merges(nonpersistent, factories)
        restored = nonpersistent
    if master_layout == "indexed_lists":
        restored["fp32_from_fp16_params"] = dict(enumerate(restored["fp32_from_fp16_params"]))
    for param in masters:
        param.data.fill_(-999)
    owner.load_state_dict(restored)
    for index, param in enumerate(masters):
        torch.testing.assert_close(param, torch.full_like(param, index + 11), rtol=0, atol=0)
        torch.testing.assert_close(
            wrapper.optimizer.state[param]["momentum_buffer"],
            torch.full_like(param, index + 101),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize('tp_axis', [0, 1])
@pytest.mark.parametrize('padding', [2, 4])
def test_plain_padded_checkpoint_coordinates(tp_axis, padding, tmp_path_dist_ckpt):
    """Non-fused KDA/output weights use unpadded TP coordinates and restore zero tails."""
    rank = dist.get_rank()
    tp_rank, gtp_rank = rank % 4 // 2, rank % 2
    rows, columns = 8 - padding, 3
    shape = (rows * 2, columns) if tp_axis == 0 else (rows, columns * 2)
    reference = torch.arange(1, math.prod(shape) + 1, device='cuda', dtype=torch.float32).reshape(
        shape
    )
    data = reference.chunk(2, dim=tp_axis)[tp_rank].contiguous()
    source = ShardedTensor.from_rank_offsets(
        'weight', data, (tp_axis, tp_rank, 2), replica_id=(0, 0, rank // 4 * 2 + gtp_rank)
    )
    path = tmp_path_dist_ckpt / f'plain-{tp_axis}-{padding}'
    path.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    dist_checkpointing.save(
        {'weight': source, 'state': make_sharded_optimizer_tensor(source, data + 1000, 'momentum')},
        path,
    )
    physical = torch.full((4, columns), -999.0, device='cuda')
    target = _make_gtp_logical_sharded_tensor(
        physical,
        'weight',
        tp_axis=tp_axis,
        tp_rank=tp_rank,
        tp_size=2,
        gtp_rank=gtp_rank,
        gtp_remat_size=2,
        pad_length=padding,
        prepend_offsets=(),
        prepend_axis_num=0,
        other_offsets=((tp_axis, tp_rank, 2),) if tp_axis else (),
        replica_id=(0, 0, rank // 4),
    )
    loaded = dist_checkpointing.load(
        {
            'weight': target,
            'state': make_sharded_optimizer_tensor(target, torch.empty_like(physical), 'momentum'),
        },
        path,
    )
    keep = target.local_shape[0]
    expected = physical.new_zeros(physical.shape)
    expected[:keep] = data[gtp_rank * 4 : gtp_rank * 4 + keep]
    torch.testing.assert_close(loaded['weight'], expected, rtol=0, atol=0)
    expected[:keep] += 1000
    torch.testing.assert_close(loaded['state'], expected, rtol=0, atol=0)


@pytest.mark.parametrize('mode', ['fully_reshardable', 'fs_model_space'])
@pytest.mark.parametrize('kind', ['kda', 'glu', 'expert'])
def test_distributed_adam_uses_local_companion(mode, kind):
    """Exercise both production Adam checkpoint formats with nonzero flat moment buffers."""
    from types import SimpleNamespace

    from megatron.core.dist_checkpointing.dict_utils import dict_list_map_outplace
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer, Range

    factory, weight, expected, valid = _factory(kind, 2, 2, dist.get_rank(), device='cpu')
    n = weight.numel()
    start, stop = 1, n - 1
    values = {
        'param': weight.detach().flatten().clone(),
        'exp_avg': weight.detach().flatten().clone() + 5000,
        'exp_avg_sq': weight.detach().flatten().clone() + 9000,
    }
    stub = SimpleNamespace(
        gbuf_ranges=[{torch.float32: [{'param_map': {weight: {'param': Range(start, stop)}}}]}],
        buffers=[SimpleNamespace(param_index_map={weight: (0, n, 0)}, buckets=[None])],
        distributed_optimizer_instance_id=0,
        _get_main_param_and_optimizer_states=lambda param: {
            key: value[start:stop].clone() for key, value in values.items()
        },
        get_parameter_state_dp_zero=lambda **kwargs: {0: {torch.float32: values}},
    )
    method = getattr(DistributedOptimizer, f'sharded_param_state_{mode}')
    with mock.patch.object(dist, 'all_gather_into_tensor', side_effect=AssertionError('gather')):
        result = method(stub, {'weight': factory}, metadata={})
    for key, entry in result[0].items():
        payload = dict_list_map_outplace(
            lambda leaf: leaf.data.clone() if isinstance(leaf, ShardedTensor) else leaf.unwrap(),
            entry.build(),
        )
        delta = {'param': 0, 'fp32_param': 0, 'exp_avg': 5000, 'exp_avg_sq': 9000}[key]
        reference = expected + delta
        reference[valid:].zero_()
        if mode == 'fs_model_space':
            reference = reference.flatten()[start:stop]
        torch.testing.assert_close(entry.merge_fn(payload), reference, rtol=0, atol=0)
