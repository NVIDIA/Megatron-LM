# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Canonical GLU checkpoints preserve parameter identity and optimizer row semantics."""

from argparse import Namespace
from dataclasses import replace
from functools import partial

import pytest
import torch
from torch.utils._pytree import tree_map

from megatron.core import dist_checkpointing, parallel_state
from megatron.core.dist_checkpointing.glu import (
    convert_glu_model_state,
    glu_checkpoint_metadata,
    glu_runtime_layout,
    permute_glu_rows,
    validate_glu_checkpoint_layout,
)
from megatron.core.dist_checkpointing.mapping import ShardedTensor, ShardedTensorFactory
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.mlp import apply_swiglu_sharded_factory
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.dist_checkpointing.test_optimizer import SwigluFactoryModel
from tests.unit_tests.dist_checkpointing.utils import setup_model_and_optimizer
from tests.unit_tests.test_utilities import Utils


def _runtime(tensor, size):
    if size is None:
        return tensor.clone()
    gate, up = tensor.chunk(2)
    return torch.cat([piece for pair in zip(gate.split(size), up.split(size)) for piece in pair])


def _config(size, tp=1):
    return Namespace(
        gated_linear_unit=True,
        moe_mlp_glu_interleave_size=size,
        tensor_model_parallel_size=tp,
        expert_tensor_parallel_size=tp,
    )


@pytest.mark.parametrize('size', [None, 1, 2, 4])
@pytest.mark.parametrize('shape', [(16,), (16, 3)])
@pytest.mark.parametrize('singleton', [False, True])
def test_factory_layout_and_identity(size, shape, singleton):
    canonical = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)
    runtime = _runtime(canonical, size)
    factory = apply_swiglu_sharded_factory(
        ShardedTensor.from_rank_offsets('fc1', runtime), (), singleton, glu_interleave_size=size
    )
    assert factory.data is runtime
    snapshot = runtime.clone()
    parts = factory.build()
    for part in parts:
        half = (
            int(part.key.endswith('_v')) if singleton else part.global_offset[0] // (shape[0] // 2)
        )
        expected = canonical.chunk(2)[half]
        torch.testing.assert_close(part.data, expected, rtol=0, atol=0)
    loaded_parts = [part.data.clone() for part in parts]
    loaded = factory.merge_fn(loaded_parts)
    torch.testing.assert_close(loaded, factory.data, rtol=0, atol=0)
    torch.testing.assert_close(runtime, snapshot, rtol=0, atol=0)


@pytest.mark.parametrize('size', [0, -1, True, 3])
def test_invalid_interleaving_fails_before_factory_expansion(size):
    with pytest.raises(ValueError, match='interleave size'):
        apply_swiglu_sharded_factory(
            ShardedTensor.from_rank_offsets('fc1', torch.empty(16, 3)), (), glu_interleave_size=size
        )


@pytest.mark.parametrize('loading_optimizer', [False, True])
def test_unmarked_interleaved_checkpoint_is_rejected_even_with_matching_settings(loading_optimizer):
    with pytest.raises(ValueError, match='unmarked interleaved'):
        validate_glu_checkpoint_layout(
            None,
            source_config=_config(2),
            target_config=_config(2),
            loading_optimizer=loading_optimizer,
        )


def test_metadata_distinguishes_model_space_from_raw_optimizer_buffers():
    source, target = _config(2), _config(4, tp=2)
    for model_space in (False, True):
        metadata = glu_checkpoint_metadata(source, optimizer_uses_model_layout=model_space)
        validate_glu_checkpoint_layout(
            metadata, source_config=None, target_config=target, loading_optimizer=False
        )
        if model_space:
            validate_glu_checkpoint_layout(
                metadata, source_config=None, target_config=target, loading_optimizer=True
            )
        else:
            with pytest.raises(ValueError, match='Raw optimizer buffers'):
                validate_glu_checkpoint_layout(
                    metadata, source_config=None, target_config=target, loading_optimizer=True
                )
            validate_glu_checkpoint_layout(
                metadata, source_config=None, target_config=source, loading_optimizer=True
            )
    validate_glu_checkpoint_layout(
        None, source_config=_config(None), target_config=_config(None), loading_optimizer=True
    )
    validate_glu_checkpoint_layout(
        {'distrib_optim_sharding_type': 'fully_reshardable'},
        source_config=_config(None),
        target_config=_config(2),
        loading_optimizer=True,
    )
    with pytest.raises(ValueError, match='unmarked interleaved'):
        validate_glu_checkpoint_layout(
            None, source_config=None, target_config=target, loading_optimizer=False
        )


@pytest.fixture
def distributed():
    Utils.initialize_model_parallel(1, 1)
    yield
    Utils.destroy_model_parallel()


@pytest.mark.parametrize('single_weight', [False, True])
def test_real_grouped_mlp_factories_use_the_configured_layout(
    distributed, single_weight, tmp_path_dist_ckpt, monkeypatch
):
    monkeypatch.setenv("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "1")
    model_parallel_cuda_manual_seed(123)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        num_moe_experts=2,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        add_bias_linear=False,
        moe_grouped_gemm=True,
        moe_use_grouped_tensor=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        moe_single_grouped_weight=single_weight,
        moe_mlp_glu_interleave_size=2,
        moe_shared_expert_intermediate_size=64,
        use_grouped_gemm_for_shared_expert=True,
        moe_shared_expert_glu_interleave_size=4,
    )
    submodules = get_submodules(
        get_gpt_layer_with_transformer_engine_submodules(2, moe_grouped_gemm=True).mlp
    )
    layer = MoELayer(config, submodules).bfloat16()
    for linear in (layer.experts.linear_fc1, layer.experts.linear_fc2):
        assert linear.single_grouped_weight == single_weight
        if single_weight:
            assert "weight" in dict(linear.named_parameters(recurse=False))
    for module, size in ((layer.experts, 2), (layer.shared_experts, 4)):
        state = module.sharded_state_dict(
            metadata={'dp_cp_group': parallel_state.get_data_parallel_group()}
        )
        from megatron.core.dist_checkpointing.optimizer import get_param_id_to_sharded_param_map

        parameters = list(module.parameters())
        parameter_map = get_param_id_to_sharded_param_map(state, parameters)
        assert len(parameter_map) == len(parameters), [
            name for name, _ in module.named_parameters()
        ]
        factories = [
            value
            for key, value in state.items()
            if 'linear_fc1.weight' in key and isinstance(value, ShardedTensorFactory)
        ]
        assert factories
        for factory in factories:
            grouped = factory.data.ndim == 3
            views = list(factory.data.unbind(0)) if grouped else [factory.data]
            canonical = []
            for expert, view in enumerate(views):
                # Keep small BF16-exact row values, distinct within each expert.
                rows = torch.arange(view.shape[0], device=view.device) % 32 + expert * 32
                value = rows[:, None].expand_as(view).to(view.dtype)
                canonical.append(value)
            runtime = [_runtime(value, size) for value in canonical]
            with torch.no_grad():
                factory.data.copy_(torch.stack(runtime) if grouped else runtime[0])
            built = factory.build()
            expert_parts = built if grouped else [built]
            for parts, expected in zip(expert_parts, canonical, strict=True):
                torch.testing.assert_close(
                    torch.cat([part.data for part in parts]),
                    expected,
                    rtol=0,
                    atol=0,
                    msg=lambda m: f"{factory.key} shape={factory.data.shape} size={size} grouped={grouped} {m}",
                )

    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer

    ddp = DistributedDataParallel(
        config, DistributedDataParallelConfig(use_distributed_optimizer=True), layer
    )
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            lr=0.001, bf16=True, params_dtype=torch.bfloat16, use_distributed_optimizer=True
        ),
        [ddp],
    )
    for parameter in layer.parameters():
        parameter.main_grad.fill_(0.1)
    assert optimizer.step()[0]
    metadata = {
        'distrib_optim_sharding_type': 'fully_reshardable',
        **glu_checkpoint_metadata([layer], optimizer_uses_model_layout=True),
    }
    model_state = layer.sharded_state_dict()

    def snapshot():
        return tree_map(
            lambda value: value.detach().clone() if isinstance(value, torch.Tensor) else value,
            {"model": layer.state_dict(), "optimizer": optimizer.state_dict()},
        )

    expected = snapshot()
    optimizer_state = optimizer.sharded_state_dict(model_state, metadata=metadata)
    with TempNamedDir(tmp_path_dist_ckpt / 'real-grouped-adam') as path:
        dist_checkpointing.save(
            {'model': model_state, 'optimizer': optimizer_state}, path, content_metadata=metadata
        )
        with torch.no_grad():
            for parameter in layer.parameters():
                parameter.zero_()
            for part in getattr(optimizer, "chained_optimizers", [optimizer]):
                for group in part.param_groups:
                    for parameter in group["params"]:
                        parameter.zero_()
                for state in part.optimizer.state.values():
                    for value in state.values():
                        if isinstance(value, torch.Tensor):
                            value.zero_()
        loaded = dist_checkpointing.load(
            {
                'model': layer.sharded_state_dict(),
                'optimizer': optimizer.sharded_state_dict(
                    layer.sharded_state_dict(), metadata=metadata, is_loading=True
                ),
            },
            path,
        )
        layer.load_state_dict(loaded['model'])
        optimizer.load_state_dict(loaded['optimizer'])
        torch.testing.assert_close(snapshot(), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    'source_size,target_size,target_tp', [(2, 2, 1), (2, 4, 2), (2, None, 1), (None, 2, 1)]
)
def test_disk_roundtrip_reshards_model_master_and_moments(
    tmp_path_dist_ckpt, distributed, source_size, target_size, target_tp
):
    def template(size, tp, *, saving):
        tp_rank = Utils.rank % tp
        canonical = torch.arange(160, dtype=torch.float32).reshape(32, 5)
        gate, up = canonical.chunk(2)
        local = torch.cat((gate.chunk(tp)[tp_rank], up.chunk(tp)[tp_rank]))
        runtime = _runtime(local, size)
        factory = apply_swiglu_sharded_factory(
            ShardedTensor.from_rank_offsets(
                'fc1',
                runtime if saving else torch.empty_like(runtime),
                (0, tp_rank, tp),
                replica_id=Utils.rank // tp,
            ),
            (),
            glu_interleave_size=size,
        )
        return {
            name: replace(
                factory,
                key=name,
                data=(runtime + index * 1000) if saving else torch.empty_like(runtime),
            )
            for index, name in enumerate(('model', 'master', 'exp_avg', 'exp_avg_sq'))
        }, runtime

    with TempNamedDir(tmp_path_dist_ckpt / 'glu-layout') as path:
        source, _ = template(source_size, 1, saving=True)
        dist_checkpointing.save(
            source,
            path,
            content_metadata=glu_checkpoint_metadata(
                _config(source_size), optimizer_uses_model_layout=True
            ),
        )
        target, runtime = template(target_size, target_tp, saving=False)
        loaded = dist_checkpointing.load(target, path)
        for index, name in enumerate(('model', 'master', 'exp_avg', 'exp_avg_sq')):
            torch.testing.assert_close(loaded[name], runtime + index * 1000, rtol=0, atol=0)


class InterleavedOptimizerModel(SwigluFactoryModel):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.config.gated_linear_unit = True
        self.config.moe_mlp_glu_interleave_size = size
        canonical = (
            torch.arange(self.linear.weight.numel(), dtype=torch.float32).reshape_as(
                self.linear.weight
            )
            / 100
        )
        with torch.no_grad():
            self.linear.weight.copy_(_runtime(canonical, size))

    def sharded_state_dict(self):
        weight = self.linear.weight
        return {
            'linear.weight': apply_swiglu_sharded_factory(
                ShardedTensor.from_rank_offsets(
                    'linear.weight', weight, replica_id=(0, 0, Utils.rank)
                ),
                (),
                glu_interleave_size=self.size,
            )
        }


def _initialize_interleaved(*, size, **kwargs):
    return InterleavedOptimizerModel(size)


@pytest.mark.parametrize("dist_opt,memory_efficient", [(True, False), (True, True), (False, False)])
def test_adam_resume_preserves_three_updates(
    tmp_path_dist_ckpt, distributed, dist_opt, memory_efficient
):
    metadata = {
        'distrib_optim_sharding_type': 'fully_reshardable',
        'distrib_optim_fully_reshardable_mem_efficient': memory_efficient,
        **glu_checkpoint_metadata(_config(2), optimizer_uses_model_layout=True),
    }

    def initialize(size):
        models, optimizer = setup_model_and_optimizer(
            seed=5,
            tp=1,
            pp=1,
            dist_opt=dist_opt,
            initialize_fn=partial(_initialize_interleaved, size=size),
        )
        for group in optimizer.param_groups:
            group['lr'] = 0.003
        if hasattr(optimizer, 'chained_optimizers'):
            assert len(optimizer.chained_optimizers) == 1
            optimizer = optimizer.chained_optimizers[0]
        return models[0], optimizer

    def update(model, optimizer, step):
        optimizer.zero_grad()
        weight = model.linear.weight
        gradient = (
            torch.arange(weight.numel(), device='cuda').reshape_as(weight) % 7 + step + 1
        ) / 100
        weight.main_grad.copy_(_runtime(gradient, model.size))
        assert optimizer.step()[0]

    model_a, optimizer_a = initialize(2)
    if not dist_opt:
        from megatron.training.checkpointing import _build_sharded_state_dict_metadata

        metadata = _build_sharded_state_dict_metadata(model=[model_a])
        metadata.pop("dp_cp_group")
        assert metadata["glu_optimizer_layout"] == "contiguous"
    for step in range(2):
        update(model_a, optimizer_a, step)
    with TempNamedDir(tmp_path_dist_ckpt / 'glu-adam-resume') as path:
        model_state = model_a.sharded_state_dict()
        dist_checkpointing.save(
            {
                'model': model_state,
                'optimizer': optimizer_a.sharded_state_dict(model_state, metadata=metadata),
            },
            path,
            content_metadata=metadata,
        )
        model_b, optimizer_b = initialize(4)
        model_state = model_b.sharded_state_dict()
        loaded = dist_checkpointing.load(
            {
                'model': model_state,
                'optimizer': optimizer_b.sharded_state_dict(
                    model_state, metadata=metadata, is_loading=True
                ),
            },
            path,
        )
        model_b.load_state_dict(loaded['model'])
        optimizer_b.load_state_dict(loaded['optimizer'])

        def assert_optimizer_state(rtol=0):
            if dist_opt:
                states = [
                    optimizer.get_parameter_state_dp_zero(
                        use_gloo_comm=False, return_on_all_ranks=True
                    )
                    for optimizer in (optimizer_a, optimizer_b)
                ]
                buffers = [next(iter(state[0].values())) for state in states]
            else:
                buffers = []
                for optimizer in (optimizer_a, optimizer_b):
                    master = optimizer.fp32_from_float16_groups[0][0]
                    buffers.append({"param": master, **optimizer.optimizer.state[master]})
            for name in ('param', 'exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(
                    permute_glu_rows(buffers[0][name].reshape(64, 5), 2, interleave=False),
                    permute_glu_rows(buffers[1][name].reshape(64, 5), 4, interleave=False),
                    rtol=rtol,
                    atol=0,
                )

        # Loading itself must be bit-exact, before any optimizer arithmetic runs.
        assert_optimizer_state()
        torch.testing.assert_close(
            permute_glu_rows(model_a.linear.weight, 2, interleave=False),
            permute_glu_rows(model_b.linear.weight, 4, interleave=False),
            rtol=0,
            atol=0,
        )
        for step in range(2, 5):
            update(model_a, optimizer_a, step)
            update(model_b, optimizer_b, step)
            torch.testing.assert_close(
                permute_glu_rows(model_a.linear.weight, 2, interleave=False),
                permute_glu_rows(model_b.linear.weight, 4, interleave=False),
                rtol=0,
                atol=0,
            )
            # Reordered ordinary FusedAdam tensors can select different vector lanes
            # and differ by one FP32 ulp after arithmetic. Restore above remains exact.
            assert_optimizer_state(rtol=0 if dist_opt else 1e-6)


def test_legacy_conversion_uses_built_modules_not_vision_fc1_names():
    model = torch.nn.Module()
    model.language = torch.nn.Module()
    model.language.linear_fc1 = torch.nn.Linear(3, 128, bias=False)
    model.language.linear_fc1._glu_interleave_size = 32
    model.vision = torch.nn.Module()
    model.vision.linear_fc1 = torch.nn.Linear(3, 4304, bias=False)
    state = model.state_dict()
    converted = convert_glu_model_state(state, model, interleave=False)
    assert converted['vision.linear_fc1.weight'] is state['vision.linear_fc1.weight']
    restored = convert_glu_model_state(converted, model, interleave=True)
    torch.testing.assert_close(
        restored['language.linear_fc1.weight'], state['language.linear_fc1.weight']
    )


def test_runtime_layout_follows_built_dense_mlp():
    class BuiltMLP(torch.nn.Module):
        def __init__(self, size):
            super().__init__()
            self.config = _config(None)
            self.config.use_grouped_gemm_for_dense_mlp = True
            self.size = size

        def _checkpoint_glu_interleave_size(self):
            return self.size

    ordinary, fused = BuiltMLP(None), BuiltMLP(32)
    assert glu_runtime_layout([ordinary])['dense'] is None
    assert glu_runtime_layout([fused])['dense'] == 32
    with pytest.raises(ValueError, match='Raw optimizer buffers'):
        validate_glu_checkpoint_layout(
            glu_checkpoint_metadata([ordinary], optimizer_uses_model_layout=False),
            source_config=None,
            target_config=[fused],
            loading_optimizer=True,
        )


@pytest.mark.parametrize("explicit_metadata", [False, True])
def test_direct_core_writer_marks_glu_and_rejects_ambiguous_old_rows(
    tmp_path_dist_ckpt, distributed, explicit_metadata
):
    canonical = torch.arange(24, device='cuda').reshape(8, 3).float()
    runtime = _runtime(canonical, 1)

    def factory(size):
        return apply_swiglu_sharded_factory(
            ShardedTensor.from_rank_offsets('fc1', runtime.clone(), replica_id=Utils.rank),
            (),
            glu_interleave_size=size,
        )

    with TempNamedDir(tmp_path_dist_ckpt / 'direct-glu') as path:
        requested = {'fc1': factory(1)}
        dist_checkpointing.save(
            requested, path, content_metadata={"application": 7} if explicit_metadata else None
        )
        assert set(requested) == {'fc1'}
        assert (
            dist_checkpointing.load_content_metadata(path)['glu_checkpoint_layout'] == 'contiguous'
        )
        loaded = dist_checkpointing.load({'fc1': factory(1)}, path)
        assert set(loaded) == ({'fc1', 'content_metadata'} if explicit_metadata else {'fc1'})
        if explicit_metadata:
            assert loaded['content_metadata']['application'] == 7
        torch.testing.assert_close(loaded['fc1'], runtime)
    with TempNamedDir(tmp_path_dist_ckpt / 'old-direct-glu') as path:
        # Reproduce the old Core writer: split runtime rows without canonicalization.
        old = factory(None)
        unmarked = ShardedTensorFactory(
            old.key, old.data, old.build_fn, old.merge_fn, old.replica_id
        )
        dist_checkpointing.save({'fc1': unmarked}, path)
        destination = factory(1)
        snapshot = destination.data.clone()
        with pytest.raises(ValueError, match='unmarked interleaved'):
            dist_checkpointing.load({'fc1': destination}, path)
        torch.testing.assert_close(destination.data, snapshot)


@pytest.mark.parametrize('single_bias', [False, True])
def test_single_grouped_parameter_is_used_by_optimizer_factories(
    distributed, tmp_path_dist_ckpt, single_bias
):
    from megatron.core.dist_checkpointing.optimizer import (
        get_param_id_to_sharded_param_map,
        make_sharded_optimizer_tensor,
    )
    from megatron.core.transformer.moe.experts import _grouped_checkpoint_factory

    param = torch.nn.Parameter(torch.arange(48, device='cuda').reshape(2, 8, 3).float())
    if single_bias:
        param = torch.nn.Parameter(param[:, :, 0].contiguous())
    with torch.no_grad():
        param.copy_(permute_glu_rows(param, 2, axis=1, interleave=True))
    templates = [
        apply_swiglu_sharded_factory(
            ShardedTensor.from_rank_offsets(
                'experts.fc1', view, (0, i, 2), prepend_axis_num=1, replica_id=Utils.rank
            ),
            ((0, i, 2),),
            glu_interleave_size=2,
        )
        for i, view in enumerate(param)
    ]
    factory = _grouped_checkpoint_factory(
        param, templates, lambda tensor, _: tensor.unbind(0), 'weight'
    )
    mapping = get_param_id_to_sharded_param_map({'weight': factory}, [param])
    assert mapping[0].data is param
    moment = make_sharded_optimizer_tensor(
        mapping[0], param.detach().clone() + 100, prefix='optimizer.exp_avg'
    )
    with TempNamedDir(tmp_path_dist_ckpt / 'grouped-optimizer') as path:
        dist_checkpointing.save({'weight': factory, 'moment': moment}, path)
        loaded = dist_checkpointing.load({'weight': factory, 'moment': moment}, path)
        torch.testing.assert_close(loaded['weight'], param)
        torch.testing.assert_close(loaded['moment'], param + 100)


@pytest.mark.parametrize("flattened", [False, True])
def test_grouped_fc2_restores_fp32_optimizer_tensors(distributed, tmp_path_dist_ckpt, flattened):
    from megatron.core.dist_checkpointing.optimizer import make_sharded_optimizer_tensor
    from megatron.core.transformer.moe.experts import _grouped_checkpoint_factory

    values = torch.arange(24, device="cuda", dtype=torch.float32).reshape(2, 4, 3)
    parameter = torch.nn.Parameter(
        values.bfloat16().reshape(8, 3) if flattened else values.bfloat16()
    )

    def split(tensor, _):
        return list(tensor.reshape(2, 4, 3).unbind(0))

    templates = [
        ShardedTensor.from_rank_offsets(
            "experts.fc2", view, (0, expert, 2), prepend_axis_num=1, replica_id=Utils.rank
        )
        for expert, view in enumerate(split(parameter, "weight"))
    ]
    factory = _grouped_checkpoint_factory(parameter, templates, split, "weight")
    expected = {"model": parameter.detach().clone()}
    state = {"model": factory}
    for index, name in enumerate(("master", "exp_avg", "exp_avg_sq")):
        expected[name] = parameter.detach().float() + index + 0.125
        state[name] = make_sharded_optimizer_tensor(factory, expected[name].clone(), prefix=name)
        assert state[name].data.dtype == torch.float32
        assert all(shard.dtype == torch.float32 for shard in state[name].build())
    with TempNamedDir(tmp_path_dist_ckpt / "grouped-fc2") as path:
        dist_checkpointing.save(dict(state), path)
        loaded = dist_checkpointing.load(dict(state), path)
        for name, value in expected.items():
            torch.testing.assert_close(loaded[name], value, rtol=0, atol=0)


def test_dtensor_conversion_preserves_local_sharding(distributed):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    mesh = init_device_mesh('cuda', (torch.distributed.get_world_size(),))
    canonical = torch.arange(96, device='cuda').reshape(32, 3).float()
    tensor = distribute_tensor(canonical, mesh, [Shard(0)])
    converted = permute_glu_rows(tensor, 2, interleave=True)
    assert converted.placements == tensor.placements
    torch.testing.assert_close(converted.full_tensor(), _runtime(canonical, 2))
    torch.testing.assert_close(
        permute_glu_rows(converted, 2, interleave=False).full_tensor(), canonical
    )
