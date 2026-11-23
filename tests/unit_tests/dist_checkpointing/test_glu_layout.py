# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Canonical GLU checkpoints preserve parameter identity and optimizer row semantics."""

from argparse import Namespace
from dataclasses import replace
from functools import partial

import pytest
import torch

from megatron.core import dist_checkpointing, parallel_state
from megatron.core.dist_checkpointing.glu import (
    glu_checkpoint_metadata,
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
def test_real_grouped_mlp_factories_use_the_configured_layout(distributed, single_weight):
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
    layer = MoELayer(config, submodules)
    for module, size in ((layer.experts, 2), (layer.shared_experts, 4)):
        state = module.sharded_state_dict(
            metadata={'dp_cp_group': parallel_state.get_data_parallel_group()}
        )
        factories = [
            value
            for key, value in state.items()
            if 'linear_fc1.weight' in key and isinstance(value, ShardedTensorFactory)
        ]
        assert factories
        for factory in factories:
            canonical = torch.arange(
                factory.data.numel(), device=factory.data.device, dtype=factory.data.dtype
            ).reshape_as(factory.data)
            with torch.no_grad():
                factory.data.copy_(_runtime(canonical, size))
            parts = factory.build()
            torch.testing.assert_close(
                torch.cat([part.data for part in parts]), canonical, rtol=0, atol=0
            )


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


@pytest.mark.parametrize('memory_efficient', [False, True])
def test_distributed_adam_resume_preserves_three_updates(
    tmp_path_dist_ckpt, distributed, memory_efficient
):
    metadata = {
        'distrib_optim_sharding_type': 'fully_reshardable',
        'distrib_optim_fully_reshardable_mem_efficient': memory_efficient,
        **glu_checkpoint_metadata(_config(2), optimizer_uses_model_layout=True),
    }

    def initialize(size):
        models, optimizer = setup_model_and_optimizer(
            seed=5, tp=1, pp=1, initialize_fn=partial(_initialize_interleaved, size=size)
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
        for step in range(2, 5):
            update(model_a, optimizer_a, step)
            update(model_b, optimizer_b, step)
            torch.testing.assert_close(
                permute_glu_rows(model_a.linear.weight, 2, interleave=False),
                permute_glu_rows(model_b.linear.weight, 4, interleave=False),
                rtol=0,
                atol=0,
            )
            states = [
                optimizer.get_parameter_state_dp_zero(use_gloo_comm=False, return_on_all_ranks=True)
                for optimizer in (optimizer_a, optimizer_b)
            ]
            buffers = [next(iter(state[0].values())) for state in states]
            for name in ('param', 'exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(
                    permute_glu_rows(buffers[0][name].reshape(64, 5), 2, interleave=False),
                    permute_glu_rows(buffers[1][name].reshape(64, 5), 4, interleave=False),
                    rtol=0,
                    atol=0,
                )
