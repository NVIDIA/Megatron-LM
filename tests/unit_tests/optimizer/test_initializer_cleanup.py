# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Saved initializer lifetime at the public optimizer factory boundary."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

import megatron.core.optimizer as optimizer_module
from megatron.core.models.mimo import optimizer as mimo_optimizer_module
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from tests.unit_tests.test_utilities import Utils


class _ParameterWithInitializer(torch.nn.Parameter):
    """Expose TE's saved-initializer API without requiring FP8-capable hardware."""

    def get_high_precision_init_val(self):
        return self._high_precision_init_val

    def clear_high_precision_init_val(self):
        assert self._high_precision_init_val is not None
        self._high_precision_init_val = None


def _parameter(device='cpu', requires_grad=True):
    param = _ParameterWithInitializer(
        torch.zeros(4, 4, dtype=torch.bfloat16, device=device), requires_grad=requires_grad
    )
    param._high_precision_init_val = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
    return param


def _model():
    model = torch.nn.Module()
    model.owned = _parameter()
    model.non_owned = _parameter()
    model.frozen = _parameter(requires_grad=False)
    model.plain = torch.nn.Parameter(torch.zeros(1))
    model.ddp_config = SimpleNamespace(
        use_megatron_fsdp=False, data_parallel_sharding_strategy='no_shard'
    )
    return model


@pytest.mark.parametrize('path', ['muon', 'mimo', 'adam', 'sgd', 'fsdp', 'fsdp_overlap'])
def test_factory_clears_initializers_after_all_master_weights(monkeypatch, path):
    """Every successful return releases non-owned storage without clearing it too early."""
    chunks = [_model(), _model()]
    config = OptimizerConfig(
        optimizer=path if path in ('muon', 'adam', 'sgd') else 'adam',
        lr=0.1,
        overlap_param_gather_with_optimizer_step=path == 'fsdp_overlap',
    )
    built = []
    masters = []

    def build(model_chunks, **kwargs):
        # All sub-optimizers must finish before sweeping the model, including chunks
        # processed by earlier sub-optimizers and parameters outside their ownership.
        for chunk in chunks:
            assert chunk.non_owned.get_high_precision_init_val() is not None
            assert chunk.frozen.get_high_precision_init_val() is not None
        for chunk in model_chunks:
            masters.append(chunk.owned.get_high_precision_init_val().float().clone())
            chunk.owned.clear_high_precision_init_val()
        optimizer = object()
        built.append(optimizer)
        return optimizer

    if path == 'mimo':
        model = MimoModel.__new__(MimoModel)
        torch.nn.Module.__init__(model)
        model.chunks = torch.nn.ModuleList(chunks)
        monkeypatch.setattr(
            mimo_optimizer_module, 'get_mimo_optimizer', lambda model, config: build(model.chunks)
        )
        factory_chunks = [model]
    else:
        factory_chunks = chunks
        monkeypatch.setattr(optimizer_module, '_get_megatron_emerging_optimizer', build)
        monkeypatch.setattr(
            optimizer_module, '_get_megatron_optimizer_based_on_param_groups', build
        )
        monkeypatch.setattr(optimizer_module, 'ChainedOptimizer', tuple)
        monkeypatch.setattr(optimizer_module, 'get_pg_rank', lambda group: 0)
        monkeypatch.setattr(optimizer_module, 'get_pg_size', lambda group: 1)
        monkeypatch.setattr(
            ProcessGroupCollection,
            'setup_process_groups_for_optimizer',
            lambda *args, **kwargs: dict.fromkeys(
                (
                    'dp_cp_group',
                    'intra_dp_cp_group',
                    'intra_expt_dp_group',
                    'mp_group',
                    'expt_tp_pp_group',
                    'expt_tp_pp_with_egtp_remat_group',
                    'intra_dp_cp_group_gloo',
                    'intra_expt_dp_group_gloo',
                    'intra_dist_opt_group',
                )
            ),
        )
        monkeypatch.setattr(
            optimizer_module, '_get_param_groups_and_buffers', lambda *args, **kwargs: ([], {})
        )
        for chunk in chunks:
            chunk.ddp_config.use_megatron_fsdp = path.startswith('fsdp')

    result = get_megatron_optimizer(config, factory_chunks, config_overrides={})

    if path in ('muon', 'mimo', 'fsdp'):
        assert result is built[0]
    else:
        assert result == tuple(built)
    assert len(masters) == len(chunks)
    for master in masters:
        torch.testing.assert_close(master, torch.arange(16).reshape(4, 4).float(), rtol=0, atol=0)
    for chunk in chunks:
        for param in (chunk.owned, chunk.non_owned, chunk.frozen):
            assert param.get_high_precision_init_val() is None
    # Repeated/shared parameters and already-consumed initializers remain safe to visit.
    optimizer_module._clear_high_precision_initializers(chunks + chunks)


def test_failed_construction_preserves_initializers(monkeypatch):
    """Do not sweep initializers if optimizer construction raises."""
    model = _model()

    def fail(**kwargs):
        raise RuntimeError('construction failed')

    monkeypatch.setattr(optimizer_module, '_get_megatron_emerging_optimizer', fail)
    with pytest.raises(RuntimeError, match='construction failed'):
        get_megatron_optimizer(OptimizerConfig(optimizer='muon'), [model])
    for param in (model.owned, model.non_owned, model.frozen):
        assert param.get_high_precision_init_val() is not None


@pytest.mark.skipif(
    not optimizer_module.HAVE_EMERGING_OPTIMIZERS, reason='requires emerging-optimizers'
)
@pytest.mark.parametrize('expert', [False, True])
def test_layer_wise_muon_clears_non_owned_initializers(expert):
    """Real distributed Muon must retain exact local masters and no remote initializers."""
    Utils.initialize_model_parallel()
    try:
        if torch.distributed.get_world_size() < 2:
            pytest.skip('requires at least two ranks')
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        model = torch.nn.Module()
        model.config = SimpleNamespace(num_attention_heads=1, num_query_groups=1, kv_channels=4)
        model.ddp_config = SimpleNamespace(
            use_distributed_optimizer=False,
            overlap_param_gather=False,
            param_sync_via_bucket_group=False,
        )
        model.weights = torch.nn.ParameterList(
            [_parameter(device='cuda') for _ in range(2 * torch.distributed.get_world_size())]
        )
        for index, param in enumerate(model.weights):
            param.allreduce = not (expert and index % 2)
        config = OptimizerConfig(
            optimizer='muon',
            lr=0.01,
            bf16=True,
            use_layer_wise_distributed_optimizer=True,
            muon_split_qkv=False,
            clip_grad=0.0,
        )

        optimizer = get_megatron_optimizer(
            config, [model], pg_collection=pg_collection, use_gloo_process_groups=False
        )

        owned = [
            param
            for child in optimizer.chained_optimizers
            for group in child.float16_groups
            for param in group
        ]
        assert 0 < len(owned) < len(model.weights)
        for param in owned:
            torch.testing.assert_close(
                param.main_param,
                torch.arange(16, device='cuda', dtype=torch.float32).reshape(4, 4),
                rtol=0,
                atol=0,
            )
        assert all(param.get_high_precision_init_val() is None for param in model.weights)

        for param in model.weights:
            param.main_grad = torch.ones_like(param, dtype=torch.float32)
        assert optimizer.step()[0]
        checkpoint = deepcopy(optimizer.state_dict())
        first_update = [param.main_param.clone() for param in owned]
        assert optimizer.step()[0]
        second_update = [param.main_param.clone() for param in owned]
        assert any(
            not torch.equal(first, second) for first, second in zip(first_update, second_update)
        )

        optimizer.load_state_dict(checkpoint)
        for param, expected in zip(owned, first_update):
            torch.testing.assert_close(param.main_param, expected, rtol=0, atol=0)
        assert optimizer.step()[0]
        for param, expected in zip(owned, second_update):
            torch.testing.assert_close(param.main_param, expected, rtol=0, atol=0)
        assert all(param.get_high_precision_init_val() is None for param in model.weights)
    finally:
        Utils.destroy_model_parallel()
