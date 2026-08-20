# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Hybrid layer-sharding scope: per-bucket optimizer class dispatch.

With ``use_layer_sharding_muon`` and ``muon_lsh_scope='expert'``,
``_create_emerging_optimizer`` is called once per (dense, expert) bucket and must
build TensorParallelMuon for the dense bucket (so ``muon_tp_mode``, including
``'auto'``, governs dense weights) and LayerShardedMuon for the expert bucket.
With the default scope ``'all'`` the pre-existing behavior is unchanged:
LayerShardedMuon for everything via the registry's ``config_to_cls``.

Single-process, no distributed init, no GPU. Constructs real optimizers at the
strict defaults (``ns_batch_size=1``), which the scoped EO version gate allows on
every emerging-optimizers release including the CI-pinned 0.2.0.
"""

import pytest
import torch

pytest.importorskip("emerging_optimizers", reason="requires emerging-optimizers")

from megatron.core.optimizer import emerging_optimizers as eo_mod
from megatron.core.optimizer.layer_sharded_muon import LayerShardedMuon
from megatron.core.optimizer.optimizer_config import OptimizerConfig


class _ModelCfg:
    num_attention_heads = 8
    num_query_groups = 2
    kv_channels = 16


class _Chunk:
    config = _ModelCfg()


def _cfg(scope: str, tp_mode: str = 'blockwise') -> OptimizerConfig:
    return OptimizerConfig(
        use_layer_sharding_muon=True, muon_lsh_scope=scope, muon_tp_mode=tp_mode
    )


def _bucket(is_expert: bool):
    """One param group shaped like _get_param_groups output for one bucket."""
    return [
        {
            'params': [torch.nn.Parameter(torch.randn(8, 8))],
            'is_expert_parallel': is_expert,
        }
    ]


def _create(cfg, groups):
    optimizer, init_state_fn = eo_mod._create_emerging_optimizer(
        cfg, groups, 'muon', [_Chunk()], pg_collection=None
    )
    assert init_state_fn is eo_mod._EMERGING_OPTIMIZERS['muon'].init_state_fn
    return optimizer


def test_scope_expert_routes_buckets_to_different_classes():
    cfg = _cfg('expert', tp_mode='auto')

    dense_opt = _create(cfg, _bucket(is_expert=False))
    assert isinstance(dense_opt, eo_mod.TensorParallelMuon)
    assert not isinstance(dense_opt, LayerShardedMuon)
    assert dense_opt.tp_mode == 'auto', "dense bucket must honor muon_tp_mode"

    expert_opt = _create(cfg, _bucket(is_expert=True))
    assert isinstance(expert_opt, LayerShardedMuon)


def test_scope_all_keeps_layer_sharded_for_every_bucket():
    cfg = _cfg('all')
    for is_expert in (False, True):
        optimizer = _create(cfg, _bucket(is_expert))
        assert isinstance(optimizer, LayerShardedMuon), (
            f"scope='all' must keep LayerShardedMuon (is_expert={is_expert})"
        )


def test_scope_expert_without_lsh_flag_is_inert():
    """muon_lsh_scope only means something under use_layer_sharding_muon."""
    cfg = OptimizerConfig(
        use_layer_sharding_muon=False, muon_lsh_scope='expert', muon_tp_mode='auto'
    )
    for is_expert in (False, True):
        optimizer = _create(cfg, _bucket(is_expert))
        assert isinstance(optimizer, eo_mod.TensorParallelMuon)
        assert not isinstance(optimizer, LayerShardedMuon)


def test_scope_expert_dense_kwargs_are_tensor_parallel_muon_kwargs():
    """The dense bucket must get the TPM kwargs surface (qkv fn, pg_collection),
    not LayerShardedMuon-only kwargs — the adaptive_muon kwargs-pollution bug class."""
    cfg = _cfg('expert')
    dense_opt = _create(cfg, _bucket(is_expert=False))
    assert hasattr(dense_opt, 'is_qkv_fn')
    assert not hasattr(dense_opt, 'gtp_group'), "LayerShardedMuon-only attr leaked"
