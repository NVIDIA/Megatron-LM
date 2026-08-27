# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Hybrid Muon NS modes: per-bucket optimizer class dispatch.

``--muon-expert-tp-mode`` gives expert-parallel weights their own NS mode
(default: follow ``--muon-tp-mode``). When the effective modes differ,
``_create_emerging_optimizer`` is called once per (dense, expert) bucket and
must build the class each mode selects — e.g. TensorParallelMuon with
``tp_mode='auto'`` for the dense bucket and LayerShardedMuon for the expert
bucket. With the modes equal (the default follow), the pre-existing registry
dispatch is unchanged.

Single-process, no distributed init, no GPU. Constructs real optimizers at the
strict defaults (``ns_batch_size=1``), which the scoped EO version gate allows
on every emerging-optimizers release.
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


class _PGStub:
    """Duck-typed ProcessGroupCollection: distinct sentinels per field so the
    tests can assert WHICH domain's groups each bucket's optimizer received."""

    def __init__(self):
        self.gtp_remat = object()
        self.expt_gtp_remat = object()
        self.tp = object()
        self.expt_tp = object()


def _cfg(
    tp_mode: str = 'duplicated',
    expert_tp_mode: "str | None" = None,
    split_qkv: bool = False,
) -> OptimizerConfig:
    # muon_split_qkv defaults False here, mirroring what validate_args enforces
    # whenever the DENSE mode is layer_sharded (LayerShardedMuon rejects
    # split_qkv at construction).
    return OptimizerConfig(
        muon_tp_mode=tp_mode,
        muon_expert_tp_mode=expert_tp_mode,
        muon_split_qkv=split_qkv,
    )


def _bucket(is_expert: bool):
    """One param group shaped like _get_param_groups output for one bucket."""
    return [
        {
            'params': [torch.nn.Parameter(torch.randn(8, 8))],
            'is_expert_parallel': is_expert,
        }
    ]


def _create(cfg, groups, pg_collection=None):
    optimizer, init_state_fn = eo_mod._create_emerging_optimizer(
        cfg, groups, 'muon', [_Chunk()], pg_collection=pg_collection
    )
    assert init_state_fn is eo_mod._EMERGING_OPTIMIZERS['muon'].init_state_fn
    return optimizer


def test_hybrid_routes_buckets_to_different_classes():
    cfg = _cfg(tp_mode='auto', expert_tp_mode='layer_sharded')

    dense_opt = _create(cfg, _bucket(is_expert=False))
    assert type(dense_opt) is eo_mod.TensorParallelMuon
    assert dense_opt.tp_mode == 'auto', "dense bucket must honor muon_tp_mode"

    expert_opt = _create(cfg, _bucket(is_expert=True))
    assert isinstance(expert_opt, LayerShardedMuon)
    # 'layer_sharded' is the registry selector; the class receives the bitwise
    # reference mode for its delegated (fallback/degenerate) paths.
    assert expert_opt.tp_mode == 'duplicated'


def test_hybrid_expert_bucket_gets_expert_domain_groups():
    pgs = _PGStub()
    cfg = _cfg(tp_mode='auto', expert_tp_mode='layer_sharded')

    expert_opt = _create(cfg, _bucket(is_expert=True), pg_collection=pgs)
    assert expert_opt.gtp_remat_group is pgs.expt_gtp_remat
    assert expert_opt.tp_group is pgs.expt_tp

    # Reverse hybrid: the DENSE bucket's LayerShardedMuon keeps dense domains.
    cfg_rev = _cfg(tp_mode='layer_sharded', expert_tp_mode='duplicated')
    dense_opt = _create(cfg_rev, _bucket(is_expert=False), pg_collection=pgs)
    assert isinstance(dense_opt, LayerShardedMuon)
    assert dense_opt.gtp_remat_group is pgs.gtp_remat
    assert dense_opt.tp_group is pgs.tp


def test_reverse_hybrid_expert_bucket_is_tensor_parallel_muon():
    cfg = _cfg(tp_mode='layer_sharded', expert_tp_mode='duplicated')
    expert_opt = _create(cfg, _bucket(is_expert=True))
    assert type(expert_opt) is eo_mod.TensorParallelMuon
    assert expert_opt.tp_mode == 'duplicated'


def test_expert_mode_follows_dense_by_default():
    """Unset expert mode is NOT hybrid: the registry path is untouched and both
    buckets get the class the (single) mode selects."""
    for tp_mode, expected in (
        ('layer_sharded', LayerShardedMuon),
        ('auto', eo_mod.TensorParallelMuon),
    ):
        cfg = _cfg(tp_mode=tp_mode)
        assert not eo_mod._muon_modes_are_hybrid(cfg)
        for is_expert in (False, True):
            optimizer = _create(cfg, _bucket(is_expert))
            assert isinstance(optimizer, expected), (
                f"tp_mode={tp_mode}, is_expert={is_expert}"
            )


def test_hybrid_expert_bucket_tolerates_split_qkv():
    """With hybrid modes, split-QKV may be enabled for the dense (TPM) bucket;
    the expert LayerShardedMuon bucket must not trip over it — the dispatcher
    forces split_qkv off there (expert weights own no QKV)."""
    cfg = _cfg(tp_mode='auto', expert_tp_mode='layer_sharded', split_qkv=True)
    dense_opt = _create(cfg, _bucket(is_expert=False))
    assert dense_opt.split_qkv is True, "dense bucket must keep split-QKV"
    expert_opt = _create(cfg, _bucket(is_expert=True))
    assert isinstance(expert_opt, LayerShardedMuon)
    assert expert_opt.split_qkv is False


def test_explicit_expert_mode_equal_to_dense_is_not_hybrid():
    cfg = _cfg(tp_mode='auto', expert_tp_mode='auto')
    assert not eo_mod._muon_modes_are_hybrid(cfg)
    optimizer = _create(cfg, _bucket(is_expert=True))
    assert type(optimizer) is eo_mod.TensorParallelMuon
