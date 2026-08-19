# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Config-to-kwargs dispatch tests for the emerging-optimizer registry.

Regression coverage for kwargs pollution across registry entries: the
layer-sharding branch must live ONLY on the ``muon`` entry's dispatcher
(``_muon_entry_config_to_kwargs``), never in the shared
``_muon_config_to_kwargs`` that ``adaptive_muon`` layers its own kwargs on.
When the branch leaked into the shared helper,
``--optimizer adaptive_muon --use-layer-sharding-muon`` crashed at
construction: TensorParallelAdaptiveMuon received LayerShardedMuon-only
kwargs (gtp_group, ns_batch_size, ...) and lost the ones it needs
(is_qkv_fn, qkv_split_shapes, pg_collection).

Pure single-process tests: no distributed init, no GPU.
"""

import inspect

import pytest

pytest.importorskip("emerging_optimizers", reason="requires emerging-optimizers")

from megatron.core.optimizer import emerging_optimizers as eo_mod
from megatron.core.optimizer.layer_sharded_muon import LayerShardedMuon
from megatron.core.optimizer.optimizer_config import OptimizerConfig

# No version guard on purpose: nothing here constructs LayerShardedMuon (whose
# __init__ raises on emerging-optimizers < 0.3.0), so these tests run — and give
# real CI signal — even on containers pinned to older emerging-optimizers.
# (A try/except around the import would guard nothing anyway: layer_sharded_muon
# swallows its own emerging-optimizers import failures behind
# HAVE_EMERGING_OPTIMIZERS, so the import always succeeds.)
# INVARIANT for future additions: tests in this file must stay signature/registry
# level (inspect, kwargs dicts) and never instantiate an optimizer — the moment
# one does, it inherits LayerShardedMuon's version-conditional construction
# behavior and needs explicit skip/bypass handling to keep below-floor CI coverage.


class _ModelCfg:
    num_attention_heads = 8
    num_query_groups = 2
    kv_channels = 16


class _Chunk:
    config = _ModelCfg()


def _Cfg(use_layer_sharding_muon: bool) -> OptimizerConfig:
    """A real OptimizerConfig, so the reflective ``_kwargs_from_config`` lookups
    exercise the actual field surface instead of a stub that makes every
    ``hasattr`` fail (which would shrink the test to the hardcoded keys only)."""
    return OptimizerConfig(use_layer_sharding_muon=use_layer_sharding_muon)


def test_shared_muon_kwargs_ignore_layer_sharding_flag():
    """The shared builder must stay pure even with the lsh flag set."""
    kwargs = eo_mod._muon_config_to_kwargs(_Cfg(True), [_Chunk()], pg_collection=None)
    for lsh_only in ("gtp_group", "tp_group", "fused_group", "ns_batch_size"):
        assert lsh_only not in kwargs, f"LayerShardedMuon-only kwarg leaked: {lsh_only}"
    assert "is_qkv_fn" in kwargs
    assert "qkv_split_shapes" in kwargs
    assert "pg_collection" in kwargs


@pytest.mark.parametrize("lsh_flag", [False, True])
def test_adaptive_muon_kwargs_match_constructor_signature(lsh_flag):
    """Every kwarg built for adaptive_muon must be accepted by its __init__.

    This is the construction-crash regression: with the flag set, a polluted
    shared helper produced gtp_group/ns_batch_size and TensorParallelAdaptiveMuon
    raised TypeError before training started.
    """
    kwargs = eo_mod._adaptive_muon_config_to_kwargs(_Cfg(lsh_flag), [_Chunk()], pg_collection=None)
    accepted = set(inspect.signature(eo_mod.TensorParallelAdaptiveMuon.__init__).parameters)
    unexpected = set(kwargs) - accepted
    assert not unexpected, f"kwargs TensorParallelAdaptiveMuon.__init__ rejects: {unexpected}"


def test_muon_entry_dispatches_on_layer_sharding_flag():
    """The muon entry's dispatcher pairs lsh kwargs with the lsh class."""
    lsh_kwargs = eo_mod._muon_entry_config_to_kwargs(_Cfg(True), [_Chunk()], pg_collection=None)
    assert "gtp_group" in lsh_kwargs
    assert eo_mod._muon_config_to_cls(_Cfg(True)) is LayerShardedMuon

    plain_kwargs = eo_mod._muon_entry_config_to_kwargs(_Cfg(False), [_Chunk()], pg_collection=None)
    assert "gtp_group" not in plain_kwargs
    assert "is_qkv_fn" in plain_kwargs
    assert eo_mod._muon_config_to_cls(_Cfg(False)) is eo_mod.TensorParallelMuon


def test_muon_entry_registered_with_dispatcher():
    """The registry wires the dispatcher to 'muon' and leaves adaptive_muon alone."""
    muon_entry = eo_mod._EMERGING_OPTIMIZERS["muon"]
    assert muon_entry.config_to_kwargs is eo_mod._muon_entry_config_to_kwargs
    adaptive_entry = eo_mod._EMERGING_OPTIMIZERS["adaptive_muon"]
    assert adaptive_entry.config_to_kwargs is eo_mod._adaptive_muon_config_to_kwargs
