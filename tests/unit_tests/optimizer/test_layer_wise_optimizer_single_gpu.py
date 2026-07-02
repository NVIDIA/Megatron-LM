# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Regression test for the single-GPU (dp_cp_size == 1) path of
``LayerWiseDistributedOptimizer.set_bucket_layerwise_params_list`` (issue #5203).

When the data-parallel-with-context-parallel group has size 1, ``_shard_params``
sets ``self.dp_cp_params_list = None``. ``set_bucket_layerwise_params_list`` must
handle that case instead of iterating over ``None``. The standard unit-test
harness runs every test at a fixed world size (> 1), so the world_size == 1
condition cannot be expressed there; this is therefore a lightweight,
collective-free test that drives the method directly with a minimal stand-in
optimizer and a mocked single-rank group.
"""
from types import SimpleNamespace

import pytest

from megatron.core.optimizer import layer_wise_optimizer as lwo
from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer


class _FakeBucket:
    def __init__(self, params):
        self.params_list = list(params)
        self.params = set(params)
        self.layerwise_params_list = None

    def set_layerwise_params_list(self, params_list):
        self.layerwise_params_list = params_list


def _make_optimizer_stub():
    """Minimal stand-in exposing only what the method under test reads."""
    return SimpleNamespace(
        dp_cp_params_list=None,  # set by _shard_params when dp_cp_size == 1
        expt_dp_params_list=None,
        pg_collection=SimpleNamespace(dp_cp="dp_cp", expt_dp="expt_dp"),
    )


def test_set_bucket_layerwise_params_list_single_gpu(monkeypatch):
    """Single-rank: the bucket's params must all be assigned to rank 0 without
    raising ``TypeError: 'NoneType' object is not iterable``."""
    monkeypatch.setattr(lwo, "get_pg_size", lambda group: 1)
    monkeypatch.setattr(
        lwo, "_bucket_is_managed_by_layer_wise_optimizer", lambda bucket, *a, **k: True
    )

    p_w, p_b = object(), object()
    bucket = _FakeBucket([p_w, p_b])
    model_chunk = SimpleNamespace(
        bucket_groups=[SimpleNamespace(buckets=[bucket])],
        expert_parallel_bucket_groups=[],
    )

    optimizer = _make_optimizer_stub()
    LayerWiseDistributedOptimizer.set_bucket_layerwise_params_list(
        optimizer, [model_chunk]
    )

    assert bucket.layerwise_params_list == [[p_w, p_b]]
