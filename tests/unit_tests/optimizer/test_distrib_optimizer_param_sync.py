# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer


def make_bucket(layer_wise=False):
    """Construct a bucket carrying the DDP optimizer-routing tag."""
    return SimpleNamespace(
        params_list=[SimpleNamespace(is_managed_by_layer_wise_optimizer=layer_wise)]
    )


@pytest.mark.parametrize('expert_owner', [False, True])
@pytest.mark.parametrize('num_chunks', [1, 2])
def test_shared_ddp_syncs_only_owned_buckets(expert_owner, num_chunks):
    """A dense/expert sibling must not resync the other optimizer's DDP groups."""
    optimizer = DistributedOptimizer.__new__(DistributedOptimizer)
    optimizer.is_stub_optimizer = False
    optimizer.buffers = []
    optimizer.model_chunks = []
    expected = []
    for _ in range(num_chunks):
        # Multiple buckets in one DDP group must retain their grouping/copyback.
        dense = SimpleNamespace(buckets=[make_bucket(), make_bucket()])
        expert = SimpleNamespace(buckets=[make_bucket(), make_bucket()])
        layer_wise = SimpleNamespace(buckets=[make_bucket(layer_wise=True)])
        chunk = SimpleNamespace(
            bucket_groups=[dense, layer_wise, SimpleNamespace(buckets=[])],
            expert_parallel_bucket_groups=[expert],
            _start_bucket_group_param_sync=Mock(),
        )
        owned = expert if expert_owner else dense
        optimizer.buffers.extend(SimpleNamespace(buckets=[b]) for b in owned.buckets)
        optimizer.model_chunks.append(chunk)
        expected.append((chunk, owned))

    optimizer.start_param_sync_for_bucket_group_subset()

    for chunk, owned in expected:
        assert chunk._start_bucket_group_param_sync.call_args_list == [
            call(owned, force_sync=False)
        ]


def test_stub_optimizer_does_not_sync_shared_model():
    """A stub has no buffers, even when its model has a trainable sibling."""
    optimizer = DistributedOptimizer.__new__(DistributedOptimizer)
    optimizer.is_stub_optimizer = True
    optimizer.model_chunks = [SimpleNamespace(_start_bucket_group_param_sync=Mock())]
    optimizer.start_param_sync_for_bucket_group_subset()
    optimizer.model_chunks[0]._start_bucket_group_param_sync.assert_not_called()


def test_mixed_ownership_group_is_rejected_before_sync():
    """Partially owned groups must fail rather than gather stale sibling parameters."""
    owned, sibling = make_bucket(), make_bucket()
    group = SimpleNamespace(buckets=[owned, sibling])
    chunk = SimpleNamespace(
        bucket_groups=[group],
        expert_parallel_bucket_groups=[],
        _start_bucket_group_param_sync=Mock(),
    )
    optimizer = DistributedOptimizer.__new__(DistributedOptimizer)
    optimizer.is_stub_optimizer = False
    optimizer.buffers = [SimpleNamespace(buckets=[owned])]
    optimizer.model_chunks = [chunk]

    with pytest.raises(AssertionError, match='must not mix optimizer-owned buckets'):
        optimizer.start_param_sync_for_bucket_group_subset()

    chunk._start_bucket_group_param_sync.assert_not_called()


def test_untagged_owned_bucket_preserves_single_optimizer_sync():
    """Legacy untagged buckets still synchronize when owned by this optimizer."""
    bucket = SimpleNamespace(params_list=[SimpleNamespace()])
    group = SimpleNamespace(buckets=[bucket])
    chunk = SimpleNamespace(
        bucket_groups=[group],
        expert_parallel_bucket_groups=[],
        _start_bucket_group_param_sync=Mock(),
    )
    optimizer = DistributedOptimizer.__new__(DistributedOptimizer)
    optimizer.is_stub_optimizer = False
    optimizer.buffers = [SimpleNamespace(buckets=[bucket])]
    optimizer.model_chunks = [chunk]

    optimizer.start_param_sync_for_bucket_group_subset()

    chunk._start_bucket_group_param_sync.assert_called_once_with(group, force_sync=False)


def test_layer_wise_group_is_skipped_even_when_listed_in_buffers():
    """The pre-existing LayerWise exclusion takes precedence over buffer ownership."""
    bucket = make_bucket(layer_wise=True)
    group = SimpleNamespace(buckets=[bucket])
    chunk = SimpleNamespace(
        bucket_groups=[group],
        expert_parallel_bucket_groups=[],
        _start_bucket_group_param_sync=Mock(),
    )
    optimizer = DistributedOptimizer.__new__(DistributedOptimizer)
    optimizer.is_stub_optimizer = False
    optimizer.buffers = [SimpleNamespace(buckets=[bucket])]
    optimizer.model_chunks = [chunk]

    optimizer.start_param_sync_for_bucket_group_subset()

    chunk._start_bucket_group_param_sync.assert_not_called()
