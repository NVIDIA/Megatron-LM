# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the layer-sharded Muon all_to_all routing primitives.

The forward exchange must assemble each home's complete matrices from the
per-rank shards; the backward exchange must scatter per-home results back to
the exact originating shards; and a fwd -> identity -> bwd roundtrip must
reproduce the input bit-for-bit. All three are pure routing properties -- no
Newton-Schulz involved -- so any failure is an indexing bug, not numerics.

Launch with torchrun (the fixtures initialize the torchrun-managed group):
  torchrun --nproc-per-node=4 -m pytest tests/unit_tests/optimizer/test_layer_sharded_a2a.py
"""

import pytest
import torch
import torch.distributed as dist

from megatron.core.optimizer.layer_sharded_a2a import (
    layer_sharded_all_to_all_bwd,
    layer_sharded_all_to_all_fwd,
)
from tests.unit_tests.test_utilities import Utils

_SEED = 42


@pytest.fixture(scope="module", autouse=True)
def _torchrun_dist_init():
    Utils.initialize_model_parallel()
    # Utils uses the NCCL backend, so the exchanged tensors must live on this
    # rank's GPU; defaulting the device keeps every torch.randn in the tests
    # device-agnostic. (Utils already bound the rank to its device.)
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    yield
    if torch.cuda.is_available():
        torch.set_default_device("cpu")
    Utils.destroy_model_parallel()


def _world():
    return dist.group.WORLD


def _require_multi_rank():
    if dist.get_world_size() < 2:
        pytest.skip("Requires >= 2 ranks (launch with torchrun --nproc-per-node=4)")


def test_fwd_reconstructs_complete_matrix():
    """Each NS-home rank assembles the full (P, Q) matrix from the row shards."""
    _require_multi_rank()
    S, r = dist.get_world_size(), dist.get_rank()
    P, Q = 16 * S, 8
    N = S

    # Same seed on all ranks: every rank derives its shard from identical full
    # tensors, so each home can be checked against ground truth locally.
    torch.manual_seed(_SEED)
    full = [torch.randn(P, Q) for _ in range(N)]
    shards = [m[r * (P // S) : (r + 1) * (P // S), :].clone() for m in full]
    homes = {i: i % S for i in range(N)}

    complete, my_indices = layer_sharded_all_to_all_fwd(shards, homes, r, S, _world(), gtp_dim=0)

    assert my_indices == [i for i in range(N) if i % S == r]
    for got, idx in zip(complete, my_indices):
        torch.testing.assert_close(
            got, full[idx], msg=lambda m: f"reconstruction of param {idx} on rank {r}\n\n{m}"
        )


def test_bwd_distributes_shards_correctly():
    """Every rank receives exactly its row shard of each home's result."""
    _require_multi_rank()
    S, r = dist.get_world_size(), dist.get_rank()
    P, Q = 16 * S, 8
    N = S

    torch.manual_seed(_SEED + 1)
    full_results = [torch.randn(P, Q) for _ in range(N)]
    my_results = [full_results[i] for i in range(N) if i % S == r]
    my_indices = [i for i in range(N) if i % S == r]
    templates = [torch.empty(P // S, Q) for _ in range(N)]
    homes = {i: i % S for i in range(N)}

    shards = layer_sharded_all_to_all_bwd(
        my_results, my_indices, templates, homes, r, S, _world(), gtp_dim=0
    )

    assert len(shards) == N
    for i, shard in enumerate(shards):
        assert shard is not None, f"missing shard for param {i}"
        expected = full_results[i][r * (P // S) : (r + 1) * (P // S), :]
        torch.testing.assert_close(
            shard, expected, msg=lambda m: f"shard of param {i} on rank {r}\n\n{m}"
        )


def test_roundtrip_without_ns_is_identity():
    """fwd -> identity -> bwd returns the original shards, heterogeneous shapes included.

    Shapes and assignments are deliberately uneven (different sizes, one rank
    with more homes than others) to exercise the per-param split-size paths.
    """
    _require_multi_rank()
    S, r = dist.get_world_size(), dist.get_rank()

    torch.manual_seed(_SEED + 2)
    shapes = [(16 * S, 8), (4 * S, 24), (16 * S, 8), (8 * S, 4)]
    full = [torch.randn(*s) for s in shapes]
    shards = [m[r * (m.size(0) // S) : (r + 1) * (m.size(0) // S), :].clone() for m in full]
    homes = {0: 0, 1: 0, 2: min(1, S - 1), 3: min(2, S - 1)}

    complete, my_indices = layer_sharded_all_to_all_fwd(shards, homes, r, S, _world(), gtp_dim=0)
    identity = [m.clone() for m in complete]
    recovered = layer_sharded_all_to_all_bwd(
        identity, my_indices, shards, homes, r, S, _world(), gtp_dim=0
    )

    for i, (orig, back) in enumerate(zip(shards, recovered)):
        assert back is not None, f"missing roundtrip result for param {i}"
        assert torch.equal(orig, back), (
            f"roundtrip changed param {i} on rank {r}: "
            f"max |diff| = {(orig - back).abs().max().item():.3e}"
        )
