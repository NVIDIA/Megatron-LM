# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fixed-capacity EP lookup (full-iteration CUDA graphs) against the dynamic exchange.

Runs under torchrun with 2, 4 or 8 ranks; each case uses an EP subgroup of the first
``ep_size`` ranks and the remaining ranks sit the case out after group creation.
"""

import os
from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.engram.distributed_embedding import (
    EPShardedMultiTableEmbedding,
    get_contiguous_row_range,
)
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) not in (2, 4, 8),
    reason="requires torchrun with two, four, or eight ranks",
)

TABLE_SIZES = (101, 103, 107, 109)
EMBEDDING_DIM = 5
TOKENS = 64


def _setup(ep_size, capacity_factor, dtype=torch.float64):
    Utils.initialize_distributed()
    rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
    if world_size < ep_size:
        pytest.skip(f"requires at least {ep_size} ranks")
    # Collective on the default group: every rank must participate in group creation.
    group = torch.distributed.new_group(ranks=list(range(ep_size)))
    if rank >= ep_size:
        return None
    config = SimpleNamespace(
        use_cpu_initialization=False,
        params_dtype=dtype,
        perform_initialization=False,
        sequence_parallel=False,
        deterministic_mode=False,
    )
    embedding = EPShardedMultiTableEmbedding(
        config=config,
        table_sizes=TABLE_SIZES,
        embedding_dim=EMBEDDING_DIM,
        init_method=lambda _: None,
        ep_group=group,
        tp_group=None,
        expt_dp_group=None,
        capacity_factor=capacity_factor,
    )
    with torch.no_grad():
        for table_id, table in enumerate(embedding.tables):
            rows = torch.arange(table.row_start, table.row_end, device="cuda", dtype=dtype)
            columns = torch.arange(EMBEDDING_DIM, device="cuda", dtype=dtype)
            table.weight.copy_(table_id * 1000.0 + rows[:, None] * 10.0 + columns[None, :])
    return rank, embedding


def _hash_ids(seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    columns = [
        torch.randint(0, size, (1, TOKENS), device="cuda", generator=generator)
        for size in TABLE_SIZES
    ]
    return torch.stack(columns, dim=-1)


def _lookup_and_grads(embedding, hash_ids, grad_output, dynamic):
    for table in embedding.tables:
        table.weight.grad = None
    # The dynamic (variable-split) exchange is what the lookup uses without a capacity factor.
    capacity_factor = embedding.capacity_factor
    if dynamic:
        embedding.capacity_factor = None
    try:
        output = embedding(hash_ids)
        output.backward(grad_output)
    finally:
        embedding.capacity_factor = capacity_factor
    return output.detach(), [table.weight.grad.clone() for table in embedding.tables]


@pytest.mark.parametrize("ep_size", [1, 2, 4])
def test_fixed_capacity_matches_dynamic_forward_and_table_gradients(ep_size):
    # A factor of ep_size gives every bucket room for all of a rank's tokens: no overflow.
    state = _setup(ep_size, capacity_factor=float(ep_size))
    if state is None:
        return
    rank, embedding = state
    hash_ids = _hash_ids(seed=17 + rank)
    grad_output = torch.randn(
        (*hash_ids.shape, EMBEDDING_DIM),
        device="cuda",
        dtype=torch.float64,
        generator=torch.Generator(device="cuda").manual_seed(5 + rank),
    )
    fixed_output, fixed_grads = _lookup_and_grads(embedding, hash_ids, grad_output, dynamic=False)
    dynamic_output, dynamic_grads = _lookup_and_grads(
        embedding, hash_ids, grad_output, dynamic=True
    )
    torch.testing.assert_close(fixed_output, dynamic_output, rtol=0, atol=0)
    for fixed_grad, dynamic_grad in zip(fixed_grads, dynamic_grads):
        torch.testing.assert_close(fixed_grad, dynamic_grad)


@pytest.mark.parametrize("ep_size", [2, 4])
def test_overflow_raises_with_the_factor_that_would_fit(ep_size):
    state = _setup(ep_size, capacity_factor=0.25)
    if state is None:
        return
    rank, embedding = state
    # Every request goes to EP rank 0, far beyond a 0.25 * TOKENS / ep_size bucket; every rank
    # overflows before the exchange, so all of them raise.
    hash_ids = torch.stack(
        [
            torch.randint(
                0, get_contiguous_row_range(size, 0, ep_size)[1], (1, TOKENS), device="cuda"
            )
            for size in TABLE_SIZES
        ],
        dim=-1,
    )
    with pytest.raises(ValueError, match=rf"engram-ep-capacity-factor.*{ep_size}\.00"):
        embedding(hash_ids)


@pytest.mark.parametrize("ep_size", [2, 4])
def test_captured_lookup_replays_with_new_tokens(ep_size):
    """Capture fwd+bwd once, replay with different tokens: nothing may be baked at capture."""
    state = _setup(ep_size, capacity_factor=2.0, dtype=torch.float32)
    if state is None:
        return
    rank, embedding = state
    static_ids = _hash_ids(seed=100 + rank)
    grad_output = torch.randn(
        (*static_ids.shape, EMBEDDING_DIM),
        device="cuda",
        generator=torch.Generator(device="cuda").manual_seed(9 + rank),
    )
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            for table in embedding.tables:
                table.weight.grad = None
            embedding(static_ids).backward(grad_output)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    for table in embedding.tables:
        table.weight.grad = None

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = embedding(static_ids)
        static_output.backward(grad_output)
    # The graph owns these gradient tensors; the eager reference below rebinds .grad.
    static_grads = [table.weight.grad for table in embedding.tables]

    for seed in (200, 300):
        new_ids = _hash_ids(seed=seed + rank)
        static_ids.copy_(new_ids)
        graph.replay()
        torch.cuda.synchronize()
        graph_output = static_output.clone()
        graph_grads = [grad.clone() for grad in static_grads]

        expected_output, expected_grads = _lookup_and_grads(
            embedding, new_ids, grad_output, dynamic=True
        )
        torch.testing.assert_close(graph_output, expected_output, rtol=0, atol=0)
        for graph_grad, expected_grad in zip(graph_grads, expected_grads):
            torch.testing.assert_close(graph_grad, expected_grad)
    del graph


@pytest.mark.parametrize("ep_size", [2, 4])
def test_invalid_requests_take_no_bucket_slot(ep_size):
    """THD padding slots hash to the same n-gram; masked out, they cannot overflow a bucket."""
    state = _setup(ep_size, capacity_factor=1.0)
    if state is None:
        return
    rank, embedding = state
    hash_ids = _hash_ids(seed=41 + rank)
    # The padded half of the row repeats one row per table: far beyond one bucket's
    # 1.0 * TOKENS / ep_size slots, exactly what a padding tail looks like.
    hash_ids[:, TOKENS // 2 :] = hash_ids[:, :1]
    valid = torch.arange(TOKENS, device="cuda").view(1, TOKENS) < TOKENS // 2
    grad_output = torch.randn(
        (*hash_ids.shape, EMBEDDING_DIM),
        device="cuda",
        dtype=torch.float64,
        generator=torch.Generator(device="cuda").manual_seed(3 + rank),
    )
    masked = valid[..., None, None].to(torch.float64)

    for table in embedding.tables:
        table.weight.grad = None
    output = embedding(hash_ids, valid=valid) * masked
    output.backward(grad_output)
    grads = [table.weight.grad.clone() for table in embedding.tables]

    # Unmasked, the same padding overflows the fixed-capacity bucket.
    with torch.no_grad(), pytest.raises(ValueError, match="engram-ep-capacity-factor"):
        embedding(hash_ids)

    expected_output, expected_grads = _lookup_and_grads(
        embedding, hash_ids, grad_output * masked, dynamic=True
    )
    torch.testing.assert_close(output.detach(), expected_output * masked, rtol=0, atol=0)
    for grad, expected_grad in zip(grads, expected_grads):
        torch.testing.assert_close(grad, expected_grad)
