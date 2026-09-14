# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""EP-sharded lookup parity tests.

Runs under the CI unit-test launcher (torch.distributed.run at 8 ranks) as well as local
torchrun with 2 or 4 ranks: each case builds an EP subgroup of the first ``ep_size`` ranks
and the remaining ranks sit the case out after the collective group creation.
"""

import os
from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.engram.distributed_embedding import (
    EPShardedMultiTableEmbedding,
    get_contiguous_row_range,
)
from megatron.training.utils import get_pipeline_prefetched_tokens, prepare_tokens_for_pipeline
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) not in (2, 4, 8),
    reason="requires torchrun with two, four, or eight ranks",
)


def _initialize():
    Utils.initialize_distributed()
    return torch.distributed.get_rank(), torch.distributed.get_world_size()


def _full_table(table_id: int, rows: int, dim: int, device) -> torch.Tensor:
    row = torch.arange(rows, dtype=torch.float64, device=device).unsqueeze(1)
    column = torch.arange(dim, dtype=torch.float64, device=device).unsqueeze(0)
    return table_id * 1000.0 + row * 10.0 + column


def test_pipeline_group_prefetches_matching_tokens_before_schedule():
    rank, world_size = _initialize()
    group = torch.distributed.group.WORLD
    tp_group = None
    for group_rank in range(world_size):
        candidate = torch.distributed.new_group(ranks=[group_rank])
        if group_rank == rank:
            tp_group = candidate
    assert tp_group is not None
    tokens = torch.arange(8, device="cuda", dtype=torch.int64).view(2, 4)
    data_iterator = iter([{"tokens": tokens}]) if rank == 0 else None
    iterator = prepare_tokens_for_pipeline(data_iterator, 1, 2, 4, group, tp_group)
    received = get_pipeline_prefetched_tokens(iterator)
    torch.testing.assert_close(received, torch.arange(8, device="cuda").view(2, 4))
    if rank == 0:
        assert next(iterator)["tokens"].data_ptr() == tokens.data_ptr()


@pytest.mark.parametrize("use_cpu_initialization", [False, True])
def test_table_init_keeps_default_rng_synchronized_across_ep_ranks(use_cpu_initialization):
    """EP shards have unequal row counts; their init must not desync the default RNG streams.

    A desynchronized default stream (CPU or any CUDA device) would silently diverge every
    subsequently initialized replicated parameter across data-parallel peers.
    """
    rank, world_size = _initialize()
    ep_size = 2
    device = torch.device("cuda", torch.cuda.current_device())
    group = torch.distributed.new_group(ranks=list(range(ep_size)))
    if rank >= ep_size:
        return

    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    config = SimpleNamespace(
        use_cpu_initialization=use_cpu_initialization,
        params_dtype=torch.float32,
        perform_initialization=True,
        sequence_parallel=False,
        deterministic_mode=False,
    )
    embedding = EPShardedMultiTableEmbedding(
        config=config,
        table_sizes=(11, 13),
        embedding_dim=4,
        init_method=lambda tensor: torch.nn.init.normal_(tensor, mean=0.0, std=0.1),
        ep_group=group,
        tp_group=None,
        expt_dp_group=None,
    )

    # The next default-stream draws (CPU and CUDA) must be identical on both EP ranks even
    # though their shard shapes differ (11 and 13 rows split unevenly over 2 ranks).
    probe = torch.cat((torch.randn(16, device=device), torch.randn(16).to(device)))
    gathered_probe = [torch.empty_like(probe) for _ in range(ep_size)]
    torch.distributed.all_gather(gathered_probe, probe, group=group)
    assert torch.equal(gathered_probe[0], gathered_probe[1])

    # The shards themselves must be decorrelated across EP ranks, not copies of one stream.
    for table in embedding.tables:
        rows = min(
            table.local_num_embeddings, table.global_num_embeddings - table.local_num_embeddings
        )
        shard_head = table.weight[:rows].to(device).contiguous()
        gathered_shards = [torch.empty_like(shard_head) for _ in range(ep_size)]
        torch.distributed.all_gather(gathered_shards, shard_head, group=group)
        assert not torch.equal(gathered_shards[0], gathered_shards[1])


@pytest.mark.parametrize("ep_size", [2, 4])
def test_variable_lookup_gradients_and_adam_step_match_full_reference(ep_size):
    rank, world_size = _initialize()
    if world_size < ep_size:
        pytest.skip(f"requires at least {ep_size} ranks")
    device = torch.device("cuda", torch.cuda.current_device())
    # Collective on the default group: every rank must participate in group creation.
    group = torch.distributed.new_group(ranks=list(range(ep_size)))
    if rank >= ep_size:
        return

    table_sizes = (11, 13)
    embedding_dim = 3
    config = SimpleNamespace(
        use_cpu_initialization=False,
        params_dtype=torch.float64,
        perform_initialization=False,
        sequence_parallel=False,
        deterministic_mode=True,
    )
    embedding = EPShardedMultiTableEmbedding(
        config=config,
        table_sizes=table_sizes,
        embedding_dim=embedding_dim,
        init_method=lambda _: None,
        ep_group=group,
        tp_group=None,
        expt_dp_group=None,
    )

    full_tables = []
    with torch.no_grad():
        for table_id, (table, rows) in enumerate(zip(embedding.tables, table_sizes)):
            full = _full_table(table_id, rows, embedding_dim, device)
            full_tables.append(full.detach().clone().requires_grad_(True))
            table.weight.copy_(full[table.row_start : table.row_end])

    # Rank-local token counts are unequal. Repeated rows exercise gradient accumulation, while
    # routing each table to only one owner creates explicit zero peer splits.
    local_tokens = rank + 1
    owner_for_table0 = (rank + 1) % ep_size
    table0_start, table0_end = get_contiguous_row_range(table_sizes[0], owner_for_table0, ep_size)
    table1_start, table1_end = get_contiguous_row_range(table_sizes[1], 0, ep_size)
    rows0 = torch.full(
        (local_tokens,), table0_start + (rank % (table0_end - table0_start)), device=device
    )
    rows1 = torch.full(
        (local_tokens,), table1_start + (rank % (table1_end - table1_start)), device=device
    )
    hash_ids = torch.stack((rows0, rows1), dim=-1).view(1, local_tokens, 2)

    output = embedding(hash_ids)
    expected_output = torch.stack((full_tables[0][rows0], full_tables[1][rows1]), dim=-2).view_as(
        output
    )
    torch.testing.assert_close(output, expected_output)

    dense = torch.nn.Linear(embedding_dim * 2, 2, bias=False, dtype=torch.float64, device=device)
    with torch.no_grad():
        dense.weight.copy_(
            torch.tensor(
                [[0.2, -0.3, 0.5, 0.7, -0.2, 0.1], [-0.4, 0.6, 0.8, -0.1, 0.3, 0.9]],
                dtype=torch.float64,
                device=device,
            )
        )
    reference_dense = torch.nn.Linear(
        embedding_dim * 2, 2, bias=False, dtype=torch.float64, device=device
    )
    reference_dense.load_state_dict(dense.state_dict())
    coefficient = float(rank + 1)
    loss = dense(output.flatten(start_dim=-2)).sum() * coefficient
    reference_loss = reference_dense(expected_output.flatten(start_dim=-2)).sum() * coefficient
    loss.backward()
    reference_loss.backward()
    torch.testing.assert_close(dense.weight.grad, reference_dense.weight.grad)

    # The full reference graphs on each rank cover only that rank's requests. Sum their full-table
    # gradients to obtain the owner-side result produced by differentiable all-to-all backward.
    for table_id, table in enumerate(embedding.tables):
        full_grad = full_tables[table_id].grad
        torch.distributed.all_reduce(full_grad, group=group)
        torch.testing.assert_close(table.weight.grad, full_grad[table.row_start : table.row_end])

    actual_optimizer = torch.optim.Adam(
        [*[table.weight for table in embedding.tables], dense.weight], lr=1.0e-3
    )
    reference_optimizer = torch.optim.Adam([*full_tables, reference_dense.weight], lr=1.0e-3)
    actual_optimizer.step()
    reference_optimizer.step()
    for table_id, table in enumerate(embedding.tables):
        torch.testing.assert_close(
            table.weight,
            full_tables[table_id][table.row_start : table.row_end],
            rtol=1e-12,
            atol=1e-12,
        )
    torch.testing.assert_close(dense.weight, reference_dense.weight, rtol=1e-12, atol=1e-12)
