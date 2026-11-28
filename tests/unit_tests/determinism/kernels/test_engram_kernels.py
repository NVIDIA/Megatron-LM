# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact replay of the Engram table lookup in ``megatron/core/models/engram/``.

Both cases run ``deterministic_mode``, where the table backward sorts the looked-up rows and
sums the gradients of a repeated row in that order (``_DeterministicEmbedding``):

* one table on a single rank, with thousands of lookups over a few dozen rows;
* two tables sharded over an expert-parallel group of two ranks, through the request-count,
  request and embedding all-to-alls of ``EPShardedMultiTableEmbedding`` and the reverse
  all-to-all that returns the gradients to the row owners.

The routed case needs torchrun with at least two ranks: the first two form the expert-parallel
group and the remaining ranks sit the case out after the collective group creation.
"""

import os

import pytest
import torch

from megatron.core.models.engram.distributed_embedding import (
    EPShardedEmbeddingTable,
    EPShardedMultiTableEmbedding,
    get_contiguous_row_range,
)
from tests.unit_tests.determinism.kernels.harness import (
    CONTENTION_TOKENS,
    assert_module_replays_bit_exact,
    bytes_equal,
    deterministic_algorithms,
    seeded,
)
from tests.unit_tests.determinism.utils import RacingStreams, collect_grads, zero_grads
from tests.unit_tests.models.engram._test_utils import make_module_config
from tests.unit_tests.test_utilities import Utils, release_process_groups

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

EP_SIZE = 2


@pytest.fixture(autouse=True)
def _release_process_groups():
    """The routed case creates a process group of its own; see release_process_groups."""
    with release_process_groups():
        yield


def test_deterministic_table_backward_replays_bit_exact():
    """Under deterministic_mode, repeated-row gradients are summed in a fixed order on GPU."""
    seeded(1234)
    num_rows, embedding_dim = 61, 1024
    table = EPShardedEmbeddingTable(
        config=make_module_config(deterministic_mode=True, dtype=torch.bfloat16),
        global_num_embeddings=num_rows,
        embedding_dim=embedding_dim,
        init_method=lambda tensor: torch.nn.init.normal_(tensor, mean=0.0, std=0.02),
    ).cuda()
    # Thousands of lookups over a few dozen rows put many gradients on every touched row.
    row_ids = torch.randint(0, num_rows, (CONTENTION_TOKENS,), device="cuda")
    grad_output = torch.randn(CONTENTION_TOKENS, embedding_dim, device="cuda", dtype=torch.bfloat16)
    assert_module_replays_bit_exact(
        table,
        {"local_row_ids": row_ids},
        grad_output=grad_output,
        replays=4,
        contention=True,
        what="Engram deterministic table lookup",
    )


def _lookup_and_backward(embedding, hash_ids, grad_output):
    """Run one lookup and its backward; return this rank's output and table gradients."""
    zero_grads(embedding)
    output = embedding(hash_ids)
    output.backward(grad_output)
    torch.cuda.synchronize()
    return {"output": output.detach().clone(), **collect_grads([embedding])}


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) < EP_SIZE,
    reason="requires torchrun with at least two ranks",
)
def test_routed_lookup_and_owner_gradients_replay_bit_exact():
    """Under deterministic_mode, the EP2 lookup of two tables and the gradients that its
    backward returns to the row owners repeat byte for byte on both ranks."""
    Utils.initialize_distributed()
    rank = torch.distributed.get_rank()
    # Collective on the default group: every rank must participate in group creation.
    group = torch.distributed.new_group(ranks=list(range(EP_SIZE)))
    if rank >= EP_SIZE:
        return
    device = torch.device("cuda", torch.cuda.current_device())

    seeded(1234 + rank)
    table_sizes, embedding_dim = (61, 67), 1024
    embedding = EPShardedMultiTableEmbedding(
        config=make_module_config(deterministic_mode=True, dtype=torch.bfloat16),
        table_sizes=table_sizes,
        embedding_dim=embedding_dim,
        init_method=lambda tensor: torch.nn.init.normal_(tensor, mean=0.0, std=0.02),
        ep_group=group,
    ).to(device)

    # Rank 0 looks up rows of both shards. Rank 1 looks up half as many tokens and only rows
    # of its own shard: it sends nothing to rank 0, and the two shards serve different numbers
    # of requests. Thousands of lookups over a few dozen rows per table put many gradients on
    # every touched row.
    num_tokens = CONTENTION_TOKENS // (rank + 1)
    row_ranges = [
        (0, table_size) if rank == 0 else get_contiguous_row_range(table_size, 1, EP_SIZE)
        for table_size in table_sizes
    ]
    hash_ids = torch.stack(
        [torch.randint(start, end, (num_tokens,), device=device) for start, end in row_ranges],
        dim=-1,
    )
    grad_output = torch.randn(
        num_tokens, len(table_sizes), embedding_dim, device=device, dtype=torch.bfloat16
    )

    # --deterministic-mode also turns the deterministic torch algorithms on.
    with deterministic_algorithms(True):
        runs = [_lookup_and_backward(embedding, hash_ids, grad_output)]
        for _ in range(3):
            with RacingStreams():
                runs.append(_lookup_and_backward(embedding, hash_ids, grad_output))

    # Compare only after the last exchange: a rank that stopped at its first mismatch would
    # leave its peer waiting in the next all-to-all.
    reference = runs[0]
    # The output, and the gradient of either table: each shard serves requests for both.
    assert len(reference) == 1 + len(table_sizes)
    assert all(torch.count_nonzero(tensor) > 0 for tensor in reference.values())
    for index, replay in enumerate(runs[1:], start=1):
        assert replay.keys() == reference.keys()
        for name, expected in reference.items():
            assert bytes_equal(expected, replay[name]), f"replay {index}: {name}"
