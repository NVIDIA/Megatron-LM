# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact replay of the Engram symmetric transport and lookup.

The exchange replay is ported from ``test_engram_device_exchange_replays`` of
NVIDIA/Megatron-LM#7224 by Hongxiao Bai (hxbai), commit 906ccd59e: striped LSA/GIN transfers
reuse both receive slots under side-stream contention, byte for byte. The lookup replay runs a
symmetric lookup with duplicate rows and its row-sparse ``main_grad`` gradient under the same
contention. Both run under torchrun on GB300 and skip unless the transport's toolchain (CuTe
DSL >= 4.8.1, nccl4py >= 0.6.0, a matching NCCL runtime and device bitcode) is installed; the
lookup replay also needs ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` in the launch
environment. Transports are closed collectively only when the case passed: after a failure on
one rank, a clean close would wait in a device barrier for peers that are still in the test.
"""

import os
from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.engram import symmetric_lookup
from megatron.core.models.engram.distributed_embedding import EPShardedMultiTableEmbedding
from tests.unit_tests.determinism.kernels.harness import (
    CONTENTION_TOKENS,
    assert_module_replays_bit_exact,
    assert_replays_bit_exact,
    bytes_equal,
)
from tests.unit_tests.test_utilities import Utils

_IS_BLACKWELL = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10


def _row_group_or_skip():
    try:
        symmetric_lookup.require_symmetric_transport()
    except ImportError as exc:
        pytest.skip(f"Engram symmetric transport unavailable: {exc}")
    Utils.initialize_distributed()
    group = torch.distributed.group.WORLD
    if torch.distributed.get_world_size(group) < 2:
        pytest.skip("Needs at least two row shards")
    return group


@pytest.mark.internal
@pytest.mark.launch_on_gb200
@pytest.mark.skipif(not _IS_BLACKWELL, reason="NCCL device transport needs Blackwell")
def test_engram_device_exchange_replays():
    """Striped transfers reuse both slots under competing stream work, byte for byte."""
    pytest.importorskip("nccl.core.device.cute", exc_type=ImportError)
    group = _row_group_or_skip()
    from megatron.core.ops.engram import EngramDeviceTransport, create_unique_id

    rank, size = torch.distributed.get_rank(group), torch.distributed.get_world_size(group)
    unique_id = [create_unique_id() if rank == 0 else None]
    torch.distributed.broadcast_object_list(unique_id, src=0, group=group)
    # Eight stripes with 4 KiB slots force several rounds for 1040-byte packets.
    workspace = (4096 * 2 * (size + min(4, size)) + size * 32) * 8
    counts = torch.tensor([[153 + (src + dst) % 7 for dst in range(size)] for src in range(size)])
    send, recv = counts[rank], counts[:, rank]
    plan = symmetric_lookup.EngramExchangePlan.from_counts(
        send.cuda(), recv.cuda(), send.tolist(), recv.tolist()
    )
    source = torch.randint(256, (sum(send.tolist()), 1040), device="cuda", dtype=torch.uint8)
    expected = source.new_empty((sum(recv.tolist()), 1040))
    torch.distributed.all_to_all_single(expected, source, recv.tolist(), send.tolist(), group=group)
    transport = EngramDeviceTransport(group, unique_id=unique_id[0], workspace_bytes=workspace)

    def exchange(values):
        return transport.exchange(values, plan)

    ok = False
    try:
        transport.initialize()
        transport.barrier()
        assert bytes_equal(exchange(source), expected)
        assert transport.stripes == 8
        assert_replays_bit_exact(
            exchange,
            (source,),
            backward=False,
            contention=True,
            what="Engram striped LSA/GIN row exchange",
        )
        ok = True
    finally:
        transport.close(clean=ok)


class _Lookup(torch.nn.Module):
    """Tables plus their symmetric lookup, the unit the replay harness runs."""

    def __init__(self, group, table_sizes, dim) -> None:
        super().__init__()
        config = SimpleNamespace(
            use_cpu_initialization=False,
            params_dtype=torch.bfloat16,
            perform_initialization=True,
            sequence_parallel=False,
            deterministic_mode=True,
        )
        self.embedding = EPShardedMultiTableEmbedding(
            config,
            table_sizes,
            dim,
            lambda tensor: torch.nn.init.normal_(tensor, std=0.02),
            ep_group=group,
        )
        self.lookup = symmetric_lookup.SymmetricEngramLookup(self.embedding)

    def forward(self, hash_ids):
        return self.lookup(hash_ids)


@pytest.mark.internal
@pytest.mark.launch_on_gb200
@pytest.mark.skipif(not _IS_BLACKWELL, reason="NCCL device transport needs Blackwell")
def test_engram_symmetric_lookup_and_gradient_replay(monkeypatch):
    """Deduplicated lookup and owner-side main_grad accumulation replay bit for bit."""
    if "expandable_segments:True" not in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
        pytest.skip("Engram table windows need PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    group = _row_group_or_skip()
    rank, size = torch.distributed.get_rank(group), torch.distributed.get_world_size(group)
    dim = 160
    module = _Lookup(group, (100003 * size, max(1, size - 1), 1009), dim)
    weights = [table.weight for table in module.embedding.tables]
    for weight in weights:
        weight.main_grad = torch.zeros(weight.shape, dtype=torch.float32, device=weight.device)
        weight.grad_added_to_main_grad = False
    monkeypatch.setattr(symmetric_lookup, "_engram_row_group", lambda pg, config: group)
    generator = torch.Generator(device="cuda").manual_seed(4321 + rank)
    # Few distinct rows per small table: long duplicate segments in both reductions.
    ids = torch.stack(
        [
            torch.randint(0, table_size, (CONTENTION_TOKENS, 1), device="cuda", generator=generator)
            for table_size in module.embedding.table_sizes
        ],
        dim=-1,
    )
    grad = torch.randn((*ids.shape, dim), device="cuda", generator=generator).to(torch.bfloat16)
    ok = False
    try:
        symmetric_lookup.initialize_engram_transports(
            [module],
            None,
            SimpleNamespace(communication_backend="symmetric", symmetric_workspace_mb=64),
        )
        assert_module_replays_bit_exact(
            module,
            (ids,),
            grad_output=grad,
            contention=True,
            what="Engram symmetric lookup and row-sparse main_grad",
        )
        ok = True
    finally:
        for weight in weights:
            # The placeholder gradient aliases the table; release it as DDP's hook would.
            weight.grad = None
        symmetric_lookup.close_engram_transports(clean=ok)
