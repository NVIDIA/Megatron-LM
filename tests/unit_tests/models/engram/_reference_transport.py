# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""An all-to-all reference for the host interface of the Engram symmetric transport.

Ported from the ``_ReferenceDeviceTransport`` of NVIDIA/Megatron-LM#7224 by Hongxiao Bai (hxbai)
(tests/unit_tests/models/test_v41_engram.py) and adapted to the interface that
``megatron.core.models.engram.symmetric_lookup`` drives: count and gradient exchanges, local
table description, the workspace allocation, collective registration and the workspace barrier
(no-ops here) and lookup. Every transfer is a ``torch.distributed.all_to_all_single`` over the
row group, so the reference runs on Gloo (CPU) and NCCL (GPU). Rows are copied, never summed,
so results are bitwise comparable with the CuTe/NCCL transport. Each exchange also checks the
plan's device splits, which only the CuTe kernels read, against its host splits.
"""

from types import SimpleNamespace

import torch
import torch.distributed as dist

from megatron.core.utils import get_pg_rank, get_pg_size


def _check_device_splits(plan) -> None:
    """The ``[N, 4]`` device splits must be (send start, send rows, receive start, receive rows)."""

    def starts(counts):
        return [sum(counts[:peer]) for peer in range(len(counts))]

    send, recv = list(plan.send_splits), list(plan.recv_splits)
    expected = [list(row) for row in zip(starts(send), send, starts(recv), recv)]
    assert plan.splits.dtype == torch.int64, plan.splits.dtype
    assert plan.splits.tolist() == expected, (plan.splits.tolist(), expected)


class ReferenceEngramTransport:
    """Engram transport whose every exchange is an ``all_to_all_single`` on the row group.

    It records what the runtime asks of it, so tests can check the count rows, the
    registration decisions and how many exchanges a lookup or a backward issues.
    """

    def __init__(self, group, **options) -> None:
        self.group = group
        self.rank = get_pg_rank(group)
        self.size = get_pg_size(group)
        # Constructor options of the device transport (unique_id, workspace_bytes, ...).
        self.options = options
        self.prepared = self.initialized = False
        self.closed = None
        self.barriers = 0
        self.exchanges = 0
        self.describes = 0
        self.lookups = []
        self.registrations = []

    def prepare(self) -> None:
        """Nothing to allocate: there is no workspace."""
        self.prepared = True

    def initialize(self) -> None:
        """Nothing to create: the row group's process group carries every exchange."""
        assert self.prepared, "start-up initializes a transport only after prepare()"
        self.initialized = True

    def barrier(self) -> None:
        """Nothing to publish or wait for: every exchange is synchronous."""
        self.barriers += 1

    def describe_tables(self, key, tables):
        """Never changed and never failing; the layout is each table's own storage offset."""
        del key
        self.describes += 1
        layout = tuple(
            (table.storage_offset() * table.element_size(), table.stride(0) * table.element_size())
            for table in tables
        )
        return False, None, layout

    def register_tables(self, key, metadata) -> None:
        """Record the collective registration decision and the metadata it received."""
        self.registrations.append((key, metadata.clone()))

    def exchange(self, source, plan, *, remote_only=False, output=None):
        """Exchange rows by the plan's host splits, ordered by source rank."""
        assert not remote_only, "the reference transport has no LSA/GIN split"
        _check_device_splits(plan)
        self.exchanges += 1
        result = source.new_empty((sum(plan.recv_splits), *source.shape[1:]))
        dist.all_to_all_single(
            result,
            source.contiguous(),
            list(plan.recv_splits),
            list(plan.send_splits),
            group=self.group,
        )
        if output is None:
            return result
        output.copy_(result)
        return output

    def lookup(self, key, tables, requests, plan, metadata):
        """Send the requests to their owners and return the rows as uint8, in request order."""
        del key
        self.lookups.append(
            SimpleNamespace(requests=requests.clone(), plan=plan, metadata=metadata.clone())
        )
        received = self.exchange(requests, plan)
        rows = tables[0].new_empty((received.shape[0], tables[0].shape[1]))
        for table, weight in enumerate(tables):
            mask = received[:, 0] == table
            rows[mask] = weight.detach()[received[mask, 1]]
        return self.exchange(rows, plan.reverse()).view(torch.uint8)

    def close(self, clean=False) -> None:
        """Record how the runtime closed the transport."""
        self.closed = clean
