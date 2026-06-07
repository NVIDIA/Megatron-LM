# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.async_scheduling import (
    AsyncDecodeTransaction,
    AsyncGraphShape,
    AsyncKVBlockLease,
    AsyncRowMap,
    AsyncTransactionState,
)


def _plan(request_ids):
    request_ids_tensor = torch.tensor(request_ids, dtype=torch.long, device="cpu")
    return SimpleNamespace(
        request_ids=request_ids_tensor,
        active_request_count=request_ids_tensor.numel(),
        active_token_count=request_ids_tensor.numel(),
        padded_active_request_count=4,
    )


@pytest.mark.internal
def test_transaction_state_transitions():
    plan = _plan([10, 11])
    txn = AsyncDecodeTransaction(
        transaction_id=7,
        prepared_layout=plan,
        graph_shape=AsyncGraphShape.from_plan(plan),
    )

    assert txn.state == AsyncTransactionState.PREPARED
    txn.launch(cuda_graph_request_count=4)
    assert txn.state == AsyncTransactionState.IN_FLIGHT
    assert txn.cuda_graph_request_count == 4

    row_map = txn.resolve_for_current(
        current_request_ids=torch.tensor([10, 11]),
        current_graph_shape=AsyncGraphShape(2, 2, 4),
    )
    assert row_map is not None
    assert not row_map.row_mapped

    txn.mark_ready(row_map)
    assert txn.state == AsyncTransactionState.READY
    txn.consume()
    assert txn.state == AsyncTransactionState.CONSUMED


@pytest.mark.internal
def test_transaction_rejects_illegal_transitions():
    plan = _plan([1])
    txn = AsyncDecodeTransaction(
        transaction_id=1,
        prepared_layout=plan,
        graph_shape=AsyncGraphShape.from_plan(plan),
    )

    with pytest.raises(RuntimeError, match="expected state ready"):
        txn.consume()

    txn.launch(cuda_graph_request_count=None)
    txn.drop("test")
    assert txn.state == AsyncTransactionState.DROPPED
    assert txn.drop_reason == "test"

    with pytest.raises(RuntimeError, match="cannot drop"):
        txn.drop("again")


@pytest.mark.internal
@pytest.mark.parametrize(
    ("pending", "current", "expected_rows", "row_mapped"),
    [
        ([10, 11, 12], [10, 11, 12], [0, 1, 2], False),
        ([10, 11, 12], [11, 12], [1, 2], True),
        ([10, 11, 12], [10, 12], [0, 2], True),
        ([10, 11, 12], [10, 11], [0, 1], False),
    ],
)
def test_row_map_resolves_finished_rows(pending, current, expected_rows, row_mapped):
    row_map = AsyncRowMap.for_current(
        pending_request_ids=torch.tensor(pending),
        current_request_ids=torch.tensor(current),
    )

    assert row_map is not None
    assert row_map.pending_rows_cpu.tolist() == expected_rows
    assert row_map.row_mapped is row_mapped


@pytest.mark.internal
@pytest.mark.parametrize(
    ("pending", "current"),
    [
        ([10, 11], [12]),
        ([10, 10], [10]),
        ([10, 11], [11, 11]),
        ([10], [10, 11]),
    ],
)
def test_row_map_rejects_incompatible_rows(pending, current):
    assert (
        AsyncRowMap.for_current(
            pending_request_ids=torch.tensor(pending),
            current_request_ids=torch.tensor(current),
        )
        is None
    )


@pytest.mark.internal
def test_transaction_rejects_graph_shape_mismatch():
    plan = _plan([1, 2])
    txn = AsyncDecodeTransaction(
        transaction_id=1,
        prepared_layout=plan,
        graph_shape=AsyncGraphShape.from_plan(plan),
    )
    txn.launch(cuda_graph_request_count=4)

    assert (
        txn.resolve_for_current(
            current_request_ids=torch.tensor([1, 2]),
            current_graph_shape=AsyncGraphShape(
                active_request_count=2,
                active_token_count=2,
                padded_active_request_count=8,
            ),
        )
        is None
    )


@pytest.mark.internal
def test_kv_lease_records_reserved_blocks():
    lease = AsyncKVBlockLease(
        reserved_request_ids=torch.tensor([7, 8]),
        reserved_block_ids=torch.tensor([100, 101], dtype=torch.int32),
        reserved_block_columns=torch.tensor([1, 1], dtype=torch.int32),
    )

    assert lease.has_reservations
    assert lease.reserved_block_ids.tolist() == [100, 101]
