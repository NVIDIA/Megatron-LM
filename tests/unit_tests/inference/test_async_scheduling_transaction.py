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
from megatron.core.inference.batch_dimensions_utils import InferenceBatchDimensions
from megatron.core.inference.ep_async_protocol import EPAsyncPhase, EPAsyncStepProtocol
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
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


class _RecordingEPCommunicator:
    def __init__(self, *, sync_results=(), world_size=2):
        self.sync_results = list(sync_results)
        self.world_size = world_size
        self.protocol_mismatch_count = 0
        self.calls = []

    def sync_all_reduce_max(self, *values, phase=None, step_id=None):
        self.calls.append((EPAsyncPhase(phase), step_id, tuple(values)))
        if self.sync_results:
            return self.sync_results.pop(0)
        return values[0] if len(values) == 1 else tuple(values)


@pytest.mark.internal
@pytest.mark.parametrize(
    ("global_state", "expected"),
    [
        ((1, 1, 1, 0, 0, 0), (True, False, False)),
        ((1, 1, 1, 1, 0, 0), (True, False, True)),
        ((1, 1, 0, 0, 1, 0), (False, True, False)),
        ((1, 1, 1, 0, 0, 1), (False, True, False)),
    ],
)
def test_ep_step_begin_agrees_on_row_mapped_reuse(global_state, expected):
    communicator = _RecordingEPCommunicator(sync_results=[global_state, 1])
    protocol = EPAsyncStepProtocol(communicator)

    decision = protocol.decide_step_begin(
        has_real_work=True,
        has_pending_forward=True,
        pending_forward_reusable=True,
        pending_forward_row_mapped=bool(global_state[3]),
    )

    assert (
        decision.reuse_pending_forward,
        decision.discard_pending_forward,
        decision.row_mapped_forward,
    ) == expected
    assert communicator.calls == [
        (
            EPAsyncPhase.STEP_BEGIN,
            0,
            (1, 1, 1, int(bool(global_state[3])), 0, 0),
        ),
        (EPAsyncPhase.STEP_BEGIN_ACK, 0, (1,)),
    ]


@pytest.mark.internal
def test_ep_graph_shape_sync_uses_protocol_tag(monkeypatch):
    communicator = _RecordingEPCommunicator(sync_results=[(16, 0)])
    protocol = EPAsyncStepProtocol(communicator)
    monkeypatch.setattr("megatron.core.inference.batch_dimensions_utils.get_pg_size", lambda _: 2)

    adjusted = InferenceBatchDimensions.adjust_batch_dims_for_expert_parallelism(
        InferenceBatchDimensions(8, 0, 4),
        ep_group=object(),
        ep_async_protocol=protocol,
    )

    assert adjusted == InferenceBatchDimensions(16, 0, 4)
    assert communicator.calls == [(EPAsyncPhase.GRAPH_SHAPE, 0, (8, 0))]


@pytest.mark.internal
@pytest.mark.parametrize("materialize_only_last_token_logits", [False, True])
def test_controller_required_logits_uses_pending_row_indices(materialize_only_last_token_logits):
    logits = torch.arange(1 * 4 * 3, dtype=torch.float32).view(1, 4, 3)
    context = SimpleNamespace(
        total_request_count=2,
        paused_request_count=0,
        padded_active_token_count=4,
        config=SimpleNamespace(
            materialize_only_last_token_logits=materialize_only_last_token_logits
        ),
        is_decode_only=lambda: True,
    )
    controller = SimpleNamespace(
        inference_wrapped_model=SimpleNamespace(inference_context=context),
        _all_logits_cuda=logits,
    )

    selected = TextGenerationController._dynamic_step_required_token_logits(
        controller, row_indices=torch.tensor([1, 3])
    )

    assert torch.equal(selected, logits.squeeze(0).index_select(0, torch.tensor([1, 3])))
