# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from collections import deque
from types import SimpleNamespace

import pytest
import torch

from megatron.core import utils as core_utils
from megatron.core.inference.async_transaction import (
    AsyncDecodePlan,
    AsyncDecodeTransaction,
    AsyncEPParticipant,
    AsyncGraphShape,
    AsyncLayoutSnapshot,
    AsyncLogprobMTPParticipant,
    AsyncMambaStateParticipant,
    AsyncResourceLedger,
    AsyncResourceParticipant,
    AsyncRowMapPolicy,
    AsyncSampleReadback,
    AsyncSampleReadbackParticipant,
    AsyncSampleTicket,
    AsyncStepTransaction,
    AsyncTransactionParticipant,
    AsyncTxnState,
    classify_async_eligibility,
    resolve_async_pending_forward,
)
from megatron.core.inference.batch_dimensions_utils import InferenceBatchDimensions
from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.contexts.mamba_slot_allocator import MambaSlotAllocator
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
from megatron.core.inference.ep_async_protocol import (
    EPAsyncHandoffDecision,
    EPAsyncPhase,
    EPAsyncStepProtocol,
    EPStepBeginDecision,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers import (
    text_generation_controller as tgc_module,
)
from megatron.core.inference.text_generation_controllers.async_decode_coordinator import (
    AsyncDecodeCoordinator,
)
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.transformer.enums import InferenceCudaGraphScope


class _RecordingEPCommunicator:
    def __init__(
        self, *, async_results=(), sync_results=(), world_size=2, fail_async=False, fail_sync=False
    ):
        self.async_results = list(async_results)
        self.sync_results = list(sync_results)
        self.world_size = world_size
        self.fail_async = fail_async
        self.fail_sync = fail_sync
        self.protocol_mismatch_count = 0
        self.calls = []

    async def all_reduce_max(self, *values, async_op=True, phase=None, step_id=None):
        self.calls.append(("async", EPAsyncPhase(phase), step_id, tuple(values), async_op))
        if self.fail_async:
            raise RuntimeError("async collective failed")
        if self.async_results:
            return self.async_results.pop(0)
        return values[0] if len(values) == 1 else tuple(values)

    def sync_all_reduce_max(self, *values, phase=None, step_id=None):
        self.calls.append(("sync", EPAsyncPhase(phase), step_id, tuple(values)))
        if self.fail_sync:
            raise RuntimeError("sync collective failed")
        if self.sync_results:
            return self.sync_results.pop(0)
        return values[0] if len(values) == 1 else tuple(values)


@pytest.mark.internal
@pytest.mark.asyncio
async def test_ep_protocol_tags_work_consensus_and_completion():
    communicator = _RecordingEPCommunicator(async_results=[(7, -1), 1])
    protocol = EPAsyncStepProtocol(communicator)

    consensus = await protocol.establish_work_consensus(
        local_work=3, signal_consensus=True, async_op=False
    )
    await protocol.complete_work_step(async_op=False)

    assert consensus.step_id == 0
    assert consensus.global_work == 7
    assert consensus.all_pausing
    assert communicator.calls == [
        ("async", EPAsyncPhase.WORK_CONSENSUS, 0, (3, -1), False),
        ("async", EPAsyncPhase.WORK_CONSENSUS_ACK, 0, (1,), False),
        ("async", EPAsyncPhase.STEP_COMPLETE, 0, (1,), False),
        ("async", EPAsyncPhase.STEP_COMPLETE_ACK, 0, (1,), False),
    ]
    assert protocol.diagnostics()["work_consensus"] == 1
    assert protocol.diagnostics()["work_completions"] == 1
    assert protocol.diagnostics()["active_step_id"] is None


@pytest.mark.internal
@pytest.mark.asyncio
async def test_ep_protocol_local_mode_nested_steps_and_collective_errors():
    local_protocol = EPAsyncStepProtocol()
    assert not local_protocol.enabled
    assert await local_protocol.all_reduce_max(EPAsyncPhase.WORK_CONSENSUS, 3, 4) == (3, 4)
    assert local_protocol.sync_all_reduce_max(EPAsyncPhase.GRAPH_SHAPE, 5) == 5

    nested_protocol = EPAsyncStepProtocol(_RecordingEPCommunicator())
    await nested_protocol.establish_work_consensus(local_work=1, signal_consensus=False)
    with pytest.raises(RuntimeError, match="still active"):
        await nested_protocol.establish_work_consensus(local_work=1, signal_consensus=False)
    await nested_protocol.complete_work_step()

    async_error_protocol = EPAsyncStepProtocol(_RecordingEPCommunicator(fail_async=True))
    with pytest.raises(RuntimeError, match="async collective failed"):
        await async_error_protocol.establish_work_consensus(local_work=1, signal_consensus=False)
    assert async_error_protocol.diagnostics()["collective_errors"] == 1

    sync_error_protocol = EPAsyncStepProtocol(_RecordingEPCommunicator(fail_sync=True))
    with pytest.raises(RuntimeError, match="sync collective failed"):
        sync_error_protocol.decide_async_handoff(has_real_work=True, can_launch_async_handoff=True)
    assert sync_error_protocol.diagnostics()["collective_errors"] == 1


@pytest.mark.internal
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("global_state", "expected"),
    [
        ((1, 1, 1, 0, 0, 0), (True, False, False)),
        ((1, 1, 1, 1, 0, 0), (True, False, True)),
        ((1, 1, 1, 0, 0, 1), (False, True, False)),
        ((1, 1, 0, 0, 1, 0), (False, True, False)),
    ],
)
async def test_ep_step_begin_reuses_or_discards_with_global_state(global_state, expected):
    communicator = _RecordingEPCommunicator(
        async_results=[(1, 0), 1], sync_results=[global_state, 1]
    )
    protocol = EPAsyncStepProtocol(communicator)
    await protocol.establish_work_consensus(local_work=1, signal_consensus=False)

    decision = protocol.decide_step_begin(
        has_real_work=True,
        has_pending_forward=True,
        pending_forward_reusable=True,
        pending_forward_row_mapped=bool(global_state[3]),
    )
    await protocol.complete_work_step()

    assert (
        decision.reuse_pending_forward,
        decision.discard_pending_forward,
        decision.row_mapped_forward,
    ) == expected
    assert communicator.calls[2] == (
        "sync",
        EPAsyncPhase.STEP_BEGIN,
        0,
        (1, 1, 1, int(bool(global_state[3])), 0, 0),
    )


@pytest.mark.internal
@pytest.mark.parametrize(
    ("has_real_work", "can_launch", "global_state", "expected"),
    [
        (True, True, (1, 1, 0), (True, False, True, False)),
        (True, False, (1, 1, 1), (False, True, True, True)),
        (False, True, (1, 1, 0), (True, False, True, False)),
        (False, False, (0, 0, 0), (False, True, False, False)),
    ],
)
def test_ep_async_handoff_launches_or_skips_with_global_state(
    has_real_work, can_launch, global_state, expected
):
    communicator = _RecordingEPCommunicator(sync_results=[global_state, 1])
    protocol = EPAsyncStepProtocol(communicator)

    decision = protocol.decide_async_handoff(
        has_real_work=has_real_work, can_launch_async_handoff=can_launch
    )

    assert (
        decision.launch_async_forward,
        decision.skip_async_forward,
        decision.any_launch_request,
        decision.any_skip_request,
    ) == expected
    assert communicator.calls == [
        (
            "sync",
            EPAsyncPhase.ASYNC_HANDOFF,
            0,
            (
                int(has_real_work),
                int(has_real_work and can_launch),
                int(has_real_work and not can_launch),
            ),
        ),
        ("sync", EPAsyncPhase.ASYNC_HANDOFF_ACK, 0, (1,)),
    ]


@pytest.mark.internal
def test_ep_protocol_diagnostics_count_reuse_discard_launch_and_skip():
    communicator = _RecordingEPCommunicator(
        sync_results=[(1, 1, 1, 0, 0, 0), 1, (1, 1, 0, 0, 1, 0), 1, (1, 1, 0), 1, (1, 1, 1), 1]
    )
    protocol = EPAsyncStepProtocol(communicator)

    reuse = protocol.decide_step_begin(
        has_real_work=True,
        has_pending_forward=True,
        pending_forward_reusable=True,
        pending_forward_row_mapped=False,
    )
    discard = protocol.decide_step_begin(
        has_real_work=True,
        has_pending_forward=True,
        pending_forward_reusable=False,
        pending_forward_row_mapped=False,
    )
    launch = protocol.decide_async_handoff(has_real_work=True, can_launch_async_handoff=True)
    skip = protocol.decide_async_handoff(has_real_work=True, can_launch_async_handoff=False)

    diagnostics = protocol.diagnostics()
    assert reuse.reuse_pending_forward
    assert discard.discard_pending_forward
    assert launch.launch_async_forward
    assert skip.skip_async_forward
    assert diagnostics["step_begin_reuses"] == 1
    assert diagnostics["step_begin_discards"] == 1
    assert diagnostics["handoff_launches"] == 1
    assert diagnostics["handoff_skips"] == 1


@pytest.mark.internal
@pytest.mark.parametrize(
    ("local_dims", "sync_result", "kwargs", "expected"),
    [
        (InferenceBatchDimensions(8, 0, 4), (16, 0, 0, 8), {}, InferenceBatchDimensions(16, 0, 4)),
        (
            InferenceBatchDimensions(8, 0, 4),
            (64, 1, 2, 10),
            {"decode_only_cuda_graphs": True},
            None,
        ),
        (
            InferenceBatchDimensions(24, 1, 4),
            (64, 1, 2, 10),
            {"decode_only_cuda_graphs": False, "strict": True},
            InferenceBatchDimensions(64, 2, 10),
        ),
    ],
)
def test_ep_graph_shape_sync_uses_tagged_protocol(
    monkeypatch, local_dims, sync_result, kwargs, expected
):
    calls = []

    class _GraphShapeProtocol:
        def sync_all_reduce_max(self, phase, *values):
            calls.append((phase, values))
            return sync_result

    monkeypatch.setattr("megatron.core.inference.batch_dimensions_utils.get_pg_size", lambda _: 2)

    adjusted = InferenceBatchDimensions.adjust_batch_dims_for_expert_parallelism(
        local_dims,
        ep_group=object(),
        ep_async_protocol=_GraphShapeProtocol(),
        num_speculative_tokens=2,
        **kwargs,
    )

    assert adjusted == expected
    assert calls == [
        (
            EPAsyncPhase.GRAPH_SHAPE,
            (
                local_dims.token_count,
                int(local_dims.prefill_req_count > 0),
                local_dims.prefill_req_count,
                local_dims.decode_req_count,
            ),
        )
    ]


@pytest.mark.internal
def test_ep_graph_shape_sync_can_use_zmq_without_protocol(monkeypatch):
    calls = []

    class _ZMQCommunicator:
        def sync_all_reduce_max(self, *values):
            calls.append(values)
            return (32, 1, 3, 7)

    monkeypatch.setattr("megatron.core.inference.batch_dimensions_utils.get_pg_size", lambda _: 2)

    adjusted = InferenceBatchDimensions.adjust_batch_dims_for_expert_parallelism(
        InferenceBatchDimensions(8, 0, 4),
        strict=True,
        decode_only_cuda_graphs=False,
        ep_group=object(),
        ep_zmq_communicator=_ZMQCommunicator(),
    )

    assert adjusted == InferenceBatchDimensions(32, 3, 7)
    assert calls == [(8, 0, 0, 4)]


def _make_controller_with_rows(pending_ids, current_ids, current_graph_count=None):
    controller = object.__new__(TextGenerationController)
    controller.num_speculative_tokens = 0
    pending_graph_count = None if pending_ids is None else len(pending_ids)
    if current_graph_count is None:
        current_graph_count = (
            pending_graph_count if pending_graph_count is not None else len(current_ids)
        )
    controller._async_discarded_forward_count = 0
    controller._async_row_mapped_forward_count = 0
    controller._async_step_transaction = None
    controller.inference_wrapped_model = SimpleNamespace(
        inference_context=SimpleNamespace(
            request_ids=torch.tensor(current_ids, dtype=torch.int64),
            paused_request_count=0,
            total_request_count=len(current_ids),
            active_token_count=len(current_ids),
            padded_active_request_count=current_graph_count,
            using_cuda_graph_this_step=lambda: True,
        )
    )
    if pending_ids is not None:
        _install_pending_transaction(
            controller,
            _make_async_layout_snapshot(pending_ids, cuda_graph_request_count=pending_graph_count),
        )
    return controller


def _make_async_layout_snapshot(
    request_ids,
    *,
    cuda_graph_request_count=None,
    tokens_per_request=1,
    **layout_fields,
):
    request_ids_tensor = torch.tensor(request_ids, dtype=torch.int64)
    request_count = int(request_ids_tensor.numel())
    if cuda_graph_request_count is None:
        cuda_graph_request_count = request_count
    return AsyncLayoutSnapshot(
        request_ids=request_ids_tensor,
        graph_shape=AsyncGraphShape(
            active_request_count=request_count,
            active_token_count=request_count * tokens_per_request,
            padded_active_request_count=cuda_graph_request_count,
            tokens_per_request=tokens_per_request,
        ),
        **layout_fields,
    )


def _install_pending_transaction(controller, snapshot, *, state=AsyncTxnState.LAUNCHED):
    transaction = AsyncStepTransaction(step_id=0, state=state, snapshot=snapshot)
    controller._async_step_transaction = transaction
    return transaction


def _sample_ticket(tokens, mtp_tokens=None):
    sampled_tokens = torch.tensor(tokens, dtype=torch.int64)
    sampled_mtp_tokens = None if mtp_tokens is None else torch.tensor(mtp_tokens, dtype=torch.int64)
    return AsyncSampleTicket(
        slot=0,
        active_request_count=int(sampled_tokens.numel()),
        sampled_tokens_cuda=None,
        sample_values_cuda=None,
        sampled_tokens_cpu=sampled_tokens,
        sampled_mtp_tokens_cpu=sampled_mtp_tokens,
        copy_done_event=SimpleNamespace(synchronize=lambda: None),
    )


@pytest.mark.internal
def test_async_decode_coordinator_owns_transaction_state_machine():
    controller = object.__new__(TextGenerationController)
    controller._async_step_transaction = None
    controller._async_transaction_next_step_id = 0
    coordinator = AsyncDecodeCoordinator(controller)
    controller._async_decode_coordinator = coordinator
    snapshot = _make_async_layout_snapshot([10, 11], cuda_graph_request_count=2)
    plan = AsyncDecodePlan.from_snapshot(snapshot)

    transaction = coordinator.begin_transaction(snapshot=snapshot, plan=plan)

    assert transaction.step_id == 0
    assert coordinator.pending_transaction() is transaction
    assert controller._pending_async_transaction() is transaction
    assert controller._has_pending_async_forward_state()

    coordinator.retire_transaction()

    assert transaction.state == AsyncTxnState.RETIRED
    assert coordinator.pending_transaction() is None
    assert controller._async_step_transaction is None
    assert not controller._has_pending_async_forward_state()


def _async_layout_snapshot_status(controller):
    transaction = controller._pending_async_transaction()
    if transaction is None:
        return True, False
    pending_snapshot = transaction.snapshot
    current_snapshot = AsyncLayoutSnapshot.from_context_current(
        controller.inference_wrapped_model.inference_context, tokens_per_request=1
    )
    row_map = pending_snapshot.row_map_to_current(current_snapshot.request_ids)
    if not pending_snapshot.graph_compatible_with(current_snapshot) or row_map is None:
        return False, False
    sequential_rows = torch.arange(row_map.numel(), dtype=torch.long, device="cpu")
    return True, not torch.equal(row_map, sequential_rows)


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="row mapping returns CUDA tensors")
@pytest.mark.parametrize(
    ("pending_ids", "current_ids", "expected_status", "expected_resolve", "expected_rows"),
    [
        (None, [10, 11], (True, False), (False, False), None),
        ([10, 11], [10, 11], (True, False), (True, False), None),
        ([10, 11, 12], [12, 10, 11], (True, True), (True, True), [2, 0, 1]),
        ([10, 11, 12], [12, 10], (True, True), (True, True), [2, 0]),
        ([10, 11], [10, 12], (False, False), (False, False), None),
        ([10, 11], [], (False, False), (False, False), None),
    ],
)
def test_pending_async_forward_rows_reuse_map_or_discard(
    pending_ids, current_ids, expected_status, expected_resolve, expected_rows
):
    controller = _make_controller_with_rows(pending_ids, current_ids)

    assert controller._pending_async_forward_row_status() == expected_status
    usable, row_indices, row_mapped = controller._resolve_pending_async_forward()

    assert (usable, row_mapped) == expected_resolve
    if expected_rows is None:
        assert row_indices is None
    else:
        assert row_indices.tolist() == expected_rows
    expected_discards = 0 if pending_ids is None else int(not expected_resolve[0])
    assert controller._async_discarded_forward_count == expected_discards
    assert controller._async_row_mapped_forward_count == int(expected_resolve[1])


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="row mapping returns CUDA tensors")
def test_pending_async_forward_rows_discard_when_graph_shape_changes():
    controller = _make_controller_with_rows([10, 11, 12], [12, 10, 11], current_graph_count=4)

    assert controller._pending_async_forward_row_status() == (False, False)
    assert controller._resolve_pending_async_forward() == (False, None, False)
    assert controller._async_discarded_forward_count == 1


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="row mapping returns CUDA tensors")
@pytest.mark.parametrize(
    ("pending_ids", "current_ids", "current_graph_count", "expected_status"),
    [
        ([10, 11], [10, 11], None, (True, False)),
        ([10, 11, 12], [12, 10, 11], None, (True, True)),
        ([10, 11, 12], [12, 10], None, (True, True)),
        ([10, 11], [10, 12], None, (False, False)),
        ([10, 11], [], None, (False, False)),
        ([10, 11, 12], [12, 10, 11], 4, (False, False)),
    ],
)
def test_async_layout_snapshot_matches_pending_forward_row_decisions(
    pending_ids, current_ids, current_graph_count, expected_status
):
    controller = _make_controller_with_rows(
        pending_ids, current_ids, current_graph_count=current_graph_count
    )

    assert controller._pending_async_forward_row_status() == expected_status
    assert _async_layout_snapshot_status(controller) == expected_status


@pytest.mark.internal
def test_async_pending_forward_decision_respects_row_map_policy():
    pending = _make_async_layout_snapshot([10, 11, 12], cuda_graph_request_count=3)
    current = _make_async_layout_snapshot([12, 10], cuda_graph_request_count=3)

    reuse = resolve_async_pending_forward(
        pending, current, row_map_policy=AsyncRowMapPolicy.REUSE
    )
    identity_only = resolve_async_pending_forward(
        pending, current, row_map_policy=AsyncRowMapPolicy.IDENTITY_ONLY
    )

    assert reuse.reusable
    assert reuse.row_mapped
    assert reuse.row_map.tolist() == [2, 0]
    assert reuse.diagnostics()["row_map_policy"] == "reuse"
    assert not identity_only.reusable
    assert identity_only.row_mapped
    assert identity_only.reason == "row map policy rejected non-identity layout"
    assert identity_only.diagnostics()["row_map_policy"] == "identity_only"


@pytest.mark.internal
def test_async_decode_plan_owns_pending_forward_layout_decision():
    pending = _make_async_layout_snapshot([10, 11, 12], cuda_graph_request_count=3)
    current = _make_async_layout_snapshot([12, 10], cuda_graph_request_count=3)
    plan = AsyncDecodePlan.from_snapshot(pending)

    decision = plan.resolve_pending_forward(current, row_map_policy=AsyncRowMapPolicy.REUSE)
    resolved_plan = plan.with_pending_forward_decision(decision)

    assert decision.reusable
    assert resolved_plan.row_mapped
    assert resolved_plan.row_map.tolist() == [2, 0]
    assert resolved_plan.graph_compatible
    assert resolved_plan.layout_compatible
    assert resolved_plan.graph_shape.padded_active_request_count == 3


@pytest.mark.internal
def test_async_transaction_owns_pending_forward_resolution_transition():
    controller = _make_controller_with_rows([10, 11, 12], [12, 10])
    context = controller.inference_wrapped_model.inference_context
    transaction = controller._pending_async_transaction()

    preview = transaction.pending_forward_decision(
        context, row_map_policy=AsyncRowMapPolicy.REUSE
    )

    assert preview.reusable
    assert preview.row_mapped
    assert preview.row_map.tolist() == [2, 0]
    assert transaction.state == AsyncTxnState.LAUNCHED
    assert transaction.row_map is None
    assert transaction.plan.row_map is None

    resolved = transaction.resolve_against_current(
        context, row_map_policy=AsyncRowMapPolicy.REUSE
    )

    assert resolved.diagnostics() == preview.diagnostics()
    assert transaction.state == AsyncTxnState.RESOLVED
    assert transaction.row_map.tolist() == [2, 0]
    assert transaction.plan.row_map.tolist() == [2, 0]
    assert transaction.plan.row_mapped


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="row mapping returns CUDA tensors")
def test_identity_only_row_map_policy_discards_row_mapped_pending_forward():
    controller = _make_controller_with_rows([10, 11, 12], [12, 10])
    controller._async_row_map_policy = AsyncRowMapPolicy.IDENTITY_ONLY

    assert controller._pending_async_forward_row_status() == (False, False)
    reused, row_indices, row_mapped = controller._resolve_pending_async_forward()

    transaction = controller._async_step_transaction
    assert not reused
    assert row_indices is None
    assert not row_mapped
    assert transaction.discard_reason == "row map policy rejected non-identity layout"
    assert transaction.plan.row_map.tolist() == [2, 0]
    assert transaction.plan.row_mapped
    assert not transaction.plan.layout_compatible
    assert controller._async_discarded_forward_count == 1
    assert controller._async_layout_mismatch_discard_count == 1
    assert controller._async_row_mapped_forward_count == 0


@pytest.mark.internal
def test_record_pending_forward_uses_prepared_request_order():
    controller = _make_controller_with_rows(None, [10, 11, 12])
    context = controller.inference_wrapped_model.inference_context
    cleared = []
    context.async_prepared_request_ids_cpu = lambda: torch.tensor([10, 12, 11], dtype=torch.int32)
    context.clear_async_prepared_decode_plan = lambda: cleared.append(True)

    transaction = controller._begin_async_step_transaction(cuda_graph_request_count=3)

    assert transaction.snapshot.request_ids.tolist() == [10, 12, 11]
    assert cleared == [True]


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="row mapping returns CUDA tensors")
def test_pending_async_forward_discards_when_planned_layout_mismatches_current():
    controller = _make_controller_with_rows([10, 11], [10, 11])
    context = controller.inference_wrapped_model.inference_context
    context.request_query_lengths = torch.tensor([1, 1], dtype=torch.int32)
    context.request_kv_length_offsets = torch.tensor([6, 11], dtype=torch.int64)
    context.token_to_request_idx = torch.tensor([0, 1], dtype=torch.int32)
    context.token_to_pos_ids = torch.tensor([6, 11], dtype=torch.int64)
    context.token_to_block_idx = torch.tensor([100, 200], dtype=torch.int32)
    context.token_to_local_position_within_kv_block = torch.tensor([6, 11], dtype=torch.int32)
    pending_snapshot = _make_async_layout_snapshot(
        [10, 11],
        cuda_graph_request_count=2,
        request_query_lengths=torch.tensor([1, 1], dtype=torch.int32),
        request_kv_length_offsets=torch.tensor([6, 11], dtype=torch.int64),
        token_to_request_idx=torch.tensor([[0], [1]], dtype=torch.int32),
        token_to_pos_ids=torch.tensor([[6], [11]], dtype=torch.int64),
        token_to_block_idx=torch.tensor([[100], [201]], dtype=torch.int32),
        token_to_local_position_within_kv_block=torch.tensor(
            [[6], [11]], dtype=torch.int32
        ),
    )
    _install_pending_transaction(controller, pending_snapshot)
    current_snapshot = AsyncLayoutSnapshot.from_context_current(context, tokens_per_request=1)
    row_map = pending_snapshot.row_map_to_current(current_snapshot.request_ids)

    assert controller._pending_async_forward_row_status() == (False, False)
    assert row_map is not None
    assert pending_snapshot.layout_compatible_with(current_snapshot, row_map=row_map) is False
    assert controller._resolve_pending_async_forward() == (False, None, False)
    assert controller._async_discarded_forward_count == 1


@pytest.mark.internal
def test_pending_async_forward_discards_next_step_position_drift():
    controller = _make_controller_with_rows([10, 11], [10, 11])
    context = controller.inference_wrapped_model.inference_context
    context.request_query_lengths = torch.tensor([1, 1], dtype=torch.int32)
    context.request_kv_length_offsets = torch.tensor([6, 11], dtype=torch.int64)
    context.token_to_request_idx = torch.tensor([0, 1], dtype=torch.int32)
    context.token_to_pos_ids = torch.tensor([6, 11], dtype=torch.int64)
    context.token_to_block_idx = torch.tensor([100, 200], dtype=torch.int32)
    context.token_to_local_position_within_kv_block = torch.tensor([6, 11], dtype=torch.int32)
    pending_snapshot = _make_async_layout_snapshot(
        [10, 11],
        cuda_graph_request_count=2,
        request_query_lengths=torch.tensor([1, 1], dtype=torch.int32),
        request_kv_length_offsets=torch.tensor([7, 12], dtype=torch.int64),
        token_to_request_idx=torch.tensor([[0], [1]], dtype=torch.int32),
        token_to_pos_ids=torch.tensor([[7], [12]], dtype=torch.int64),
        token_to_block_idx=torch.tensor([[100], [200]], dtype=torch.int32),
        token_to_local_position_within_kv_block=torch.tensor([[7], [12]], dtype=torch.int32),
    )
    _install_pending_transaction(controller, pending_snapshot)

    assert controller._pending_async_forward_row_status() == (False, False)
    assert controller._resolve_pending_async_forward() == (False, None, False)
    transaction = controller._async_step_transaction
    assert transaction.discard_reason == "layout mismatch"
    assert controller._async_discarded_forward_count == 1


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="row mapping returns CUDA tensors")
def test_pending_async_forward_reuses_subset_when_finished_row_left():
    controller = _make_controller_with_rows([10, 11, 12], [12, 10])
    context = controller.inference_wrapped_model.inference_context
    context.request_query_lengths = torch.tensor([1, 1], dtype=torch.int32)
    context.request_kv_length_offsets = torch.tensor([21, 5], dtype=torch.int64)
    context.token_to_request_idx = torch.tensor([0, 1], dtype=torch.int32)
    context.token_to_pos_ids = torch.tensor([21, 5], dtype=torch.int64)
    context.token_to_block_idx = torch.tensor([300, 100], dtype=torch.int32)
    context.token_to_local_position_within_kv_block = torch.tensor([21, 5], dtype=torch.int32)
    pending_snapshot = _make_async_layout_snapshot(
        [10, 11, 12],
        cuda_graph_request_count=3,
        request_query_lengths=torch.tensor([1, 1, 1], dtype=torch.int32),
        request_kv_length_offsets=torch.tensor([5, 13, 21], dtype=torch.int64),
        token_to_request_idx=torch.tensor([[0], [1], [2]], dtype=torch.int32),
        token_to_pos_ids=torch.tensor([[5], [13], [21]], dtype=torch.int64),
        token_to_block_idx=torch.tensor([[100], [200], [300]], dtype=torch.int32),
        token_to_local_position_within_kv_block=torch.tensor(
            [[5], [13], [21]], dtype=torch.int32
        ),
    )
    _install_pending_transaction(controller, pending_snapshot)
    current_snapshot = AsyncLayoutSnapshot.from_context_current(context, tokens_per_request=1)
    row_map = pending_snapshot.row_map_to_current(current_snapshot.request_ids)

    reused, row_indices, row_mapped = controller._resolve_pending_async_forward()

    assert reused
    assert row_mapped
    assert row_indices.tolist() == [2, 0]
    assert row_map is not None
    assert row_map.tolist() == [2, 0]
    assert pending_snapshot.layout_compatible_with(current_snapshot, row_map=row_map)
    transaction = controller._async_step_transaction
    assert transaction.plan.row_map.tolist() == [2, 0]
    assert transaction.plan.row_mapped
    assert transaction.plan.graph_compatible
    assert transaction.plan.layout_compatible
    assert controller._async_discarded_forward_count == 0


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="row mapping returns CUDA tensors")
def test_pending_async_forward_discards_when_token_request_layout_mismatches():
    controller = _make_controller_with_rows([10, 11], [10, 11])
    context = controller.inference_wrapped_model.inference_context
    context.request_query_lengths = torch.tensor([1, 1], dtype=torch.int32)
    context.request_kv_length_offsets = torch.tensor([6, 11], dtype=torch.int64)
    context.token_to_request_idx = torch.tensor([0, 1], dtype=torch.int32)
    _install_pending_transaction(
        controller,
        _make_async_layout_snapshot(
            [10, 11],
            cuda_graph_request_count=2,
            token_to_request_idx=torch.tensor([[0], [0]], dtype=torch.int32),
        ),
    )

    assert controller._pending_async_forward_row_status() == (False, False)
    assert controller._resolve_pending_async_forward() == (False, None, False)
    assert controller._async_discarded_forward_count == 1


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="row mapping returns CUDA tensors")
def test_pending_async_forward_discards_when_mamba_bank_layout_mismatches():
    controller = _make_controller_with_rows([10, 11], [10, 11])
    context = controller.inference_wrapped_model.inference_context
    context.is_hybrid_model = True
    context.request_query_lengths = torch.tensor([1, 1], dtype=torch.int32)
    context.request_kv_length_offsets = torch.tensor([6, 11], dtype=torch.int64)
    read_indices = torch.tensor([6, 9], dtype=torch.int32)
    write_indices = torch.tensor([7, 8], dtype=torch.int32)
    context._mamba_flat_indices = lambda _active_slice, use_candidate_bank=False: (
        write_indices if use_candidate_bank else read_indices
    )
    _install_pending_transaction(
        controller,
        _make_async_layout_snapshot(
            [10, 11],
            cuda_graph_request_count=2,
            mamba_read_indices=torch.tensor([6, 8], dtype=torch.int32),
            mamba_write_indices=torch.tensor([7, 8], dtype=torch.int32),
        ),
    )

    assert controller._pending_async_forward_row_status() == (False, False)
    assert controller._resolve_pending_async_forward() == (False, None, False)
    assert controller._async_discarded_forward_count == 1


@pytest.mark.internal
@pytest.mark.asyncio
async def test_reused_pending_forward_prepares_next_step_before_sampling(monkeypatch):
    monkeypatch.setattr(tgc_module, "range_push", lambda _msg: None)
    monkeypatch.setattr(tgc_module, "range_pop", lambda: None)
    events = []
    context = SimpleNamespace(
        active_token_count=2,
        total_request_count=2,
        paused_request_count=0,
        num_decode_requests=2,
        is_hybrid_model=False,
        config=SimpleNamespace(materialize_only_last_token_logits=True),
        kv_block_allocator=SimpleNamespace(
            store_routing_per_block=lambda _routing: events.append("routing")
        ),
        release_deferred_async_resources=lambda: events.append("release"),
    )
    controller = object.__new__(TextGenerationController)
    controller.inference_wrapped_model = SimpleNamespace(inference_context=context)
    controller.num_speculative_tokens = 0
    _install_pending_transaction(
        controller, _make_async_layout_snapshot([10, 11], cuda_graph_request_count=2)
    )
    controller._async_prepare_deferred_until_after_sampling = False
    controller._decide_ep_step_begin = lambda **_kwargs: EPStepBeginDecision(
        step_id=0,
        has_real_work=True,
        reuse_pending_forward=True,
        discard_pending_forward=False,
        row_mapped_forward=False,
    )
    controller._resolve_pending_async_forward = lambda: (True, None, False)
    controller._router_record_bookkeeping = lambda: None
    controller._should_collect_dynamic_sampling_bookkeeping = lambda **_kwargs: False
    controller._try_prepare_async_decode_before_sampling = lambda: events.append("precheck") or True
    controller._dynamic_step_sample_logits_to_next_input_ids = (
        lambda: events.extend(["sample", "copy"])
    )
    controller._try_prepare_async_decode_after_sampling = lambda: pytest.fail(
        "reused non-row-mapped forward should prepare before sampling"
    )
    controller._transfer_async_samples_to_cpu = (
        lambda count: events.append(("d2h", count)) or _sample_ticket([4, 5])
    )
    controller._confirm_prepared_ep_async_handoff = lambda: False
    controller._ensure_ep_async_handoff_decided = lambda **_kwargs: events.append("ep_done")

    result = await controller.async_generate_output_tokens_dynamic_batch(skip_bookkeeping=True)

    ordering = [event for event in events if event in ("precheck", "sample", "copy")]
    assert ordering == ["precheck", "sample", "copy"]
    assert ("d2h", 2) in events
    assert result["sample"].tolist() == [4, 5]


@pytest.mark.internal
@pytest.mark.asyncio
async def test_reused_pending_forward_falls_back_after_sampling_when_presampling_declines(
    monkeypatch,
):
    monkeypatch.setattr(tgc_module, "range_push", lambda _msg: None)
    monkeypatch.setattr(tgc_module, "range_pop", lambda: None)
    events = []
    context = SimpleNamespace(
        active_token_count=2,
        total_request_count=2,
        paused_request_count=0,
        num_decode_requests=2,
        is_hybrid_model=False,
        config=SimpleNamespace(materialize_only_last_token_logits=True),
        kv_block_allocator=SimpleNamespace(
            store_routing_per_block=lambda _routing: events.append("routing")
        ),
        release_deferred_async_resources=lambda: events.append("release"),
    )
    controller = object.__new__(TextGenerationController)
    controller.inference_wrapped_model = SimpleNamespace(inference_context=context)
    controller.num_speculative_tokens = 0
    _install_pending_transaction(
        controller, _make_async_layout_snapshot([10, 11], cuda_graph_request_count=2)
    )
    controller._async_prepare_deferred_until_after_sampling = False
    controller._decide_ep_step_begin = lambda **_kwargs: EPStepBeginDecision(
        step_id=0,
        has_real_work=True,
        reuse_pending_forward=True,
        discard_pending_forward=False,
        row_mapped_forward=False,
    )
    controller._resolve_pending_async_forward = lambda: (True, None, False)
    controller._router_record_bookkeeping = lambda: None
    controller._should_collect_dynamic_sampling_bookkeeping = lambda **_kwargs: False
    controller._try_prepare_async_decode_before_sampling = lambda: events.append("precheck") or False
    controller._dynamic_step_sample_logits = lambda **_kwargs: events.append("sample")
    controller._try_prepare_async_decode_after_sampling = (
        lambda: events.append("prepare_after") or True
    )
    controller._copy_sampled_decode_tokens_to_next_input_ids = lambda count: events.append(
        "copy" if count == 2 else ("copy", count)
    )
    controller._transfer_async_samples_to_cpu = (
        lambda count: events.append(("d2h", count)) or _sample_ticket([4, 5])
    )
    controller._confirm_prepared_ep_async_handoff = lambda: False
    controller._ensure_ep_async_handoff_decided = lambda **_kwargs: events.append("ep_done")

    result = await controller.async_generate_output_tokens_dynamic_batch(skip_bookkeeping=True)

    ordering = [
        event
        for event in events
        if event in ("precheck", "sample", "prepare_after", "copy")
    ]
    assert ordering == ["precheck", "sample", "prepare_after", "copy"]
    assert ("d2h", 2) in events
    assert result["sample"].tolist() == [4, 5]


@pytest.mark.internal
@pytest.mark.asyncio
async def test_prepare_async_decode_before_sampling_steady_state_ordering(monkeypatch):
    monkeypatch.setattr(tgc_module, "range_push", lambda _msg: None)
    monkeypatch.setattr(tgc_module, "range_pop", lambda: None)
    events = []
    context = SimpleNamespace(
        active_token_count=2,
        total_request_count=2,
        paused_request_count=0,
        num_decode_requests=2,
        padded_active_request_count=2,
        is_hybrid_model=False,
        config=SimpleNamespace(materialize_only_last_token_logits=True),
        kv_block_allocator=SimpleNamespace(
            store_routing_per_block=lambda _routing: events.append("routing")
        ),
        publish_async_prepared_decode_plan=lambda: events.append("publish"),
        transfer_bookkeeping_to_gpu=lambda **_kwargs: events.append("h2d") or None,
        current_input_and_position_ids=lambda: (
            torch.tensor([[1, 2]], dtype=torch.int64),
            torch.tensor([[0, 1]], dtype=torch.int64),
        ),
        using_cuda_graph_this_step=lambda: True,
        mark_async_resources_in_flight=lambda: events.append("mark_in_flight") or "ledger",
    )

    def _prepare(**kwargs):
        events.append("prepare" if kwargs.get("pre_sampling") else "prepare_after")
        return True

    context.prepare_async_decode_next_step = _prepare
    controller = object.__new__(TextGenerationController)
    controller.inference_wrapped_model = SimpleNamespace(
        inference_context=context,
        model=SimpleNamespace(config=SimpleNamespace(moe_enable_routing_replay=False)),
    )
    controller.num_speculative_tokens = 0
    controller._async_step_transaction = None
    controller._async_prepare_deferred_until_after_sampling = False
    controller._async_forward_launch_count = 0
    controller._decide_ep_step_begin = lambda **_kwargs: EPStepBeginDecision(
        step_id=0,
        has_real_work=True,
        reuse_pending_forward=False,
        discard_pending_forward=False,
        row_mapped_forward=False,
    )
    controller._dynamic_step_context_init = lambda: (
        torch.tensor([[1, 2]], dtype=torch.int64),
        torch.tensor([[0, 1]], dtype=torch.int64),
    )
    controller._dynamic_step_forward_logits = lambda *_args: events.append("forward")
    controller._router_record_bookkeeping = lambda: None
    controller._async_scheduling_disabled_reason = lambda **_kwargs: None
    controller._record_async_eligibility_result = lambda _reason: None
    controller._record_async_disable_reason = lambda reason: events.append(("disable", reason))
    controller._decide_ep_async_handoff = (
        lambda **_kwargs: events.append("handoff")
        or EPAsyncHandoffDecision(
            step_id=0,
            has_real_work=True,
            launch_async_forward=True,
            skip_async_forward=False,
            any_launch_request=True,
            any_skip_request=False,
        )
    )
    controller._should_collect_dynamic_sampling_bookkeeping = lambda **_kwargs: False
    controller._dynamic_step_sample_logits_to_next_input_ids = (
        lambda: events.extend(["sample", "copy"])
    )
    controller._transfer_async_samples_to_cpu = (
        lambda count: events.append(("d2h", count)) or _sample_ticket([4, 5])
    )
    controller._confirm_prepared_ep_async_handoff = lambda: True
    controller._begin_async_step_transaction = lambda _count: events.append(
        "record"
    ) or SimpleNamespace(mark_launched=lambda **_kwargs: events.append("tx_launch"))
    controller._ensure_ep_async_handoff_decided = lambda **_kwargs: events.append("ep_done")

    result = await controller.async_generate_output_tokens_dynamic_batch(skip_bookkeeping=True)

    ordering = [
        event
        for event in events
        if event in ("prepare", "handoff", "sample", "copy", "publish", "h2d")
    ]
    assert ordering == ["prepare", "handoff", "sample", "copy", "publish", "h2d"]
    assert result["sample"].tolist() == [4, 5]


@pytest.mark.internal
@pytest.mark.asyncio
async def test_prepare_async_decode_before_sampling_unsafe_fallback_ordering(monkeypatch):
    monkeypatch.setattr(tgc_module, "range_push", lambda _msg: None)
    monkeypatch.setattr(tgc_module, "range_pop", lambda: None)
    events = []
    context = SimpleNamespace(
        active_token_count=2,
        total_request_count=2,
        paused_request_count=0,
        num_decode_requests=2,
        padded_active_request_count=2,
        is_hybrid_model=False,
        config=SimpleNamespace(materialize_only_last_token_logits=True),
        kv_block_allocator=SimpleNamespace(
            store_routing_per_block=lambda _routing: events.append("routing")
        ),
        publish_async_prepared_decode_plan=lambda: events.append("publish"),
        transfer_bookkeeping_to_gpu=lambda **_kwargs: events.append("h2d") or None,
        current_input_and_position_ids=lambda: (
            torch.tensor([[1, 2]], dtype=torch.int64),
            torch.tensor([[0, 1]], dtype=torch.int64),
        ),
        using_cuda_graph_this_step=lambda: True,
        mark_async_resources_in_flight=lambda: events.append("mark_in_flight") or "ledger",
    )

    def _prepare(**kwargs):
        if kwargs.get("pre_sampling"):
            events.append("prepare_pre")
            return False
        events.append("prepare_after")
        return True

    context.prepare_async_decode_next_step = _prepare
    controller = object.__new__(TextGenerationController)
    controller.inference_wrapped_model = SimpleNamespace(
        inference_context=context,
        model=SimpleNamespace(config=SimpleNamespace(moe_enable_routing_replay=False)),
    )
    controller.num_speculative_tokens = 0
    controller._async_step_transaction = None
    controller._async_prepare_deferred_until_after_sampling = False
    controller._async_forward_launch_count = 0
    controller._decide_ep_step_begin = lambda **_kwargs: EPStepBeginDecision(
        step_id=0,
        has_real_work=True,
        reuse_pending_forward=False,
        discard_pending_forward=False,
        row_mapped_forward=False,
    )
    controller._dynamic_step_context_init = lambda: (
        torch.tensor([[1, 2]], dtype=torch.int64),
        torch.tensor([[0, 1]], dtype=torch.int64),
    )
    controller._dynamic_step_forward_logits = lambda *_args: events.append("forward")
    controller._router_record_bookkeeping = lambda: None
    controller._async_scheduling_disabled_reason = lambda **_kwargs: None
    controller._record_async_eligibility_result = lambda _reason: None
    controller._record_async_disable_reason = lambda reason: events.append(("disable", reason))
    controller._decide_ep_async_handoff = (
        lambda **_kwargs: EPAsyncHandoffDecision(
            step_id=0,
            has_real_work=True,
            launch_async_forward=True,
            skip_async_forward=False,
            any_launch_request=True,
            any_skip_request=False,
        )
    )
    controller._should_collect_dynamic_sampling_bookkeeping = lambda **_kwargs: False
    controller._dynamic_step_sample_logits = lambda **_kwargs: events.append("sample")
    controller._copy_sampled_decode_tokens_to_next_input_ids = lambda count: events.append(
        ("copy", count)
    )
    controller._transfer_async_samples_to_cpu = (
        lambda count: events.append(("d2h", count)) or _sample_ticket([4, 5])
    )
    controller._confirm_prepared_ep_async_handoff = lambda: True
    controller._begin_async_step_transaction = lambda _count: events.append(
        "record"
    ) or SimpleNamespace(mark_launched=lambda **_kwargs: events.append("tx_launch"))
    controller._ensure_ep_async_handoff_decided = lambda **_kwargs: events.append("ep_done")

    result = await controller.async_generate_output_tokens_dynamic_batch(skip_bookkeeping=True)

    ordering = [
        event
        for event in events
        if event in ("sample", "prepare_after", "publish", "h2d") or event == ("copy", 2)
    ]
    assert ordering == ["sample", "prepare_after", ("copy", 2), "publish", "h2d"]
    assert not any(event[0] == "disable" for event in events if isinstance(event, tuple))
    assert result["sample"].tolist() == [4, 5]


@pytest.mark.internal
def test_controller_handoff_decision_is_cached_and_skip_can_be_forced():
    class _Protocol:
        enabled = True

        def __init__(self):
            self.calls = []

        def decide_async_handoff(self, *, has_real_work, can_launch_async_handoff):
            self.calls.append((has_real_work, can_launch_async_handoff))
            return EPAsyncHandoffDecision(
                step_id=12,
                has_real_work=has_real_work,
                launch_async_forward=can_launch_async_handoff,
                skip_async_forward=not can_launch_async_handoff,
                any_launch_request=can_launch_async_handoff,
                any_skip_request=not can_launch_async_handoff,
            )

    controller = object.__new__(TextGenerationController)
    controller._ep_async_protocol = _Protocol()
    controller._ep_async_handoff_decision_this_step = None
    controller._ep_async_handoff_decided_this_step = False

    first = controller._decide_ep_async_handoff(has_real_work=True, can_launch_async_handoff=True)
    second = controller._decide_ep_async_handoff(
        has_real_work=False, can_launch_async_handoff=False
    )

    assert first is second
    assert controller._ep_async_protocol.calls == [(True, True)]

    controller._ep_async_handoff_decision_this_step = None
    controller._ep_async_handoff_decided_this_step = False
    controller._ensure_ep_async_handoff_decided(has_real_work=True)
    assert controller._ep_async_protocol.calls[-1] == (True, False)


@pytest.mark.internal
@pytest.mark.parametrize(
    ("use_protocol", "pending_forward", "row_status", "expected"),
    [
        (False, True, (True, False), (True, False, False)),
        (False, True, (False, False), (False, True, False)),
        (False, True, (True, True), (True, False, True)),
        (False, False, (True, False), (False, False, False)),
        (True, True, (True, True), (True, False, True)),
    ],
)
def test_controller_step_begin_bridges_local_and_ep_protocol_decisions(
    use_protocol, pending_forward, row_status, expected
):
    controller = object.__new__(TextGenerationController)
    controller._has_pending_async_forward_state = lambda: pending_forward
    controller._ep_async_handoff_decided_this_step = True
    controller._ep_async_handoff_decision_this_step = object()
    controller._pending_async_forward_row_status = lambda: row_status
    protocol_calls = []

    class _Protocol:
        enabled = True

        def decide_step_begin(self, **kwargs):
            protocol_calls.append(kwargs)
            return EPStepBeginDecision(
                step_id=4,
                has_real_work=kwargs["has_real_work"],
                reuse_pending_forward=kwargs["pending_forward_reusable"],
                discard_pending_forward=not kwargs["pending_forward_reusable"],
                row_mapped_forward=kwargs["pending_forward_row_mapped"],
            )

    if use_protocol:
        controller._ep_async_protocol = _Protocol()
    else:
        controller._ep_async_protocol = None

    decision = controller._decide_ep_step_begin(has_real_work=True)

    assert controller._ep_async_handoff_decided_this_step is False
    assert controller._ep_async_handoff_decision_this_step is None
    if use_protocol:
        assert protocol_calls == [
            {
                "has_real_work": True,
                "has_pending_forward": pending_forward,
                "pending_forward_reusable": row_status[0],
                "pending_forward_row_mapped": row_status[1],
            }
        ]
        assert (
            decision.reuse_pending_forward,
            decision.discard_pending_forward,
            decision.row_mapped_forward,
        ) == expected
    else:
        assert (
            decision.reuse_pending_forward,
            decision.discard_pending_forward,
            decision.row_mapped_forward,
        ) == expected


@pytest.mark.internal
def test_controller_step_begin_records_ep_decision_on_transaction_participant():
    controller = object.__new__(TextGenerationController)
    controller._ep_async_protocol = None
    controller._ep_async_handoff_decided_this_step = True
    controller._ep_async_handoff_decision_this_step = object()
    transaction = _install_pending_transaction(
        controller, _make_async_layout_snapshot([10, 11], cuda_graph_request_count=2)
    )
    controller._pending_async_forward_row_status = lambda: (True, True)

    decision = controller._decide_ep_step_begin(has_real_work=True)

    participant = next(
        participant
        for participant in transaction.participants
        if isinstance(participant, AsyncEPParticipant)
    )
    diagnostics = participant.diagnostics()
    assert decision.row_mapped_forward
    assert diagnostics["prepared"]
    assert diagnostics["step_begin"] == {
        "step_id": -1,
        "has_real_work": True,
        "reuse_pending_forward": True,
        "discard_pending_forward": False,
        "row_mapped_forward": True,
    }


@pytest.mark.internal
def test_controller_ep_handoff_participant_attaches_to_launch_transaction():
    controller = object.__new__(TextGenerationController)
    controller._ep_async_protocol = None
    controller._ep_async_handoff_decided_this_step = False
    controller._ep_async_handoff_decision_this_step = None
    controller.num_speculative_tokens = 0
    controller._async_logprob_requests_seen = False
    controller.inference_wrapped_model = SimpleNamespace(
        inference_context=SimpleNamespace(is_hybrid_model=False)
    )
    snapshot = _make_async_layout_snapshot([30], cuda_graph_request_count=1)
    transaction = AsyncDecodeTransaction(
        step_id=8,
        state=AsyncTxnState.PREPARED,
        snapshot=snapshot,
        plan=AsyncDecodePlan.from_snapshot(snapshot),
    )

    handoff = controller._decide_ep_async_handoff(
        has_real_work=True, can_launch_async_handoff=True
    )
    controller._attach_async_transaction_participants(
        transaction, resources=None, sample_ticket=None
    )

    participant = next(
        participant
        for participant in transaction.participants
        if isinstance(participant, AsyncEPParticipant)
    )
    assert handoff.launch_async_forward
    assert controller._async_ep_participant_this_step is None
    assert transaction.participant_state["AsyncEPParticipant"]["handoff"] == {
        "step_id": -1,
        "has_real_work": True,
        "launch_async_forward": True,
        "skip_async_forward": False,
        "any_launch_request": True,
        "any_skip_request": False,
    }
    transaction.rollback("handoff invalidated")
    assert participant.diagnostics()["rolled_back"]


def _make_async_gate_controller(active_request_count=2):
    controller = object.__new__(TextGenerationController)
    controller._async_scheduling_enabled = True
    controller._async_step_barrier_reason = None
    controller._enable_cuda_graph = True
    controller.model_config = SimpleNamespace(
        inference_cuda_graph_scope=InferenceCudaGraphScope.block
    )
    controller.model_is_pipeline_parallel = False
    controller.num_speculative_tokens = 0
    controller._num_mtp_depths = 0
    controller._sampling_backend = "torch"
    controller._async_admission_barrier_requested = False
    controller._async_prepare_deferred_until_after_sampling = False
    context = SimpleNamespace(
        total_request_count=active_request_count,
        paused_request_count=0,
        padded_batch_dimensions=InferenceBatchDimensions(
            active_request_count, 0, active_request_count
        ),
        is_decode_only=lambda: True,
        using_cuda_graph_this_step=lambda: True,
        is_hybrid_model=False,
        discard_async_prepared_decode_plan=lambda: None,
    )
    controller.inference_wrapped_model = SimpleNamespace(inference_context=context)
    return controller, context


@pytest.mark.internal
@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("eligible", None),
        ("disabled", "disabled"),
        ("step_barrier", "logging step"),
        ("no_cuda_graph", "requires local cuda graphs"),
        ("scope_none", "requires block-scope inference cuda graphs"),
        ("scope_layer", "requires block-scope inference cuda graphs"),
        ("pipeline_parallel", "pipeline parallel is unsupported"),
        ("mtp_presampling", "mtp pre-sampling graph is unsupported"),
        ("mtp_depth_mismatch", "not enough mtp heads"),
        ("flashinfer", "sampling backend is unsupported"),
        ("prefill", "not decode-only"),
        ("eager_step", "not using cuda graph"),
        ("empty", "no active requests"),
        ("admission_barrier", "waiting request admission deferred"),
        ("stride_mismatch", "cuda graph shape does not match decode stride"),
    ],
)
def test_async_scheduling_disabled_reason_matrix(case, expected):
    controller, context = _make_async_gate_controller()
    allow_mtp = False
    if case == "disabled":
        controller._async_scheduling_enabled = False
    elif case == "step_barrier":
        controller._async_step_barrier_reason = "logging step"
    elif case == "no_cuda_graph":
        controller._enable_cuda_graph = False
    elif case == "scope_none":
        controller.model_config.inference_cuda_graph_scope = InferenceCudaGraphScope.none
    elif case == "scope_layer":
        controller.model_config.inference_cuda_graph_scope = InferenceCudaGraphScope.layer
    elif case == "pipeline_parallel":
        controller.model_is_pipeline_parallel = True
    elif case == "mtp_presampling":
        controller.num_speculative_tokens = 2
        controller._num_mtp_depths = 2
    elif case == "mtp_depth_mismatch":
        controller.num_speculative_tokens = 2
        controller._num_mtp_depths = 1
        allow_mtp = True
    elif case == "flashinfer":
        controller._sampling_backend = "flashinfer"
    elif case == "prefill":
        context.is_decode_only = lambda: False
    elif case == "eager_step":
        context.using_cuda_graph_this_step = lambda: False
    elif case == "empty":
        context.total_request_count = 0
        context.padded_batch_dimensions = InferenceBatchDimensions(0, 0, 0)
    elif case == "admission_barrier":
        controller._async_admission_barrier_requested = True
    elif case == "stride_mismatch":
        controller.num_speculative_tokens = 1
        controller._num_mtp_depths = 1
        allow_mtp = True

    decision = classify_async_eligibility(controller, context, allow_mtp=allow_mtp)
    assert decision.reason == expected
    assert decision.can_prepare is (expected is None)
    assert decision.can_launch is (expected is None)
    if case == "admission_barrier":
        assert not controller._async_admission_barrier_requested
        controller._async_admission_barrier_requested = True

    assert controller._async_scheduling_disabled_reason(allow_mtp=allow_mtp) == expected
    if case == "admission_barrier":
        assert not controller._async_admission_barrier_requested


@pytest.mark.internal
@pytest.mark.parametrize(
    (
        "top_k",
        "top_p",
        "return_log_probs",
        "top_n_logprobs",
        "logprobs_seen",
        "bookkeeping_state",
        "expected",
    ),
    [
        ([1, 1], [0.0, 0.0], [False, False], [0, 0], False, (False, False), True),
        ([4, 1], [0.0, 0.0], [False, False], [0, 0], False, (True, False), True),
        ([1, 1], [0.0, 0.0], [True, False], [0, 0], True, (True, False), True),
        ([1, 1], [0.0, 0.0], [False, False], [2, 0], True, (True, False), True),
        ([1, 1], [0.0, 0.0], [False, False], [0, 0], True, (True, True), False),
    ],
)
def test_async_sampling_and_logprob_bookkeeping_matrix(
    top_k, top_p, return_log_probs, top_n_logprobs, logprobs_seen, bookkeeping_state, expected
):
    controller, context = _make_async_gate_controller(active_request_count=2)
    controller._sampling_backend = "torch"
    controller._async_logprob_requests_seen = logprobs_seen
    context.active_request_metadata = {
        "top_k": torch.tensor(top_k, dtype=torch.int32),
        "top_p": torch.tensor(top_p, dtype=torch.float32),
        "return_log_probs": torch.tensor(return_log_probs, dtype=torch.bool),
        "top_n_logprobs": torch.tensor(top_n_logprobs, dtype=torch.int32),
    }
    async_next_prepared, pending_forward_reused = bookkeeping_state

    assert (
        controller._should_collect_dynamic_sampling_bookkeeping(
            async_next_prepared=async_next_prepared, pending_forward_reused=pending_forward_reused
        )
        is expected
    )


def _install_async_prepare_stubs(
    controller, *, disabled_reason=None, prepare_result=True, launch_decision=True
):
    events = []
    context = controller.inference_wrapped_model.inference_context
    context.prepare_async_decode_next_step = (
        lambda **kwargs: events.append(("prepare", kwargs)) or prepare_result
    )
    controller._async_scheduling_disabled_reason = (
        lambda **_kwargs: events.append(("disabled", _kwargs)) or disabled_reason
    )
    controller._record_async_eligibility_result = lambda reason: events.append(
        ("eligibility", reason)
    )
    controller._record_async_disable_reason = lambda reason: events.append(("disable", reason))

    def _handoff(*, has_real_work, can_launch_async_handoff):
        events.append(("handoff", has_real_work, can_launch_async_handoff))
        return EPAsyncHandoffDecision(
            step_id=0,
            has_real_work=has_real_work,
            launch_async_forward=launch_decision,
            skip_async_forward=not launch_decision,
            any_launch_request=can_launch_async_handoff,
            any_skip_request=not can_launch_async_handoff,
        )

    controller._decide_ep_async_handoff = _handoff
    return events


@pytest.mark.internal
@pytest.mark.parametrize(
    ("case", "expected_ok", "expected_deferred", "expected_disable_reason"),
    [
        ("disabled", False, False, None),
        ("prepare_deferred", False, True, None),
        ("handoff_skipped", False, False, "ep async handoff skipped"),
        ("success", True, False, None),
    ],
)
def test_prepare_async_decode_before_sampling_handoff_paths(
    monkeypatch, case, expected_ok, expected_deferred, expected_disable_reason
):
    monkeypatch.setattr(tgc_module, "range_push", lambda _msg: None)
    monkeypatch.setattr(tgc_module, "range_pop", lambda: None)
    controller, _context = _make_async_gate_controller()
    events = _install_async_prepare_stubs(
        controller,
        disabled_reason="blocked" if case == "disabled" else None,
        prepare_result=case != "prepare_deferred",
        launch_decision=case != "handoff_skipped",
    )

    ok = controller._try_prepare_async_decode_before_sampling()

    assert ok is expected_ok
    assert controller._async_prepare_deferred_until_after_sampling is expected_deferred
    if case == "disabled":
        assert ("handoff", True, False) in events
        assert not any(event[0] == "prepare" for event in events if isinstance(event, tuple))
    elif case == "prepare_deferred":
        assert ("prepare", {"pre_sampling": True}) in events
        assert not any(event[0] == "handoff" for event in events if isinstance(event, tuple))
    else:
        assert ("prepare", {"pre_sampling": True}) in events
        assert ("handoff", True, True) in events
    if expected_disable_reason is not None:
        assert ("disable", expected_disable_reason) in events
    if case == "prepare_deferred":
        assert not any(event[0] == "disable" for event in events if isinstance(event, tuple))


@pytest.mark.internal
def test_prepare_async_decode_before_sampling_deferral_is_not_disable_reason(monkeypatch):
    monkeypatch.setattr(tgc_module, "range_push", lambda _msg: None)
    monkeypatch.setattr(tgc_module, "range_pop", lambda: None)
    controller, context = _make_async_gate_controller()
    controller._async_disable_reason = None
    controller._async_disable_reason_counts = {}
    controller._async_eligibility_check_count = 0
    controller._async_eligibility_pass_count = 0
    events = []
    controller._async_scheduling_disabled_reason = lambda **_kwargs: None
    controller._decide_ep_async_handoff = lambda **kwargs: events.append(
        ("handoff", kwargs)
    )
    context.prepare_async_decode_next_step = (
        lambda **kwargs: events.append(("prepare", kwargs)) or False
    )

    assert not controller._try_prepare_async_decode_before_sampling()

    assert controller._async_prepare_deferred_until_after_sampling
    assert controller._async_disable_reason_counts == {}
    assert controller._async_eligibility_check_count == 1
    assert controller._async_eligibility_pass_count == 1
    assert events == [("prepare", {"pre_sampling": True})]


@pytest.mark.internal
def test_prepare_async_decode_before_sampling_keeps_hybrid_fast_path(monkeypatch):
    monkeypatch.setattr(tgc_module, "range_push", lambda _msg: None)
    monkeypatch.setattr(tgc_module, "range_pop", lambda: None)
    controller, context = _make_async_gate_controller()
    context.is_hybrid_model = True
    events = []
    controller._async_scheduling_disabled_reason = (
        lambda **kwargs: events.append(("disabled", kwargs)) or None
    )
    controller._record_async_eligibility_result = lambda reason: events.append(
        ("eligibility", reason)
    )
    controller._decide_ep_async_handoff = (
        lambda **kwargs: events.append(("handoff", kwargs))
        or SimpleNamespace(launch_async_forward=True)
    )
    context.prepare_async_decode_next_step = (
        lambda **kwargs: events.append(("prepare", kwargs)) or True
    )

    assert controller._try_prepare_async_decode_before_sampling()

    assert not controller._async_prepare_deferred_until_after_sampling
    assert events == [
        ("disabled", {}),
        ("eligibility", None),
        ("prepare", {"pre_sampling": True}),
        ("handoff", {"has_real_work": True, "can_launch_async_handoff": True}),
    ]


@pytest.mark.internal
@pytest.mark.parametrize(
    ("case", "expected_ok", "expected_disable_reason"),
    [
        ("disabled", False, None),
        ("prepare_failed", False, "failed to prepare next-step metadata"),
        ("success", True, None),
    ],
)
def test_prepare_async_decode_after_sampling_handoff_paths(
    monkeypatch, case, expected_ok, expected_disable_reason
):
    monkeypatch.setattr(tgc_module, "range_push", lambda _msg: None)
    monkeypatch.setattr(tgc_module, "range_pop", lambda: None)
    controller, _context = _make_async_gate_controller()
    events = _install_async_prepare_stubs(
        controller,
        disabled_reason="blocked" if case == "disabled" else None,
        prepare_result=case != "prepare_failed",
        launch_decision=case != "handoff_skipped",
    )

    assert controller._try_prepare_async_decode_after_sampling() is expected_ok
    assert ("disabled", {"allow_mtp": True}) in events
    if case == "disabled":
        assert not any(event[0] == "prepare" for event in events if isinstance(event, tuple))
        assert ("handoff", True, False) in events
    elif case == "prepare_failed":
        assert ("handoff", True, False) in events
    else:
        assert not any(event[0] == "handoff" for event in events if isinstance(event, tuple))
    if expected_disable_reason is not None:
        assert ("disable", expected_disable_reason) in events


@pytest.mark.internal
@pytest.mark.parametrize(
    ("launch_decision", "expected_ok", "expected_disable_reason"),
    [(True, True, None), (False, False, "ep async handoff skipped")],
)
def test_confirm_prepared_ep_async_handoff_paths(
    launch_decision, expected_ok, expected_disable_reason
):
    controller, _context = _make_async_gate_controller()
    events = _install_async_prepare_stubs(controller, launch_decision=launch_decision)

    assert controller._confirm_prepared_ep_async_handoff() is expected_ok
    assert ("handoff", True, True) in events
    if expected_disable_reason is not None:
        assert ("disable", expected_disable_reason) in events


@pytest.mark.internal
@pytest.mark.parametrize(
    ("case", "expected_ok", "expected_reset", "expected_forwards"),
    [("disabled", False, 0, 0), ("handoff_skipped", False, 0, 0), ("success", True, 1, 1)],
)
def test_dummy_async_handoff_mirrors_real_rank_launch(
    monkeypatch, case, expected_ok, expected_reset, expected_forwards
):
    monkeypatch.setattr(tgc_module, "range_push", lambda _msg: None)
    monkeypatch.setattr(tgc_module, "range_pop", lambda: None)
    context = SimpleNamespace(reset_count=0)
    context.reset = lambda: setattr(context, "reset_count", context.reset_count + 1)
    controller = object.__new__(TextGenerationController)
    controller.inference_wrapped_model = SimpleNamespace(inference_context=context)
    controller._async_forward_launch_count = 0
    controller._dummy_async_handoff_disabled_reason = lambda: (
        "blocked" if case == "disabled" else None
    )
    events = []
    controller._record_async_eligibility_result = lambda reason: events.append(
        ("eligibility", reason)
    )
    controller._record_async_disable_reason = lambda reason: events.append(("disable", reason))
    controller._wait_for_dummy_context_h2d = lambda: events.append("wait_h2d")
    controller._decide_ep_async_handoff = lambda **kwargs: events.append(
        ("handoff", kwargs)
    ) or EPAsyncHandoffDecision(
        step_id=0,
        has_real_work=False,
        launch_async_forward=(case == "success"),
        skip_async_forward=(case != "success"),
        any_launch_request=kwargs["can_launch_async_handoff"],
        any_skip_request=not kwargs["can_launch_async_handoff"],
    )
    controller._dynamic_step_context_init = lambda is_dummy_forward=False: events.append(
        ("context_init", is_dummy_forward)
    ) or ("input_ids", "position_ids")
    controller._dynamic_step_forward_logits = lambda *_args: events.append("forward")

    assert controller._try_launch_dummy_async_handoff() is expected_ok
    assert context.reset_count == expected_reset
    assert events.count("forward") == expected_forwards
    assert events.count("wait_h2d") == expected_reset
    assert controller._async_forward_launch_count == expected_forwards
    if case == "handoff_skipped":
        assert ("disable", "ep async handoff skipped") in events


@pytest.mark.internal
def test_wait_for_dummy_context_h2d_synchronizes_once():
    controller = object.__new__(TextGenerationController)
    events = []
    controller._dummy_context_h2d_done_event = SimpleNamespace(
        synchronize=lambda: events.append("sync")
    )

    controller._wait_for_dummy_context_h2d()
    controller._wait_for_dummy_context_h2d()

    assert events == ["sync"]
    assert controller._dummy_context_h2d_done_event is None


@pytest.mark.internal
def test_pending_async_forward_cleanup_releases_only_when_needed():
    context = SimpleNamespace(release_count=0)
    context.release_deferred_async_resources = lambda: setattr(
        context, "release_count", context.release_count + 1
    )
    controller = object.__new__(TextGenerationController)
    controller.inference_wrapped_model = SimpleNamespace(inference_context=context)
    controller._async_step_transaction = None
    controller._async_discarded_forward_count = 0

    controller._discard_pending_async_forward()
    assert context.release_count == 0
    assert controller._async_discarded_forward_count == 0

    transaction = _install_pending_transaction(
        controller, _make_async_layout_snapshot([1, 2], cuda_graph_request_count=2)
    )
    controller._discard_pending_async_forward()
    assert context.release_count == 1
    assert controller._async_step_transaction is None
    assert transaction.state == AsyncTxnState.DISCARDED
    assert controller._async_discarded_forward_count == 1


@pytest.mark.internal
def test_pending_async_forward_row_discard_releases_without_accepting_mamba_bank():
    controller = _make_controller_with_rows([10, 11], [10, 12])
    context = controller.inference_wrapped_model.inference_context
    context.release_deferred_async_resources = lambda: None
    context.accept_async_mamba_state = lambda _request_ids: pytest.fail(
        "discarded forwards must not commit candidate Mamba banks"
    )

    assert controller._resolve_pending_async_forward() == (False, None, False)

    assert controller._async_discarded_forward_count == 1


@pytest.mark.internal
@pytest.mark.parametrize(
    ("sampling_params", "expected_seen"),
    [
        (SamplingParams(), False),
        (SamplingParams(return_log_probs=True), True),
        (SamplingParams(top_n_logprobs=2), True),
    ],
)
def test_note_sampling_params_tracks_async_logprob_requests(sampling_params, expected_seen):
    controller = object.__new__(TextGenerationController)
    controller._async_logprob_requests_seen = False

    controller.note_request_sampling_params(sampling_params)

    assert controller._async_logprob_requests_seen is expected_seen


@pytest.mark.internal
def test_async_diagnostics_report_pending_forward_disable_counts_and_ep_protocol():
    controller = object.__new__(TextGenerationController)
    controller._async_scheduling_enabled = True
    _install_pending_transaction(
        controller, _make_async_layout_snapshot([1], cuda_graph_request_count=1)
    )
    controller._async_step_barrier_reason = "logging step"
    controller._async_eligibility_check_count = 4
    controller._async_eligibility_pass_count = 2
    controller._async_disable_reason_counts = {"not decode-only": 1, "logging step": 1}
    controller._async_disable_reason = "logging step"
    controller._async_forward_launch_count = 3
    controller._async_prepared_forward_count = 4
    controller._async_launched_forward_count = 3
    controller._async_reused_forward_count = 2
    controller._async_committed_forward_count = 2
    controller._async_rolled_back_forward_count = 1
    controller._async_discarded_forward_count = 1
    controller._async_row_mapped_forward_count = 1
    controller._async_identity_forward_count = 1
    controller._async_graph_mismatch_discard_count = 1
    controller._async_layout_mismatch_discard_count = 0
    controller._async_row_map_policy = AsyncRowMapPolicy.IDENTITY_ONLY
    controller._ep_async_protocol = SimpleNamespace(
        diagnostics=lambda: {"step_begin_reuses": 1, "handoff_launches": 2}
    )

    diagnostics = controller.get_async_scheduling_diagnostics()

    assert diagnostics == {
        "enabled": True,
        "pending_forward": True,
        "row_map_policy": "identity_only",
        "step_barrier_reason": "logging step",
        "eligibility_checks": 4,
        "eligibility_passes": 2,
        "disable_reason_counts": {"not decode-only": 1, "logging step": 1},
        "last_disable_reason": "logging step",
        "forward_launches": 3,
        "prepared_forwards": 4,
        "launched_forwards": 3,
        "reused_forwards": 2,
        "committed_forwards": 2,
        "rolled_back_forwards": 1,
        "discarded_forwards": 1,
        "row_mapped_forwards": 1,
        "identity_reused_forwards": 1,
        "graph_mismatch_discards": 1,
        "layout_mismatch_discards": 0,
        "ep_protocol": {"step_begin_reuses": 1, "handoff_launches": 2},
    }


@pytest.mark.internal
def test_async_row_map_policy_setter_switches_exact_parity_mode():
    controller = object.__new__(TextGenerationController)
    controller._async_row_map_policy = AsyncRowMapPolicy.REUSE

    resolved = controller.set_async_row_map_policy("identity_only")

    assert resolved == AsyncRowMapPolicy.IDENTITY_ONLY
    assert controller._async_row_map_policy == AsyncRowMapPolicy.IDENTITY_ONLY
    with pytest.raises(ValueError, match="Unknown async row-map policy"):
        controller.set_async_row_map_policy("invalid")


@pytest.mark.internal
def test_async_transaction_and_resource_diagnostics_are_stable():
    snapshot = _make_async_layout_snapshot([10, 11], cuda_graph_request_count=2)
    ledger = AsyncResourceLedger(in_flight=True)
    ledger.record_reservations(
        request_ids=torch.tensor([10], dtype=torch.int32),
        block_ids=torch.tensor([100], dtype=torch.int32),
        block_columns=torch.tensor([1], dtype=torch.int32),
    )
    ledger.defer_kv_blocks(torch.tensor([200], dtype=torch.int32))
    transaction = AsyncStepTransaction(
        step_id=7,
        state=AsyncTxnState.LAUNCHED,
        snapshot=snapshot,
        resources=ledger,
        row_map=torch.tensor([1, 0], dtype=torch.long),
    )

    diagnostics = transaction.diagnostics()
    assert {
        key: diagnostics[key]
        for key in (
            "step_id",
            "state",
            "request_ids",
            "row_map",
            "discard_reason",
            "has_sample_ticket",
            "has_resources",
            "has_h2d_done_event",
            "has_forward_done_event",
            "has_ep_decision",
            "participants",
        )
    } == {
        "step_id": 7,
        "state": "launched",
        "request_ids": [10, 11],
        "row_map": [1, 0],
        "discard_reason": None,
        "has_sample_ticket": False,
        "has_resources": True,
        "has_h2d_done_event": False,
        "has_forward_done_event": False,
        "has_ep_decision": False,
        "participants": {},
    }
    assert diagnostics["plan"]["request_ids"] == [10, 11]
    assert diagnostics["plan"]["active_request_count"] == 2
    assert ledger.diagnostics() == {
        "in_flight": True,
        "reservations": 1,
        "deferred_kv_blocks": 1,
        "deferred_mamba_slots": 0,
        "mamba_leases": 0,
        "consumed_reservations": 0,
    }


@pytest.mark.internal
def test_async_decode_plan_and_transaction_participant_hooks_are_canonical():
    events = []

    class _Participant:
        def diagnostics(self):
            return {"events": list(events)}

        def prepare(self, plan):
            events.append(("prepare", plan.request_ids.tolist()))
            return "prepared"

        def validate(self, plan, current_state):
            events.append(("validate", plan.active_request_count, current_state))
            return True

        def commit(self, plan):
            events.append(("commit", plan.active_request_count))

        def rollback(self, plan):
            events.append(("rollback", plan.active_request_count))

    participant = _Participant()
    snapshot = _make_async_layout_snapshot([21, 22], cuda_graph_request_count=2)
    plan = AsyncDecodePlan.from_snapshot(snapshot)
    transaction = AsyncDecodeTransaction(
        step_id=3,
        state=AsyncTxnState.PLANNED,
        snapshot=snapshot,
        plan=plan,
        participants=(participant,),
    )

    assert isinstance(participant, AsyncTransactionParticipant)
    assert AsyncStepTransaction is AsyncDecodeTransaction
    assert plan.graph_shape.padded_active_request_count == 2
    assert plan.diagnostics()["request_ids"] == [21, 22]

    transaction.prepare_participants()
    assert transaction.participant_state == {"_Participant": "prepared"}
    assert transaction.validate_participants("current")
    transaction.mark_committed()

    rollback_transaction = AsyncDecodeTransaction(
        step_id=4,
        state=AsyncTxnState.PREPARED,
        snapshot=snapshot,
        plan=plan,
        participants=(participant,),
    )
    rollback_transaction.rollback("validation failed")

    assert events == [
        ("prepare", [21, 22]),
        ("validate", 2, "current"),
        ("commit", 2),
        ("rollback", 2),
    ]
    assert transaction.state == AsyncTxnState.COMMITTED
    assert rollback_transaction.state == AsyncTxnState.ROLLED_BACK
    assert rollback_transaction.discard_reason == "validation failed"


@pytest.mark.internal
def test_async_transaction_terminal_lifecycle_fences_rollback_not_retire():
    events = []

    class _Event:
        def synchronize(self):
            events.append("forward_sync")

    class _Participant:
        def prepare(self, plan):
            events.append(("prepare", plan.active_request_count))
            return "prepared"

        def validate(self, plan, current_state):
            return True

        def commit(self, plan):
            events.append(("commit", plan.active_request_count))

        def rollback(self, plan):
            events.append(("rollback", plan.active_request_count))

        def retire(self, plan):
            events.append(("retire", plan.active_request_count))

        def diagnostics(self):
            return {}

    snapshot = _make_async_layout_snapshot([31, 32], cuda_graph_request_count=2)
    participant = _Participant()
    transaction = AsyncDecodeTransaction(
        step_id=9,
        state=AsyncTxnState.LAUNCHED,
        snapshot=snapshot,
        forward_done_event=_Event(),
        participants=(participant,),
    )

    transaction.prepare_participants()
    transaction.prepare_participants()
    transaction.rollback("invalidated")
    transaction.rollback("late rollback")
    transaction.discard("late discard")
    transaction.mark_committed()

    assert events == [
        ("prepare", 2),
        "forward_sync",
        ("rollback", 2),
    ]
    assert transaction.state == AsyncTxnState.ROLLED_BACK
    assert transaction.discard_reason == "invalidated"
    assert transaction.diagnostics()["participants_prepared"]
    assert transaction.diagnostics()["participants_rolled_back"]
    assert not transaction.diagnostics()["participants_committed"]

    committed = AsyncDecodeTransaction(
        step_id=10,
        state=AsyncTxnState.LAUNCHED,
        snapshot=snapshot,
        forward_done_event=_Event(),
        participants=(participant,),
    )
    committed.mark_committed()
    committed.mark_committed()
    committed.rollback("late rollback")
    committed.discard("late discard")
    committed.mark_retired()
    committed.mark_retired()

    assert events == [
        ("prepare", 2),
        "forward_sync",
        ("rollback", 2),
        ("commit", 2),
        ("retire", 2),
    ]
    assert committed.state == AsyncTxnState.RETIRED
    assert committed.discard_reason is None
    assert committed.diagnostics()["participants_committed"]
    assert committed.diagnostics()["participants_retired"]


@pytest.mark.internal
def test_async_transaction_prepares_late_participants_once():
    events = []

    class _FirstParticipant:
        def prepare(self, plan):
            events.append(("prepare_first", plan.active_request_count))
            return "first"

        def validate(self, plan, current_state):
            return True

        def commit(self, plan):
            events.append(("commit_first", plan.active_request_count))

        def rollback(self, plan):
            events.append(("rollback_first", plan.active_request_count))

        def diagnostics(self):
            return {}

    class _SecondParticipant:
        def prepare(self, plan):
            events.append(("prepare_second", plan.active_request_count))
            return "second"

        def validate(self, plan, current_state):
            return True

        def commit(self, plan):
            events.append(("commit_second", plan.active_request_count))

        def rollback(self, plan):
            events.append(("rollback_second", plan.active_request_count))

        def diagnostics(self):
            return {}

    snapshot = _make_async_layout_snapshot([41, 42], cuda_graph_request_count=2)
    first = _FirstParticipant()
    second = _SecondParticipant()
    transaction = AsyncDecodeTransaction(
        step_id=11,
        state=AsyncTxnState.PREPARED,
        snapshot=snapshot,
        participants=(first,),
    )

    transaction.prepare_participants()
    transaction.add_participants(second)
    transaction.prepare_participants()
    transaction.mark_committed()

    assert events == [
        ("prepare_first", 2),
        ("prepare_second", 2),
        ("commit_first", 2),
        ("commit_second", 2),
    ]
    assert transaction.has_participant(_FirstParticipant)
    assert transaction.has_participant(_SecondParticipant)
    assert transaction.participant_state == {
        "_FirstParticipant": "first",
        "_SecondParticipant": "second",
    }
    with pytest.raises(RuntimeError, match="after transaction finalization"):
        transaction.add_participants(_SecondParticipant())


@pytest.mark.internal
def test_async_resource_participant_rolls_back_speculative_resources_once():
    released_blocks = []
    context = SimpleNamespace(
        is_hybrid_model=False,
        async_kv_deferred_release_count=0,
        kv_block_allocator=SimpleNamespace(
            release_memory_blocks=lambda blocks: released_blocks.append(blocks.clone())
        ),
    )
    ledger = AsyncResourceLedger(in_flight=True)
    ledger.record_reservations(
        request_ids=torch.tensor([10], dtype=torch.int32),
        block_ids=torch.tensor([100], dtype=torch.int32),
        block_columns=torch.tensor([1], dtype=torch.int32),
    )
    ledger.defer_kv_blocks(torch.tensor([200], dtype=torch.int32))
    participant = AsyncResourceParticipant(ledger, context)
    snapshot = _make_async_layout_snapshot([10], cuda_graph_request_count=1)
    transaction = AsyncDecodeTransaction(
        step_id=5,
        state=AsyncTxnState.PREPARED,
        snapshot=snapshot,
        participants=(participant,),
    )

    transaction.prepare_participants()
    transaction.rollback("invalidated")
    transaction.rollback("invalidated again")

    assert [blocks.tolist() for blocks in released_blocks] == [[200, 100]]
    assert context.async_kv_deferred_release_count == 2
    assert participant.diagnostics()["rolled_back"]
    assert not participant.diagnostics()["committed"]
    assert ledger.diagnostics() == {
        "in_flight": False,
        "reservations": 0,
        "deferred_kv_blocks": 0,
        "deferred_mamba_slots": 0,
        "mamba_leases": 0,
        "consumed_reservations": 0,
    }


@pytest.mark.internal
def test_async_mamba_participant_commits_candidate_bank_once():
    accepted_request_ids = []
    context = SimpleNamespace(
        is_hybrid_model=True,
        accept_async_mamba_state=lambda request_ids: accepted_request_ids.append(
            request_ids.clone()
        ),
    )
    snapshot = _make_async_layout_snapshot([21, 22], cuda_graph_request_count=2)
    plan = AsyncDecodePlan.from_snapshot(snapshot)
    participant = AsyncMambaStateParticipant(context)
    transaction = AsyncDecodeTransaction(
        step_id=6,
        state=AsyncTxnState.RESOLVED,
        snapshot=snapshot,
        plan=plan,
        participants=(participant,),
    )

    transaction.mark_committed()
    transaction.mark_committed()
    transaction.rollback("late rollback")

    assert [request_ids.tolist() for request_ids in accepted_request_ids] == [[21, 22]]
    assert participant.diagnostics() == {"committed": True, "rolled_back": False}
    assert transaction.state == AsyncTxnState.COMMITTED


@pytest.mark.internal
def test_async_mamba_participant_commits_row_mapped_current_request_ids():
    accepted_request_ids = []
    context = SimpleNamespace(
        is_hybrid_model=True,
        accept_async_mamba_state=lambda request_ids: accepted_request_ids.append(
            request_ids.clone()
        ),
    )
    pending_snapshot = _make_async_layout_snapshot([21, 22, 23], cuda_graph_request_count=3)
    current_snapshot = _make_async_layout_snapshot([23, 21], cuda_graph_request_count=3)
    decision = resolve_async_pending_forward(
        pending_snapshot,
        current_snapshot,
        row_map_policy=AsyncRowMapPolicy.REUSE,
    )
    plan = AsyncDecodePlan.from_snapshot(pending_snapshot).with_pending_forward_decision(
        decision
    )
    participant = AsyncMambaStateParticipant(context)
    transaction = AsyncDecodeTransaction(
        step_id=7,
        state=AsyncTxnState.RESOLVED,
        snapshot=pending_snapshot,
        plan=plan,
        participants=(participant,),
    )

    transaction.mark_committed()

    assert decision.reusable
    assert decision.row_map.tolist() == [2, 0]
    assert plan.current_request_ids().tolist() == [23, 21]
    assert [request_ids.tolist() for request_ids in accepted_request_ids] == [[23, 21]]


@pytest.mark.internal
def test_controller_attaches_speculative_resource_participants_to_launch_transaction():
    accepted_request_ids = []
    released_blocks = []
    context = SimpleNamespace(
        is_hybrid_model=True,
        accept_async_mamba_state=lambda request_ids: accepted_request_ids.append(
            request_ids.clone()
        ),
        async_kv_deferred_release_count=0,
        kv_block_allocator=SimpleNamespace(
            release_memory_blocks=lambda blocks: released_blocks.append(blocks.clone())
        ),
    )
    controller = object.__new__(TextGenerationController)
    controller.inference_wrapped_model = SimpleNamespace(inference_context=context)
    controller.num_speculative_tokens = 2
    controller._async_logprob_requests_seen = False
    snapshot = _make_async_layout_snapshot([31, 32], cuda_graph_request_count=2)
    transaction = AsyncDecodeTransaction(
        step_id=7,
        state=AsyncTxnState.PREPARED,
        snapshot=snapshot,
        plan=AsyncDecodePlan.from_snapshot(snapshot),
    )
    ledger = AsyncResourceLedger(in_flight=True)
    ledger.defer_kv_blocks(torch.tensor([300], dtype=torch.int32))
    sample_ticket = _sample_ticket([4, 5], mtp_tokens=[[6, 7], [8, 9]])

    controller._attach_async_transaction_participants(
        transaction, resources=ledger, sample_ticket=sample_ticket
    )

    assert tuple(type(participant) for participant in transaction.participants) == (
        AsyncMambaStateParticipant,
        AsyncSampleReadbackParticipant,
        AsyncLogprobMTPParticipant,
        AsyncResourceParticipant,
    )
    assert transaction.participant_state.keys() == {
        "AsyncMambaStateParticipant",
        "AsyncSampleReadbackParticipant",
        "AsyncLogprobMTPParticipant",
        "AsyncResourceParticipant",
    }

    transaction.mark_committed()

    assert [request_ids.tolist() for request_ids in accepted_request_ids] == [[31, 32]]
    assert [blocks.tolist() for blocks in released_blocks] == [[300]]
    assert context.async_kv_deferred_release_count == 1
    assert not ledger.in_flight


@pytest.mark.internal
@pytest.mark.asyncio
async def test_async_h2d_and_forward_launch_before_cpu_bookkeeping_drains(monkeypatch):
    monkeypatch.setattr(tgc_module, "range_push", lambda _msg: None)
    monkeypatch.setattr(tgc_module, "range_pop", lambda: None)
    events = []

    class _Event:
        def __init__(self, name):
            self.name = name

        def synchronize(self):
            events.append(f"{self.name}_sync")

    context = SimpleNamespace(
        active_token_count=2,
        total_request_count=2,
        paused_request_count=0,
        num_decode_requests=2,
        padded_active_request_count=2,
        is_hybrid_model=False,
        config=SimpleNamespace(materialize_only_last_token_logits=True),
        kv_block_allocator=SimpleNamespace(
            store_routing_per_block=lambda _routing: events.append("routing")
        ),
        publish_async_prepared_decode_plan=lambda: events.append("publish"),
        transfer_bookkeeping_to_gpu=lambda **_kwargs: events.append("h2d")
        or _Event("h2d"),
        current_input_and_position_ids=lambda: (
            torch.tensor([[1, 2]], dtype=torch.int64),
            torch.tensor([[0, 1]], dtype=torch.int64),
        ),
        using_cuda_graph_this_step=lambda: True,
        mark_async_resources_in_flight=lambda: events.append("mark_in_flight") or "ledger",
    )
    controller = object.__new__(TextGenerationController)
    controller.inference_wrapped_model = SimpleNamespace(
        inference_context=context,
        model=SimpleNamespace(config=SimpleNamespace(moe_enable_routing_replay=False)),
    )
    controller.num_speculative_tokens = 0
    controller._async_step_transaction = None
    controller._async_prepare_deferred_until_after_sampling = False
    controller._async_forward_launch_count = 0
    controller._decide_ep_step_begin = lambda **_kwargs: EPStepBeginDecision(
        step_id=0,
        has_real_work=True,
        reuse_pending_forward=False,
        discard_pending_forward=False,
        row_mapped_forward=False,
    )
    controller._dynamic_step_context_init = lambda: (
        torch.tensor([[1, 2]], dtype=torch.int64),
        torch.tensor([[0, 1]], dtype=torch.int64),
    )

    def _forward(*_args):
        events.append("async_forward" if "h2d" in events else "forward")

    controller._dynamic_step_forward_logits = _forward
    controller._router_record_bookkeeping = lambda: None
    controller._async_scheduling_disabled_reason = lambda **_kwargs: None
    controller._record_async_eligibility_result = lambda _reason: None
    controller._record_async_disable_reason = lambda reason: events.append(("disable", reason))
    controller._try_prepare_async_decode_before_sampling = lambda: events.append(
        "prepare"
    ) or True
    controller._should_collect_dynamic_sampling_bookkeeping = lambda **_kwargs: False
    controller._dynamic_step_sample_logits_to_next_input_ids = (
        lambda: events.extend(["sample", "copy"])
    )
    controller._transfer_async_samples_to_cpu = lambda count: events.append(
        ("d2h", count)
    ) or _sample_ticket([4, 5])
    controller._confirm_prepared_ep_async_handoff = lambda: True
    controller._begin_async_step_transaction = lambda _count: events.append(
        "record"
    ) or SimpleNamespace(mark_launched=lambda **_kwargs: events.append("tx_launch"))
    controller._ensure_ep_async_handoff_decided = lambda **_kwargs: events.append("ep_done")

    def _bookkeeping(**kwargs):
        events.append("bookkeeping_start")
        kwargs["sample_ready_event"].synchronize()
        kwargs["h2d_done_event"].synchronize()
        events.append("bookkeeping_done")
        return {"sample": torch.tensor([4, 5], dtype=torch.int64)}

    controller._dynamic_step_context_bookkeeping = _bookkeeping

    result = await controller.async_generate_output_tokens_dynamic_batch(skip_bookkeeping=False)

    assert result["sample"].tolist() == [4, 5]
    assert events.index("h2d") < events.index("async_forward") < events.index(
        "bookkeeping_start"
    )
    assert events.index("async_forward") < events.index("h2d_sync")
    assert "h2d_sync" not in events[: events.index("async_forward")]


@pytest.mark.internal
@pytest.mark.asyncio
async def test_async_prepare_and_launch_do_not_move_after_cpu_bookkeeping(monkeypatch):
    monkeypatch.setattr(tgc_module, "range_push", lambda _msg: None)
    monkeypatch.setattr(tgc_module, "range_pop", lambda: None)
    events = []

    class _Event:
        def __init__(self, name):
            self.name = name

        def synchronize(self):
            events.append(f"{self.name}_sync")

    context = SimpleNamespace(
        active_token_count=2,
        total_request_count=2,
        paused_request_count=0,
        num_decode_requests=2,
        padded_active_request_count=2,
        is_hybrid_model=False,
        config=SimpleNamespace(materialize_only_last_token_logits=True),
        kv_block_allocator=SimpleNamespace(
            store_routing_per_block=lambda _routing: events.append("routing")
        ),
        publish_async_prepared_decode_plan=lambda: events.append("publish"),
        transfer_bookkeeping_to_gpu=lambda **_kwargs: events.append("h2d")
        or _Event("h2d"),
        current_input_and_position_ids=lambda: (
            torch.tensor([[1, 2]], dtype=torch.int64),
            torch.tensor([[0, 1]], dtype=torch.int64),
        ),
        using_cuda_graph_this_step=lambda: True,
        mark_async_resources_in_flight=lambda: events.append("mark_in_flight") or "ledger",
    )

    def _prepare(**kwargs):
        events.append("prepare_pre" if kwargs.get("pre_sampling") else "prepare_after")
        return True

    context.prepare_async_decode_next_step = _prepare
    controller = object.__new__(TextGenerationController)
    controller.inference_wrapped_model = SimpleNamespace(
        inference_context=context,
        model=SimpleNamespace(config=SimpleNamespace(moe_enable_routing_replay=False)),
    )
    controller.num_speculative_tokens = 0
    controller._async_step_transaction = None
    controller._async_prepare_deferred_until_after_sampling = False
    controller._async_forward_launch_count = 0
    controller._decide_ep_step_begin = lambda **_kwargs: EPStepBeginDecision(
        step_id=0,
        has_real_work=True,
        reuse_pending_forward=False,
        discard_pending_forward=False,
        row_mapped_forward=False,
    )
    controller._dynamic_step_context_init = lambda: (
        torch.tensor([[1, 2]], dtype=torch.int64),
        torch.tensor([[0, 1]], dtype=torch.int64),
    )

    def _forward(*_args):
        events.append("async_forward" if "h2d" in events else "forward")

    controller._dynamic_step_forward_logits = _forward
    controller._router_record_bookkeeping = lambda: None
    controller._async_scheduling_disabled_reason = lambda **_kwargs: None
    controller._record_async_eligibility_result = lambda _reason: None
    controller._record_async_disable_reason = lambda reason: events.append(("disable", reason))
    controller._decide_ep_async_handoff = lambda **_kwargs: EPAsyncHandoffDecision(
        step_id=0,
        has_real_work=True,
        launch_async_forward=True,
        skip_async_forward=False,
        any_launch_request=True,
        any_skip_request=False,
    )
    controller._should_collect_dynamic_sampling_bookkeeping = lambda **_kwargs: False
    controller._dynamic_step_sample_logits_to_next_input_ids = (
        lambda: events.extend(["sample", "copy"])
    )
    controller._transfer_async_samples_to_cpu = lambda count: events.append(
        ("d2h", count)
    ) or _sample_ticket([4, 5])
    controller._begin_async_step_transaction = lambda _count: events.append(
        "record"
    ) or SimpleNamespace(mark_launched=lambda **_kwargs: events.append("tx_launch"))
    controller._attach_async_transaction_participants = (
        lambda transaction, **kwargs: events.append(("attach", kwargs))
    )
    controller._ensure_ep_async_handoff_decided = lambda **_kwargs: events.append("ep_done")

    def _bookkeeping(**kwargs):
        events.append("bookkeeping_start")
        kwargs["sample_ready_event"].synchronize()
        kwargs["h2d_done_event"].synchronize()
        events.append("bookkeeping_done")
        return {"sample": torch.tensor([4, 5], dtype=torch.int64)}

    controller._dynamic_step_context_bookkeeping = _bookkeeping

    result = await controller.async_generate_output_tokens_dynamic_batch(skip_bookkeeping=False)

    assert result["sample"].tolist() == [4, 5]
    ordering = [
        event
        for event in events
        if event
        in (
            "prepare_pre",
            "prepare_after",
            "sample",
            "copy",
            "publish",
            "h2d",
            "async_forward",
            "bookkeeping_start",
        )
    ]
    assert ordering == [
        "prepare_pre",
        "sample",
        "copy",
        "publish",
        "h2d",
        "async_forward",
        "bookkeeping_start",
    ]
    assert "prepare_after" not in events


@pytest.mark.internal
def test_activate_async_sample_slot_uses_readback_owned_buffers_and_events():
    controller = object.__new__(TextGenerationController)
    controller.num_speculative_tokens = 2
    controller._async_sample_readback = AsyncSampleReadback(
        sample_slot_count=2,
        current_sample_slot=0,
        sampled_tokens_cuda_slots=["cuda_tokens_0", "cuda_tokens_1"],
        sample_values_cuda_slots=["cuda_values_0", "cuda_values_1"],
        sampled_tokens_cpu_slots=["cpu_tokens_0", "cpu_tokens_1"],
        source_ready_events=("source_ready_0", "source_ready_1"),
        copy_done_events=("copy_ready_0", "copy_ready_1"),
        copy_stream="copy_stream",
        sampled_mtp_tokens_cuda_slots=["cuda_mtp_0", "cuda_mtp_1"],
        sampled_mtp_tokens_cpu_slots=["cpu_mtp_0", "cpu_mtp_1"],
    )

    controller._activate_async_sample_slot(1)

    assert controller._async_sample_readback.current_sample_slot == 1
    assert controller._sampled_tokens_cuda == "cuda_tokens_1"
    assert controller._async_sample_values_cuda == "cuda_values_1"
    assert controller._async_sampled_tokens_cpu == "cpu_tokens_1"
    assert controller._async_sample_source_ready_event == "source_ready_1"
    assert controller._async_sample_ready_event == "copy_ready_1"
    assert controller._sampled_mtp_tokens_cuda == "cuda_mtp_1"
    assert controller._async_sampled_mtp_tokens_cpu == "cpu_mtp_1"


class _FakeBookkeepingContext:
    def __init__(self, request_ids, sequence_lengths, max_sequence_lengths, termination_ids):
        self.request_ids = torch.tensor(request_ids, dtype=torch.int64)
        self.paused_request_count = 0
        self.total_request_count = len(request_ids)
        self.active_request_metadata = {
            "termination_id": torch.tensor(termination_ids, dtype=torch.int64)
        }
        self.sequence_lengths = torch.tensor(sequence_lengths, dtype=torch.int64)
        self.max_sequence_lengths = torch.tensor(max_sequence_lengths, dtype=torch.int64)
        self.request_to_kv_block_ids = torch.tensor(
            [[5, -1, -1], [6, 7, -1], [8, -1, -1]], dtype=torch.int32
        )[: len(request_ids)]
        self.kv_block_allocator = SimpleNamespace(block_routing=True)
        self.update_calls = []

    def get_active_sequence_lengths(self):
        return self.sequence_lengths.clone()

    def get_max_sequence_lengths(self):
        return self.max_sequence_lengths.clone()

    def update_requests(self, active_request_mask, new_tokens, new_speculative_tokens):
        self.update_calls.append(
            (
                active_request_mask.clone(),
                new_tokens.clone(),
                None if new_speculative_tokens is None else new_speculative_tokens.clone(),
            )
        )
        return {
            "newly_paused_request_ids": torch.tensor([91], dtype=torch.int64),
            "evict_request_ids": torch.tensor([92], dtype=torch.int64),
        }


class _SyncEvent:
    def __init__(self):
        self.sync_count = 0

    def synchronize(self):
        self.sync_count += 1


@pytest.mark.internal
@pytest.mark.parametrize(
    (
        "num_speculative_tokens",
        "sampled_tokens",
        "accepted_tokens",
        "termination_ids",
        "stop_ids",
        "sequence_lengths",
        "max_sequence_lengths",
        "expected_finished",
        "expected_finish_counter",
    ),
    [
        (0, [1, 2, 3], None, [-1, 42, -1], {12}, [5, 6, 7], [10, 7, 10], [11, 12], "base"),
        (
            2,
            [1, 2, 3],
            [[99, -1], [-1, -1], [-1, -1]],
            [99, -1, -1],
            set(),
            [5, 5, 5],
            [9, 9, 9],
            [10],
            "mtp",
        ),
    ],
)
def test_dynamic_bookkeeping_marks_lifecycle_boundaries(
    monkeypatch,
    num_speculative_tokens,
    sampled_tokens,
    accepted_tokens,
    termination_ids,
    stop_ids,
    sequence_lengths,
    max_sequence_lengths,
    expected_finished,
    expected_finish_counter,
):
    monkeypatch.setattr(tgc_module, "range_push", lambda _msg: None)
    monkeypatch.setattr(tgc_module, "range_pop", lambda: None)
    context = _FakeBookkeepingContext(
        request_ids=[10, 11, 12],
        sequence_lengths=sequence_lengths,
        max_sequence_lengths=max_sequence_lengths,
        termination_ids=termination_ids,
    )
    controller = object.__new__(TextGenerationController)
    controller.inference_wrapped_model = SimpleNamespace(inference_context=context)
    controller.num_speculative_tokens = num_speculative_tokens
    controller._request_sampling_rngs = {10: object(), 11: object(), 12: object()}
    controller._get_stop_word_finished_ids_callback = lambda _ids: stop_ids
    controller._async_finish_boundary_count = 0
    controller._async_mtp_finish_boundary_count = 0
    controller._async_pause_boundary_count = 0
    controller._async_evict_boundary_count = 0
    if accepted_tokens is not None:
        controller._accepted_tokens_per_request = torch.tensor(accepted_tokens, dtype=torch.int64)
    sample_ready_event = _SyncEvent()
    h2d_done_event = _SyncEvent()

    result = controller._dynamic_step_context_bookkeeping(
        sampled_tokens_cpu=torch.tensor(sampled_tokens, dtype=torch.int64),
        sampled_mtp_tokens_cpu=(
            None
            if num_speculative_tokens == 0
            else torch.zeros(num_speculative_tokens, 3, dtype=torch.int64)
        ),
        sample_ready_event=sample_ready_event,
        h2d_done_event=h2d_done_event,
    )

    expected_mask = torch.tensor(
        [request_id not in expected_finished for request_id in [10, 11, 12]], dtype=torch.uint8
    )
    assert torch.equal(context.update_calls[0][0], expected_mask)
    assert result["finished_request_ids"].tolist() == expected_finished
    assert result["finished_routing_block_ids"] == {
        request_id: context.request_to_kv_block_ids[i][
            context.request_to_kv_block_ids[i] >= 0
        ].tolist()
        for i, request_id in enumerate([10, 11, 12])
        if request_id in expected_finished
    }
    assert sample_ready_event.sync_count == 1
    assert h2d_done_event.sync_count == 1
    assert controller._async_pause_boundary_count == 1
    assert controller._async_evict_boundary_count == 1
    assert controller._async_finish_boundary_count == int(expected_finish_counter == "base")
    assert controller._async_mtp_finish_boundary_count == int(expected_finish_counter == "mtp")
    for request_id in expected_finished:
        assert request_id not in controller._request_sampling_rngs


class _ReleaseRecordingAllocator:
    dummy_block_idx = -2

    def __init__(self):
        self.released = []

    def release_memory_blocks(self, blocks):
        self.released.append(blocks.clone())


class _MambaMetadataWithFreeRecording:
    def __init__(self, slots, banks):
        self.request_to_mamba_state_idx = torch.tensor(slots, dtype=torch.int32)
        self.request_to_mamba_state_bank = torch.tensor(banks, dtype=torch.int32)
        self.freed_slots = []
        self.reset_called = False
        self.reset_varlen_count = 0

    def free_slot_ids(self, slots):
        self.freed_slots.append(slots.clone())

    def free_slots(self, _request_indexes):
        raise AssertionError("async cleanup must defer slot free by concrete slot id")

    def reset(self):
        self.reset_called = True

    def reset_varlen_metadata(self):
        self.reset_varlen_count += 1


@pytest.mark.internal
def test_async_pending_resources_are_quarantined_until_forward_retires():
    context = object.__new__(DynamicInferenceContext)
    context.request_to_kv_block_ids = torch.tensor(
        [[10, 11, -1], [12, -1, -1]], dtype=torch.int32
    )
    context.is_hybrid_model = False
    context.mamba_slot_allocator = None
    context.kv_block_allocator = _ReleaseRecordingAllocator()
    context._async_resource_ledger = AsyncResourceLedger(in_flight=True)
    context.async_kv_deferred_release_count = 0
    context.async_mamba_deferred_release_count = 0

    context.release_memory_blocks_from_request_indexes(torch.tensor([0], dtype=torch.long))

    assert context.kv_block_allocator.released == []
    assert context._async_resource_ledger.deferred_kv_tensor().tolist() == [10, 11]
    assert context.request_to_kv_block_ids.tolist() == [[-1, -1, -1], [12, -1, -1]]

    context.release_deferred_async_resources()

    assert [blocks.tolist() for blocks in context.kv_block_allocator.released] == [[10, 11]]
    assert not context._async_resource_ledger.in_flight
    assert context._async_resource_ledger.deferred_kv_tensor().numel() == 0
    assert context.async_kv_deferred_release_count == 2


@pytest.mark.internal
def test_async_transaction_defers_mamba_slot_free_until_forward_retires():
    context = object.__new__(DynamicInferenceContext)
    context.request_to_kv_block_ids = torch.tensor(
        [[10, 11, -1], [12, -1, -1]], dtype=torch.int32
    )
    context.is_hybrid_model = True
    context.mamba_metadata = _MambaMetadataWithFreeRecording(slots=[3, 5], banks=[0, 1])
    context.mamba_slot_allocator = None
    context.kv_block_allocator = _ReleaseRecordingAllocator()
    context._async_resource_ledger = AsyncResourceLedger(in_flight=True)
    context.async_kv_deferred_release_count = 0
    context.async_mamba_deferred_release_count = 0

    context.release_memory_blocks_from_request_indexes(torch.tensor([0], dtype=torch.long))

    assert context.kv_block_allocator.released == []
    assert context.mamba_metadata.freed_slots == []
    assert context._async_resource_ledger.deferred_kv_tensor().tolist() == [10, 11]
    assert context._async_resource_ledger.deferred_mamba_tensor().tolist() == [3]
    assert context.mamba_metadata.request_to_mamba_state_idx.tolist() == [-1, 5]
    assert context.mamba_metadata.request_to_mamba_state_bank.tolist() == [0, 1]

    context.release_deferred_async_resources()

    assert [blocks.tolist() for blocks in context.kv_block_allocator.released] == [[10, 11]]
    assert [slots.tolist() for slots in context.mamba_metadata.freed_slots] == [[3]]
    assert context._async_resource_ledger.deferred_mamba_tensor().numel() == 0
    assert context.async_mamba_deferred_release_count == 1
    assert not context._async_resource_ledger.in_flight


@pytest.mark.internal
def test_async_mamba_reset_defers_allocated_slots_while_forward_is_in_flight():
    context = object.__new__(DynamicInferenceContext)
    context.is_hybrid_model = True
    context.mamba_metadata = _MambaMetadataWithFreeRecording(slots=[2, -1, 4], banks=[0, 0, 1])
    context._async_resource_ledger = AsyncResourceLedger(in_flight=True)
    context.async_mamba_deferred_release_count = 0

    context.reset_mamba_state()

    assert not context.mamba_metadata.reset_called
    assert context.mamba_metadata.reset_varlen_count == 1
    assert context._async_resource_ledger.deferred_mamba_tensor().tolist() == [2, 4]
    assert context.mamba_metadata.request_to_mamba_state_idx.tolist() == [-1, -1, -1]
    assert context.mamba_metadata.request_to_mamba_state_bank.tolist() == [0, 0, 0]

    context.release_deferred_async_resources()

    assert [slots.tolist() for slots in context.mamba_metadata.freed_slots] == [[2, 4]]
    assert context._async_resource_ledger.deferred_mamba_tensor().numel() == 0
    assert context.async_mamba_deferred_release_count == 2


@pytest.mark.internal
def test_async_kv_reservations_are_adopted_or_deferred_then_released():
    context = object.__new__(DynamicInferenceContext)
    context.paused_request_count = 0
    context.total_request_count = 3
    context.request_ids = torch.tensor([10, 11, 12], dtype=torch.int32)
    context.request_kv_block_counts = torch.tensor([1, 1, 1], dtype=torch.int32)
    context.request_to_kv_block_ids = torch.tensor(
        [[1, -1, -1], [2, -1, -1], [3, -1, -1]], dtype=torch.int32
    )
    context.request_last_kv_block_id = torch.tensor([1, 2, 3], dtype=torch.int32)
    context._async_resource_ledger = AsyncResourceLedger()
    context._async_resource_ledger.record_reservations(
        request_ids=torch.tensor([10, 11, 99], dtype=torch.int32),
        block_ids=torch.tensor([100, 101, 102], dtype=torch.int32),
        block_columns=torch.tensor([1, 1, 1], dtype=torch.int32),
    )
    context.async_kv_reservation_adoption_count = 0
    context.async_kv_deferred_release_count = 0
    context.async_mamba_deferred_release_count = 0
    context.is_hybrid_model = False
    context.kv_block_allocator = _ReleaseRecordingAllocator()

    consumed = context.consume_async_kv_reservations(
        torch.tensor([10], dtype=torch.int32), torch.tensor([1], dtype=torch.int32)
    )

    assert consumed.tolist() == [100]
    assert context.request_to_kv_block_ids.tolist() == [[1, -1, -1], [2, -1, -1], [3, -1, -1]]
    assert context.request_kv_block_counts.tolist() == [1, 1, 1]
    assert context.request_last_kv_block_id.tolist() == [1, 2, 3]
    assert context._async_resource_ledger.reservation_count == 2
    assert context._async_resource_ledger.reserved_block_ids_tensor().tolist() == [101, 102]
    assert context.async_kv_reservation_adoption_count == 1

    context.defer_unused_async_kv_reservations()

    assert context._async_resource_ledger.reservation_count == 0
    assert context._async_resource_ledger.deferred_kv_tensor().tolist() == [101, 102]

    context.release_deferred_async_resources()

    assert [blocks.tolist() for blocks in context.kv_block_allocator.released] == [[101, 102]]
    assert context._async_resource_ledger.deferred_kv_tensor().numel() == 0
    assert context.async_kv_deferred_release_count == 2


@pytest.mark.internal
def test_speculative_top_n_logprobs_cover_decode_and_prefill_rows():
    controller = object.__new__(TextGenerationController)
    controller.num_speculative_tokens = 2
    controller._accepted_token_counts_per_request = torch.tensor([2, 0], dtype=torch.int64)
    context = SimpleNamespace(
        total_request_count=3,
        paused_request_count=0,
        config=SimpleNamespace(materialize_only_last_token_logits=False),
        gpu_view=SimpleNamespace(
            request_in_prefill_status=torch.tensor([0, 0, 1], dtype=torch.int32),
            request_query_lengths=torch.tensor([3, 3, 2], dtype=torch.int32),
        ),
        active_request_metadata={
            "top_n_logprobs": torch.tensor([2, 1, 2], dtype=torch.int32),
            "skip_prompt_log_probs": torch.tensor([False, False, True], dtype=torch.bool),
        },
    )
    controller.inference_wrapped_model = SimpleNamespace(inference_context=context)
    log_probs = torch.arange(8 * 6, dtype=torch.float32).view(8, 6)

    result = controller._dynamic_step_calculate_top_n_logprobs_speculative(log_probs)

    assert sorted(result) == [0, 1, 2]
    assert [len(result[i]) for i in [0, 1, 2]] == [3, 1, 1]
    assert result[0][0][1].tolist() == [5, 4]
    assert result[1][0][1].tolist() == [5]
    assert result[2][0][1].tolist() == [5, 4]


@pytest.mark.internal
@pytest.mark.parametrize(
    ("top_n_logprobs", "only_last", "expected_lengths"),
    [([0, 0, 0], False, None), ([0, 2, 1], True, {1: 1, 2: 1}), ([0, 2, 1], False, {1: 2, 2: 2})],
)
def test_speculative_top_n_logprobs_zero_and_prefill_materialization_modes(
    top_n_logprobs, only_last, expected_lengths
):
    controller = object.__new__(TextGenerationController)
    controller.num_speculative_tokens = 1
    controller._accepted_token_counts_per_request = torch.tensor([0], dtype=torch.int64)
    context = SimpleNamespace(
        total_request_count=3,
        paused_request_count=0,
        config=SimpleNamespace(materialize_only_last_token_logits=only_last),
        gpu_view=SimpleNamespace(
            request_in_prefill_status=torch.tensor([0, 1, 1], dtype=torch.int32),
            request_query_lengths=torch.tensor([2, 2, 2], dtype=torch.int32),
        ),
        active_request_metadata={
            "top_n_logprobs": torch.tensor(top_n_logprobs, dtype=torch.int32),
            "skip_prompt_log_probs": torch.tensor([False, False, False], dtype=torch.bool),
        },
    )
    controller.inference_wrapped_model = SimpleNamespace(inference_context=context)
    row_count = 4 if only_last else 6
    log_probs = torch.arange(row_count * 5, dtype=torch.float32).view(row_count, 5)

    result = controller._dynamic_step_calculate_top_n_logprobs_speculative(log_probs)

    if expected_lengths is None:
        assert result is None
    else:
        assert {idx: len(values) for idx, values in result.items()} == expected_lengths
        assert result[1][0][1].tolist() == [4, 3]
        assert result[2][0][1].tolist() == [4]


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="MTP sampling test requires CUDA tensors")
@pytest.mark.parametrize("materialize_only_last_token_logits", [False, True])
def test_row_mapped_mtp_sampling_uses_pending_forward_logits(materialize_only_last_token_logits):
    controller = object.__new__(TextGenerationController)
    stride = 3
    vocab_size = 5
    pending_request_count = 3
    active_request_count = 2
    expected_indices = torch.tensor([6, 7, 8, 0, 1, 2], device="cuda")
    source_logits = torch.arange(
        pending_request_count * stride * vocab_size, device="cuda", dtype=torch.float32
    ).view(1, pending_request_count * stride, vocab_size)

    context = SimpleNamespace(
        total_request_count=active_request_count,
        paused_request_count=0,
        num_decode_requests=active_request_count,
        num_prefill_requests=0,
        gpu_view=SimpleNamespace(
            request_in_prefill_status=torch.zeros(
                active_request_count, dtype=torch.int32, device="cuda"
            )
        ),
        config=SimpleNamespace(
            materialize_only_last_token_logits=materialize_only_last_token_logits
        ),
        using_cuda_graph_this_step=lambda: False,
        speculative_required_logit_indices=lambda: torch.arange(
            active_request_count * stride, device="cuda"
        ),
    )
    controller.inference_wrapped_model = SimpleNamespace(inference_context=context)
    controller._sampling_backend = "torch"
    controller._enable_cuda_graph = False
    controller._all_logits_cuda = source_logits
    controller.num_speculative_tokens = stride - 1
    controller.vocab_size = vocab_size
    captures = {}

    def _sample(required_logits, request_in_prefill_status):
        captures["required_logits"] = required_logits.detach().clone()
        captures["request_in_prefill_status"] = request_in_prefill_status.detach().clone()
        return torch.arange(active_request_count * stride, device="cuda"), None

    def _verify(output_tokens, input_tokens_required, *args):
        captures["input_tokens_required"] = input_tokens_required.detach().clone()
        return (
            torch.arange(active_request_count, device="cuda"),
            torch.ones(active_request_count, stride, dtype=torch.bool, device="cuda"),
            input_tokens_required,
        )

    def _prepare(_num_decode_requests, _output_tokens, required_logit_indices, *_args):
        captures["prepared_required_indices"] = required_logit_indices.detach().clone()

    controller._sample_speculative_logits = _sample
    controller._verify_speculative_tokens = _verify
    controller._prepare_speculative_tokens_for_next_forward_pass = _prepare

    controller._dynamic_step_sample_logits_and_verify_tokens(
        input_ids=torch.arange(active_request_count * stride, device="cuda").view(1, -1),
        row_indices=torch.tensor([2, 0], device="cuda"),
    )

    assert torch.equal(captures["required_logits"], source_logits.squeeze(0)[expected_indices])
    assert torch.equal(
        captures["request_in_prefill_status"], torch.zeros(2, dtype=torch.int32, device="cuda")
    )
    assert torch.equal(captures["input_tokens_required"], torch.arange(6, device="cuda"))
    assert torch.equal(captures["prepared_required_indices"], expected_indices)


@pytest.mark.internal
def test_row_mapped_full_logits_sampling_uses_pending_decode_rows():
    controller = object.__new__(TextGenerationController)
    source_logits = torch.arange(4 * 5, dtype=torch.float32).view(1, 4, 5)

    def _fail_last_token_logits(_logits):
        pytest.fail("row-mapped decode reuse should gather pending rows before last_token_logits")

    context = SimpleNamespace(
        total_request_count=2,
        paused_request_count=0,
        padded_active_token_count=4,
        config=SimpleNamespace(materialize_only_last_token_logits=False),
        is_decode_only=lambda: True,
        last_token_logits=_fail_last_token_logits,
    )
    controller.inference_wrapped_model = SimpleNamespace(inference_context=context)
    controller._all_logits_cuda = source_logits

    result = controller._dynamic_step_required_token_logits(row_indices=torch.tensor([2, 0]))

    assert torch.equal(result, source_logits.squeeze(0).index_select(0, torch.tensor([2, 0])))


@pytest.mark.internal
def test_row_mapped_greedy_sampling_writes_tokens_in_current_request_order():
    controller = object.__new__(TextGenerationController)
    source_logits = torch.tensor(
        [[[0.0, 20.0, 0.0, 0.0], [0.0, 0.0, 30.0, 0.0], [0.0, 0.0, 0.0, 40.0]]]
    )
    next_input_ids = torch.empty(2, dtype=torch.long)
    context = SimpleNamespace(
        total_request_count=2,
        paused_request_count=0,
        config=SimpleNamespace(materialize_only_last_token_logits=False),
        is_decode_only=lambda: True,
        copy_async_prepared_decode_input_ids_from_samples=lambda *_args, **_kwargs: False,
        gpu_view=SimpleNamespace(token_to_input_ids=next_input_ids),
        active_request_metadata={
            "top_k": torch.tensor([1, 1], dtype=torch.int32),
            "top_p": torch.tensor([0.0, 0.0], dtype=torch.float32),
        },
    )
    controller.inference_wrapped_model = SimpleNamespace(inference_context=context)
    controller._all_logits_cuda = source_logits
    controller._async_sample_values_cuda = torch.empty(2)
    controller._sampled_tokens_cuda = torch.empty(2, dtype=torch.long)
    controller._sampled_mtp_tokens_cuda = None
    controller.num_speculative_tokens = 0

    controller._dynamic_step_sample_logits_greedy_to_next_input_ids(row_indices=torch.tensor([2, 0]))

    assert controller._sampled_tokens_cuda.tolist() == [3, 1]
    assert next_input_ids.tolist() == [3, 1]


@pytest.mark.internal
def test_decode_full_logits_sampling_uses_current_rows_after_async_prepare_without_row_map():
    controller = object.__new__(TextGenerationController)
    source_logits = torch.arange(4 * 5, dtype=torch.float32).view(1, 4, 5)

    def _fail_last_token_logits(_logits):
        pytest.fail("decode-only full logits should not use mutated last-token metadata")

    context = SimpleNamespace(
        total_request_count=2,
        paused_request_count=0,
        config=SimpleNamespace(materialize_only_last_token_logits=False),
        is_decode_only=lambda: True,
        last_token_logits=_fail_last_token_logits,
    )
    controller.inference_wrapped_model = SimpleNamespace(inference_context=context)
    controller._all_logits_cuda = source_logits

    result = controller._dynamic_step_required_token_logits()

    assert torch.equal(result, source_logits.squeeze(0)[:2])


@pytest.mark.internal
def test_decode_full_logits_sampling_kernel_uses_contiguous_rows_after_async_prepare():
    controller = object.__new__(TextGenerationController)
    source_logits = torch.arange(4 * 5, dtype=torch.float32).view(1, 4, 5)
    captures = {}

    class _Sampling:
        def sample_kernel(
            self, logits, n, context, gather_indices=None, eager=True, cache_key=None
        ):
            captures["logits"] = logits
            captures["n"] = n
            captures["gather_indices"] = gather_indices
            captures["eager"] = eager
            captures["cache_key"] = cache_key
            return torch.tensor([4, 3, 2, 1])

    context = SimpleNamespace(
        total_request_count=2,
        paused_request_count=0,
        padded_active_request_count=4,
        config=SimpleNamespace(materialize_only_last_token_logits=False),
        is_decode_only=lambda: True,
        using_cuda_graph_this_step=lambda: True,
        gpu_view=SimpleNamespace(active_request_last_token_idxs=torch.tensor([3, 2, 1, 0])),
    )
    controller.inference_wrapped_model = SimpleNamespace(inference_context=context)
    controller._all_logits_cuda = source_logits
    controller._sampled_tokens_cuda = torch.empty(2, dtype=torch.long)
    controller._sampling = _Sampling()
    controller._sampling_backend = "flashinfer"
    controller._enable_cuda_graph = True

    controller._dynamic_step_sample_logits()

    assert captures["gather_indices"] is None
    assert captures["n"] == 4
    assert captures["cache_key"] == ("sample", 4)
    assert torch.equal(controller._sampled_tokens_cuda, torch.tensor([4, 3]))


@pytest.mark.internal
def test_row_mapped_full_logits_logprobs_use_gathered_decode_rows_as_last_logits():
    controller = object.__new__(TextGenerationController)
    source_logits = torch.arange(4 * 5, dtype=torch.float32).view(1, 4, 5)
    sampled_tokens = torch.tensor([4, 1], dtype=torch.long)
    captures = {}

    def _calculate_log_probs(logits, new_tokens, only_last_token_logits=False):
        captures["logits"] = logits.detach().clone()
        captures["new_tokens"] = new_tokens.detach().clone()
        captures["only_last_token_logits"] = only_last_token_logits
        return [[-1.0], [-2.0]], logits.squeeze(0)

    context = SimpleNamespace(
        total_request_count=2,
        paused_request_count=0,
        padded_active_token_count=2,
        config=SimpleNamespace(materialize_only_last_token_logits=False),
        is_decode_only=lambda: True,
        calculate_log_probs=_calculate_log_probs,
    )
    controller.inference_wrapped_model = SimpleNamespace(inference_context=context)
    controller._all_logits_cuda = source_logits
    controller._sampled_tokens_cuda = sampled_tokens
    controller.num_speculative_tokens = 0

    log_probs, log_probs_tensor = controller._dynamic_step_calculate_log_probs(
        row_indices=torch.tensor([3, 1])
    )

    expected_logits = source_logits.index_select(1, torch.tensor([3, 1]))
    assert log_probs == [[-1.0], [-2.0]]
    assert torch.equal(log_probs_tensor, expected_logits.squeeze(0))
    assert torch.equal(captures["logits"], expected_logits)
    assert torch.equal(captures["new_tokens"], sampled_tokens)
    assert captures["only_last_token_logits"]


@pytest.mark.internal
@pytest.mark.parametrize(
    ("has_pending", "availability", "active_token_count", "chunked_request_id", "expected_defer"),
    [
        (False, (True, True, True), 2, -1, False),
        (True, (True, True, True), 2, -1, True),
        (True, (True, False, True), 2, -1, True),
        (True, (False, False, True), 2, 123, True),
        (True, (True, True, False), 2, -1, False),
        (True, (False, False, False), 2, -1, False),
        (True, (True, False, True), 4, -1, False),
    ],
)
def test_pending_async_forward_defers_waiting_request_admission_that_can_mutate_context(
    has_pending, availability, active_token_count, chunked_request_id, expected_defer
):
    engine = object.__new__(DynamicInferenceEngine)
    engine.waiting_request_ids = deque([123])
    engine.enable_chunked_prefill = True
    engine.controller = SimpleNamespace(
        barrier_count=0,
        has_pending_async_forward=lambda: has_pending,
        request_async_admission_barrier=lambda: setattr(
            engine.controller, "barrier_count", engine.controller.barrier_count + 1
        ),
    )
    engine.context = SimpleNamespace(
        active_token_count=active_token_count,
        max_tokens=4,
        chunked_prefill_request_id=chunked_request_id,
        check_availability=lambda _req: availability,
    )
    engine.get_request = lambda request_id: SimpleNamespace(request_id=request_id)

    assert engine._defer_waiting_request_admission_for_async() is expected_defer
    assert engine.controller.barrier_count == int(expected_defer)


@pytest.mark.internal
def test_mamba_prefix_cache_uses_current_live_bank_for_store_and_restore():
    allocator = object.__new__(MambaSlotAllocator)
    allocator.block_to_slot = torch.tensor([0], dtype=torch.int64)
    allocator.conv_states = torch.zeros((1, 1, 1), dtype=torch.float32)
    allocator.ssm_states = torch.zeros((1, 1, 1), dtype=torch.float32)
    allocator.context = SimpleNamespace(
        mamba_state_bank_count=2,
        mamba_metadata=SimpleNamespace(
            request_to_mamba_state_idx=torch.tensor([1], dtype=torch.int64),
            request_to_mamba_state_bank=torch.tensor([1], dtype=torch.int64),
        ),
        mamba_conv_states=torch.tensor([[[10.0]], [[11.0]], [[12.0]], [[13.0]]]).transpose(0, 1),
        mamba_ssm_states=torch.tensor([[[20.0]], [[21.0]], [[22.0]], [[23.0]]]).transpose(0, 1),
    )

    allocator.store_from_live_batch(slots=[0], request_indices=[0])

    assert allocator.conv_states[:, 0].item() == 13.0
    assert allocator.ssm_states[:, 0].item() == 23.0

    allocator.conv_states[:, 0] = 31.0
    allocator.ssm_states[:, 0] = 41.0
    allocator.restore_to_live(request_idx=0, block_id=0)

    assert allocator.context.mamba_conv_states[:, 3].item() == 31.0
    assert allocator.context.mamba_ssm_states[:, 3].item() == 41.0
    assert allocator.context.mamba_conv_states[:, 1].item() == 11.0
    assert allocator.context.mamba_ssm_states[:, 1].item() == 21.0


@pytest.mark.internal
def test_inference_config_async_scheduling_flags_are_opt_in():
    default_config = InferenceConfig()
    enabled_config = InferenceConfig(
        enable_async_scheduling=True,
        async_row_map_policy=AsyncRowMapPolicy.IDENTITY_ONLY.value,
        logging_step_interval=0,
    )

    assert not default_config.enable_async_scheduling
    assert default_config.async_row_map_policy == AsyncRowMapPolicy.REUSE.value
    assert enabled_config.enable_async_scheduling
    assert enabled_config.async_row_map_policy == AsyncRowMapPolicy.IDENTITY_ONLY.value
    assert enabled_config.logging_step_interval == 0


@pytest.mark.internal
def test_nvtx_range_stack_is_thread_local(monkeypatch):
    events = []

    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda msg: events.append(("push", msg)))
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: events.append(("pop", None)))

    try:
        core_utils.configure_nvtx_profiling(True)
        core_utils.nvtx_range_push("outer")
        core_utils.nvtx_range_pop("outer")

        core_utils.nvtx_range_push("main")
        errors = []

        def worker():
            try:
                core_utils.nvtx_range_push("worker")
                core_utils.nvtx_range_pop("worker")
            except Exception as exc:
                errors.append(exc)

        import threading

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

        core_utils.nvtx_range_pop("main")
    finally:
        core_utils.configure_nvtx_profiling(False)

    assert errors == []
    assert events == [
        ("push", "outer"),
        ("pop", None),
        ("push", "main"),
        ("push", "worker"),
        ("pop", None),
        ("pop", None),
    ]
