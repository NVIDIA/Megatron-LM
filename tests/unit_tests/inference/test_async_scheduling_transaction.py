# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import MethodType, SimpleNamespace

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
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
from megatron.core.inference.ep_async_protocol import EPAsyncPhase, EPAsyncStepProtocol
from megatron.core.inference.text_generation_controllers import (
    text_generation_controller as tgc_module,
)
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
def test_identity_row_map_does_not_materialize_device_gather_rows():
    row_map = AsyncRowMap.for_current(
        pending_request_ids=torch.tensor([10, 11, 12]),
        current_request_ids=torch.tensor([10, 11]),
        device="cpu",
    )

    assert row_map is not None
    assert not row_map.row_mapped
    assert row_map.pending_rows is None


@pytest.mark.internal
def test_row_mapped_forward_materializes_device_gather_rows():
    row_map = AsyncRowMap.for_current(
        pending_request_ids=torch.tensor([10, 11, 12]),
        current_request_ids=torch.tensor([11, 12]),
        device="cpu",
    )

    assert row_map is not None
    assert row_map.row_mapped
    assert row_map.pending_rows is not None
    assert row_map.pending_rows.tolist() == [1, 2]


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


@pytest.mark.internal
@pytest.mark.parametrize("uses_mamba_candidate_bank", [False, True])
def test_transaction_from_plan_records_resource_leases(uses_mamba_candidate_bank):
    plan = SimpleNamespace(
        request_ids=torch.tensor([10, 11], dtype=torch.int32),
        reserved_request_ids=torch.tensor([11], dtype=torch.int32),
        reserved_block_ids=torch.tensor([101], dtype=torch.int32),
        reserved_block_columns=torch.tensor([2], dtype=torch.int32),
        active_request_count=2,
        active_token_count=6,
        padded_active_request_count=4,
    )

    txn = AsyncDecodeTransaction.from_plan(
        transaction_id=9,
        prepared_layout=plan,
        tokens_per_request=3,
        uses_mamba_candidate_bank=uses_mamba_candidate_bank,
        cuda_graph_request_count=4,
    )

    assert txn.state == AsyncTransactionState.IN_FLIGHT
    assert txn.graph_shape == AsyncGraphShape(
        active_request_count=2,
        active_token_count=6,
        padded_active_request_count=4,
        tokens_per_request=3,
    )
    assert txn.kv_lease.reserved_request_ids.tolist() == [11]
    assert txn.kv_lease.reserved_block_ids.tolist() == [101]
    assert txn.mamba_lease.uses_candidate_bank is uses_mamba_candidate_bank
    assert txn.mamba_lease.candidate_request_ids.tolist() == (
        [10, 11] if uses_mamba_candidate_bank else []
    )
    assert txn.mtp_state is not None
    assert txn.mtp_state.tokens_per_request == 3
    assert txn.cuda_graph_request_count == 4


@pytest.mark.internal
def test_transaction_rejects_tokens_per_request_mismatch():
    plan = SimpleNamespace(
        request_ids=torch.tensor([1, 2], dtype=torch.int32),
        reserved_request_ids=torch.empty(0, dtype=torch.int32),
        reserved_block_ids=torch.empty(0, dtype=torch.int32),
        reserved_block_columns=torch.empty(0, dtype=torch.int32),
        active_request_count=2,
        active_token_count=6,
        padded_active_request_count=4,
    )
    txn = AsyncDecodeTransaction.from_plan(
        transaction_id=3,
        prepared_layout=plan,
        tokens_per_request=3,
    )

    assert (
        txn.resolve_for_current(
            current_request_ids=torch.tensor([1, 2], dtype=torch.int32),
            current_graph_shape=AsyncGraphShape(
                active_request_count=2,
                active_token_count=2,
                padded_active_request_count=4,
                tokens_per_request=1,
            ),
        )
        is None
    )


@pytest.mark.internal
def test_transaction_resolves_row_mapped_mtp_stride_subset():
    plan = SimpleNamespace(
        request_ids=torch.tensor([10, 11], dtype=torch.int32),
        reserved_request_ids=torch.empty(0, dtype=torch.int32),
        reserved_block_ids=torch.empty(0, dtype=torch.int32),
        reserved_block_columns=torch.empty(0, dtype=torch.int32),
        active_request_count=2,
        active_token_count=6,
        padded_active_request_count=4,
    )
    txn = AsyncDecodeTransaction.from_plan(
        transaction_id=5,
        prepared_layout=plan,
        tokens_per_request=3,
    )

    row_map = txn.resolve_for_current(
        current_request_ids=torch.tensor([11], dtype=torch.int32),
        current_graph_shape=AsyncGraphShape(
            active_request_count=1,
            active_token_count=3,
            padded_active_request_count=4,
            tokens_per_request=3,
        ),
    )

    assert row_map is not None
    assert row_map.row_mapped
    assert row_map.pending_rows_cpu.tolist() == [1]


def _eligible_async_context(**overrides):
    active_request_count = overrides.pop("active_request_count", 2)
    block_size_tokens = overrides.pop("block_size_tokens", 8)
    top_k = overrides.pop("top_k", [1] * active_request_count)
    top_p = overrides.pop("top_p", [0.0] * active_request_count)
    return_log_probs = overrides.pop("return_log_probs", [False] * active_request_count)
    top_n_logprobs = overrides.pop("top_n_logprobs", [0] * active_request_count)
    last_offsets = overrides.pop("last_offsets", [2] * active_request_count)
    context = SimpleNamespace(
        enable_async_scheduling=True,
        _async_reserved_kv_block_count=0,
        num_speculative_tokens=0,
        is_decode_only=lambda: True,
        paused_request_count=0,
        total_request_count=active_request_count,
        get_index_of_chunked_prefill_request=lambda safe=True: -1,
        request_kv_length_offsets=torch.arange(active_request_count, dtype=torch.int64) + 10,
        request_query_lengths=torch.ones(active_request_count, dtype=torch.int64),
        request_metadata={
            "termination_id": torch.full((active_request_count,), -1, dtype=torch.int64)
        },
        request_output_lengths=torch.full((active_request_count,), 100, dtype=torch.int64),
        request_last_kv_block_offset=torch.tensor(last_offsets, dtype=torch.int64),
        block_size_tokens=block_size_tokens,
        kv_block_allocator=SimpleNamespace(total_avail=active_request_count),
        active_request_metadata={
            "temperature": torch.ones(active_request_count, dtype=torch.float32),
            "top_k": torch.tensor(top_k, dtype=torch.int32),
            "top_p": torch.tensor(top_p, dtype=torch.float32),
            "return_log_probs": torch.tensor(return_log_probs, dtype=torch.bool),
            "top_n_logprobs": torch.tensor(top_n_logprobs, dtype=torch.int32),
        },
    )
    context.active_request_metadata_needs_logprob_results = lambda active_count=None: bool(
        context.active_request_metadata["return_log_probs"][
            : active_request_count if active_count is None else active_count
        ].any()
        or (
            context.active_request_metadata["top_n_logprobs"][
                : active_request_count if active_count is None else active_count
            ]
            > 0
        ).any()
    )
    for name, value in overrides.items():
        setattr(context, name, value)
    return context


@pytest.mark.internal
@pytest.mark.parametrize(
    ("overrides", "kwargs", "expected"),
    [
        ({"num_speculative_tokens": 2}, {}, "skip_mtp"),
        ({}, {"skip_bookkeeping": True}, "skip_bookkeeping"),
        ({"return_log_probs": [True, False]}, {}, "skip_logprob_results"),
        ({"top_n_logprobs": [0, 3]}, {}, "skip_logprob_results"),
        ({}, {"active_stop_words": True}, "skip_stop_words"),
        ({"last_offsets": [7, 2]}, {}, "skip_mixed_block_boundary"),
        (
            {"last_offsets": [7, 7], "kv_block_allocator": SimpleNamespace(total_avail=1)},
            {},
            "skip_kv_block_unavailable",
        ),
        ({"top_k": [4, 1]}, {}, None),
        ({"top_p": [0.7, 0.0]}, {}, None),
    ],
)
def test_async_decode_planner_skip_reason_matrix(overrides, kwargs, expected):
    context = _eligible_async_context(**overrides)

    reason = DynamicInferenceContext.async_decode_next_step_skip_reason(context, **kwargs)

    assert reason == expected


@pytest.mark.internal
def test_prompt_logprob_history_does_not_disable_async_after_active_metadata_clears():
    context = _eligible_async_context(return_log_probs=[True, False], top_n_logprobs=[0, 0])

    assert (
        DynamicInferenceContext.async_decode_next_step_skip_reason(context)
        == "skip_logprob_results"
    )

    context.active_request_metadata["return_log_probs"].zero_()
    assert DynamicInferenceContext.async_decode_next_step_skip_reason(context) is None


@pytest.mark.internal
def test_controller_delegates_async_prepare_eligibility_to_context_planner():
    events = []
    context = SimpleNamespace(
        paused_request_count=0,
        total_request_count=2,
        request_ids=torch.tensor([10, 11], dtype=torch.int64),
        prepare_async_decode_next_step=lambda **kwargs: events.append(kwargs) or True,
    )
    controller = object.__new__(TextGenerationController)
    controller.inference_wrapped_model = SimpleNamespace(inference_context=context)
    controller._async_stop_word_requests_seen = True
    controller._has_active_stop_words_callback = lambda request_ids: {11}.intersection(
        request_ids
    )

    assert controller._try_prepare_async_decode_next_step(skip_bookkeeping=True)
    assert events == [{"skip_bookkeeping": True, "active_stop_words": True}]


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
def test_ep_handoff_diagnostics_count_launch_and_skip():
    communicator = _RecordingEPCommunicator(sync_results=[(1, 1, 0), 1, (1, 1, 1), 1])
    protocol = EPAsyncStepProtocol(communicator)

    launch = protocol.decide_async_handoff(
        has_real_work=True, can_launch_async_handoff=True
    )
    skip = protocol.decide_async_handoff(
        has_real_work=True, can_launch_async_handoff=True
    )

    assert launch.launch_async_forward
    assert skip.skip_async_forward
    diagnostics = protocol.diagnostics()
    assert diagnostics["handoff_launches"] == 1
    assert diagnostics["handoff_skips"] == 1
    assert communicator.calls == [
        (EPAsyncPhase.ASYNC_HANDOFF, 0, (1, 1, 0)),
        (EPAsyncPhase.ASYNC_HANDOFF_ACK, 0, (1,)),
        (EPAsyncPhase.ASYNC_HANDOFF, 1, (1, 1, 0)),
        (EPAsyncPhase.ASYNC_HANDOFF_ACK, 1, (1,)),
    ]


@pytest.mark.internal
def test_dummy_async_handoff_fences_h2d_before_context_reset(monkeypatch):
    monkeypatch.setattr(tgc_module, "range_push", lambda _msg: None)
    monkeypatch.setattr(tgc_module, "range_pop", lambda: None)
    events = []
    context = SimpleNamespace(
        reset=lambda: events.append("reset"),
        record_async_scheduling_counter=lambda name: events.append(("counter", name)),
    )
    controller = object.__new__(TextGenerationController)
    controller.inference_wrapped_model = SimpleNamespace(inference_context=context)
    controller._dummy_async_handoff_disabled_reason = lambda: None
    controller._decide_ep_async_handoff = lambda **_kwargs: SimpleNamespace(
        launch_async_forward=True
    )
    controller._wait_for_dummy_context_h2d = lambda: events.append("wait_h2d")
    controller._dynamic_step_context_init = lambda is_dummy_forward=False: events.append(
        ("context_init", is_dummy_forward)
    ) or (None, None)
    controller._dynamic_step_forward_logits = lambda _input_ids, _position_ids: events.append(
        "forward"
    )

    assert controller._try_launch_dummy_async_handoff()
    assert events == [
        "wait_h2d",
        "reset",
        ("context_init", True),
        "forward",
        ("counter", "dummy_launch"),
    ]


@pytest.mark.internal
@pytest.mark.asyncio
async def test_ep_work_step_sends_coordinator_replies_after_step_completion():
    events = []
    engine = object.__new__(DynamicInferenceEngine)

    async def _async_step(*, send_coordinator_replies=True):
        events.append(("step", send_coordinator_replies))
        return {"finished_request_records": ["finished"]}

    async def _ep_complete_work_step():
        events.append("ep_complete")

    engine.async_step = _async_step
    engine._ep_complete_work_step = _ep_complete_work_step
    engine._send_finished_records_to_coordinator = lambda records: events.append(
        ("coordinator_reply", records)
    )

    await engine._run_ep_work_step(local_pending=1)

    assert events == [
        ("step", False),
        "ep_complete",
        ("coordinator_reply", ["finished"]),
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


@pytest.mark.internal
def test_controller_generated_logprobs_use_pending_row_indices():
    logits = torch.arange(1 * 4 * 5, dtype=torch.float32).view(1, 4, 5)
    sampled = torch.tensor([3, 1], dtype=torch.long)
    recorded = {}

    def _calculate_log_probs(logits_arg, new_tokens, only_last_token_logits=False):
        recorded["logits"] = logits_arg.clone()
        recorded["new_tokens"] = new_tokens.clone()
        recorded["only_last"] = only_last_token_logits
        return [[0.0], [0.0]], torch.log_softmax(logits_arg.squeeze(0), dim=-1)

    context = SimpleNamespace(
        total_request_count=2,
        paused_request_count=0,
        config=SimpleNamespace(materialize_only_last_token_logits=False),
        is_decode_only=lambda: True,
        calculate_log_probs=_calculate_log_probs,
    )
    controller = SimpleNamespace(
        num_speculative_tokens=0,
        inference_wrapped_model=SimpleNamespace(inference_context=context),
        _all_logits_cuda=logits,
        _sampled_tokens_cuda=sampled,
    )

    log_probs, _ = TextGenerationController._dynamic_step_calculate_log_probs(
        controller, row_indices=torch.tensor([3, 1])
    )

    assert log_probs == [[0.0], [0.0]]
    assert recorded["only_last"]
    assert torch.equal(recorded["new_tokens"], sampled)
    assert torch.equal(recorded["logits"], logits.index_select(1, torch.tensor([3, 1])))


@pytest.mark.internal
def test_controller_top_n_logprobs_use_selected_row_tensor():
    context = SimpleNamespace(
        total_request_count=2,
        paused_request_count=0,
        active_request_metadata={
            "top_n_logprobs": torch.tensor([1, 2], dtype=torch.int32),
        },
        config=SimpleNamespace(materialize_only_last_token_logits=False),
        is_decode_only=lambda: True,
    )
    controller = SimpleNamespace(
        inference_wrapped_model=SimpleNamespace(inference_context=context),
    )
    selected_log_probs = torch.tensor(
        [
            [0.0, 10.0, 2.0, 3.0],
            [9.0, 7.0, 11.0, 1.0],
        ],
        dtype=torch.float32,
    )

    top_n = TextGenerationController._dynamic_step_calculate_top_n_logprobs(
        controller,
        selected_log_probs,
        row_indices=torch.tensor([3, 1]),
    )

    assert top_n is not None
    assert top_n[0][0][1].tolist() == [1]
    assert top_n[1][0][1].tolist() == [2, 0]


@pytest.mark.internal
def test_row_mapped_sampled_token_transfer_uses_current_request_order():
    logits = torch.zeros(1, 4, 5, dtype=torch.float32)
    logits[0, 2, 4] = 10.0
    logits[0, 0, 1] = 9.0
    copied = {}

    def _copy_async_prepared_decode_input_ids_from_samples(sampled_tokens):
        copied["sampled_tokens"] = sampled_tokens[:2].clone()
        return True

    context = SimpleNamespace(
        total_request_count=2,
        paused_request_count=0,
        config=SimpleNamespace(materialize_only_last_token_logits=True),
        is_decode_only=lambda: True,
        active_request_metadata={
            "top_k": torch.tensor([1, 1], dtype=torch.int32),
            "top_p": torch.tensor([0.0, 0.0], dtype=torch.float32),
        },
        copy_async_prepared_decode_input_ids_from_samples=(
            _copy_async_prepared_decode_input_ids_from_samples
        ),
    )
    controller = object.__new__(TextGenerationController)
    controller._async_non_greedy_requests_seen = False
    controller._all_logits_cuda = logits
    controller._greedy_sample_values_cuda = torch.empty(2, dtype=torch.float32)
    controller._greedy_sampled_tokens_cuda = torch.empty(2, dtype=torch.int64)
    controller.inference_wrapped_model = SimpleNamespace(inference_context=context)

    TextGenerationController._dynamic_step_sample_logits_to_next_input_ids(
        controller,
        row_indices=torch.tensor([2, 0], dtype=torch.long),
    )

    assert copied["sampled_tokens"].tolist() == [4, 1]


class _ReleaseRecordingAllocator:
    def __init__(self):
        self.released = []

    def release_memory_blocks(self, blocks):
        self.released.append(blocks.clone())


@pytest.mark.internal
def test_discard_prepared_plan_releases_unlaunched_kv_reservations():
    allocator = _ReleaseRecordingAllocator()
    context = SimpleNamespace(
        _async_prepared_decode_plan=SimpleNamespace(
            reserved_block_ids=torch.tensor([30, 31], dtype=torch.int32)
        ),
        _async_reserved_kv_block_count=2,
        _async_reserved_kv_block_request_ids=torch.tensor([10, 11], dtype=torch.int32),
        _async_reserved_kv_block_ids=torch.tensor([30, 31], dtype=torch.int32),
        _async_reserved_kv_block_columns=torch.tensor([1, 1], dtype=torch.int32),
        kv_block_allocator=allocator,
        async_scheduling_counters={},
    )
    context.record_async_scheduling_counter = MethodType(
        DynamicInferenceContext.record_async_scheduling_counter, context
    )
    context._clear_async_reserved_kv_blocks = MethodType(
        DynamicInferenceContext._clear_async_reserved_kv_blocks, context
    )
    context.clear_async_prepared_decode_plan = MethodType(
        DynamicInferenceContext.clear_async_prepared_decode_plan, context
    )

    DynamicInferenceContext.discard_async_prepared_decode_plan(context)

    assert [blocks.tolist() for blocks in allocator.released] == [[30, 31]]
    assert context._async_prepared_decode_plan is None
    assert context._async_reserved_kv_block_count == 0
    assert context._async_reserved_kv_block_ids.tolist() == [-1, -1]
    assert context.async_scheduling_counters["kv_lease_dropped"] == 2


@pytest.mark.internal
def test_async_reserved_kv_blocks_are_adopted_or_deferred_then_released():
    context = object.__new__(DynamicInferenceContext)
    context._async_reserved_kv_block_count = 3
    context._async_reserved_kv_block_request_ids = torch.tensor([10, 11, 99], dtype=torch.int32)
    context._async_reserved_kv_block_ids = torch.tensor([100, 101, 102], dtype=torch.int32)
    context._async_reserved_kv_block_columns = torch.tensor([1, 1, 1], dtype=torch.int32)
    context._async_deferred_kv_blocks_to_release = torch.empty(0, dtype=torch.int32)
    context._async_deferred_mamba_slots_to_free = torch.empty(0, dtype=torch.int32)
    context._async_reserved_kv_block_adoption_count = 0
    context._async_deferred_kv_block_release_count = 0
    context._async_deferred_mamba_slot_release_count = 0
    context.is_hybrid_model = False
    context.async_scheduling_counters = {}
    context.kv_block_allocator = _ReleaseRecordingAllocator()

    consumed = context._consume_async_reserved_kv_blocks(
        torch.tensor([10], dtype=torch.int32), torch.tensor([1], dtype=torch.int32)
    )

    assert consumed.tolist() == [100]
    assert context._async_reserved_kv_block_ids.tolist() == [-1, 101, 102]
    assert context._async_reserved_kv_block_adoption_count == 1
    assert context.async_scheduling_counters["kv_lease_adopted"] == 1

    context._defer_remaining_async_reserved_kv_blocks()

    assert context._async_reserved_kv_block_count == 0
    assert context._async_reserved_kv_block_request_ids.tolist() == [-1, -1, -1]
    assert context._async_deferred_kv_blocks_to_release.tolist() == [101, 102]

    context.release_deferred_async_resources()

    assert [blocks.tolist() for blocks in context.kv_block_allocator.released] == [[101, 102]]
    assert context._async_deferred_kv_blocks_to_release.numel() == 0
    assert context._async_deferred_kv_block_release_count == 2
    assert context.async_scheduling_counters["kv_lease_released"] == 2


class _MambaMetadataWithFreeRecording:
    def __init__(self):
        self.request_to_mamba_state_idx = torch.tensor([2, -1, 4], dtype=torch.int32)
        self.request_to_mamba_state_bank = torch.tensor([1, 0, 1], dtype=torch.int32)
        self.mamba_state_free_slots = torch.empty(0, dtype=torch.int32)
        self.freed_slots = []
        self.reset_called = False
        self.reset_varlen_count = 0

    def free_slot_ids(self, slots):
        self.freed_slots.append(slots.clone().cpu())

    def reset(self):
        self.reset_called = True

    def reset_varlen_metadata(self):
        self.reset_varlen_count += 1


@pytest.mark.internal
def test_async_mamba_reset_defers_allocated_slots_while_forward_is_in_flight():
    metadata = _MambaMetadataWithFreeRecording()
    context = SimpleNamespace(
        is_hybrid_model=True,
        _async_forward_in_flight=True,
        _async_deferred_kv_blocks_to_release=torch.empty(0, dtype=torch.int32),
        _async_deferred_mamba_slots_to_free=torch.empty(0, dtype=torch.int32),
        _async_deferred_kv_block_release_count=0,
        _async_deferred_mamba_slot_release_count=0,
        mamba_metadata=metadata,
        async_scheduling_counters={},
    )
    context.record_async_scheduling_counter = MethodType(
        DynamicInferenceContext.record_async_scheduling_counter, context
    )
    context._append_deferred_async_mamba_slots = MethodType(
        DynamicInferenceContext._append_deferred_async_mamba_slots, context
    )
    context._release_deferred_async_mamba_slots = MethodType(
        DynamicInferenceContext._release_deferred_async_mamba_slots, context
    )

    DynamicInferenceContext.reset_mamba_state(context)

    assert not metadata.reset_called
    assert metadata.reset_varlen_count == 1
    assert metadata.request_to_mamba_state_idx.tolist() == [-1, -1, -1]
    assert metadata.request_to_mamba_state_bank.tolist() == [0, 0, 0]
    assert context._async_deferred_mamba_slots_to_free.tolist() == [2, 4]

    DynamicInferenceContext._release_deferred_async_mamba_slots(context)

    assert [slots.tolist() for slots in metadata.freed_slots] == [[2, 4]]
    assert context._async_deferred_mamba_slots_to_free.numel() == 0
    assert context._async_deferred_mamba_slot_release_count == 2
