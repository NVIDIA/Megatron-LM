# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core import utils as core_utils
from megatron.core.inference.batch_dimensions_utils import InferenceBatchDimensions
from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
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
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.transformer.enums import CudaGraphScope


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
        ("async", EPAsyncPhase.STEP_COMPLETE, 0, (1,), False),
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
        ((1, 1, 1, 1, 0, 0, 0, 0), (True, True, False, False)),
        ((1, 1, 1, 1, 1, 0, 0, 0), (True, True, False, True)),
        ((1, 1, 1, 1, 0, 0, 1, 0), (True, False, True, False)),
        ((1, 1, 1, 0, 0, 1, 0, 0), (True, False, True, False)),
        ((1, 1, 1, 1, 0, 0, 0, 1), (False, True, False, False)),
    ],
)
async def test_ep_step_begin_reuses_or_discards_with_global_state(global_state, expected):
    communicator = _RecordingEPCommunicator(async_results=[(1, 0), 1], sync_results=[global_state])
    protocol = EPAsyncStepProtocol(communicator)
    await protocol.establish_work_consensus(local_work=1, signal_consensus=False)

    decision = protocol.decide_step_begin(
        has_real_work=True,
        has_pending_forward=True,
        has_pending_async_sample=True,
        pending_forward_reusable=True,
        pending_forward_row_mapped=bool(global_state[4]),
    )
    await protocol.complete_work_step()

    assert (
        decision.use_pending_async_sample,
        decision.reuse_pending_forward,
        decision.discard_pending_forward,
        decision.row_mapped_forward,
    ) == expected
    assert communicator.calls[1] == (
        "sync",
        EPAsyncPhase.STEP_BEGIN,
        0,
        (1, 1, 1, 1, int(bool(global_state[4])), 0, 0, 0),
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
    communicator = _RecordingEPCommunicator(sync_results=[global_state])
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
        )
    ]


@pytest.mark.internal
def test_ep_protocol_diagnostics_count_reuse_discard_launch_and_skip():
    communicator = _RecordingEPCommunicator(
        sync_results=[(1, 1, 1, 1, 0, 0, 0, 0), (1, 1, 1, 0, 0, 1, 0, 0), (1, 1, 0), (1, 1, 1)]
    )
    protocol = EPAsyncStepProtocol(communicator)

    reuse = protocol.decide_step_begin(
        has_real_work=True,
        has_pending_forward=True,
        has_pending_async_sample=True,
        pending_forward_reusable=True,
        pending_forward_row_mapped=False,
    )
    discard = protocol.decide_step_begin(
        has_real_work=True,
        has_pending_forward=True,
        has_pending_async_sample=True,
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


def _make_controller_with_rows(pending_ids, current_ids):
    controller = object.__new__(TextGenerationController)
    controller._async_pending_forward_request_ids = (
        None if pending_ids is None else torch.tensor(pending_ids, dtype=torch.int64)
    )
    controller._async_discarded_forward_count = 0
    controller._async_row_mapped_forward_count = 0
    controller.inference_wrapped_model = SimpleNamespace(
        inference_context=SimpleNamespace(
            request_ids=torch.tensor(current_ids, dtype=torch.int64),
            paused_request_count=0,
            total_request_count=len(current_ids),
        )
    )
    return controller


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="row mapping returns CUDA tensors")
@pytest.mark.parametrize(
    ("pending_ids", "current_ids", "expected_status", "expected_resolve", "expected_rows"),
    [
        (None, [10, 11], (True, False), (True, False), None),
        ([10, 11], [10, 11], (True, False), (True, False), None),
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
    usable, row_indices, row_mapped = controller._resolve_pending_async_forward_rows()

    assert (usable, row_mapped) == expected_resolve
    if expected_rows is None:
        assert row_indices is None
    else:
        assert row_indices.tolist() == expected_rows
    assert controller._async_discarded_forward_count == int(not expected_resolve[0])
    assert controller._async_row_mapped_forward_count == int(expected_resolve[1])
    assert controller._async_pending_forward_request_ids is None


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
    controller._async_pending_forward = pending_forward
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
                use_pending_async_sample=kwargs["has_pending_async_sample"],
                reuse_pending_forward=kwargs["pending_forward_reusable"],
                discard_pending_forward=not kwargs["pending_forward_reusable"],
                row_mapped_forward=kwargs["pending_forward_row_mapped"],
            )

    if use_protocol:
        controller._ep_async_protocol = _Protocol()
    else:
        controller._ep_async_protocol = None

    decision = controller._decide_ep_step_begin(has_real_work=True, pending_async_sample=True)

    assert controller._ep_async_handoff_decided_this_step is False
    assert controller._ep_async_handoff_decision_this_step is None
    if use_protocol:
        assert protocol_calls == [
            {
                "has_real_work": True,
                "has_pending_forward": pending_forward,
                "has_pending_async_sample": True,
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


def _make_async_gate_controller(active_request_count=2):
    controller = object.__new__(TextGenerationController)
    controller._async_scheduling_enabled = True
    controller._async_step_barrier_reason = None
    controller._enable_cuda_graph = True
    controller.model_config = SimpleNamespace(
        cuda_graph_scope=[CudaGraphScope.full_iteration_inference]
    )
    controller.model_is_pipeline_parallel = False
    controller.num_speculative_tokens = 0
    controller._num_mtp_depths = 0
    controller._sampling_backend = "torch"
    controller._async_admission_barrier_requested = False
    context = SimpleNamespace(
        total_request_count=active_request_count,
        paused_request_count=0,
        padded_batch_dimensions=InferenceBatchDimensions(
            active_request_count, 0, active_request_count
        ),
        is_decode_only=lambda: True,
        using_cuda_graph_this_step=lambda: True,
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
        ("no_full_iteration_graph", "requires full-iteration inference cuda graphs"),
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
    elif case == "no_full_iteration_graph":
        controller.model_config.cuda_graph_scope = []
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
        ([1, 1], [0.0, 0.0], [False, False], [0, 0], False, (False, False, False), True),
        ([4, 1], [0.0, 0.0], [False, False], [0, 0], False, (True, False, True), True),
        ([1, 1], [0.0, 0.0], [True, False], [0, 0], True, (True, False, False), True),
        ([1, 1], [0.0, 0.0], [False, False], [2, 0], True, (True, False, False), True),
        ([1, 1], [0.0, 0.0], [True, False], [0, 0], True, (True, False, True), False),
        ([1, 1], [0.0, 0.0], [False, False], [0, 0], True, (True, True, False), False),
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
    async_next_prepared, pending_forward_reused, async_sample_already_launched = bookkeeping_state

    assert (
        controller._should_collect_dynamic_sampling_bookkeeping(
            async_next_prepared=async_next_prepared,
            pending_forward_reused=pending_forward_reused,
            async_sample_already_launched=async_sample_already_launched,
        )
        is expected
    )


def _install_async_prepare_stubs(
    controller,
    *,
    disabled_reason=None,
    prepare_result=True,
    launch_decision=True,
    logprob_results=False,
    greedy=True,
    async_decode_graph=None,
):
    events = []
    context = controller.inference_wrapped_model.inference_context
    context.prepare_async_decode_next_step = lambda: events.append("prepare") or prepare_result
    controller._async_scheduling_disabled_reason = (
        lambda **_kwargs: events.append(("disabled", _kwargs)) or disabled_reason
    )
    controller._record_async_eligibility_result = lambda reason: events.append(
        ("eligibility", reason)
    )
    controller._record_async_disable_reason = lambda reason: events.append(("disable", reason))
    controller._get_async_decode_graph = lambda: async_decode_graph
    controller._active_requests_need_logprob_results = lambda: logprob_results
    controller._active_requests_use_greedy_sampling = lambda _count: greedy

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
    ("case", "expected_ok", "expected_disable_reason"),
    [
        ("disabled", False, None),
        ("prepare_failed", False, "failed to prepare next-step metadata"),
        ("handoff_skipped", False, "ep async handoff skipped"),
        ("logprobs", True, None),
        ("non_greedy", True, None),
        ("fallback_forward", True, None),
    ],
)
def test_launch_async_decode_graph_handoff_paths(
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
        logprob_results=case == "logprobs",
        greedy=case != "non_greedy",
    )

    ok, sampled, mtp_sampled, sample_event, h2d_event, graph_launched = (
        controller._try_launch_async_decode_graph(active_request_count=2)
    )

    assert ok is expected_ok
    assert (sampled, mtp_sampled, sample_event, h2d_event, graph_launched) == (
        None,
        None,
        None,
        None,
        False,
    )
    if case == "disabled":
        assert ("handoff", True, False) in events
        assert "prepare" not in events
    elif case == "prepare_failed":
        assert ("handoff", True, False) in events
    else:
        assert ("handoff", True, True) in events
    if expected_disable_reason is not None:
        assert ("disable", expected_disable_reason) in events


@pytest.mark.internal
@pytest.mark.parametrize("require_captured_graph", [False, True])
def test_launch_async_decode_graph_captured_graph_launch_and_required_gate(
    monkeypatch, require_captured_graph
):
    monkeypatch.setattr(tgc_module, "range_push", lambda _msg: None)
    monkeypatch.setattr(tgc_module, "range_pop", lambda: None)
    controller, _context = _make_async_gate_controller()
    sample_ready_event = object()
    h2d_done_event = object()
    fake_graph = SimpleNamespace(
        sample_ready_events=(sample_ready_event,), h2d_done_events=(h2d_done_event,)
    )
    events = _install_async_prepare_stubs(
        controller, async_decode_graph=None if require_captured_graph else fake_graph
    )
    controller._launch_async_decode_graph = lambda graph: events.append(("launch", graph)) or 0
    controller._async_transfer_samples_to_cpu = (
        lambda active_count, event, sample_slot: events.append(
            ("transfer", active_count, event, sample_slot)
        )
        or (torch.tensor([1, 2]), None, event)
    )

    ok, sampled, mtp_sampled, sample_event, h2d_event, graph_launched = (
        controller._try_launch_async_decode_graph(
            active_request_count=2, require_captured_graph=require_captured_graph
        )
    )

    if require_captured_graph:
        assert not ok
        assert (sampled, mtp_sampled, sample_event, h2d_event, graph_launched) == (
            None,
            None,
            None,
            None,
            False,
        )
        assert ("disable", "async decode graph not captured") in events
        assert "prepare" not in events
    else:
        assert ok
        assert sampled.tolist() == [1, 2]
        assert mtp_sampled is None
        assert sample_event is sample_ready_event
        assert h2d_event is h2d_done_event
        assert graph_launched
        assert ("launch", fake_graph) in events
        assert ("transfer", 2, sample_ready_event, 0) in events


@pytest.mark.internal
@pytest.mark.parametrize(
    ("case", "expected_ok", "expected_disable_reason"),
    [
        ("disabled", False, None),
        ("handoff_skipped", False, "ep async handoff skipped"),
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
        assert "prepare" not in events
        assert ("handoff", True, False) in events
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
    assert controller._async_forward_launch_count == expected_forwards
    if case == "handoff_skipped":
        assert ("disable", "ep async handoff skipped") in events


@pytest.mark.internal
def test_pending_async_forward_and_sample_cleanup_releases_only_when_needed():
    context = SimpleNamespace(release_count=0)
    context.release_deferred_async_kv_blocks = lambda: setattr(
        context, "release_count", context.release_count + 1
    )
    controller = object.__new__(TextGenerationController)
    controller.inference_wrapped_model = SimpleNamespace(inference_context=context)
    controller._async_pending_forward = False
    controller._async_pending_cuda_graph_request_count = 3
    controller._async_pending_forward_request_ids = torch.tensor([1, 2])
    controller._async_discarded_forward_count = 0

    controller._discard_pending_async_forward()
    assert context.release_count == 0
    assert controller._async_discarded_forward_count == 0

    controller._async_pending_forward = True
    controller._discard_pending_async_forward()
    assert context.release_count == 1
    assert not controller._async_pending_forward
    assert controller._async_pending_cuda_graph_request_count is None
    assert controller._async_pending_forward_request_ids is None
    assert controller._async_discarded_forward_count == 1

    controller._async_pending_sampled_tokens_cpu = torch.tensor([1])
    controller._async_pending_sampled_mtp_tokens_cpu = torch.tensor([[2]])
    controller._async_pending_sample_ready_event = object()
    controller._async_pending_h2d_done_event = object()
    controller._async_pending_sample_cuda_graph_request_count = 4
    controller._clear_pending_async_sample()
    assert controller._async_pending_sampled_tokens_cpu is None
    assert controller._async_pending_sampled_mtp_tokens_cpu is None
    assert controller._async_pending_sample_ready_event is None
    assert controller._async_pending_h2d_done_event is None
    assert controller._async_pending_sample_cuda_graph_request_count is None


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
    consumed_events = []
    controller._mark_async_sample_copy_consumed = consumed_events.append
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
    assert consumed_events == [sample_ready_event]
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


@pytest.mark.internal
def test_async_reserved_kv_blocks_are_adopted_or_deferred_then_released():
    context = object.__new__(DynamicInferenceContext)
    context.paused_request_count = 0
    context.total_request_count = 3
    context.request_ids = torch.tensor([10, 11, 12], dtype=torch.int32)
    context.request_kv_block_counts = torch.tensor([1, 1, 1], dtype=torch.int32)
    context.request_to_kv_block_ids = torch.tensor(
        [[1, -1, -1], [2, -1, -1], [3, -1, -1]], dtype=torch.int32
    )
    context.request_last_kv_block_id = torch.tensor([1, 2, 3], dtype=torch.int32)
    context._async_reserved_kv_block_count = 3
    context._async_reserved_kv_block_request_ids = torch.tensor([10, 11, 99], dtype=torch.int32)
    context._async_reserved_kv_block_ids = torch.tensor([100, 101, 102], dtype=torch.int32)
    context._async_reserved_kv_block_columns = torch.tensor([1, 1, 1], dtype=torch.int32)
    context._async_deferred_kv_blocks_to_release = torch.empty(0, dtype=torch.int32)
    context._async_reserved_kv_block_adoption_count = 0
    context._async_deferred_kv_block_release_count = 0
    context.kv_block_allocator = _ReleaseRecordingAllocator()

    adopted = context._adopt_or_defer_async_reserved_kv_blocks(
        torch.tensor([1, 0, 1], dtype=torch.uint8)
    )

    assert adopted.tolist() == [10]
    assert context.request_to_kv_block_ids.tolist() == [[1, 100, -1], [2, -1, -1], [3, -1, -1]]
    assert context.request_kv_block_counts.tolist() == [2, 1, 1]
    assert context.request_last_kv_block_id.tolist() == [100, 2, 3]
    assert context._async_reserved_kv_block_count == 0
    assert context._async_reserved_kv_block_request_ids.tolist() == [-1, -1, -1]
    assert context._async_deferred_kv_blocks_to_release.tolist() == [101, 102]
    assert context._async_reserved_kv_block_adoption_count == 1

    context.release_deferred_async_kv_blocks()

    assert [blocks.tolist() for blocks in context.kv_block_allocator.released] == [[101, 102]]
    assert context._async_deferred_kv_blocks_to_release.numel() == 0
    assert context._async_deferred_kv_block_release_count == 2


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
def test_inference_config_async_scheduling_flags_are_opt_in():
    default_config = InferenceConfig()
    enabled_config = InferenceConfig(
        enable_async_scheduling=True, enable_async_decode_graphs=True, logging_step_interval=0
    )

    assert not default_config.enable_async_scheduling
    assert not default_config.enable_async_decode_graphs
    assert enabled_config.enable_async_scheduling
    assert enabled_config.enable_async_decode_graphs
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
