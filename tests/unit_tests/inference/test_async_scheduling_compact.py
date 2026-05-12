# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core import utils as core_utils
from megatron.core.inference.batch_dimensions_utils import InferenceBatchDimensions
from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.ep_async_protocol import (
    EPAsyncHandoffDecision,
    EPAsyncPhase,
    EPAsyncStepProtocol,
)
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)


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
