# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.async_txn import (
    AsyncTxnDiagnostics,
    AsyncTxnSkipReason,
    StepTxn,
    TxnRetireQueue,
)
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CORE_INFERENCE_ROOT = REPO_ROOT / "megatron" / "core" / "inference"


def _core_inference_source() -> str:
    return "\n".join(path.read_text() for path in CORE_INFERENCE_ROOT.rglob("*.py"))


@pytest.mark.parametrize(
    "symbol",
    [
        "AsyncDecodeLifecyclePlan",
        "_AsyncPendingForwardView",
        "pending_forward_signature",
        "_validate_pending_forward",
        "_consume_pending_forward",
        "full_layout_signature",
        "row_map",
        "rowmap",
        "candidate_bank",
        "candidate_mamba",
        "mamba_candidate",
        "four_vote",
        "four-vote",
        "vote_protocol",
    ],
)
def test_prediction_and_reconciliation_symbols_are_absent(symbol):
    assert symbol not in _core_inference_source()


class _AdoptionContext:
    async_scheduling = True
    paused_request_count = 0
    total_request_count = 1
    padded_active_request_count = 1
    padded_active_token_count = 1

    def __init__(self):
        self.async_txn_diagnostics = AsyncTxnDiagnostics(enabled=True)
        self.request_ids = torch.tensor([11], dtype=torch.int64)

    def active_decode_slot(self):
        return None

    def is_decode_only(self):
        return True


def test_guard_failure_drops_launched_child_without_plain_decode_rerun():
    context = _AdoptionContext()
    controller = object.__new__(TextGenerationController)
    controller.inference_wrapped_model = SimpleNamespace(inference_context=context)
    controller._async_launched_child_txn = StepTxn(
        step_id=1, request_ids=(10,), launched=True
    )

    with pytest.raises(RuntimeError, match="async decode child adoption invariant failed"):
        controller._try_adopt_async_child_logits()

    assert controller._async_launched_child_txn is None
    assert context.async_txn_diagnostics.guard_failures == 1
    assert context.async_txn_diagnostics.adopted == 0


def test_step_txn_tracks_single_bank_mamba_slots_without_candidate_state():
    step_txn_fields = {field.name for field in fields(StepTxn)}

    assert "mamba_slot_ids" in step_txn_fields
    assert not any("candidate" in field_name for field_name in step_txn_fields)
    assert not any("bank" in field_name for field_name in step_txn_fields)


def test_async_skip_reasons_are_concrete():
    reason_values = {reason.value for reason in AsyncTxnSkipReason}

    assert "unknown_barrier" not in reason_values
    assert "skip_bookkeeping" in reason_values
    assert "active_count_changed" in reason_values
    assert not any("unknown" in value for value in reason_values)


class _LaunchGateContext:
    total_request_count = 2
    paused_request_count = 0

    def using_cuda_graph_this_step(self):
        return False


def _make_launch_gate_controller():
    controller = object.__new__(TextGenerationController)
    controller.model_config = SimpleNamespace(
        expert_model_parallel_size=1,
        num_moe_experts=None,
    )
    controller.inference_wrapped_model = SimpleNamespace(
        inference_context=_LaunchGateContext()
    )
    controller.num_speculative_tokens = 0
    controller._enable_cuda_graph = False
    return controller


def test_plain_decode_launch_gate_uses_concrete_local_reasons():
    controller = _make_launch_gate_controller()
    child_txn = StepTxn(step_id=1, request_ids=(101,))

    skip_bookkeeping_reason = controller._async_child_launch_skip_reason(
        child_txn,
        return_log_probs=False,
        return_top_n_logprobs=False,
        skip_bookkeeping=True,
    )
    active_count_reason = controller._async_child_launch_skip_reason(
        child_txn,
        return_log_probs=False,
        return_top_n_logprobs=False,
        skip_bookkeeping=False,
    )

    assert skip_bookkeeping_reason == AsyncTxnSkipReason.SKIP_BOOKKEEPING
    assert active_count_reason == AsyncTxnSkipReason.ACTIVE_COUNT_CHANGED


def test_cuda_graph_and_ep_are_not_blanket_async_skip_reasons():
    controller = _make_launch_gate_controller()
    controller._enable_cuda_graph = True
    controller.model_config.expert_model_parallel_size = 4
    controller.model_config.num_moe_experts = 128
    child_txn = StepTxn(step_id=1, request_ids=(101, 102))

    reason = controller._async_child_launch_skip_reason(
        child_txn,
        return_log_probs=False,
        return_top_n_logprobs=False,
        skip_bookkeeping=False,
    )

    assert reason is None


def test_diagnostics_show_compact_transaction_lifecycle():
    diagnostics = AsyncTxnDiagnostics(enabled=True)
    released = []
    retire_queue = TxnRetireQueue(diagnostics)

    diagnostics.record_prepared(under_forward=True)
    diagnostics.record_launched(h2d_ready_before_sampling=True)
    diagnostics.record_adopted()
    retire_queue.enqueue(None, lambda: released.append("retired"))
    retire_queue.drain_ready()
    diagnostics.record_barrier_skip(AsyncTxnSkipReason.SKIP_BOOKKEEPING)

    snapshot = diagnostics.snapshot()
    assert released == ["retired"]
    assert snapshot["prepared"] == 1
    assert snapshot["launched"] == 1
    assert snapshot["adopted"] == 1
    assert snapshot["retired"] == 1
    assert snapshot["prepare_under_forward"] == 1
    assert snapshot["h2d_ready_before_sampling"] == 1
    assert snapshot["top_skip_reason"] == AsyncTxnSkipReason.SKIP_BOOKKEEPING.value
