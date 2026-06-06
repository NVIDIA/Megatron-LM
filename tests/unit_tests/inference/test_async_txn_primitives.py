# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch

from megatron.core.inference.async_txn import AsyncTxnDiagnostics, StepTxn, TxnRetireQueue


class FakeEvent:
    def __init__(self, done: bool = False):
        self.done = done

    def query(self) -> bool:
        return self.done


def test_step_txn_owns_request_snapshot_and_events():
    ids = [7, 8]
    event = FakeEvent(done=False)
    txn = StepTxn(step_id=3, request_ids=ids, h2d_done_event=event)
    ids.append(9)

    assert txn.request_ids == (7, 8)
    assert not txn.h2d_ready()
    event.done = True
    assert txn.h2d_ready()


def test_step_txn_guard_allows_only_terminal_disappearances():
    txn = StepTxn(step_id=1, request_ids=[1, 2, 3], cuda_graph_key=("decode", 4))
    txn.launched = True

    assert txn.guard_adoption([1, 3], terminal_request_ids=[2], cuda_graph_key=("decode", 4))
    assert not txn.guard_adoption([1, 3], terminal_request_ids=[], cuda_graph_key=("decode", 4))
    assert not txn.guard_adoption([1, 4], terminal_request_ids=[2], cuda_graph_key=("decode", 4))
    assert not txn.guard_adoption([1, 3], terminal_request_ids=[2], cuda_graph_key=("decode", 8))
    assert not txn.guard_adoption([1, 3], terminal_request_ids=[2], decode_only=False)


def test_retire_queue_releases_only_after_event_completion():
    released = []
    first = FakeEvent(done=False)
    second = FakeEvent(done=True)
    diagnostics = AsyncTxnDiagnostics(enabled=True)
    queue = TxnRetireQueue(diagnostics)

    queue.enqueue(first, lambda: released.append("first"))
    queue.enqueue(second, lambda: released.append("second"))
    assert queue.drain_ready() == 0
    assert released == []
    assert diagnostics.retire_queue_depth == 2

    first.done = True
    assert queue.drain_ready() == 2
    assert released == ["first", "second"]
    assert diagnostics.retired == 2
    assert diagnostics.retire_queue_depth == 0


def test_diagnostics_are_noop_when_disabled():
    diagnostics = AsyncTxnDiagnostics(enabled=False)
    diagnostics.record_prepared(under_forward=True)
    diagnostics.record_launched(h2d_ready_before_sampling=True)
    diagnostics.record_adopted()
    diagnostics.record_sync_step("serial_wrapped")
    diagnostics.record_barrier_skip("not_decode_only")
    diagnostics.record_retired()
    diagnostics.record_guard_failure()
    diagnostics.record_launched(sample_to_launch_latency_us=12.0)
    diagnostics.record_commit_duration(34.0)

    assert diagnostics.snapshot() == {
        "enabled": False,
        "prepared": 0,
        "launched": 0,
        "adopted": 0,
        "sync_steps": 0,
        "barrier_skips": 0,
        "retired": 0,
        "guard_failures": 0,
        "prepare_under_forward": 0,
        "h2d_ready_before_sampling": 0,
        "sample_to_launch_latency_us": 0.0,
        "commit_duration_us": 0.0,
        "retire_queue_depth": 0,
        "eligibility_skip_reasons": {},
        "top_skip_reason": None,
    }


def test_serial_wrapped_diagnostics_are_cheap_dict_counters():
    diagnostics = AsyncTxnDiagnostics(enabled=True)
    diagnostics.record_prepared()
    diagnostics.record_adopted()
    diagnostics.record_sync_step("serial_wrapped")

    snapshot = diagnostics.snapshot()
    assert snapshot["prepared"] == 1
    assert snapshot["adopted"] == 1
    assert snapshot["sync_steps"] == 1
    assert snapshot["eligibility_skip_reasons"] == {"serial_wrapped": 1}
    assert torch.tensor(snapshot["prepared"]).item() == 1


def test_diagnostics_record_latest_latency_samples():
    diagnostics = AsyncTxnDiagnostics(enabled=True)

    diagnostics.record_launched(
        h2d_ready_before_sampling=True, sample_to_launch_latency_us=7.25
    )
    diagnostics.record_commit_duration(11.5)

    snapshot = diagnostics.snapshot()
    assert snapshot["launched"] == 1
    assert snapshot["h2d_ready_before_sampling"] == 1
    assert snapshot["sample_to_launch_latency_us"] == 7.25
    assert snapshot["commit_duration_us"] == 11.5
