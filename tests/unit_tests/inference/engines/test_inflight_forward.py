# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""C2 unit tests for :class:`InflightForward` (the thin StepTxn-backed facade).

All pure CPU bookkeeping over ``StepTxn`` objects -- no model, no allocators, no GPU buffer
(mirrors the C5 handoff-lifecycle battery). The bar is the adopt truth table: ``resolve``
is exactly ``adopt_child``'s pre-existing SUBSET + bucket guard (no new policy invented), and
``discard`` routes the forward's leases through the fence-gated retire queue.
"""

import pytest

from megatron.core.inference.engines.inflight_forward import InflightForward
from megatron.core.inference.engines.step_transaction import (
    LaunchGateReason,
    PrestagedDecodePlan,
    StepTransactionManager,
    TxnPhase,
)


class _StubEvent:
    """A CUDA-event stand-in with a controllable ``query()`` for deterministic tests."""

    def __init__(self, ready: bool):
        self.ready = ready
        self.synchronized = False

    def query(self) -> bool:
        return self.ready

    def synchronize(self) -> None:
        self.synchronized = True
        self.ready = True


def _eligible_plan(step_id=3, request_ids=(10, 11, 12)):
    return PrestagedDecodePlan(
        step_id=step_id,
        eligible=True,
        snapshot_request_ids=list(request_ids),
        active_request_count=len(request_ids),
        kv_block_leases=[5, 6, 7],
        reserved_boundary_blocks=[(11, 2)],
        mamba_slot_leases=[1, 2, 3],
        kv_blocks_needed=1,
    )


@pytest.mark.internal
class TestInflightForward:
    """The typed handle for the single speculative forward launched before commit."""

    def test_launch_prestages_and_launches(self) -> None:
        """launch() prestages + launches the child and carries its snapshot/bucket/ticket."""
        mgr = StepTransactionManager(context=None)
        fwd, h2d = _StubEvent(ready=False), _StubEvent(ready=True)
        inflight = InflightForward.launch(
            mgr,
            _eligible_plan(),
            bucket=("decode", 4),
            forward_done_event=fwd,
            h2d_done_event=h2d,
            sample_ticket="ticket-K",
        )
        assert inflight.txn is mgr.child_txn
        assert inflight.txn.phase is TxnPhase.LAUNCHED
        assert inflight.snapshot_request_ids == [10, 11, 12]
        assert inflight.cuda_graph_request_count == ("decode", 4)
        assert inflight.forward_done_event is fwd
        assert inflight.h2d_done_event is h2d
        assert inflight.sample_ticket == "ticket-K"
        assert mgr.diagnostics.prepared == 1 and mgr.diagnostics.launched == 1

    def test_launch_requires_eligible_plan(self) -> None:
        """An ineligible plan cannot back an in-flight forward."""
        mgr = StepTransactionManager(context=None)
        ineligible = PrestagedDecodePlan(
            step_id=1, eligible=False, skip_reason=LaunchGateReason.RESUME
        )
        with pytest.raises(AssertionError):
            InflightForward.launch(mgr, ineligible)

    def test_resolve_true_exact_survivors(self) -> None:
        """survivors == snapshot AND bucket match -> adopted; child promoted to current_txn."""
        mgr = StepTransactionManager(context=None)
        inflight = InflightForward.launch(mgr, _eligible_plan(), bucket=("decode", 4))
        assert inflight.resolve([10, 11, 12], committed_bucket=("decode", 4)) is True
        assert inflight.txn.phase is TxnPhase.ADOPTED
        assert mgr.current_txn is inflight.txn and mgr.child_txn is None
        assert mgr.diagnostics.adopted == 1 and mgr.diagnostics.guard_failures == 0

    def test_resolve_true_subset_midbatch_finish(self) -> None:
        """A finish removes a row (survivors strict subset) -> still adopted, no rerun."""
        mgr = StepTransactionManager(context=None)
        inflight = InflightForward.launch(mgr, _eligible_plan(request_ids=(10, 11, 12)))
        assert inflight.resolve([10, 12]) is True  # request 11 finished mid-batch
        assert mgr.diagnostics.adopted == 1 and mgr.diagnostics.guard_failures == 0

    def test_resolve_false_superset_barrier(self) -> None:
        """A survivor absent from the snapshot -> guard failure (no promotion, no rerun)."""
        mgr = StepTransactionManager(context=None)
        inflight = InflightForward.launch(mgr, _eligible_plan(request_ids=(1, 2)))
        assert inflight.resolve([1, 2, 3]) is False  # request 3 never computed by the forward
        assert mgr.diagnostics.guard_failures == 1 and mgr.diagnostics.adopted == 0
        # Recovery is a barrier handled by the caller; the child stays LAUNCHED, not discarded.
        assert mgr.current_txn is None
        assert mgr.child_txn is inflight.txn and inflight.txn.phase is TxnPhase.LAUNCHED

    def test_resolve_false_bucket_mismatch(self) -> None:
        """A graph-bucket mismatch trips the guard even when survivors are a subset."""
        mgr = StepTransactionManager(context=None)
        inflight = InflightForward.launch(mgr, _eligible_plan(request_ids=(1, 2)), bucket=("decode", 4))
        assert inflight.resolve([1, 2], committed_bucket=("decode", 8)) is False
        assert mgr.diagnostics.guard_failures == 1 and mgr.diagnostics.adopted == 0

    def test_resolve_default_bucket_skips_check(self) -> None:
        """With no committed_bucket supplied, the bucket check is skipped (subset only)."""
        mgr = StepTransactionManager(context=None)
        inflight = InflightForward.launch(mgr, _eligible_plan(request_ids=(1, 2)), bucket=("decode", 4))
        assert inflight.resolve([1, 2]) is True  # bucket not supplied -> not checked
        assert mgr.diagnostics.adopted == 1

    def test_discard_enqueues_leases_with_forward_fence(self) -> None:
        """discard routes the leases through retire_txn_leases gated by the forward fence."""
        mgr = StepTransactionManager(context=None)
        fwd = _StubEvent(ready=False)
        inflight = InflightForward.launch(
            mgr, _eligible_plan(), forward_done_event=fwd, h2d_done_event=_StubEvent(ready=True)
        )
        freed = []
        inflight.discard(current_step=5, release=lambda: freed.append("kv"))
        # Child cleared so no stale snapshot is consumed later.
        assert mgr.child_txn is None
        # Held until BOTH the two-step delay and the forward fence are satisfied.
        assert mgr.retire(5) == 0 and freed == []  # neither satisfied
        assert mgr.retire(7) == 0 and freed == []  # delay ok, fence not ready
        fwd.ready = True
        assert mgr.retire(7) == 1 and freed == ["kv"]  # both satisfied
        assert mgr.diagnostics.retired == 1

    def test_discard_event_override(self) -> None:
        """An explicit fence overrides the txn's forward fence (coalesced multi-resource free)."""
        mgr = StepTransactionManager(context=None)
        inflight = InflightForward.launch(
            mgr, _eligible_plan(), forward_done_event=_StubEvent(ready=False)
        )
        override = _StubEvent(ready=True)
        freed = []
        inflight.discard(current_step=0, release=lambda: freed.append("slot"), event=override)
        assert mgr.retire(2) == 1 and freed == ["slot"]

    def test_facade_is_thin_no_extra_state(self) -> None:
        """The facade holds only (mgr, txn, sample_ticket): properties read straight through."""
        mgr = StepTransactionManager(context=None)
        inflight = InflightForward.launch(mgr, _eligible_plan(), bucket=("decode", 2))
        assert set(vars(inflight)) == {"mgr", "txn", "sample_ticket"}
        assert inflight.snapshot_request_ids is inflight.txn.request_ids
        assert inflight.cuda_graph_request_count is inflight.txn.bucket
