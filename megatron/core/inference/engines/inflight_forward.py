# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""A typed handle for the single speculative forward F(K+1) launched before commit.

The launch-before-commit decode pipeline launches *one* forward speculatively before the
current step's ``update_requests`` runs. :class:`InflightForward` is the first-class handle for
that forward, backed entirely by the already-built ``StepTxn`` / ``StepTransactionManager``
lifecycle (``prestage_child`` -> ``launch_child`` -> ``adopt_child`` -> ``retire_txn_leases``).
It carries:

* the consume-by-request-id snapshot (``StepTxn.request_ids``),
* the CUDA-graph bucket the forward replayed under (``StepTxn.bucket``),
* the read-only resource lease manifest (KV blocks / reserved boundary blocks / mamba slots,
  carried on the ``StepTxn``),
* the two and only two CUDA fences (``h2d_done_event``, ``forward_done_event``),
* the :class:`AsyncSampleRing` ticket whose D2H this forward's launch overlapped.

It is a *thin facade*: it invents no new lifecycle or policy. ``resolve`` delegates to the
pre-existing ``adopt_child`` consume-by-request-id guard (committed survivors must be a SUBSET
of the snapshot AND the committed graph bucket must match). ``True`` => adopt; ``False`` => the
caller takes the **barrier** recovery (drain the forward, consume its still-valid subset, take
a sync prime step) -- never discard-and-rerun.

It deliberately does NOT reimplement a three-way row-remap: the prestage is identity-+1-only
and bails on any reshape (the ``SPECULATIVE_BOUNDARY_WINDOW`` gate), so a reused forward is
always in identity row order -- the only outcomes are adopt-as-is or barrier.

Extension points (do not implement here):

* **mamba candidate-bank lease.** Hybrid models stay on the serial path (single mamba bank),
  so no new field is added. ``StepTxn`` already carries ``mamba_slot_leases``; a future 2-bank
  port would lease the candidate bank through the same handle without changing this facade.
* **EP step-begin consensus.** The launch decision is a host-local boolean today; a future EP
  port slots an all-reduce-max ``{has_real_work, launch_eligible, in_flight_present}`` consensus
  in front of :meth:`launch` / :meth:`resolve` / :meth:`discard` without re-shaping this object.

Pure addition this commit: no runtime call site uses it yet (the overlap wiring lands next).
"""

from typing import TYPE_CHECKING, Any, Optional

from megatron.core.inference.engines.step_transaction import (
    _UNSET,
    PrestagedDecodePlan,
    StepTransactionManager,
    StepTxn,
)

if TYPE_CHECKING:
    from megatron.core.inference.text_generation_controllers.async_sample_ring import RingTicket


class InflightForward:
    """Typed handle for the in-flight speculative forward F(K+1), backed by a ``StepTxn``."""

    def __init__(
        self,
        mgr: StepTransactionManager,
        txn: StepTxn,
        sample_ticket: Optional["RingTicket"] = None,
    ):
        self.mgr = mgr
        self.txn = txn
        self.sample_ticket = sample_ticket

    @classmethod
    def launch(
        cls,
        mgr: StepTransactionManager,
        plan: PrestagedDecodePlan,
        *,
        bucket: Any = None,
        forward_done_event: Any = None,
        h2d_done_event: Any = None,
        sample_ticket: Optional["RingTicket"] = None,
    ) -> "InflightForward":
        """Prestage + launch the speculative child and return its typed handle.

        ``prestage_child`` seeds the PRESTAGED child from the (eligible) plan; ``launch_child``
        attaches the two fences and transitions it to LAUNCHED. The plan must be eligible
        (``prestage_child`` asserts this via ``StepTxn.from_prestaged_plan``).
        """
        child = mgr.prestage_child(plan, bucket=bucket)
        assert child is not None, "InflightForward.launch requires an eligible prestaged plan"
        txn = mgr.launch_child(forward_done_event=forward_done_event, h2d_done_event=h2d_done_event)
        return cls(mgr, txn, sample_ticket=sample_ticket)

    def resolve(self, committed_survivor_ids: Any, committed_bucket: Any = _UNSET) -> bool:
        """Consume-by-request-id adopt guard: SUBSET(survivors, snapshot) AND bucket match.

        Delegates to ``adopt_child`` -- the system's only adopt guard. ``True`` promotes the
        child to the current transaction; ``False`` leaves it LAUNCHED for the caller's barrier
        recovery (consume the valid subset, sync prime) and bumps ``guard_failures``.
        """
        return self.mgr.adopt_child(committed_survivor_ids, committed_bucket=committed_bucket)

    def discard(self, *, current_step: int, release, event: Any = _UNSET, tag: str = "discard") -> None:
        """Route the forward's leases through the fence-gated retire queue and drop the child.

        Used on suspend / reset / shutdown / empty-batch boundary. The leases are freed only
        after the in-flight forward's ``forward_done_event`` (the default fence) and the
        two-step retire delay, so a still-running forward never has its KV/mamba pulled out from
        under it. Clears ``mgr.child_txn`` so no stale snapshot is later consumed.
        """
        self.mgr.retire_txn_leases(
            self.txn, current_step=current_step, release=release, event=event, tag=tag
        )
        if self.mgr.child_txn is self.txn:
            self.mgr.child_txn = None

    @property
    def snapshot_request_ids(self) -> Any:
        """The consume-by-request-id snapshot (== the launched ``StepTxn.request_ids``)."""
        return self.txn.request_ids

    @property
    def cuda_graph_request_count(self) -> Any:
        """The CUDA-graph bucket the forward replayed under (== ``StepTxn.bucket``)."""
        return self.txn.bucket

    @property
    def forward_done_event(self) -> Any:
        """The fence signalled when the in-flight forward completes."""
        return self.txn.forward_done_event

    @property
    def h2d_done_event(self) -> Any:
        """The fence signalled when the forward's coalesced metadata H2D completes."""
        return self.txn.h2d_done_event
