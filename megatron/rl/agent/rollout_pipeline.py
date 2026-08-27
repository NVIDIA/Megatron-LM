# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Caller-owned orchestration of grouped rollout generation over an agent."""

import asyncio
import logging
import queue as thread_queue
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, AsyncIterator, Iterable, NamedTuple

import numpy as np

from megatron.core.inference.utils import asyncio_Queue, asyncio_QueueShutDown
from megatron.core.utils import trace_async_exceptions

from ..inference import ReturnsRaw
from ..rollout_bank import _PendingProblem, _PendingRollout
from ..rollout_granularity import GRANULARITY_RANK, ConsumptionGranularity, SubmissionGranularity
from .api import EpisodeResult, GroupedRolloutRequest, GroupRolloutParams, RolloutGroup

logger = logging.getLogger(__name__)

BANK_WRITE_MAX_RECORDS = 64


class _PendingConsumedMarkers(NamedTuple):
    """A consumption-marker batch riding the bank queue behind its records.

    FIFO order through the single writer thread is what guarantees the marker
    reaches disk only after every record enqueued before it — the invariant a
    blocking ``drain_bank`` used to provide at the inference-to-training switch.
    """

    uids: tuple[str, ...]
    iteration: int

if TYPE_CHECKING:
    from ..rollout_bank import RolloutBank
    from .api import GroupedRolloutGenerator, Rollout, TokenRollout


class _GranularityConfig(NamedTuple):
    submission: SubmissionGranularity
    consumption: ConsumptionGranularity
    num_groups_per_batch: int
    rollouts_per_group: int
    num_groups_per_env: tuple[int, ...]

    @classmethod
    def from_request(
        cls, request: GroupedRolloutRequest, num_groups_per_env: list[int]
    ) -> "_GranularityConfig":
        """Build the per-request granularity policy.

        Args:
            request: Grouped rollout request carrying the granularity choices.
            num_groups_per_env: Groups each env contributes to one batch, in env order.

        Returns:
            A validated _GranularityConfig.
        """
        cls._validate(request, num_groups_per_env)
        return cls(
            submission=request.submission_granularity,
            consumption=request.consumption_granularity,
            num_groups_per_batch=request.num_groups,
            rollouts_per_group=request.rollouts_per_group,
            num_groups_per_env=tuple(num_groups_per_env),
        )

    def env_of_index(self, index_in_batch: int) -> int:
        """Map a batch slot to the env owning it (slots are env-blocked, in env order).

        Args:
            index_in_batch: Slot index within one trainer batch.

        Returns:
            The env_index owning the slot.
        """
        boundary = 0
        for env_index, groups in enumerate(self.num_groups_per_env):
            boundary += groups
            if index_in_batch < boundary:
                return env_index
        raise IndexError(
            f"index_in_batch {index_in_batch} outside batch of {self.num_groups_per_batch}"
        )

    def units_per_batch(self) -> int:
        """Submission units in one batch; gate capacity = depth-in-batches x this.

        Returns:
            The number of submission units one trainer batch contains.
        """
        return {
            "R": self.num_groups_per_batch * self.rollouts_per_group,
            "G": self.num_groups_per_batch,
            "B": 1,
        }[self.submission]

    @staticmethod
    def _validate(request: GroupedRolloutRequest, num_groups_per_env: list[int]) -> None:
        """Reject invalid granularity and layout combinations.

        Args:
            request: Grouped rollout request to check.
            num_groups_per_env: Constant per-env group layout.
        """
        assert (
            GRANULARITY_RANK[request.consumption_granularity]
            >= GRANULARITY_RANK[request.submission_granularity]
        ), (
            f"Consumption granularity ({request.consumption_granularity}) must be no finer "
            f"than submission granularity ({request.submission_granularity})."
        )
        assert all(
            groups > 0 for groups in num_groups_per_env
        ), "Each environment must request at least one group per batch."
        assert (
            sum(num_groups_per_env) == request.num_groups
        ), "The sum of groups per environment must equal the total number of groups requested."


class _SubmissionGate:
    """Gate capacity is measured in units of the configured submission granularity.

    Each granularity has a single release point: R slots free when inference
    completes, so the gate bounds engine concurrency in rollouts. G and B
    slots free when the trainer consumes the group/batch, so the gate
    enforces the --rl-generation-lag run-ahead cap in groups/batches
    respectively.
    """

    def __init__(
        self,
        *,
        capacity: int,
        submission: SubmissionGranularity,
    ) -> None:
        """Create a gate with `capacity` slots counted at `submission` granularity.

        Args:
            capacity: Maximum submission units in flight.
            submission: Configured submission granularity.
        """
        self._sem = asyncio.Semaphore(capacity)
        self._submission = submission
        self.capacity = capacity
        # Observability counters, updated only on the configured submission
        # granularity (the only path that touches the semaphore). `held`
        # counts slots currently held; `prepare_blocked_seconds` accumulates
        # time stage_prepare spent waiting on the semaphore.
        self.held = 0
        self.prepare_blocked_seconds = 0.0
        self.acquire_calls = 0
        self.release_calls = 0

    async def acquire_for(self, granularity: SubmissionGranularity) -> None:
        """Take one slot when crossing a boundary of the configured granularity.

        Args:
            granularity: The dispatch boundary being crossed.
        """
        if self._submission == granularity:
            start = time.monotonic()
            await self._sem.acquire()
            self.prepare_blocked_seconds += time.monotonic() - start
            self.held += 1
            self.acquire_calls += 1

    def release_for(self, granularity: SubmissionGranularity) -> None:
        """Release one slot when work at the given granularity reaches its release point.

        Args:
            granularity: The granularity whose release point was just reached.
        """
        if self._submission == granularity:
            self._sem.release()
            self.held -= 1
            self.release_calls += 1


class _InferWorkItem(NamedTuple):
    """One rollout's worth of work flowing from prepare to infer.

    Timestamps are wall-clock monotonic seconds: `prepared_at` is stamped at
    construction and `infer_dequeued_at` is filled in via `_replace` when an
    infer worker dequeues the item. Zero means "not yet reached".

    `bank_uid` is the owning group's durable bank identity, reserved before any
    member exists and None whenever the durable rollout bank is disabled. It is
    distinct from `group_id`, which is the pipeline-local assemble-bucket key and
    is always present.
    """

    group_id: int
    rollout_idx: int
    batch_id: int
    index_in_batch: int
    params: GroupRolloutParams
    env_index: int = 0
    bank_uid: str | None = None
    prepared_at: float = 0.0
    infer_dequeued_at: float = 0.0


class _InferredItem(NamedTuple):
    """One rollout post-inference, flowing from infer to assemble."""

    item: _InferWorkItem
    episode: EpisodeResult
    inferred_at: float = 0.0


class RolloutPipeline:
    """Orchestrates grouped rollout generation over an agent, one instance per request.

    Constructed and driven by the caller (e.g. the trainer via run());
    the agent only supplies the env allocations, per-group preparation, and inference calls.
    """

    def __init__(
        self,
        agent: "GroupedRolloutGenerator",
        request: GroupedRolloutRequest,
        parallel_generation_tasks: int,
        bank: "RolloutBank | None" = None,
        initial_batch_id: int = 0,
    ) -> None:
        """Validate the request and size the gate, queues, and worker pool.

        Args:
            agent: Agent supplying the env layout, preparation, and inference.
            request: Grouped rollout request to serve; one pipeline per request.
            parallel_generation_tasks: Submission gate depth in trainer batches.
            bank: Optional durable store for freshly completed rollout groups.
            initial_batch_id: Batch ID assigned to the first batch in this pipeline.
        """
        assert isinstance(
            request.inference_interface, ReturnsRaw
        ), "InferenceInterface must support raw_text return to provide rollouts."
        self.agent = agent
        self.request = request
        self.bank = bank
        self.initial_batch_id = initial_batch_id
        self.allocations = agent.rollout_allocations(request.num_groups)
        self.gran_policy = _GranularityConfig.from_request(
            request, [allocation.num_groups for allocation in self.allocations]
        )
        # Lag may be fractional or negative (>= -1): clamp and round the slot math at granularity.
        self.gate = _SubmissionGate(
            capacity=max(
                1, round(parallel_generation_tasks * self.gran_policy.units_per_batch())
            ),
            submission=self.gran_policy.submission,
        )
        rollouts_per_batch = self.gran_policy.num_groups_per_batch * request.rollouts_per_group
        self.num_infer_workers = self.gate.capacity * (
            rollouts_per_batch // self.gran_policy.units_per_batch()
        )

        # Core queues.
        self.infer_queue = asyncio_Queue()
        self.assemble_queue = asyncio_Queue()
        self.output_queue = asyncio_Queue()
        # The bank queue is a plain thread queue, not an asyncio queue: its
        # consumer must keep writing while the trainer holds the loop thread
        # inside a training step and nothing pumps the event loop.
        self.bank_queue: thread_queue.Queue = thread_queue.Queue()
        self._bank_error: Exception | None = None
        self._bank_writer: threading.Thread | None = None
        if bank is not None:
            self._bank_writer = threading.Thread(
                target=self._bank_writer_loop, name="rollout-bank-writer", daemon=True
            )
            self._bank_writer.start()

        # Track regenerated groups and tasks for proper shutdown.
        self._next_regen_group_id = -1
        self._prepare_done = False
        self._groups_in_flight = 0
        self._regen_tasks: set[asyncio.Task] = set()

        # Buffers of partial results.
        self._assemble_pending: dict[int, dict[int, RolloutGroup]] = {}
        self._consume_pending: dict[int, list[RolloutGroup]] = {}
        self._output_enqueued_at: dict[tuple[int, int], float] = {}

        # Observability accumulators.
        self.infer_queue_dwell: list[float] = []
        self.engine_dwell: list[float] = []
        self.assemble_queue_dwell: list[float] = []
        self.output_queue_dwell: list[float] = []
        self.prepared_count = 0
        self.inferred_count = 0
        self.assembled_count = 0
        # All drops, any reason; the two counters below attribute them for metrics.
        self.dropped_count = 0
        self.filtered_count = 0
        self.refilled_placeholder_groups = 0
        self.restored_count = 0
        self.restored_rollout_count = 0
        self.yielded_count = 0
        self.prepared_groups_per_env = [0] * len(self.gran_policy.num_groups_per_env)
        self.assembled_groups_per_env = [0] * len(self.gran_policy.num_groups_per_env)
        self.yielded_groups_per_env = [0] * len(self.gran_policy.num_groups_per_env)
        self.refill_failure_reasons: dict[str, int] = {}
        self._lifetime_refilled_groups = 0

    async def run(self) -> AsyncIterator[RolloutGroup]:
        """Run the pipeline stages; cancels them when the iterator is closed.

        Yields:
            RolloutGroup: Groups in consumption-granularity order.
        """
        tasks = [
            asyncio.create_task(self.stage_prepare()),
            asyncio.create_task(self.stage_infer()),
            asyncio.create_task(self.stage_assemble()),
        ]
        try:
            async for group in self.stage_consume():
                yield group
            for task in tasks:
                task.cancel()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            failure = next(
                (
                    result
                    for result in results
                    if isinstance(result, BaseException)
                    and not isinstance(result, asyncio.CancelledError)
                ),
                None,
            )
            raise RuntimeError(
                "RolloutPipeline output stream ended: a pipeline stage died"
                + ("" if failure is not None else " (no stage exception was recovered)")
            ) from failure
        finally:
            regen_tasks = tuple(self._regen_tasks)
            for task in (*tasks, *regen_tasks):
                task.cancel()
            await asyncio.gather(*tasks, *regen_tasks, return_exceptions=True)

    async def _submit_group_to_infer_queue(
        self,
        *,
        group_id: int,
        batch_id: int,
        index_in_batch: int,
        restored: RolloutGroup | None = None,
    ) -> None:
        """Enqueue one group's inference items, acquiring per-rollout gate slots.

        Args:
            group_id: Pipeline-local key for this group's assemble bucket.
            batch_id: Trainer batch this group belongs to.
            index_in_batch: Slot within that batch.
            restored: An incomplete group recovered from the bank. Its members seed
                the assemble bucket and only its missing slots are generated, using
                the persisted problem state so they answer the same prompt.
        """
        env_index = self.gran_policy.env_of_index(index_in_batch)
        agent = self.allocations[env_index].agent
        params: GroupRolloutParams = await agent.prepare_group_rollout(
            self.request, problem_state=restored.problem_state if restored else None
        )
        self.prepared_groups_per_env[env_index] += 1

        if restored is not None:
            bank_uid = restored.uid
            indices = restored.missing_indices(self.request.rollouts_per_group)
            self._assemble_pending[group_id] = dict(
                zip(restored.member_indices, restored.rollouts, strict=True)
            )
            self.restored_rollout_count += len(restored.rollouts)
        else:
            bank_uid = self.bank.reserve_group_uid() if self.bank is not None else None
            indices = list(range(self.request.rollouts_per_group))
            if self.bank is not None and params.problem_state is not None:
                self.bank_queue.put_nowait(_PendingProblem(bank_uid, params.problem_state))

        for rollout_idx in indices:
            await self.gate.acquire_for("R")
            item = _InferWorkItem(
                group_id=group_id,
                rollout_idx=rollout_idx,
                batch_id=batch_id,
                index_in_batch=index_in_batch,
                params=params,
                env_index=env_index,
                bank_uid=bank_uid,
                prepared_at=time.monotonic(),
            )
            await self.infer_queue.put(item)
            self.prepared_count += 1

    def _maybe_close_intake(self) -> None:
        """Shut down infer_queue once no work can ever be submitted again."""
        if self._prepare_done and self._groups_in_flight <= 0:
            self.infer_queue.shutdown()

    def assert_no_inflight_rollouts(self) -> None:
        """Verify no rollouts are buffered or in-flight at an iteration boundary for lag=0.

        Counts rollouts rather than groups: dropped groups consume prepared
        rollouts without ever being yielded, and a partially restored group is
        yielded having consumed only some of its members from the gate.
        """
        buffered = {
            "infer_queue": self.infer_queue.qsize(),
            "assemble_queue": self.assemble_queue.qsize(),
            "output_queue": self.output_queue.qsize(),
            "assemble_pending": sum(len(items) for items in self._assemble_pending.values()),
            "consume_pending": sum(len(groups) for groups in self._consume_pending.values()),
            "regen_pending": sum(1 for task in self._regen_tasks if not task.done()),
        }
        assert not any(buffered.values()), (
            f"The rollout pipeline has buffered rollouts at iteration boundary: {buffered}. "
            "The generator has run ahead under a stale policy."
        )
        yielded_rollouts = (
            self.yielded_count * self.request.rollouts_per_group - self.restored_rollout_count
        )
        in_flight = self.prepared_count - (
            yielded_rollouts + self.dropped_count * self.request.rollouts_per_group
        )
        assert in_flight == 0, (
            f"The rollout pipeline prepared {self.prepared_count} rollout(s) but yielded "
            f"{self.yielded_count} group(s) ({self.restored_rollout_count} restored member(s)) "
            f"and dropped {self.dropped_count} group(s) of {self.request.rollouts_per_group} "
            f"({self.filtered_count} same-reward-filtered, "
            f"{self.refilled_placeholder_groups} placeholder-refilled); "
            f"{in_flight} rollout(s) in flight at iteration boundary."
        )

    async def stage_prepare(self) -> None:
        """Generate gated inference work items."""
        group_id = 0
        batch_id = self.initial_batch_id
        try:
            while True:
                await self.gate.acquire_for("B")

                for index_in_batch in range(self.gran_policy.num_groups_per_batch):
                    await self.gate.acquire_for("G")
                    self._groups_in_flight += 1
                    env_index = self.gran_policy.env_of_index(index_in_batch)
                    allocation = self.allocations[env_index]
                    restored: RolloutGroup | None = self.agent.take_restored_group(allocation.env_id)
                    if restored is not None:
                        assert all(rollout.env_id == allocation.env_id for rollout in restored), (
                            f"Restored rollout group routed to env {allocation.env_id!r} contains "
                            f"members for {[rollout.env_id for rollout in restored]}"
                        )
                        if restored.is_complete(self.request.rollouts_per_group):
                            restored.batch_id = batch_id
                            restored.index_in_batch = index_in_batch
                            self._output_enqueued_at[(batch_id, index_in_batch)] = time.monotonic()
                            await self.output_queue.put(restored)
                            self.restored_count += 1
                            self.restored_rollout_count += len(restored.rollouts)
                            self._groups_in_flight -= 1
                            self._maybe_close_intake()
                            group_id += 1
                            continue
                        await self._submit_group_to_infer_queue(
                            group_id=group_id,
                            batch_id=batch_id,
                            index_in_batch=index_in_batch,
                            restored=restored,
                        )
                        group_id += 1
                        continue
                    await self._submit_group_to_infer_queue(
                        group_id=group_id, batch_id=batch_id, index_in_batch=index_in_batch
                    )
                    group_id += 1
                batch_id += 1
        except BaseException:
            self.infer_queue.shutdown()
            raise
        finally:
            self._prepare_done = True
            self._maybe_close_intake()

    def _bank_writer_loop(self) -> None:
        """Consume the bank queue on a dedicated thread, for the pipeline's lifetime.

        The trainer only runs the asyncio loop while collecting rollouts, so a
        loop-driven consumer would freeze for the whole training step and force
        the inference-to-training switch to drain the backlog with the GPUs
        idle. This thread keeps writing regardless of what the loop is doing.

        A single consumer means only one write is ever in flight, which is what
        lets the bank keep its running sidecar offsets without a lock. All
        failures are latched, never raised, so the thread cannot die and a
        ``bank_queue.join()`` cannot deadlock.
        """
        while True:
            items = [self.bank_queue.get()]
            while len(items) < BANK_WRITE_MAX_RECORDS:
                try:
                    items.append(self.bank_queue.get_nowait())
                except thread_queue.Empty:
                    break
            try:
                self._process_bank_items(items)
            finally:
                for _ in items:
                    self.bank_queue.task_done()

    def _process_bank_items(self, items: list) -> None:
        """Apply queued items in FIFO order: coalesced record writes, markers in place.

        A marker batch flushes the records queued before it first, so a marker
        can never reach disk ahead of a record it names.
        """
        records: list = []
        for item in items:
            if isinstance(item, _PendingConsumedMarkers):
                self._write_records_latching(records)
                records = []
                self._mark_consumed_latching(item)
            else:
                records.append(item)
        self._write_records_latching(records)

    def _write_records_latching(self, records: list) -> None:
        """Write one set of records, recording any failure instead of raising.

        Latching keeps the writer thread alive through a bad write; the failure
        surfaces at the next durability barrier or marker enqueue instead.
        """
        if not records:
            return
        try:
            self.bank.write_records(records)
        except Exception as exc:
            self._bank_error = exc
            logger.exception("Rollout bank write failed; %d record(s) lost", len(records))

    def _mark_consumed_latching(self, markers: _PendingConsumedMarkers) -> None:
        """Apply one consumption-marker batch, recording any failure instead of raising."""
        try:
            self.bank.mark_consumed_many(markers.uids, markers.iteration)
        except Exception as exc:
            self._bank_error = exc
            logger.exception(
                "Rollout bank consumption-marker write failed; %d marker(s) lost",
                len(markers.uids),
            )

    def _raise_latched_bank_error(self) -> None:
        """Raise the last latched bank failure once, then clear it."""
        if self._bank_error is not None:
            error, self._bank_error = self._bank_error, None
            raise error

    def enqueue_consumed_markers(self, uids: Iterable[str | None], iteration: int) -> None:
        """Queue consumption markers behind the records they name, without blocking.

        The single FIFO writer guarantees the markers reach disk only after
        every record enqueued before them — the same records-before-marker
        invariant a blocking drain used to enforce at the inference-to-training
        switch, minus the idle GPUs. A crash mid-queue loses the tail records
        and their marker together, which recovery already treats as "never
        generated": the groups regenerate.

        Raises any bank failure latched since the last barrier, so a broken
        disk surfaces within one training iteration.
        """
        if self.bank is None:
            return
        self._raise_latched_bank_error()
        markers = _PendingConsumedMarkers(
            uids=tuple(uid for uid in uids if uid), iteration=iteration
        )
        if not markers.uids:
            return
        self.bank_queue.put(markers)
        if self._bank_writer is None or not self._bank_writer.is_alive():
            self.drain_bank()

    def drain_bank(self) -> None:
        """Block until every queued bank record and marker is durable.

        The barrier before segment switches and compaction, both of which
        rewrite state that queued items would be written against. Needs no
        event loop: the writer runs on its own thread, and has had the whole
        training step to work through the backlog, so this is normally instant.

        If the writer thread is unavailable (it never started, or the
        interpreter is tearing it down), the remaining items are applied inline
        instead of dropped.
        """
        if self.bank is None:
            return
        if self._bank_writer is not None and self._bank_writer.is_alive():
            self.bank_queue.join()
        else:
            pending = []
            while True:
                try:
                    pending.append(self.bank_queue.get_nowait())
                except thread_queue.Empty:
                    break
            try:
                self._process_bank_items(pending)
            finally:
                for _ in pending:
                    self.bank_queue.task_done()
        self._raise_latched_bank_error()

    async def stage_infer(self) -> None:
        """Run a persistent pool of inference workers, spawned once per pipeline."""
        workers = [
            asyncio.create_task(self._infer_worker()) for _ in range(self.num_infer_workers)
        ]
        try:
            await asyncio.gather(*workers, return_exceptions=True)
        finally:
            for worker in workers:
                worker.cancel()
            self.assemble_queue.shutdown()

    async def _infer_worker(self) -> None:
        while True:
            try:
                item = await self.infer_queue.get()
            except asyncio_QueueShutDown:
                return
            item = item._replace(infer_dequeued_at=time.monotonic())
            if item.prepared_at:
                self.infer_queue_dwell.append(item.infer_dequeued_at - item.prepared_at)
            await self._infer_one(item)

    @trace_async_exceptions(verbose=True)
    async def _infer_one(self, item: _InferWorkItem) -> None:
        """Run one episode for one work item and hand the result to assemble.

        Args:
            item: The dequeued work item; its params carry the episode closure.
        """
        episode = await item.params.run_episode()
        inferred_at = time.monotonic()
        self.gate.release_for("R")
        if item.infer_dequeued_at:
            self.engine_dwell.append(inferred_at - item.infer_dequeued_at)
        self.inferred_count += 1
        await self.assemble_queue.put(
            _InferredItem(item=item, episode=episode, inferred_at=inferred_at)
        )

    async def stage_assemble(self) -> None:
        """Grade each rollout as it arrives, emit groups once they fill, and
        regenerate dropped ones.

        Grading moved here from group completion so a member can be persisted the
        moment it is durable-worthy. It does not run more often: every rollout is
        graded either way, because the drop decision needs the rewards.

        Placeholders are not persisted. A failed episode has an empty trajectory
        and no decode work to save, and banking one would let a restored group
        carry a placeholder past the refill path in ``_decide_drop``; leaving the
        slot empty means restore regenerates it instead.
        """
        pending = self._assemble_pending
        try:
            while True:
                try:
                    inferred = await self.assemble_queue.get()
                except asyncio_QueueShutDown:
                    break
                dequeued_at = time.monotonic()
                if inferred.inferred_at:
                    self.assemble_queue_dwell.append(dequeued_at - inferred.inferred_at)

                item = inferred.item
                rollout = await item.params.build_rollout(inferred.episode)
                if (
                    self.bank is not None
                    and item.bank_uid is not None
                    and not rollout.is_placeholder
                ):
                    self.bank_queue.put_nowait(
                        _PendingRollout(item.bank_uid, item.rollout_idx, rollout)
                    )

                bucket = pending.setdefault(item.group_id, {})
                bucket[item.rollout_idx] = rollout
                if len(bucket) < self.request.rollouts_per_group:
                    continue

                pending.pop(item.group_id)
                indices = sorted(bucket)
                self.assembled_count += 1
                self.assembled_groups_per_env[item.env_index] += 1
                group = RolloutGroup(
                    rollouts=[bucket[index] for index in indices],
                    batch_id=item.batch_id,
                    index_in_batch=item.index_in_batch,
                    uid=item.bank_uid,
                    member_indices=indices,
                )
                if self._decide_drop(group):
                    self.dropped_count += 1
                    task = asyncio.create_task(
                        self._regenerate_group(group.batch_id, group.index_in_batch)
                    )
                    self._regen_tasks.add(task)
                    task.add_done_callback(self._regen_tasks.discard)
                    continue
                self._output_enqueued_at[(group.batch_id, group.index_in_batch)] = (
                    time.monotonic()
                )
                await self.output_queue.put(group)
                self._groups_in_flight -= 1
                self._maybe_close_intake()
        finally:
            self.output_queue.shutdown()

    @staticmethod
    def _is_refillable_placeholder(rollout: "Rollout | TokenRollout") -> bool:
        """A failed episode's placeholder, eligible for a refill."""
        return rollout.is_placeholder and rollout.rollout_status in ('ok', 'placeholder')

    def _decide_drop(self, group: RolloutGroup) -> bool:
        """Decide whether to drop this group instead of delivering it."""
        if all(self._is_refillable_placeholder(rollout) for rollout in group.rollouts):
            self.refilled_placeholder_groups += 1
            self._lifetime_refilled_groups += 1
            reasons: dict[str, int] = {}
            for rollout in group.rollouts:
                if rollout.failure_reason:
                    reasons[rollout.failure_reason] = reasons.get(rollout.failure_reason, 0) + 1
                    self.refill_failure_reasons[rollout.failure_reason] = (
                        self.refill_failure_reasons.get(rollout.failure_reason, 0) + 1
                    )
            if self._lifetime_refilled_groups == 1 or self._lifetime_refilled_groups % 32 == 0:
                logger.warning(
                    "Refilling all-placeholder group (batch %s slot %s): %d refilled over this "
                    "pipeline's lifetime%s. Groups whose episodes all failed are regenerated; "
                    "a climbing count means a likely failure upstream of the pipeline.",
                    group.batch_id,
                    group.index_in_batch,
                    self._lifetime_refilled_groups,
                    f" (member failure reasons: {sorted(reasons.items())})" if reasons else "",
                )
            return True
        if self.request.filter_groups_with_same_reward:
            real_rewards = [
                rollout.reward for rollout in group.rollouts if not rollout.is_placeholder
            ]
            if real_rewards and np.std(real_rewards) <= 1e-6:
                self.filtered_count += 1
                return True
        return False

    @trace_async_exceptions(verbose=True)
    async def _regenerate_group(self, batch_id: int, index_in_batch: int) -> None:
        """Resubmit a replacement group for a dropped group's batch slot."""
        group_id = self._next_regen_group_id
        self._next_regen_group_id -= 1
        try:
            await self._submit_group_to_infer_queue(
                group_id=group_id, batch_id=batch_id, index_in_batch=index_in_batch
            )
        except asyncio_QueueShutDown:
            # Intake closed mid-regeneration (teardown or other failure).
            self.gate.release_for("G")
            self._groups_in_flight -= 1
            self._maybe_close_intake()
        except BaseException:
            self.infer_queue.shutdown()
            raise

    def _record_output_dwell(self, group: RolloutGroup) -> None:
        """Record how long a group sat in output_queue before being yielded."""
        key = (group.batch_id, group.index_in_batch)
        enqueued_at = self._output_enqueued_at.pop(key, 0.0)
        if enqueued_at:
            self.output_queue_dwell.append(time.monotonic() - enqueued_at)
        self.yielded_count += 1
        self.yielded_groups_per_env[self.gran_policy.env_of_index(group.index_in_batch)] += 1

    async def _next_complete_group(self) -> RolloutGroup | None:
        """Pop the next group off output_queue and record its dwell."""
        try:
            group = await self.output_queue.get()
        except asyncio_QueueShutDown:
            return None
        self._record_output_dwell(group)
        return group

    async def stage_consume(self) -> AsyncIterator[RolloutGroup]:
        """Deliver groups in the order defined by the consumption granularity."""
        consume = {
            "G": self._consume_completion_order,
            "B": self._consume_batch_order,
        }[self.gran_policy.consumption]
        async for group in consume():
            yield group

    async def _consume_completion_order(self) -> AsyncIterator[RolloutGroup]:
        """G consumption: deliver groups in completion order, balanced across envs."""
        groups_per_env_per_batch = self.gran_policy.num_groups_per_env
        pending_groups_by_env: list[deque[RolloutGroup]] = [
            deque() for _ in groups_per_env_per_batch
        ]
        delivered_groups_by_env = [0] * len(groups_per_env_per_batch)
        while (group := await self._next_complete_group()) is not None:
            env_index = self.gran_policy.env_of_index(group.index_in_batch)
            pending_groups_by_env[env_index].append(group)
            yielded_any = True
            while yielded_any:
                yielded_any = False
                for env, queue in enumerate(pending_groups_by_env):
                    if queue and delivered_groups_by_env[env] < groups_per_env_per_batch[env]:
                        yield queue.popleft()
                        self.gate.release_for("G")
                        delivered_groups_by_env[env] += 1
                        yielded_any = True
                if all(
                    count == quota
                    for count, quota in zip(delivered_groups_by_env, groups_per_env_per_batch)
                ):
                    delivered_groups_by_env = [0] * len(groups_per_env_per_batch)
        # The stream is over; nothing is left to balance against, so drain any pending groups.
        for queue in pending_groups_by_env:
            while queue:
                yield queue.popleft()
                self.gate.release_for("G")

    async def _consume_batch_order(self) -> AsyncIterator[RolloutGroup]:
        """B consumption: deliver whole batches in dataset order."""
        next_batch_id = self.initial_batch_id
        pending = self._consume_pending
        while (group := await self._next_complete_group()) is not None:
            pending.setdefault(group.batch_id, []).append(group)
            while (
                len(pending.get(next_batch_id, []))
                >= self.gran_policy.num_groups_per_batch
            ):
                batch = pending.pop(next_batch_id)
                batch.sort(key=lambda group: group.index_in_batch)
                next_batch_id += 1
                for group in batch:
                    yield group
                    self.gate.release_for("G")
                self.gate.release_for("B")
