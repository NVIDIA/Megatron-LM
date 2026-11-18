# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""One-batch lookahead for a CPU-only, leader-owned DCP planner."""

import atexit
import copy
import hashlib
import itertools
import json
import logging
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections import deque
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from megatron.core.datasets import dcp_runtime_worker
from megatron.core.datasets.data_schedule_utils import get_batch_and_global_seqlens
from megatron.core.datasets.dcp_runtime_scheduler.models.solver_section import pipeline_schedule
from megatron.core.datasets.dcp_runtime_scheduler.runtime.config import (
    SCHEDULERS,
    load_runtime_config,
)
from megatron.core.datasets.dcp_runtime_scheduler.runtime.planner import (
    MAX_PRICED_GROUP_SEQUENCES,
    RuntimePlanGeometry,
    RuntimeScheduleError,
    longest_priced_sample,
)
from megatron.core.packed_seq_params import resolve_thd_tail_padding_policy
from megatron.core.utils import log_single_rank

if TYPE_CHECKING:
    from megatron.core.model_parallel_config import ModelParallelConfig
    from megatron.core.rerun_state_machine import RerunDataIterator

logger = logging.getLogger(__name__)


# Seconds the DPxCP leader waits for the CPU worker's reply to one batch before every rank
# raises an error. A plan takes at most tens of seconds (dynamic_usp, thousands of samples)
# and loading a cost model a few; the limit only has to tell a busy worker from one that
# will never answer. It is below the timeout of the process groups (10 minutes by default,
# --distributed-timeout-minutes), so that the other ranks get this error first.
PLANNER_REPLY_TIMEOUT_S = 300.0


class _CpuPlanner:
    """Isolated executable: never re-import the GPU training entry point via spawn."""

    # Seconds to wait for the worker after closing its request stream. A process forked
    # later (a dataloader worker) can keep the pipe open, so EOF alone is not reliable.
    close_timeout_s = 10.0
    in_flight = False

    def __init__(self, spec: dict) -> None:
        self.process = subprocess.Popen(
            [sys.executable, '-u', str(Path(__file__).with_name('dcp_runtime_worker.py'))],
            env={
                **os.environ,
                'CUDA_VISIBLE_DEVICES': '',
                'OMP_NUM_THREADS': '1',
                'OPENBLAS_NUM_THREADS': '1',
            },
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.process.stdin.write(json.dumps(spec) + '\n')
        self.process.stdin.flush()

    def solve(self, lengths: list[int], padded: list[int], baseline_rows: list) -> dict:
        """Send one global batch to the worker (from the I/O thread); return or raise its reply."""
        self.in_flight = True
        try:
            try:
                self.process.stdin.write(json.dumps([lengths, padded, baseline_rows]) + '\n')
                self.process.stdin.flush()
            except BrokenPipeError:
                # Initialization failures already wrote an error response before exiting.
                pass
            response = self.process.stdout.readline()
        finally:
            self.in_flight = False
        if not response:
            raise RuntimeError(f'CPU DCP worker exited unexpectedly: {self.process.poll()}')
        payload = json.loads(response)
        if payload.get('schedule_error') is not None:
            raise RuntimeScheduleError(payload['schedule_error'])
        if 'error' in payload:
            raise RuntimeError(payload['error'])
        return payload

    def stop(self) -> None:
        """End the worker: EOF, then kill after a timeout; at once when a request is in flight."""
        if self.in_flight:
            self.process.kill()
        try:
            self.process.stdin.close()
        except (OSError, ValueError):
            pass
        try:
            self.process.wait(timeout=self.close_timeout_s)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()

    def close(self) -> None:
        """Close the request stream and reap the CPU worker process."""
        self.stop()
        self.process.stdout.close()


class _DaemonExecutor:
    """Run submitted calls one at a time on a daemon thread; ``submit`` returns a Future.

    concurrent.futures joins its threads before atexit callbacks run, so a thread that waits
    for the CPU worker would keep the interpreter from reaching ``close``; a daemon does not.
    """

    # Seconds ``shutdown(wait=True)`` waits for the running call before leaving it behind.
    join_timeout_s = 10.0

    def __init__(self, name: str) -> None:
        self._calls = queue.SimpleQueue()
        self._closed = False
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(self, function, *args) -> Future:
        """Queue one call; its result or exception is delivered through the Future."""
        if self._closed:
            raise RuntimeError('cannot schedule new futures after shutdown')
        future = Future()
        self._calls.put((future, function, args))
        return future

    def _run(self) -> None:
        while True:
            call = self._calls.get()
            if call is None:
                return
            future, function, args = call
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(function(*args))
            except BaseException as error:
                future.set_exception(error)

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        """Stop after the queued calls, or drop them; ``wait`` joins for a bounded time."""
        self._closed = True
        if cancel_futures:
            while True:
                try:
                    call = self._calls.get_nowait()
                except queue.Empty:
                    break
                if call is not None:
                    call[0].cancel()
        self._calls.put(None)
        if wait:
            self._thread.join(self.join_timeout_s)


class _BufferedIterator:
    def __init__(self, source):
        self.source = source
        self.items = deque()

    def __iter__(self):
        return self

    def __next__(self):
        return self.items.popleft() if self.items else next(self.source)

    def __getattr__(self, name):
        # The training loop asks the iterable for more than items, for example the
        # save_state of an external dataloader at a checkpoint: answer for the source.
        if name in ('source', 'items'):
            raise AttributeError(name)
        return getattr(self.source, name)

    def peek(self, count):
        """Buffer upcoming items without advancing the owning rerun iterator."""
        while len(self.items) < count:
            self.items.append(next(self.source))
        return list(self.items)[:count]


class DcpSchedulePrefetcher:
    """Prepare one future batch without consuming the rerun iterator early.

    Only the owning training process reads data and calls distributed/CUDA APIs.
    One isolated CPU worker, on the DPxCP leader only, owns the persistent planner.
    The global length collective is moved earlier, not duplicated or threaded.

    With more than one pipeline stage every stage has its own DPxCP leader and CPU worker
    and plans the same global batch alone. No plan is sent between stages: ``consume``
    proves on every step, with one collective over the pipeline group, that all stages
    hold the same plan, before any of them uses it.

    Args:
        data_iterator: Original RerunDataIterator.
        dp_group: Existing data-parallel process group.
        dp_cp_group: Existing data/context-parallel process group.
        spec: Pickleable CPU planner specification; contains no tensors or groups.
        asynchronous: Whether to prepare and solve one batch ahead.
        trace_dir: Optional per-rank JSONL output directory.
        pp_group: Existing pipeline-parallel process group of a run with more than one
            pipeline stage (``spec['pipeline']``); None with one stage.
    """

    leader_error = None
    # True on a DPxCP leader whose executor could not start its thread: consume() sends
    # leader_error for it, as for a leader whose CPU worker could not be started.
    leader_without_executor = False
    pp_group = None
    # Training and evaluation iterators each have a prefetcher writing to the same per-rank
    # trace file; this index, in creation order within the process, tells their records apart.
    iterator_index = 0
    _created = itertools.count()
    # Set once this process has reported a failing Stage 2 (``_report_stage2_failure``).
    _stage2_failure_reported = False

    def __init__(
        self,
        data_iterator: 'RerunDataIterator',
        dp_group: torch.distributed.ProcessGroup,
        dp_cp_group: torch.distributed.ProcessGroup,
        spec: dict,
        asynchronous: bool,
        trace_dir: str | None,
        pp_group: torch.distributed.ProcessGroup | None = None,
    ) -> None:
        self.iterator = data_iterator
        self.buffer = _BufferedIterator(data_iterator.iterable)
        data_iterator.iterable = self.buffer
        self.dp_group = dp_group
        self.group = dp_cp_group
        self.pp_group = pp_group
        self.spec = spec
        self.asynchronous = asynchronous
        self.pending = None
        self.sequence_number = 0
        self.iterator_index = next(DcpSchedulePrefetcher._created)
        self.last_telemetry = None
        self.executor = None
        self.worker = None
        self.baseline = None
        self.leader_error = None
        self.trace_path = None
        if trace_dir:
            directory = Path(trace_dir)
            directory.mkdir(parents=True, exist_ok=True)
            self.trace_path = directory / f'rank{torch.distributed.get_rank():05d}.jsonl'
        if dp_cp_group.rank() == 0:
            # Nothing the leader does alone may raise here: the other ranks would wait in a
            # collective for a leader that already failed. consume() broadcasts the error.
            try:
                self.executor = _DaemonExecutor('dcp-planner-io')
                from megatron.core.datasets.data_schedule import DefaultDynamicCPScheduler

                geometry = spec['geometry']
                # The default layout is built with the count multiple of the run's pipeline
                # (the stages under virtual stages, else 1; None with one stage), as Megatron
                # aligns it: the planner prices no other incumbent.
                multiple = (spec.get('pipeline') or {}).get('count_multiple')
                self.baseline = DefaultDynamicCPScheduler(
                    geometry['tokens_per_rank'],
                    geometry['world_size'],
                    1,
                    multiple,
                    max_num_seqs=geometry['max_group_seqs'],
                    min_cp_size=min(geometry['allowed_cp_sizes']),
                ).get_groups_and_subsamples
                self.worker = _CpuPlanner(spec)
            except Exception:
                self.leader_error = traceback.format_exc()
                self.leader_without_executor = self.executor is None
        atexit.register(self.close)

    def _submit(self, prepared, count):
        lengths, padded = prepared[-1], prepared[4]
        digest = hashlib.sha256(json.dumps([lengths, padded]).encode()).hexdigest()
        submitted = {'monotonic_ns': time.monotonic_ns(), 'wall_ns': time.time_ns()}
        # Only the leader runs the code below, between two collectives. Nothing here may
        # raise: an error is carried in the pending entry and broadcast by consume(), so
        # every rank raises it together (also when it happened one step ahead in prefetch).
        future, error, incumbent_error = None, self.leader_error, None
        if self.executor and error is None:
            try:
                baseline_rows = None
                if self.baseline:
                    try:
                        baseline_rows = self.baseline(list(enumerate(padded)))
                    except Exception as failure:
                        # The default layout only seeds the search and is optional: the
                        # heuristic asserts on batches it cannot place, and the worker, which
                        # scores and searches, reports truly infeasible samples.
                        incumbent_error = f'{type(failure).__name__}: {failure}'
                future = self.executor.submit(self.worker.solve, lengths, padded, baseline_rows)
            except Exception:
                error = traceback.format_exc()
        return dict(
            prepared=prepared,
            count=count,
            future=future,
            error=error,
            incumbent_error=incumbent_error,
            digest=digest,
            submit=submitted,
        )

    def consume(self, batch_list: list) -> tuple:
        """Consume this step's raw data and the leader's completed schedule.

        With more than one pipeline stage the scheduler calls this only on a step for which
        no stage ran out of data: it reduces that flag over the pipeline group first.

        Args:
            batch_list: The items the scheduler read from the rerun iterator for this step.
                Reading them there keeps the end-of-data protocol of the scheduler; with a
                lookahead they are the items ``prefetch`` peeked, already prepared.

        Returns:
            Original five-field batch metadata tuple and the scheduled rank rows.
        """
        begin = time.monotonic_ns()
        count = len(batch_list)
        pending, self.pending = self.pending, None
        lookahead = (
            pending is not None and pending['count'] == count and not self.iterator.replaying
        )
        if not lookahead:
            if pending is not None and pending['future'] is not None:
                pending['future'].cancel()
            prepared = get_batch_and_global_seqlens(
                iter(batch_list), count, self.dp_group, include_original_seqlens=True
            )
            pending = self._submit(prepared, count)
        payload = [None]
        waiting = time.monotonic_ns()
        if self.executor or self.leader_without_executor:
            try:
                if pending['error'] is not None:
                    raise RuntimeError(pending['error'])
                ready = pending['future'].done()
                result = pending['future'].result(timeout=PLANNER_REPLY_TIMEOUT_S)
                payload[0] = dict(
                    result=result,
                    digest=pending['digest'],
                    ready_when_consumed=ready,
                    planner_wait_s=(time.monotonic_ns() - waiting) / 1e9,
                )
            except RuntimeScheduleError as error:
                payload[0] = dict(error=traceback.format_exc(), schedule_error=str(error))
            except FutureTimeoutError:
                # Without a limit a worker that never answers would hold the leader here
                # and, through the broadcast below, every other rank.
                payload[0] = dict(error=self._no_reply_error())
            except Exception:
                payload[0] = dict(error=traceback.format_exc())
        torch.distributed.broadcast_object_list(
            payload,
            src=torch.distributed.get_global_rank(self.group, 0),
            group=self.group,
            device=torch.device('cuda', torch.cuda.current_device()),
        )
        consumed = {'monotonic_ns': time.monotonic_ns(), 'wall_ns': time.time_ns()}
        if payload[0] is None and self.pp_group is not None:
            # Only a leader whose prefetcher was closed sends nothing. The other pipeline
            # stages wait in the collective below, so this fails the stage like a planner
            # error, after that collective. With one stage nothing waits, and the payload
            # raises where it is read first.
            payload[0] = dict(error='The DPxCP leader sent no schedule: its prefetcher was closed')
        failed = payload[0] is not None and 'error' in payload[0]
        mismatch, stage_key, stage_failed, stages_differ = None, None, False, False
        if self.pp_group is not None:
            # More than one pipeline stage. Every rank of every stage issues the next two
            # collectives here, whatever its own stage reported: a rank that raised first
            # would leave the other stages waiting in the collective over the pipeline
            # group. Every error is therefore raised after them, on every stage.
            mismatch = _count_over_group(
                payload[0].get('digest', pending['digest']) != pending['digest'], self.group
            )
            stage_key = 0
            if not failed and not mismatch:
                # The lengths the leader planned for and the rows of the plan: stages that
                # gathered other lengths differ in the key even when their rows are equal.
                try:
                    stage_key = _agreement_key([payload[0]['digest'], payload[0]['result']['rows']])
                except Exception:
                    # A reply without usable rows is the same on every rank of this stage:
                    # it fails the stage like any planner error, after the collective.
                    payload[0] = dict(
                        error='The reply of the CPU planner has no rows to compare:\n'
                        + traceback.format_exc()
                    )
                    failed = True
            stage_failed, stages_differ = _compare_pipeline_stages(
                failed or mismatch, stage_key, self.pp_group
            )
        if payload[0].get('schedule_error') is not None:
            # Same type and text on every rank: no layout exists for this batch.
            raise RuntimeScheduleError(payload[0]['schedule_error'])
        if failed:
            raise RuntimeError(f"CPU DCP planner failed:\n{payload[0]['error']}")
        if mismatch is None:
            mismatch = _count_over_group(payload[0]['digest'] != pending['digest'], self.group)
        if mismatch:
            raise RuntimeError('CP replicas disagree on the global logical/physical lengths')
        if stage_failed:
            raise RuntimeError(
                'The CPU DCP planner of another pipeline stage failed; the ranks of that stage '
                'report why (a planner error, a batch without a layout, or global lengths that '
                'differ between its ranks)'
            )
        if stages_differ:
            # The values of this stage, the same on all of its ranks: the logs of two stages
            # show whether the lengths (input digest) or only the plans (key) differ.
            raise RuntimeError(
                'Pipeline stages disagree on the runtime DCP schedule of this step: the global '
                'lengths they gathered or the plans they computed differ (this stage, pipeline '
                f"rank {self.pp_group.rank()}: input digest {pending['digest']}, agreement key "
                f'{stage_key:016x})'
            )
        result = payload[0]['result']
        pipeline_fields = {}
        if self.spec.get('pipeline') is not None:
            pipeline_fields = dict(
                pipeline=self.spec['pipeline'],
                pipeline_terms_are_default=self.spec.get('pipeline_terms_are_default'),
                pipeline_terms_fitted_at=self.spec.get('pipeline_terms_fitted_at'),
                pipeline_agreement_key=None if stage_key is None else f'{stage_key:016x}',
            )
        self.last_telemetry = dict(
            event='consume',
            iterator=self.iterator_index,
            sequence_number=self.sequence_number,
            asynchronous=self.asynchronous,
            lookahead=lookahead,
            submit=pending['submit'],
            solve_start=result['solve_start'],
            solve_end=result['solve_end'],
            consume=consumed,
            worker_pid=result['worker_pid'],
            worker_cuda_initialized=result['worker_cuda_initialized'],
            worker_distributed_initialized=result['worker_distributed_initialized'],
            external_scheduler_imported=result.get('external_scheduler_imported', False),
            input_digest=pending['digest'],
            global_lengths=pending['prepared'][-1],
            padded_lengths=pending['prepared'][4],
            global_ids=list(range(len(pending['prepared'][-1]))),
            residual_wait_s=payload[0]['planner_wait_s'],
            plan_receive_s=(consumed['monotonic_ns'] - waiting) / 1e9,
            ready_when_consumed=payload[0]['ready_when_consumed'],
            consume_s=(consumed['monotonic_ns'] - begin) / 1e9,
            queue_depth=0,
            queue_depth_at_submit=1,
            predicted_ms=result['predicted_ms'],
            model_metadata=result.get('model_metadata'),
            statistics=result['statistics'],
            rows=result['rows'],
            scheduler_type=self.spec.get('type'),
            max_cp_size=self.spec['geometry'].get('max_cp_size'),
            solver_options=self.spec.get('solver_options', {}),
            planner_options=result.get('planner_options', self.spec.get('solver_options', {})),
            solver_entry=result.get('solver_entry'),
            incumbent_error=pending.get('incumbent_error'),
            **pipeline_fields,
        )
        self._trace(self.last_telemetry)
        self._report_stage2_failure(result['statistics'])
        self.sequence_number += 1
        return pending['prepared'][:5], result['rows']

    def prefetch(self, count: int) -> None:
        """Prepare one batch and submit its solve before current-step GPU training.

        Nothing here crosses pipeline stages: their collective is issued in ``consume``.
        """
        if not self.asynchronous or self.pending is not None:
            return
        try:
            items = self.buffer.peek(count)
        except StopIteration:
            items = None
        if _count_over_group(items is None, self.group):
            return
        prepared = get_batch_and_global_seqlens(
            iter(items), count, self.dp_group, include_original_seqlens=True
        )
        self.pending = self._submit(prepared, count)
        self._trace(
            dict(
                event='submit',
                iterator=self.iterator_index,
                sequence_number=self.sequence_number,
                submit=self.pending['submit'],
                input_digest=self.pending['digest'],
                queue_depth=1,
            )
        )

    def _no_reply_error(self) -> str:
        process, state = getattr(self.worker, 'process', None), ''
        if process is not None:
            code = process.poll()
            ended = 'is still running' if code is None else f'exited with code {code}'
            state = f' (worker process {process.pid} {ended})'
        return (
            f'The CPU planner did not answer within {PLANNER_REPLY_TIMEOUT_S:g} s{state}. The '
            'DPxCP leader stops waiting so that every rank raises this error instead of '
            'blocking in the schedule broadcast'
        )

    def _trace(self, payload):
        if self.trace_path:
            with self.trace_path.open('a') as stream:
                stream.write(json.dumps(payload) + '\n')

    def _report_stage2_failure(self, statistics) -> None:
        """Log one WARNING per process, on rank 0, for the first plan in which Stage 2 failed.

        The planner contains an error of Stage 2: it keeps the Stage 1 layout and records
        the error in ``statistics['stage2']``, as ``first_error`` when a microbatch could
        not be re-packed and as ``rescore_error`` when the re-packed layout could not be
        scored. Without this notice only the trace would show that the run trains on Stage 1
        layouts. No collective is issued and nothing is raised.
        """
        stage2 = statistics.get('stage2') if isinstance(statistics, dict) else None
        if not isinstance(stage2, dict) or DcpSchedulePrefetcher._stage2_failure_reported:
            return
        kept = dict(
            first_error='the Stage 1 layout of every microbatch that failed was kept',
            rescore_error='the Stage 1 layout of the step was kept',
        )
        errors = [
            f'{name}: {stage2[name]} ({note})'
            for name, note in kept.items()
            if stage2.get(name) is not None
        ]
        if not errors:
            return
        DcpSchedulePrefetcher._stage2_failure_reported = True
        log_single_rank(
            logger,
            logging.WARNING,
            'Runtime DCP scheduler: Stage 2 of the solver failed while a plan was made '
            '(iterator %s, batch %s): %s. Later failures of Stage 2 are not reported again in '
            'this process: statistics["stage2"] of every plan records them (last_telemetry, '
            'and the trace when "trace_dir" is set)',
            self.iterator_index,
            self.sequence_number,
            '; '.join(errors),
            rank=0,
        )

    def close(self) -> None:
        """Finish outstanding CPU work; no worker ever accesses a process group."""
        worker, self.worker = self.worker, None
        if worker is not None:
            # Ending the process first unblocks an I/O thread still waiting for its reply;
            # a request in flight is abandoned, the worker is not given time to finish it.
            worker.stop()
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None
        if worker is not None:
            worker.close()


def _agreement_key(value) -> int:
    """62 bits of the SHA-256 of ``value`` written as JSON with sorted keys.

    The key and its negative both fit a signed 64-bit tensor, so one MAX reduction of
    ``[key, -key]`` gives the largest and the smallest key of a process group.
    """
    text = json.dumps(value, sort_keys=True, separators=(',', ':'))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'big') >> 2


def _compare_pipeline_stages(failed, key: int, pp_group) -> tuple[bool, bool]:
    """One collective over the pipeline group: has a stage failed, and do the keys differ.

    A MAX reduction over ``[failed, key, -key]`` (``_agreement_key``), with the same answer on
    every rank. Nothing is raised here: the callers raise once every rank has left it.
    """
    probe = torch.tensor([int(bool(failed)), key, -key], dtype=torch.int64, device='cuda')
    torch.distributed.all_reduce(probe, op=torch.distributed.ReduceOp.MAX, group=pp_group)
    any_failed, largest, negated_smallest = (int(value) for value in probe.tolist())
    return bool(any_failed), largest != -negated_smallest


_MISSING = '<unset>'


class _RecordedConfig:
    """Read-through view of the training configuration that records every field it serves.

    The recorded values are the fingerprint of what a runtime specification was derived
    from, so a later change to any of them is detected instead of ignored.
    """

    def __init__(self, config):
        self._config = config
        self._reads = {}

    def __getattr__(self, name):
        value = getattr(self._config, name, _MISSING)
        self._reads[name] = repr(value)
        if value is _MISSING:
            raise AttributeError(name)
        return value


def _check_layer_types(model, non_attention):
    """Name the cost model's layer types when a per-type mapping does not match them."""
    if non_attention is None:
        return
    required = getattr(model, 'non_attention_layer_types', None)
    if required is None:
        # Fail closed: a model that takes these inputs must say which types it expects.
        raise ValueError(
            f'{model.__name__} takes non_attention inputs but does not declare '
            'non_attention_layer_types'
        )
    optional = getattr(model, 'non_attention_optional_layer_types', ())
    for key in ('forward_ms_per_layer', 'backward_ms_per_layer'):
        value = non_attention[key]
        if isinstance(value, dict) and (
            set(required) - set(value) or set(value) - set(required) - set(optional)
        ):
            raise ValueError(
                f'non_attention.{key} must be one number, or one number per layer type '
                f'{list(required)}'
                + (f' (optionally also {list(optional)})' if optional else '')
                + f'; got {sorted(value)}'
            )


def _from_domain_check(error):
    """Whether a cost model raised ``error`` from its ``check_domain`` method."""
    return any(frame.name == 'check_domain' for frame in traceback.extract_tb(error.__traceback__))


def _pipeline_of_run(where, scheduler_type, hardware, model, config, pp_size, virtual):
    """``spec['pipeline']`` of a run with more than one pipeline stage; None with one stage.

    The one place that decides which pipeline shapes the runtime scheduler takes (README,
    "More than one pipeline stage"): the DynamicUSP solver with a cost model ``model`` that
    has a step time for several stages, with virtual stages (``virtual`` of 2 or more: the
    interleaved schedule, counts a multiple of the stages) or without (``virtual`` None: the
    non-interleaved schedule, any count). A virtual size of 1 is not taken: Megatron reads
    it as virtual stages where the schedule is chosen and as none where the data iterator
    is taken. The model is not looked at, as in DynamicUSP: a run whose stages are not
    equal shares of the model (an explicit pipeline layout, other layer counts in the first
    or last stage, the embedding or the loss accounted in the split, MTP layers, a hybrid
    model) starts like any other, and only its predicted step time is less accurate.
    ``config`` is the recording view, so the option read here is part of the fingerprint of
    the specification.
    """
    if pp_size == 1:
        return None
    if SCHEDULERS[scheduler_type]['solver'] != 'dynamic_usp' or not callable(
        getattr(model, 'pipeline_group_times', None)
    ):
        raise ValueError(
            f'{where}: the runtime DCP scheduler plans for more than one pipeline stage only '
            'with type "dynamic_usp" without "hardware" (the original DynamicUSP profile, '
            'whose cost model has a step time for several pipeline stages); this run has '
            f'pipeline_model_parallel_size={pp_size} with type={scheduler_type!r}, '
            f'hardware={hardware!r}'
        )
    if virtual is None:
        # No virtual stages: one model chunk per stage, any count of microbatches.
        return dict(stages=pp_size, virtual_stages=1, count_multiple=1)
    if virtual < 2:
        raise ValueError(
            'With more than one pipeline stage the runtime DCP scheduler takes '
            'virtual_pipeline_model_parallel_size unset (no virtual pipeline stages, the '
            'non-interleaved schedule) or at least 2 (the interleaved schedule); got '
            f'pipeline_model_parallel_size={pp_size} with '
            f'virtual_pipeline_model_parallel_size={virtual}'
        )
    # Megatron sets an unset group size to the pipeline-parallel size.
    group = getattr(config, 'microbatch_group_size_per_vp_stage', None)
    if group is not None and group != pp_size:
        raise ValueError(
            'With more than one pipeline stage the runtime DCP scheduler requires '
            'microbatch_group_size_per_vp_stage equal to pipeline_model_parallel_size '
            f'({pp_size}): its step time replays the interleaved schedule of that group size, '
            f'and its counts of microbatches are multiples of {pp_size}; got '
            f'microbatch_group_size_per_vp_stage={group}'
        )
    return dict(stages=pp_size, virtual_stages=virtual, count_multiple=pp_size)


def _pipeline_terms_fitted_at(terms, pipeline):
    """What ``solver.pipeline.fitted_at`` of the coefficient file says about this run.

    The record of the run in which the pipeline step terms were fitted, with ``matches_run``
    added; ``'not recorded'`` when the file gives a term but no record; None when it gives
    neither, so that every term is a default and there is no fit to describe.
    """
    record = terms.get('fitted_at')
    if record is None:
        defaults = terms.get('default_terms', ())
        given = [name for name in terms if name not in ('default_terms', 'fitted_at', *defaults)]
        return 'not recorded' if given else None
    run = dict(
        pipeline_parallel=pipeline['stages'],
        virtual_stages=pipeline['virtual_stages'],
        schedule=pipeline_schedule(pipeline['virtual_stages']),
    )
    return dict(record, matches_run=all(record.get(key) == value for key, value in run.items()))


def build_runtime_spec(
    settings: dict,
    config: 'ModelParallelConfig',
    world_size: int,
    *,
    pp_size: int = 1,
    max_num_seqs: int | None = None,
) -> dict:
    """Bind validated runtime settings to the training configuration.

    ``settings`` comes from ``load_runtime_config`` or ``validate_runtime_config``. The cost
    model and its paired solver are built once in this process, exactly as the CPU worker
    builds them, and the model's domain is checked against ``config``. The returned
    specification is CPU-only and serializable; ``solver_options`` is a Python-only knob.

    Only with more than one pipeline stage (``pp_size``) it has ``pipeline``
    (``_pipeline_of_run``), ``pipeline_terms_are_default``, ``pipeline_terms_fitted_at``
    and two more facts that the stages compare at start-up, ``cost_model_table_sha256``
    and ``python_version``; ``stage2`` is present only when the coefficient file switches
    Stage 2 of the solver on (the comments where they are set describe them).
    """
    from .dcp_runtime_scheduler.models import measured_dynamic_usp_model, model_class
    from .dcp_runtime_scheduler.models.selection import (
        is_full_qwen35_config,
        validate_full_qwen35_config,
    )

    scheduler_type = settings['type']
    where = (
        f"Runtime configuration {settings.get('config_path', '<settings>')!r} "
        f'(type {scheduler_type!r})'
    )
    # Read and checked first: with one pipeline stage a virtual size is reported before any
    # other fault, and the fingerprint of the specification lists this field first.
    virtual = getattr(config, 'virtual_pipeline_model_parallel_size', None)
    if pp_size == 1 and virtual is not None:
        raise ValueError(
            'The runtime DCP scheduler takes virtual pipeline stages only with more than '
            f'one pipeline stage; got virtual_pipeline_model_parallel_size={virtual} with '
            'pipeline_model_parallel_size=1'
        )
    if (
        getattr(config, 'sequence_parallel', False)
        and getattr(config, 'tensor_model_parallel_size', 1) != 1
    ):
        raise ValueError(
            'The runtime DCP scheduler currently requires sequence_parallel off or '
            'tensor_model_parallel_size=1'
        )
    options = dict(settings['options'])
    minimum = config.min_dynamic_context_parallel_size
    maximum = settings['max_cp_size']
    if scheduler_type == 'dynamic_usp':
        # "hardware" selects a packaged measured calibration with its own CP sizes; decide
        # that before applying the tiers of the original profile.
        try:
            model = measured_dynamic_usp_model(options.pop('hardware', None))
        except ValueError as error:
            raise ValueError(f'{where}: {error}') from error
        if model is None:
            model = 'usp_calib_v1'
            if world_size not in (8, 16, 32) or minimum not in (8, 16, 32) or minimum > world_size:
                raise ValueError(
                    "type 'dynamic_usp' requires DPxCP and min_dynamic_context_parallel_size in "
                    f'(8, 16, 32); got DPxCP={world_size}, minimum={minimum}. These are the CP '
                    'sizes of the original DynamicUSP profile; "hardware": "h100" selects the '
                    'calibration measured for classic P2P groups of 1, 2, 4 or 8 ranks'
                )
            options['allow_provisional'] = True
    elif is_full_qwen35_config(config):
        model = 'qwen35_full'
        # Reported in the plan metadata next to the tail policy of the measured source.
        options['tail_padding'] = resolve_thd_tail_padding_policy(config)
    else:
        model = 'dcp_benchmark'
        heads = tuple(
            getattr(config, name, None)
            for name in ('num_attention_heads', 'num_query_groups', 'kv_channels')
        )
        layers = getattr(config, 'num_layers', None)
        supported = getattr(model_class(model), 'attention_geometries', None)
        largest = getattr(model_class(model), 'max_dp_cp_size', None)
        if largest is not None and world_size > largest:
            raise ValueError(
                f'{where} does not match the training configuration: the {scheduler_type!r} '
                'cost model has ordinary-GQA attention parameters for DPxCP '
                f'(data-parallel size times context-parallel size) up to {largest}; this run '
                f'has DPxCP={world_size}'
            )
        if supported is not None:
            # Name the training settings; the predictor itself only reports a bare tuple.
            if heads not in supported:
                raise ValueError(
                    f'{where} does not match the training configuration: the '
                    f'{scheduler_type!r} cost model has attention parameters for '
                    '(num_attention_heads, num_query_groups, kv_channels) in '
                    f'{list(supported)}; this model has {heads}'
                )
            if type(layers) is not int or layers < 1:
                raise ValueError(
                    f'{where} does not match the training configuration: num_layers must be '
                    f'a positive integer, got {layers!r}'
                )
        options.update(
            query_heads=heads[0],
            kv_heads=heads[1],
            head_dim=heads[2],
            layer_count=layers,
            # The layer program replays the forward pass when every layer is recomputed.
            recompute_forward=getattr(config, 'recompute_granularity', None) == 'full',
        )
    pipeline = _pipeline_of_run(
        where,
        scheduler_type,
        settings['options'].get('hardware'),
        model_class(model),
        config,
        pp_size,
        virtual,
    )
    if maximum is not None and maximum < minimum:
        raise ValueError(
            f'max_cp_size={maximum} must be >= min_dynamic_context_parallel_size={minimum}'
        )
    # A cost model class declares the largest CP group it prices. Without a declaration
    # every size up to DPxCP is a candidate, and the model's own domain check or the
    # start-up probe reports the sizes it does not price.
    ceiling = getattr(model_class(model), 'largest_cp_size', None) or world_size
    if minimum > ceiling:
        raise ValueError(
            f'The {scheduler_type!r} cost model prices CP groups of at most {ceiling} ranks; '
            f'min_dynamic_context_parallel_size={minimum} is above that'
        )
    # Classic CP creates canonical, size-aligned power-of-two subgroups.
    # Restrict candidates, while leaving the selected solver's search enabled.
    formable = tuple(
        cp
        for cp in range(minimum, world_size + 1)
        if cp & (cp - 1) == 0 and world_size % cp == 0 and cp <= ceiling
    )
    if minimum not in formable:
        raise ValueError(
            'The runtime DCP scheduler requires min_dynamic_context_parallel_size to be a '
            f'power-of-two minimum CP dividing DPxCP={world_size}, got {minimum}'
        )
    if maximum is not None and maximum not in formable:
        # No rounding: the cap is the largest group the user accepts, so it has to be one.
        raise ValueError(
            f'{where}: max_cp_size={maximum} is not a CP group size this run can form. With '
            f'DPxCP={world_size} and min_dynamic_context_parallel_size={minimum} the '
            f'{scheduler_type!r} scheduler forms groups of {list(formable)} ranks '
            '(power-of-two sizes dividing DPxCP that its cost model covers); set '
            '"max_cp_size" to one of them'
        )
    allowed = tuple(cp for cp in formable if maximum is None or cp <= maximum)
    spec = dict(
        type=scheduler_type,
        geometry=dict(
            world_size=world_size,
            tokens_per_rank=config.max_seqlen_per_dp_cp_rank,
            ep_size=getattr(config, 'expert_model_parallel_size', 1),
            allowed_cp_sizes=allowed,
            dataset_alignment=2 * world_size,
            # Execution cap (thd_max_packed_sequences, when set) intersected with the
            # largest pack the cost models can price.
            max_group_seqs=(
                MAX_PRICED_GROUP_SEQUENCES
                if max_num_seqs is None
                else min(max_num_seqs, MAX_PRICED_GROUP_SEQUENCES)
            ),
            align_group_starts=True,
            max_cp_size=maximum,
        ),
        model=model,
        model_options=options,
        solver_options={},
        hardware=settings['options'].get('hardware'),
    )
    spec['async'] = settings['async']
    spec['trace_dir'] = settings['trace_dir']
    for key in ('config_path', 'config_sha256'):
        if key in settings:
            spec[key] = settings[key]
    if pipeline is not None:
        # The planner below, and the one of the CPU worker, are built for these stages.
        spec['pipeline'] = pipeline
    try:
        # The geometry comes from the training configuration alone (EP size, DPxCP, ...).
        RuntimePlanGeometry(**spec['geometry'])
        if model == 'qwen35_full':
            validate_full_qwen35_config(config)
    except ValueError as error:
        raise ValueError(f'{where} does not match the training configuration: {error}') from error
    try:
        _check_layer_types(model_class(model), options.get('non_attention'))
        # The same construction the CPU worker performs, done once here so that this
        # rank reports an unusable configuration before training starts.
        planner = dcp_runtime_worker._initialize_planner(spec)
    except (TypeError, ValueError) as error:
        # A model may check the planner geometry (DPxCP, tokens per rank, EP size, CP sizes) in
        # check_domain; a failure there is a training-configuration fault, not the file's.
        blame = ' does not match the training configuration' if _from_domain_check(error) else ''
        raise ValueError(f'{where}{blame}: {error}') from error
    # What the cost model takes as one NVLink domain: it follows from the hardware of the model,
    # is shown in the start-up line and is not an input of the CPU worker.
    spec['nvlink_domain_size'] = getattr(planner.model, 'nvlink_domain_size', None)
    if options.get('calibration') is not None:
        # The CPU worker loads the file again when it starts; it must still be this one.
        spec['calibration_sha256'] = getattr(planner.model, 'calibration_sha256', None)
        if getattr(planner.model, 'sample_weight_is_default', False):
            # A user's table without solver.sample_weight: the start-up line names the default.
            spec['sample_weight_is_default'] = True
    if getattr(planner, 'stage2_enabled', False):
        # The start-up line says that Stage 2 is on; ``idle`` is the planner's reason when it
        # has no work in any plan of this run, else None.
        spec['stage2'] = dict(
            keep_threshold=planner.stage2_keep_threshold, idle=getattr(planner, 'stage2_idle', None)
        )
    if pipeline is not None:
        # Three step terms of the coefficient file change every score. The start-up line and
        # the trace name those the file does not give (defaults, not fitted values) and say
        # when the terms were fitted in a run of another pipeline shape.
        terms = planner.model.pipeline_terms
        spec['pipeline_terms_are_default'] = list(terms.get('default_terms', ()))
        spec['pipeline_terms_fitted_at'] = _pipeline_terms_fitted_at(terms, pipeline)
        # Every stage plans alone, so the stages compare at start-up what a plan depends on:
        # also the coefficient table the cost model loaded and the Python version (the solver
        # adds floats with the built-in sum(), whose rounding CPython 3.12 changed). The
        # digest does not guard the bytes of the packaged table: the cost model checks those
        # against its own pin, so a stage with other bytes stops alone, before the comparison.
        spec['cost_model_table_sha256'] = getattr(planner.model, 'calibration_sha256', None)
        spec['python_version'] = list(sys.version_info[:2])
    required_batch = getattr(planner.model, 'required_global_batch_size', None)
    if required_batch is not None:
        # A cost model whose step overhead was measured with one global batch size prices
        # no other; the training entry compares it with the configured batch sizes.
        spec['required_global_batch_size'] = required_batch
    try:
        planner.model.check_domain(planner.geometry, config)
    except ValueError as error:
        raise ValueError(f'{where} does not match the training configuration: {error}') from error
    if maximum is None and max(allowed) != world_size:
        # Without the full DPxCP group a sample that the training configuration admits may
        # have no CP group, which would only show at some training step: ask for the limit.
        remedy = 'Use a power-of-two DPxCP, or set' if world_size & (world_size - 1) else 'Set'
        group_limit = max(allowed) * config.max_seqlen_per_dp_cp_rank
        limit = f'{max(allowed)} * max_seqlen_per_dp_cp_rank = {group_limit} tokens'
        # The smaller of the two limits: what the largest group holds, and the longest
        # sample the cost model prices (its measured sample-length range).
        longest = longest_priced_sample(planner.geometry, planner.model)
        if longest < group_limit:
            limit = (
                f'{longest} tokens, the longest sample the {scheduler_type!r} cost model '
                f'prices (less than {limit})'
            )
        raise ValueError(
            f'The runtime DCP scheduler forms CP groups of {list(allowed)} ranks for '
            f'DPxCP={world_size} (power-of-two sizes dividing DPxCP that the cost model covers); '
            f'the full DPxCP group is not one of them, so samples longer than '
            f'{limit} cannot be scheduled. '
            f'{remedy} "max_cp_size": {max(allowed)} in the runtime '
            f'configuration to accept this limit.'
        )
    return spec


def _check_trace_dir(trace_dir):
    """Create the trace directory and prove that this rank can write into it."""
    if trace_dir is None:
        return
    try:
        Path(trace_dir).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=trace_dir):
            pass
    except (OSError, ValueError) as error:
        # ValueError: a path the operating system cannot represent (an embedded NUL).
        raise ValueError(
            f'trace_dir {trace_dir!r} from the runtime configuration cannot be created or '
            f'written on this rank: {error}'
        ) from error


def describe_runtime_spec(spec: dict) -> str:
    """One line for the training log: what the validated runtime file selected."""
    geometry = spec['geometry']
    domain = spec.get('nvlink_domain_size')
    topology = f'NVLink domain: {domain} consecutive ranks' if domain else ''
    if spec.get('hardware'):
        hardware = f", hardware={spec['hardware']}" + (f' ({topology})' if topology else '')
    else:
        # The original DynamicUSP profile names no hardware; its packaged table has a domain.
        hardware = f', {topology}' if topology else ''
    weight = (
        ", Stage 1 sample weight: DynamicUSP's published default A, B, C "
        '(the calibration file gives no solver.sample_weight)'
        if spec.get('sample_weight_is_default')
        else ''
    )
    return (
        f"Runtime DCP scheduler: type={spec['type']}, cost model={spec['model']}{hardware}, "
        f"CP group sizes={list(geometry['allowed_cp_sizes'])}, at most "
        f"{geometry['max_group_seqs']} sequences per CP group, "
        f"async={'on' if spec['async'] else 'off'}, runtime file={spec.get('config_path')}"
        f'{weight}{_describe_stage2(spec)}{_describe_pipeline(spec)}'
    )


def _describe_stage2(spec: dict) -> str:
    """The part of the start-up line that a coefficient file with Stage 2 switched on adds."""
    stage2 = spec.get('stage2')
    if stage2 is None:
        return ''
    if stage2['idle'] is not None:
        return f", Stage 2 switched on but idle: {stage2['idle']}"
    return f", Stage 2 on (keep_threshold={stage2['keep_threshold']})"


def _describe_pipeline(spec: dict) -> str:
    """The part of the start-up line that a run with more than one pipeline stage adds."""
    pipeline = spec.get('pipeline')
    if pipeline is None:
        return ''
    schedule = pipeline_schedule(pipeline['virtual_stages'])
    if pipeline['virtual_stages'] > 1:
        text = (
            f", pipeline stages={pipeline['stages']}, virtual stages="
            f"{pipeline['virtual_stages']}, microbatch count a multiple of "
            f"{pipeline['count_multiple']}"
        )
    else:
        # A run without virtual pipeline stages: the line names its schedule.
        text = (
            f", pipeline stages={pipeline['stages']}, no virtual stages ({schedule} schedule), "
            'any microbatch count'
        )
    defaults = spec.get('pipeline_terms_are_default')
    if defaults:
        text += (
            f", pipeline step terms at DynamicUSP's defaults, which are not fitted values: "
            f"{', '.join(defaults)} (the coefficient file does not give them in solver.pipeline)"
        )
    fitted_at = spec.get('pipeline_terms_fitted_at')
    if fitted_at == 'not recorded':
        text += (
            ', run in which the pipeline step terms were fitted: not recorded (the coefficient '
            'file has no solver.pipeline.fitted_at)'
        )
    elif fitted_at is not None and not fitted_at['matches_run']:
        text += (
            ', pipeline step terms were fitted in another run than this one: '
            f"solver.pipeline.fitted_at has pipeline_parallel={fitted_at['pipeline_parallel']}, "
            f"virtual_stages={fitted_at['virtual_stages']}, schedule={fitted_at['schedule']}; "
            f"this run has pipeline_parallel={pipeline['stages']}, "
            f"virtual_stages={pipeline['virtual_stages']}, schedule={schedule}"
        )
    return text


def runtime_spec(
    config: 'ModelParallelConfig',
    world_size: int,
    *,
    pp_size: int = 1,
    max_num_seqs: int | None = None,
) -> dict:
    """Validated planner specification for ``config.dynamic_cp_runtime_config``.

    The first call reads the file, validates it against the training configuration,
    creates the trace directory when one is configured, and keeps the result on
    ``config``; later calls, including every training step, return a copy without touching
    the file. The geometry and the configuration fields it was derived from must not
    change afterwards.
    """
    request = (world_size, pp_size, max_num_seqs)
    entry = getattr(config, '_dynamic_cp_runtime', None)
    if entry is not None:
        changed = [
            name
            for name, value in entry['reads'].items()
            if repr(getattr(config, name, _MISSING)) != value
        ]
        if entry['request'] != request or changed:
            detail = (
                f'configuration fields {changed}'
                if changed
                else f"(DPxCP, PP, sequence cap) {entry['request']} -> {request}"
            )
            raise ValueError(
                'Runtime DCP geometry/model cannot change after it was validated: ' + detail
            )
        return copy.deepcopy(entry['spec'])
    view = _RecordedConfig(config)
    settings = load_runtime_config(getattr(view, 'dynamic_cp_runtime_config', None))
    spec = build_runtime_spec(
        settings, view, world_size, pp_size=pp_size, max_num_seqs=max_num_seqs
    )
    _check_trace_dir(spec['trace_dir'])
    config._dynamic_cp_runtime = dict(
        request=request, reads=view._reads, spec=spec, coverage_checked=False, coverage_error=None
    )
    return copy.deepcopy(spec)


def _coverage_error(spec):
    """None, or why the cost model cannot price every CP width the planner may use."""
    gaps = dcp_runtime_worker.coverage_gaps(dcp_runtime_worker._initialize_planner(spec))
    if not gaps:
        return None
    reasons = {}
    for cp, reason in sorted(gaps.items()):
        reasons.setdefault(reason, []).append(cp)
    sizes = spec['geometry']['allowed_cp_sizes']
    covered = list(itertools.takewhile(lambda cp: cp not in gaps, sizes))
    advice = (
        f'It prices CP groups of {covered} ranks; set "max_cp_size": {covered[-1]} in the '
        'runtime configuration to schedule with those sizes only.'
        if covered
        else 'The smallest allowed CP size is among them, so no "max_cp_size" avoids them.'
    )
    return (
        f"The {spec['type']!r} cost model cannot price CP groups of {sorted(gaps)} ranks with "
        f"max_seqlen_per_dp_cp_rank={spec['geometry']['tokens_per_rank']}: "
        + '; '.join(f'{reason} (CP sizes {sizes})' for reason, sizes in reasons.items())
        + '. '
        + advice
    )


def _counts_over_group(flags, group) -> list[int]:
    """Number of ranks of ``group`` that set each of ``flags``; the same values on every rank."""
    counts = torch.tensor([int(bool(flag)) for flag in flags], dtype=torch.int32, device='cuda')
    torch.distributed.all_reduce(counts, group=group)
    return counts.tolist()


def _count_over_group(flag, group) -> int:
    """``_counts_over_group`` for one flag."""
    return _counts_over_group((flag,), group)[0]


def _stage_facts(spec: dict) -> dict:
    """What every rank of every pipeline stage must hold alike, by name.

    Every stage plans alone, so these are the facts that a plan depends on, and ``async``,
    which decides the collectives of a step.
    """
    return {
        'runtime file sha256': spec.get('config_sha256'),
        'calibration file sha256': spec.get('calibration_sha256'),
        'cost model table sha256': spec.get('cost_model_table_sha256'),
        'Stage 2': spec.get('stage2'),
        'pipeline': spec['pipeline'],
        'geometry': spec['geometry'],
        'async': spec['async'],
        'Python version': spec.get('python_version'),
    }


# The facts of ``_stage_facts`` as the errors of the start-up check list them.
_STAGE_FACTS_LISTED = (
    'the runtime file, the calibration file, the coefficient table of the cost model, Stage 2 '
    'of the solver, the pipeline sizes, the planner geometry, "async" and the Python version'
)


def check_cost_model_coverage(config: 'ModelParallelConfig', dp_cp_group, pp_group=None) -> None:
    """Probe every allowed CP width once per process; all ranks of the group raise together.

    Loading predictor parameters is the expensive part, so only the DPxCP leader probes
    and broadcasts the outcome, together with the digest of the runtime file it read; one
    reduction over ``dp_cp_group`` counts the ranks that read other bytes. Call it on
    every rank of ``dp_cp_group`` at the same point, after ``runtime_spec`` succeeded.

    With more than one pipeline stage every stage plans alone, so every rank of every stage
    must hold the same facts (``_stage_facts``). Within a stage each rank compares the key
    of its facts with the one its leader broadcasts, and the reduction gets a second
    count, the ranks whose key is another; a stage with such a rank has failed. Between
    the stages one collective over ``pp_group`` (``pg_collection.pp``) carries whether a
    stage has failed and otherwise the key of its facts. Every rank issues the same three
    collectives whatever it found and raises after the last one, so a difference on one
    rank stops every rank of every stage that the two groups reach: the ranks of one
    tensor-parallel rank. The ranks of other tensor-parallel ranks pass the check and then
    wait in a collective until the time-out of the process groups, as they do when a step
    fails. With one stage no facts are sent or compared.

    The outcome is kept: a later call raises the same error again, without a collective,
    so a caller that caught the first error cannot go on to train.
    """
    entry = config._dynamic_cp_runtime
    if entry['coverage_checked']:
        if entry.get('coverage_error') is not None:
            raise ValueError(entry['coverage_error'])
        return
    spec = entry['spec']
    has_pipeline = spec.get('pipeline') is not None
    if has_pipeline and pp_group is None:
        # Every rank is called alike, so every rank raises this before any collective.
        raise ValueError(
            'The start-up check of the runtime DCP scheduler needs the pipeline process group '
            'of a run with more than one pipeline stage: pass it as pp_group '
            '(pg_collection.pp)'
        )
    # The facts of this rank and their key; none with one pipeline stage. Both are ready
    # before the first collective, so nothing is computed between two collectives.
    facts = _stage_facts(spec) if has_pipeline else {}
    facts_key = _agreement_key(facts) if has_pipeline else 0
    # [probe outcome, what the leader read]; followers receive both.
    outcome = [None, None]
    if dp_cp_group.rank() == 0:
        read = dict(sha256=spec.get('config_sha256'), path=spec.get('config_path'))
        if has_pipeline:
            read['facts'] = facts_key
        outcome[1] = read
        # The leader must reach the broadcast: any failure travels as its outcome.
        try:
            outcome[0] = _coverage_error(spec)
        except Exception:
            outcome[0] = 'The cost-model coverage probe failed:\n' + traceback.format_exc()
    torch.distributed.broadcast_object_list(
        outcome,
        src=torch.distributed.get_global_rank(dp_cp_group, 0),
        group=dp_cp_group,
        device=torch.device('cuda', torch.cuda.current_device()),
    )
    leader = outcome[1] or {}
    # A rank that read other bytes would schedule with other settings; a different "async"
    # alone desynchronises the collectives. Count such ranks so that all of them stop.
    other_file = leader.get('sha256') != spec.get('config_sha256')
    if has_pipeline:
        # The same reduction with a second count: the ranks whose facts differ from those of
        # the leader of their stage. Their stage fails as a whole, and every stage stops.
        differing, other_facts = _counts_over_group(
            (other_file, leader.get('facts') != facts_key), dp_cp_group
        )
    else:
        differing, other_facts = _count_over_group(other_file, dp_cp_group), 0
    error = outcome[0]
    if has_pipeline:
        failed = error is not None or bool(differing) or bool(other_facts)
        another_failed, stages_differ = _compare_pipeline_stages(
            failed, 0 if failed else facts_key, pp_group
        )
    # The last collective is done: the texts of the errors are composed from here on.
    if differing:
        error = (
            f"{differing} of {dp_cp_group.size()} DPxCP ranks read a runtime configuration "
            f"that differs from the one the DPxCP leader read ({leader.get('path')!r}). The "
            'file must be identical on every rank and must not change while training starts'
        )
    if has_pipeline:
        # The values of this rank, which are those of its stage unless ranks of the stage
        # differ: the logs of the ranks show which values differ.
        own = ', '.join(f'{name} {value}' for name, value in facts.items())
        if error is None and other_facts:
            error = (
                f'{other_facts} of {dp_cp_group.size()} DPxCP ranks of this pipeline stage '
                'hold other settings of the runtime DCP scheduler than the DPxCP leader of '
                f'the stage: {_STAGE_FACTS_LISTED} must be the same on every rank of every '
                'stage, because every stage plans alone (this rank, DPxCP rank '
                f'{dp_cp_group.rank()} of pipeline rank {pp_group.rank()}: {own})'
            )
        elif error is None and another_failed:
            error = (
                'The start-up check of the runtime DCP scheduler failed on another pipeline '
                'stage; the ranks of that stage report why'
            )
        elif error is None and stages_differ:
            error = (
                'Pipeline stages read different settings of the runtime DCP scheduler: '
                f'{_STAGE_FACTS_LISTED} must be the same on every stage, because every stage '
                f'plans alone (this stage, pipeline rank {pp_group.rank()}: {own})'
            )
    # Every collective is done on every rank; the outcome, good or bad, is final.
    entry['coverage_checked'], entry['coverage_error'] = True, error
    if error is not None:
        raise ValueError(error)
