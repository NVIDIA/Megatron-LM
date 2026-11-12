# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""One-batch lookahead for a CPU-only, leader-owned DCP planner."""

import atexit
import copy
import hashlib
import itertools
import json
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

from megatron.core.datasets.data_schedule_utils import get_batch_and_global_seqlens
from megatron.core.datasets.dcp_runtime_scheduler.runtime.config import load_runtime_config
from megatron.core.datasets.dcp_runtime_scheduler.runtime.planner import (
    MAX_PRICED_GROUP_SEQUENCES,
    RuntimePlanGeometry,
    RuntimeScheduleError,
    longest_priced_sample,
)
from megatron.core.packed_seq_params import resolve_thd_tail_padding_policy

if TYPE_CHECKING:
    from megatron.core.model_parallel_config import ModelParallelConfig
    from megatron.core.rerun_state_machine import RerunDataIterator


# Seconds the DPxCP leader waits for the CPU worker's reply to one batch before every rank
# raises an error. Plans take milliseconds to seconds, loading a cost model a few seconds;
# the limit only has to tell a busy worker from one that will never answer.
PLANNER_REPLY_TIMEOUT_S = 600.0


class _CpuPlanner:
    """Isolated executable: never re-import the GPU training entry point via spawn."""

    # Seconds to wait for the worker after closing its request stream. A process forked
    # later (a dataloader worker) can keep the pipe open, so EOF alone is not reliable.
    close_timeout_s = 10.0
    # True while a request waits for its reply.
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
        # This method runs in an I/O-only thread; the separate process solves.
        """Send one global batch to the isolated worker and return its result or error."""
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
        """End the worker process: request-stream EOF first, then kill after a timeout.

        A request that is still in flight is not waited for: its reply is of no use any
        more, and the worker may never produce it.
        """
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

    The thread talks to the CPU worker over its pipes and may wait for a reply when the
    process exits. concurrent.futures joins its threads before atexit callbacks run, which
    would keep the interpreter from exiting and from ever reaching ``close``. A daemon
    thread is not joined, so the atexit ``close`` runs and ends the worker.
    """

    # Seconds ``shutdown(wait=True)`` waits for the running call; the thread is a daemon and
    # is left behind after that.
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

    Args:
        data_iterator: Original RerunDataIterator.
        dp_group: Existing data-parallel process group.
        dp_cp_group: Existing data/context-parallel process group.
        spec: Pickleable CPU planner specification; contains no tensors or groups.
        asynchronous: Whether to prepare and solve one batch ahead.
        trace_dir: Optional per-rank JSONL output directory.
    """

    leader_error = None
    # Training and evaluation iterators each have a prefetcher writing to the same per-rank
    # trace file; this index, in creation order within the process, tells their records apart.
    iterator_index = 0
    _created = itertools.count()

    def __init__(
        self,
        data_iterator: 'RerunDataIterator',
        dp_group: torch.distributed.ProcessGroup,
        dp_cp_group: torch.distributed.ProcessGroup,
        spec: dict,
        asynchronous: bool,
        trace_dir: str | None,
    ) -> None:
        self.iterator = data_iterator
        self.buffer = _BufferedIterator(data_iterator.iterable)
        data_iterator.iterable = self.buffer
        self.dp_group = dp_group
        self.group = dp_cp_group
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
            # The executor marks this rank as the leader. Everything else the leader
            # does alone must not raise here: the other ranks would wait in a collective
            # for a leader that already failed. consume() broadcasts the error instead.
            self.executor = _DaemonExecutor('dcp-planner-io')
            try:
                from megatron.core.datasets.data_schedule import DefaultDynamicCPScheduler

                geometry = spec['geometry']
                self.baseline = DefaultDynamicCPScheduler(
                    geometry['tokens_per_rank'],
                    geometry['world_size'],
                    1,
                    None,
                    max_num_seqs=geometry['max_group_seqs'],
                    min_cp_size=min(geometry['allowed_cp_sizes']),
                ).get_groups_and_subsamples
                self.worker = _CpuPlanner(spec)
            except Exception:
                self.leader_error = traceback.format_exc()
        atexit.register(self.close)

    def _submit(self, prepared, count):
        lengths = prepared[-1]
        padded = prepared[4]
        digest = hashlib.sha256(json.dumps([lengths, padded]).encode()).hexdigest()
        # Construct the default heuristic's layout as the incumbent that seeds the search.
        # This cheap incumbent construction remains on the main process; scoring/search
        # and the predictor's initialization run exclusively in the CPU worker.
        geometry = self.spec['geometry']
        cap_exceeded = geometry.get('max_cp_size') is not None and any(
            length > max(geometry['allowed_cp_sizes']) * geometry['tokens_per_rank']
            for length in padded
        )
        submitted = {'monotonic_ns': time.monotonic_ns(), 'wall_ns': time.time_ns()}
        # Only the leader runs the code below, between two collectives. Nothing here may
        # raise: an error is carried in the pending entry and broadcast by consume(), so
        # every rank raises it together (also when it happened one step ahead in prefetch).
        future, error, incumbent_error = None, self.leader_error, None
        if self.executor and error is None:
            try:
                baseline_rows = None
                if self.baseline and not cap_exceeded:
                    try:
                        baseline_rows = self.baseline(list(enumerate(padded)))
                    except Exception as failure:
                        # The incumbent is optional. The heuristic asserts on batches it
                        # cannot place; the worker reports truly infeasible samples.
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

        Args:
            batch_list: The items the scheduler read from the rerun iterator for this step.
                Reading them there keeps the end-of-data protocol of the scheduler; with a
                lookahead they are the items ``prefetch`` peeked, already prepared.

        Returns:
            Original five-field batch metadata tuple and the scheduled rank rows.
        """
        begin = time.monotonic_ns()
        count = len(batch_list)
        if (
            self.pending is not None
            and self.pending['count'] == count
            and not self.iterator.replaying
        ):
            pending = self.pending
            lookahead = True
        else:
            lookahead = False
            if self.pending is not None and self.pending['future'] is not None:
                self.pending['future'].cancel()
            prepared = get_batch_and_global_seqlens(
                iter(batch_list), count, self.dp_group, include_original_seqlens=True
            )
            pending = self._submit(prepared, count)
        self.pending = None
        payload = [None]
        waiting = time.monotonic_ns()
        if self.executor:
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
        if payload[0].get('schedule_error') is not None:
            # Same type and text on every rank: no layout exists for this batch.
            raise RuntimeScheduleError(payload[0]['schedule_error'])
        if 'error' in payload[0]:
            raise RuntimeError(f"CPU DCP planner failed:\n{payload[0]['error']}")
        mismatch = torch.tensor(
            [int(payload[0]['digest'] != pending['digest'])], device='cuda', dtype=torch.int32
        )
        torch.distributed.all_reduce(mismatch, group=self.group)
        if mismatch.item():
            raise RuntimeError('CP replicas disagree on the global logical/physical lengths')
        result = payload[0]['result']
        self.last_telemetry = dict(
            event='consume',
            iterator=self.iterator_index,
            sequence_number=self.sequence_number,
            # "async" as configured, and whether this batch was in fact planned one step
            # ahead: that needs lookahead from the training loop (see train_step).
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
        )
        self._trace(self.last_telemetry)
        self.sequence_number += 1
        return pending['prepared'][:5], result['rows']

    def prefetch(self, count: int) -> None:
        """Prepare one batch and submit its solve before current-step GPU training."""
        if not self.asynchronous or self.pending is not None:
            return
        ended = False
        try:
            items = self.buffer.peek(count)
        except StopIteration:
            ended = True
        end_flag = torch.tensor([int(ended)], dtype=torch.int32, device='cuda')
        torch.distributed.all_reduce(end_flag, group=self.group)
        if end_flag.item():
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
        process = getattr(self.worker, 'process', None)
        code = None if process is None else process.poll()
        state = (
            ''
            if process is None
            else (
                f' (worker process {process.pid} is still running)'
                if code is None
                else f' (worker process {process.pid} exited with code {code})'
            )
        )
        return (
            f'The CPU planner did not answer within {PLANNER_REPLY_TIMEOUT_S:g} s{state}. The '
            'DPxCP leader stops waiting so that every rank raises this error instead of '
            'blocking in the schedule broadcast'
        )

    def _trace(self, payload):
        if self.trace_path:
            with self.trace_path.open('a') as stream:
                stream.write(json.dumps(payload) + '\n')

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
    """
    from . import dcp_runtime_worker
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
    if pp_size != 1 or getattr(config, 'virtual_pipeline_model_parallel_size', None) is not None:
        raise ValueError('The runtime DCP scheduler currently supports PP1/VPP1 only')
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
            # The packaged GQA parameters are TE P2P fits over NCCL process groups.
            legacy_transport_backend='nccl_p2p',
            # The layer program replays the forward pass when every layer is recomputed.
            recompute_forward=getattr(config, 'recompute_granularity', None) == 'full',
        )
    if maximum is not None and maximum < minimum:
        raise ValueError(
            f'max_cp_size={maximum} must be >= min_dynamic_context_parallel_size={minimum}'
        )
    # A cost model class declares the largest CP group it prices. Without a declaration
    # every size up to DPxCP is a candidate, and the model's own domain check or the
    # start-up probe reports the sizes it does not price.
    ceiling = getattr(model_class(model), 'largest_cp_size', None)
    if ceiling is None:
        ceiling = world_size
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
        # A model checks the planner geometry (DPxCP, tokens per rank, EP size, CP sizes) in
        # check_domain; a failure there is a training-configuration fault, not the file's.
        blame = ' does not match the training configuration' if _from_domain_check(error) else ''
        raise ValueError(f'{where}{blame}: {error}') from error
    # What the cost model takes as one NVLink domain: it follows from the hardware of the model,
    # is shown in the start-up line and is not an input of the CPU worker.
    spec['nvlink_domain_size'] = getattr(planner.model, 'nvlink_domain_size', None)
    if options.get('calibration') is not None:
        # The CPU worker loads the file again when it starts; it must still be this one.
        spec['calibration_sha256'] = getattr(planner.model, 'calibration_sha256', None)
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
        # Without the full DPxCP group a sample that the training configuration admits
        # (up to DPxCP * max_seqlen_per_dp_cp_rank tokens) may have no legal CP group, which
        # would only surface as an error at some training step. Make the limit explicit.
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
    return (
        f"Runtime DCP scheduler: type={spec['type']}, cost model={spec['model']}{hardware}, "
        f"CP group sizes={list(geometry['allowed_cp_sizes'])}, at most "
        f"{geometry['max_group_seqs']} sequences per CP group, "
        f"async={'on' if spec['async'] else 'off'}, runtime file={spec.get('config_path')}"
    )


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
    from . import dcp_runtime_worker

    gaps = dcp_runtime_worker.coverage_gaps(dcp_runtime_worker._initialize_planner(spec))
    if not gaps:
        return None
    reasons = {}
    for cp, reason in sorted(gaps.items()):
        reasons.setdefault(reason, []).append(cp)
    covered = []
    for cp in spec['geometry']['allowed_cp_sizes']:
        if cp in gaps:
            break
        covered.append(cp)
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


def _count_over_group(flag, group) -> int:
    """Number of ranks of ``group`` whose ``flag`` is set; the same value on every rank."""
    count = torch.tensor([int(bool(flag))], dtype=torch.int32, device='cuda')
    torch.distributed.all_reduce(count, group=group)
    return int(count.item())


def check_cost_model_coverage(config: 'ModelParallelConfig', dp_cp_group) -> None:
    """Probe every allowed CP width once per process; all ranks of the group raise together.

    Loading predictor parameters is the expensive part, so only the DPxCP leader probes
    and the outcome is broadcast, together with the digest of the runtime file the leader
    read: every rank reads the file itself and must have read the same bytes. Call it on
    every rank of ``dp_cp_group`` at the same point, after ``runtime_spec`` succeeded.

    The outcome is kept: a later call raises the same error again, without a collective,
    so a caller that caught the first error cannot go on to train.
    """
    entry = getattr(config, '_dynamic_cp_runtime', None)
    if entry is None:
        return
    if entry['coverage_checked']:
        if entry.get('coverage_error') is not None:
            raise ValueError(entry['coverage_error'])
        return
    spec = entry['spec']
    # [probe outcome, what the leader read]; followers receive both.
    outcome = [None, None]
    if dp_cp_group.rank() == 0:
        outcome[1] = dict(sha256=spec.get('config_sha256'), path=spec.get('config_path'))
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
    differing = _count_over_group(leader.get('sha256') != spec.get('config_sha256'), dp_cp_group)
    error = outcome[0]
    if differing:
        error = (
            f"{differing} of {dp_cp_group.size()} DPxCP ranks read a runtime configuration "
            f"that differs from the one the DPxCP leader read ({leader.get('path')!r}). The "
            'file must be identical on every rank and must not change while training starts'
        )
    # Both collectives are done on every rank; the outcome, good or bad, is final.
    entry['coverage_checked'], entry['coverage_error'] = True, error
    if error is not None:
        raise ValueError(error)
