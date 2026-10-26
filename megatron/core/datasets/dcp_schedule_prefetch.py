# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""One-batch lookahead for a CPU-only, leader-owned DCP planner."""

import atexit
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from megatron.core.datasets.data_schedule_utils import get_batch_and_global_seqlens

if TYPE_CHECKING:
    from megatron.core.model_parallel_config import ModelParallelConfig
    from megatron.core.rerun_state_machine import RerunDataIterator


class _CpuPlanner:
    """Isolated executable: never re-import the GPU training entry point via spawn."""

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
        self.process.stdin.write(json.dumps([lengths, padded, baseline_rows]) + '\n')
        self.process.stdin.flush()
        response = self.process.stdout.readline()
        if not response:
            raise RuntimeError(f'CPU DCP worker exited unexpectedly: {self.process.poll()}')
        payload = json.loads(response)
        if 'error' in payload:
            raise RuntimeError(payload['error'])
        return payload

    def close(self) -> None:
        self.process.stdin.close()
        self.process.wait()
        self.process.stdout.close()


class _BufferedIterator:
    def __init__(self, source):
        self.source = source
        self.items = deque()

    def __iter__(self):
        return self

    def __next__(self):
        return self.items.popleft() if self.items else next(self.source)

    def peek(self, count):
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
        self.last_telemetry = None
        self.executor = None
        self.worker = None
        self.baseline = None
        self.trace_path = None
        if trace_dir:
            directory = Path(trace_dir)
            directory.mkdir(parents=True, exist_ok=True)
            self.trace_path = directory / f'rank{torch.distributed.get_rank():05d}.jsonl'
        if dp_cp_group.rank() == 0:
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
            if spec.get('baseline'):
                module, factory = spec['baseline'].split(':', 1)
                self.baseline = getattr(importlib.import_module(module), factory)(geometry)
            self.worker = _CpuPlanner(spec)
            self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='dcp-planner-io')
        atexit.register(self.close)

    def _submit(self, prepared, count):
        lengths = prepared[-1]
        padded = prepared[4]
        digest = hashlib.sha256(json.dumps([lengths, padded]).encode()).hexdigest()
        # Construct the configured P2 incumbent, never the arbitrary heuristic.
        # This cheap incumbent construction remains on the main process; scoring/search
        # and the predictor's initialization run exclusively in the CPU worker.
        baseline_rows = self.baseline(list(enumerate(padded))) if self.baseline else None
        submitted = {'monotonic_ns': time.monotonic_ns(), 'wall_ns': time.time_ns()}
        future = (
            self.executor.submit(self.worker.solve, lengths, padded, baseline_rows)
            if self.executor
            else None
        )
        return dict(prepared=prepared, count=count, future=future, digest=digest, submit=submitted)

    def consume(self, count: int) -> tuple:
        """Consume this step's raw data and the leader's completed schedule.

        Returns:
            Original five-field batch metadata tuple and the scheduled rank rows.
        """
        begin = time.monotonic_ns()
        if (
            self.pending is not None
            and self.pending['count'] == count
            and not self.iterator.replaying
        ):
            pending = self.pending
            for _ in range(count):
                next(self.iterator)
        else:
            if self.pending is not None and self.pending['future'] is not None:
                self.pending['future'].cancel()
            prepared = get_batch_and_global_seqlens(
                self.iterator, count, self.dp_group, include_original_seqlens=True
            )
            pending = self._submit(prepared, count)
        self.pending = None
        payload = [None]
        waiting = time.monotonic_ns()
        if self.executor:
            try:
                ready = pending['future'].done()
                result = pending['future'].result()
                payload[0] = dict(
                    result=result,
                    digest=pending['digest'],
                    ready_when_consumed=ready,
                    planner_wait_s=(time.monotonic_ns() - waiting) / 1e9,
                )
            except Exception:
                payload[0] = dict(error=traceback.format_exc())
        torch.distributed.broadcast_object_list(
            payload,
            src=torch.distributed.get_global_rank(self.group, 0),
            group=self.group,
            device=torch.device('cuda', torch.cuda.current_device()),
        )
        consumed = {'monotonic_ns': time.monotonic_ns(), 'wall_ns': time.time_ns()}
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
            sequence_number=self.sequence_number,
            asynchronous=self.asynchronous,
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
            solver=self.spec.get('solver'),
            solver_options=self.spec.get('solver_options', {}),
            planner_options=result.get('planner_options', self.spec.get('solver_options', {})),
            solver_entry=result.get('solver_entry'),
            baseline=self.spec.get('baseline') or 'checkout_default_powers_not_historical_pin',
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
                sequence_number=self.sequence_number,
                submit=self.pending['submit'],
                input_digest=self.pending['digest'],
                queue_depth=1,
            )
        )

    def _trace(self, payload):
        if self.trace_path:
            with self.trace_path.open('a') as stream:
                stream.write(json.dumps(payload) + '\n')

    def close(self) -> None:
        """Finish outstanding CPU work; no worker ever accesses a process group."""
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None
        if self.worker is not None:
            self.worker.close()
            self.worker = None


def _json_options(options):
    if isinstance(options, str):
        options = (
            json.loads(options)
            if options.lstrip().startswith('{')
            else json.loads(Path(options).read_text())
        )
    if options is not None and not isinstance(options, dict):
        raise ValueError('Runtime model/solver options must be a JSON object')
    return options or {}


def runtime_spec(
    config: 'ModelParallelConfig', world_size: int, *, max_num_seqs: int | None = None
) -> dict:
    """Build a CPU-only, serializable planner specification from model configuration."""
    factory = getattr(config, 'dynamic_cp_runtime_model', None)
    options = _json_options(getattr(config, 'dynamic_cp_runtime_model_options', None))
    minimum = config.min_dynamic_context_parallel_size
    native = getattr(config, 'use_native_cp_transport', False)
    # Classic CP creates canonical, size-aligned power-of-two subgroups.
    # Restrict candidates, while leaving the selected solver's search enabled.
    allowed = tuple(
        cp
        for cp in range(minimum, world_size + 1)
        if native or (cp & (cp - 1) == 0 and world_size % cp == 0)
    )
    if minimum not in allowed:
        raise ValueError('Runtime DCP requires a power-of-two minimum CP dividing DPxCP')
    if factory == 'usp_calib_v1':
        if world_size not in (8, 16, 32) or minimum not in (8, 16, 32) or minimum > world_size:
            raise ValueError('usp_calib_v1 requires DPxCP and minimum CP in (8, 16, 32)')
        allowed = tuple(cp for cp in (8, 16, 32) if minimum <= cp <= world_size)
    return dict(
        geometry=dict(
            world_size=world_size,
            tokens_per_rank=config.max_seqlen_per_dp_cp_rank,
            ep_size=getattr(config, 'expert_model_parallel_size', 1),
            allowed_cp_sizes=allowed,
            dataset_alignment=2 * world_size,
            max_group_seqs=min(max_num_seqs, 64) if max_num_seqs is not None else 64,
            align_group_starts=not native,
        ),
        model=factory or 'megatron.core.datasets.dcp_runtime_worker:_AttentionWorkModel',
        model_options=options,
        solver=getattr(config, 'dynamic_cp_runtime_solver', None),
        solver_options=_json_options(getattr(config, 'dynamic_cp_runtime_solver_options', None)),
        baseline=getattr(config, 'dynamic_cp_runtime_baseline', None),
        default_model=(
            factory is None
            or factory == 'megatron.core.datasets.dcp_runtime_worker:_AttentionWorkModel'
        ),
        force_powers=getattr(config, 'dynamic_cp_runtime_force_powers', False),
    )
