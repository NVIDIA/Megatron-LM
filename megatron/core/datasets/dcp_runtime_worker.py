# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU-only DCP worker, executed by file path without importing Megatron or TE."""

import importlib
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, is_dataclass


def _runtime_module(name):
    prefix = f'{__package__}.' if __package__ else ''
    return importlib.import_module(f'{prefix}dcp_runtime_scheduler.{name}')


class _AttentionWorkModel:
    """Quadratic-work smoke-test proxy, not a calibrated GPU timing prediction."""

    name = 'megatron_quadratic_work'
    source_hashes = {'model': 'sum(stored_length**2)/cp'}
    program_costs = None
    layer_count = 1
    forward_non_attention_ms = 0.0
    backward_non_attention_ms = 0.0
    placement_invariant = True

    def __init__(self, geometry):
        self.geometry = geometry

    def forward(self, lengths, cp, *, ranks=None):
        return sum(self.geometry.stored_length(length, cp) ** 2 for length in lengths) / cp / 1e6

    def backward(self, lengths, cp, *, ranks=None):
        return 2 * self.forward(lengths, cp, ranks=ranks)

    def timing(self):
        return None

    def clear(self):
        return None

    def check_domain(self, geometry, config):
        return None


def _initialize_planner(spec):
    geometry_options = dict(spec['geometry'])
    geometry_options['allowed_cp_sizes'] = tuple(sorted(set(geometry_options['allowed_cp_sizes'])))
    geometry = _runtime_module('runtime.planner').RuntimePlanGeometry(**geometry_options)
    if spec['default_model']:
        model = _AttentionWorkModel(geometry=geometry, **spec['model_options'])
    elif spec['model'] in (
        'calibrated_attention',
        'qwen35_gb200',
        'dcp_scheduler.adapters:qwen35_new_core_model',
    ):
        model = _runtime_module('models').make_model(
            spec['model'], geometry, **spec['model_options']
        )
    else:
        module, factory = spec['model'].split(':', 1)
        model = getattr(importlib.import_module(module), factory)(**spec['model_options'])
    options = spec.get('solver_options', {})
    if spec.get('solver') == 'dynamic_usp':
        return _runtime_module('runtime.dynamic_usp').DynamicUSPRuntimePlanner(
            geometry, model, **options
        )
    if spec.get('solver'):
        module, factory = spec['solver'].split(':', 1)
        return getattr(importlib.import_module(module), factory)(geometry, model, **options)
    runtime = _runtime_module('runtime.lightweight')
    return runtime.LightweightRuntimePlanner(geometry, model, runtime.LightweightOptions(**options))


def _solve(planner, spec, lengths, padded, baseline_rows):
    validate_rows = _runtime_module('runtime.planner').validate_rows

    started = {'monotonic_ns': time.monotonic_ns(), 'wall_ns': time.time_ns()}
    alignment = planner.geometry.dataset_alignment
    if len(lengths) != len(padded) or any(
        ((length + alignment - 1) // alignment) * alignment != physical
        for length, physical in zip(lengths, padded)
    ):
        raise ValueError('Runtime DCP requires lengths matching the dataset padding alignment')
    logical = dict(enumerate(lengths))
    if spec['force_powers']:
        groups = validate_rows(
            baseline_rows,
            dict(enumerate(padded)),
            planner.geometry,
            min(planner.geometry.allowed_cp_sizes),
        )
        if any(cp & (cp - 1) for row in groups for _, cp, _ in row):
            raise ValueError('The P2 comparison baseline contains a non-power-of-two group')
        rows, statistics, predicted_ms = baseline_rows, {'powers_baseline': True}, None
    else:
        plan = planner.plan_padded(
            dict(enumerate(padded)), logical_lengths=logical, incumbent_rows=baseline_rows
        )
        rows, statistics, predicted_ms = plan.rows, planner.last_statistics, plan.predicted_ms
    torch = sys.modules.get('torch')
    cuda_initialized = bool(torch is not None and torch.cuda.is_initialized())
    distributed_initialized = bool(torch is not None and torch.distributed.is_initialized())
    if cuda_initialized or distributed_initialized:
        raise RuntimeError('CPU predictor must not initialize CUDA or distributed')
    return dict(
        rows=rows,
        statistics=statistics,
        predicted_ms=predicted_ms,
        planner_options=(
            asdict(planner.options)
            if is_dataclass(getattr(planner, 'options', None))
            else spec.get('solver_options', {})
        ),
        solver_entry=(
            spec['solver']
            if spec.get('solver') and ':' in spec['solver']
            else f'{type(planner).__module__}:{type(planner).__name__}'
        )
        + '.plan_padded',
        solve_start=started,
        solve_end={'monotonic_ns': time.monotonic_ns(), 'wall_ns': time.time_ns()},
        worker_pid=os.getpid(),
        worker_cuda_initialized=cuda_initialized,
        worker_distributed_initialized=distributed_initialized,
        external_scheduler_imported=any(
            name == 'dcp_scheduler' or name.startswith('dcp_scheduler.') for name in sys.modules
        ),
    )


def main() -> None:
    # Keep third-party diagnostics out of the JSON protocol, which uses stdout.
    output = sys.stdout
    sys.stdout = sys.stderr
    try:
        spec = json.loads(sys.stdin.readline())
        planner = _initialize_planner(spec)
        for request in sys.stdin:
            try:
                result = _solve(planner, spec, *json.loads(request))
            except Exception:
                result = {'error': traceback.format_exc()}
            output.write(json.dumps(result) + '\n')
            output.flush()
    except Exception:
        output.write(json.dumps({'error': traceback.format_exc()}) + '\n')
        output.flush()


if __name__ == '__main__':
    main()
