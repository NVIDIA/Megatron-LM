# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU-only DCP worker, executed by file path without importing Megatron or TE."""

import importlib
import json
import os
import sys
import time
import traceback


def _runtime_module(name):
    prefix = f'{__package__}.' if __package__ else ''
    return importlib.import_module(f'{prefix}dcp_runtime_scheduler.{name}')


def _initialize_planner(spec):
    """Build the cost model and the solver that the scheduler type binds to it."""
    geometry_options = dict(spec['geometry'])
    geometry_options['allowed_cp_sizes'] = tuple(sorted(set(geometry_options['allowed_cp_sizes'])))
    geometry = _runtime_module('runtime.planner').RuntimePlanGeometry(**geometry_options)
    schedulers = _runtime_module('runtime.config').SCHEDULERS
    model = _runtime_module('models').make_model(spec['model'], geometry, **spec['model_options'])
    pinned = spec.get('calibration_sha256')
    if pinned is not None and getattr(model, 'calibration_sha256', None) != pinned:
        # Start-up validated other bytes than this process just loaded.
        raise ValueError(
            f"The calibration file {spec['model_options'].get('calibration')!r} changed after "
            f'start-up validation (sha256 {pinned} then, '
            f"{getattr(model, 'calibration_sha256', None)} now); it must not change during a run"
        )
    options = spec.get('solver_options', {})
    # Present only for a run with more than one pipeline stage.
    pipeline = spec.get('pipeline')
    if schedulers[spec['type']]['solver'] == 'dynamic_usp':
        if pipeline is not None:
            # The planner then scores the pipeline schedule of the run.
            schedule = _runtime_module('models.solver_section').pipeline_schedule(
                pipeline['virtual_stages']
            )
            options = dict(
                options,
                pipeline=(pipeline['stages'], pipeline['virtual_stages']),
                pipeline_schedule=schedule,
            )
        return _runtime_module('runtime.dynamic_usp').DynamicUSPRuntimePlanner(
            geometry, model, **options
        )
    runtime = _runtime_module('runtime.lightweight')
    return runtime.LightweightRuntimePlanner(geometry, model, runtime.LightweightOptions(**options))


def _solve(planner, spec, lengths, padded, baseline_rows):
    planner_module = _runtime_module('runtime.planner')
    started = {'monotonic_ns': time.monotonic_ns(), 'wall_ns': time.time_ns()}
    geometry = planner.geometry
    alignment = geometry.dataset_alignment
    if len(lengths) != len(padded):
        raise ValueError('Runtime DCP requires one padded length per logical length')
    for sample_id, (length, physical) in enumerate(zip(lengths, padded)):
        if ((length + alignment - 1) // alignment) * alignment != physical:
            raise planner_module.RuntimeScheduleError(
                f'sample {sample_id}: padded length {physical} is not seq_len={length} rounded '
                f'up to the dataset alignment {alignment} (2 * DPxCP) of the runtime scheduler'
            )
    logical = dict(enumerate(lengths))
    geometry.validate_lengths(logical)
    required = getattr(planner.model, 'required_global_batch_size', None)
    if required is not None and len(lengths) != required:
        # A measured step overhead describes one global batch size; every rank reports it.
        raise planner_module.RuntimeScheduleError(
            f"The {spec['type']!r} calibration was measured with {required} samples per "
            f'global batch; this global batch has {len(lengths)}'
        )
    unpriced = planner_module.unpriced_sample(geometry, planner.model, logical)
    if unpriced is not None:
        # The sample fits a CP group, but the cost model prices no pack that holds it: say
        # so here, with the limit, instead of failing later on some rejected pack.
        longest = planner_module.longest_priced_sample(geometry, planner.model)
        raise planner_module.RuntimeScheduleError(
            f'sample {unpriced}: seq_len={logical[unpriced]} is longer than the {longest} '
            f"tokens that the {spec['type']!r} cost model prices (its measured sample-length "
            f'range); it cannot be scheduled with allowed_cp_sizes={geometry.allowed_cp_sizes}, '
            f'max_seqlen_per_dp_cp_rank={geometry.tokens_per_rank}'
        )
    incumbent_rejection = {}
    if baseline_rows is not None:
        # The default heuristic only seeds the search. It does not know the runtime
        # contract, so a violation discards the seed and never fails the step.
        try:
            planner_module.validate_rows(baseline_rows, dict(enumerate(padded)), geometry)
        except planner_module.RuntimeScheduleError as error:
            unsupported = sorted(
                {cp for row in baseline_rows for _, cp, _ in planner_module.decode_groups(row)}
                - set(geometry.allowed_cp_sizes)
            )
            if geometry.max_cp_size is not None and any(
                cp > geometry.max_cp_size for cp in unsupported
            ):
                reason = 'default_incumbent_exceeds_max_cp_size'
            elif unsupported:
                reason = 'default_incumbent_outside_predictor_cp_domain'
            else:
                reason = 'default_incumbent_violates_runtime_contract'
            incumbent_rejection = {
                'incumbent_rejection': reason,
                'incumbent_rejection_detail': str(error),
                'incumbent_unsupported_cp_sizes': unsupported,
            }
            baseline_rows = None
    # The planner validates the final rows against the same geometry, cap included.
    plan = planner.plan_padded(
        dict(enumerate(padded)), logical_lengths=logical, incumbent_rows=baseline_rows
    )
    metadata = getattr(planner.model, 'metadata', None)
    model_metadata = metadata() if callable(metadata) else None
    torch = sys.modules.get('torch')
    if torch is not None and (torch.cuda.is_initialized() or torch.distributed.is_initialized()):
        raise RuntimeError('CPU predictor must not initialize CUDA or distributed')
    return dict(
        rows=plan.rows,
        statistics={**planner.last_statistics, **incumbent_rejection},
        predicted_ms=plan.predicted_ms,
        model_metadata=model_metadata,
        solve_start=started,
        solve_end={'monotonic_ns': time.monotonic_ns(), 'wall_ns': time.time_ns()},
        worker_pid=os.getpid(),
    )


def _error_response(error):
    """Report a failure; a schedule error keeps its message so every rank can re-raise it."""
    response = {'error': traceback.format_exc()}
    if any(base.__name__ == 'RuntimeScheduleError' for base in type(error).__mro__):
        response['schedule_error'] = str(error)
    return response


def main() -> None:
    # Keep third-party diagnostics out of the JSON protocol, which uses stdout.
    """Serve newline-delimited CPU planner requests over standard input/output."""
    output = sys.stdout
    sys.stdout = sys.stderr
    try:
        spec = json.loads(sys.stdin.readline())
        planner = _initialize_planner(spec)
        for request in sys.stdin:
            try:
                result = _solve(planner, spec, *json.loads(request))
            except Exception as error:
                result = _error_response(error)
            output.write(json.dumps(result) + '\n')
            output.flush()
    except Exception as error:
        output.write(json.dumps(_error_response(error)) + '\n')
        output.flush()


if __name__ == '__main__':
    main()
