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


def _initialize_planner(spec):
    """Build the cost model and the solver that the scheduler type binds to it."""
    geometry_options = dict(spec['geometry'])
    geometry_options['allowed_cp_sizes'] = tuple(sorted(set(geometry_options['allowed_cp_sizes'])))
    geometry = _runtime_module('runtime.planner').RuntimePlanGeometry(**geometry_options)
    schedulers = _runtime_module('runtime.config').SCHEDULERS
    if spec['type'] not in schedulers:
        raise ValueError(f"Unknown scheduler type: {spec['type']!r}")
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
    if schedulers[spec['type']]['solver'] == 'dynamic_usp':
        return _runtime_module('runtime.dynamic_usp').DynamicUSPRuntimePlanner(
            geometry, model, **options
        )
    runtime = _runtime_module('runtime.lightweight')
    return runtime.LightweightRuntimePlanner(geometry, model, runtime.LightweightOptions(**options))


def _probe_packs(geometry, cp):
    """Full CP packs without padding: one sequence across the group, and ``cp`` sequences.

    The second pack holds ``tokens_per_rank`` tokens per sequence, with fewer and longer
    sequences when the per-group sequence cap is below ``cp``. Nothing is returned when no
    aligned sequence fits.
    """
    packs = []
    for count in (1, min(cp, geometry.max_group_seqs or cp)):
        pack = geometry.full_pack(cp, count)
        if pack is not None and pack not in packs:
            packs.append(pack)
    return packs


def coverage_gaps(planner):
    """Allowed CP widths the planner cannot rely on, with the predictor's own reason.

    The planner treats a pack the predictor rejects as unavailable, so a width without
    parameters would only surface as a failed plan at some training step. A width counts
    as covered when the model prices one of its full packs on ranks ``0..cp-1`` and also
    the full pack with the most sequences the planner puts in one group of that width:
    ``planner.sequence_limit``, the sequence cap lowered to what the model declares it can
    price. What the probe accepts is therefore what the planner builds.
    """
    geometry, model = planner.geometry, planner.model
    gaps = {}

    def rejection(lengths, cp):
        try:
            ranks = list(range(cp))
            if getattr(model, 'program_costs', None) is not None:
                model.program_costs(lengths, cp, ranks=ranks)
            model.forward(lengths, cp, ranks=ranks)
            model.backward(lengths, cp, ranks=ranks)
        except ValueError as error:
            return str(error)
        return None

    for cp in geometry.allowed_cp_sizes:
        packs = _probe_packs(geometry, cp)
        reasons, priced = [], None
        for lengths in packs:
            reason = rejection(lengths, cp)
            if reason is None:
                priced = lengths
                break
            reasons.append(reason)
        if packs and priced is None:
            gaps[cp] = reasons[0]
            continue
        limit = planner.sequence_limit(cp)
        if limit < 1:
            gaps[cp] = 'the cost model declares that it prices no pack of this CP size'
            continue
        count = min(limit, geometry.capacity(cp) // geometry.pack_unit(cp))
        largest = geometry.full_pack(cp, count) if count >= 1 else None
        if largest is not None and largest != priced:
            reason = rejection(largest, cp)
            if reason is not None:
                gaps[cp] = (
                    f'{reason} (for a full pack of {count} sequences, the most the planner '
                    'puts in one group of this size)'
                )
    model.clear()
    return gaps


def _solve(planner, spec, lengths, padded, baseline_rows):
    planner_module = _runtime_module('runtime.planner')
    validate_rows = planner_module.validate_rows

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
        # contract (per-group sequence cap, max_cp_size, the cost model's CP widths),
        # so a violation discards the seed and never fails the step.
        try:
            validate_rows(
                baseline_rows, dict(enumerate(padded)), geometry, min(geometry.allowed_cp_sizes)
            )
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
    # plan_padded validates the final rows against the same geometry, cap included.
    plan = planner.plan_padded(
        dict(enumerate(padded)), logical_lengths=logical, incumbent_rows=baseline_rows
    )
    rows = plan.rows
    statistics = {**planner.last_statistics, **incumbent_rejection}
    predicted_ms = plan.predicted_ms
    metadata = getattr(planner.model, 'metadata', None)
    model_metadata = metadata() if callable(metadata) else None
    torch = sys.modules.get('torch')
    cuda_initialized = bool(torch is not None and torch.cuda.is_initialized())
    distributed_initialized = bool(torch is not None and torch.distributed.is_initialized())
    if cuda_initialized or distributed_initialized:
        raise RuntimeError('CPU predictor must not initialize CUDA or distributed')
    return dict(
        rows=rows,
        statistics=statistics,
        predicted_ms=predicted_ms,
        model_metadata=model_metadata,
        planner_options=(
            asdict(planner.options)
            if is_dataclass(getattr(planner, 'options', None))
            else spec.get('solver_options', {})
        ),
        solver_entry=f'{type(planner).__module__}:{type(planner).__name__}.plan_padded',
        solve_start=started,
        solve_end={'monotonic_ns': time.monotonic_ns(), 'wall_ns': time.time_ns()},
        worker_pid=os.getpid(),
        worker_cuda_initialized=cuda_initialized,
        worker_distributed_initialized=distributed_initialized,
        external_scheduler_imported=any(
            name == 'dcp_scheduler' or name.startswith('dcp_scheduler.') for name in sys.modules
        ),
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
        # Ask the cost model for its pack limits while this process starts: a model may
        # price packs to answer, which must not use up the first plan's time budget.
        for cp in planner.geometry.allowed_cp_sizes:
            planner.sequence_limit(cp)
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
