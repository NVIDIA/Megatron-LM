# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Frozen benchmark snapshot: preserve the original prediction code body.
# isort: skip_file
# fmt: off
# ruff: noqa
# pylint: skip-file

"""Rank-level training timeline for dynamic CP with pipeline parallelism.

The evaluator models the execution dependencies that matter to the scheduler:

* CP groups rendezvous only on their member ranks.
* Disjoint CP groups can progress into later microbatches independently.
* Contiguous EP groups rendezvous after every modeled transformer layer.
* As a conservative parameter-gather approximation, DPxCP replicas rendezvous
  once after the first forward invocation on each physical PP stage.
* PP=1 follows Megatron's no-pipeline forward/backward order (F0, B0, ...).
* PP>1 follows Megatron's non-interleaved warmup/1F1B/cooldown order and
  carries activation/gradient readiness rank by rank between adjacent stages.
* VPP>1 follows Megatron's interleaved schedule table, rank-dependent warmup,
  and chunk-major logical pipeline dependencies.

Core-attention costs are per transformer layer. Non-attention costs are
rank-local, per layer and per microbatch; this is the appropriate contract for
full-pack THD/DCP runs where every rank executes the same physical token
capacity outside core attention. A final iteration tail can be supplied for
work such as optimizer/final gradient synchronization that is independent of
the pack schedule.
"""

from __future__ import annotations

import heapq

import math

from dataclasses import dataclass

from typing import Sequence

from .rank_timeline import RankTimelineError, rank_intervals

from .execution_program import execute_program, validate_program

class TrainingTimelineError(ValueError):
    """Raised when a training schedule or timing contract is invalid."""

@dataclass(frozen=True, order=True)
class StageOperation:
    """One forward or backward microbatch invocation on one PP stage."""

    stage: int
    phase: str
    microbatch: int
    model_chunk: int = 0

@dataclass(frozen=True)
class OperationTimeline:
    operation_index: int
    stage: int
    model_chunk: int
    phase: str
    microbatch: int
    layer_count: int
    earliest_start_ms: float
    latest_finish_ms: float
    dependency_wait_gpu_ms: float
    cp_rendezvous_wait_gpu_ms: float
    ep_sync_wait_gpu_ms: float
    first_forward_dp_cp_sync_wait_gpu_ms: float
    rank_work_ms: float
    cp_group_effective_work_ms: float
    cp_group_internal_padding_wait_gpu_ms: float
    input_rank_times_ms: tuple[float, ...]
    output_rank_times_ms: tuple[float, ...]

@dataclass(frozen=True)
class TrainingTimeline:
    world_size_per_stage: int
    pp_size: int
    vpp_size: int
    ep_size: int
    layers_per_stage: tuple[int, ...]
    layers_per_virtual_stage: tuple[int, ...]
    microbatch_group_size_per_vp_stage: int | None
    microbatch_order: tuple[int, ...]
    first_forward_dp_cp_sync: bool
    compute_makespan_ms: float
    iteration_tail_ms: float
    makespan_ms: float
    modeled_iterations_per_second: float
    rank_work_ms: float
    cp_group_effective_work_ms: float
    cp_group_internal_padding_wait_gpu_ms: float
    total_idle_gpu_ms: float
    dependency_wait_gpu_ms: float
    cp_rendezvous_wait_gpu_ms: float
    ep_sync_wait_gpu_ms: float
    first_forward_dp_cp_sync_wait_gpu_ms: float
    final_sync_wait_gpu_ms: float
    final_stage_rank_times_ms: tuple[tuple[float, ...], ...]
    stage_orders: tuple[tuple[StageOperation, ...], ...]
    operations: tuple[OperationTimeline, ...]

def distribute_layers(layer_count: int, pp_size: int) -> tuple[int, ...]:
    """Distribute contiguous transformer layers as evenly as possible."""
    if layer_count <= 0:
        raise TrainingTimelineError("layer_count must be positive")
    if pp_size <= 0:
        raise TrainingTimelineError("pp_size must be positive")
    if layer_count < pp_size:
        raise TrainingTimelineError(
            "layer_count must be at least pp_size when layers_per_stage is omitted"
        )
    base, remainder = divmod(layer_count, pp_size)
    return tuple(base + (stage < remainder) for stage in range(pp_size))

def resolve_layers_per_stage(
    *,
    layer_count: int | None,
    pp_size: int,
    layers_per_stage: Sequence[int] | None,
) -> tuple[int, ...]:
    if layers_per_stage is None:
        if layer_count is None:
            raise TrainingTimelineError(
                "layer_count is required when layers_per_stage is omitted"
            )
        return distribute_layers(int(layer_count), pp_size)
    result = tuple(int(value) for value in layers_per_stage)
    if len(result) != pp_size or any(value <= 0 for value in result):
        raise TrainingTimelineError(
            "layers_per_stage must contain one positive value per PP stage"
        )
    if layer_count is not None and sum(result) != int(layer_count):
        raise TrainingTimelineError(
            "sum(layers_per_stage) must equal layer_count"
        )
    return result

def resolve_pipeline_layers(
    *,
    layer_count: int | None,
    pp_size: int,
    vpp_size: int,
    layers_per_stage: Sequence[int] | None,
    layers_per_virtual_stage: Sequence[int] | None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Resolve physical-stage and chunk-major virtual-stage layer counts."""
    if vpp_size <= 0:
        raise TrainingTimelineError("vpp_size must be positive")
    if vpp_size == 1:
        physical = resolve_layers_per_stage(
            layer_count=layer_count,
            pp_size=pp_size,
            layers_per_stage=layers_per_stage,
        )
        if layers_per_virtual_stage is not None:
            virtual = tuple(int(value) for value in layers_per_virtual_stage)
            if virtual != physical:
                raise TrainingTimelineError(
                    "with vpp_size=1, layers_per_virtual_stage must equal "
                    "layers_per_stage"
                )
        return physical, physical

    if pp_size <= 1:
        raise TrainingTimelineError("vpp_size > 1 requires pp_size > 1")
    logical_stage_count = pp_size * vpp_size
    if layers_per_virtual_stage is not None:
        virtual = tuple(int(value) for value in layers_per_virtual_stage)
        if len(virtual) != logical_stage_count or any(value <= 0 for value in virtual):
            raise TrainingTimelineError(
                "layers_per_virtual_stage must contain pp_size * vpp_size "
                "positive values in chunk-major order"
            )
        if layer_count is not None and sum(virtual) != int(layer_count):
            raise TrainingTimelineError(
                "sum(layers_per_virtual_stage) must equal layer_count"
            )
        physical = tuple(
            sum(virtual[chunk * pp_size + stage] for chunk in range(vpp_size))
            for stage in range(pp_size)
        )
        if layers_per_stage is not None:
            requested_physical = resolve_layers_per_stage(
                layer_count=layer_count,
                pp_size=pp_size,
                layers_per_stage=layers_per_stage,
            )
            if requested_physical != physical:
                raise TrainingTimelineError(
                    "layers_per_stage disagrees with layers_per_virtual_stage"
                )
        return physical, virtual

    if layers_per_stage is not None:
        physical = resolve_layers_per_stage(
            layer_count=layer_count,
            pp_size=pp_size,
            layers_per_stage=layers_per_stage,
        )
        if any(value % vpp_size for value in physical):
            raise TrainingTimelineError(
                "each layers_per_stage value must be divisible by vpp_size when "
                "layers_per_virtual_stage is omitted"
            )
        virtual = tuple(
            physical[stage] // vpp_size
            for chunk in range(vpp_size)
            for stage in range(pp_size)
        )
        return physical, virtual

    if layer_count is None:
        raise TrainingTimelineError(
            "layer_count is required when pipeline layer counts are omitted"
        )
    if int(layer_count) <= 0 or int(layer_count) % logical_stage_count:
        raise TrainingTimelineError(
            "layer_count must be positive and divisible by pp_size * vpp_size "
            "when VPP layer counts are omitted"
        )
    per_virtual_stage = int(layer_count) // logical_stage_count
    virtual = (per_virtual_stage,) * logical_stage_count
    physical = (per_virtual_stage * vpp_size,) * pp_size
    return physical, virtual

def _validate_order(
    microbatch_count: int, order: Sequence[int] | None
) -> tuple[int, ...]:
    result = (
        tuple(range(microbatch_count))
        if order is None
        else tuple(int(value) for value in order)
    )
    if len(result) != microbatch_count or set(result) != set(range(microbatch_count)):
        raise TrainingTimelineError(
            "microbatch_order must be a permutation of every microbatch index"
        )
    return result

def one_f_one_b_stage_order(
    *,
    num_microbatches: int,
    pp_size: int,
    stage: int,
    microbatch_order: Sequence[int] | None = None,
) -> tuple[StageOperation, ...]:
    """Return Megatron's non-interleaved warmup/1F1B/cooldown order."""
    if num_microbatches <= 0:
        raise TrainingTimelineError("num_microbatches must be positive")
    if pp_size <= 0 or not 0 <= stage < pp_size:
        raise TrainingTimelineError("stage must be in [0, pp_size)")
    order = _validate_order(num_microbatches, microbatch_order)
    warmup = min(pp_size - stage - 1, num_microbatches)
    remaining = num_microbatches - warmup
    operations: list[StageOperation] = []
    for index in range(warmup):
        operations.append(StageOperation(stage, "forward", order[index]))
    for index in range(remaining):
        operations.append(
            StageOperation(stage, "forward", order[warmup + index])
        )
        operations.append(StageOperation(stage, "backward", order[index]))
    for index in range(remaining, num_microbatches):
        operations.append(StageOperation(stage, "backward", order[index]))
    return tuple(operations)

def resolve_microbatch_group_size_per_vp_stage(
    *,
    num_microbatches: int,
    pp_size: int,
    vpp_size: int,
    microbatch_group_size_per_vp_stage: int | None,
) -> int | None:
    """Apply Megatron's interleaved schedule constraints and default."""
    if vpp_size == 1:
        if microbatch_group_size_per_vp_stage is not None:
            raise TrainingTimelineError(
                "microbatch_group_size_per_vp_stage requires vpp_size > 1"
            )
        return None
    if pp_size <= 1:
        raise TrainingTimelineError("vpp_size > 1 requires pp_size > 1")
    group_size = (
        pp_size
        if microbatch_group_size_per_vp_stage is None
        else int(microbatch_group_size_per_vp_stage)
    )
    if group_size < pp_size or group_size > num_microbatches:
        raise TrainingTimelineError(
            "microbatch_group_size_per_vp_stage must be in "
            "[pp_size, num_microbatches]"
        )
    remainder = num_microbatches % group_size
    if 0 < remainder < pp_size:
        raise TrainingTimelineError(
            "the final VPP microbatch group must be empty or contain at least "
            "pp_size microbatches"
        )
    return group_size

def interleaved_schedule_table(
    *,
    num_microbatches: int,
    vpp_size: int,
    microbatch_group_size_per_vp_stage: int,
) -> tuple[tuple[int, int], ...]:
    """Return Megatron's ``(microbatch, model_chunk)`` VPP lookup table."""
    table: list[tuple[int, int]] = []
    group_size = int(microbatch_group_size_per_vp_stage)
    for first_microbatch in range(0, num_microbatches, group_size):
        last_microbatch = min(first_microbatch + group_size, num_microbatches)
        table.extend(
            (microbatch, model_chunk)
            for model_chunk in range(vpp_size)
            for microbatch in range(first_microbatch, last_microbatch)
        )
    return tuple(table)

def interleaved_one_f_one_b_stage_order(
    *,
    num_microbatches: int,
    pp_size: int,
    stage: int,
    vpp_size: int,
    microbatch_group_size_per_vp_stage: int,
    microbatch_order: Sequence[int] | None = None,
) -> tuple[StageOperation, ...]:
    """Return one physical rank's Megatron interleaved 1F1B order."""
    if not 0 <= stage < pp_size:
        raise TrainingTimelineError("stage must be in [0, pp_size)")
    order = _validate_order(num_microbatches, microbatch_order)
    table = interleaved_schedule_table(
        num_microbatches=num_microbatches,
        vpp_size=vpp_size,
        microbatch_group_size_per_vp_stage=(
            microbatch_group_size_per_vp_stage
        ),
    )
    total_virtual_microbatches = num_microbatches * vpp_size
    if len(table) != total_virtual_microbatches:
        raise AssertionError("interleaved schedule table lost work")
    warmup = (pp_size - stage - 1) * 2
    warmup += (vpp_size - 1) * microbatch_group_size_per_vp_stage
    warmup = min(warmup, total_virtual_microbatches)
    remaining = total_virtual_microbatches - warmup

    def operation(virtual_microbatch: int, phase: str) -> StageOperation:
        microbatch_slot, forward_chunk = table[virtual_microbatch]
        model_chunk = (
            forward_chunk
            if phase == "forward"
            else vpp_size - forward_chunk - 1
        )
        return StageOperation(
            stage,
            phase,
            order[microbatch_slot],
            model_chunk,
        )

    operations: list[StageOperation] = [
        operation(virtual_microbatch, "forward")
        for virtual_microbatch in range(warmup)
    ]
    for offset in range(remaining):
        operations.append(operation(warmup + offset, "forward"))
        operations.append(operation(offset, "backward"))
    operations.extend(
        operation(virtual_microbatch, "backward")
        for virtual_microbatch in range(remaining, total_virtual_microbatches)
    )
    return tuple(operations)

def _finite_nonnegative(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise TrainingTimelineError(f"{name} must be finite and non-negative")
    return result

def resolve_stage_values(
    value: float | Sequence[float],
    *,
    pp_size: int,
    name: str,
) -> tuple[float, ...]:
    """Broadcast one timing value or validate one value per PP stage."""
    if pp_size <= 0:
        raise TrainingTimelineError("pp_size must be positive")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = tuple(
            _finite_nonnegative(stage_value, f"{name}[{stage}]")
            for stage, stage_value in enumerate(value)
        )
        if len(result) != pp_size:
            raise TrainingTimelineError(
                f"{name} must contain one value per PP stage"
            )
        return result
    scalar = _finite_nonnegative(value, name)
    return (scalar,) * pp_size

def resolve_stage_local_token_models(
    value: Sequence[LocalTokenModel] | None,
    *,
    pp_size: int,
    name: str,
) -> tuple[LocalTokenModel, ...] | None:
    """Validate an optional local-token timing model for every PP stage."""
    if value is None:
        return None
    result = tuple(value)
    if len(result) != pp_size:
        raise TrainingTimelineError(
            f"{name} must contain one local-token model per PP stage"
        )
    if any(not hasattr(model, "predict_ms_per_layer") for model in result):
        raise TrainingTimelineError(
            f"{name} entries must implement predict_ms_per_layer"
        )
    return result

def resolve_stage_ep_group_token_models(
    value: Sequence[EPGroupTokenModel] | None,
    *,
    pp_size: int,
    name: str,
) -> tuple[EPGroupTokenModel, ...] | None:
    if value is None:
        return None
    result = tuple(value)
    if len(result) != pp_size:
        raise TrainingTimelineError(
            f"{name} must contain one EP-group model per PP stage"
        )
    if any(not hasattr(model, "predict_ms_per_layer") for model in result):
        raise TrainingTimelineError(
            f"{name} entries must implement predict_ms_per_layer"
        )
    return result

def _group_local_non_attention_ms(
    group: dict,
    *,
    cp: int,
    model: LocalTokenModel,
) -> tuple[float, ...]:
    raw_tokens = group.get("local_valid_tokens_per_rank")
    if raw_tokens is None:
        raise TrainingTimelineError(
            "token-aware non-attention requires local_valid_tokens_per_rank "
            "on every CP group"
        )
    tokens = tuple(int(value) for value in raw_tokens)
    if len(tokens) != cp:
        raise TrainingTimelineError(
            "local_valid_tokens_per_rank length must equal the CP size"
        )
    try:
        values = tuple(float(model.predict_ms_per_layer(value)) for value in tokens)
    except ValueError as error:
        raise TrainingTimelineError(str(error)) from error
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise TrainingTimelineError(
            "local-token non-attention predictions must be finite and non-negative"
        )
    return values

def _phase_group_cost_ms(group: dict, phase: str, stage: int = 0) -> float:
    if phase == "forward":
        value = group.get("forward_cost_ms", group.get("cost_ms"))
    elif phase == "backward":
        value = group.get("backward_cost_ms", 0.0)
    else:
        raise TrainingTimelineError(f"unsupported phase {phase!r}")
    if f"{phase}_cost_ms_by_stage" in group:
        values = group[f"{phase}_cost_ms_by_stage"]
        if stage >= len(values):
            raise TrainingTimelineError("missing per-stage attention cost")
        value = values[stage]
    if value is None:
        raise TrainingTimelineError(
            f"group is missing a {phase} core-attention cost"
        )
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise TrainingTimelineError(
            f"group has invalid {phase} core-attention cost {result}"
        )
    return result

def _execute_stage_operation(
    input_rank_times: Sequence[float],
    groups: Sequence[dict],
    *,
    world_size: int,
    ep_size: int,
    layer_count: int,
    phase: str,
    stage: int = 0,
    non_attention_ms_per_layer: float,
    local_token_non_attention_model: LocalTokenModel | None,
    ep_group_non_attention_model: EPGroupTokenModel | None,
    program_operations: Sequence[dict] | None = None,
) -> tuple[list[float], float, float, float, float]:
    """Apply one F/B stage invocation to a rank-readiness vector."""
    if program_operations is not None:
        return execute_program(input_rank_times, groups, world_size=world_size,
            ep_size=ep_size, stage=stage, operations=program_operations)
    try:
        intervals = rank_intervals(groups, world_size)
    except RankTimelineError as error:
        raise TrainingTimelineError(str(error)) from error
    available = [float(value) for value in input_rank_times]
    cp_wait = 0.0
    ep_wait = 0.0
    rank_work = 0.0
    cp_group_effective_work = 0.0
    attention_costs = [_phase_group_cost_ms(group, phase, stage) for group in groups]
    rank_valid_tokens: list[int] | None = None
    ep_group_non_attention_durations: tuple[float, ...] | None = None
    if ep_group_non_attention_model is not None:
        rank_valid_tokens = [0] * world_size
        for group, ranks in zip(groups, intervals):
            raw_tokens = group.get("local_valid_tokens_per_rank")
            if raw_tokens is None or len(raw_tokens) != len(ranks):
                raise TrainingTimelineError(
                    "EP-group non-attention requires one local valid-token "
                    "count per CP rank"
                )
            for rank, tokens in zip(ranks, raw_tokens):
                rank_valid_tokens[rank] = int(tokens)
        if int(ep_group_non_attention_model.ep_size) != ep_size:
            raise TrainingTimelineError(
                "EP-group non-attention model ep_size disagrees with timeline"
            )
        ep_group_non_attention_durations = tuple(
            float(
                ep_group_non_attention_model.predict_ms_per_layer(
                    rank_valid_tokens[start_rank : start_rank + ep_size]
                )
            )
            for start_rank in range(0, world_size, ep_size)
        )
        if any(
            not math.isfinite(duration) or duration < 0.0
            for duration in ep_group_non_attention_durations
        ):
            raise TrainingTimelineError(
                "EP-group non-attention prediction must be finite and non-negative"
            )

    local_values = (
        [
            _group_local_non_attention_ms(
                group,
                cp=int(group["cp"]),
                model=local_token_non_attention_model,
            )
            for group in groups
        ]
        if local_token_non_attention_model is not None
        else None
    )
    scalar_group_costs = (
        [duration + non_attention_ms_per_layer for duration in attention_costs]
        if local_token_non_attention_model is None
        and ep_group_non_attention_model is None
        else None
    )
    attention_effective_work_per_layer = sum(
        int(group["cp"]) * duration
        for group, duration in zip(groups, attention_costs)
    )
    ep_effective_work_per_layer = (
        ep_size * sum(ep_group_non_attention_durations)
        if ep_group_non_attention_durations is not None
        else 0.0
    )
    local_effective_work_per_layer = (
        sum(
            int(group["cp"])
            * (attention_duration + max(non_attention_durations))
            for group, attention_duration, non_attention_durations in zip(
                groups, attention_costs, local_values
            )
        )
        if local_values is not None
        else 0.0
    )

    def execute_attention() -> None:
        nonlocal cp_wait, rank_work
        for group, ranks, duration in zip(groups, intervals, attention_costs):
            cp = int(group["cp"])
            arrivals = [available[rank] for rank in ranks]
            start = max(arrivals)
            finish = start + duration
            cp_wait += sum(start - value for value in arrivals)
            rank_work += cp * duration
            for rank in ranks:
                available[rank] = finish

    def execute_rank_local_non_attention(
        values_by_group: Sequence[Sequence[float]],
    ) -> None:
        nonlocal rank_work
        for ranks, durations in zip(intervals, values_by_group):
            for rank, duration in zip(ranks, durations):
                available[rank] += duration
                rank_work += duration

    def execute_ep_sync() -> None:
        nonlocal ep_wait
        for start_rank in range(0, world_size, ep_size):
            end_rank = start_rank + ep_size
            synchronized = max(available[start_rank:end_rank])
            ep_wait += sum(
                synchronized - value for value in available[start_rank:end_rank]
            )
            available[start_rank:end_rank] = [synchronized] * ep_size

    def execute_ep_group_non_attention() -> None:
        nonlocal ep_wait, rank_work
        assert ep_group_non_attention_durations is not None
        for ep_group, start_rank in enumerate(range(0, world_size, ep_size)):
            end_rank = start_rank + ep_size
            arrivals = available[start_rank:end_rank]
            start = max(arrivals)
            duration = ep_group_non_attention_durations[ep_group]
            ep_wait += sum(start - value for value in arrivals)
            rank_work += ep_size * duration
            available[start_rank:end_rank] = [start + duration] * ep_size

    for _ in range(layer_count):
        if ep_group_non_attention_model is not None:
            cp_group_effective_work += attention_effective_work_per_layer
            if phase == "forward":
                execute_attention()
                execute_ep_group_non_attention()
            elif phase == "backward":
                execute_ep_group_non_attention()
                execute_attention()
            else:
                raise TrainingTimelineError(f"unsupported phase {phase!r}")
            cp_group_effective_work += ep_effective_work_per_layer
            continue
        if local_token_non_attention_model is None:
            assert scalar_group_costs is not None
            for group, ranks, duration in zip(
                groups, intervals, scalar_group_costs
            ):
                cp = int(group["cp"])
                arrivals = [available[rank] for rank in ranks]
                start = max(arrivals)
                finish = start + duration
                cp_wait += sum(start - value for value in arrivals)
                rank_work += cp * duration
                cp_group_effective_work += cp * duration
                for rank in ranks:
                    available[rank] = finish
            execute_ep_sync()
            continue

        assert local_values is not None
        cp_group_effective_work += local_effective_work_per_layer
        if phase == "forward":
            execute_attention()
            execute_rank_local_non_attention(local_values)
            execute_ep_sync()
        elif phase == "backward":
            # Transformer-layer backward traverses the MoE/non-attention path
            # before core-attention backward.
            execute_rank_local_non_attention(local_values)
            execute_ep_sync()
            execute_attention()
        else:
            raise TrainingTimelineError(f"unsupported phase {phase!r}")
    return available, cp_wait, ep_wait, rank_work, cp_group_effective_work

def evaluate_training_schedule(
    rounds: Sequence[Sequence[dict]],
    *,
    world_size: int,
    ep_size: int = 1,
    pp_size: int = 1,
    vpp_size: int = 1,
    layer_count: int | None = 1,
    layers_per_stage: Sequence[int] | None = None,
    layers_per_virtual_stage: Sequence[int] | None = None,
    microbatch_group_size_per_vp_stage: int | None = None,
    forward_non_attention_ms_per_layer: float | Sequence[float] = 0.0,
    backward_non_attention_ms_per_layer: float | Sequence[float] = 0.0,
    forward_local_token_models: Sequence[LocalTokenModel] | None = None,
    backward_local_token_models: Sequence[LocalTokenModel] | None = None,
    forward_ep_group_token_models: Sequence[EPGroupTokenModel] | None = None,
    backward_ep_group_token_models: Sequence[EPGroupTokenModel] | None = None,
    forward_pipeline_latency_ms: float = 0.0,
    backward_pipeline_latency_ms: float = 0.0,
    iteration_tail_ms: float = 0.0,
    first_forward_dp_cp_sync: bool = True,
    microbatch_order: Sequence[int] | None = None,
    execution_program: Sequence[dict] | None = None,
) -> TrainingTimeline:
    """Evaluate no-PP, non-interleaved, or interleaved 1F1B dependencies.

    ``world_size`` is the number of dynamic-CP ranks *per pipeline stage*.
    Pipeline dependencies connect equal rank indices on adjacent stages.

    ``first_forward_dp_cp_sync`` is a conservative approximation for the
    distributed-optimizer parameter all-gather overlap. It synchronizes all
    stage-local DPxCP ranks after the first forward invocation on each physical
    PP stage. It does not add another barrier for later forwards, VPP model
    chunks, or any backward operation.

    When local-token models are supplied, every group must carry
    ``local_valid_tokens_per_rank``. Forward layers execute core attention,
    rank-local residual/MoE work, then the EP rendezvous. Backward layers use
    the reverse dependency order: rank-local work, EP rendezvous, then core
    attention backward.
    """
    if not rounds:
        raise TrainingTimelineError("rounds must contain at least one microbatch")
    if world_size <= 0:
        raise TrainingTimelineError("world_size must be positive")
    if ep_size <= 0 or world_size % ep_size:
        raise TrainingTimelineError(
            "ep_size must be a positive divisor of world_size"
        )
    if pp_size <= 0:
        raise TrainingTimelineError("pp_size must be positive")
    if vpp_size <= 0:
        raise TrainingTimelineError("vpp_size must be positive")
    resolved_layers, resolved_virtual_layers = resolve_pipeline_layers(
        layer_count=layer_count,
        pp_size=pp_size,
        vpp_size=vpp_size,
        layers_per_stage=layers_per_stage,
        layers_per_virtual_stage=layers_per_virtual_stage,
    )
    resolved_microbatch_group_size = (
        resolve_microbatch_group_size_per_vp_stage(
            num_microbatches=len(rounds),
            pp_size=pp_size,
            vpp_size=vpp_size,
            microbatch_group_size_per_vp_stage=(
                microbatch_group_size_per_vp_stage
            ),
        )
    )
    forward_non_attention = resolve_stage_values(
        forward_non_attention_ms_per_layer,
        pp_size=pp_size,
        name="forward_non_attention_ms_per_layer",
    )
    backward_non_attention = resolve_stage_values(
        backward_non_attention_ms_per_layer,
        pp_size=pp_size,
        name="backward_non_attention_ms_per_layer",
    )
    forward_token_models = resolve_stage_local_token_models(
        forward_local_token_models,
        pp_size=pp_size,
        name="forward_local_token_models",
    )
    backward_token_models = resolve_stage_local_token_models(
        backward_local_token_models,
        pp_size=pp_size,
        name="backward_local_token_models",
    )
    forward_ep_models = resolve_stage_ep_group_token_models(
        forward_ep_group_token_models,
        pp_size=pp_size,
        name="forward_ep_group_token_models",
    )
    backward_ep_models = resolve_stage_ep_group_token_models(
        backward_ep_group_token_models,
        pp_size=pp_size,
        name="backward_ep_group_token_models",
    )
    if sum(
        value is not None
        for value in (forward_token_models, forward_ep_models)
    ) > 1 or (
        any(forward_non_attention)
        and (forward_token_models is not None or forward_ep_models is not None)
    ):
        raise TrainingTimelineError(
            "forward scalar, local-token, and EP-group non-attention models "
            "are mutually exclusive"
        )
    if sum(
        value is not None
        for value in (backward_token_models, backward_ep_models)
    ) > 1 or (
        any(backward_non_attention)
        and (backward_token_models is not None or backward_ep_models is not None)
    ):
        raise TrainingTimelineError(
            "backward scalar, local-token, and EP-group non-attention models "
            "are mutually exclusive"
        )
    forward_latency = _finite_nonnegative(
        forward_pipeline_latency_ms, "forward_pipeline_latency_ms"
    )
    backward_latency = _finite_nonnegative(
        backward_pipeline_latency_ms, "backward_pipeline_latency_ms"
    )
    tail = _finite_nonnegative(iteration_tail_ms, "iteration_tail_ms")
    order = _validate_order(len(rounds), microbatch_order)
    if execution_program is not None:
        validate_program(execution_program, resolved_virtual_layers)
        if (any(forward_non_attention) or any(backward_non_attention)
                or any(model is not None for model in (forward_token_models,
                    backward_token_models, forward_ep_models, backward_ep_models))):
            raise TrainingTimelineError("execution_program replaces per-layer non-attention models")

    if vpp_size == 1:
        stage_orders = tuple(
            one_f_one_b_stage_order(
                num_microbatches=len(rounds),
                pp_size=pp_size,
                stage=stage,
                microbatch_order=order,
            )
            for stage in range(pp_size)
        )
    else:
        assert resolved_microbatch_group_size is not None
        stage_orders = tuple(
            interleaved_one_f_one_b_stage_order(
                num_microbatches=len(rounds),
                pp_size=pp_size,
                stage=stage,
                vpp_size=vpp_size,
                microbatch_group_size_per_vp_stage=(
                    resolved_microbatch_group_size
                ),
                microbatch_order=order,
            )
            for stage in range(pp_size)
        )
    first_forward_operations: set[StageOperation] = set()
    for stage_order in stage_orders:
        for operation in stage_order:
            if operation.phase == "forward":
                first_forward_operations.add(operation)
                break
    operations = {
        operation
        for stage_order in stage_orders
        for operation in stage_order
    }
    if len(operations) != 2 * pp_size * vpp_size * len(rounds):
        raise AssertionError("1F1B operation construction lost or duplicated work")

    dependencies: dict[StageOperation, set[StageOperation]] = {
        operation: set() for operation in operations
    }
    local_predecessor: dict[StageOperation, StageOperation] = {}
    for stage_order in stage_orders:
        for previous, current in zip(stage_order, stage_order[1:]):
            dependencies[current].add(previous)
            local_predecessor[current] = previous
    def logical_stage(operation: StageOperation) -> int:
        return operation.model_chunk * pp_size + operation.stage

    logical_stage_count = pp_size * vpp_size
    for operation in operations:
        logical = logical_stage(operation)
        if operation.phase == "forward" and logical > 0:
            predecessor = logical - 1
            dependencies[operation].add(
                StageOperation(
                    predecessor % pp_size,
                    "forward",
                    operation.microbatch,
                    predecessor // pp_size,
                )
            )
        if operation.phase == "backward":
            dependencies[operation].add(
                StageOperation(
                    operation.stage,
                    "forward",
                    operation.microbatch,
                    operation.model_chunk,
                )
            )
            if logical + 1 < logical_stage_count:
                successor = logical + 1
                dependencies[operation].add(
                    StageOperation(
                        successor % pp_size,
                        "backward",
                        operation.microbatch,
                        successor // pp_size,
                    )
                )

    dependents: dict[StageOperation, list[StageOperation]] = {
        operation: [] for operation in operations
    }
    indegree = {}
    for operation, sources in dependencies.items():
        indegree[operation] = len(sources)
        for source in sources:
            dependents[source].append(operation)
    stage_position = {
        operation: index
        for stage_order in stage_orders
        for index, operation in enumerate(stage_order)
    }
    ready = [
        (stage_position[operation], operation.stage, operation)
        for operation in operations
        if indegree[operation] == 0
    ]
    heapq.heapify(ready)
    output_times: dict[StageOperation, tuple[float, ...]] = {}
    operation_rows: list[OperationTimeline] = []
    dependency_wait = 0.0
    cp_wait = 0.0
    ep_wait = 0.0
    first_forward_dp_cp_sync_wait = 0.0
    rank_work = 0.0
    cp_group_effective_work = 0.0

    while ready:
        _, _, operation = heapq.heappop(ready)
        local_source = local_predecessor.get(operation)
        local_ready = (
            list(output_times[local_source])
            if local_source is not None
            else [0.0] * world_size
        )
        input_ready = list(local_ready)
        for source in dependencies[operation]:
            if source == local_source:
                continue
            latency = 0.0
            if (
                operation.phase == "forward"
                and source.phase == "forward"
                and source.microbatch == operation.microbatch
                and logical_stage(source) + 1 == logical_stage(operation)
            ):
                latency = forward_latency
            elif (
                operation.phase == "backward"
                and source.phase == "backward"
                and source.microbatch == operation.microbatch
                and logical_stage(source) == logical_stage(operation) + 1
            ):
                latency = backward_latency
            source_times = output_times[source]
            for rank in range(world_size):
                input_ready[rank] = max(
                    input_ready[rank], source_times[rank] + latency
                )
        operation_dependency_wait = sum(
            input_ready[rank] - local_ready[rank] for rank in range(world_size)
        )
        non_attention = (
            forward_non_attention[operation.stage]
            if operation.phase == "forward"
            else backward_non_attention[operation.stage]
        )
        operation_layer_count = resolved_virtual_layers[
            operation.model_chunk * pp_size + operation.stage
        ]
        (
            final_times,
            operation_cp_wait,
            operation_ep_wait,
            operation_work,
            operation_cp_group_effective_work,
        ) = (
            _execute_stage_operation(
                input_ready,
                rounds[operation.microbatch],
                world_size=world_size,
                ep_size=ep_size,
                layer_count=operation_layer_count,
                phase=operation.phase,
                stage=operation.stage,
                program_operations=(execution_program[operation.model_chunk * pp_size
                    + operation.stage][operation.phase] if execution_program is not None else None),
                non_attention_ms_per_layer=non_attention,
                local_token_non_attention_model=(
                    forward_token_models[operation.stage]
                    if operation.phase == "forward"
                    and forward_token_models is not None
                    else backward_token_models[operation.stage]
                    if operation.phase == "backward"
                    and backward_token_models is not None
                    else None
                ),
                ep_group_non_attention_model=(
                    forward_ep_models[operation.stage]
                    if operation.phase == "forward"
                    and forward_ep_models is not None
                    else backward_ep_models[operation.stage]
                    if operation.phase == "backward"
                    and backward_ep_models is not None
                    else None
                ),
            )
        )
        operation_first_forward_dp_cp_sync_wait = 0.0
        if first_forward_dp_cp_sync and operation in first_forward_operations:
            synchronized = max(final_times)
            operation_first_forward_dp_cp_sync_wait = sum(
                synchronized - value for value in final_times
            )
            final_times = [synchronized] * world_size
        output_times[operation] = tuple(final_times)
        dependency_wait += operation_dependency_wait
        cp_wait += operation_cp_wait
        ep_wait += operation_ep_wait
        first_forward_dp_cp_sync_wait += operation_first_forward_dp_cp_sync_wait
        rank_work += operation_work
        cp_group_effective_work += operation_cp_group_effective_work
        operation_rows.append(
            OperationTimeline(
                operation_index=len(operation_rows),
                stage=operation.stage,
                model_chunk=operation.model_chunk,
                phase=operation.phase,
                microbatch=operation.microbatch,
                layer_count=operation_layer_count,
                earliest_start_ms=min(input_ready),
                latest_finish_ms=max(final_times),
                dependency_wait_gpu_ms=operation_dependency_wait,
                cp_rendezvous_wait_gpu_ms=operation_cp_wait,
                ep_sync_wait_gpu_ms=operation_ep_wait,
                first_forward_dp_cp_sync_wait_gpu_ms=(
                    operation_first_forward_dp_cp_sync_wait
                ),
                rank_work_ms=operation_work,
                cp_group_effective_work_ms=operation_cp_group_effective_work,
                cp_group_internal_padding_wait_gpu_ms=max(
                    0.0,
                    operation_cp_group_effective_work - operation_work,
                ),
                input_rank_times_ms=tuple(input_ready),
                output_rank_times_ms=tuple(final_times),
            )
        )
        for dependent in dependents[operation]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                heapq.heappush(
                    ready,
                    (
                        stage_position[dependent],
                        dependent.stage,
                        dependent,
                    ),
                )

    if len(output_times) != len(operations):
        unresolved = [
            operation for operation in operations if operation not in output_times
        ]
        raise TrainingTimelineError(
            f"1F1B dependency graph contains a cycle; unresolved={unresolved[:4]}"
        )
    final_stage_times = tuple(
        output_times[stage_order[-1]] for stage_order in stage_orders
    )
    compute_makespan = max(max(values) for values in final_stage_times)
    final_sync_wait = sum(
        compute_makespan - value
        for stage_times in final_stage_times
        for value in stage_times
    )
    gpu_count = pp_size * world_size
    total_idle = gpu_count * compute_makespan - rank_work
    decomposed_idle = (
        dependency_wait
        + cp_wait
        + ep_wait
        + first_forward_dp_cp_sync_wait
        + final_sync_wait
    )
    if not math.isclose(
        total_idle,
        decomposed_idle,
        rel_tol=1.0e-9,
        abs_tol=1.0e-6,
    ):
        raise AssertionError(
            "training-timeline idle decomposition does not close: "
            f"total={total_idle}, components={decomposed_idle}"
        )
    makespan = compute_makespan + tail
    return TrainingTimeline(
        world_size_per_stage=world_size,
        pp_size=pp_size,
        vpp_size=vpp_size,
        ep_size=ep_size,
        layers_per_stage=resolved_layers,
        layers_per_virtual_stage=resolved_virtual_layers,
        microbatch_group_size_per_vp_stage=resolved_microbatch_group_size,
        microbatch_order=order,
        first_forward_dp_cp_sync=bool(first_forward_dp_cp_sync),
        compute_makespan_ms=compute_makespan,
        iteration_tail_ms=tail,
        makespan_ms=makespan,
        modeled_iterations_per_second=1_000.0 / makespan,
        rank_work_ms=rank_work,
        cp_group_effective_work_ms=cp_group_effective_work,
        cp_group_internal_padding_wait_gpu_ms=max(
            0.0, cp_group_effective_work - rank_work
        ),
        total_idle_gpu_ms=max(0.0, total_idle),
        dependency_wait_gpu_ms=max(0.0, dependency_wait),
        cp_rendezvous_wait_gpu_ms=max(0.0, cp_wait),
        ep_sync_wait_gpu_ms=max(0.0, ep_wait),
        first_forward_dp_cp_sync_wait_gpu_ms=max(
            0.0, first_forward_dp_cp_sync_wait
        ),
        final_sync_wait_gpu_ms=max(0.0, final_sync_wait),
        final_stage_rank_times_ms=final_stage_times,
        stage_orders=stage_orders,
        operations=tuple(operation_rows),
    )
