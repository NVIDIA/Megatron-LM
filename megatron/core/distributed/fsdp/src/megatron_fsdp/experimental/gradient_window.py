# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Plan bounded packets of local contributions before the original reduce-scatter."""

import hashlib
from types import MappingProxyType

import torch
import torch.distributed as dist
from torch.distributed.tensor import Partial, Shard

from .execution_order import ExecutionAction


def _group_metadata(group, agreement_ranks):
    """Describe global storage; local shard sizes may differ across peers."""
    grad = group.main_grad
    mesh, layout = group.mesh, group.main_weight.layout
    placements = tuple(grad.placements)
    axes = [axis for axis, placement in enumerate(placements) if placement != Partial("avg")]
    sizes = tuple(mesh.size(axis) for axis in range(mesh.ndim))
    eligible = (
        len(axes) == 1
        and isinstance(placements[axes[0]], Shard)
        and placements[axes[0]].dim == 0
        and sizes[axes[0]] > 1
    )
    charge = 0
    if eligible:
        peers = dist.get_process_group_ranks(mesh.get_group(axes[0]))
        if not set(peers).issubset(agreement_ranks):
            raise ValueError("The RS agreement group must cover every affected inner communicator.")
        if grad.layout != layout:
            raise ValueError("RS input and main-gradient layouts must match.")
        numel = layout.get_local_range(mesh, (Partial("avg"),) * mesh.ndim).numel
        accumulator = torch.finfo(grad.dtype).bits // 8
        communication = torch.finfo(group.grad_comm_dtype).bits // 8
        # Reserve the accumulator, optional cast and one full-input RS scratch bound.
        charge = numel * (
            accumulator
            + (communication if grad.dtype != group.grad_comm_dtype else 0)
            + communication
        )
    signature = (
        sizes,
        tuple((type(placement).__name__, repr(placement)) for placement in placements),
        tuple(tuple(shape) for shape in layout.tensor_shapes),
        tuple(layout.tensor_to_offset),
        layout.size,
        tuple(layout.rank_to_offset),
        str(grad.dtype),
        str(group.grad_comm_dtype),
        group.grad_scale,
        charge,
    )
    return signature, charge


def _build_plan(records, agreement_ranks, budget_bytes):
    if type(budget_bytes) is not int or budget_bytes < 0:
        raise ValueError("RS window budget must be a nonnegative integer.")
    if any(event is None for event in records):
        raise RuntimeError("An FSDP owner expired before RS planning.")

    owners = {}
    for event in records:
        context = getattr(event.module, "context", None)
        for owner in getattr(context, "_module_names", ()):
            owners.setdefault(owner, len(owners))
    trace, groups, occurrences = [], {}, {}
    gradient_events, epochs, epoch = [], [], 0
    for event in records:
        owner = owners.setdefault(event.module, len(owners))
        trace.append(
            (
                event.event_id,
                owner,
                event.phase,
                event.action,
                event.rowwise_only,
                event.finalizes_outer,
            )
        )
        if event.action is not ExecutionAction.GRAD_READY:
            continue
        position = len(gradient_events)
        gradient_events.append(event)
        epochs.append(epoch)
        for ordinal, group in enumerate(event.module.parameter_groups):
            if not group.requires_grad:
                continue
            key = (owner, ordinal)
            if key not in groups:
                groups[key] = _group_metadata(group, agreement_ranks)
            if groups[key][1]:
                occurrences.setdefault(key, []).append(position)
        if event.finalizes_outer:
            epoch += 1

    usage = [0] * len(gradient_events)
    plan = {}
    next_occurrence = dict.fromkeys(occurrences, 0)
    at_position = {}
    for key, positions in occurrences.items():
        for position in positions:
            at_position.setdefault(position, []).append(key)
    # First-fit, longest packet first; outer boundaries and the terminal cut every packet.
    for position in range(len(gradient_events)):
        for key in at_position.get(position, ()):
            positions = occurrences[key]
            start = next_occurrence[key]
            if start == len(positions) or positions[start] != position:
                continue
            candidates = positions[start : start + 4]
            candidates = [end for end in candidates if epochs[end] == epochs[position]]
            charge = groups[key][1]
            for count in range(len(candidates), 1, -1):
                end = candidates[count - 1]
                if any(value + charge > budget_bytes for value in usage[position : end + 1]):
                    continue
                for index in range(position, end + 1):
                    usage[index] += charge
                flush = gradient_events[end].event_id
                for member in candidates[:count]:
                    plan[(gradient_events[member].event_id, key[1])] = flush
                next_occurrence[key] = start + count
                break
            else:
                next_occurrence[key] = start + 1
    return plan, (trace, tuple(groups.items()), 4, budget_bytes, tuple(sorted(plan.items())))


def build_rs_plan(execution_order, agreement_group, *, device, budget_bytes):
    """Agree once on immutable packets of two to four GRAD_READY contributions.

    Keys are ``(source event ID, parameter-group ordinal)`` and values name the
    packet's final GRAD_READY event. Missing keys retain the baseline reduction.
    The budget covers admitted packets' buffers through their flush, inclusive;
    it does not cap persistent gradients, baseline staging or whole-GPU memory.
    """
    if agreement_group is None:
        raise ValueError("RS windows require an explicit agreement group.")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Build the RS window plan before CUDA graph capture.")
    error, plan, digest = None, {}, 0
    try:
        ranks = set(dist.get_process_group_ranks(agreement_group))
        records = [execution_order.seek(index) for index in range(len(execution_order))]
        plan, signature = _build_plan(records, ranks, budget_bytes)
        digest = int(hashlib.sha256(repr(signature).encode()).hexdigest()[:15], 16)
    except Exception as exc:
        error = exc
    header = torch.tensor(
        [int(error is None), len(execution_order), -len(execution_order), digest, -digest],
        dtype=torch.int64,
        device=device,
    )
    dist.all_reduce(header, op=dist.ReduceOp.MIN, group=agreement_group)
    valid, count, negative_count, minimum, negative_maximum = header.tolist()
    if not valid:
        raise ValueError(
            "Invalid FSDP RS topology, layout or window policy on an agreement rank."
        ) from error
    if count != -negative_count or minimum != -negative_maximum:
        raise RuntimeError("FSDP RS trace, layout, budget or plan differs across agreement ranks.")
    return MappingProxyType(plan)
