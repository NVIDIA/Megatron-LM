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

"""Reuse gathered weights within the normal single-successor prefetch window."""

import hashlib
from types import MappingProxyType

import torch
import torch.distributed as dist

from .execution_order import ExecutionAction
from .phase import Phase


def build_retain_mask(execution_order, agreement_group, *, device):
    """Share two buffers between compute, prefetch and retained weights.

    Preserve single-successor prefetch and admit complete release-to-next-use gaps
    within its owner-slot envelope. Nested execution may normally need more than
    two owners; retention does not enlarge that envelope. Zero-byte aliases use
    no gathered buffer. The returned mask owns no modules.
    """
    if agreement_group is None:
        raise ValueError("Weight retention requires an explicit agreement group.")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Build the weight-retention mask before CUDA graph capture.")

    records = [execution_order.seek(index) for index in range(len(execution_order))]
    owner_indices, policies, trace = {}, {}, []
    for index, event in enumerate(records):
        if event is None:
            trace.append((index, None))
            continue
        owner_index = owner_indices.setdefault(event.module, len(owner_indices))
        policy = getattr(event.module, "_schedule_policy", None)
        policies[event.module] = tuple(
            getattr(policy, name, None)
            for name in ("forward_prefetch_size", "backward_prefetch_size")
        )
        trace.append(
            (
                index,
                owner_index,
                event.phase,
                event.action,
                event.rowwise_only,
                event.unsharded_bytes,
            )
        )
    digest = int(
        hashlib.sha256(repr((trace, tuple(policies.values()))).encode()).hexdigest()[:15], 16
    )
    header = torch.tensor(
        [len(records), -len(records), digest, -digest], dtype=torch.int64, device=device
    )
    dist.all_reduce(header, op=dist.ReduceOp.MIN, group=agreement_group)
    values = header.tolist()
    if values[0] != -values[1] or values[2] != -values[3]:
        raise RuntimeError("FSDP weight trace or prefetch policy differs across ranks.")
    if any(size not in (None, 0) for sizes in policies.values() for size in sizes):
        raise ValueError("Weight retention requires single-successor prefetch (None or 0).")

    sizes = {}
    for event in records:
        if event is None:
            raise RuntimeError("An FSDP weight owner expired before mask planning.")
        if event.action not in (ExecutionAction.UNSHARD, ExecutionAction.RESHARD):
            raise ValueError("Weight-retention planning requires UNSHARD/RESHARD events.")
        if event.module in sizes and sizes[event.module] != event.unsharded_bytes:
            raise RuntimeError("A weight owner's recorded unsharded_bytes changed.")
        sizes[event.module] = event.unsharded_bytes

    successors, next_uses, reuse_ends = {}, {}, {}
    successor = None
    for index in reversed(range(len(records))):
        event = records[index]
        if event.action is ExecutionAction.UNSHARD:
            successors[index] = successor
            next_uses[event.module] = index
            successor = event.module
        elif event.module in next_uses:
            reuse_ends[index] = next_uses[event.module]

    # Simulate demand and any configured successor before admitting retained gaps.
    live, usage, limits = set(), [], []
    for index, event in enumerate(records):
        if event.action is ExecutionAction.UNSHARD:
            live.add(event.module)
            sizes_by_phase = policies[event.module]
            prefetch = (event.phase is Phase.FORWARD and sizes_by_phase[0] is None) or (
                event.phase is Phase.BACKWARD and sizes_by_phase[1] is None
            )
            if prefetch and (successor := successors[index]) is not None:
                live.add(successor)
        elif not event.rowwise_only:
            live.discard(event.module)
        buffered = {owner for owner in live if sizes[owner] != 0}
        usage.append(buffered)
        limits.append(max(2, len(buffered)))

    retain_mask = {}
    for start, event in enumerate(records):
        end = reuse_ends.get(start)
        if end is None or event.event_id in retain_mask:
            continue
        owner = {event.module} if sizes[event.module] != 0 else set()
        if any(len(usage[index] | owner) > limits[index] for index in range(start, end)):
            continue
        for index in range(start, end):
            usage[index].update(owner)
            release = records[index]
            if release.module is event.module and release.action is ExecutionAction.RESHARD:
                retain_mask[release.event_id] = True
    return MappingProxyType(retain_mask)
