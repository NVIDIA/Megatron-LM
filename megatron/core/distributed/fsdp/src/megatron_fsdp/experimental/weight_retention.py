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

"""Plan weight reuse and prefetch within the standard double-buffer window."""

import hashlib
from collections.abc import Mapping
from types import MappingProxyType

import torch
import torch.distributed as dist

from .execution_order import ExecutionAction, ExecutionOrder
from .phase import Phase


def build_weight_plan(
    execution_order: ExecutionOrder,
    agreement_group: dist.ProcessGroup | None,
    *,
    device: torch.device | str,
) -> tuple[Mapping[int, bool], Mapping[int, int]]:
    """Return a retain mask and a mapping from unshard anchors to prefetch targets.

    Reserve the standard single-successor prefetch window before retaining
    weights. Prefetches can move earlier within that window. Nested execution
    can have more than two resident owners including the prefetched successor.
    """
    if agreement_group is None:
        raise ValueError("Weight retention requires an explicit agreement group.")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Build the weight plan before CUDA graph capture.")

    records = [execution_order.seek(index) for index in range(len(execution_order))]
    owner_indices, trace = {}, []
    for index, event in enumerate(records):
        if event is None:
            trace.append((index, None))
            continue
        owner_index = owner_indices.setdefault(event.module, len(owner_indices))
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
    digest = int(hashlib.sha256(repr(trace).encode()).hexdigest()[:15], 16)
    header = torch.tensor(
        [len(records), -len(records), digest, -digest], dtype=torch.int64, device=device
    )
    dist.all_reduce(header, op=dist.ReduceOp.MIN, group=agreement_group)
    values = header.tolist()
    if values[0] != -values[1] or values[2] != -values[3]:
        raise RuntimeError("FSDP weight trace differs across ranks.")

    sizes = {}
    for event in records:
        if event is None:
            raise RuntimeError("An FSDP weight owner expired before mask planning.")
        if event.action not in (ExecutionAction.UNSHARD, ExecutionAction.RESHARD):
            raise ValueError("Weight-retention planning requires UNSHARD/RESHARD events.")
        if event.module in sizes and sizes[event.module] != event.unsharded_bytes:
            raise RuntimeError("A weight owner's recorded unsharded_bytes changed.")
        sizes[event.module] = event.unsharded_bytes

    next_uses, reuse_ends, successors = {}, {}, {}
    successor = None
    for index in reversed(range(len(records))):
        event = records[index]
        if event.action is ExecutionAction.UNSHARD:
            next_uses[event.module] = index
            successors[index] = successor
            successor = event.module
        elif event.module in next_uses:
            reuse_ends[index] = next_uses[event.module]

    # A rowwise release can leave columnwise storage resident. An alias uses no buffer.
    live, usage, limits = set(), [], []
    for index, event in enumerate(records):
        if event.action is ExecutionAction.UNSHARD:
            if sizes[event.module] != 0:
                live.add(event.module)
            successor = successors[index]
            if successor is not None and sizes[successor] != 0:
                live.add(successor)
        elif event.action is ExecutionAction.RESHARD and not event.rowwise_only:
            live.discard(event.module)
        usage.append(live.copy())
        limits.append(max(2, len(live)))

    def fits(owner, index):
        return owner in usage[index] or len(usage[index]) < limits[index]

    retain_mask = {}
    for start in sorted(reuse_ends):
        end = reuse_ends[start]
        owner = records[start].module
        if sizes[owner] == 0:
            retain_mask[start] = True
            continue
        if not all(fits(owner, index) for index in range(start, end)):
            continue
        retain_mask[start] = True
        for index in range(start, end):
            usage[index].add(owner)

    ready, last_release, prefetch_at = {}, {}, {}
    previous_miss = 0
    for end, event in enumerate(records):
        owner = event.module
        available = ready.setdefault(owner, set())
        if event.action is ExecutionAction.RESHARD:
            if end not in retain_mask:
                last_release[owner] = end
                if event.rowwise_only:
                    available.discard(Phase.FORWARD)
                else:
                    available.clear()
            continue

        # Treat planes independently. The runtime can also reuse BF16 across phases.
        requested = {Phase.FORWARD, Phase.BACKWARD} if event.phase is None else {event.phase}
        if sizes[owner] != 0 and not requested <= available:
            # Intervals between successive misses are disjoint. Within each interval
            # use the earliest unshard that can carry the target until its demand.
            first = max(previous_miss, last_release.get(owner, -1) + 1)
            anchor = None
            for start in range(end - 1, first - 1, -1):
                if not fits(owner, start):
                    break
                if records[start].action is ExecutionAction.UNSHARD:
                    anchor = start
            if anchor is not None:
                prefetch_at[anchor] = end
            previous_miss = end
        available.update(requested)

    return MappingProxyType(retain_mask), MappingProxyType(prefetch_at)
