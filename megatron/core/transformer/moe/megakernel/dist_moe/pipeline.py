# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Activation leases for the ordinary, sequential PP/VPP schedules."""

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType

import torch


@dataclass(frozen=True)
class PipelinePlan:
    """One physical slot per live (model chunk, microbatch) invocation."""

    events: tuple
    slots: MappingProxyType
    num_slots: int


def plan_activation_slots(schedule_table, num_warmup, forward_only=False, tracked_chunks=None):
    """Color live intervals using the same table and warmup as the native schedule.

    The backward table reverses chunk IDs, but keeps each chunk's microbatches
    in FIFO order. Computation is sequential even with P2P communication overlap.
    """
    table = tuple(schedule_table)
    chunks = tuple(
        sorted({chunk for _, chunk in table} if tracked_chunks is None else set(tracked_chunks))
    )
    return _plan_activation_slots(table, num_warmup, forward_only, chunks)


@lru_cache(maxsize=128)
def _plan_activation_slots(table, num_warmup, forward_only, tracked_chunks):
    if not table or not 0 <= num_warmup <= len(table):
        raise ValueError("dist_moe requires a nonempty schedule and valid warmup")
    num_chunks = max(chunk for _, chunk in table) + 1
    forwards = tuple(("forward", chunk, mb) for mb, chunk in table)
    if forward_only:
        return PipelinePlan(forwards, MappingProxyType({}), 1)
    backwards = tuple(("backward", num_chunks - chunk - 1, mb) for mb, chunk in table)
    events = list(forwards[:num_warmup])
    for forward, backward in zip(forwards[num_warmup:], backwards):
        events.extend((forward, backward))
    events.extend(backwards[len(table) - num_warmup :])
    slots, live = {}, {}
    num_slots = 0
    for direction, chunk, mb in events:
        if chunk not in tracked_chunks:
            continue
        key = (chunk, mb)
        if direction == "forward":
            if key in slots:
                raise ValueError(f"duplicate dist_moe forward {key}")
            slot = next((i for i in range(num_slots) if i not in live.values()), num_slots)
            num_slots = max(num_slots, slot + 1)
            slots[key] = live[key] = slot
        elif key not in live:
            raise ValueError(f"dist_moe backward precedes forward {key}")
        else:
            del live[key]
    if live:
        raise ValueError("dist_moe schedule has unfinished forwards")
    return PipelinePlan(tuple(events), MappingProxyType(slots), max(1, num_slots))


class _LayerTicket:
    def __init__(self, schedule, key, module, stack) -> None:
        self.schedule, self.key, self.module, self.stack = schedule, key, module, stack
        self.started = self.completed = False

    def before_backward(self):
        """Reject an invalid consumption order before launching Dist kernels."""
        if (
            self.started
            or self.schedule.current != ("backward", *self.key)
            or not self.stack
            or next((ticket for ticket in reversed(self.stack) if not ticket.started), None)
            is not self
        ):
            raise RuntimeError("dist_moe backward must consume each slot in reverse layer order")
        self.started = True

    def complete(self):
        """Mark this layer consumed; the scheduler releases the entire invocation."""
        if not self.started or self.completed or not self.stack or self.stack[-1] is not self:
            raise RuntimeError("dist_moe layer completion is missing, repeated, or out of order")
        self.stack.pop()
        self.completed = True


class DistMoEPipelineSchedule:
    """Bind one native schedule execution to independent activation slots.

    The object itself identifies this scheduler execution, independently of
    optimizer step numbers. A failed execution stays attached to its modules,
    so an unfinished backward cannot silently leak into another schedule.
    """

    def __init__(self, model_chunks, schedule_table, num_warmup, forward_only=False) -> None:
        from .runtime import _backends, configure_dist_moe_activation_slots

        chunks = model_chunks if isinstance(model_chunks, (list, tuple)) else [model_chunks]
        # Native schedules also accept duck-typed models without Module.modules().
        chunks = [chunk for chunk in chunks if isinstance(chunk, torch.nn.Module)]
        self.modules = _backends(chunks)
        self.current = None
        self.forward_only = forward_only
        self.position = 0
        self.leases = {}
        if not self.modules:
            return
        if any(getattr(module, "_pipeline_schedule", None) is not None for module in self.modules):
            raise RuntimeError("dist_moe previous pipeline schedule did not finish")
        pools = {}
        for module in self.modules:
            if module._runtime_pool is None:
                raise RuntimeError("call initialize_dist_moe_runtime before the pipeline schedule")
            pools.setdefault(id(module._runtime_pool), set()).add(module._chunk_index)
        self.pool_plans = {
            pool: plan_activation_slots(schedule_table, num_warmup, forward_only, chunks)
            for pool, chunks in pools.items()
        }
        # Every pool shares the native event order, but colors only its own chunks.
        self.plan = next(iter(self.pool_plans.values()))
        configure_dist_moe_activation_slots(
            chunks, {pool: plan.num_slots for pool, plan in self.pool_plans.items()}
        )
        for module in self.modules:
            module._pipeline_schedule = self

    @contextmanager
    def _step(self, direction, chunk, microbatch):
        if not self.modules:
            yield
            return
        event = (direction, chunk, microbatch)
        if self.current is not None or self.position >= len(self.plan.events):
            raise RuntimeError("dist_moe pipeline schedule was reentered or exceeded its plan")
        if self.plan.events[self.position] != event:
            raise RuntimeError(f"dist_moe expected {self.plan.events[self.position]}, got {event}")
        key = (chunk, microbatch)
        if direction == "forward" and not self.forward_only:
            for plan in self.pool_plans.values():
                if key in plan.slots and plan.slots[key] in (
                    plan.slots[k] for k in self.leases if k in plan.slots
                ):
                    raise RuntimeError("dist_moe cannot reuse a live activation slot")
            self.leases[key] = {}
        self.current = event
        # Evaluation has no backward and must never allocate a training lease.
        with torch.no_grad() if self.forward_only else nullcontext():
            yield
        if not self.forward_only:
            stacks = self.leases[key]
            if direction == "forward":
                observed = {ticket.module for stack in stacks.values() for ticket in stack}
                expected = {module for module in self.modules if module._chunk_index == chunk}
                if observed != expected:
                    raise RuntimeError("dist_moe pipeline forward skipped or repeated a layer")
            else:
                if any(stacks.values()):
                    raise RuntimeError("dist_moe pipeline backward left unfinished layers")
                del self.leases[key]
        self.current = None
        self.position += 1

    def forward_step(self, function, *args, **kwargs):
        """Run the native forward step without changing its communication hooks."""
        with self._step("forward", kwargs.get("vp_stage", 0), kwargs["current_microbatch"]):
            return function(*args, **kwargs)

    def backward_step(self, function, chunk, microbatch, *args, **kwargs):
        """Keep the lease until the complete native backward, including owner hooks."""
        with self._step("backward", chunk, microbatch):
            return function(*args, **kwargs)

    def graph_key(self, module):
        """Bind captured runners to the schedule order and immutable slot view."""
        if self.current is None or self.current[0] != "forward":
            raise RuntimeError("dist_moe forward must run inside its native pipeline schedule")
        _, chunk, microbatch = self.current
        if module not in self.modules or module._chunk_index != chunk:
            raise RuntimeError("dist_moe forward used a module from another pipeline chunk")
        if self.forward_only:
            return None
        depth = sum(
            other.context is module.context and other._chunk_index == chunk
            for other in self.modules
        )
        plan = self.pool_plans[id(module._runtime_pool)]
        return self.plan.events, plan.slots[(chunk, microbatch)], depth

    def register_forward(self, module):
        """Select the invocation's slot and return its per-layer completion ticket."""
        graph_key = self.graph_key(module)
        if graph_key is None:
            return None
        if not torch.is_grad_enabled():
            raise RuntimeError("dist_moe training pipeline forward requires gradients")
        _, chunk, microbatch = self.current
        key = (chunk, microbatch)
        stack = self.leases[key].setdefault(id(module.context), [])
        if any(ticket.module is module for ticket in stack):
            raise RuntimeError("dist_moe pipeline layer executed twice in one invocation")
        module.context.select_activation_slot(*graph_key[1:])
        ticket = _LayerTicket(self, key, module, stack)
        stack.append(ticket)
        return ticket

    def finish(self):
        """Require a drained execution before final gradient synchronization."""
        if not self.modules:
            return
        if self.current is not None or self.leases or self.position != len(self.plan.events):
            raise RuntimeError("dist_moe pipeline schedule did not drain")
        for module in self.modules:
            module._pipeline_schedule = None


def begin_forward(module):
    """Use pipeline leases for PP/VPP; keep the existing PP1 execution contract."""
    schedule = getattr(module, "_pipeline_schedule", None)
    if schedule is not None:
        return schedule.register_forward(module)
    if module.config.pipeline_model_parallel_size > 1:
        raise RuntimeError("dist_moe PP/VPP requires the native pipeline schedule")
    return None
