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

"""Shared runtime state for Megatron-FSDP modules."""

import enum
import hashlib
from collections import deque
from collections.abc import Callable
from typing import NamedTuple
from weakref import WeakKeyDictionary

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed import ProcessGroup

from .execution_order import ExecutionEvent, ExecutionOrder
from .phase import Phase
from .runtime_policy import RuntimePolicy


class PrefetchOrder(enum.Enum):
    """How to choose the module order used for prefetching."""

    REGISTRATION = enum.auto()
    RECORDED = enum.auto()


class _GradBufferBatch(NamedTuple):
    """Inputs and their last-consumer event, owned by the allocation stream."""

    buffers: tuple[torch.Tensor, ...]
    done: torch.cuda.Event
    allocation_stream: torch.cuda.Stream
    position: int = -1


class FsdpContext:
    """Runtime stream and prefetch state shared by FSDP roots constructed together."""

    allgather_stream: torch.cuda.Stream
    inner_reduction_stream: torch.cuda.Stream
    # Aliases inner_reduction_stream unless DP-outer overlap is enabled.
    outer_reduction_stream: torch.cuda.Stream
    # HFSDP/HSDP need explicit last-microbatch state. First-microbatch state is
    # unnecessary because each parameter group tracks whether model_weight is stale
    # after syncing from main_weight.
    is_last_microbatch: bool
    use_symmetric_memory: bool
    caller_managed_grad_sync: bool
    # Prefetch uses construction order unless execution-order recording is enabled.
    execution_order: ExecutionOrder
    # Topology metadata must not keep modules alive after construction.
    _roots: WeakKeyDictionary[nn.Module, None]
    # Names are relative to each FSDP root and absent until finalization.
    _module_names: WeakKeyDictionary[nn.Module, str]
    # The optimizer runs on the current stream and must wait for reductions on
    # this context's reduce-scatter stream. Each context owns its own stream, so
    # independent roots sharing a context need only one completion callback.
    _post_backward_hook_registered: bool

    def __init__(
        self,
        device: torch.device | None = None,
        runtime_policy: RuntimePolicy = RuntimePolicy(),
        parameter_to_owner: dict[nn.Parameter, int] | None = None,
        prefetch_order: PrefetchOrder = PrefetchOrder.REGISTRATION,
        prefetch_process_group: ProcessGroup | None = None,
    ) -> None:
        """Create rank-local runtime state for FSDP modules on ``device``.

        Args:
            device: CUDA device on which this context schedules communication. Defaults to
                the current CUDA device.
            runtime_policy: Communication allocation, stream, and gradient-synchronization
                settings shared by all modules constructed in this context.
            parameter_to_owner: Construction-time owner assignments for TensorAtomic
                parameters, keyed by the original parameters before sharding. Owners are
                ranks in each parameter group's 1-D data-parallel mesh and must agree across
                that mesh. Every TensorAtomic parameter needs an entry; other entries are
                ignored. Tensors are packed by owner without changing logical parameter order.
            prefetch_order: ``REGISTRATION`` expects a forward/backward pass matching module
                registration order. ``RECORDED`` gathers parameters on demand and records
                module calls until the first ``post_backward()``. With caller-managed
                gradient synchronization, recording spans the full schedule until
                ``finish_grad_sync()`` instead. Subsequent passes must repeat the complete
                sequence consistently across data-parallel ranks.
                Recording currently requires the ordinary module hooks; combined 1F1B
                bypasses them. Discard the context if the recorded pass raises.
            prefetch_process_group: Ranks sharing the recorded schedule. Synchronizes
                recorded gradient-input retirement. Without this explicit group, inputs
                are retired only in backward or gradient synchronization.
        """
        overlap_allgather_reduction = runtime_policy.overlap_allgather_reduction
        overlap_outer_communication = runtime_policy.overlap_outer_communication
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if device.type != "cuda":
            raise ValueError(f"FsdpContext requires a CUDA device, got {device}.")

        self.is_last_microbatch = True
        self.use_symmetric_memory = runtime_policy.use_symmetric_memory
        self.caller_managed_grad_sync = runtime_policy.caller_managed_grad_sync
        self.prefetch_order = prefetch_order
        self._prefetch_process_group = prefetch_process_group
        self.execution_order = ExecutionOrder()
        self._recording_execution_order = prefetch_order is PrefetchOrder.RECORDED
        self._post_backward_hook_registered = False
        self._pending_grad_buffers: deque[_GradBufferBatch] = deque()
        # Forward occurrence -> producer occurrences whose gradient inputs can retire.
        self._grad_buffer_release_plan: dict[int, tuple[int, ...]] = {}
        self._recorded_grad_buffer_positions: list[int] = []
        self._delayed_post_backward_callbacks: list[
            tuple[nn.Module, Callable[[nn.Module], None]]
        ] = []
        # Construction-only; empty after finalization.
        self._registered_modules: list[nn.Module] = []
        self._roots = WeakKeyDictionary()
        self._module_names = WeakKeyDictionary()
        self.parameter_to_owner = parameter_to_owner
        self._is_finalized = False
        # The two overlap options independently split streams from the inner
        # reduction stream. With (overlap_allgather_reduction,
        # overlap_outer_communication), the assignments are:
        #   (False, False): all-gather and both reductions share one stream.
        #   (True, False):  all-gather is separate; both reductions share a stream.
        #   (False, True):  all-gather and inner reduction share; outer is separate.
        #   (True, True):   all-gather, inner reduction, and outer reduction each
        #                  have a separate stream.
        # Outer all-gather separation is not implemented yet; the outer option
        # currently affects only reduction.
        self.inner_reduction_stream = torch.cuda.Stream(device)
        self.allgather_stream = (
            torch.cuda.Stream(device)
            if overlap_allgather_reduction
            else self.inner_reduction_stream
        )
        self.outer_reduction_stream = (
            torch.cuda.Stream(device)
            if overlap_outer_communication
            else self.inner_reduction_stream
        )

    @property
    def is_finalized(self) -> bool:
        """Whether construction has completed for this context."""
        return self._is_finalized

    @property
    def is_recording_execution_order(self) -> bool:
        """Whether module calls are being recorded for subsequent prefetching."""
        return self._recording_execution_order

    def register_module(self, module: nn.Module) -> None:
        """Register a module after its FSDP initialization completes in this context."""
        if self._is_finalized:
            raise RuntimeError("Cannot register an FSDP module after its context is finalized.")
        self._registered_modules.append(module)

    def finalize(self) -> None:
        """Finalize roots, names, and cross-root prefetch orders."""
        if self._is_finalized:
            raise RuntimeError("FSDP context is already finalized.")

        # Successful FsdpModule initialization registers the module, and fully_shard
        # rejects children from another context. Membership identifies this context's
        # FSDP modules without importing FsdpModule.
        registered_modules = set(self._registered_modules)
        children: set[nn.Module] = set()
        for module in self._registered_modules:
            _collect_fsdp_children(module, registered_modules, children)
        for root in self._registered_modules:
            if root in children:
                continue
            self._roots[root] = None
            for name, module in root.named_modules():
                if module not in registered_modules:
                    continue
                self._module_names[module] = name
                if not self._recording_execution_order:
                    self.execution_order.append(ExecutionEvent(module, Phase.FORWARD))

        if not self._recording_execution_order:
            for root in reversed(list(self._roots)):
                _collect_backward_order(root, registered_modules, self.execution_order)

        self._registered_modules.clear()
        self.parameter_to_owner = None
        self._is_finalized = True

    def is_root(self, module: nn.Module) -> bool:
        """Return whether ``module`` is an outermost FSDP module in this context."""
        self.ensure_finalized()
        return module in self._roots

    def module_name(self, module: nn.Module) -> str:
        """Return ``module``'s name relative to its FSDP root."""
        self.ensure_finalized()
        return self._module_names[module]

    def ensure_finalized(self) -> None:
        """Raise if construction has not completed for this context."""
        if not self._is_finalized:
            raise RuntimeError(
                "FSDP context is not finalized. Exit fully_shard_context to complete construction."
            )

    def current_stream(self) -> torch.cuda.Stream:
        """Current stream on this context's device."""
        return torch.cuda.current_stream(self.allgather_stream.device)

    def validate_grad_sync(self) -> None:
        """Require a caller-owned wait or a pending autograd completion callback."""
        if self.caller_managed_grad_sync or self._post_backward_hook_registered:
            return
        raise RuntimeError(
            "Gradient reduction has no pending autograd completion callback. "
            "Use FsdpContext(runtime_policy=RuntimePolicy(caller_managed_grad_sync=True)) "
            "and call context.finish_grad_sync() after all backward work, including "
            "delayed weight-gradient computation, before consuming gradients."
        )

    def finish_grad_sync(self) -> None:
        """Wait on submitted gradient reductions.

        Call after all backward work, including delayed weight-gradient computation,
        and before reading or modifying gradients. This enqueues a stream dependency;
        it does not block the CPU or wait for reductions that have not yet been launched.
        """
        self.current_stream().wait_stream(self.inner_reduction_stream)
        if self.outer_reduction_stream is not self.inner_reduction_stream:
            self.current_stream().wait_stream(self.outer_reduction_stream)
        self._release_grad_buffers()
        if self.caller_managed_grad_sync:
            self._finish_execution_order()

    def _retain_grad_buffers(
        self,
        buffers: list[torch.Tensor],
        allocation_stream: torch.cuda.Stream,
        *,
        position: int | None = None,
    ) -> None:
        """Keep module inputs alive through inner reduction and local accumulation."""
        if buffers:
            # DP-outer reduction consumes persistent main_grad, not these inputs.
            done = self.inner_reduction_stream.record_event()
            if position is None:
                order = self.execution_order
                position = (
                    len(order) - 1
                    if self._recording_execution_order
                    else ((order.position - 1) % len(order) if len(order) else -1)
                )
            if self._recording_execution_order:
                self._recorded_grad_buffer_positions.append(position)
            self._pending_grad_buffers.append(
                _GradBufferBatch(tuple(buffers), done, allocation_stream, position)
            )

    def _release_grad_buffers(self, *, keep: int = 0) -> None:
        """Order storage reuse on its allocation stream before dropping references."""
        while len(self._pending_grad_buffers) > keep:
            batch = self._pending_grad_buffers.popleft()
            batch.allocation_stream.wait_event(batch.done)

    def _release_grad_buffers_in_forward(self) -> None:
        """Record completed inputs, then replay the group-wide retirement plan."""
        if self.prefetch_order is not PrefetchOrder.RECORDED or not self._pending_grad_buffers:
            return
        if self._recording_execution_order:
            with torch.cuda.device(self.allgather_stream.device):
                if torch.cuda.is_current_stream_capturing():
                    return
            positions = []
            while self._pending_grad_buffers and self._pending_grad_buffers[0].done.query():
                positions.append(self._pending_grad_buffers.popleft().position)
            if positions:
                self._grad_buffer_release_plan[len(self.execution_order)] = tuple(positions)
        else:
            positions = self._grad_buffer_release_plan.get(self.execution_order.position, ())
            # Backward's bounded staging may already have retired a planned input.
            # Match producers instead of applying a count to a different FIFO batch.
            while (
                self._pending_grad_buffers and self._pending_grad_buffers[0].position in positions
            ):
                self._release_grad_buffers(keep=len(self._pending_grad_buffers) - 1)

    def _compile_grad_buffer_release_plan(self) -> None:
        """Agree once on each input's latest observed retirement occurrence."""
        group = self._prefetch_process_group
        if group is None:
            self._grad_buffer_release_plan.clear()
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Gradient-input retirement requires one eager recording iteration.")
        events = list(self.execution_order)
        indices = {module: index for index, module in enumerate(self._module_names)}
        signature = repr(
            (
                [(indices[event.module], event.module.name, event.phase.name) for event in events],
                self._recorded_grad_buffer_positions,
            )
        )
        digest = int.from_bytes(hashlib.sha256(signature.encode()).digest()[:8], "little")
        digest &= (1 << 62) - 1
        valid = len(set(self._recorded_grad_buffer_positions)) == len(
            self._recorded_grad_buffer_positions
        ) and all(0 <= position < len(events) for position in self._recorded_grad_buffer_positions)
        header = torch.tensor(
            [len(events), -len(events), digest, -digest, int(valid)],
            dtype=torch.int64,
            device=self.allgather_stream.device,
        )
        dist.all_reduce(header, op=dist.ReduceOp.MIN, group=group)
        values = header.tolist()
        if values[0] != -values[1] or values[2] != -values[3] or not values[4]:
            raise RuntimeError(
                "Recorded FSDP events or gradient-input producers differ across ranks."
            )

        # A missing observation means this rank kept the input until backward/grad sync.
        # MAX therefore retains it unless every rank observed a forward retirement.
        release_positions = torch.full(
            (len(events),), len(events), dtype=torch.int64, device=self.allgather_stream.device
        )
        for position, producers in self._grad_buffer_release_plan.items():
            for producer in producers:
                release_positions[producer] = position
        dist.all_reduce(release_positions, op=dist.ReduceOp.MAX, group=group)
        plan: dict[int, list[int]] = {}
        for producer, position in enumerate(release_positions.tolist()):
            if position < len(events):
                plan.setdefault(position, []).append(producer)
        self._grad_buffer_release_plan = {
            position: tuple(producers) for position, producers in plan.items()
        }

    def _finish_execution_order(self) -> None:
        """Finish recording or validate replay at the execution-window boundary."""
        if self._recording_execution_order:
            # A synchronization before any compute must not end recording.
            if self.execution_order.peek() is not None:
                self._compile_grad_buffer_release_plan()
                self._recording_execution_order = False
        elif self.prefetch_order is PrefetchOrder.RECORDED and self.execution_order.position != 0:
            raise RuntimeError("FSDP execution ended before the recorded order was complete.")

    def post_backward(self) -> None:
        """Finish backward cleanup and automatically synchronize the last microbatch."""
        self._run_delayed_post_backward_callbacks()
        if not self.caller_managed_grad_sync:
            self._finish_execution_order()
        # Standalone training can call optimizer.step() immediately after backward.
        # Wait for reductions on the last microbatch; earlier microbatches keep
        # overlapping reduction with compute. MCore calls finish_grad_sync() itself
        # after its schedule completes, including delayed weight gradients.
        if not self.caller_managed_grad_sync and self.is_last_microbatch:
            self.finish_grad_sync()
        self._post_backward_hook_registered = False

    def _run_delayed_post_backward_callbacks(self) -> None:
        """Run deferred module completion callbacks once."""
        # Finish children before parents to close nested backward NVTX ranges.
        while self._delayed_post_backward_callbacks:
            module, callback = self._delayed_post_backward_callbacks.pop()
            callback(module)

    def register_post_backward_hook(self) -> None:
        """Register one final callback for deferred cleanup and optional gradient sync.

        Multiple FSDP roots can share this context. Waiting for the
        reduce-scatter stream in each root's ``post_backward()`` would prevent
        one root's backward compute from overlapping another root's gradient
        reductions. Wait once at context-level autograd completion instead.
        """

        if self._post_backward_hook_registered:
            # Another root sharing this context already queued the completion wait.
            return
        self._post_backward_hook_registered = True

        # TODO(wujingyue): Switch to torch.autograd.graph.queue_callback() when Megatron-LM
        # requires a PyTorch version that includes it:
        # https://github.com/pytorch/pytorch/pull/193958
        torch.autograd.Variable._execution_engine.queue_callback(self.post_backward)


def _collect_backward_order(
    module: nn.Module, registered_modules: set[nn.Module], order: ExecutionOrder
) -> None:
    """Collect one root's static backward prefetch order."""
    if module in registered_modules:
        order.append(ExecutionEvent(module, Phase.BACKWARD))

    for child in reversed(list(module.children())):
        _collect_backward_order(child, registered_modules, order)


def _collect_fsdp_children(
    module: nn.Module, registered_modules: set[nn.Module], children: set[nn.Module]
) -> None:
    """Collect the nearest registered FSDP descendants of ``module``."""
    for child in module.children():
        if child in registered_modules:
            children.add(child)
        else:
            _collect_fsdp_children(child, registered_modules, children)
