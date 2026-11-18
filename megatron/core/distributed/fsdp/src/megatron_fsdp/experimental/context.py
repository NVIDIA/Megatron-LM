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
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

import torch
from torch import nn

from .execution_order import ExecutionAction, ExecutionEvent, ExecutionOrder, ExecutionOrderView
from .gradient_window import build_rs_plan
from .phase import Phase
from .runtime_policy import PrefetchOrder, RuntimePolicy

if TYPE_CHECKING:
    from .dbuffer import DBuffer
    from .parameter_group import FsdpParameterGroup


@dataclass
class _RsInput:
    buffer: "DBuffer"
    ready: torch.cuda.Event
    write_stream: torch.cuda.Stream
    bound_grad_dtypes: dict[int, torch.dtype | None] = field(default_factory=dict)


class GradientFinalizationMode(enum.Enum):
    """Control outer reduction and the automatic wait for gradient reductions.

    | Mode | Outer reduction | Automatic wait at autograd completion |
    | --- | --- | --- |
    | SKIP_OUTER | Skip | No |
    | REDUCE_OUTER | Launch as gradients become ready | No |
    | REDUCE_OUTER_AND_WAIT (default) | Launch as gradients become ready | Yes |

    Inner reduction or local accumulation runs in every mode. The automatic wait
    covers both inner and outer reductions. The default applies outside a
    ``finalize_gradients()`` scope.
    """

    SKIP_OUTER = enum.auto()
    """Skip outer reduction and the automatic completion wait during accumulation."""

    REDUCE_OUTER = enum.auto()
    """Launch outer reduction; the caller must call ``finish_grad_sync()`` afterward."""

    REDUCE_OUTER_AND_WAIT = enum.auto()
    """Launch outer reduction and wait for all gradient reductions at autograd completion."""


class FsdpContext:
    """Runtime stream and prefetch state shared by FSDP roots constructed together."""

    allgather_stream: torch.cuda.Stream
    inner_reduction_stream: torch.cuda.Stream
    # Aliases inner_reduction_stream unless DP-outer overlap is enabled.
    outer_reduction_stream: torch.cuda.Stream
    # Active mode, controlled by finalize_gradients().
    _gradient_finalization_mode: GradientFinalizationMode
    use_symmetric_memory: bool
    # Prefetch uses construction order unless execution-order recording is enabled.
    execution_order: ExecutionOrder
    # Topology metadata must not keep modules alive after construction.
    _roots: WeakKeyDictionary[nn.Module, None]
    # Names are relative to each FSDP root and absent until finalization.
    _module_names: WeakKeyDictionary[nn.Module, str]
    # The optimizer runs on the current stream and must wait for reductions on
    # this context's reduce-scatter stream. Each context owns its own stream, so
    # independent roots sharing a context need only one completion callback.
    _post_autograd_hook_registered: bool

    def __init__(
        self,
        device: torch.device | None = None,
        runtime_policy: RuntimePolicy = RuntimePolicy(),
        parameter_to_owner: dict[nn.Parameter, int] | None = None,
        gradient_finalization_mode: GradientFinalizationMode = (
            GradientFinalizationMode.REDUCE_OUTER_AND_WAIT
        ),
        *,
        rs_window_budget_bytes: int = 0,
    ) -> None:
        """Create rank-local runtime state for FSDP modules on ``device``.

        Args:
            device: CUDA device on which this context schedules communication. Defaults to
                the current CUDA device.
            runtime_policy: Communication allocation, stream, and prefetch settings shared by all
                modules constructed in this context.
            parameter_to_owner: Construction-time owner assignments for TensorAtomic
                parameters, keyed by the original parameters before sharding. Owners are
                ranks in each parameter group's 1-D data-parallel mesh and must agree across
                that mesh. Every TensorAtomic parameter needs an entry; other entries are
                ignored. Tensors are packed by owner without changing logical parameter order.
            gradient_finalization_mode: Initial mode restored when a gradient-finalization
                scope exits. The default waits automatically; callers that own the wait
                can select REDUCE_OUTER and call finish_grad_sync() after backward.
            rs_window_budget_bytes: Buffer budget for grouping inner reduce-scatters. Zero
                reduces each contribution as soon as it is ready.
        """
        overlap_allgather_reduction = runtime_policy.overlap_allgather_reduction
        overlap_outer_communication = runtime_policy.overlap_outer_communication
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if device.type != "cuda":
            raise ValueError(f"FsdpContext requires a CUDA device, got {device}.")
        if rs_window_budget_bytes < 0:
            raise ValueError("The FSDP reduce-scatter window budget must be nonnegative.")
        if rs_window_budget_bytes and (
            runtime_policy.prefetch_order is not PrefetchOrder.RECORDED
            or runtime_policy.autograd_completes_execution
            or runtime_policy.use_symmetric_memory
        ):
            raise ValueError("RS windows require recorded, caller-completed, ordinary buffers.")

        self._gradient_finalization_mode = gradient_finalization_mode
        self.autograd_completes_execution = runtime_policy.autograd_completes_execution
        self.use_symmetric_memory = runtime_policy.use_symmetric_memory
        self.prefetch_order = runtime_policy.prefetch_order
        self.execution_order = ExecutionOrder()
        self.retain_mask: Mapping[int, bool] = {}
        self.rs_window_budget_bytes = rs_window_budget_bytes
        self.rs_plan: Mapping[tuple[int, int], int] | None = None
        self.rs_inputs: dict[FsdpParameterGroup, _RsInput] = {}
        self._execution_order_view: ExecutionOrderView | None = None
        self._prefetch_order_view: ExecutionOrderView | None = None
        self._recording_execution_order = runtime_policy.prefetch_order is PrefetchOrder.RECORDED
        self._post_autograd_hook_registered = False
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
    def gradient_finalization_mode(self) -> GradientFinalizationMode:
        """Active mode, controlled by ``finalize_gradients()``."""
        return self._gradient_finalization_mode

    @property
    def is_finalized(self) -> bool:
        """Whether construction has completed for this context."""
        return self._is_finalized

    @property
    def training(self) -> bool:
        """Whether any FSDP root is in training mode."""
        return any(root.training for root in self._roots)

    @property
    def is_recording_execution_order(self) -> bool:
        """Whether module calls are being recorded for subsequent prefetching."""
        return self._recording_execution_order and self.training

    @property
    def execution_order_view(self) -> ExecutionOrderView | None:
        """The complete recorded U/R sequence, consumed without wrapping."""
        view = self._execution_order_view
        if view is not None and view.source is not self.execution_order:
            raise RuntimeError("The recorded FSDP execution order was replaced.")
        return view

    @property
    def prefetch_order_view(self) -> ExecutionOrderView:
        """Independent unshard hints for speculative prefetch."""
        view = self._prefetch_order_view
        if view is None or view.source is not self.execution_order:
            view = self.execution_order.view(action=ExecutionAction.UNSHARD)
            self._prefetch_order_view = view
        return view

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

        if self.prefetch_order is PrefetchOrder.RECORDED and any(
            not module._schedule_policy.register_hooks for module in self._module_names
        ):
            raise ValueError("Recorded FSDP execution requires ordinary module hooks.")

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

    @contextmanager
    def finalize_gradients(
        self, mode: GradientFinalizationMode = GradientFinalizationMode.REDUCE_OUTER_AND_WAIT
    ) -> Iterator[None]:
        """Scope outer reduction and completion-wait behavior for backward calls.

        Use SKIP_OUTER while accumulating gradients across backward calls. For the last
        backward before the optimizer step, use REDUCE_OUTER_AND_WAIT for automatic
        completion, or REDUCE_OUTER and call finish_grad_sync() after all backward work,
        including backward_dw(). Inner reduction runs in every mode; post_backward()
        closes model execution and clears remaining gathered weights independently.

        Outside a scope, backward uses the mode selected at context construction.
        Wrap both forward and backward. The scope itself does not launch reductions
        or wait on entry or exit.

        Nested scopes are allowed. Disallowing nesting could make mode selection
        clearer, but requires substantial changes to pipeline-parallel synchronization
        scopes and is deferred to a separate PR.

        Args:
            mode: Outer reduction and completion-wait behavior within the scope.
        """
        self.ensure_finalized()
        previous_mode = self._gradient_finalization_mode
        self._gradient_finalization_mode = mode

        try:
            yield
        finally:
            self._gradient_finalization_mode = previous_mode

    def validate_grad_sync(self) -> None:
        """Require a caller-owned wait or a pending autograd completion callback."""
        if (
            self._gradient_finalization_mode is not GradientFinalizationMode.REDUCE_OUTER_AND_WAIT
            or self._post_autograd_hook_registered
        ):
            return
        raise RuntimeError(
            "Gradient reduction has no pending autograd completion callback. "
            "Use context.finalize_gradients(mode=GradientFinalizationMode.REDUCE_OUTER) "
            "and call context.finish_grad_sync() after all backward work, including "
            "delayed weight-gradient computation, before consuming gradients."
        )

    def finish_grad_sync(self) -> None:
        """Wait on submitted gradient reductions without ending model execution.

        Call after all backward work, including delayed weight-gradient computation,
        and before reading or modifying gradients. This enqueues a stream dependency;
        it does not block the CPU or wait for reductions that have not yet been launched.
        """
        # TODO(wujingyue): Support SKIP_OUTER even on the last microbatch by launching pending
        # outer reductions here before waiting, instead of requiring backward to launch them.

        self.current_stream().wait_stream(self.inner_reduction_stream)
        if self.outer_reduction_stream is not self.inner_reduction_stream:
            self.current_stream().wait_stream(self.outer_reduction_stream)

    def pre_backward(self) -> None:
        """Fork the reduce-scatter stream into any active CUDA-graph capture.

        A stream joins a capture only via a wait_stream edge; without it, the first
        allocation on the reduce-scatter stream is illegal during capture. The DP-outer
        reduction stream needs no fork: it joins via its wait on the reduce-scatter stream.
        """
        self.inner_reduction_stream.wait_stream(self.current_stream())

    def _finish_execution_order(self) -> None:
        """Close one recording or require the complete recorded replay."""
        if self._recording_execution_order:
            self._recording_execution_order = False
            self._execution_order_view = self.execution_order.view()
            self._prefetch_order_view = self.execution_order.view(action=ExecutionAction.UNSHARD)
        elif self.prefetch_order is PrefetchOrder.RECORDED:
            view = self.execution_order_view
            if view is None or view.position != len(view):
                raise RuntimeError("FSDP execution ended before the recorded order was complete.")
            view.reset()
            self.prefetch_order_view.reset()

    def post_backward(self) -> None:
        """Complete the execution order after all backwards in this schedule."""
        if not self.training:
            return
        if self.rs_inputs:
            raise RuntimeError("FSDP execution ended with unreduced RS window inputs.")
        self._finish_execution_order()
        # Speculative prefetch may gather an owner that never runs backward.
        # Final cleanup is physical work outside the logical U/R trace.
        for module in self._module_names:
            module.reshard(force=True)

    def compile_rs_window(self, agreement_group: torch.distributed.ProcessGroup) -> None:
        """Plan once after the complete eager iteration and before graph capture."""
        view = self.execution_order_view
        if view is None or view.position or self.rs_plan is not None:
            raise RuntimeError("Compile the RS window once, at the completed recording boundary.")
        self.rs_plan = build_rs_plan(
            self.execution_order,
            agreement_group,
            device=self.inner_reduction_stream.device,
            budget_bytes=self.rs_window_budget_bytes,
        )
        groups = {
            self.execution_order.seek(event_id).module.parameter_groups[index]
            for event_id, index in self.rs_plan
        }
        for group in groups:
            group.register_rs_grad_view_hooks(self)

    def post_microbatch(self) -> None:
        """Release deferred module weights after each microbatch backward."""
        # Finish children before parents to close nested backward NVTX ranges.
        while self._delayed_post_backward_callbacks:
            module, callback = self._delayed_post_backward_callbacks.pop()
            callback(module)

    def _post_autograd(self) -> None:
        """Run deferred cleanup and the configured automatic completion and gradient wait."""
        self.post_microbatch()
        self._post_autograd_hook_registered = False
        if self.autograd_completes_execution:
            self.post_backward()
        if self._gradient_finalization_mode is GradientFinalizationMode.REDUCE_OUTER_AND_WAIT:
            self.finish_grad_sync()

    def register_post_backward_hook(self) -> None:
        """Register one final callback for deferred cleanup and optional gradient sync.

        Multiple FSDP roots can share this context. Waiting for the
        reduce-scatter stream in each root's ``post_backward()`` would prevent
        one root's backward compute from overlapping another root's gradient
        reductions. Wait once at context-level autograd completion instead.
        """

        if self._post_autograd_hook_registered:
            # Another root sharing this context already queued the completion wait.
            return
        self._post_autograd_hook_registered = True

        # TODO(wujingyue): Switch to torch.autograd.graph.queue_callback() when Megatron-LM
        # requires a PyTorch version that includes it:
        # https://github.com/pytorch/pytorch/pull/193958
        torch.autograd.Variable._execution_engine.queue_callback(self._post_autograd)


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
