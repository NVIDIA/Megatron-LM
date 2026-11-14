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
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from weakref import WeakKeyDictionary

import torch
from torch import nn

from .execution_order import ExecutionEvent, ExecutionOrder
from .phase import Phase
from .runtime_policy import PrefetchOrder, RuntimePolicy


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
    _post_backward_hook_registered: bool

    def __init__(
        self,
        device: torch.device | None = None,
        runtime_policy: RuntimePolicy = RuntimePolicy(),
        parameter_to_owner: dict[nn.Parameter, int] | None = None,
        gradient_finalization_mode: GradientFinalizationMode = (
            GradientFinalizationMode.REDUCE_OUTER_AND_WAIT
        ),
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
        """
        overlap_allgather_reduction = runtime_policy.overlap_allgather_reduction
        overlap_outer_communication = runtime_policy.overlap_outer_communication
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if device.type != "cuda":
            raise ValueError(f"FsdpContext requires a CUDA device, got {device}.")

        self._gradient_finalization_mode = gradient_finalization_mode
        self.use_symmetric_memory = runtime_policy.use_symmetric_memory
        self.execution_order = ExecutionOrder()
        self._recording_execution_order = runtime_policy.prefetch_order is PrefetchOrder.RECORDED
        self._post_backward_hook_registered = False
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

    @contextmanager
    def finalize_gradients(
        self, mode: GradientFinalizationMode = GradientFinalizationMode.REDUCE_OUTER_AND_WAIT
    ) -> Iterator[None]:
        """Scope outer reduction and completion-wait behavior for backward calls.

        Use SKIP_OUTER while accumulating gradients across backward calls. For the last
        backward before the optimizer step, use REDUCE_OUTER_AND_WAIT for automatic
        completion, or REDUCE_OUTER and call finish_grad_sync() after all backward work,
        including backward_dw(). Inner reduction and weight cleanup run in every mode.

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
            or self._post_backward_hook_registered
        ):
            return
        raise RuntimeError(
            "Gradient reduction has no pending autograd completion callback. "
            "Use context.finalize_gradients(mode=GradientFinalizationMode.REDUCE_OUTER) "
            "and call context.finish_grad_sync() after all backward work, including "
            "delayed weight-gradient computation, before consuming gradients."
        )

    def finish_grad_sync(self) -> None:
        """Reshard remaining weights and wait on submitted gradient reductions.

        Call after all backward work, including delayed weight-gradient computation,
        and before reading or modifying gradients. This enqueues a stream dependency;
        it does not block the CPU or wait for reductions that have not yet been launched.
        """
        # TODO(wujingyue): Support SKIP_OUTER even on the last microbatch by launching pending
        # outer reductions here before waiting, instead of requiring backward to launch them.

        # A backward prefetch may gather a frozen module that never runs backward.
        # Restore bindings and clear its gather state so the next iteration
        # materializes fresh weights. All backward work, including delayed wgrad,
        # must be complete here.
        for module in self._module_names:
            module.reshard()
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

    def post_backward(self) -> None:
        """Finish backward cleanup and synchronize gradients when finalization is enabled."""
        self._run_delayed_post_backward_callbacks()
        self._recording_execution_order = False
        # Standalone training can call optimizer.step() immediately after backward.
        # Wait for reductions on the last microbatch; earlier microbatches keep
        # overlapping reduction with compute. MCore calls finish_grad_sync() itself
        # after its schedule completes, including delayed weight gradients.
        if self._gradient_finalization_mode is GradientFinalizationMode.REDUCE_OUTER_AND_WAIT:
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
