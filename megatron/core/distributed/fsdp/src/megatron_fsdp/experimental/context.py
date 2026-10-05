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

from contextvars import ContextVar, Token
from weakref import WeakKeyDictionary

import torch
from torch import nn

from .indexed_order import IndexedOrder

_FSDP_CONTEXT = ContextVar["FsdpContext | None"]("mfsdp_context", default=None)


class FsdpContext:
    """Runtime stream and prefetch state shared by FSDP roots constructed together."""

    allgather_stream: torch.cuda.Stream
    reduce_scatter_stream: torch.cuda.Stream
    # HFSDP/HSDP need explicit last-microbatch state. First-microbatch state is
    # unnecessary because each parameter group tracks whether model_weight is stale
    # after syncing from main_weight.
    is_last_microbatch: bool
    use_symmetric_memory: bool
    unify_communication_stream: bool
    caller_managed_grad_sync: bool
    """Leave gradient synchronization to the caller instead of an autograd callback.

    The caller must call ``finish_grad_sync()`` after all gradient producers,
    including delayed weight-gradient computation, and before consuming gradients.
    """
    # Static orders used to drive all-gather prefetch. We may want to switch to
    # capturing runtime order if static module order proves too fragile. Each
    # FsdpModule tracks its own materialized state via ``FsdpModule._unshard_event``.
    forward_order: IndexedOrder[nn.Module]
    backward_order: IndexedOrder[nn.Module]
    # Topology metadata must not keep modules alive after construction.
    _roots: IndexedOrder[nn.Module]
    # Names are relative to each FSDP root and absent until finalization.
    _module_names: WeakKeyDictionary[nn.Module, str]
    # The optimizer runs on the current stream and must wait for reductions on
    # this context's reduce-scatter stream. Each context owns its own stream, so
    # independent roots sharing a context need only one completion callback.
    _post_backward_hook_registered: bool

    def __init__(
        self,
        device: torch.device | None = None,
        use_symmetric_memory: bool = False,
        unify_communication_stream: bool = False,
        parameter_to_owner: dict[nn.Parameter, int] | None = None,
        caller_managed_grad_sync: bool = False,
    ) -> None:
        """Create rank-local runtime state for FSDP modules on ``device``.

        Args:
            device: CUDA device on which this context schedules communication. Defaults to
                the current CUDA device.
            use_symmetric_memory: Whether modules constructed in this context allocate
                communication staging buffers from PyTorch's NCCL symmetric-memory pool.
            unify_communication_stream: Whether all-gathers and reduce-scatters share one
                communication stream to reduce peak transient memory. See
                https://github.com/NVIDIA/Megatron-LM/issues/6471.
            parameter_to_owner: Construction-time owner assignments for TensorAtomic
                parameters, keyed by the original parameters before sharding. Owners are
                ranks in each parameter group's 1-D data-parallel mesh and must agree across
                that mesh. Every TensorAtomic parameter needs an entry; other entries are
                ignored. Tensors are packed by owner without changing logical parameter order.
            caller_managed_grad_sync: Disable the automatic autograd completion callback,
                allowing delayed weight gradients or custom backward schedules. The caller must
                call ``finish_grad_sync()`` after all backward work and before reading or
                modifying gradients.
        """
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if device.type != "cuda":
            raise ValueError(
                f"fully_shard_context/FsdpContext requires a CUDA device, got {device}."
            )

        self.is_last_microbatch = True
        self.use_symmetric_memory = use_symmetric_memory
        self.unify_communication_stream = unify_communication_stream
        self.caller_managed_grad_sync = caller_managed_grad_sync
        self.forward_order = IndexedOrder()
        self.backward_order = IndexedOrder()
        self._post_backward_hook_registered = False
        # Construction-only; empty after finalization.
        self._registered_modules: list[nn.Module] = []
        self._roots = IndexedOrder()
        self._module_names = WeakKeyDictionary()
        self.parameter_to_owner = parameter_to_owner
        self._is_finalized = False
        self._context_token: Token[FsdpContext | None] | None = None
        self.allgather_stream = torch.cuda.Stream(device)
        if unify_communication_stream:
            # A unified stream lets an all-gather reuse the storage released by a
            # preceding reduce-scatter.
            self.reduce_scatter_stream = self.allgather_stream
        else:
            self.reduce_scatter_stream = torch.cuda.Stream(device)

    def __enter__(self) -> "FsdpContext":
        """Activate this context for FSDP module construction."""
        if _FSDP_CONTEXT.get() is not None:
            raise RuntimeError("fully_shard_context does not support nesting.")
        if self._is_finalized:
            raise RuntimeError("Cannot enter fully_shard_context after construction is finalized.")
        self._context_token = _FSDP_CONTEXT.set(self)
        return self

    def __exit__(self, exc_type: type[BaseException] | None, *_: object) -> None:
        """Finalize successful construction and always clear the active scope."""
        assert self._context_token is not None
        try:
            if exc_type is None:
                self.finalize()
        finally:
            _FSDP_CONTEXT.reset(self._context_token)
            self._context_token = None

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
            self._roots.append(root)
            for name, module in root.named_modules():
                if module not in registered_modules:
                    continue
                self._module_names[module] = name
                self.forward_order.append(module)

        for root in reversed(self._roots):
            _collect_backward_order(root, registered_modules, self.backward_order)

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
            "Use fully_shard_context(caller_managed_grad_sync=True) and call "
            "context.finish_grad_sync() after all backward work, including "
            "delayed weight-gradient computation, before consuming gradients."
        )

    def finish_grad_sync(self) -> None:
        """Order current-stream consumers after all gradient reductions submitted so far.

        Call after all backward work, including delayed weight-gradient computation,
        and before reading or modifying gradients. This enqueues a stream dependency;
        it does not block the CPU or wait for reductions that have not yet been launched.
        """
        self.current_stream().wait_stream(self.reduce_scatter_stream)

    def post_backward(self) -> None:
        """Order current-stream consumers after this backward's gradient reductions."""
        self.finish_grad_sync()
        self._post_backward_hook_registered = False

    def register_post_backward_hook(self) -> None:
        """Register one final callback unless the caller manages gradient synchronization.

        Multiple FSDP roots can share this context. Waiting for the
        reduce-scatter stream in each root's ``post_backward()`` would prevent
        one root's backward compute from overlapping another root's gradient
        reductions. Wait once at context-level autograd completion instead.
        """

        if self.caller_managed_grad_sync:
            # Leave the wait to the caller's finish_grad_sync(), after all backward work,
            # including delayed weight-gradient computation, has been launched.
            return
        if self._post_backward_hook_registered:
            # Another root sharing this context already queued the completion wait.
            return
        self._post_backward_hook_registered = True

        # TODO(wujingyue): Switch to torch.autograd.graph.queue_callback() when Megatron-LM
        # requires a PyTorch version that includes it:
        # https://github.com/pytorch/pytorch/pull/193958
        torch.autograd.Variable._execution_engine.queue_callback(self.post_backward)


def current_fully_shard_context() -> FsdpContext | None:
    """Return the innermost active ``fully_shard_context``, or ``None``.

    Read-only counterpart of :func:`fully_shard_context`: it never creates, joins, or
    finalizes a context, and returns ``None`` whenever no ``fully_shard_context`` scope is
    active. Callers that must share one context -- for example per-chunk wrappers built by
    a single wrap call -- use it to join the caller's ambient context instead of opening a
    second one.
    """
    return _FSDP_CONTEXT.get()


def _collect_backward_order(
    module: nn.Module, registered_modules: set[nn.Module], order: IndexedOrder[nn.Module]
) -> None:
    """Collect one root's static backward prefetch order."""
    if module in registered_modules:
        order.append(module)

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
