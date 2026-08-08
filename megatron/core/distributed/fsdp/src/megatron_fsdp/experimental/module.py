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

"""Module mixin for the minimal Megatron-FSDP path."""

import enum
from collections.abc import Callable
from typing import Literal, cast
from weakref import ref

import torch
from torch import nn
from torch.distributed import DeviceMesh

from ..mixed_precision import MixedPrecisionPolicy
from .context import FsdpContext
from .parameter_group import FsdpParameterGroup, get_containing_parameter_group
from .placement import Placements


def _is_in_backward() -> bool:
    """Return whether the current thread is executing an autograd GraphTask."""
    return torch._C._current_graph_task_id() != -1


class FsdpModule:
    """Mixin attached to modules managed by the minimal FSDP path."""

    class Phase(enum.Enum):
        """Lifecycle phase of this FsdpModule."""

        RESTING = enum.auto()
        FORWARD = enum.auto()
        BACKWARD = enum.auto()

    # Name relative to the root FSDP module from named_modules().
    # Root uses "" and None means uninitialized.
    _name: str | None
    _parameter_groups: tuple[FsdpParameterGroup, ...]
    _context: FsdpContext
    _num_ready_grad_parameters: int
    _is_root: bool
    _num_trainable_parameters: int
    # Event recorded after this FsdpModule's full parameters are materialized.
    # ``None`` lets pre_forward enqueue an all-gather unless an earlier FsdpModule
    # already prefetched this module.
    _unshard_event: torch.cuda.Event | None
    # Backward-pre hook sets this to BACKWARD before activation recomputation
    # can run. Forward and backward hooks own all other transitions.
    _phase: Phase

    def __init__(
        self,
        context: FsdpContext,
        mesh: DeviceMesh,
        placements: Placements,
        mixed_precision_policy: MixedPrecisionPolicy,
        use_symm_mem: bool = False,
        grad_divisor: int = 1,
    ) -> None:
        """Initialize FSDP runtime state on an already-constructed module."""
        self._context = context
        self._is_root = False
        self._name = None
        self._unshard_event = None
        self._phase = FsdpModule.Phase.RESTING
        owned_parameters = _collect_owned_parameters(self)
        assert tuple(placements.dp_axes) == tuple(
            range(mesh.ndim)
        ), "FSDP requires dp_axes to match every mesh axis in mesh order for now."
        if grad_divisor <= 0:
            raise ValueError(f"grad_divisor must be positive, got {grad_divisor}.")
        parameter_groups = [
            FsdpParameterGroup(
                owning_module=self,
                parameters=group_parameters,
                mesh=mesh,
                placements=placements,
                mixed_precision_policy=mixed_precision_policy,
                reduce_scatter_stream=context.reduce_scatter_stream,
                use_symm_mem=use_symm_mem,
                grad_divisor=grad_divisor,
            )
            for group_parameters in _group_parameters(owned_parameters)
        ]
        self._parameter_groups = tuple(parameter_groups)
        self._num_ready_grad_parameters = 0
        self._num_trainable_parameters = sum(
            len(group.fsdp_parameters) for group in self._parameter_groups if group.requires_grad
        )
        self._register_hooks()
        context.register_module(self)

    @property
    def context(self) -> FsdpContext:
        """Return the FSDP context."""
        return self._context

    @property
    def name(self) -> str:
        """Return this FsdpModule's name."""
        name = self._name
        if name is None:
            raise RuntimeError("FSDP module name has not been initialized.")
        return name

    def is_root(self) -> bool:
        """Return whether this module is an outermost FsdpModule in its context."""
        return self._is_root

    def _register_hooks(self) -> None:
        module = cast(nn.Module, self)
        # Use PyTorch's callback module argument instead of capturing self so
        # these hooks do not retain a deleted FSDP module.
        module.register_forward_pre_hook(
            lambda hooked_module, _args: cast(FsdpModule, hooked_module).pre_forward()
        )
        module.register_forward_hook(
            lambda hooked_module, _args, _output: cast(FsdpModule, hooked_module).post_forward()
        )
        module.register_full_backward_pre_hook(
            lambda hooked_module, _grad_output: cast(FsdpModule, hooked_module).pre_backward()
        )
        if self._num_trainable_parameters == 0:
            module.register_full_backward_hook(
                lambda hooked_module, _grad_input, _grad_output: cast(
                    FsdpModule, hooked_module
                ).post_backward()
            )
            return

        # Gradient reduction for trainable parameters is parameter-completion
        # based: once every owned Parameter has accumulated its grad, this
        # FsdpModule can reduce and reshard. Module full-backward hooks can fire
        # before that when module inputs do not require grad.
        for group in self._parameter_groups:
            if not group.requires_grad:
                continue
            for fsdp_parameter in group.fsdp_parameters:
                fsdp_parameter.unsharded.register_post_accumulate_grad_hook(self._make_grad_hook())

    def _make_grad_hook(self) -> Callable[[nn.Parameter], None]:
        module_ref = ref(self)

        def grad_hook(_parameter: nn.Parameter) -> None:
            module = module_ref()
            if module is None:
                return
            module._num_ready_grad_parameters += 1
            if module._num_ready_grad_parameters == module._num_trainable_parameters:
                module.post_backward()

        return grad_hook

    def pre_forward(self) -> None:
        """Prepare full parameters for forward compute and prefetch the next FsdpModule.

        While this FsdpModule computes, we issue the next FsdpModule's all-gather
        on the comm stream, so ``AG_{i+1}`` is launched before ``F_i`` finishes.
        """
        context = self.context
        context.ensure_finalized()
        # post_forward() resets the phase after a non-recomputed forward, so a
        # FORWARD phase here means this forward-pre hook ran while the previous
        # forward was still in progress.
        assert self._phase is not FsdpModule.Phase.FORWARD
        # A reentrant checkpoint recomputes before the child module's backward-pre
        # hook can set its phase. Its forward still runs inside the active autograd
        # GraphTask, which is the signal PyTorch FSDP2 uses as well.
        is_recomputing = self._phase is FsdpModule.Phase.BACKWARD or _is_in_backward()
        if not is_recomputing:
            self._phase = FsdpModule.Phase.FORWARD
        torch.cuda.nvtx.range_push(self._nvtx_label("forward"))
        self._num_ready_grad_parameters = 0
        allgather_stream = context.allgather_stream
        current_stream = context.current_stream()

        if self.is_root():
            allgather_stream.wait_stream(current_stream)

        self._unshard_parameter_groups()
        assert self._unshard_event is not None
        # Compute waits only for this FsdpModule's all-gather (the prefetch below is
        # issued afterwards, so it is free to run concurrently with this FsdpModule).
        current_stream.wait_event(self._unshard_event)

        # Activation recomputation runs forward hooks inside backward. Do not
        # prefetch the next module in forward order: its backward may already
        # be complete, so no later backward hook would reshard it.
        if not is_recomputing:
            next_module = context.forward_order.next_item(self)
            if next_module is not None:
                next_module._unshard_parameter_groups()

    def _unshard_parameter_groups(self) -> None:
        """Unshard this FsdpModule's parameter groups on the all-gather stream.

        If ``_unshard_event`` is already set, this FsdpModule was already
        unsharded or prefetched and this method is a no-op. Otherwise, this
        method records ``_unshard_event`` after materialization so compute
        can wait without depending on later release work.
        """
        if self._unshard_event is not None:
            return

        allgather_stream = self.context.allgather_stream
        with torch.cuda.stream(allgather_stream):
            for group in self._parameter_groups:
                group.unshard_parameters()
            self._unshard_event = allgather_stream.record_event()

    def post_forward(self) -> None:
        """Return parameters to their sharded resting state after forward compute."""
        # Recomputed parameters are consumed immediately by this module's
        # backward. Keep them materialized to avoid an unnecessary all-gather;
        # post_backward() will reshard them after gradient reduction.
        is_recomputing = self._phase is FsdpModule.Phase.BACKWARD or _is_in_backward()
        if not is_recomputing:
            self._reshard_parameter_groups()
            self._phase = FsdpModule.Phase.RESTING
        torch.cuda.nvtx.range_pop()

    def _reshard_parameter_groups(self) -> None:
        """Reshard parameter groups and release unsharded storage after compute.

        This method clears ``_unshard_event`` after queuing the release, so
        future users enqueue a fresh all-gather.
        """
        for group in self._parameter_groups:
            group.reshard_parameters()

        allgather_stream = self.context.allgather_stream
        allgather_stream.wait_stream(self.context.current_stream())
        # Release on the all-gather stream where unsharded storage was allocated,
        # so no record_stream() call is required for the storage.
        with torch.cuda.stream(allgather_stream):
            for group in self._parameter_groups:
                group.release_unsharded_storage()
            self._unshard_event = None

    def pre_backward(self) -> None:
        """Prepare full parameters and prefetch the next FsdpModule in backward order."""
        self._phase = FsdpModule.Phase.BACKWARD
        torch.cuda.nvtx.range_push(self._nvtx_label("backward"))
        context = self.context
        current_stream = context.current_stream()
        if self.is_root():
            context.register_post_backward_final_callback()
            # Fork the reduce-scatter stream from the current stream once, at the
            # start of backward, so every module's post-backward reduce-scatter is
            # part of any active CUDA-graph capture. A stream only joins the
            # capture via this wait_stream edge; without it the first allocation on
            # the reduce-scatter stream falls back to a raw cudaMalloc, which is
            # illegal during capture. Later modules are covered by the post-copy
            # fork each preceding module issues before its collective.
            context.reduce_scatter_stream.wait_stream(current_stream)

        self._unshard_parameter_groups()
        assert self._unshard_event is not None
        current_stream.wait_event(self._unshard_event)

        next_module = context.backward_order.next_item(self)
        if next_module is not None:
            next_module._unshard_parameter_groups()

    def post_backward(self) -> None:
        """Reduce gradients and return parameters to their sharded resting state."""
        self._reduce_gradient_groups()
        self._reshard_parameter_groups()
        self._phase = FsdpModule.Phase.RESTING
        torch.cuda.nvtx.range_pop()

    def _reduce_gradient_groups(self) -> None:
        """Pack gradients and immediately launch their reduce-scatters."""
        context = self.context
        reduce_scatter_stream = context.reduce_scatter_stream
        current_stream = context.current_stream()

        for group in self._parameter_groups:
            if not group.requires_grad:
                continue

            with torch.cuda.stream(reduce_scatter_stream):
                partial_grad = group.allocate_partial_grad_buffer()

            current_stream.wait_stream(reduce_scatter_stream)
            group.copy_gradients_to_partial_buffer(partial_grad)

            reduce_scatter_stream.wait_stream(current_stream)
            with torch.cuda.stream(reduce_scatter_stream):
                group.reduce_partial_gradients(partial_grad, self.context.is_last_microbatch)

    @property
    def parameter_groups(self) -> tuple[FsdpParameterGroup, ...]:
        """Parameter groups owned by this FsdpModule."""
        return self._parameter_groups

    def _nvtx_label(self, phase: Literal["forward", "backward"]) -> str:
        name = self.name if self.name else "<root>"
        return f"MFSDP {name} {phase}"


def _collect_owned_parameters(root_module: nn.Module) -> dict[str, nn.Parameter]:
    parameters: dict[str, nn.Parameter] = {}

    def visit(submodule: nn.Module, submodule_fqn: str) -> None:
        direct_parameters = submodule.named_parameters(recurse=False, remove_duplicate=False)

        for local_parameter_name, parameter in direct_parameters:
            parameter_fqn = (
                f"{submodule_fqn}.{local_parameter_name}" if submodule_fqn else local_parameter_name
            )
            if get_containing_parameter_group(parameter) is not None:
                raise ValueError(
                    f"Parameter {parameter_fqn!r} is already owned by another FsdpModule."
                )
            parameters[parameter_fqn] = parameter

        for child_name, child_module in submodule.named_children():
            if isinstance(child_module, FsdpModule):
                continue
            child_fqn = f"{submodule_fqn}.{child_name}" if submodule_fqn else child_name
            visit(child_module, child_fqn)

    visit(root_module, "")
    return parameters


def _group_parameters(parameters: dict[str, nn.Parameter]) -> list[dict[str, nn.Parameter]]:
    grouped: dict[tuple[torch.dtype, bool], dict[str, nn.Parameter]] = {}
    for name, parameter in parameters.items():
        key = (parameter.dtype, parameter.requires_grad)
        grouped.setdefault(key, {})[name] = parameter
    return [grouped[key] for key in grouped]
