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

# Postpone annotations so FsdpModule can refer to itself in method signatures.
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import cast
from weakref import ref

import torch
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Shard
from torch.distributed.tensor.placement_types import Placement

from ..mixed_precision import MixedPrecisionPolicy
from .context import FsdpContext
from .countdown import Countdown
from .execution_order import ExecutionOrder
from .module_utils import get_parameter_owner, is_in_backward
from .parameter_group import (
    FsdpParameter,
    FsdpParameterGroup,
    effective_dtype,
    get_containing_parameter_group,
)
from .phase import Phase
from .placement import BlockAtomic, RowAtomic, ShardLayout
from .schedule_policy import SchedulePolicy


class FsdpModule:
    """Mixin attached to modules managed by the minimal FSDP path."""

    _parameter_groups: tuple[FsdpParameterGroup, ...]
    _context: FsdpContext
    _trainable_parameter_countdown: Countdown
    _num_trainable_parameters: int
    _schedule_policy: SchedulePolicy
    # ``phase`` is FORWARD between pre_forward() and post_forward(), BACKWARD
    # between pre_backward() and post_backward(), and RESTING otherwise. The only
    # exception is non-reentrant activation recomputation: it runs between pre_backward()
    # and post_backward(), preserving BACKWARD through its nested forward hooks.
    _phase: Phase

    def __init__(
        self,
        context: FsdpContext,
        mesh: DeviceMesh,
        model_weight_placements: tuple[Placement, ...],
        main_grad_placements: tuple[Placement, ...],
        main_weight_placements: tuple[Placement, ...],
        mixed_precision_policy: MixedPrecisionPolicy,
        grad_scale: float = 1.0,
        schedule_policy: SchedulePolicy = SchedulePolicy(),
        use_symmetric_memory: bool = False,
        shard_layout: ShardLayout = ShardLayout.PADDED,
    ) -> None:
        """Initialize FSDP runtime state on an already-constructed module.

        Args:
            shard_layout: Padded or unpadded rank segments for every parameter group.
                See ``fully_shard``.
        """
        self._context = context
        self._phase = Phase.RESTING
        self._schedule_policy = schedule_policy
        owned_parameters = _collect_owned_parameters(self)
        if grad_scale <= 0:
            raise ValueError(f"grad_scale must be positive, got {grad_scale}.")
        parameter_groups = []
        for group_parameters in _group_parameters(owned_parameters):
            first_parameter = next(iter(group_parameters.values()))
            group_dtype = effective_dtype(first_parameter)
            parameter_groups.append(
                FsdpParameterGroup(
                    owning_module=self,
                    fqn_to_parameter=group_parameters,
                    mesh=mesh,
                    model_weight_placements=_specialize_placements(
                        model_weight_placements, group_dtype
                    ),
                    main_grad_placements=_specialize_placements(main_grad_placements, group_dtype),
                    main_weight_placements=_specialize_placements(
                        main_weight_placements, group_dtype
                    ),
                    mixed_precision_policy=mixed_precision_policy,
                    grad_scale=grad_scale,
                    use_symmetric_memory=use_symmetric_memory,
                    parameter_to_owner=context.parameter_to_owner,
                    shard_layout=shard_layout,
                )
            )
        self._parameter_groups = tuple(parameter_groups)
        self._trainable_parameter_countdown = Countdown(
            schedule_policy.grad_accumulation_count
            if schedule_policy.grad_accumulation_count is not None
            else sum(1 for _ in self._trainable_fsdp_parameters())
        )
        # The state-dict safety hook is registered unconditionally. It is still
        # required to keep loading a state dict safe when ``schedule_policy.register_hooks`` is
        # False (i.e. execution hooks are disabled), so it must not be turned off
        # together with the auto-execution hooks below.
        self.register_load_state_dict_pre_hook(FsdpModule._pre_load_state_dict)
        if schedule_policy.register_hooks:
            self._register_hooks()
        context.register_module(cast(nn.Module, self))

    @property
    def context(self) -> FsdpContext:
        """Return the FSDP context."""
        return self._context

    @property
    def phase(self) -> Phase:
        """Return this module's lifecycle phase."""
        return self._phase

    @phase.setter
    def phase(self, phase: Phase) -> None:
        """Transition this module between its valid lifecycle phases."""
        allowed_transitions = {
            (Phase.RESTING, Phase.FORWARD),
            (Phase.FORWARD, Phase.RESTING),
            (Phase.RESTING, Phase.BACKWARD),
            (Phase.BACKWARD, Phase.RESTING),
        }
        if (self._phase, phase) not in allowed_transitions:
            raise RuntimeError(f"Invalid FSDP module phase transition: {self._phase} -> {phase}.")
        self._phase = phase

    @property
    def name(self) -> str:
        """Return this FsdpModule's name."""
        return self.context.module_name(cast(nn.Module, self))

    def is_root(self) -> bool:
        """Return whether this module is an outermost FsdpModule in its context."""
        return self.context.is_root(cast(nn.Module, self))

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
        self.register_post_backward_hook(FsdpModule.post_backward)

    def register_post_backward_hook(self, post_backward_hook: Callable[[FsdpModule], None]) -> None:
        """Register a post-backward hook to run after this module's backward completes.

        The hook runs when this module's backward is complete, so it can reshard
        this module's parameters and reduce their gradients. It is invoked once
        all expected gradient-accumulation callbacks have arrived, or
        via a full-backward hook when the module owns no trainable parameters.

        Args:
            post_backward_hook: Callback receiving this FSDP module after all of its
                trainable parameters have accumulated gradients.
        """
        module = cast(nn.Module, self)
        if self._trainable_parameter_countdown.initial_value == 0:

            def grad_checking_post_backward_hook(module, grad_input, _grad_output):
                if all(grad is None for grad in grad_input):
                    # With no input gradients, this hook runs before internal backward.
                    module.context._delayed_post_backward_callbacks.append(
                        (module, post_backward_hook)
                    )
                else:
                    post_backward_hook(module)

            module.register_full_backward_hook(grad_checking_post_backward_hook)
            return

        # Gradient reduction for trainable parameters is parameter-completion
        # based: once every owned Parameter has accumulated its grad, this
        # FsdpModule can reduce and reshard. Module full-backward hooks can fire
        # before that when module inputs do not require grad.
        module_ref = ref(self)

        def grad_hook(_: nn.Parameter) -> None:
            module = module_ref()
            if module is None:
                return
            if module._trainable_parameter_countdown.decrement():
                post_backward_hook(module)

        for group in self._parameter_groups:
            if not group.requires_grad:
                continue
            for fsdp_parameter in group.fsdp_parameters:
                parameter = fsdp_parameter.unsharded
                # ``skip_backward_post_hook`` is TE's delayed-wgrad contract: these
                # gradients are materialized by ``backward_dw()``, not autograd.
                if not getattr(parameter, "skip_backward_post_hook", False):
                    parameter.register_post_accumulate_grad_hook(grad_hook)
                    continue
                if len(fsdp_parameter.fqns) > 1:
                    raise ValueError(
                        "Tied parameters with delayed wgrad are not supported because "
                        "Transformer Engine does not accumulate their gradients. See "
                        "https://github.com/NVIDIA/TransformerEngine/issues/3437"
                    )
                parameter_module, _ = get_parameter_owner(module, fsdp_parameter.fqns[0])
                parameter_module.register_wgrad_accumulation_and_reduce_hooks(
                    lambda parameter=parameter: grad_hook(parameter)
                )

    def _trainable_fsdp_parameters(self) -> Iterator[FsdpParameter]:
        """Iterate this module's trainable parameters in a stable order."""
        for group in self._parameter_groups:
            if not group.requires_grad:
                continue
            yield from group.fsdp_parameters

    @staticmethod
    def _pre_load_state_dict(
        _module: nn.Module,
        _state_dict: dict[str, object],
        _prefix: str,
        local_metadata: dict[str, object],
        _strict: bool,
        _missing_keys: list[str],
        _unexpected_keys: list[str],
        _error_msgs: list[str],
    ) -> None:
        """Reject state-dict loads that replace parameters managed by FSDP."""
        # PyTorch propagates load_state_dict(assign=True) to pre-hooks through
        # this internal metadata key before replacing parameters and buffers.
        if local_metadata.get("assign_to_params_buffers", False):
            raise RuntimeError(
                "load_state_dict(assign=True) is not supported after fully_shard(). "
                "Load before fully_shard() or use an in-place load path with assign=False."
            )

    def pre_forward(self) -> None:
        """Prepare full parameters for forward compute and prefetch the next FsdpModule.

        While this FsdpModule computes, we issue the next FsdpModule's all-gather
        on the comm stream, so ``AG_{i+1}`` is launched before ``F_i`` finishes.
        """
        context = self.context
        # This is the first MFSDP hook to run, so finalize the context here once
        # before any module begins communication.
        context.ensure_finalized()
        # A reentrant checkpoint recomputes before the child module's backward-pre
        # hook runs. The active autograd GraphTask identifies that recomputation.
        is_recomputing = self.phase is Phase.BACKWARD or is_in_backward()
        if self.phase is not Phase.BACKWARD:
            self.phase = Phase.FORWARD
        # forward/backward each span multiple lifecycle methods (pre_forward ->
        # post_forward and pre_backward -> post_backward), so they keep explicit
        # push/pop instead of a single context scope.
        torch.cuda.nvtx.range_push(self._nvtx_label("forward"))

        if self.is_root():
            context.allgather_stream.wait_stream(context.current_stream())

        # TODO: Recomputation gathers MXFP8 rowwise planes on demand, because backward
        # prefetch gathers only columnwise planes. Thw two planes can be prefetched at once.
        if context.is_recording_execution_order and not is_recomputing:
            context.forward_order.append(cast(nn.Module, self))
        self.unshard(
            Phase.FORWARD, prefetch=not is_recomputing and not context.is_recording_execution_order
        )

    def unshard(self, phase: Phase | None = None, prefetch: bool = False) -> None:
        """Unshard this FsdpModule's parameter groups immediately.

        Args:
            phase: In what phase this module runs. None unshards for every phase.
            prefetch: Whether to also prefetch successors in the module order of
                ``phase``, which must then be FORWARD or BACKWARD.

        External schedulers invoking this directly (rather than through the
        automatic ``pre_forward`` hook) must first synchronize the all-gather
        stream on the root by calling
        ``context.allgather_stream.wait_stream(context.current_stream())``
        before this when ``self.is_root()``; the automatic forward path
        performs that root sync in ``pre_forward()`` immediately before this.
        """
        with self._nvtx_range("unshard"):
            compute_stream = self.context.current_stream()
            self._unshard_parameter_groups(phase)
            # Compute waits only for this FsdpModule's all-gather (the prefetch below is
            # issued afterwards, so it is free to run concurrently with this FsdpModule).
            with torch.cuda.stream(compute_stream):
                for group in self._parameter_groups:
                    group.wait_for_unshard(phase)

            if not prefetch:
                return
            context = self.context
            match phase:
                case Phase.FORWARD:
                    self._prefetch_parameter_groups(
                        context.forward_order, self._schedule_policy.forward_prefetch_size
                    )
                case Phase.BACKWARD:
                    self._prefetch_parameter_groups(
                        context.backward_order, self._schedule_policy.backward_prefetch_size
                    )
                case _:
                    raise ValueError(
                        f"Prefetching requires a FORWARD or BACKWARD phase, got {phase!r}."
                    )

    def _prefetch_parameter_groups(
        self, order: ExecutionOrder[nn.Module], prefetch_size: int | None
    ) -> None:
        """Prefetch successors from ``order`` according to this module's budget.

        Successors are unsharded for this module's phase, which is the compute they
        run next in ``order``.
        """
        module = cast(nn.Module, self)
        order.advance(module)
        next_module = order.next_item()
        if prefetch_size is None:
            if next_module is not None:
                cast(FsdpModule, next_module)._unshard_parameter_groups(self.phase)
            return

        prefetched_size = 0
        offset = 1
        while next_module is not None and prefetched_size < prefetch_size:
            fsdp_module = cast(FsdpModule, next_module)
            fsdp_module._unshard_parameter_groups(self.phase)
            prefetched_size += fsdp_module.num_parameter_elements
            offset += 1
            next_module = order.next_item(offset)

    def _unshard_parameter_groups(self, phase: Phase | None) -> None:
        """Ask each group to gather missing weights on the all-gather stream."""
        if phase not in (Phase.FORWARD, Phase.BACKWARD, None):
            raise ValueError(
                f"Unsharding requires a FORWARD, BACKWARD, or None phase, got {phase!r}."
            )
        # Recomputation that does not re-enter this module's pre_forward reads MXFP8 rowwise
        # planes in backward, so gather them together with columnwise ones. Other dtype
        # weights serve both phases already.
        if phase is Phase.BACKWARD and self._schedule_policy.unshard_all_planes_in_backward:
            phase = None
        with torch.cuda.stream(self.context.allgather_stream):
            for group in self._parameter_groups:
                group.unshard_parameters(phase)

    def post_forward(self) -> None:
        """Return parameters to their sharded resting state after forward compute."""
        is_recomputing = self.phase is Phase.BACKWARD or is_in_backward()
        if is_recomputing:
            self._release_unsharded_storage(rowwise_only=True)
        else:
            self.reshard()
        if self.phase is Phase.FORWARD:
            self.phase = Phase.RESTING
        torch.cuda.nvtx.range_pop()

    def reshard(self) -> None:
        """Reshard this FsdpModule's parameter groups."""
        with self._nvtx_range("reshard"):
            self._reshard_parameter_groups()

    def _reshard_parameter_groups(self) -> None:
        """Reshard parameter groups and release unsharded storage after compute."""
        for group in self._parameter_groups:
            group.reshard_parameters()
        self._release_unsharded_storage()

    def _release_unsharded_storage(self, *, rowwise_only: bool = False) -> None:
        """Select and release gathered storage after the current compute finishes."""
        if rowwise_only:
            groups_to_release = tuple(
                group for group in self._parameter_groups if group.is_rowwise_unsharded
            )
        else:
            groups_to_release = self._parameter_groups
        if not groups_to_release:
            return
        allgather_stream = self.context.allgather_stream
        allgather_stream.wait_stream(self.context.current_stream())
        # Release on the all-gather stream where unsharded storage was allocated,
        # so no record_stream() call is required for the storage.
        with torch.cuda.stream(allgather_stream):
            for group in groups_to_release:
                group.release_unsharded_storage(rowwise_only=rowwise_only)

    def pre_backward(self) -> None:
        """Prepare full parameters and prefetch the next FsdpModule in backward order."""
        # Shared-layer MTP (mtp_use_repeated_layer=True) reuses the same FSDP block:
        #   h1 = block(fuse(h0, shifted_embeddings_1))
        #   h2 = block(fuse(h1, shifted_embeddings_2))
        #   (loss1(h1) + loss2(h2)).backward()
        # In ordinary autograd, backward enters depth 2 then depth 1, invoking this
        # pre-hook twice on the same wrapper. Each shared parameter's post-accumulate
        # hook fires once after both uses contribute, so our post_backward() has not
        # reset BACKWARD when depth 1 enters.
        # TODO: Separate post_backward (phase/weight release) from post_accumulate_grad
        # (gradient reduction): https://github.com/NVIDIA/Megatron-LM/pull/7902
        # For invocations with gradient-requiring inputs:
        # In shared MTP, post_backward would run twice (once per depth), resetting
        # the phase after each invocation. Depth 1 could then enter from RESTING
        # and gather its weights normally, without this guard. post_accumulate_grad would
        # run once, after both depths' local gradient contributions have accumulated,
        # and perform the reduction.
        if self.phase is Phase.BACKWARD:
            return
        self.phase = Phase.BACKWARD
        torch.cuda.nvtx.range_push(self._nvtx_label("backward"))
        context = self.context
        if context.is_recording_execution_order:
            context.backward_order.append(cast(nn.Module, self))
        current_stream = context.current_stream()
        if self.is_root():
            context.register_post_backward_hook()
            # Fork the reduce-scatter stream from the current stream once, at the
            # start of backward, so every module's post-backward reduce-scatter is
            # part of any active CUDA-graph capture. A stream only joins the
            # capture via this wait_stream edge; without it the first allocation on
            # the reduce-scatter stream falls back to a raw cudaMalloc, which is
            # illegal during capture. Later modules are covered by the post-copy
            # fork each preceding module issues before its collective.
            context.reduce_scatter_stream.wait_stream(current_stream)

        self.unshard(Phase.BACKWARD, prefetch=not context.is_recording_execution_order)

    def post_backward(self) -> None:
        """Reduce gradients and return parameters to their sharded resting state."""
        self.reshard()
        self.context.validate_grad_sync()
        self._reduce_gradient_groups()
        self.phase = Phase.RESTING
        torch.cuda.nvtx.range_pop()

    def _reduce_gradient_groups(self) -> None:
        """Pack gradients and immediately launch their reduce-scatters."""
        with self._nvtx_range("reduce_gradients"):
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
                    group.reduce_partial_gradients(
                        partial_grad, is_last_microbatch=self.context.is_last_microbatch
                    )

    @property
    def parameter_groups(self) -> tuple[FsdpParameterGroup, ...]:
        """Parameter groups owned by this FsdpModule."""
        return self._parameter_groups

    @property
    def num_parameter_elements(self) -> int:
        """Return the number of unsharded parameter elements owned by this module."""
        return sum(
            parameter.unsharded.numel()
            for group in self._parameter_groups
            for parameter in group.fsdp_parameters
        )

    def _nvtx_label(self, operation: str) -> str:
        module_name = self.name or "<root>"
        return f"MFSDP {module_name} {operation}"

    @contextmanager
    def _nvtx_range(self, operation: str) -> Iterator[None]:
        """Scope an nvtx range to this context so an early return still pops it."""
        torch.cuda.nvtx.range_push(self._nvtx_label(operation))
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()


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
        key = (effective_dtype(parameter), parameter.requires_grad)
        grouped.setdefault(key, {})[name] = parameter
    return [grouped[key] for key in grouped]


def _specialize_placements(
    placements: tuple[Placement, ...], group_dtype: torch.dtype
) -> tuple[Placement, ...]:
    """Specialize public placements for one homogeneous parameter group.

    Map Torch's user-facing ``Shard(0)`` to ``BlockAtomic`` for MXFP8 groups
    and ``RowAtomic`` for ordinary floating-point groups.
    """
    if group_dtype not in (torch.uint8, torch.float32, torch.bfloat16, torch.float16):
        raise NotImplementedError(f"Unsupported group dtype: {group_dtype}.")
    for placement in placements:
        if type(placement) is Shard and placement.dim != 0:
            raise NotImplementedError(
                "MFSDP currently supports only dim-0 Shard placements, " f"got {placement!r}."
            )
    placement_type = BlockAtomic(32) if group_dtype == torch.uint8 else RowAtomic()
    return tuple(
        placement_type if type(placement) is Shard else placement for placement in placements
    )
