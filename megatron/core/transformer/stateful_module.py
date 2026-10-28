# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tensor-only execution boundaries for modules sharing differentiable state.

A region implements ``forward(hidden, state) -> (hidden, state)``. State is a
fresh mapping for each invocation. Only declared fields cross the region; model
code owns their meaning, while eager, checkpoint and graph execution use the
same tensor signature. The module never retains a forward's state mapping.
"""

import gc
from contextlib import contextmanager
from functools import partial
from typing import Callable, Mapping

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from megatron.core.tensor_observation import suspend_tensor_observations
from megatron.core.tensor_parallel import random as rng
from megatron.core.transformer.state_boundary import TensorField, TensorSchema


class StatefulModule(nn.Module):
    """Expose a stateful region as a tuple of ordinary autograd tensors.

    Schemas describe one shape/layout profile. A caller can construct another
    adapter over the same module for a different profile without copying weights.
    Input and output fields can differ, including relayed inputs and absent slots.
    Names shared between independent components must have an explicit owner.
    """

    def __init__(
        self,
        module: nn.Module,
        input_fields: tuple[TensorField, ...] = (),
        output_fields: tuple[TensorField, ...] = (),
    ) -> None:
        super().__init__()
        self.module = module
        self.training = module.training
        self.inputs, self.outputs = TensorSchema(input_fields), TensorSchema(output_fields)

    def forward(self, hidden: Tensor, *values: Tensor) -> tuple[Tensor, ...]:
        """Run from explicit tensors, rebuilding state on checkpoint replay/capture."""
        state = self.inputs.unpack(values)
        output, state = self.module(hidden, state)
        if not isinstance(output, Tensor):
            raise TypeError("Stateful regions must return one hidden tensor and a state mapping")
        return (output, *self.outputs.pack(state))

    def run(
        self,
        hidden: Tensor,
        state: Mapping[str, Tensor | None],
        *,
        recompute: bool = False,
        checkpoint_fn: Callable | None = None,
    ) -> tuple[Tensor, dict[str, Tensor | None]]:
        """Run eagerly or checkpoint explicit state, including parameter-only gradients.

        ``checkpoint_fn(region, *tensors)`` replaces the default non-reentrant
        checkpoint, e.g. with an FP8-aware ``te_checkpoint`` for quantized regions.
        """
        args = (hidden, *self.inputs.pack(state))
        if recompute and torch.is_grad_enabled():
            if hidden.is_cuda and torch.cuda.is_current_stream_capturing():
                raise ValueError("Stateful full recomputation cannot be nested in CUDA capture")
            if checkpoint_fn is None:
                if _fp8_autocast_enabled():
                    # Replay would run outside the forward's FP8 recipe state.
                    raise ValueError("FP8 recomputation requires an FP8-aware checkpoint_fn")
                checkpoint_fn = partial(
                    checkpoint, use_reentrant=False, context_fn=_checkpoint_contexts
                )
            result = checkpoint_fn(self, *args)
        else:
            result = self(*args)
        return self.restore_result(result, state)

    def restore_result(self, result, state):
        """Publish live outputs and retire consumed inputs, preserving unrelated components."""
        updated = dict(state)
        for field in self.inputs.fields:
            updated.pop(field.key, None)
        updated.update(self.outputs.unpack(result[1:]))
        return result[0], updated


def _fp8_autocast_enabled() -> bool:
    try:
        from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
    except ImportError:
        return False
    return FP8GlobalStateManager.is_fp8_enabled()


def _checkpoint_contexts():
    """Preserve Megatron's model-parallel RNG in addition to PyTorch's default RNG."""
    saved = {}

    def clone_states(states):
        # Graph-safe trackers hold mutable Generators, not immutable byte snapshots.
        return {
            name: value.clone_state() if isinstance(value, torch.Generator) else value.clone()
            for name, value in states.items()
        }

    @contextmanager
    def forward():
        saved["tracker"] = clone_states(rng.get_cuda_rng_tracker().get_states())
        previous = rng.is_checkpointing()
        rng._set_checkpointing()
        try:
            yield
        finally:
            if not previous:
                rng._unset_checkpointing()

    @contextmanager
    def replay():
        tracker = rng.get_cuda_rng_tracker()
        previous_states, previous = tracker.get_states(), rng.is_checkpointing()
        tracker.set_states(clone_states(saved["tracker"]))
        rng._set_checkpointing()
        try:
            with suspend_tensor_observations():
                yield
        finally:
            tracker.set_states(previous_states)
            if not previous:
                rng._unset_checkpointing()

    return forward(), replay()


class _GraphInputGrad(torch.autograd.Function):
    """Give shared producers gradients independent of a graph's reusable buffers."""

    @staticmethod
    def forward(ctx, tensor):
        """Alias the input so its gradient passes through this node."""
        return tensor.view_as(tensor)

    @staticmethod
    def backward(ctx, gradient):
        """Copy the gradient out of the graph's reusable static buffer."""
        return gradient.clone()


class _GraphOutputs(torch.autograd.Function):
    """Protect captured buffers and reject unsupported dynamic gradient activity."""

    @staticmethod
    def forward(ctx, owner, slot, *outputs):
        """Track a graph slot's outputs without materializing absent gradients."""
        ctx.owner, ctx.slot = owner, slot
        ctx.generation = owner._generation[slot]
        ctx.active = tuple(t.requires_grad for t in outputs)
        ctx.set_materialize_grads(False)
        result = tuple(t.view_as(t) for t in outputs)
        ctx.mark_non_differentiable(*(t for t, active in zip(result, ctx.active) if not active))
        return result

    @staticmethod
    def backward(ctx, *grads):
        """Validate output gradients and release the slot after captured backward."""
        if ctx.owner is not None and ctx.generation != ctx.owner._generation[ctx.slot]:
            raise RuntimeError("This graph invocation was aborted; discard its outputs")
        if ctx.owner is None:
            raise RuntimeError("Stateful graph outputs support one backward per invocation")
        if any(active and grad is None for active, grad in zip(ctx.active, grads)):
            raise RuntimeError(
                "Every differentiable captured output must participate in backward; "
                "use a separate output schema/capture profile for an unused state branch"
            )
        # Releasing before the nested captured backward runs would let a new
        # microbatch overwrite its buffers. Retire the slot at engine completion.
        owner, slot = ctx.owner, ctx.slot
        for parameter in owner._fused_grad_params[slot]:
            parameter.grad_added_to_main_grad = True
        ctx.owner = None
        torch.autograd.Variable._execution_engine.queue_callback(lambda: owner._release(slot))
        return (None, None, *grads)


class StatefulGraphs:
    """Capture independent graph slots for outstanding pipeline microbatches.

    Each slot has its own graph memory pool and retains no caller state mapping.
    Graphs share model parameters. Input shapes, dtype, strides, gradient activity
    and training mode match capture; all differentiable outputs must be consumed.
    State fields whose backward activity can change require distinct capture profiles.
    """

    def __init__(
        self,
        region: StatefulModule,
        hidden: Tensor,
        state: Mapping[str, Tensor | None],
        *,
        slots: int = 1,
        backend: str = "torch",
        num_warmup_iters: int = 3,
        debug_checks: bool = False,
        capture_kwargs: Mapping | None = None,
        make_forward_pre_hook=None,
    ) -> None:
        if not hidden.is_cuda or slots < 1:
            raise ValueError("Stateful CUDA graphs require a CUDA sample and positive slot count")
        te_graph = None
        if backend == "torch":
            capture = torch.cuda.make_graphed_callables
        elif backend == "transformer_engine":
            from transformer_engine.pytorch import graph as te_graph
            from transformer_engine.pytorch import make_graphed_callables
            from transformer_engine.pytorch.distributed import (
                get_all_rng_states,
                graph_safe_rng_available,
            )

            # TE changes its global capture flag and module __call__ before
            # saving RNG state. Reject legacy byte states before those changes.
            if graph_safe_rng_available() and any(
                not isinstance(value, torch.Generator) for value in get_all_rng_states().values()
            ):
                raise ValueError(
                    "Transformer Engine CUDA graphs require graph-safe RNG states; "
                    "initialize and seed the TE RNG tracker during model setup"
                )
            capture = make_graphed_callables
        else:
            raise ValueError(f"Unsupported stateful graph backend: {backend}")
        self.region, self.device = region, hidden.device
        self.debug_checks = debug_checks
        # The backends' own eval fallback only sees the unregistered slot wrappers,
        # so every replay compares the live module tree with the captured mode.
        self._modules = tuple(region.module.modules())
        self._modes = tuple(module.training for module in self._modules)
        samples = (hidden, *region.inputs.pack(state))
        self._signature = tuple(self._tensor_signature(t) for t in samples)
        self._busy = [False] * slots
        self._generation = [0] * slots
        self._fused_grad_params = []
        self._events = [torch.cuda.Event() for _ in range(slots)]
        self._recorded = [False] * slots
        self._callables = []
        self._wrappers = []
        self._manual_forward_hooks = []
        if make_forward_pre_hook is not None:
            self.setup_manual_hooks(make_forward_pre_hook)
            # Lazy capture may follow an asynchronous parameter gather. Drain it
            # before capture suppresses the ordinary DDP parameter-owner hooks.
            for hook, module in self._manual_forward_hooks:
                hook(module)
        for _ in range(slots):
            wrapper = StatefulModule(region.module, region.inputs.fields, region.outputs.fields)
            wrapper.training = region.training
            self._wrappers.append(wrapper)
            args = tuple(t.detach().clone().requires_grad_(t.requires_grad) for t in samples)
            with self._capture_grad_state(te_graph) as fused:
                self._callables.append(
                    capture(
                        wrapper,
                        args,
                        num_warmup_iters=num_warmup_iters,
                        allow_unused_input=True,
                        **(capture_kwargs or {}),
                    )
                )
            self._fused_grad_params.append(tuple(fused))
        self._module_profile = self._profile_module()

    @contextmanager
    def _capture_grad_state(self, te_graph):
        """Keep warmup/capture out of DDP reduction and preserve accumulated gradients."""
        from megatron.core.transformer import cuda_graphs

        missing = object()
        original_call = StatefulModule.__dict__.get("__call__", missing)
        te_was_capturing = te_graph is not None and te_graph.is_graph_capturing()
        saved = [
            (
                p,
                p.grad,
                getattr(p, "main_grad", None),
                getattr(p, "grad_added_to_main_grad", missing),
            )
            for p in self.region.parameters()
        ]
        backups = [None if main is None else main.clone() for _, _, main, _ in saved]
        # Router statistics and other registered state must count real invocations,
        # not the extra forward/backward calls used to build each graph slot.
        buffers = [(buffer, buffer.clone()) for buffer in self.region.buffers()]
        was_capturing = cuda_graphs.is_graph_capturing()
        # DDP retains AccumulateGrad nodes created before the capture stream.
        # Match MCore's full-graph setup, but restore this process-global switch.
        get_stream_override = getattr(torch._C, "_override_stale_capture_stream", None)
        previous_override = None if get_stream_override is None else get_stream_override()
        if previous_override is not None:
            torch.autograd.graph.set_override_stale_capture_stream(True)
        cuda_graphs._set_capture_start()
        fused = []
        for p, _, _, flag in saved:
            p.grad = None
            if flag is not missing:
                p.grad_added_to_main_grad = False
        try:
            yield fused
            fused.extend(p for p, _, _, _ in saved if getattr(p, "grad_added_to_main_grad", False))
        finally:
            try:
                with torch.no_grad():
                    for buffer, value in buffers:
                        buffer.copy_(value)
                for (p, grad, main, flag), backup in zip(saved, backups):
                    if main is not None:
                        main.copy_(backup)
                    p.grad = grad
                    if flag is missing:
                        if hasattr(p, "grad_added_to_main_grad"):
                            delattr(p, "grad_added_to_main_grad")
                    else:
                        p.grad_added_to_main_grad = flag
            finally:
                if te_graph is not None:
                    if original_call is missing:
                        if "__call__" in StatefulModule.__dict__:
                            delattr(StatefulModule, "__call__")
                    else:
                        StatefulModule.__call__ = original_call
                    if not te_was_capturing:
                        te_graph.set_capture_end()
                if previous_override is not None:
                    torch.autograd.graph.set_override_stale_capture_stream(previous_override)
                if not was_capturing:
                    cuda_graphs._set_capture_end()

    def validate_module(self) -> None:
        """Check stable parameter storage and training mode outside the replay hot path."""
        if self._profile_module() != self._module_profile:
            raise ValueError("Stateful graph module storage or training mode changed after capture")

    def abort(self) -> None:
        """Retire a discarded iteration after its communication has been drained.

        Old outputs must be discarded; backward through them is rejected. Call
        only once the caller has stopped scheduling work for this graph set.
        """
        torch.cuda.synchronize(self.device)
        for slot in range(len(self._busy)):
            self._generation[slot] += 1
            self._busy[slot] = False
            self._recorded[slot] = False

    @staticmethod
    def _tensor_signature(tensor):
        return (
            tuple(tensor.shape),
            tensor.dtype,
            tensor.device,
            # Singleton dimensions do not advance the storage offset.
            tuple(
                stride if size != 1 else 0 for size, stride in zip(tensor.shape, tensor.stride())
            ),
            tensor.requires_grad,
        )

    def _profile_module(self):
        tensors = (*self.region.module.parameters(), *self.region.module.buffers())
        return (
            tuple((id(t), t.data_ptr(), self._tensor_signature(t)) for t in tensors),
            tuple((id(m), m.training) for m in self.region.module.modules()),
        )

    def _release(self, slot):
        self._events[slot].record(torch.cuda.current_stream(self.device))
        self._recorded[slot], self._busy[slot] = True, False

    def close(self) -> None:
        """Release captures before destroying their process groups.

        Complete backward and discard its output tensors first. Graphed module
        forward closures can form cycles; breaking them lets NCCL release the
        communicators referenced by captured collectives before process-group
        shutdown waits for those graph references.
        """
        if any(self._busy):
            raise RuntimeError("Cannot close stateful graphs with outstanding backward")
        torch.cuda.synchronize(self.device)
        for wrapper in self._wrappers:
            wrapper.forward = StatefulModule.forward.__get__(wrapper, StatefulModule)
        self._callables.clear()
        self._wrappers.clear()
        self._manual_forward_hooks.clear()
        gc.collect()

    def setup_manual_hooks(self, make_hook_func):
        """Run parameter-gather pre-hooks for the captured parameter owners."""
        self._manual_forward_hooks = [
            (make_hook_func(), module)
            for module in self.region.module.modules()
            if next(module.parameters(recurse=False), None) is not None
        ]

    def run(
        self, hidden: Tensor, state: Mapping[str, Tensor | None], *, slot: int = 0
    ) -> tuple[Tensor, dict[str, Tensor | None]]:
        """Replay one reserved slot and publish its side outputs into a fresh mapping."""
        if not 0 <= slot < len(self._callables) or self._busy[slot]:
            raise RuntimeError("Graph slot is invalid or still belongs to an outstanding backward")
        if tuple(module.training for module in self._modules) != self._modes:
            raise ValueError("Stateful graph training mode differs from capture")
        if self.debug_checks:
            self.validate_module()
        args = (hidden, *self.region.inputs.pack(state))
        if tuple(self._tensor_signature(t) for t in args) != self._signature:
            raise ValueError("Stateful graph input profile differs from capture")
        if self._recorded[slot]:
            torch.cuda.current_stream(self.device).wait_event(self._events[slot])
        for hook, module in self._manual_forward_hooks:
            hook(module)
        args = (args[0], *(_GraphInputGrad.apply(t) if t.requires_grad else t for t in args[1:]))
        outputs = self._callables[slot](*args)
        if torch.is_grad_enabled() and any(t.requires_grad for t in outputs):
            self._busy[slot] = True
            outputs = _GraphOutputs.apply(self, slot, *outputs)
        else:
            # An eval/frozen invocation has no backward to hold the slot. Give
            # callers (including asynchronous P2P) independent output storage.
            outputs = tuple(output.clone() for output in outputs)
            self._release(slot)
        return self.region.restore_result(outputs, state)
