# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tensor-only execution boundaries for modules sharing differentiable state.

A region implements ``forward(hidden, state) -> (hidden, state)``. State is a
fresh mapping for each invocation. Only declared fields cross the region; model
code owns their meaning, while eager, checkpoint and graph execution use the
same tensor signature. The module never retains a forward's state mapping.
"""

import gc
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Mapping

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
        self, hidden: Tensor, state: Mapping[str, Tensor | None], *, recompute: bool = False
    ) -> tuple[Tensor, dict[str, Tensor | None]]:
        """Run eagerly or checkpoint explicit state, including parameter-only gradients."""
        args = (hidden, *self.inputs.pack(state))
        if recompute and torch.is_grad_enabled():
            if hidden.is_cuda and torch.cuda.is_current_stream_capturing():
                raise ValueError("Stateful full recomputation cannot be nested in CUDA capture")
            result = checkpoint(self, *args, use_reentrant=False, context_fn=_checkpoint_contexts)
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


class _StatefulGraphModule(StatefulModule):
    """Keep TE's reused gradient slots compatible across different state schemas.

    TE versions that key gradient buffers by differentiable outputs alone still
    cache a slot for every output, including nondifferentiable tensors. Put all
    active outputs first and pad every callable to the same length with None.
    Equal TE keys then imply identical gradient slot positions and counts.
    Packing only rearranges tensor references; padding allocates no tensors.
    """

    def __init__(self, region, output_count):
        super().__init__(region.module, region.inputs.fields, region.outputs.fields)
        self._output_count = output_count
        self._active_outputs = None
        self._output_order = None

    def forward(self, hidden, *values):
        outputs = super().forward(hidden, *values)
        active = tuple(t.requires_grad for t in outputs)
        if self._active_outputs is None:
            self._active_outputs = active
            self._output_order = tuple(i for i, grad in enumerate(active) if grad) + tuple(
                i for i, grad in enumerate(active) if not grad
            )
        elif active != self._active_outputs:
            raise ValueError('Stateful graph output gradient activity changed during capture')
        return tuple(outputs[i] for i in self._output_order) + (None,) * (
            self._output_count - len(outputs)
        )

    def restore_outputs(self, outputs):
        """Restore model order and gradient eligibility after TE's autograd wrapper."""
        restored = [None] * len(self._output_order)
        for position, original in enumerate(self._output_order):
            tensor = outputs[position]
            # A custom autograd Function makes all floating outputs require
            # gradients unless marked otherwise, including detached metadata.
            restored[original] = tensor if self._active_outputs[original] else tensor.detach()
        return tuple(restored)


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
        return tensor.view_as(tensor)

    @staticmethod
    def backward(ctx, gradient):
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

    Independent captures have separate pools; ``capture_many`` shares a pool
    according to the pipeline schedule. Slots retain no caller state mapping.
    Graphs share model parameters. Input shapes, dtype, strides and gradient
    activity match capture; all differentiable outputs must be consumed. State
    fields whose backward activity can change require distinct capture profiles.
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
        _defer_capture: bool = False,
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
        self._backend = backend
        self.debug_checks = debug_checks
        samples = (hidden, *region.inputs.pack(state))
        self._signature = tuple(self._tensor_signature(t) for t in samples)
        self._busy = [False] * slots
        self._generation = [0] * slots
        self._fused_grad_params = []
        self._events = [torch.cuda.Event() for _ in range(slots)]
        self._recorded = [False] * slots
        self._callables = []
        self._wrappers = []
        self._output_wrapper = None
        self._manual_forward_hooks = []
        if make_forward_pre_hook is not None:
            self.setup_manual_hooks(make_forward_pre_hook)
            # Lazy capture may follow an asynchronous parameter gather. Drain it
            # before capture suppresses the ordinary DDP parameter-owner hooks.
            for hook, module in self._manual_forward_hooks:
                hook(module)
        if _defer_capture:
            return
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

    @classmethod
    def capture_many(
        cls,
        profiles,
        *,
        order,
        num_layers_per_chunk,
        capture_kwargs=None,
        preserve_gradients=True,
        clone_inputs=True,
    ):
        """Capture model-owned regions together in the pipeline execution order.

        Each profile is ``(region, hidden, state, make_forward_pre_hook)``. Unlike
        independent captures, TE can reuse backward workspaces across layers and
        static inputs across microbatches whose backward has completed. Capture
        must run between training steps, with no outstanding region forwards.
        Set ``clone_inputs=False`` only for detached, exclusively owned warmup
        samples: capture takes ownership of those buffers instead of copying them.
        """
        if (
            not profiles
            or not num_layers_per_chunk
            or any(count < 0 for count in num_layers_per_chunk)
            or sum(num_layers_per_chunk) != len(profiles)
        ):
            raise ValueError("Stateful capture profiles must match the layers in each chunk")
        chunks = len(num_layers_per_chunk)
        microbatches = len(order) // (2 * chunks)
        if (
            microbatches < 1
            or len(order) != 2 * chunks * microbatches
            or any(
                order.count(chunk) != microbatches or order.count(-chunk) != microbatches
                for chunk in range(1, chunks + 1)
            )
        ):
            raise ValueError("Stateful capture requires a balanced forward/backward schedule")
        owners = [
            cls(
                region,
                hidden,
                state,
                slots=microbatches,
                backend="transformer_engine",
                make_forward_pre_hook=hook,
                _defer_capture=True,
            )
            for region, hidden, state, hook in profiles
        ]
        output_count = max(1 + len(region.outputs.packed_fields) for region, _, _, _ in profiles)
        wrappers = tuple(_StatefulGraphModule(region, output_count) for region, _, _, _ in profiles)
        for wrapper, (region, _, _, _) in zip(wrappers, profiles):
            wrapper.training = region.training
        samples = tuple(
            (hidden, *region.inputs.pack(state)) for region, hidden, state, _ in profiles
        )
        sample_args = cls._scheduled_sample_args(
            samples, order, num_layers_per_chunk, clone_inputs=clone_inputs
        )
        offsets = [0]
        for count in num_layers_per_chunk:
            offsets.append(offsets[-1] + count)

        from transformer_engine.pytorch import graph as te_graph
        from transformer_engine.pytorch import make_graphed_callables

        # A single guard deduplicates shared parameters/buffers. CLI capture can
        # discard completed-step gradients instead of cloning every main_grad.
        guard = object.__new__(cls)
        guard.region = nn.ModuleList(wrappers)
        with guard._capture_grad_state(te_graph, preserve_gradients=preserve_gradients) as fused:
            captured = make_graphed_callables(
                wrappers,
                tuple(sample_args),
                allow_unused_input=True,
                _order=order,
                _num_layers_per_chunk=num_layers_per_chunk,
                _reuse_graph_input_output_buffers=True,
                **(capture_kwargs or {}),
            )
        fused_ids = {id(p) for p in fused}
        for chunk, count in enumerate(num_layers_per_chunk):
            for local_layer in range(count):
                layer = offsets[chunk] + local_layer
                owner = owners[layer]
                owner._wrappers.append(wrappers[layer])
                owner._output_wrapper = wrappers[layer]
                owner._callables = [
                    captured[offsets[chunk] * microbatches + mb * count + local_layer]
                    for mb in range(microbatches)
                ]
                parameters = tuple(p for p in owner.region.parameters() if id(p) in fused_ids)
                owner._fused_grad_params = [parameters] * microbatches
                owner._module_profile = owner._profile_module()
        return owners

    @staticmethod
    def _scheduled_sample_args(samples, order, num_layers_per_chunk, *, clone_inputs=True):
        """Allocate static inputs only for the peak live forwards of each signature."""
        chunks = len(num_layers_per_chunk)
        microbatches = len(order) // (2 * chunks)
        result = [None] * (len(samples) * microbatches)
        offsets = [0]
        for count in num_layers_per_chunk:
            offsets.append(offsets[-1] + count)
        signatures = [tuple(StatefulGraphs._tensor_signature(t) for t in args) for args in samples]
        available, pending = defaultdict(deque), [deque() for _ in range(chunks)]
        forward_index = [0] * chunks
        donated = set()
        for operation in order:
            chunk = abs(operation) - 1
            count = num_layers_per_chunk[chunk]
            if operation > 0:
                for layer in range(offsets[chunk], offsets[chunk + 1]):
                    key = signatures[layer]
                    if available[key]:
                        args = available[key].popleft()
                    elif not clone_inputs and layer not in donated:
                        if any(not t.is_leaf for t in samples[layer]):
                            raise ValueError('Donated graph samples must be detached leaves')
                        args = samples[layer]
                        donated.add(layer)
                    else:
                        args = tuple(
                            t.detach().clone().requires_grad_(t.requires_grad)
                            for t in samples[layer]
                        )
                    index = offsets[chunk] * microbatches + forward_index[chunk] * count
                    result[index + layer - offsets[chunk]] = args
                    pending[chunk].append((key, args))
                forward_index[chunk] += 1
            else:
                if len(pending[chunk]) < count:
                    raise ValueError("Stateful capture backward precedes its forward")
                for _ in range(count):
                    key, args = pending[chunk].popleft()
                    available[key].append(args)
        return result

    @contextmanager
    def _capture_grad_state(self, te_graph, *, preserve_gradients=True):
        """Keep warmup/capture out of DDP reduction and preserve accumulated gradients."""
        from megatron.core.transformer import cuda_graphs

        missing = object()
        original_calls = {
            type(module): type(module).__dict__.get("__call__", missing)
            for module in self.region.modules()
            if isinstance(module, StatefulModule)
        }
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
        backups = [
            main.clone() if main is not None and preserve_gradients else None
            for _, _, main, _ in saved
        ]
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
                        if preserve_gradients:
                            main.copy_(backup)
                        else:
                            main.zero_()
                    p.grad = grad
                    if flag is missing:
                        if hasattr(p, "grad_added_to_main_grad"):
                            delattr(p, "grad_added_to_main_grad")
                    else:
                        p.grad_added_to_main_grad = flag
            finally:
                if te_graph is not None:
                    for module_cls, original_call in original_calls.items():
                        if original_call is missing:
                            if "__call__" in module_cls.__dict__:
                                delattr(module_cls, "__call__")
                        else:
                            module_cls.__call__ = original_call
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
        for captured in self._callables:
            reset = getattr(captured, "reset", None)
            if reset is not None:
                reset()
        for wrapper in self._wrappers:
            wrapper.forward = type(wrapper).forward.__get__(wrapper, type(wrapper))
        self._callables.clear()
        self._wrappers.clear()
        self._output_wrapper = None
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
        self,
        hidden: Tensor,
        state: Mapping[str, Tensor | None],
        *,
        slot: int = 0,
        is_first_microbatch: bool = True,
    ) -> tuple[Tensor, dict[str, Tensor | None]]:
        """Replay a slot, refreshing TE's weight cache on the step's first microbatch."""
        if not 0 <= slot < len(self._callables) or self._busy[slot]:
            raise RuntimeError("Graph slot is invalid or still belongs to an outstanding backward")
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
        if self._backend == "transformer_engine":
            outputs = self._callables[slot](*args, is_first_microbatch=is_first_microbatch)
        else:
            outputs = self._callables[slot](*args)
        if self._output_wrapper is not None:
            outputs = self._output_wrapper.restore_outputs(outputs)
        if torch.is_grad_enabled() and any(t.requires_grad for t in outputs):
            self._busy[slot] = True
            outputs = _GraphOutputs.apply(self, slot, *outputs)
        else:
            # An eval/frozen invocation has no backward to hold the slot. Give
            # callers (including asynchronous P2P) independent output storage.
            outputs = tuple(output.clone() for output in outputs)
            self._release(slot)
        return self.region.restore_result(outputs, state)
