# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Checkpoint strategies for child modules and bound methods."""

from __future__ import annotations

from typing import Any, Iterable, Optional

import torch

from megatron.core.activation_policy.recompute import checkpointing as _checkpointing
from megatron.core.activation_policy.recompute.checkpointing import precision_aware_checkpoint
from megatron.core.activation_policy.runtime import (
    Event,
    activation_point_owner,
    installed_strategy,
    update_strategy,
)


class Strategy:
    """Base type for strategies declared on a module during construction."""

    __slots__ = ("region",)

    def __init__(self, region: Any):
        self.region = region

    def observes(self) -> Iterable[Any]:
        """Additional regions where this strategy schedules boundary callbacks."""
        return ()

    def run(self, owner, defer, forward, args, kwargs):
        """Run the region. Defer a callback with ``defer(event, self, callback)``.

        For module calls, enter callbacks receive (args, kwargs) and exit callbacks receive
        the returned value. Executors that fire a boundary directly supply its value.
        ``forward`` includes the module hooks on both the initial call and replay.
        """
        raise NotImplementedError

    def _update(self, **changes):
        """Apply a setup-time edit and refresh observed points if already installed."""
        owner = activation_point_owner(self.region)
        if owner is not None and installed_strategy(owner, self.region) is self:
            update_strategy(owner, self.region, **changes)
        else:
            for field, value in changes.items():
                setattr(self, field, value)


class CheckpointWithoutOutputStrategy(Strategy):
    """Discard the region's output in forward and rebuild it in backward.

    Implemented with ``CheckpointWithoutOutput``.

    Args:
        region: Child module or bound method whose call is replayed in backward.
        release_event: Child module or bound method where the region's output storage is
            released. By default, gradients at this boundary trigger replay.
        release_on_exit: Release at its output by default, or at its inputs when False.
        recompute_on: Optional separate boundary whose gradients trigger replay. Both events
            must occur after the checkpointed call. Every backward path that uses a discarded
            output must reach a trigger before using it.
        retain_input_tensors: Keep outputs that share storage with an input when releasing.
        training_only: Only recompute while the owner is in training mode.
    """

    __slots__ = ("release_on", "retain_input_tensors", "training_only", "recompute_on")

    def __init__(
        self,
        region: Any,
        *,
        release_event: Any,
        release_on_exit: bool = True,
        recompute_on: Optional[Event] = None,
        retain_input_tensors: bool = True,
        training_only: bool = False,
    ):
        super().__init__(region)
        self.release_on = (
            Event.exit(release_event) if release_on_exit else Event.enter(release_event)
        )
        self.recompute_on = recompute_on
        self.retain_input_tensors = retain_input_tensors
        self.training_only = training_only

    def release_on_enter(self, target) -> None:
        """Move release to the inputs of target during model setup."""
        self._set_release(Event.enter(target))

    def release_on_exit(self, target) -> None:
        """Move release to the outputs of target during model setup."""
        self._set_release(Event.exit(target))

    def _set_release(self, event: Event) -> None:
        if event == self.release_on:
            return
        self._update(release_on=event)

    def observes(self) -> Iterable[Any]:
        if self.recompute_on is None:
            return (self.release_on.region,)
        return (self.release_on.region, self.recompute_on.region)

    def run(self, owner, defer, forward, args, kwargs):
        if not torch.is_grad_enabled() or (self.training_only and not owner.training):
            return forward(*args, **kwargs)
        ckpt = _checkpointing.CheckpointWithoutOutput(
            retain_input_tensors=self.retain_input_tensors
        )
        tensors, flat_forward = _checkpoint_call(forward, args, kwargs)
        out = ckpt.checkpoint(flat_forward, *tensors)
        if ckpt.outputs is None:
            return out  # Warmup or a call without differentiable inputs.
        if self.recompute_on is None or self.recompute_on == self.release_on:

            def release_and_hook(value):
                for tensor in _grad_tensors(value):
                    ckpt.discard_output_and_register_recompute(tensor)

            defer(self.release_on, self, release_and_hook)
        else:

            def install_hooks(value):
                tensors = _grad_tensors(value)
                if not tensors:
                    raise ValueError("recompute_on requires a differentiable boundary tensor")
                for tensor in tensors:
                    ckpt.register_recompute_hook(tensor)

            defer(self.recompute_on, self, install_hooks)
            defer(self.release_on, self, lambda _: ckpt.release_outputs())
        return out


def _grad_tensors(value):
    """Unique differentiable tensor leaves at an observed boundary."""
    from torch.utils._pytree import tree_leaves

    return list(
        dict.fromkeys(
            t for t in tree_leaves(value) if isinstance(t, torch.Tensor) and t.requires_grad
        )
    )


def _checkpoint_call(forward, args, kwargs):
    """Pass tensor inputs through autograd and reconstruct the original call on replay."""
    from torch.utils._pytree import tree_flatten, tree_unflatten

    values, spec = tree_flatten((args, kwargs))
    positions = [i for i, value in enumerate(values) if isinstance(value, torch.Tensor)]
    tensors = [values[i] for i in positions]
    for i in positions:
        values[i] = None

    def flat_forward(*flat):
        rebuilt = values.copy()
        for i, tensor in zip(positions, flat):
            rebuilt[i] = tensor
        args, kwargs = tree_unflatten(rebuilt, spec)
        return forward(*args, **kwargs)

    return tensors, flat_forward


class CheckpointStrategy(Strategy):
    """Keep the region's output and replay its computation in backward."""

    def run(self, owner, defer, forward, args, kwargs):
        tensors, flat_forward = _checkpoint_call(forward, args, kwargs)
        return precision_aware_checkpoint(
            flat_forward, owner.config, owner.pg_collection.tp, *tensors
        )
