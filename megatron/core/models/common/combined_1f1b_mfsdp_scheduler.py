# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from typing import Callable

from torch import nn

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule


def _make_unshard_forward_hook(owner: FsdpModule):
    """Unshard the owning FSDP unit before the submodule forward."""

    def hook(submodule, _args, _kwargs):
        if owner.is_root():
            context = owner.context
            context.allgather_stream.wait_stream(context.current_stream())
        owner.unshard()

    return hook


def _make_unshard_backward_hook(owner: FsdpModule):
    """Create a backward pre-hook that unshards the owning FSDP module before the submodule
    backward."""

    def hook(submodule, _grad_output):
        if owner.is_root():
            owner.context.pre_backward()
        owner.unshard()

    return hook


def _module_post_forward_hook(module: FsdpModule, _args, _output) -> None:
    """Reshard after evaluation, which bypasses the combined schedule's release hooks."""
    if not module.training:
        module.reshard()


def post_fsdp_backward(module: FsdpModule) -> None:
    """Reshard the unit and submit gradient reductions after its last backward use."""
    module.reshard()
    module.post_accumulate_grad()


def reshard_fsdp_modules(module: nn.Module) -> None:
    """Reshard a layer and its independently sharded descendants after forward."""
    units = [unit for unit in module.modules() if isinstance(unit, FsdpModule)]
    for unit in reversed(units):
        unit.reshard()


def register_fsdp_module_hooks(module: FsdpModule) -> None:
    """Register PyTorch hooks for on-demand unshard and eval-time reshard."""

    def register_hooks(submodule, owner):
        if isinstance(submodule, FsdpModule):
            owner = submodule  # BEFORE registering: an FSDP unit owns itself
        if len(list(submodule.parameters(recurse=False))) > 0:
            submodule.register_forward_pre_hook(
                _make_unshard_forward_hook(owner), prepend=True, with_kwargs=True
            )
            submodule.register_full_backward_pre_hook(_make_unshard_backward_hook(owner))
        for child in submodule.children():
            register_hooks(child, owner)

    assert isinstance(module, FsdpModule), "Owner must be an FsdpModule."

    register_hooks(module, module)

    for submodule in module.modules():
        if isinstance(submodule, FsdpModule):
            submodule.register_forward_hook(_module_post_forward_hook)


def register_fsdp_ep_overlap_schedule_hooks(
    model: nn.Module, layer_plans
) -> Callable[[], None] | None:
    """Register layer hooks and return a chunk-level backward completion hook.

    Assign shared units in forward order so they finish at their last backward use.
    Units outside the layer plans, such as the root, finish after pre-process backward.
    """
    remaining = dict.fromkeys(unit for unit in model.modules() if isinstance(unit, FsdpModule))
    if not remaining:
        return None
    for plan in layer_plans:
        units = tuple(unit for unit in plan.layer.modules() if unit in remaining)
        for unit in units:
            del remaining[unit]

        def finish_layer(_module, units=units):
            for unit in units:
                post_fsdp_backward(unit)

        plan.set_fsdp_reshard_hooks(reshard_fsdp_modules, finish_layer)

    if not remaining:
        return None
    chunk_units = tuple(remaining)

    def finish_chunk():
        for unit in chunk_units:
            post_fsdp_backward(unit)

    return finish_chunk
