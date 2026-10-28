# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from torch import nn

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.parameter_group import (
    get_containing_parameter_group,
)


def _make_unshard_forward_hook(owner: FsdpModule, weight_module: nn.Module | None = None):
    """Create a forward pre-hook that unshards the owning FSDP module before the submodule
    forward."""

    def hook(submodule, _args, _kwargs):
        if owner.is_root():
            context = owner.context
            context.allgather_stream.wait_stream(context.current_stream())
        owner.unshard()
        if weight_module is not None:
            # The caller may have passed the sharded embedding weight before this hook runs.
            _kwargs["weight"] = weight_module.weight
            return _args, _kwargs

    return hook


def _make_unshard_backward_hook(owner: FsdpModule):
    """Create a backward pre-hook that unshards the owning FSDP module before the submodule
    backward."""

    def hook(submodule, _grad_output):
        owner.unshard()

    return hook


def _module_post_forward_hook(module: FsdpModule, _args, _output) -> None:
    """Reshard after evaluation, which bypasses the combined schedule's release hooks."""
    if not module.training:
        module.reshard()


def finish_fsdp_backward(module: FsdpModule) -> None:
    """Release weights and reduce gradients after all scheduled consumers finish."""
    module.reshard()
    module._reduce_gradient_groups()


def reshard_fsdp_module(module: FsdpModule) -> None:
    """Reshard the FSDP module after fine-grained computation."""
    assert isinstance(module, FsdpModule), "Expected an FsdpModule."
    module.reshard()


def register_combined_1f1b_hooks(module: FsdpModule) -> None:
    """Install demand-gather and evaluation-release hooks for the EP schedule."""

    tied_outputs: dict[nn.Module, nn.Module] = {}

    def register_hooks(submodule, owner):
        if getattr(submodule, "share_embeddings_and_output_weights", False):
            embedding = getattr(submodule, "embedding", None)
            output_layer = getattr(submodule, "output_layer", None)
            if embedding is not None and output_layer is not None:
                tied_outputs[output_layer] = embedding.word_embeddings
        if isinstance(submodule, FsdpModule):
            owner = submodule  # BEFORE registering: an FSDP unit owns itself
        if len(list(submodule.parameters(recurse=False))) > 0:
            submodule.register_forward_pre_hook(
                _make_unshard_forward_hook(owner), prepend=True, with_kwargs=True
            )
            submodule.register_full_backward_pre_hook(_make_unshard_backward_hook(owner))
        elif submodule in tied_outputs:
            weight_module = tied_outputs[submodule]
            group = get_containing_parameter_group(weight_module.weight)
            weight_owner = group._owning_module() if group is not None else None
            if weight_owner is not None:
                submodule.register_forward_pre_hook(
                    _make_unshard_forward_hook(weight_owner, weight_module),
                    prepend=True,
                    with_kwargs=True,
                )
                submodule.register_full_backward_pre_hook(_make_unshard_backward_hook(weight_owner))
        for child in submodule.children():
            register_hooks(child, owner)

    assert isinstance(module, FsdpModule), "Owner must be an FsdpModule."

    register_hooks(module, module)

    for submodule in module.modules():
        if isinstance(submodule, FsdpModule):
            submodule.register_forward_hook(_module_post_forward_hook)


def register_fsdp_schedule_hooks(model: nn.Module, layer_plans) -> tuple[FsdpModule, ...]:
    """Reduce each unit after its last scheduled backward use, including delayed wgrad.

    Operations:
        1. Collect FSDP units from the current chunk's outer model.
        2. Visit layers in forward order and assign each unit once, so shared
           units finish at their last backward use (after delayed wgrad).
        3. Return the remaining units for completion after pre-process backward.
           These are typically root units owning embeddings, tied/untied output
           heads, final normalization, and MTP projections/norms.
    """
    remaining = dict.fromkeys(unit for unit in model.modules() if isinstance(unit, FsdpModule))
    if not remaining:
        return ()
    for plan in layer_plans:
        units = tuple(unit for unit in plan.layer.modules() if unit in remaining)
        for unit in units:
            del remaining[unit]

        def finish_layer(_module, units=units):
            for unit in units:
                finish_fsdp_backward(unit)

        plan.set_fsdp_reshard_hooks(reshard_fsdp_module, finish_layer)
    return tuple(remaining)
