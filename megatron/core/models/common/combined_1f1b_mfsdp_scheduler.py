# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.phase import Phase
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.runtime_policy import (
    PrefetchOrder,
)


def _make_unshard_forward_hook(owner: FsdpModule, phase: Phase | None):
    """Create a forward pre-hook that unshards the owning FSDP module before the submodule
    forward."""

    def hook(submodule, _args, _kwargs):
        context = owner.context
        # A pipeline chunk can enter capture through a child with its first weights.
        if owner.is_root() or context.prefetch_order is PrefetchOrder.RECORDED:
            context.allgather_stream.wait_stream(context.current_stream())
        owner.unshard(phase)

    return hook


def _make_unshard_backward_hook(owner: FsdpModule, phase: Phase | None):
    """Create a backward pre-hook that unshards the owning FSDP module before the submodule
    backward."""

    def hook(submodule, _grad_output):
        context = owner.context
        if owner.is_root() or context.prefetch_order is PrefetchOrder.RECORDED:
            context.pre_backward()
        owner.unshard(phase)

    return hook


def _module_post_backward_hook(module: FsdpModule) -> None:
    module.reshard()
    module.post_accumulate_grad()


def reshard_fsdp_modules(module: FsdpModule) -> None:
    """Reshard the layer and its independently sharded children."""
    assert isinstance(module, FsdpModule), "Expected an FsdpModule."
    units = [unit for unit in module.modules() if isinstance(unit, FsdpModule)]
    for unit in reversed(units):
        unit.reshard()


def register_combined_1f1b_hooks(module: FsdpModule, *, recompute: bool) -> None:
    """Install the sub-module hooks required by MCore combined 1F1B."""
    recorded = module.context.prefetch_order is PrefetchOrder.RECORDED
    # Backward recomputation can read rowwise weights without re-entering a forward hook.
    backward_phase = Phase.BACKWARD if recorded and not recompute else None

    def register_hooks(submodule, owner):
        if isinstance(submodule, FsdpModule):
            owner = submodule  # BEFORE registering: an FSDP unit owns itself
        if len(list(submodule.parameters(recurse=False))) > 0:
            # TE GroupedTensor forward also prepares columnwise weights for dgrad.
            forward_phase = (
                Phase.FORWARD
                if recorded and not getattr(submodule, "use_grouped_tensor", False)
                else None
            )
            submodule.register_forward_pre_hook(
                _make_unshard_forward_hook(owner, forward_phase), prepend=True, with_kwargs=True
            )
            submodule.register_full_backward_pre_hook(
                _make_unshard_backward_hook(owner, backward_phase)
            )
        for child in submodule.children():
            register_hooks(child, owner)

    assert isinstance(module, FsdpModule), "Owner must be an FsdpModule."
    register_hooks(module, module)

    for submodule in module.modules():
        if isinstance(submodule, FsdpModule):
            if any(group.requires_grad for group in submodule.parameter_groups):
                submodule.register_post_accumulate_grad_hook(_module_post_backward_hook)
            else:
                submodule.register_post_backward_hook(_module_post_backward_hook)
