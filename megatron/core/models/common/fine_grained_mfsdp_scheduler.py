# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from weakref import ref

from torch import nn

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule

_MFSDP_PARENT_MODULE_REF_ATTR = "_fsdp_parent_module_ref"
_MFSDP_SCHEDULER_ATTR = "_fsdp_scheduler"


def _get_owning_fsdp_module(submodule: nn.Module) -> FsdpModule | None:
    """Return the FsdpModule that owns a fine-grained schedule submodule."""
    if isinstance(submodule, FsdpModule):
        return submodule
    parent_ref = getattr(submodule, _MFSDP_PARENT_MODULE_REF_ATTR, None)
    return parent_ref() if parent_ref is not None else None


def _unshard_before_submodule_forward(submodule: nn.Module, _args, _kwargs) -> None:
    """Materialize the owning FSDP unit before a schedule sub-module runs."""
    fsdp_module = _get_owning_fsdp_module(submodule)
    assert fsdp_module is not None, "FSDP module not found for submodule."
    if fsdp_module.is_root():
        context = fsdp_module.context
        context.allgather_stream.wait_stream(context.current_stream())

    fsdp_module.unshard()


def _unshard_before_submodule_backward(submodule: nn.Module, _grad_output) -> None:
    """Enter the owning FSDP unit's backward lifecycle before sub-module backward."""
    fsdp_module = _get_owning_fsdp_module(submodule)
    assert fsdp_module is not None, "FSDP module not found for submodule."
    fsdp_module.unshard()
    if fsdp_module.is_root():
        context = fsdp_module.context
        context.reduce_scatter_stream.wait_stream(context.current_stream())


def _set_owning_fsdp_module_refs(module: FsdpModule) -> None:
    """Recursively set the nearest FsdpModule reference on all sub-modules.

    This is used by fine-grained schedules to find the owning FSDP unit for a
    sub-module without traversing the module tree.
    """

    def register_refs(submodule: nn.Module, owner: FsdpModule) -> None:
        """Register references recursively while preserving the nearest FSDP owner."""
        if isinstance(submodule, FsdpModule):
            owner = submodule
        object.__setattr__(submodule, _MFSDP_PARENT_MODULE_REF_ATTR, ref(owner))
        for child in submodule.children():
            register_refs(child, owner)

    register_refs(module, module)


def reshard_fsdp_module(module: FsdpModule) -> None:
    """Reshard the FSDP module after fine-grained computation."""
    assert isinstance(module, FsdpModule), "Expected an FsdpModule."
    module.reshard()


def module_post_backward_hook(module: FsdpModule) -> None:
    reshard_fsdp_module(module)
    module._reduce_gradient_groups()


def _register_combined_1f1b_hooks(module: FsdpModule) -> None:
    """Install the sub-module hooks required by MCore combined 1F1B."""

    for submodule in module.modules():
        submodule.register_forward_pre_hook(
            _unshard_before_submodule_forward, prepend=True, with_kwargs=True
        )
        submodule.register_full_backward_pre_hook(_unshard_before_submodule_backward)
        if isinstance(submodule, FsdpModule):
            submodule.register_post_backward_hook(module_post_backward_hook)


def setup_combined_1f1b_hooks(module: nn.Module) -> None:
    """Install the parameter lifecycle hooks used by combined 1F1B."""
    # The scheduler is driven by the submodule hooks below; it must not rely on
    # the static prefetch/lifecycle assumptions of the automatic FSDP hooks. The
    # FSDP unit's own idempotent ``unshard()``/``reshard()`` are used directly so
    # the scheduler works on the generic MFSDP v2 lifecycle.
    root_modules = [
        submodule
        for submodule in module.modules()
        if isinstance(submodule, FsdpModule) and submodule.is_root()
    ]
    assert root_modules, "Root FSDP module not found."

    for root_module in root_modules:
        if hasattr(root_module, _MFSDP_SCHEDULER_ATTR):
            continue
        _set_owning_fsdp_module_refs(root_module)
        _register_combined_1f1b_hooks(root_module)
        object.__setattr__(root_module, _MFSDP_SCHEDULER_ATTR, True)
