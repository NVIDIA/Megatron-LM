# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from weakref import ref

from torch import nn

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule

_MFSDP_PARENT_MODULE_REF_ATTR = "_mfsdp_parent_module_ref"


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


def _module_post_backward_hook(module: FsdpModule) -> None:
    module.reshard()
    module._reduce_gradient_groups()


def reshard_fsdp_module(module: FsdpModule) -> None:
    """Reshard the FSDP module after fine-grained computation."""
    assert isinstance(module, FsdpModule), "Expected an FsdpModule."
    module.reshard()


def register_combined_1f1b_hooks(module: FsdpModule) -> None:
    """Install the sub-module hooks required by MCore combined 1F1B."""

    def register_refs(submodule: nn.Module, owner: FsdpModule) -> None:
        """Register references recursively while preserving the nearest FSDP owner."""
        if isinstance(submodule, FsdpModule):
            owner = submodule
        object.__setattr__(submodule, _MFSDP_PARENT_MODULE_REF_ATTR, ref(owner))
        for child in submodule.children():
            register_refs(child, owner)

    register_refs(module, module)

    for submodule in module.modules():
        has_parameters = len(list(submodule.parameters(recurse=False))) > 0
        if has_parameters:
            submodule.register_forward_pre_hook(
                _unshard_before_submodule_forward, prepend=True, with_kwargs=True
            )
            submodule.register_full_backward_pre_hook(_unshard_before_submodule_backward)
        if isinstance(submodule, FsdpModule):
            submodule.register_post_backward_hook(_module_post_backward_hook)
