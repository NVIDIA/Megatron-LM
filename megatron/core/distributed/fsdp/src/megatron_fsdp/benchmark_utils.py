# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import functools
import itertools

from .fsdp_hook_utils import register_backward_hook, register_post_backward_hook
from .megatron_fsdp import MegatronFSDP
from .utils import is_submodule


def disable_megatron_fsdp_communication(module: MegatronFSDP, fill_buckets: bool = True):
    """Disable Megatron-FSDP communication by allocating and deallocating
    parameter buckets around forward and backward passes. This is useful for benchmarking
    the overhead of Megatron-FSDP communication."""
    ddp_config = module.ddp_config
    assert ddp_config.data_parallel_sharding_strategy == "optim_grads_params"

    module._root_pre_backward_hook_handle.remove()
    for handle in itertools.chain(
        module.backward_pre_hooks.values(),
        module.forward_pre_hooks.values(),
        module.forward_hooks.values(),
    ):
        handle.remove()

    bucket_map = {}

    def _allocate_params(param_and_grad_buffer, params, *args, **kwargs):
        for param in params:
            group_id = param_and_grad_buffer.param_to_param_group[param]
            if group_id in bucket_map:
                continue
            bucket_map[group_id] = True
            group = param_and_grad_buffer.parameter_groups[group_id]
            wbuf = group.model_weight_buffer
            bucket = wbuf.fetch_bucket()
            if fill_buckets:
                bucket.data.uniform_(0, 1)

    def _deallocate_params(param_and_grad_buffer, params, *args, **kwargs):
        for param in params:
            group_id = param_and_grad_buffer.param_to_param_group[param]
            if group_id not in bucket_map:
                continue
            group = param_and_grad_buffer.parameter_groups[group_id]
            wbuf = group.model_weight_buffer
            wbuf.free_bucket_storage()
            del bucket_map[group_id]

    fake_fsdp_hooks = []
    module._replace_param_with_raw_if_needed()
    try:
        root_module = module.module
        param_and_grad_buffer = module.param_and_grad_buffer
        fsdp_unit_modules = module.fsdp_unit_modules
        fsdp_modules = []
        for m in root_module.modules():
            # Skip if the module is already registered in fsdp_modules.
            if any(is_submodule(m, fsdp_module) for fsdp_module in fsdp_modules):
                continue

            if isinstance(m, tuple(fsdp_unit_modules)):
                fsdp_modules.append(m)
                param_list = list(m.parameters())
                pre_forward_hook = m.register_forward_pre_hook(
                    functools.partial(_allocate_params, param_and_grad_buffer, param_list)
                )
                post_forward_hook = m.register_forward_hook(
                    functools.partial(_deallocate_params, param_and_grad_buffer, param_list)
                )
                backward_pre_hook = register_backward_hook(
                    m, functools.partial(_allocate_params, param_and_grad_buffer, param_list)
                )
                post_backward_hook = register_post_backward_hook(
                    m, functools.partial(_deallocate_params, param_and_grad_buffer, param_list)
                )
                fake_fsdp_hooks += [
                    pre_forward_hook,
                    post_forward_hook,
                    backward_pre_hook,
                    post_backward_hook,
                ]
            else:
                param_list = list(m.parameters(recurse=False))
                pre_forward_hook = m.register_forward_pre_hook(
                    functools.partial(_allocate_params, param_and_grad_buffer, param_list)
                )
                fake_fsdp_hooks.append(pre_forward_hook)
    finally:
        module._replace_param_with_distributed_if_needed()

    return fake_fsdp_hooks
