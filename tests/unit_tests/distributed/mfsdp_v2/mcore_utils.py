# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared helpers for MFSDP v2 MCore tests."""

import torch
from torch.distributed.distributed_c10d import _world
from torch.distributed.tensor.debug import _clear_sharding_prop_cache

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from tests.unit_tests.test_utilities import Utils


def destroy_model_parallel():
    """Utils.destroy_model_parallel, plus the groups it leaves behind.

    It clears Megatron's references but frees only a few of the groups; c10d holds its own
    reference to the rest, so their NCCL communicators -- and, with NVLS enabled, their
    multicast reservations -- would outlive the test. See #6897.
    """
    # TODO(#7322): Remove this workaround when runtime parameters become local tensors.
    # Clear both sharding caches (Python and C++) before destroying groups; otherwise
    # nn.Parameter(DTensor) can reuse stale meshes in later tests through detach().
    _clear_sharding_prop_cache()
    Utils.destroy_model_parallel()
    for group in list(_world.pg_map):
        if group is not torch.distributed.group.WORLD:
            torch.distributed.destroy_process_group(group)


def wrap_with_fsdp_v2(model, config, pg_collection, **kwargs):
    """Wrap a model with the MCore MFSDP v2 adapter under ZeRO-3."""
    return FullyShardedDataParallel(
        config=config,
        ddp_config=DistributedDataParallelConfig(
            use_megatron_fsdp=True,
            megatron_fsdp_version=2,
            use_distributed_optimizer=False,
            data_parallel_sharding_strategy="optim_grads_params",
            fp8_param_gather=config.fp8_param,
        ),
        module=model,
        pg_collection=pg_collection,
        **kwargs,
    )


def units_unsharding_all_planes_in_backward(model) -> set[str]:
    """Return the names of FSDP units whose backward unshards gather all MXFP8 planes."""
    return {
        name
        for name, module in model.module.named_modules()
        if isinstance(module, FsdpModule) and module._schedule_policy.unshard_all_planes_in_backward
    }
