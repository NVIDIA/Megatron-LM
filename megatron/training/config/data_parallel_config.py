# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CLI names and migration of saved data-parallel configurations.

Legacy names belong at the training input boundary. Core DDP and FSDP configs
and model constructors only consume their own native schemas.
"""

from dataclasses import fields

from megatron.core.distributed import DistributedDataParallelConfig, FullyShardedDataParallelConfig

FSDP_ARGUMENT_NAMES = {
    'version': 'megatron_fsdp_version',
    'padded_shards': 'megatron_fsdp_padded_shards',
    'double_buffer': 'fsdp_double_buffer',
    'db_use_persist_buf_on_alloc_fail': 'fsdp_db_use_persist_buf_on_alloc_fail',
    'all_gather_in_start_param_sync': 'fsdp_all_gather_in_start_param_sync',
    'manual_registration': 'fsdp_manual_registration',
    'main_params_dtype': 'megatron_fsdp_main_params_dtype',
    'main_grads_dtype': 'megatron_fsdp_main_grads_dtype',
    'grad_comm_dtype': 'megatron_fsdp_grad_comm_dtype',
    'use_decoupled_grad': 'megatron_fsdp_use_decoupled_grad',
    'cuda_graph_mode': 'megatron_fsdp_cuda_graph_mode',
    'enable_fine_grained_param_gather': 'megatron_fsdp_enable_fine_grained_param_gather',
    'max_pool_double_buffer': 'megatron_fsdp_max_pool_double_buffer',
}

# Old DDP payloads serialized these fields even when FSDP was disabled.
_LEGACY_FSDP_FIELDS = frozenset(
    [
        'data_parallel_sharding_strategy',
        'expert_data_parallel_sharding_strategy',
        'expert_outer_dp_sharding_strategy',
        'fsdp_all_gather_in_start_param_sync',
        'fsdp_db_use_persist_buf_on_alloc_fail',
        'fsdp_double_buffer',
        'fsdp_manual_registration',
        'gradient_reduce_div_fusion',
        'hfsdp_param_gather_overlap',
        'keep_fp8_transpose_cache',
        'megatron_fsdp_cuda_graph_mode',
        'megatron_fsdp_enable_fine_grained_param_gather',
        'megatron_fsdp_grad_comm_dtype',
        'megatron_fsdp_main_grads_dtype',
        'megatron_fsdp_main_params_dtype',
        'megatron_fsdp_max_pool_double_buffer',
        'megatron_fsdp_padded_shards',
        'megatron_fsdp_use_decoupled_grad',
        'megatron_fsdp_version',
        'outer_dp_sharding_strategy',
        'overlap_outer_communication',
        'suggested_communication_unit_size',
        'use_custom_fsdp',
        'use_megatron_fsdp',
    ]
)
_DDP_TARGETS = {
    "megatron.core.distributed.DistributedDataParallelConfig",
    "megatron.core.distributed.distributed_data_parallel_config.DistributedDataParallelConfig",
}
_TORCH_FSDP_TARGETS = {
    "megatron.core.distributed.TorchFullyShardedDataParallelConfig",
    "megatron.core.distributed.torch_fully_sharded_data_parallel_config.TorchFullyShardedDataParallelConfig",
}
_FSDP_TARGET = (
    "megatron.core.distributed.fully_sharded_data_parallel_config.FullyShardedDataParallelConfig"
)


def migrate_data_parallel_config(payload: dict) -> dict:
    """Translate a saved training container before any dataclass is instantiated.

    The caller owns a deep copy. Unknown settings survive so strict deserialization
    can reject them; only known, inactive fields of the old union schema are dropped.
    """
    legacy = payload.pop("ddp", None)
    if legacy is not None:
        if payload.get("data_parallel") is not None:
            raise ValueError("Specify only one of data_parallel and ddp")
        payload["data_parallel"] = legacy
    config = payload.get("data_parallel")
    if not isinstance(config, dict):
        return payload
    target = config.get("_target_")
    if target not in _DDP_TARGETS | _TORCH_FSDP_TARGETS:
        return payload
    use_fsdp = config.get("use_megatron_fsdp", False) or config.get("use_custom_fsdp", False)
    if use_fsdp and target in _TORCH_FSDP_TARGETS:
        raise ValueError("Cannot select both Megatron-FSDP and Torch FSDP")
    if not use_fsdp:
        for name in _LEGACY_FSDP_FIELDS:
            config.pop(name, None)
        return payload

    ddp_fields = {f.name for f in fields(DistributedDataParallelConfig)}
    fsdp_fields = {f.name for f in fields(FullyShardedDataParallelConfig)}
    old_to_new = {old: new for new, old in FSDP_ARGUMENT_NAMES.items()}
    converted = {"_target_": _FSDP_TARGET}
    for name, value in config.items():
        if name in ("_target_", "use_megatron_fsdp", "use_custom_fsdp"):
            continue
        native_name = old_to_new.get(name, name)
        if native_name in fsdp_fields or name not in ddp_fields:
            converted[native_name] = value
    # Old training saved the requested bucket count alongside the derived size.
    if converted.get("num_buckets") is not None and converted.get("bucket_size") is not None:
        if converted.get("version", 1) == 1:
            converted["num_buckets"] = None
        else:
            converted["bucket_size"] = None  # Preserve the unsupported v2 request for validation.
    payload["data_parallel"] = converted
    return payload
