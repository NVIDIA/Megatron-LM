# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Engram parameter policy and native Megatron optimizer integration."""

import copy

import torch

from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.optimizer import (
    USING_PYTORCH_OPTIMIZER,
    Adam,
    _get_megatron_optimizer_based_on_param_groups,
    get_megatron_optimizer,
)
from megatron.core.optimizer.optimizer import ChainedOptimizer, MegatronOptimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import unwrap_model

from .config import ENGRAM_TABLE_LR_MULTIPLIER, ENGRAM_TABLE_WEIGHT_DECAY
from .parallel import EngramParallelGroups


class EngramAdam(Adam):
    """Native Adam update with storage-preserving FP32 checkpoint loading."""

    def load_state_dict(self, state_dict):
        """Restore FP32 Adam state while retaining native parameter storage ownership."""
        for group in self.param_groups:
            if any(param.dtype != torch.float32 for param in group['params']):
                raise ValueError("Engram Adam parameters must be FP32")
        for state in state_dict['state'].values():
            for name, value in state.items():
                if name not in ('exp_avg', 'exp_avg_sq', 'step'):
                    raise ValueError(f"Unsupported Engram Adam state: {name}")
                if name != 'step' and value.dtype != torch.float32:
                    raise ValueError("Engram Adam moments must be FP32")
        return torch.optim.Optimizer.load_state_dict(self, state_dict)


def _validate_owner_config(config: OptimizerConfig, model_chunks: list) -> None:
    if config.optimizer_cpu_offload or config.optimizer_cuda_graph:
        raise ValueError('Engram owners do not support optimizer offload or CUDA graphs')
    if config.fp16 or config.params_dtype == torch.float16:
        raise ValueError('Engram owners support only FP32 and BF16 parameters')
    if config.loss_scale not in (None, 1.0):
        raise ValueError('Engram owners require unity loss scaling')
    if config.optimizer == 'sgd':
        raise ValueError('Engram tables require Adam or an emerging backbone optimizer')
    if config.overlap_param_gather_with_optimizer_step:
        raise ValueError('Engram requires synchronous owner parameter synchronization')
    for chunk in model_chunks:
        if not hasattr(chunk, 'buffers'):
            raise ValueError('Engram owners require native Megatron DDP buffers')
        if chunk.ddp_config.use_megatron_fsdp or chunk.ddp_config.use_custom_fsdp:
            raise ValueError('Engram owners require ordinary Megatron DDP')


def _build_table_optimizer(
    config: OptimizerConfig,
    model_chunks: list,
    param_groups: list[dict],
    groups: EngramParallelGroups,
    pg_collection: ProcessGroupCollection,
) -> MegatronOptimizer:
    """Build native Adam over table buffers already allocated by DDP."""
    _validate_owner_config(config, model_chunks)
    adam_config = copy.copy(config)
    adam_config.optimizer = 'adam'
    adam_config.use_distributed_optimizer = True
    adam_config.use_layer_wise_distributed_optimizer = False
    adam_config.overlap_param_gather = False
    adam_config.reuse_grad_buf_for_mxfp8_param_ag = False
    adam_config.use_precision_aware_optimizer = False
    adam_config.use_precision_aware_optimizer_no_fp8_or_ds_fp8 = False
    owner_params = {param for group in param_groups for param in group['params']}
    if any(param.dtype not in (torch.float32, torch.bfloat16) for param in owner_params):
        raise ValueError('Engram owners support only FP32 and BF16 parameters')
    per_model_buffers = {
        index: [
            buffer
            for buffer in chunk.buffers
            if all(param in owner_params for param in buffer.params)
        ]
        for index, chunk in enumerate(model_chunks)
    }
    bound = {
        param
        for buffers in per_model_buffers.values()
        for buffer in buffers
        for param in buffer.params
    }
    if bound != owner_params:
        raise ValueError('Every Engram parameter must have a native DDP buffer')
    for buffers in per_model_buffers.values():
        for buffer in buffers:
            if (
                buffer.data_parallel_group != groups.replica_group
                or buffer.grad_dtype != torch.float32
            ):
                raise ValueError('Engram buffers require their R replica group and FP32 gradients')
    ddp_config = DistributedDataParallelConfig(
        use_distributed_optimizer=True, grad_reduce_in_fp32=True
    )
    # Native construction copies this ownership marker to model and FP32 master shards.
    for param in owner_params:
        param.tensor_model_parallel = True
        param.main_param_model_shard = None
        if param.dtype == torch.float32:
            param.main_param_sharded = True
            param.main_param = None
    optimizer = _get_megatron_optimizer_based_on_param_groups(
        adam_config,
        model_chunks,
        param_groups,
        per_model_buffers=per_model_buffers,
        data_parallel_group=groups.replica_group,
        data_parallel_group_idx=groups.shard_rank,
        intra_dist_opt_group=groups.stats_group,
        pg_collection=pg_collection,
        ddp_config=ddp_config,
        checkpoint_sharding_type="fully_sharded_model_space",
        checkpoint_step_group=groups.stats_group,
        optimizer_class=None if USING_PYTORCH_OPTIMIZER else EngramAdam,
    )
    # Native construction already binds BF16 masters, including empty-rank ownership.
    # Parameter norms also need FP32 shard ownership and model-value views.
    for model_groups, model_shards in (
        (optimizer.model_float16_groups, optimizer.shard_float16_groups),
        (optimizer.model_fp32_groups, optimizer.shard_fp32_groups),
    ):
        for models, shards in zip(model_groups, model_shards):
            for param, shard in zip(models, shards):
                param.main_param_model_shard = shard
                if param.dtype == torch.float32:
                    param.main_param = shard
    _initialize_optimizer_state(optimizer)
    return optimizer


def _initialize_optimizer_state(optimizer: MegatronOptimizer) -> None:
    """Materialize native optimizer state so initialization checkpoints are complete."""
    children = getattr(optimizer, 'chained_optimizers', None)
    if children is not None:
        for child in children:
            _initialize_optimizer_state(child)
    else:
        init_state_fn = getattr(optimizer, 'init_state_fn', None)
        inner = getattr(optimizer, 'optimizer', None)
        if init_state_fn is not None and inner is not None:
            init_state_fn(inner, optimizer.config)


def get_engram_optimizer(
    config: OptimizerConfig,
    model_chunks: list,
    config_overrides: dict | None = None,
    *,
    use_gloo_process_groups: bool = True,
    pg_collection: ProcessGroupCollection | None = None,
    dump_param_to_param_group_map: str | None = None,
    parallel_groups: EngramParallelGroups | None = None,
) -> MegatronOptimizer:
    """Apply table policy once, then construct native backbone and owner optimizers."""
    if dump_param_to_param_group_map is not None:
        raise ValueError(
            'Engram parameter-group map export is not supported for mixed owner optimizers'
        )
    owners = []

    def table_policy(param, name, merged_override):
        metadata = getattr(param, 'engram_table_metadata', None)
        if metadata is None:
            return None
        override = {'wd_mult': ENGRAM_TABLE_WEIGHT_DECAY, 'lr_mult': ENGRAM_TABLE_LR_MULTIPLIER}
        if config.lr is not None:
            override['max_lr'] = config.lr * ENGRAM_TABLE_LR_MULTIPLIER
        if config.min_lr is not None:
            override['min_lr'] = config.min_lr * ENGRAM_TABLE_LR_MULTIPLIER
        if config.optimizer not in ('adam', 'sgd'):
            override['optimizer'] = 'adam'
        if metadata.row_parallel:
            override.update(is_engram_row_parallel=True, weight_decay=ENGRAM_TABLE_WEIGHT_DECAY)
            if config.lr is not None:
                override['lr'] = (merged_override or {}).get('lr', override['max_lr'])
        return override

    def backbone_filter(group):
        if group.get('is_engram_row_parallel', False):
            owners.append(group)
            return False
        return True

    backbone = get_megatron_optimizer(
        config,
        model_chunks,
        config_overrides,
        use_gloo_process_groups=use_gloo_process_groups,
        pg_collection=pg_collection,
        final_override_fn=table_policy,
        param_group_filter=backbone_filter,
    )
    if not owners:
        return backbone
    if pg_collection is None:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    if parallel_groups is None:
        for module in unwrap_model(model_chunks):
            provider = getattr(module, '_token_context_provider', None)
            engram = provider if provider is not None else getattr(module, 'engram', None)
            if engram is not None and engram.parallel_groups is not None:
                parallel_groups = engram.parallel_groups
                break
    if parallel_groups is None:
        raise ValueError('Engram row optimization requires the model-owned parallel groups')
    owner = _build_table_optimizer(config, model_chunks, owners, parallel_groups, pg_collection)
    _initialize_optimizer_state(backbone)
    # Preserve specialized wrappers, but avoid nesting an ordinary optimizer chain.
    children = backbone.chained_optimizers if type(backbone) is ChainedOptimizer else [backbone]
    return ChainedOptimizer(
        [*children, owner],
        synchronize_nonfinite_grads=True,
        nonfinite_grad_group=parallel_groups.stats_group,
    )
