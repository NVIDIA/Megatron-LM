# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import copy
import logging
import warnings
from dataclasses import astuple
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.optim import SGD as CPUSGD
from torch.optim import AdamW as CPUAdam

try:
    from transformer_engine.pytorch.optimizers import FusedAdam as Adam
    from transformer_engine.pytorch.optimizers import FusedSGD as SGD

    USING_PYTORCH_OPTIMIZER = False
except ImportError:
    try:
        from apex.optimizers import FusedAdam as Adam
        from apex.optimizers import FusedSGD as SGD

        USING_PYTORCH_OPTIMIZER = False
    except ImportError:
        warnings.warn(
            f'Transformer Engine and Apex are not installed. Falling back to Torch optimizers.'
        )

        # Apex's FusedAdam is a drop-in replacement for torch's AdamW.
        # pylint: disable-next=line-too-long.
        # See https://github.com/NVIDIA/apex/blob/7b73b12361068a10b0f44844534613f252a5ea75/apex/optimizers/fused_adam.py#L16.
        from torch.optim import SGD
        from torch.optim import AdamW as Adam

        USING_PYTORCH_OPTIMIZER = True

from megatron.core import parallel_state
from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.fsdp_dtensor_checkpoint import get_global_unique_param_name

from ..distributed.param_and_grad_buffer import _ParamAndGradBuffer
from ..transformer.module import MegatronModule
from ..utils import get_model_config, get_pg_rank, get_pg_size, is_te_min_version, log_single_rank
from .distrib_optimizer import DistributedOptimizer
from .grad_scaler import ConstantGradScaler, DynamicGradScaler
from .optimizer import (
    ChainedOptimizer,
    Float16OptimizerWithFloat16Params,
    FP32Optimizer,
    MegatronOptimizer,
    param_group_identifier_keys,
)
from .optimizer_config import AdamOptimizerConfig, OptimizerConfig, ParamKey, SGDOptimizerConfig

logger = logging.getLogger(__name__)


def _matches(param: torch.nn.Parameter, param_name: str, param_key: ParamKey) -> bool:
    """Returns true if passed-in parameter (with name) matches `param_key`.

    Args:
        param (torch.nn.Parameter): Handle to parameter object.
        param_name (str): Name of parameter in underlying PyTorch module.
        param_key (ParamKey): ParamKey object.

    Returns:
        bool: True if parameter matches passed-in param_key.
    """

    # Check if name matches.
    if isinstance(param_key.name, str):
        target_names = [param_key.name]
    else:
        target_names = list(param_key.name)
    for target_name in target_names:
        if param_name in target_name:
            return True

    # Check if attribute matches.
    if isinstance(param_key.attr, str):
        target_attrs = [param_key.attr]
    else:
        target_attrs = list(param_key.attr)
    for target_attr in target_attrs:
        if getattr(param, target_attr, False):
            return True

    return False


def _get_param_groups(
    model_chunks: List[MegatronModule],
    config: OptimizerConfig,
    config_overrides: Optional[Dict[ParamKey, OptimizerConfig]],
) -> List[Dict]:
    """Create parameter groups for optimizer.

    Creates parameter groups from provided optimizer config object.

    Args:
        model_chunks (List[MegatronModule]): model chunks to create parameter
            groups for.
        config (OptimizerConfig): optimizer configuration object.
        config_overrides (Optional[Dict[LayerKey, OptimizerConfig]): optimizer overrides,
            specified on a per-layer basis.
    Returns:
        List of parameter groups.
    """

    # Map (wd_mult, is_expert_parallel, param_group_hyperparameters_config) to params.
    params_map = {}
    configs_map = {}

    for model_chunk in model_chunks:
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue

            uses_default_config = False
            # Get optimizer config for this parameter.
            if config_overrides is None:
                config_for_param = config
                uses_default_config = True
            else:
                config_for_param = None
                for param_key in config_overrides:
                    if _matches(param, name, param_key):
                        config_for_param = config_overrides[param_key]
                        break
                # Fall back to default config.
                if config_for_param is None:
                    config_for_param = config
                    uses_default_config = True

            is_expert_parallel = not getattr(param, 'allreduce', True)

            # TODO: Make sure there is a way to support old no_weight_decay_func functionality
            # and default_skip_embedding_weight_decay:
            #     or (default_skip_embedding_weight_decay and "embedding" in name)
            no_wd = name.endswith(".bias") or len(param.shape) == 1
            if not no_wd:
                wd_mult = 1.0
            else:
                wd_mult = 0.0

            # Create config_tuple that is hash-able. Remove timers object before
            # creating config_tuple.
            config_for_param_copy = copy.deepcopy(config_for_param)
            config_for_param_copy.timers = None
            config_tuple = astuple(config_for_param_copy)
            key = (wd_mult, is_expert_parallel, config_tuple)
            if key not in params_map:
                params_map[key] = []
            params_map[key].append(param)

            if key in configs_map:
                assert (config_for_param, uses_default_config) == configs_map[key]
            else:
                configs_map[key] = (config_for_param, uses_default_config)

    # Distributed checkpoint requires all ranks to have the same param groups,
    # so we need to align the param groups across ranks, otherwise we may have
    # runtime error when loading the checkpoint or numerical error when resuming training.
    params_key = list(params_map.keys())
    gathered_params_key = [None for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(gathered_params_key, params_key)
    for keys in gathered_params_key:
        for key in keys:
            if key not in params_key:
                params_key.append(key)

    param_groups = []
    for key in params_key:
        wd_mult, is_expert_parallel, _ = key
        params = params_map[key] if key in params_map else []
        config, uses_default_config = None, True
        if key not in configs_map:
            assert params == []
        else:
            config, uses_default_config = configs_map[key]
            assert config is not None

        # TODO: Remove "backwards compatible" fields below eventually.
        param_group = {
            'params': params,
            'wd_mult': wd_mult,  # For backwards compatibility.
            'lr_mult': 1.0,  # For backwards compatibility.
            'is_expert_parallel': is_expert_parallel,
            'is_decoupled_lr': False,  # For backwards compatibility.
            'default_config': uses_default_config,
        }

        # Stick relevant fields into param_group from config object.
        if config is not None:
            param_group['max_lr'] = config.lr
            param_group['min_lr'] = config.min_lr
            # TODO: Add other relevant arguments (e.g., weight decay, optimizer)
            # here as well.
        param_groups.append(param_group)

    return param_groups


def _get_param_groups_and_buffers(
    model_chunks: List[MegatronModule],
    model_chunk_offset: int,
    config: OptimizerConfig,
    config_overrides: Optional[Dict[ParamKey, OptimizerConfig]],
    filter_fn: Callable,
    buffer_name: str,
) -> Tuple[List[Dict], Dict[int, List[_ParamAndGradBuffer]]]:
    """Returns parameter groups and buffer for optimizer.

    Args:
        model_chunks (List[MegatronModule]): model chunks to create parameter
            groups for.
        model_chunk_offset (int): offset of model_chunks in global model_chunks list.
        config (OptimizerConfig): optimizer configuration object.
        config_overrides (Optional[Dict[LayerKey, OptimizerConfig]): optimizer overrides,
            specified on a per-layer basis.
        lr (float): learning rate.
        min_lr (float): minimum learning rate.
        filter_fn (callable): filtering function for param_groups.
        buffer_name (str): name of buffer.

    Returns:
        List of parameter groups and dictionary of model chunk IDs to buffers.
    """
    param_groups = _get_param_groups(model_chunks, config, config_overrides)
    param_groups = list(filter(filter_fn, param_groups))
    buffers = {}
    for model_chunk_idx, model_chunk in enumerate(model_chunks):
        if hasattr(model_chunk, buffer_name):
            buffers[model_chunk_idx + model_chunk_offset] = getattr(model_chunk, buffer_name)

    return param_groups, buffers


def _get_megatron_optimizer_based_on_param_groups(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    param_groups: List,
    per_model_buffers: Optional[Dict[int, List[_ParamAndGradBuffer]]] = None,
    model_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    data_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    data_parallel_group_gloo: Optional[torch.distributed.ProcessGroup] = None,
    data_parallel_group_idx: Optional[int] = None,
    intra_dist_opt_group: Optional[torch.distributed.ProcessGroup] = None,
    distributed_optimizer_instance_id: Optional[int] = 0,
) -> MegatronOptimizer:
    """Get Megatron optimizer based on parameter groups.

    Args:
        config (OptimizerConfig): optimizer configuration object.
        model_chunks (list): list of model chunks.
        param_groups (list): list of parameter groups.
        per_model_buffers (dict, optional): buffers for distributed optimizer. Defaults to None.
        data_parallel_group (torch.distributed.ProcessGroup, optional): data-parallel group for
            distributed optimizer. Defaults to None.
        data_parallel_group_gloo (torch.distributed.ProcessGroup, optional): gloo data-parallel
            group for distributed optimizer. Defaults to None.
        data_parallel_group_idx (int, optional): data-parallel group index for distributed
            optimizer. Defaults to None.
        distributed_optimizer_instance_id (int, optional): Distributed optimizer instance. Defaults
            0.

    Returns:
        Instance of MegatronOptimizer.
    """
    # TODO: Logic needs to be updated to handle different optimizer types (i.e., param_groups
    # passed into this function need to correspond to the same optimizer).

    # When freezing sub-models we may have no trainable parameters on a rank and
    # hence an empty param_groups. However, we still need to create an optimizer
    # for the purposes of grad stats reductions.
    if param_groups:
        if config.optimizer_cpu_offload:
            if torch.__version__ < '2.3.0':
                warnings.warn(
                    "CPU offload is recommended for PyTorch >= 2.3.0, "
                    "untested versions below this may have convergence issues."
                )
            assert (
                config.decoupled_weight_decay
            ), "CPU offloading only supported with decoupled_weight_decay enabled (AdamW mode)."
            gpu_optimizer_cls = Adam if config.optimizer == 'adam' else SGD
            cpu_optimizer_cls = CPUAdam if config.optimizer == 'adam' else CPUSGD
            if config.use_torch_optimizer_for_cpu_offload:
                gpu_optimizer_cls = cpu_optimizer_cls
            if config.optimizer == 'adam':
                gpu_optimizer_cls = Adam
                cpu_optimizer_cls = CPUAdam
                optimizer_defaults = dict(
                    lr=config.lr,
                    weight_decay=config.weight_decay,
                    betas=(config.adam_beta1, config.adam_beta2),
                    eps=config.adam_eps,
                    bias_correction=True,
                    fused=True,  # this flag is used to improve the performance of the cpu optimizer
                )
            else:
                gpu_optimizer_cls = SGD
                cpu_optimizer_cls = CPUSGD
                optimizer_defaults = dict(
                    lr=config.lr, weight_decay=config.weight_decay, momentum=config.sgd_momentum
                )
            optimizer = HybridDeviceOptimizer(
                param_groups,
                offload_fraction=config.optimizer_offload_fraction,
                cpu_optimizer_cls=cpu_optimizer_cls,
                gpu_optimizer_cls=gpu_optimizer_cls,
                overlap_cpu_optimizer_d2h_h2d=config.overlap_cpu_optimizer_d2h_h2d,
                pin_cpu_grads=config.pin_cpu_grads,
                pin_cpu_params=config.pin_cpu_params,
                param_update_in_fp32=True,
                **optimizer_defaults,
            )
            init_state_fn = None
        elif config.optimizer == 'adam':
            kwargs = {
                "params": param_groups,
                "lr": config.lr,
                "weight_decay": config.weight_decay,
                "betas": (config.adam_beta1, config.adam_beta2),
                "eps": config.adam_eps,
            }

            # set Adam class and weight decay mode depending
            # on source of optimizer (Torch or TE/Apex)
            if USING_PYTORCH_OPTIMIZER:
                adam_cls = torch.optim.AdamW if config.decoupled_weight_decay else torch.optim.Adam
            else:
                kwargs["adam_w_mode"] = config.decoupled_weight_decay
                adam_cls = Adam

            if config.use_precision_aware_optimizer:
                kwargs.update(
                    {
                        "exp_avg_dtype": config.exp_avg_dtype,
                        "exp_avg_sq_dtype": config.exp_avg_sq_dtype,
                    }
                )
                # Master weight is managed by MCore when main_params_dtype is fp32. This is
                # because we want to use fp8 primary weight with precision aware optimizer.
                # Otherwise, master weight will be managed by TransformerEngine.
                # Delayed scaling is an exception because casting as well as the computation
                # of the scaling factor can be conducted in the adam kernel.
                if config.use_precision_aware_optimizer_no_fp8_or_ds_fp8:
                    kwargs.update(
                        {
                            "master_weights": True,
                            "use_decoupled_grad": True,
                            "master_weight_dtype": config.main_params_dtype,
                        }
                    )

                if is_te_min_version("2.1.0.dev0"):
                    kwargs.update({"store_param_remainders": config.store_param_remainders})

            optimizer = adam_cls(**kwargs)

            def init_state_fn(opt, config=None):
                for group in opt.param_groups:
                    for p in group['params']:
                        if len(opt.state[p]) == 0:
                            if config is None or not config.use_precision_aware_optimizer:
                                opt.state[p]['exp_avg'] = torch.zeros_like(p.data)
                                opt.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)
                            else:
                                opt.initialize_state(p)

        elif config.optimizer == 'sgd':
            optimizer = SGD(
                param_groups,
                lr=config.lr,
                weight_decay=config.weight_decay,
                momentum=config.sgd_momentum,
            )
            init_state_fn = None
        else:
            raise Exception('{} optimizer is not supported.'.format(config.optimizer))
    else:
        optimizer = None
        init_state_fn = None

    # Mixed precision optimizer.
    # - Note: both the Float16Optimizer and the DistributedOptimizer inherit
    #   from the MixedPrecisionOptimizer, which manages any optimizer where
    #   the model params and main params are distinct.
    if config.fp16 or config.bf16 or config.use_distributed_optimizer:

        # Grad scaler:
        #    if loss-scale is provided, instantiate the constant scaler.
        #    if we are using fp16 and loss-scale is not present, use a
        #       dynamic scaler.
        #    otherwise we are running in bf16 with no loss-scale so
        #       leave it as None.
        grad_scaler = None

        # Constant loss scale.
        if config.loss_scale:
            grad_scaler = ConstantGradScaler(config.loss_scale)

        # Dynamic loss scale.
        else:
            if config.fp16:
                grad_scaler = DynamicGradScaler(
                    initial_scale=config.initial_loss_scale,
                    min_scale=config.min_loss_scale,
                    growth_factor=2.0,
                    backoff_factor=0.5,
                    growth_interval=config.loss_scale_window,
                    hysteresis=config.hysteresis,
                )

        optimizer_args = [optimizer, config, grad_scaler, init_state_fn]
        if config.use_distributed_optimizer:
            optimizer = DistributedOptimizer(
                *optimizer_args,
                model_chunks=model_chunks,
                per_model_buffers=per_model_buffers,
                data_parallel_group=data_parallel_group,
                data_parallel_group_gloo=data_parallel_group_gloo,
                data_parallel_group_idx=data_parallel_group_idx,
                distributed_optimizer_instance_id=distributed_optimizer_instance_id,
            )
            # This is needed for case where num_distributed_optimizer_instances > 1. In this case,
            # weight gradients are all-reduced across optimizer instances, so each instance has
            # the duplicated weight gradients, need to reduce gradient stats inside each instance.
            setattr(optimizer, 'grad_stats_parallel_group', intra_dist_opt_group)
        else:
            optimizer = Float16OptimizerWithFloat16Params(*optimizer_args)
            setattr(optimizer, 'grad_stats_parallel_group', model_parallel_group)
    else:
        # FP32 optimizer.
        optimizer = FP32Optimizer(optimizer, config, init_state_fn)
        setattr(optimizer, 'grad_stats_parallel_group', model_parallel_group)

    return optimizer


def get_megatron_optimizer(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    config_overrides: Optional[Dict[ParamKey, OptimizerConfig]] = None,
    use_gloo_process_groups: bool = True,
    pg_collection: Optional[ProcessGroupCollection] = None,
    dump_param_to_param_group_map: Optional[str] = None,
) -> MegatronOptimizer:
    """Retrieve the Megatron optimizer for model chunks.

    We use separate optimizers for expert parameters and non-expert parameters.

    Args:
        config (OptimizerConfig): optimizer configuration object.
        model_chunks (List[MegatronModule]): model chunks to get optimizer for.
        config_overrides (Optional[Dict[ParamKey, OptimizerConfig]]): optional dictionary of
            optimizer configuration objects to override default optimizer behavior for different
            subsets of parameters (identified by ParamKey).
        use_gloo_process_groups (bool): if false, disable use of Gloo process groups
            in underlying Megatron optimizers.
        pg_collection: Optional unified process group for distributed training.
        dump_param_to_param_group_map (Optional[str]): path to dump parameter to param group map.

    Returns:
        Instance of MegatronOptimizer.
    """

    log_single_rank(logger, logging.INFO, f'Setting up optimizer with config {config}')

    # TODO: Remove `optimizer` from this eventually (e.g., if we use Muon for some layers and
    # Adam for other layers). This would need some more refactoring to work though (param_groups
    # filtered by optimizer passed into _get_megatron_optimizer_based_on_param_groups).
    fields_to_check_for_consistency = [
        'overlap_param_gather_with_optimizer_step',
        'optimizer',
        'optimizer_cpu_offload',
    ]
    for field_name in fields_to_check_for_consistency:
        field = getattr(config, field_name, None)
        if config_overrides is not None:
            all_configs = list(config_overrides.values())
            assert all([getattr(x, field_name, None) == field for x in all_configs])

    # Separate out first model chunk if overlapping param AG with optimizer step.
    if config.overlap_param_gather_with_optimizer_step:
        all_dense_model_chunks = [[model_chunks[0]], model_chunks[1:]]
        overlap_param_gather_with_optimizer_step_flags = [True, False]
    else:
        all_dense_model_chunks = [model_chunks]
        overlap_param_gather_with_optimizer_step_flags = [False]

    # Setup process groups using helper method
    process_groups = ProcessGroupCollection.setup_process_groups_for_optimizer(
        pg_collection, model_chunks, use_gloo_process_groups
    )

    dp_cp_group = process_groups['dp_cp_group']
    intra_dp_cp_group = process_groups['intra_dp_cp_group']
    intra_expt_dp_group = process_groups['intra_expt_dp_group']
    mp_group = process_groups['mp_group']
    expt_tp_pp_group = process_groups['expt_tp_pp_group']
    intra_dp_cp_group_gloo = process_groups['intra_dp_cp_group_gloo']
    intra_expt_dp_group_gloo = process_groups['intra_expt_dp_group_gloo']
    intra_dist_opt_group = process_groups['intra_dist_opt_group']

    model_parallel_rank = get_pg_rank(mp_group)

    if get_pg_size(dp_cp_group) > get_pg_size(intra_dp_cp_group):
        inter_dist_opt_group = process_groups['inter_dist_opt_group']
        distributed_optimizer_instance_id = get_pg_rank(inter_dist_opt_group)
    else:
        distributed_optimizer_instance_id = 0

    optimizers = []
    model_chunk_offset = 0
    ddp_config = model_chunks[0].ddp_config  # Use the first model chunk's DDP config
    if ddp_config.use_megatron_fsdp:
        for model_chunk, overlap_param_gather_with_optimizer_step in zip(
            all_dense_model_chunks, overlap_param_gather_with_optimizer_step_flags
        ):
            param_groups, buffers = _get_param_groups_and_buffers(
                model_chunk,
                model_chunk_offset=model_chunk_offset,
                config=config,
                config_overrides=config_overrides,
                filter_fn=lambda g: True,
                buffer_name='buffers',
            )

            optimizers.append(
                _get_megatron_optimizer_based_on_param_groups(
                    config=config,
                    model_chunks=model_chunk,
                    param_groups=param_groups,
                    per_model_buffers=buffers,
                    model_parallel_group=mp_group,
                    data_parallel_group=dp_cp_group,
                    data_parallel_group_gloo=intra_dp_cp_group_gloo,
                    data_parallel_group_idx=model_parallel_rank,
                    intra_dist_opt_group=intra_dist_opt_group,
                    distributed_optimizer_instance_id=distributed_optimizer_instance_id,
                )
            )
            model_chunk_offset += 1

        if len(optimizers) == 1:
            return optimizers[0]

        return ChainedOptimizer(optimizers)

    if dump_param_to_param_group_map is not None:
        param_to_param_group = {}
        param_group_id = 0
    for dense_model_chunks, overlap_param_gather_with_optimizer_step in zip(
        all_dense_model_chunks, overlap_param_gather_with_optimizer_step_flags
    ):
        param_groups, buffers = _get_param_groups_and_buffers(
            dense_model_chunks,
            model_chunk_offset=model_chunk_offset,
            config=config,
            config_overrides=config_overrides,
            filter_fn=lambda g: not g['is_expert_parallel'],
            buffer_name='buffers',
        )
        for model_chunk in dense_model_chunks:
            model_chunk.overlap_param_gather_with_optimizer_step = (
                overlap_param_gather_with_optimizer_step
            )
        if dump_param_to_param_group_map is not None:
            for param_group in param_groups:
                for param in param_group["params"]:
                    param_name = get_global_unique_param_name(model_chunks, param)
                    param_to_param_group[param_name] = param_group_id
                param_group_id += 1

        # Pass Gloo process groups into optimizer only if needed.
        optimizers.append(
            _get_megatron_optimizer_based_on_param_groups(
                config=config,
                model_chunks=dense_model_chunks,
                param_groups=param_groups,
                per_model_buffers=buffers,
                model_parallel_group=mp_group,
                data_parallel_group=intra_dp_cp_group,
                data_parallel_group_gloo=intra_dp_cp_group_gloo,
                data_parallel_group_idx=model_parallel_rank,
                intra_dist_opt_group=intra_dist_opt_group,
                distributed_optimizer_instance_id=distributed_optimizer_instance_id,
            )
        )
        model_chunk_offset += 1

    moe_param_groups, moe_buffers = _get_param_groups_and_buffers(
        model_chunks,
        model_chunk_offset=0,
        config=config,
        config_overrides=config_overrides,
        filter_fn=lambda g: g['is_expert_parallel'],
        buffer_name='expert_parallel_buffers',
    )
    if dump_param_to_param_group_map is not None:
        for param_group in moe_param_groups:
            for param in param_group["params"]:
                param_name = get_global_unique_param_name(model_chunks, param)
                param_to_param_group[param_name] = param_group_id
            param_group_id += 1
    if len(moe_param_groups) > 0:
        expt_model_parallel_rank = get_pg_rank(expt_tp_pp_group)
        # Pass Gloo process groups into optimizer only if needed.
        if use_gloo_process_groups:
            expt_data_parallel_group_gloo = intra_expt_dp_group_gloo
        else:
            expt_data_parallel_group_gloo = None
        optimizers.append(
            _get_megatron_optimizer_based_on_param_groups(
                config=config,
                model_chunks=model_chunks,
                param_groups=moe_param_groups,
                per_model_buffers=moe_buffers,
                model_parallel_group=expt_tp_pp_group,
                data_parallel_group=intra_expt_dp_group,
                data_parallel_group_gloo=expt_data_parallel_group_gloo,
                data_parallel_group_idx=expt_model_parallel_rank,
                intra_dist_opt_group=intra_dist_opt_group,
                distributed_optimizer_instance_id=distributed_optimizer_instance_id,
            )
        )

    if dump_param_to_param_group_map is not None:
        torch.distributed.checkpoint.save(
            state_dict=param_to_param_group, checkpoint_id=dump_param_to_param_group_map
        )

    return ChainedOptimizer(optimizers)
