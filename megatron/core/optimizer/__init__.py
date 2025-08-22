# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import logging
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.optim import SGD as CPUSGD
from torch.optim import AdamW as CPUAdam

try:
    from transformer_engine.pytorch.optimizers import FusedAdam as Adam
    from transformer_engine.pytorch.optimizers import FusedSGD as SGD
except ImportError:
    try:
        from apex.optimizers import FusedAdam as Adam
        from apex.optimizers import FusedSGD as SGD
    except ImportError:
        warnings.warn(
            f'Transformer Engine and Apex are not installed. Falling back to Torch optimizers.'
        )

        # Apex's FusedAdam is a drop-in replacement for torch's AdamW.
        # pylint: disable-next=line-too-long.
        # See https://github.com/NVIDIA/apex/blob/7b73b12361068a10b0f44844534613f252a5ea75/apex/optimizers/fused_adam.py#L16.
        from torch.optim import AdamW as Adam, SGD

from megatron.core import parallel_state
from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer
from megatron.core.process_groups_config import GradCommProcessGroups, ModelCommProcessGroups

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
from .optimizer_config import OptimizerConfig

logger = logging.getLogger(__name__)


def _get_param_groups(
    model_chunks: List[MegatronModule],
    no_weight_decay_cond: Optional[Callable],
    scale_lr_cond: Optional[Callable],
    lr_mult: float,
    lr: float,
    min_lr: float,
    decoupled_lr: Optional[float],
    decoupled_min_lr: Optional[float],
    default_skip_embedding_weight_decay: bool = False,
) -> List[Dict]:
    """Create parameter groups for optimizer.

    Creates parameter groups based on weight decay condition (regularized vs
    non regularized), learning rate scale condition (lr vs lr_mult * lr),
    and whether it is expert parameters. scale_lr_cond is used during finetuning
    where head of the network requires a scaled version of the base learning rate.

    Args:
        model_chunks (List[MegatronModule]): model chunks to create parameter
            groups for.
        no_weight_decay_cond (func, optional): function to determine whether a
            parameter should not perform weight decay.
        scale_lr_cond (func, optional): function to determine whether a parameter
            should have a scaled learning rate.
        lr_mult (float): learning rate multiplier for parameters that
            satisfy scale_lr_cond.
        lr (float): learning rate.
        min_lr (float): minimum learning rate.
        decoupled_lr (Optional[float]): optional decoupled learning rate.
        decoupled_min_lr (Optional[float]): optional decoupled minimum learning rate.
        default_skip_embedding_weight_decay (bool): whether to skip weight decay for embedding
            parameters by default, if no_weight_decay_cond is not provided.

    Returns:
        List of parameter groups.
    """

    use_decoupled_learning_rate = decoupled_lr is not None

    # Map (wd_mult, lr_mult, is_expert_parallel, is_decoupled_lr) to params.
    params_map = {}
    for model_chunk in model_chunks:
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue

            is_expert_parallel = not getattr(param, 'allreduce', True)

            if no_weight_decay_cond is not None:
                no_wd: bool = no_weight_decay_cond(name, param)
            else:
                # Do not regularize biases and norm parameters.
                #  optionally, also skip weight decay for embedding parameters if requested
                #  (useful if you do not want embeddings to shrink to zero in training
                #  https://arxiv.org/abs/2312.16903)
                no_wd = (
                    name.endswith(".bias")
                    or len(param.shape) == 1
                    or (default_skip_embedding_weight_decay and "embedding" in name)
                )

            if scale_lr_cond is not None:
                scale_lr = scale_lr_cond(name, param)
            else:
                scale_lr = False

            if not no_wd and not scale_lr:
                wd_mult, _lr_mult = 1.0, 1.0
            elif not no_wd and scale_lr:
                wd_mult, _lr_mult = 1.0, lr_mult
            elif no_wd and not scale_lr:
                wd_mult, _lr_mult = 0.0, 1.0
            else:
                wd_mult, _lr_mult = 0.0, lr_mult

            is_decoupled_lr = False
            # For input/embedding and output layer: embedding.word_embeddings.weight /
            # output_layer.weight.
            if use_decoupled_learning_rate and getattr(
                param, 'is_embedding_or_output_parameter', False
            ):
                is_decoupled_lr = True

            key = (wd_mult, _lr_mult, is_expert_parallel, is_decoupled_lr)
            if key not in params_map:
                params_map[key] = []
            params_map[key].append(param)

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
        wd_mult, _lr_mult, is_expert_parallel, is_decoupled_lr = key
        params = params_map[key] if key in params_map else []
        param_group = {
            'params': params,
            'wd_mult': wd_mult,
            'lr_mult': _lr_mult,
            'is_expert_parallel': is_expert_parallel,
            'is_decoupled_lr': is_decoupled_lr,
        }
        # Ensure param_group has required keys for matching when loading optimizer state
        # See MegatronOptimizer._filter_and_reorder_param_groups.
        assert set(param_group.keys()) - set(param_group_identifier_keys) == {'params'}
        param_groups.append(param_group)

    param_groups = _update_min_and_max_lr_in_param_groups(
        param_groups,
        lr=lr,
        min_lr=min_lr,
        decoupled_lr=decoupled_lr,
        decoupled_min_lr=decoupled_min_lr,
    )

    return param_groups


def _update_min_and_max_lr_in_param_groups(
    param_groups: List[Dict],
    lr: float,
    min_lr: float,
    decoupled_lr: Optional[float],
    decoupled_min_lr: Optional[float],
) -> List[Dict]:
    """
    Updates `max_lr` and `min_lr` values in each parameter group, and returns new list.
    By default, each group will use `lr` / `min_lr` as `max_lr` / `min_lr`.
    If `decoupled_lr` is provided, then `decoupled_lr` / `decoupled_min_lr` will be used
    as `max_lr` / `min_lr` for the input and output layer.

    Args:
        param_groups (List): parameter groups whose 'max_lr' and `min_lr` fields need to
            be adjusted.
        lr (float): learning rate.
        min_lr (float): minimum learning rate.
        decoupled_lr (Optional[float]): optional decoupled learning rate.
        decoupled_min_lr (Optional[float]): optional decoupled minimum learning rate.

    Returns:
        List of adjusted parameter groups.
    """

    if decoupled_min_lr is None:
        decoupled_min_lr = min_lr

    for param_group in param_groups:
        if param_group['is_decoupled_lr']:
            assert decoupled_lr is not None
            param_group['max_lr'] = decoupled_lr
            param_group['min_lr'] = decoupled_min_lr
        else:
            param_group['max_lr'] = lr
            param_group['min_lr'] = min_lr
    return param_groups


def _get_param_groups_and_buffers(
    model_chunks: List[MegatronModule],
    model_chunk_offset: int,
    config: OptimizerConfig,
    no_weight_decay_cond: Optional[Callable],
    scale_lr_cond: Optional[Callable],
    lr_mult: float,
    filter_fn: Callable,
    buffer_name: str,
    default_skip_embedding_weight_decay: bool = False,
) -> Tuple[List[Dict], Dict[int, List[_ParamAndGradBuffer]]]:
    """Returns parameter groups and buffer for optimizer.

    Args:
        model_chunks (List[MegatronModule]): model chunks to create parameter
            groups for.
        model_chunk_offset (int): offset of model_chunks in global model_chunks list.
        config (OptimizerConfig): optimizer configuration object.
        no_weight_decay_cond (func, optional): function to determine whether a
            parameter should not perform weight decay.
        scale_lr_cond (func, optional): function to determine whether a parameter
            should have a scaled learning rate.
        lr_mult (float): learning rate multiplier for parameters that
            satisfy scale_lr_cond.
        lr (float): learning rate.
        min_lr (float): minimum learning rate.
        filter_fn (callable): filtering function for param_groups.
        buffer_name (str): name of buffer.
        default_skip_embedding_weight_decay (bool): whether to skip weight decay for
            embedding parameters by default, if no_weight_decay_cond is not provided.

    Returns:
        List of parameter groups and dictionary of model chunk IDs to buffers.
    """
    param_groups = _get_param_groups(
        model_chunks,
        no_weight_decay_cond,
        scale_lr_cond,
        lr_mult,
        lr=config.lr,
        min_lr=config.min_lr,
        decoupled_lr=config.decoupled_lr,
        decoupled_min_lr=config.decoupled_min_lr,
        default_skip_embedding_weight_decay=default_skip_embedding_weight_decay,
    )
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
    # when freezing sub-models we may have no trainable parameters on a rank and
    # hence an empty param_groups. However, we still need to create an optimizer
    # for the purposes of grad stats reductions
    if param_groups:
        if config.optimizer_cpu_offload:
            if torch.__version__ < '2.3.0':
                warnings.warn(
                    "CPU offload is recommended for PyTorch >= 2.3.0, "
                    "untested versions below this may have convergence issues."
                )
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
                if config.main_params_dtype != torch.float32 or config.fp8_recipe == "delayed":
                    kwargs.update(
                        {
                            "master_weights": True,
                            "use_decoupled_grad": True,
                            "master_weight_dtype": config.main_params_dtype,
                        }
                    )

                if is_te_min_version("2.1.0.dev0"):
                    kwargs.update({"store_param_remainders": config.store_param_remainders})

            optimizer = Adam(**kwargs)

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
            setattr(
                optimizer,
                'grad_stats_parallel_group',
                parallel_state.get_intra_distributed_optimizer_instance_group(),
            )
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
    no_weight_decay_cond: Optional[Callable] = None,
    scale_lr_cond: Optional[Callable] = None,
    lr_mult: float = 1.0,
    use_gloo_process_groups: bool = True,
    default_skip_embedding_weight_decay: bool = False,
    grad_comm_pgs: Optional[GradCommProcessGroups] = None,
    model_comm_pgs: Optional[ModelCommProcessGroups] = None,
) -> MegatronOptimizer:
    """Retrieve the Megatron optimizer for model chunks.

    We use separate optimizers for expert parameters and non-expert parameters.

    Args:
        config (OptimizerConfig): optimizer configuration object.
        model_chunks (List[MegatronModule]): model chunks to get optimizer for.
        no_weight_decay_cond (func, optional): function to determine whether a parameter
            should not perform weight decay. Defaults to None.
        scale_lr_cond (func, optional): function to determine whether a parameter
            should have a scaled learning rate. Defaults to None.
        lr_mult (float, optional): learning rate multiplier for parameters that
            satisfy scale_lr_cond. Defaults to 1.0.
        use_gloo_process_groups (bool): if false, disable use of Gloo process groups
            in underlying Megatron optimizers.
        default_skip_embedding_weight_decay (bool): whether to skip weight decay for
            embedding parameters by default, if no_weight_decay_cond is not provided.
            This is useful if you do not want embeddings to shrink to zero in training
            as recommended in https://arxiv.org/abs/2312.16903
        grad_comm_pgs (Optional[GradCommProcessGroups]): gradient communication process groups.
            If None, uses default parallel_state groups.
        model_comm_pgs (Optional[ModelCommProcessGroups]): model communication process groups.
            If None, uses default parallel_state groups.

    Returns:
        Instance of MegatronOptimizer.
    """

    log_single_rank(logger, logging.INFO, f'Setting up optimizer with config {config}')

    # Separate out first model chunk if overlapping param AG with optimizer step.
    if config.overlap_param_gather_with_optimizer_step:
        all_dense_model_chunks = [[model_chunks[0]], model_chunks[1:]]
        overlap_param_gather_with_optimizer_step_flags = [True, False]
    else:
        all_dense_model_chunks = [model_chunks]
        overlap_param_gather_with_optimizer_step_flags = [False]

    if grad_comm_pgs is None and model_comm_pgs is None:
        # Gradient communication groups
        dp_cp_group = parallel_state.get_data_parallel_group(
            with_context_parallel=True, partial_data_parallel=False
        )
        intra_dp_cp_group = parallel_state.get_data_parallel_group(
            with_context_parallel=True, partial_data_parallel=True
        )

        intra_expt_dp_group = parallel_state.get_expert_data_parallel_group(
            partial_expert_data_parallel=True
        )

        # Gloo groups
        if use_gloo_process_groups:
            intra_dp_cp_group_gloo = parallel_state.get_data_parallel_group_gloo(
                with_context_parallel=True, partial_data_parallel=True
            )
            intra_expt_dp_group_gloo = parallel_state.get_expert_data_parallel_group_gloo(
                partial_expert_data_parallel=True
            )
        else:
            intra_dp_cp_group_gloo = None
            intra_expt_dp_group_gloo = None

        # Model communication groups
        mp_group = parallel_state.get_model_parallel_group()
        expt_tp_pp_group = parallel_state.get_expert_tensor_model_pipeline_parallel_group()
    elif grad_comm_pgs is not None and model_comm_pgs is not None:
        # 1. dp group - this is always required
        if not hasattr(grad_comm_pgs, 'dp'):
            raise ValueError("dp process group is required but not provided in grad_comm_pgs")
        dp_group = grad_comm_pgs.dp

        # 2. dp_cp group:
        # - If provided in grad_comm_pgs, use it
        # - Otherwise check context_parallel_size
        #   - If cp_size is 1, use same as dp
        #   - If cp_size > 1, raise error as dp_cp is needed
        if hasattr(grad_comm_pgs, 'dp_cp'):
            dp_cp_group = grad_comm_pgs.dp_cp
        else:
            model_config = get_model_config(model_chunks[0])
            cp_size = getattr(model_config, 'context_parallel_size', 1)
            if cp_size == 1:
                # If no context parallelism, dp_cp is same as dp
                dp_cp_group = dp_group
            else:
                raise ValueError(
                    "dp_cp process group is required when context_parallel_size > 1 "
                    "but not provided in grad_comm_pgs"
                )

        # 3. Handle expert data parallel group
        assert hasattr(grad_comm_pgs, 'expt_dp'), (
            "expt_dp process group is required but not provided in grad_comm_pgs",
            "please explicitly set it to None if you don't need it",
        )
        expt_dp_group = grad_comm_pgs.expt_dp

        # 4. Handle intra_dp_cp, intra_expt_dp, and inter_dist_opt
        #    based on optimizer instances:
        # Get ddp_config from model chunks to determine optimizer instances
        ddp_config = model_chunks[0].ddp_config
        if ddp_config.num_distributed_optimizer_instances == 1:
            # With a single optimizer instance:
            # - intra_dp_cp is same as dp_cp
            # - intra_expt_dp is same as expt_dp
            # - inter_dist_opt is not needed (set to None)
            intra_dp_cp_group = dp_cp_group
            intra_expt_dp_group = expt_dp_group
        else:
            # With multiple optimizer instances, both groups must be provided
            if not (
                hasattr(grad_comm_pgs, 'intra_dp_cp')
                and hasattr(grad_comm_pgs, 'intra_expt_dp')
                and hasattr(grad_comm_pgs, 'inter_dist_opt')
            ):
                raise ValueError(
                    "intra_dp_cp, intra_expt_dp, and inter_dist_opt "
                    "process groups are required when using multiple optimizer "
                    "instances (>1) but not provided in grad_comm_pgs"
                )
            intra_dp_cp_group = grad_comm_pgs.intra_dp_cp
            intra_expt_dp_group = grad_comm_pgs.intra_expt_dp

        # 5. Model communication groups
        assert hasattr(model_comm_pgs, 'mp'), (
            "mp process group is required but not provided in model_comm_pgs",
            "please explicitly set it to None if you don't need it",
        )
        mp_group = model_comm_pgs.mp

        # Expert tensor-model-pipeline group for MoE
        assert hasattr(model_comm_pgs, 'tp_ep_pp'), (
            "tp_ep_pp process group is required but not provided in model_comm_pgs",
            "please explicitly set it to None if you don't need it",
        )
        expt_tp_pp_group = model_comm_pgs.tp_ep_pp

        # Set up gloo groups - these might not be provided in process groups config
        # so we need to create them or set to None
        assert not use_gloo_process_groups, (
            "Gloo process groups are not supported when grad_comm_pgs and model_comm_pgs are "
            "provided. Please set use_gloo_process_groups to False."
        )
        intra_dp_cp_group_gloo = None
        intra_expt_dp_group_gloo = None

    else:
        raise ValueError("Grad and model comm process groups must be provided or both must be None")

    model_parallel_rank = get_pg_rank(mp_group)

    if get_pg_size(dp_cp_group) > get_pg_size(intra_dp_cp_group):
        if grad_comm_pgs is not None:
            inter_dist_opt_group = grad_comm_pgs.inter_dist_opt
        else:
            inter_dist_opt_group = parallel_state.get_inter_distributed_optimizer_instance_group()
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
                no_weight_decay_cond=no_weight_decay_cond,
                scale_lr_cond=scale_lr_cond,
                lr_mult=lr_mult,
                filter_fn=lambda g: True,
                buffer_name='buffers',
                default_skip_embedding_weight_decay=default_skip_embedding_weight_decay,
            )

            optimizers.append(
                _get_megatron_optimizer_based_on_param_groups(
                    config,
                    model_chunks=model_chunk,
                    param_groups=param_groups,
                    per_model_buffers=buffers,
                    model_parallel_group=mp_group,
                    data_parallel_group=dp_cp_group,
                    data_parallel_group_gloo=intra_dp_cp_group_gloo,
                    data_parallel_group_idx=model_parallel_rank,
                    distributed_optimizer_instance_id=distributed_optimizer_instance_id,
                )
            )
            model_chunk_offset += 1

        if len(optimizers) == 1:
            return optimizers[0]

        return ChainedOptimizer(optimizers)

    for dense_model_chunks, overlap_param_gather_with_optimizer_step in zip(
        all_dense_model_chunks, overlap_param_gather_with_optimizer_step_flags
    ):
        param_groups, buffers = _get_param_groups_and_buffers(
            dense_model_chunks,
            model_chunk_offset=model_chunk_offset,
            config=config,
            no_weight_decay_cond=no_weight_decay_cond,
            scale_lr_cond=scale_lr_cond,
            lr_mult=lr_mult,
            filter_fn=lambda g: not g['is_expert_parallel'],
            buffer_name='buffers',
            default_skip_embedding_weight_decay=default_skip_embedding_weight_decay,
        )
        for model_chunk in dense_model_chunks:
            model_chunk.overlap_param_gather_with_optimizer_step = (
                overlap_param_gather_with_optimizer_step
            )

        # Pass Gloo process groups into optimizer only if needed.
        optimizers.append(
            _get_megatron_optimizer_based_on_param_groups(
                config,
                model_chunks=dense_model_chunks,
                param_groups=param_groups,
                per_model_buffers=buffers,
                model_parallel_group=mp_group,
                data_parallel_group=intra_dp_cp_group,
                data_parallel_group_gloo=intra_dp_cp_group_gloo,
                data_parallel_group_idx=model_parallel_rank,
                distributed_optimizer_instance_id=distributed_optimizer_instance_id,
            )
        )
        model_chunk_offset += 1

    moe_param_groups, moe_buffers = _get_param_groups_and_buffers(
        model_chunks,
        model_chunk_offset=0,
        config=config,
        no_weight_decay_cond=no_weight_decay_cond,
        scale_lr_cond=scale_lr_cond,
        lr_mult=lr_mult,
        filter_fn=lambda g: g['is_expert_parallel'],
        buffer_name='expert_parallel_buffers',
        default_skip_embedding_weight_decay=default_skip_embedding_weight_decay,
    )
    if len(moe_param_groups) > 0:
        expt_model_parallel_rank = get_pg_rank(expt_tp_pp_group)
        # Pass Gloo process groups into optimizer only if needed.
        if use_gloo_process_groups:
            expt_data_parallel_group_gloo = intra_expt_dp_group_gloo
        else:
            expt_data_parallel_group_gloo = None
        optimizers.append(
            _get_megatron_optimizer_based_on_param_groups(
                config,
                model_chunks=model_chunks,
                param_groups=moe_param_groups,
                per_model_buffers=moe_buffers,
                model_parallel_group=expt_tp_pp_group,
                data_parallel_group=intra_expt_dp_group,
                data_parallel_group_gloo=expt_data_parallel_group_gloo,
                data_parallel_group_idx=expt_model_parallel_rank,
                distributed_optimizer_instance_id=distributed_optimizer_instance_id,
            )
        )

    return ChainedOptimizer(optimizers)
