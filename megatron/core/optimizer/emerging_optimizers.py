# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Megatron emerging optimizers (e.g. Muon) factory and wiring."""

import logging
from copy import copy
from typing import Any, Dict, List, Optional

import torch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import log_single_rank

from . import _get_param_groups, get_megatron_optimizer
from .layer_wise_optimizer import LayerWiseDistributedOptimizer
from .muon import TensorParallelMuon
from .optimizer import (
    ChainedOptimizer,
    Float16OptimizerWithFloat16Params,
    FP32Optimizer,
    MegatronOptimizer,
)
from .optimizer_config import OptimizerConfig, ParamGroupOverride, ParamKey

logger = logging.getLogger(__name__)


def get_megatron_emerging_optimizer(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    config_overrides: Optional[Dict[ParamKey, ParamGroupOverride]] = None,
    use_gloo_process_groups: bool = True,
    layer_wise_distributed_optimizer: bool = False,
    pg_collection: Optional[ProcessGroupCollection] = None,
) -> MegatronOptimizer:
    """Get the emerging optimizer for the model chunks.

    Args:
        config (OptimizerConfig): optimizer configuration object.
        model_chunks (List[MegatronModule]): model chunks to get optimizer for.
        use_gloo_process_groups (bool): if false, disable use of Gloo process groups
            in underlying Megatron optimizers.
        layer_wise_distributed_optimizer (bool): if true, use layer-wise distributed optimizer.
            Defaults to False.
    """
    eopt_name = copy(config.optimizer)
    if eopt_name == 'muon':
        eopt_cls = TensorParallelMuon
    else:
        raise ValueError(f"Unsupported emerging optimizer: {eopt_name}")

    # FIXME(skyw): Confirm whether we still need this hack with MuonOptimizerConfig
    config.optimizer = 'adam'

    # Dist-opt is not supported due to strong coupling with how DDP init grad buffer
    # In theory we can change DDP to enable use emerging optimizer and dist-opt-adam together
    if config.use_distributed_optimizer:
        raise Exception('emerging optimizer with dist optimizer is not supported.')
    # only support bf16 w/o loss scale now
    if config.fp16:
        raise Exception('emerging optimizer with fp16 is not supported.')

    # before this function receive properly created collection
    if pg_collection is None:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    log_single_rank(logger, logging.INFO, f'Setting up emerging optimizer with config {config}')

    # Needed for torch_dist ckpt_format, unlike torch ckpt_format
    # For other emerging optimizers, need to implement init_state_fn as well
    # TODO(boxiangw): Improve usability after optimizer refactor
    # TODO(boxiangw): support precision aware optimizer
    def eopt_init_state_fn(opt, config=None):
        for group in opt.param_groups:
            for p in group['params']:
                if len(opt.state[p]) == 0:
                    opt.state[p]['momentum_buffer'] = torch.zeros_like(p.data)

    def adam_init_state_fn(opt, config=None):
        for group in opt.param_groups:
            for p in group['params']:
                if len(opt.state[p]) == 0:
                    if config is None or not config.use_precision_aware_optimizer:
                        opt.state[p]['exp_avg'] = torch.zeros_like(p.data)
                        opt.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)
                    else:
                        opt.initialize_state(p)

    optimizers = []
    # record list of non/linear params
    linear_params = []
    nonlinear_params = []
    for model_chunk in model_chunks:
        # use config to determine qkv split shapes.
        # no need to check tp since tp splits by head and this is per head(group) dimension
        num_attention_heads = model_chunk.config.num_attention_heads
        num_query_groups = model_chunk.config.num_query_groups
        kv_channels = model_chunk.config.kv_channels
        qkv_split_shapes = [
            num_attention_heads // num_query_groups * kv_channels,
            kv_channels,
            kv_channels,
        ]
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue
            # add flag for expert weight so optimizer can figure which tp group it uses
            # alternatively, create new param group and save tp_group. this require more
            # change in optimizer
            if 'experts' in name and 'shared' not in name:
                param.expert_tp = True
            # add flag for qkv parameter
            # TODO(deyuf): support MLA
            if 'linear_qkv.weight' in name and len(param.shape) == 2:
                param.is_qkv = True
            # TODO(deyuf): currently only allow 2D non-embedding weight to avoid breaking
            if (
                not getattr(param, 'is_embedding_or_output_parameter', False)
                and len(param.shape) == 2
            ):
                linear_params.append(param)
            else:
                nonlinear_params.append(param)

    eopt_kwargs = _config_to_eopt_kwargs(config)
    if eopt_kwargs.get("split_qkv", False):
        eopt_kwargs["is_qkv_fn"] = lambda p: getattr(p, "is_qkv", False)
        eopt_kwargs["qkv_split_shapes"] = qkv_split_shapes
    eopt_kwargs['pg_collection'] = pg_collection

    # freezing nonlinear params and get param groups for emerging optimizer
    for param in nonlinear_params:
        param.requires_grad = False

    linear_param_groups = _get_param_groups(model_chunks, config, config_overrides)
    # if layerwise distributed optimizer is not used, need to handle ep params separately
    expert_param_groups = []
    if not layer_wise_distributed_optimizer:
        for group in linear_param_groups:
            if group['is_expert_parallel']:
                expert_param_groups.append(group)
                linear_param_groups.remove(group)

    optimizer = eopt_cls(linear_param_groups, **eopt_kwargs)

    reset_config_bf16 = False
    if config.bf16:
        if layer_wise_distributed_optimizer:
            # creating master weight before layerwise sharding will lead to unnecessary master
            # weight so here we delay master weight creation into layer_wise unset config.bf16
            # will also result in all optimizers below(adam) to also not be wrapped
            config.bf16 = False
            reset_config_bf16 = True
        else:
            # if not using layer_wise wrapper, just create master weight here is fine
            optimizer = Float16OptimizerWithFloat16Params(
                optimizer, config, None, eopt_init_state_fn
            )
    else:
        optimizer = FP32Optimizer(optimizer, config, eopt_init_state_fn)

    optimizers.append(optimizer)

    # expert optimizer exists meaning layerwise distributed optimizer is not used
    if len(expert_param_groups) > 0:
        expert_optimizer = eopt_cls(expert_param_groups, **eopt_kwargs)
        if config.bf16:
            expert_optimizer = Float16OptimizerWithFloat16Params(
                expert_optimizer, config, None, eopt_init_state_fn
            )
        else:
            expert_optimizer = FP32Optimizer(expert_optimizer, config, eopt_init_state_fn)
        setattr(expert_optimizer, 'grad_stats_parallel_group', pg_collection.tp_ep_pp)
        optimizers.append(expert_optimizer)

    # done with emerging optimizer, unfreeze nonlinear and freeze linear
    for param in nonlinear_params:
        param.requires_grad = True
    for param in linear_params:
        param.requires_grad = False

    # call original get. linear params will be skipped since they're freezed
    chained_adam = get_megatron_optimizer(
        config,
        model_chunks,
        config_overrides=config_overrides,
        use_gloo_process_groups=use_gloo_process_groups,
    )

    # unfreeze everything
    for param in linear_params:
        param.requires_grad = True

    # chain everything together
    init_fns = [eopt_init_state_fn] + len(chained_adam.chained_optimizers) * [adam_init_state_fn]
    optimizers += chained_adam.chained_optimizers

    if layer_wise_distributed_optimizer:
        log_single_rank(
            logger, logging.INFO, f'Using LayerWiseDistributedOptimizer for {eopt_name}'
        )
        if reset_config_bf16:
            config.bf16 = True
        return LayerWiseDistributedOptimizer(
            optimizers, config, pg_collection, init_state_fn_list=init_fns
        )
    return ChainedOptimizer(optimizers)


def _config_to_eopt_kwargs(config: OptimizerConfig) -> Dict[str, Any]:
    """Convert a Megatron optimizer config to a dictionary of kwargs for the emerging optimizer.

    Note:
        Ideally this can be a method of OptimizerConfig class, which would require a naming
        convention being enforced. Currently because optimizer implementation is coming from
        EmergingOptimizers, we don't have any mechanism to enforce same naming convention between
        MegatronCore and EmergingOptimizers. Before a reliable syncing mechanism is in place,
        we'll have this dispatch function.

    Args:
        config (OptimizerConfig): the Megatron optimizer config.

    Returns:
        Dict[str, Any]: a dictionary of kwargs for the emerging optimizer.
    """
    if config.optimizer == 'muon':
        eopt_kwargs = {
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "momentum_beta": config.muon_momentum,
            "use_nesterov": config.muon_use_nesterov,
            "fp32_matmul_prec": config.muon_fp32_matmul_prec,
            "num_ns_steps": config.muon_num_ns_steps,
            "scale_mode": config.muon_scale_mode,
            "extra_scale_factor": config.muon_extra_scale_factor,
            "mode": config.muon_tp_mode,
            "split_qkv": config.muon_split_qkv,
        }
    else:
        raise ValueError(f"Unsupported emerging optimizer: {config.optimizer}")
    return eopt_kwargs
