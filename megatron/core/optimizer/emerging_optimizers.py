# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Megatron emerging optimizers (e.g. Muon) factory and wiring."""

import copy
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import log_single_rank

from . import (
    _get_megatron_optimizer_based_on_param_groups,
    _get_param_groups,
    get_standard_config_overrides,
)
from .layer_wise_optimizer import LayerWiseDistributedOptimizer
from .muon import TensorParallelMuon
from .optimizer import (
    ChainedOptimizer,
    Float16OptimizerWithFloat16Params,
    FP32Optimizer,
    MegatronOptimizer,
)
from .optimizer_config import OptimizerConfig, ParamKey, ParamPredicate

logger = logging.getLogger(__name__)

_EMERGING_OPTIMIZERS: Dict[str, type] = {'muon': TensorParallelMuon}


def _muon_init_state_fn(opt, config=None):
    """Initialize Muon optimizer state for torch_dist checkpoint format."""
    for group in opt.param_groups:
        for p in group['params']:
            if len(opt.state[p]) == 0:
                opt.state[p]['momentum_buffer'] = torch.zeros_like(p.data)


def _adam_init_state_fn(opt, config=None):
    """Initialize Adam optimizer state for torch_dist checkpoint format."""
    for group in opt.param_groups:
        for p in group['params']:
            if len(opt.state[p]) == 0:
                if config is None or not config.use_precision_aware_optimizer:
                    opt.state[p]['exp_avg'] = torch.zeros_like(p.data)
                    opt.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)
                else:
                    opt.initialize_state(p)


_INIT_STATE_FNS = {'muon': _muon_init_state_fn, 'adam': _adam_init_state_fn}


def _is_nonlinear_or_embedding(param):
    """True for parameters that should NOT use the emerging optimizer."""
    return getattr(param, 'is_embedding_or_output_parameter', False) or len(param.shape) != 2


def get_megatron_emerging_optimizer(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    config_overrides: Optional[Dict[ParamKey, Any]] = None,
    use_gloo_process_groups: bool = True,
    layer_wise_distributed_optimizer: bool = False,
    pg_collection: Optional[ProcessGroupCollection] = None,
) -> MegatronOptimizer:
    """Get the emerging optimizer for the model chunks.

    Parameter separation (e.g., linear weights -> Muon, rest -> Adam) is expressed as a
    config_override, the same mechanism used for weight-decay and learning-rate overrides.
    Adam/SGD groups are delegated to _get_megatron_optimizer_based_on_param_groups so they
    go through the exact same code path as get_megatron_optimizer.

    Args:
        config (OptimizerConfig): optimizer configuration object.
        model_chunks (List[MegatronModule]): model chunks to get optimizer for.
        config_overrides: optional per-parameter overrides (ParamKey -> ParamGroupOverride).
        use_gloo_process_groups (bool): if false, disable use of Gloo process groups.
        layer_wise_distributed_optimizer (bool): if true, use layer-wise distributed optimizer.
        pg_collection: Optional unified process group for distributed training.
    """
    eopt_name = config.optimizer
    if eopt_name not in _EMERGING_OPTIMIZERS:
        raise ValueError(f"Unsupported emerging optimizer: {eopt_name}")
    if config.use_distributed_optimizer:
        raise Exception('emerging optimizer with dist optimizer is not supported.')
    if config.fp16:
        raise Exception('emerging optimizer with fp16 is not supported.')

    if pg_collection is None:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    pg_dict = ProcessGroupCollection.setup_process_groups_for_optimizer(
        pg_collection, model_chunks, use_gloo_process_groups
    )

    log_single_rank(logger, logging.INFO, f'Setting up emerging optimizer with config {config}')

    # Tag parameters with attributes the emerging optimizer needs.
    model_cfg = model_chunks[0].config
    qkv_split_shapes = [
        model_cfg.num_attention_heads // model_cfg.num_query_groups * model_cfg.kv_channels,
        model_cfg.kv_channels,
        model_cfg.kv_channels,
    ]
    for model_chunk in model_chunks:
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue
            if 'experts' in name and 'shared' not in name:
                param.expert_tp = True
            # TODO(deyuf): support MLA
            if 'linear_qkv.weight' in name and len(param.shape) == 2:
                param.is_qkv = True

    # Non-linear / embedding params fall back to adam; linear 2D weights use the emerging opt.
    if config_overrides is None:
        config_overrides = get_standard_config_overrides(config)
    config_overrides[
        ParamKey(
            predicate=ParamPredicate(name="nonlinear_or_embedding", fn=_is_nonlinear_or_embedding)
        )
    ] = {'optimizer': 'adam'}

    # Build param groups and bucket by (optimizer_name, is_expert_parallel).
    # Layer-wise distributed optimizer handles expert params internally so we skip that split.
    all_param_groups = _get_param_groups(model_chunks, config, config_overrides)
    grouped_param_groups: Dict[Tuple[str, bool], List[dict]] = defaultdict(list)
    for group in all_param_groups:
        opt_name = group.get('optimizer', eopt_name)
        is_expert = group['is_expert_parallel'] and not layer_wise_distributed_optimizer
        grouped_param_groups[(opt_name, is_expert)].append(group)

    # For layer-wise, delay master weight creation -- LayerWiseDistributedOptimizer wraps later.
    reset_config_bf16 = False
    if layer_wise_distributed_optimizer and config.bf16:
        config.bf16 = False
        reset_config_bf16 = True

    # Prepare emerging optimizer kwargs.
    eopt_cls = _EMERGING_OPTIMIZERS[eopt_name]
    eopt_init_state_fn = _INIT_STATE_FNS[eopt_name]
    eopt_kwargs = _config_to_eopt_kwargs(config, eopt_name)
    eopt_kwargs["is_qkv_fn"] = lambda p: getattr(p, "is_qkv", False)
    eopt_kwargs["qkv_split_shapes"] = qkv_split_shapes
    eopt_kwargs['pg_collection'] = pg_collection

    # Create one optimizer per bucket.
    optimizers: List[MegatronOptimizer] = []
    init_fns: List = []

    for (opt_name, is_expert), groups in grouped_param_groups.items():
        if not groups:
            continue

        model_parallel_group = pg_dict['expt_tp_pp_group' if is_expert else 'mp_group']

        if opt_name in _EMERGING_OPTIMIZERS:
            opt = eopt_cls(groups, **eopt_kwargs)
            if config.bf16:
                opt = Float16OptimizerWithFloat16Params(opt, config, None, eopt_init_state_fn)
            else:
                opt = FP32Optimizer(opt, config, eopt_init_state_fn)
            setattr(opt, 'grad_stats_parallel_group', model_parallel_group)
            optimizers.append(opt)
            init_fns.append(eopt_init_state_fn)
        else:
            # Delegate to _get_megatron_optimizer_based_on_param_groups for identical
            # precision / offload / scaler handling as get_megatron_optimizer.
            fallback_config = copy.copy(config)
            fallback_config.optimizer = opt_name
            opt = _get_megatron_optimizer_based_on_param_groups(
                config=fallback_config,
                model_chunks=model_chunks,
                param_groups=groups,
                model_parallel_group=model_parallel_group,
                pg_collection=pg_collection,
            )
            optimizers.append(opt)
            init_fns.append(_INIT_STATE_FNS.get(opt_name, _adam_init_state_fn))

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


def _config_to_eopt_kwargs(config: OptimizerConfig, optimizer_name: str) -> Dict[str, Any]:
    """Convert a Megatron optimizer config to kwargs for the emerging optimizer."""
    if optimizer_name == 'muon':
        return {
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
        raise ValueError(f"Unsupported emerging optimizer: {optimizer_name}")
