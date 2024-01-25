# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

from apex.optimizers import FusedAdam as Adam
from apex.optimizers import FusedSGD as SGD

from megatron import get_args

from .distrib_optimizer import DistributedOptimizer
from .grad_scaler import ConstantGradScaler, DynamicGradScaler
from .optimizer import (
    Float16OptimizerWithFloat16Params,
    FP32Optimizer,
    ChainedOptimizer,
)


def get_param_groups(model_chunks, no_weight_decay_cond, scale_lr_cond, lr_mult):
    """Create parameter groups for optimizer.

    Creates parameter groups based on weight decay condition (regularized vs
    non regularized), learning rate scale condition (args.lr vs lr_mult * args.lr),
    and whether it is expert parameters. scale_lr_cond is used during finetuning
    where head of the network requires a scaled version of the base learning rate.

    Args:
        model_chunks (List[MegatronModule]): model chunks to create parameter
            groups for.
        no_weight_decay_cond (func): function to determine whether a parameter
            should not perform weight decay.
        scale_lr_cond (func): function to determine whether a parameter
            should have a scaled learning rate.
        lr_mult (float): learning rate multiplier for parameters that
            satisfy scale_lr_cond.
    """
    # map (wd_mult, lr_mult, is_expert_parallel) to params
    params_map = {
        (1.0, 1.0, False): [],
        (1.0, 1.0, True): [],
        (1.0, lr_mult, False): [],
        (1.0, lr_mult, True): [],
        (0.0, 1.0, False): [],
        (0.0, 1.0, True): [],
        (0.0, lr_mult, False): [],
        (0.0, lr_mult, True): [],
    }

    for model_chunk in model_chunks:
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue

            is_expert_parallel = not getattr(param, 'allreduce', True)

            if no_weight_decay_cond is not None:
                no_wd = no_weight_decay_cond(name, param)
            else:
                # do not regularize biases nor Norm parameters
                no_wd = name.endswith(".bias") or len(param.shape) == 1

            if scale_lr_cond is not None:
                scale_lr = scale_lr_cond(name, param)
            else:
                scale_lr = False

            if not no_wd and not scale_lr:
                wd_mult, lr_mult = 1.0, 1.0
            elif not no_wd and scale_lr:
                wd_mult, lr_mult = 1.0, lr_mult
            elif no_wd and not scale_lr:
                wd_mult, lr_mult = 0.0, 1.0
            else:
                wd_mult, lr_mult = 0.0, lr_mult

            params_map[(wd_mult, lr_mult, is_expert_parallel)].append(param)

    param_groups = []
    for (wd_mult, lr_mult, is_expert_parallel), params in params_map.items():
        if len(params) == 0:
            continue
        param_groups.append(
            {
                'params': params,
                'wd_mult': wd_mult,
                'lr_mult': lr_mult,
                'is_expert_parallel': is_expert_parallel,
            }
        )

    return param_groups


def get_megatron_optimizer_based_on_param_groups(param_groups, grad_buffers=None):
    """Get megatron optimizer based on parameter groups.

    For distributed optimizer, we need the parameter gradients to be stored in a
    contiguous grad_buffer.

    Args:
        param_groups (list): list of parameter groups.
        grad_buffers (list, optional): list of gradient buffers. Defaults to None.
    """
    args = get_args()

    if args.optimizer == 'adam':
        optimizer = Adam(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
        )
    elif args.optimizer == 'sgd':
        optimizer = SGD(
            param_groups, lr=args.lr, weight_decay=args.weight_decay, momentum=args.sgd_momentum
        )
    else:
        raise Exception('{} optimizer is not supported.'.format(args.optimizer))

    # Determine whether the params have main-grad field.
    params_have_main_grad = True

    # If it is expert parameters, we do not use the distributed optimizer.
    # TODO: enable support for distributed optimizer with expert parameters
    # (need to support DistOpt across process group with size dp_size / ep_size).
    use_distributed_optimizer = args.use_distributed_optimizer and not any(
        [pg['is_expert_parallel'] for pg in param_groups]
    )

    # Mixed precision optimizer.
    # - Note: both the Float16Optimizer and the DistributedOptimizer inherit
    #   from the MixedPrecisionOptimizer, which manages any optimizer where
    #   the model params and main params are distinct.
    if args.fp16 or args.bf16 or use_distributed_optimizer:

        # Grad scaler:
        #    if loss-scale is provided, instantiate the constant scaler.
        #    if we are using fp16 and loss-scale is not present, use a
        #       dynamic scaler.
        #    otherwise we are running in bf16 with no loss-scale so
        #       leave it as None.
        grad_scaler = None

        # Constant loss scale.
        if args.loss_scale:
            grad_scaler = ConstantGradScaler(args.loss_scale)

        # Dynamic loss scale.
        else:
            if args.fp16:
                grad_scaler = DynamicGradScaler(
                    initial_scale=args.initial_loss_scale,
                    min_scale=args.min_loss_scale,
                    growth_factor=2.0,
                    backoff_factor=0.5,
                    growth_interval=args.loss_scale_window,
                    hysteresis=args.hysteresis,
                )

        optimizer_args = [
            optimizer,
            args.clip_grad,
            args.log_num_zeros_in_grad,
            args.check_for_nan_in_loss_and_grad,
            params_have_main_grad,
            args.fp16,
            args.bf16,
            args.params_dtype,
            grad_scaler,
        ]
        if use_distributed_optimizer:
            optimizer = DistributedOptimizer(*optimizer_args, grad_buffers)
        else:
            optimizer = Float16OptimizerWithFloat16Params(*optimizer_args)

        return optimizer

    # FP32.
    return FP32Optimizer(
        optimizer,
        args.clip_grad,
        args.log_num_zeros_in_grad,
        args.check_for_nan_in_loss_and_grad,
        params_have_main_grad,
    )


def get_megatron_optimizer(
    model_chunks, no_weight_decay_cond=None, scale_lr_cond=None, lr_mult=1.0
):
    """Retrieve the Megatron optimizer for model chunks.

    We use separate optimizers for expert parameters and non-expert parameters.
    
    Args:
        model_chunks (List[MegatronModule]): model chunks to get optimizer for.
        no_weight_decay_cond (func, optional): function to determine whether a parameter
            should not perform weight decay. Defaults to None.
        scale_lr_cond (func, optional): function to determine whether a parameter
            should have a scaled learning rate. Defaults to None.
        lr_mult (float, optional): learning rate multiplier for parameters that
            satisfy scale_lr_cond. Defaults to 1.0.
    """
    # Collect param groups.
    param_groups = get_param_groups(model_chunks, no_weight_decay_cond, scale_lr_cond, lr_mult)

    # Collect grad buffers for distributed optimizer.
    per_model_grad_buffers = {}
    for model_idx, model_chunk in enumerate(model_chunks):
        if hasattr(model_chunk, 'grad_buffers'):
            per_model_grad_buffers[model_idx] = list(model_chunk.grad_buffers.values())

    # Split param groups into dense and moe.
    dense_param_groups = list(filter(lambda g: not g['is_expert_parallel'], param_groups))
    moe_param_groups = list(filter(lambda g: g['is_expert_parallel'], param_groups))

    # Create optimizers.
    optimizers = [
        get_megatron_optimizer_based_on_param_groups(dense_param_groups, per_model_grad_buffers)
    ]
    if len(moe_param_groups):
        optimizers.append(get_megatron_optimizer_based_on_param_groups(moe_param_groups))

    if len(optimizers) == 1:
        return optimizers[0]

    return ChainedOptimizer(optimizers)
