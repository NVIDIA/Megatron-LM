# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from apex.optimizers import FusedAdam as Adam

from megatron import get_args
from megatron.model import import_layernorm

from .grad_scaler import ConstantGradScaler, DynamicGradScaler
from .optimizer import FP16OptimizerWithFP16Params, FP32Optimizer


def _get_params_for_weight_decay_optimization(module):
    """Divide params into with-weight-decay and without-weight-decay groups.
    Layernorms and baises will have no weight decay but the rest will.
    """
    args = get_args()
    LayerNorm = import_layernorm(args.fp32_residual_connection)

    weight_decay_params = {'params': []}
    no_weight_decay_params = {'params': [], 'weight_decay': 0.0}
    for module_ in module.modules():
        if isinstance(module_, LayerNorm):
            no_weight_decay_params['params'].extend(
                [p for p in list(module_._parameters.values())
                 if p is not None])
        else:
            weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and n != 'bias'])
            no_weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and n == 'bias'])

    return weight_decay_params, no_weight_decay_params


def get_megatron_optimizer(model):
    args = get_args()

    # Base optimizer.
    param_groups = _get_params_for_weight_decay_optimization(model)
    optimizer = Adam(param_groups,
                     lr=args.lr,
                     weight_decay=args.weight_decay,
                     betas=(args.adam_beta1, args.adam_beta2),
                     eps=args.adam_eps)

    if args.fp16:
        # Constant loss scale.
        if args.loss_scale:
            grad_scaler = ConstantGradScaler(args.loss_scale)
        # Dynamic loss scale.
        else:
            grad_scaler = DynamicGradScaler(
                initial_scale=args.initial_loss_scale,
                min_scale=args.min_loss_scale,
                growth_factor=2.0,
                backoff_factor=0.5,
                growth_interval=args.loss_scale_window,
                hysteresis=args.hysteresis)
        # Megatron optimizer.
        return FP16OptimizerWithFP16Params(optimizer, grad_scaler,
                                           args.clip_grad)

    # FP32.
    return FP32Optimizer(optimizer, args.clip_grad)
