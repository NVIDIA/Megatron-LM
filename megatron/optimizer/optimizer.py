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

"""Megatron optimizer."""

from abc import ABC
from abc import abstractmethod

import torch
from torch._six import inf

from apex.multi_tensor_apply import multi_tensor_applier
import amp_C

from megatron import get_timers
from megatron import mpu


def _zero_grad_group_helper(group, set_to_none):
    """Zero out the gradient for a group of parameters.
    Note: copied from torch.optim.optimizer."""
    for param in group:
        if param.grad is not None:
            if set_to_none:
                param.grad = None
            else:
                if param.grad.grad_fn is not None:
                    param.grad.detach_()
                else:
                    param.grad.requires_grad_(False)
                param.grad.zero_()


def _clip_grad_norm(parameters, max_norm, norm_type=2):
    """Clips gradient norm of an iterable of parameters.

    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters. Note that
    the gradients are modified in place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    # Filter parameters based on:
    #   - grad should not be none
    #   - parameter should not be shared
    #   - should not be a replica due to tensor model parallelism
    filtered_parameters = []
    for param in parameters:
        grad_not_none = param.grad is not None
        is_not_shared = not hasattr(param, 'shared') or not param.shared
        is_not_tp_duplicate = param.tensor_model_parallel or \
                              (mpu.get_tensor_model_parallel_rank() == 0)
        if grad_not_none and is_not_shared and is_not_tp_duplicate:
            filtered_parameters.append(param)
    parameters = filtered_parameters

    # Norm parameters.
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    total_norm = 0.0

    # Calculate norm.
    if norm_type == inf:
        total_norm = max(param.grad.detach().abs().max()
                         for param in parameters)
        total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])
        # Take max across all model-parallel GPUs.
        torch.distributed.all_reduce(total_norm_cuda,
                                     op=torch.distributed.ReduceOp.MAX,
                                     group=mpu.get_model_parallel_group())
        total_norm = total_norm_cuda[0].item()

    else:
        for param in parameters:
            param_norm = torch.norm(param.grad.detach(), norm_type)
            total_norm += param_norm.item() ** norm_type
        # Sum across all model-parallel GPUs.
        total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])
        torch.distributed.all_reduce(total_norm_cuda,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=mpu.get_model_parallel_group())
        total_norm = total_norm_cuda[0].item() ** (1. / norm_type)

    # Scale.
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for param in parameters:
            param.grad.detach().mul_(clip_coef)

    return total_norm



class MegatronOptimizer(ABC):

    def __init__(self, optimizer):
        """Input optimizer is the base optimizer for example Adam."""
        self.optimizer = optimizer
        assert self.optimizer, 'no optimizer is provided.'

    def clip_grad_norm(self, clip_grad):
        params = []
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                params.append(param)
        _clip_grad_norm(params, clip_grad)

    @abstractmethod
    def zero_grad(self, set_to_none=True):
        pass

    @abstractmethod
    def get_loss_scale(self):
        pass

    def scale_loss(self, loss):
        """Simple scaling."""
        return self.get_loss_scale() * loss

    @abstractmethod
    def step(self):
        pass

    '''
    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self, state_dict):
        pass
    '''

    # Promote state so it can be retrieved or set via
    # "optimizer_instance.state"
    def _get_state(self):
        return self.optimizer.state

    def _set_state(self, value):
        self.optimizer.state = value

    state = property(_get_state, _set_state)

    # Promote param_groups so it can be retrieved or set via
    # "optimizer_instance.param_groups"
    # (for example, to adjust the learning rate)
    def _get_param_groups(self):
        return self.optimizer.param_groups

    def _set_param_groups(self, value):
        self.optimizer.param_groups = value

    param_groups = property(_get_param_groups, _set_param_groups)



class FP16OptimizerWithFP16Params(MegatronOptimizer):


    def __init__(self, optimizer, grad_scaler, clip_grad):
        super(FP16OptimizerWithFP16Params, self).__init__(optimizer)

        self.grad_scaler = grad_scaler
        self.clip_grad = clip_grad

        # Tensor used to determine if a nan/if has happend.
        # Any non-zero value indicates inf/nan.
        self.found_inf = torch.cuda.FloatTensor([0.0])

        # Dummy tensor needed for apex multi-apply tensor.
        self._dummy_overflow_buf = torch.cuda.IntTensor([0])

        # ======================
        # master parameter stuff
        # ======================

        # Three groups of parameters:
        #   fp16_groups: original fp16 parameters
        #   fp32_from_fp16_groups: fp32 copy of fp16 parameters
        #   fp32_from_fp32_groups: original fp32 parameters
        self.fp16_groups = []
        self.fp32_from_fp16_groups = []
        self.fp32_from_fp32_groups = []

        # For all the groups in the original optimizer:
        for param_group in self.optimizer.param_groups:
            fp16_params_this_group = []
            fp32_params_this_group = []
            fp32_from_fp16_params_this_group = []
            # For all the parameters in this group:
            for i, param in enumerate(param_group['params']):
                if param.requires_grad:

                    # fp16 params:
                    if param.type() == 'torch.cuda.HalfTensor':
                        fp16_params_this_group.append(param)
                        # Create a copy
                        master_param = param.detach().clone().float()
                        # Store grads
                        master_param.requires_grad = True
                        # Copy tensor model parallel attributes.
                        mpu.copy_tensor_model_parallel_attributes(master_param,
                                                                  param)
                        if hasattr(param, 'shared'):
                            master_param.shared = param.shared
                        # Replace the optimizer params with the new fp32 copy.
                        param_group['params'][i] = master_param
                        fp32_from_fp16_params_this_group.append(master_param)
                        # Reset existing state dict key to the new master param.
                        if param in self.optimizer.state:
                            self.optimizer.state[master_param] \
                                = self.optimizer.state.pop(param)

                    # fp32 params.
                    elif param.type() == 'torch.cuda.FloatTensor':
                        fp32_params_this_group.append(param)
                        param_group['params'][i] = param

                    else:
                        raise TypeError("Wrapped parameters must be either "
                                        "torch.cuda.FloatTensor or "
                                        "torch.cuda.HalfTensor. "
                                        "Received {}".format(param.type()))

            self.fp16_groups.append(fp16_params_this_group)
            self.fp32_from_fp16_groups.append(fp32_from_fp16_params_this_group)
            self.fp32_from_fp32_groups.append(fp32_params_this_group)

        # Leverage state_dict() and load_state_dict() to
        # recast preexisting per-param state tensors
        self.optimizer.load_state_dict(self.optimizer.state_dict())


    def zero_grad(self, set_to_none=True):
        """We only need to zero the model related parameters, i.e.,
                fp16_groups & fp32_from_fp32_groups."""
        for group in self.fp16_groups:
            _zero_grad_group_helper(group, set_to_none)
        for group in self.fp32_from_fp32_groups:
            _zero_grad_group_helper(group, set_to_none)


    def get_loss_scale(self):
        return self.grad_scaler.scale


    @torch.no_grad()
    def step(self):

        timers = get_timers()

        # ==================================================
        # Copy gradients from model params to master params.
        # ==================================================

        timers('optimizer-copy-to-master-grad').start()
        # This only needs to be done for the fp16 group.
        model_grads = []
        master_grads = []
        for model_group, master_group in zip(self.fp16_groups,
                                             self.fp32_from_fp16_groups):
            for model_param, master_param in zip(model_group, master_group):
                if model_param.grad is not None:
                    if master_param.grad is None:
                        master_param.grad = torch.empty_like(master_param)
                    model_grads.append(model_param.grad)
                    master_grads.append(master_param.grad)
        self._dummy_overflow_buf.fill_(0)
        # Scaling with factor `1.0` is equivalent to copy.
        multi_tensor_applier(amp_C.multi_tensor_scale,
                             self._dummy_overflow_buf,
                             [model_grads, master_grads],
                             1.0)
        timers('optimizer-copy-to-master-grad').stop()

        # ==============================
        # Unscale and check for inf/nan.
        # ==============================

        timers('optimizer-unscale-and-check-inf').start()
        # Append fp32 parameters.
        for master_group in self.fp32_from_fp32_groups:
            for master_param in master_group:
                if master_param.grad is not None:
                    master_grads.append(master_param.grad)
        # Reset found inf.
        self.found_inf.fill_(0.0)
        # Unscale and set found inf/nan
        torch._amp_foreach_non_finite_check_and_unscale_(
            master_grads, self.found_inf, self.grad_scaler.inv_scale)
        # Update across all model parallel instances.
        torch.distributed.all_reduce(self.found_inf,
                                     op=torch.distributed.ReduceOp.MAX,
                                     group=mpu.get_model_parallel_group())
        timers('optimizer-unscale-and-check-inf').stop()

        # ==================================
        # We are done with scaling gradients
        # so we can update the loss scale.
        # ==================================
        found_inf_flag = (self.found_inf.item() > 0)
        self.grad_scaler.update(found_inf_flag)

        # =====================================
        # If we found inf/nan, skip the update.
        # =====================================
        if found_inf_flag:
            return False

        # ==========================
        # Clip the master gradients.
        # ==========================

        timers('optimizer-clip-master-grad').start()
        self.clip_grad_norm(self.clip_grad)
        timers('optimizer-clip-master-grad').stop()

        # ===================
        # Step the optimizer.
        # ===================

        self.optimizer.step()

        # =================================
        # Update params from master params.
        # =================================

        timers('optimizer-copy-master-to-model-params').start()
        # Only needed for the fp16 params.
        model_data = []
        master_data = []
        for model_group, master_group in zip(self.fp16_groups,
                                             self.fp32_from_fp16_groups):
            for model_param, master_param in zip(model_group, master_group):
                model_data.append(model_param.data)
                master_data.append(master_param.data)
        self._dummy_overflow_buf.fill_(0)
        # Scaling with factor `1.0` is equivalent to copy.
        multi_tensor_applier(amp_C.multi_tensor_scale,
                             self._dummy_overflow_buf,
                             [master_data, model_data],
                             1.0)
        timers('optimizer-copy-master-to-model-params').stop()

        return True


class FP32Optimizer(MegatronOptimizer):

    def __init__(self, optimizer, model, clip_grad):

        super(FP32Optimizer, self).__init__(optimizer)
        self.model = model
        self.clip_grad = clip_grad
        self._scale = torch.cuda.FloatTensor([1.0])


    def zero_grad(self, set_to_none=True):
        """Copied from torch.optim.optimizer"""
        for group in self.optimizer.param_groups:
            _zero_grad_group_helper(group['params'], set_to_none)


    def get_loss_scale(self):
        """FP32 optimizer does not do any scaling."""
        return self._scale


    @torch.no_grad()
    def step(self):
        """Clip gradients (if needed) and step the base optimizer.
        Always return auccessful since there is no overflow."""

        # Clip gradients.
        if self.clip_grad > 0.0:
            self.clip_grad_norm(self.clip_grad)

        # Update parameters.
        self.optimizer.step()

        # No overflow for FP32 optimizer.
        return True


    def state_dict(self):
        return self.optimizer.state_dict()


    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
