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

from apex.multi_tensor_apply import multi_tensor_applier
import amp_C

from megatron import get_timers
from megatron import mpu
from megatron import print_rank_0

from .clip_grads import clip_grad_norm_fp32


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


def _multi_tensor_copy_this_to_that(this, that, overflow_buf=None):
    """Use multi-tensor-applier to copy values from one list to another."""
    if overflow_buf:
        overflow_buf.fill_(0)
    else:
        overflow_buf = torch.cuda.IntTensor([0])
    # Scaling with factor `1.0` is equivalent to copy.
    multi_tensor_applier(amp_C.multi_tensor_scale,
                         overflow_buf,
                         [this, that],
                         1.0)


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
        clip_grad_norm_fp32(params, clip_grad)

    @abstractmethod
    def zero_grad(self, set_to_none=True):
        pass

    @abstractmethod
    def get_loss_scale(self):
        """The output should be a cuda tensor of size 1."""
        pass

    def scale_loss(self, loss):
        """Simple scaling."""
        return self.get_loss_scale() * loss

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def reload_model_params(self):
        """Refreshes any internal state from the current model parameters.
        Call whenever the parameters are changed outside of the optimizer.
        For example, when we load a model from a checkpoint  without loading
        the optimizer, the model parameters are updated but for fp16 optimizer
        with main parameters, the main parameters need to also be updated."""
        pass

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self, state_dict):
        pass

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
        # main parameter stuff
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
                        main_param = param.detach().clone().float()
                        # Store grads
                        main_param.requires_grad = True
                        # Copy tensor model parallel attributes.
                        mpu.copy_tensor_model_parallel_attributes(main_param,
                                                                  param)
                        if hasattr(param, 'shared'):
                            main_param.shared = param.shared
                        # Replace the optimizer params with the new fp32 copy.
                        param_group['params'][i] = main_param
                        fp32_from_fp16_params_this_group.append(main_param)
                        # Reset existing state dict key to the new main param.
                        if param in self.optimizer.state:
                            self.optimizer.state[main_param] \
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


    def _copy_model_grads_to_main_grads(self):
        # This only needs to be done for the fp16 group.
        model_grads = []
        main_grads = []
        for model_group, main_group in zip(self.fp16_groups,
                                           self.fp32_from_fp16_groups):
            for model_param, main_param in zip(model_group, main_group):
                if model_param.grad is not None:
                    if main_param.grad is None:
                        main_param.grad = torch.empty_like(main_param)
                    model_grads.append(model_param.grad.data)
                    main_grads.append(main_param.grad.data)
        _multi_tensor_copy_this_to_that(this=model_grads, that=main_grads,
                                        overflow_buf=self._dummy_overflow_buf)


    def _unscale_main_grads_and_check_for_nan(self):
        main_grads = []
        # fp32 params fromm fp16 ones.
        for main_group in self.fp32_from_fp16_groups:
            for main_param in main_group:
                if main_param.grad is not None:
                    main_grads.append(main_param.grad.data)
        # Append fp32 parameters.
        for main_group in self.fp32_from_fp32_groups:
            for main_param in main_group:
                if main_param.grad is not None:
                    main_grads.append(main_param.grad.data)
        # Reset found inf.
        self.found_inf.fill_(0.0)
        # Unscale and set found inf/nan
        torch._amp_foreach_non_finite_check_and_unscale_(
            main_grads, self.found_inf, self.grad_scaler.inv_scale)
        # Update across all model parallel instances.
        torch.distributed.all_reduce(self.found_inf,
                                     op=torch.distributed.ReduceOp.MAX,
                                     group=mpu.get_model_parallel_group())

        # Check for nan.
        found_inf_flag = (self.found_inf.item() > 0)
        return found_inf_flag


    def _get_model_and_main_params_data_fp16(self):
        model_data = []
        main_data = []
        for model_group, main_group in zip(self.fp16_groups,
                                           self.fp32_from_fp16_groups):
            for model_param, main_param in zip(model_group, main_group):
                model_data.append(model_param.data)
                main_data.append(main_param.data)
        return model_data, main_data


    def _copy_main_params_to_model_params(self):
        # Only needed for the fp16 params.
        model_data, main_data = self._get_model_and_main_params_data_fp16()
        _multi_tensor_copy_this_to_that(this=main_data, that=model_data,
                                        overflow_buf=self._dummy_overflow_buf)


    def _copy_model_params_to_main_params(self):
        # Only needed for the fp16 params.
        model_data, main_data = self._get_model_and_main_params_data_fp16()
        _multi_tensor_copy_this_to_that(this=model_data, that=main_data,
                                        overflow_buf=self._dummy_overflow_buf)


    def reload_model_params(self):
        self._copy_model_params_to_main_params()


    @torch.no_grad()
    def step(self):

        timers = get_timers()

        # Copy gradients from model params to main params.
        timers('optimizer-copy-to-main-grad').start()
        self._copy_model_grads_to_main_grads()
        timers('optimizer-copy-to-main-grad').stop()

        # Unscale and check for inf/nan.
        timers('optimizer-unscale-and-check-inf').start()
        found_inf_flag = self._unscale_main_grads_and_check_for_nan()
        timers('optimizer-unscale-and-check-inf').stop()

        # We are done with scaling gradients
        # so we can update the loss scale.
        self.grad_scaler.update(found_inf_flag)

        # If we found inf/nan, skip the update.
        if found_inf_flag:
            return False

        # Clip the main gradients.
        timers('optimizer-clip-main-grad').start()
        self.clip_grad_norm(self.clip_grad)
        timers('optimizer-clip-main-grad').stop()

        # Step the optimizer.
        self.optimizer.step()

        # Update params from main params.
        timers('optimizer-copy-main-to-model-params').start()
        self._copy_main_params_to_model_params()
        timers('optimizer-copy-main-to-model-params').stop()

        # Successful update.
        return True


    def state_dict(self):
        state_dict = {}
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['grad_scaler'] = self.grad_scaler.state_dict()
        state_dict['fp32_from_fp16_params'] = self.fp32_from_fp16_groups
        return state_dict


    def load_state_dict(self, state_dict):
        # Optimizer.
        optimizer_key = 'optimizer'
        if optimizer_key not in state_dict:
            optimizer_key = 'optimizer_state_dict'
            print_rank_0('***WARNING*** loading optimizer from '
                         'an old checkpoint ...')
        self.optimizer.load_state_dict(state_dict[optimizer_key])

        # Grad scaler.
        if 'grad_scaler' not in state_dict:
            print_rank_0('***WARNING*** found an old checkpoint, will not '
                         'load grad scaler ...')
        else:
            self.grad_scaler.load_state_dict(state_dict['grad_scaler'])

        # Copy data for the main params.
        fp32_from_fp16_params_key = 'fp32_from_fp16_params'
        if fp32_from_fp16_params_key not in state_dict:
            fp32_from_fp16_params_key = 'fp32_from_fp16'
        for current_group, saved_group in zip(
                self.fp32_from_fp16_groups,
                state_dict[fp32_from_fp16_params_key]):
            for current_param, saved_param in zip(current_group, saved_group):
                current_param.data.copy_(saved_param.data)



class FP32Optimizer(MegatronOptimizer):

    def __init__(self, optimizer, clip_grad):

        super(FP32Optimizer, self).__init__(optimizer)
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
        Always return successful since there is no overflow."""

        # Clip gradients.
        if self.clip_grad > 0.0:
            self.clip_grad_norm(self.clip_grad)

        # Update parameters.
        self.optimizer.step()

        # No overflow for FP32 optimizer.
        return True


    def reload_model_params(self):
        pass


    def state_dict(self):
        return self.optimizer.state_dict()


    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
