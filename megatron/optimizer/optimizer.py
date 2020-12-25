

from abc import ABC
from abc import abstractmethod

import torch

from apex.multi_tensor_apply import multi_tensor_applier
import amp_C

from megatron import mpu
from megatron import get_args


def get_megatron_optimizer(optimizer):

    args = get_args()

    grad_scaler = DynamicGradScaler(
        initial_scale=2**32,
        min_scale=args.min_scale,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=args.loss_scale_window,
        hysteresis=args.hysteresis)

    megatron_optimizer = FP16OptimizerWithFP16Params(
        optimizer, grad_scaler, args.clip_grad)

    return megatron_optimizer



class MegatronGradScaler(ABC):

    def __init__(self, initial_scale):
        """Initialize scale value with the input initial scale."""
        assert initial_scale > 0.0
        self._scale = torch.cuda.FloatTensor([initial_scale])

    @property
    def scale(self):
        return self._scale

    @property
    def inv_scale(self):
        return self._scale.double().reciprocal().float()

    @abstractmethod
    def update(self, found_inf):
        pass

    '''
    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self, state_dict):
        pass
    '''


class ConstantGradScaler(MegatronGradScaler):
    pass


class DynamicGradScaler(MegatronGradScaler):

    def __init__(self, initial_scale, min_scale,
                 growth_factor, backoff_factor,
                 growth_interval, hysteresis):
        """"Grad scaler with dynamic scale that gets adjusted
        during training."""
        super(DynamicGradScaler, self).__init__(initial_scale)

        # Lower bound on the scale.
        assert min_scale > 0.0
        assert min_scale <= initial_scale
        self.min_scale = torch.cuda.FloatTensor([min_scale])
        # Growth and backoff factors for the scale.
        assert growth_factor > 1.0
        self.growth_factor = torch.cuda.FloatTensor([growth_factor])
        assert backoff_factor < 1.0
        assert backoff_factor > 0.0
        self.backoff_factor = torch.cuda.FloatTensor([backoff_factor])
        # Interval over which if we don't see any inf/nan,
        # we will scale the grad scale by the growth factor.
        assert growth_interval > 0
        self.growth_interval = growth_interval
        # Number of inf/nans we should see before scaling down
        # the grad scale by the backoff factor.
        assert hysteresis > 0
        self.hysteresis = hysteresis

        # Trackers.
        self._growth_tracker = 0
        self._hysteresis_tracker = self.hysteresis


    def update(self, found_inf):

        # If we have an inf/nan, growth tracker is set to 0
        # and hysterisis tracker is reduced by 1.
        if found_inf:
            self._growth_tracker = 0
            self._hysteresis_tracker -= 1
            # Now if we are our of hysteresis count, scale down the loss.
            if self._hysteresis_tracker <= 0:
                self._scale = torch.max(self._scale * self.backoff_factor,
                                        self.min_scale)
        else:
            # If there is no nan/inf, increment the growth tracker.
            self._growth_tracker += 1
            # If we have had enough consequitive intervals with no nan/inf:
            if self._growth_tracker == self.growth_interval:
                # Reset the tracker and hysteresis trackers,
                self._growth_tracker = 0
                self._hysteresis_tracker = self.hysteresis
                # and scale up the loss scale.
                self._scale = self._scale * self.growth_factor



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



class MegatronOptimizer(ABC):

    def __init__(self, optimizer):
        """Input optimizer is the base optimizer for example Adam."""
        self.optimizer = optimizer
        assert self.optimizer, 'no optimizer is provided.'

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
                        master_param.tensor_model_parallel = param.tensor_model_parallel
                        #mpu.copy_tensor_model_parallel_attributes(master_param,
                        #                                          param)
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

        # ==================================================
        # Copy gradients from model params to master params.
        # ==================================================

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

        # ==============================
        # Unscale and check for inf/nan.
        # ==============================

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

        fp32_params = []
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                fp32_params.append(param)
        mpu.clip_grad_norm(fp32_params, self.clip_grad)

        # ===================
        # Step the optimizer.
        # ===================

        self.optimizer.step()

        # =================================
        # Update params from master params.
        # =================================

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

        return True
