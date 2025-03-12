# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Megatron optimizer."""

import copy
import math
import warnings
from abc import ABC, abstractmethod
from itertools import chain
from logging import getLogger
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

try:
    from transformer_engine.pytorch.optimizers import multi_tensor_applier, multi_tensor_scale

    multi_tensor_scale_impl = multi_tensor_scale
except ImportError:
    try:
        import amp_C
        from apex.multi_tensor_apply import multi_tensor_applier

        multi_tensor_scale_impl = amp_C.multi_tensor_scale
    except ImportError:
        import warnings

        warnings.warn(
            'Transformer Engine and Apex are not installed. '
            'Falling back to local implementations of '
            'multi_tensor_applier and multi_tensor_scale'
        )

        from megatron.core.utils import local_multi_tensor_applier, local_multi_tensor_scale

        multi_tensor_applier = local_multi_tensor_applier
        multi_tensor_scale_impl = local_multi_tensor_scale

from .. import parallel_state, tensor_parallel
from ..config_logger import has_config_logger_enabled, log_config_to_disk
from ..dist_checkpointing.mapping import ShardedStateDict
from ..dist_checkpointing.optimizer import (
    get_param_id_to_sharded_param_map,
    make_sharded_optimizer_tensor,
    optim_state_to_sharding_state,
)
from ..dist_checkpointing.utils import add_prefix_for_sharding
from ..transformer.module import param_is_not_shared
from .clip_grads import clip_grad_by_total_norm_fp32, count_zeros_fp32, get_grad_norm_fp32
from .grad_scaler import MegatronGradScaler
from .optimizer_config import OptimizerConfig

logger = getLogger(__name__)


def _zero_grad_group_helper(
    group: List[torch.nn.Parameter], set_to_none: bool, use_decoupled_grad: bool = False
):
    """
    Zero out the gradient for a group of parameters.
    Note: copied from torch.optim.optimizer.
    """
    for param in group:
        grad_attr = "decoupled_grad" if use_decoupled_grad else "grad"
        if hasattr(param, grad_attr) and getattr(param, grad_attr) is not None:
            if set_to_none:
                setattr(param, grad_attr, None)
            else:
                grad_obj = getattr(param, grad_attr)
                if grad_obj.grad_fn is not None:
                    grad_obj.detach_()
                else:
                    grad_obj.requires_grad_(False)
                grad_obj.zero_()


def _multi_tensor_copy_this_to_that(
    this: List[torch.Tensor], that: List[torch.Tensor], overflow_buf: Optional[torch.Tensor] = None
):
    """
    Use multi-tensor-applier to copy values from one list to another.
    We don't have a bfloat16 implementation so for now if the overflow_buf
    is not provided, we default back to simple loop copy to be compatible
    with bfloat16.
    """
    if overflow_buf is not None:
        overflow_buf.fill_(0)
        # Scaling with factor `1.0` is equivalent to copy.
        multi_tensor_applier(multi_tensor_scale_impl, overflow_buf, [this, that], 1.0)
    else:
        for this_, that_ in zip(this, that):
            that_.copy_(this_)


class MegatronOptimizer(ABC):
    """
    Base class for all Megatron optimizers.

    Args:
        optimizer (torch.optim.Optimizer): base optimizer such as Adam or SGD.
        config (OptimizerConfig): configuration object for optimizer.
        init_state_fn (Callable, optional): function to initialize state in the optimizer.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: OptimizerConfig,
        init_state_fn: Callable = lambda x: None,
    ):
        """Input optimizer is the base optimizer (e.g., Adam)."""
        self.optimizer = optimizer
        if self.optimizer is None:
            warnings.warn(
                f"WARNING: there is no optimizer on RANK {torch.distributed.get_rank()}. "
                "This may be expected if you have frozen sub-models."
            )
        self.config = config
        self.init_state_fn = init_state_fn

    def get_parameters(self) -> List[torch.nn.Parameter]:
        """
        Get list of parameters wrapped in optimizer.
        """
        params = []
        if hasattr(self.optimizer, 'param_groups'):
            for param_group in self.optimizer.param_groups:
                for param in param_group['params']:
                    params.append(param)
        return params

    def get_main_grads_for_grad_norm(self) -> List[torch.Tensor]:
        """
        Get main_grads that should be taken into account to compute the grad norm.
        Filter parameters based on:
          - grad should not be None.
          - parameter should not be shared (i.e., grads shouldn't be double counted while
            computing norms).
          - should not be a replica due to tensor model parallelism.
        """
        params = self.get_parameters()
        grads_for_norm = []
        for param in params:
            if self.config.use_precision_aware_optimizer:
                grad = param.decoupled_grad if hasattr(param, "decoupled_grad") else None
            else:
                grad = param.grad
            grad_not_none = grad is not None
            is_not_shared = param_is_not_shared(param)
            is_not_tp_duplicate = tensor_parallel.param_is_not_tensor_parallel_duplicate(param)
            if grad_not_none and is_not_shared and is_not_tp_duplicate:
                grads_for_norm.append(grad)

        return grads_for_norm

    def get_grad_stats_parallel_group(self) -> torch.distributed.ProcessGroup:
        """Process group for reducing gradient statistics (num_zeros & norm).

        The two most common cases are:
        - Non-distributed optimizer (default): Return the model-parallel group.
        - Distributed optimizer (overridden in distrib_optimizer.py): Return the entire world.
        """
        if hasattr(self, 'model_parallel_group'):
            warnings.warn(
                "WARNING: `optimizer.model_parallel_group` deprecated and renamed to "
                "`optimizer.grad_stats_parallel_group`. The previous name will be "
                "removed in a future release."
            )
            self.grad_stats_parallel_group = self.model_parallel_group
            delattr(self, "model_parallel_group")
            return self.grad_stats_parallel_group
        if hasattr(self, 'grad_stats_parallel_group'):
            return self.grad_stats_parallel_group
        return parallel_state.get_model_parallel_group()

    @abstractmethod
    def prepare_grads(self) -> bool:
        """Pre-processing gradients before the optimizer step, returns whether inf/nan is found."""
        return False

    @abstractmethod
    def step_with_ready_grads(self) -> bool:
        """Step the optimizer with ready gradients, return successful."""
        return True

    @torch.no_grad()
    def get_grad_norm(self):
        """Compute and return grad norm."""
        grads_for_norm = self.get_main_grads_for_grad_norm()
        total_norm = get_grad_norm_fp32(
            grads_for_norm, grad_stats_parallel_group=self.get_grad_stats_parallel_group()
        )
        return total_norm

    def clip_grad_norm(self, clip_grad: float) -> float:
        """Compute and return grad norm, also clip grads."""
        params = self.get_parameters()
        if params:
            grads_for_norm = self.get_main_grads_for_grad_norm()
        else:
            grads_for_norm = []
        grad_norm = get_grad_norm_fp32(
            grads_for_norm, grad_stats_parallel_group=self.get_grad_stats_parallel_group()
        )

        if params:
            clip_grad_by_total_norm_fp32(
                params, clip_grad, grad_norm, self.config.use_precision_aware_optimizer
            )
        return grad_norm

    def count_zeros(self) -> float:
        """Count number of zeros in model's gradients."""
        params = self.get_parameters()
        return count_zeros_fp32(
            params,
            grad_stats_parallel_group=self.get_grad_stats_parallel_group(),
            use_decoupled_grad=self.config.use_precision_aware_optimizer,
        )

    @abstractmethod
    def zero_grad(self, set_to_none: bool = True):
        """Zero gradients and prepare for next forward pass."""
        pass

    @abstractmethod
    def get_loss_scale(self) -> torch.Tensor:
        """
        Get current loss scale factor.
        NOTE: The output should be a CUDA tensor of size 1.
        """
        pass

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Simple scaling."""
        return self.get_loss_scale() * loss

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
        """Return state_dict."""
        pass

    @abstractmethod
    def load_state_dict(self, state_dict):
        """Load pass-in `state_dict`."""
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
        if self.is_stub_optimizer:
            return []
        else:
            return self.optimizer.param_groups

    def _set_param_groups(self, value):
        self.optimizer.param_groups = value

    param_groups = property(_get_param_groups, _set_param_groups)

    @abstractmethod
    def step(self):
        """Step the optimizer."""
        pass

    @abstractmethod
    def sharded_state_dict(
        self, model_sharded_state_dict: ShardedStateDict, is_loading: bool = False
    ) -> ShardedStateDict:
        """Builds sharded state dict for the optimizer, based on model's sharded state dict.

        Args:
            model_sharded_state_dict (ShardedStateDict): sharded state dict of the model
            is_loading (bool, optional): flag indicating whether the state dict will be
                used to save or load the optimizer state. Defaults to False.

        Returns: optimizer sharded state dict
        """

    @staticmethod
    def _extract_common_per_param_step(state_dict) -> Union[int, torch.Tensor, None]:
        common_step = None
        for param_idx, param_state in state_dict['state'].items():
            param_step = param_state.get('step', None)
            if param_step is not None:
                if common_step is None:
                    common_step = param_step
                elif common_step != param_step:
                    raise ValueError(
                        "The optimizer step differs per parameter. Mcore only supports "
                        "optimizers whose step is shared across all parameters."
                    )
        return common_step

    @staticmethod
    def _restore_common_per_param_step(state_dict: Dict, step: Union[int, torch.Tensor]):
        for param_idx, param_state in state_dict['state'].items():
            param_state['step'] = copy.deepcopy(step)


class MixedPrecisionOptimizer(MegatronOptimizer):
    """Base class for both the float-16 and the distributed optimizer.

    Args:
        optimizer (torch.optim.Optimizer): base optimizer such as Adam or SGD.
        config (OptimizerConfig): configuration object for optimizer.
        grad_scaler (MegatronGradScaler): used for scaling gradients. Note that
            this can be None. This case happens when `bf16 = True` and we don't
            use any loss scale. Note that for `bf16 = True`, we can have
            a constant gradient scaler. Also for `bf16 = False`, we
            always require a grad scaler.
        init_state_fn (Callable, optional): function to initialize state in the optimizer.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: OptimizerConfig,
        grad_scaler: Optional[MegatronGradScaler],
        init_state_fn: Callable,
    ):
        if has_config_logger_enabled(config):
            log_config_to_disk(config, locals(), prefix=type(self).__name__)

        super().__init__(optimizer, config, init_state_fn)
        self.grad_scaler = grad_scaler

        # None grad scaler is only supported for bf16.
        if self.grad_scaler is None:
            assert not self.config.fp16, 'fp16 expects a grad scaler.'

        # Tensor used to determine if a nan/if has happend.
        # Any non-zero value indicates inf/nan.
        # Note that we keep this for the cases that grad scaler is none.
        # We still record nan/inf if we have a bfloat16 with a grad scaler.
        if self.grad_scaler:
            self.found_inf = torch.tensor([0.0], dtype=torch.float, device='cuda')

        # Dummy tensor needed for apex multi-apply tensor.
        # For bfloat, we don't have multi-tensor apply and for now
        # we set it to none so the multi-tensor apply gets ignored.
        if self.config.bf16:
            self._dummy_overflow_buf = None
        else:
            self._dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')

        # In case grad scaler is not passed, define the unity scale.
        if self.grad_scaler is None:
            self._scale_one = torch.tensor([1.0], dtype=torch.float, device='cuda')

    def get_loss_scale(self):
        if self.grad_scaler is None:
            return self._scale_one
        return self.grad_scaler.scale

    def reload_model_params(self):
        if self.param_groups:
            self._copy_model_params_to_main_params()

    def _unscale_main_grads_and_check_for_nan(self):

        # Collect main grads.
        if not self.is_stub_optimizer:
            main_grads = self._collect_main_grad_data_for_unscaling()

        # Reset found inf.
        self.found_inf.fill_(0.0)

        if not self.is_stub_optimizer:
            # Unscale and set found inf/nan
            torch._amp_foreach_non_finite_check_and_unscale_(
                main_grads, self.found_inf, self.grad_scaler.inv_scale
            )

        # Update across all model parallel instances.
        torch.distributed.all_reduce(
            self.found_inf,
            op=torch.distributed.ReduceOp.MAX,
            group=self.get_grad_stats_parallel_group(),
        )

        # Check for nan.
        found_inf_flag = self.found_inf.item() > 0

        return found_inf_flag

    @torch.no_grad()
    def prepare_grads(self) -> bool:
        """Pre-processing gradients before the optimizer step, returns whether inf/nan is found."""
        timers = self.config.timers

        # Copy gradients from model params to main params.
        if timers is not None:
            timers('optimizer-copy-to-main-grad', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        if not self.is_stub_optimizer:
            self._copy_model_grads_to_main_grads()
        if timers is not None:
            timers('optimizer-copy-to-main-grad').stop()

        # Do unscale, check for inf, and update grad scaler only for
        # the case that grad scaler is provided.
        if self.grad_scaler:

            # Unscale and check for inf/nan.
            if timers is not None:
                timers('optimizer-unscale-and-check-inf', log_level=1).start(
                    barrier=self.config.barrier_with_L1_time
                )
            found_inf_flag = self._unscale_main_grads_and_check_for_nan()
            if timers is not None:
                timers('optimizer-unscale-and-check-inf').stop()

            # We are done with scaling gradients
            # so we can update the loss scale.
            self.grad_scaler.update(found_inf_flag)

            return found_inf_flag

        return False

    @torch.no_grad()
    def step_with_ready_grads(self) -> bool:
        """Step the optimizer with ready gradients, return successful."""
        timers = self.config.timers
        # Step the optimizer.
        if timers is not None:
            timers('optimizer-inner-step', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        if not self.is_stub_optimizer:
            self.optimizer.step()
        if timers is not None:
            timers('optimizer-inner-step').stop()

        # Update params from main params.
        if timers is not None:
            timers('optimizer-copy-main-to-model-params', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        if not self.is_stub_optimizer:
            self._copy_main_params_to_model_params()
        if timers is not None:
            timers('optimizer-copy-main-to-model-params').stop()

        return True

    @torch.no_grad()
    def step(self):
        timers = self.config.timers

        found_inf_flag = self.prepare_grads()
        if found_inf_flag:
            return False, None, None

        # Clip the main gradients.
        if timers is not None:
            timers('optimizer-clip-main-grad', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        grad_norm = 0.0
        if self.config.clip_grad > 0.0:
            grad_norm = self.clip_grad_norm(self.config.clip_grad)
        if timers is not None:
            timers('optimizer-clip-main-grad').stop()

        # Count the zeros in the grads.
        if timers is not None:
            timers('optimizer-count-zeros', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        num_zeros_in_grad = self.count_zeros() if self.config.log_num_zeros_in_grad else 0
        if timers is not None:
            timers('optimizer-count-zeros').stop()

        success = self.step_with_ready_grads()

        # Successful update.
        return success, grad_norm, num_zeros_in_grad


class Float16OptimizerWithFloat16Params(MixedPrecisionOptimizer):
    """Float16 optimizer for fp16 and bf16 data types.

    Args:
        optimizer (torch.optim.Optimizer): base optimizer such as Adam or SGD.
        config (OptimizerConfig): configuration object for optimizer.
        grad_scaler (MegatronGradScaler): used for scaling gradients. Note that
            this can be None. This case happens when `bf16 = True` and we don't
            use any loss scale. Note that for `bf16 = True`, we can have
            a constant gradient scaler. Also for `bf16 = False`, we
            always require a grad scaler.
        init_state_fn (Callable, optional): function to initialize state in the optimizer.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: OptimizerConfig,
        grad_scaler: MegatronGradScaler,
        init_state_fn: Callable,
    ):

        super().__init__(optimizer, config, grad_scaler, init_state_fn)

        # Handle main parameters.

        if optimizer:
            # Three groups of parameters:
            #   float16_groups: original float16 parameters
            #   fp32_from_float16_groups: fp32 copy of float16 parameters
            #   fp32_from_fp32_groups: original fp32 parameters
            self.float16_groups = []
            self.fp32_from_float16_groups = []
            self.fp32_from_fp32_groups = []

            # For all the groups in the original optimizer:
            for param_group in self.optimizer.param_groups:
                float16_params_this_group = []
                fp32_params_this_group = []
                fp32_from_float16_params_this_group = []
                # For all the parameters in this group:
                for i, param in enumerate(param_group['params']):
                    if param.requires_grad:

                        # float16 params:
                        if param.type() in ['torch.cuda.HalfTensor', 'torch.cuda.BFloat16Tensor']:
                            float16_params_this_group.append(param)
                            # Create a copy
                            main_param = param.detach().clone().float()
                            # Copy tensor model parallel attributes.
                            tensor_parallel.copy_tensor_model_parallel_attributes(main_param, param)
                            if hasattr(param, 'shared'):
                                main_param.shared = param.shared
                            # Replace the optimizer params with the new fp32 copy.
                            param_group['params'][i] = main_param

                            # Store handle to main_param.
                            param.main_param = main_param

                            fp32_from_float16_params_this_group.append(main_param)
                            # Reset existing state dict key to the new main param.
                            if param in self.optimizer.state:
                                self.optimizer.state[main_param] = self.optimizer.state.pop(param)
                        # fp32 params.
                        elif param.type() == 'torch.cuda.FloatTensor':
                            fp32_params_this_group.append(param)
                            param_group['params'][i] = param

                        else:
                            raise TypeError(
                                'Wrapped parameters must be one of '
                                'torch.cuda.FloatTensor,  '
                                'torch.cuda.HalfTensor, or '
                                'torch.cuda.BFloat16Tensor. '
                                'Received {}'.format(param.type())
                            )

                self.float16_groups.append(float16_params_this_group)
                self.fp32_from_float16_groups.append(fp32_from_float16_params_this_group)
                self.fp32_from_fp32_groups.append(fp32_params_this_group)
            self.is_stub_optimizer = False
        else:
            self.is_stub_optimizer = True

    def zero_grad(self, set_to_none=True):
        """We only need to zero the model related parameters, i.e.,
        float16_groups & fp32_from_fp32_groups. We additionally zero
        fp32_from_float16_groups as a memory optimization to reduce
        fragmentation; in the case of set_to_none==True, the space
        used by this field can be safely deallocated at this point."""
        if self.is_stub_optimizer:
            return
        for group in self.float16_groups:
            _zero_grad_group_helper(group, set_to_none)
        for group in self.fp32_from_float16_groups:
            _zero_grad_group_helper(group, set_to_none)
        for group in self.fp32_from_fp32_groups:
            _zero_grad_group_helper(group, set_to_none)

    def _collect_main_grad_data_for_unscaling(self):
        if self.is_stub_optimizer:
            return

        main_grads = []

        # fp32 params from float16 ones.
        for main_group in self.fp32_from_float16_groups:
            for main_param in main_group:
                if main_param.grad is not None:
                    main_grads.append(main_param.grad.data)

        # Append fp32 parameters.
        for main_group in self.fp32_from_fp32_groups:
            for main_param in main_group:
                if main_param.grad is not None:
                    main_grads.append(main_param.grad.data)

        return main_grads

    def _get_model_and_main_params_data_float16(self):
        model_data = []
        main_data = []
        for model_group, main_group in zip(self.float16_groups, self.fp32_from_float16_groups):
            for model_param, main_param in zip(model_group, main_group):
                model_data.append(model_param.data)
                main_data.append(main_param.data)
        return model_data, main_data

    def _copy_model_grads_to_main_grads(self):
        # This only needs to be done for the float16 group.
        for model_group, main_group in zip(self.float16_groups, self.fp32_from_float16_groups):
            for model_param, main_param in zip(model_group, main_group):
                if hasattr(model_param, 'main_grad'):
                    main_param.grad = model_param.main_grad.float()
                else:
                    if model_param.grad is not None:
                        main_param.grad = model_param.grad.float()

                # Safe to deallocate model's grad/main_grad after copying.
                # (If using contiguous buffers, main_grad's memory should
                # persist and therefore should not be deallocated.)
                model_param.grad = None

        # For fp32 grads, we need to reset the grads to main grad.
        for model_group in self.fp32_from_fp32_groups:
            for model_param in model_group:
                model_param.grad = model_param.main_grad

    def _copy_main_params_to_model_params(self):
        # Only needed for the float16 params.
        model_data, main_data = self._get_model_and_main_params_data_float16()
        _multi_tensor_copy_this_to_that(
            this=main_data, that=model_data, overflow_buf=self._dummy_overflow_buf
        )

    def _copy_model_params_to_main_params(self):
        # Only needed for the float16 params.
        model_data, main_data = self._get_model_and_main_params_data_float16()
        _multi_tensor_copy_this_to_that(
            this=model_data, that=main_data, overflow_buf=self._dummy_overflow_buf
        )

    def state_dict(self):
        state_dict = {}
        state_dict['optimizer'] = self.optimizer.state_dict()
        if self.grad_scaler:
            state_dict['grad_scaler'] = self.grad_scaler.state_dict()
        state_dict['fp32_from_fp16_params'] = self.fp32_from_float16_groups
        return state_dict

    def sharded_state_dict(
        self, model_sharded_state_dict: ShardedStateDict, is_loading: bool = False
    ):

        if is_loading:
            self.init_state_fn(self.optimizer, self.config)

        state_dict = self.state_dict()

        id_to_sharded_param_map = get_param_id_to_sharded_param_map(
            model_sharded_state_dict, chain.from_iterable(g for g in self.float16_groups)
        )

        # Convert fp32_from_fp16_params
        assert len(state_dict['fp32_from_fp16_params']) == len(
            state_dict['optimizer']['param_groups']
        )
        state_dict['fp32_from_fp16_params'] = [
            [
                make_sharded_optimizer_tensor(
                    id_to_sharded_param_map[param_id],
                    fp32_param,
                    prefix=f'optimizer.state.fp32_param',
                )
                for param_id, fp32_param in zip(state_group['params'], fp32_group)
            ]
            for fp32_group, state_group in zip(
                state_dict['fp32_from_fp16_params'], state_dict['optimizer']['param_groups']
            )
        ]

        step = self._extract_common_per_param_step(state_dict['optimizer'])

        # Convert regular optimizer state
        # all optimizer parameters passed to optim_state_to_sharding_state are
        # expected to have the same shape as the model parameters,
        # so we save the step separately and ignore it here
        optim_state_to_sharding_state(
            state_dict['optimizer'], id_to_sharded_param_map, exclude_keys="step"
        )
        # save step as a shared step among all parameters. Separate per-parameter
        # steps are not supported
        if step:
            state_dict['optimizer']['state']['common_step'] = step
        return state_dict

    def load_state_dict(self, state_dict):
        pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
        # Optimizer.
        optimizer_key = 'optimizer'
        if optimizer_key not in state_dict:
            optimizer_key = 'optimizer_state_dict'
            logger.info('***WARNING*** loading optimizer from an old checkpoint ...')
        if 'common_step' in state_dict[optimizer_key]['state']:
            common_step = state_dict[optimizer_key]['state'].pop('common_step')
            self._restore_common_per_param_step(state_dict[optimizer_key], common_step)
        self.optimizer.load_state_dict(state_dict[optimizer_key])

        # Grad scaler.
        if 'grad_scaler' not in state_dict:
            if self.config.fp16:
                logger.info('***WARNING*** found an old checkpoint, will not load grad scaler ...')
        else:
            if self.grad_scaler:
                self.grad_scaler.load_state_dict(state_dict['grad_scaler'])
            else:
                logger.info(
                    '***WARNING*** fould the grad scaler in the '
                    'checkpoint but it is None in the class. '
                    'Skipping loading grad scaler ...'
                )

        # Copy data for the main params.
        fp32_from_float16_params_key = 'fp32_from_fp16_params'
        if fp32_from_float16_params_key not in state_dict:
            fp32_from_float16_params_key = 'fp32_from_fp16'
        for current_group, saved_group in zip(
            self.fp32_from_float16_groups, state_dict[fp32_from_float16_params_key]
        ):
            for current_param, saved_param in zip(current_group, saved_group):
                current_param.data.copy_(saved_param.data)


class FP32Optimizer(MegatronOptimizer):
    """Float32 optimizer.

    Args:
        optimizer (torch.optim.Optimizer): base optimizer such as Adam or SGD.
        config (OptimizerConfig): configuration object for optimizer.
        init_state_fn (Callable, optional): function to initialize state in the optimizer.
    """

    def __init__(
        self, optimizer: torch.optim.Optimizer, config: OptimizerConfig, init_state_fn: Callable
    ):
        if has_config_logger_enabled(config):
            log_config_to_disk(config, locals(), prefix=type(self).__name__)

        super(FP32Optimizer, self).__init__(optimizer, config, init_state_fn)

        self._scale = torch.tensor([1.0], dtype=torch.float, device='cuda')
        self.is_stub_optimizer = True if optimizer is None else False

    def zero_grad(self, set_to_none=True):
        """Copied from torch.optim.optimizer"""
        if self.is_stub_optimizer:
            return
        for group in self.optimizer.param_groups:
            _zero_grad_group_helper(group['params'], set_to_none)

    def get_loss_scale(self):
        """FP32 optimizer does not do any scaling."""
        return self._scale

    @torch.no_grad()
    def prepare_grads(self) -> bool:
        """Pre-processing gradients before the optimizer step, returns whether inf/nan is found."""
        if self.is_stub_optimizer:
            return False
        timers = self.config.timers

        # Copy main_grads to grads.
        if timers is not None:
            timers('optimizer-copy-to-main-grad', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                if hasattr(param, 'main_grad'):
                    param.grad = param.main_grad
        if timers is not None:
            timers('optimizer-copy-to-main-grad').stop()

        return False

    @torch.no_grad()
    def step_with_ready_grads(self) -> bool:
        """Step the optimizer with ready gradients, return successful."""
        if self.is_stub_optimizer:
            return True
        timers = self.config.timers

        # Update parameters.
        if timers is not None:
            timers('optimizer-inner-step', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        self.optimizer.step()
        if timers is not None:
            timers('optimizer-inner-step').stop()

        return True

    @torch.no_grad()
    def step(self):
        """Clip gradients (if needed) and step the base optimizer.
        Always return successful since there is no overflow."""
        timers = self.config.timers

        found_inf_flag = self.prepare_grads()
        if found_inf_flag:
            return False, None, None

        # Clip gradients.
        if timers is not None:
            timers('optimizer-clip-main-grad', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        grad_norm = None
        if self.config.clip_grad > 0.0:
            grad_norm = self.clip_grad_norm(self.config.clip_grad)
        if timers is not None:
            timers('optimizer-clip-main-grad').stop()

        # Count the zeros in the grads.
        if timers is not None:
            timers('optimizer-count-zeros', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        num_zeros_in_grad = self.count_zeros() if self.config.log_num_zeros_in_grad else None
        if timers is not None:
            timers('optimizer-count-zeros').stop()

        success = self.step_with_ready_grads()

        # No overflow for FP32 optimizer.
        return success, grad_norm, num_zeros_in_grad

    def reload_model_params(self):
        pass

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
        if 'common_step' in state_dict['state']:
            common_step = state_dict['state'].pop('common_step')
            self._restore_common_per_param_step(state_dict, common_step)
        self.optimizer.load_state_dict(state_dict)

    def sharded_state_dict(
        self, model_sharded_state_dict: ShardedStateDict, is_loading: bool = False
    ):
        if is_loading:
            self.init_state_fn(self.optimizer, self.config)

        state_dict = self.state_dict()
        id_to_sharded_param_map = get_param_id_to_sharded_param_map(
            model_sharded_state_dict, self.get_parameters()
        )
        step = self._extract_common_per_param_step(state_dict)

        # all optimizer parameters passed to optim_state_to_sharding_state are
        # expected to have the same shape as the model parameters,
        # so we save the step separately and ignore it here
        optim_state_to_sharding_state(state_dict, id_to_sharded_param_map, exclude_keys="step")
        # save step as a shared step among all parameters. Separate per-parameter
        # steps are not supported
        if step:
            state_dict['state']['common_step'] = step
        return state_dict


class ProxyDict:
    """
    A dictionary-like object that proxies to a list of dictionaries.

    e.g., ProxyDict([{'a': 1}, {'b': 2}]) behaves like:
    {
        (0, 'a'): 1,
        (1, 'b'): 2,
    }
    We use tuples as keys to avoid ambiguity with the keys of the inner dicts.
    """

    def __init__(self, inner_dicts: List[dict]):
        self._inner_dicts = inner_dicts

    def __getitem__(self, key: Tuple[int, str]):
        idx, inner_key = key
        return self._inner_dicts[idx].get(inner_key)

    def __setitem__(self, key: Tuple[int, str], value: Any):
        idx, inner_key = key
        self._inner_dicts[idx][inner_key] = value

    def __len__(self) -> int:
        return sum([len(inner_dict) for inner_dict in self._inner_dicts])

    def __iter__(self):
        for idx, inner_dict in enumerate(self._inner_dicts):
            for inner_key in inner_dict:
                yield (idx, inner_key)

    def items(self):
        """Return generator over underlying items."""
        for idx, inner_dict in enumerate(self._inner_dicts):
            for inner_key, value in inner_dict.items():
                yield (idx, inner_key), value


class ChainedOptimizer(MegatronOptimizer):
    """ChainedOptimizer is designed for a collection of optimizers.

    These optimizers are responsible for different parts of multiple models for
    a training task and will be executed one-by-one when the model is updated.

    Args:
        chained_optimizers: a list of optimizers.
    """

    def __init__(self, chained_optimizers: List[MegatronOptimizer]):
        self.model_chunks = []
        # chained_optimizers would be empty in the case that a rank
        # has no trainable parameters
        if chained_optimizers:
            self.config = getattr(chained_optimizers[0], 'config', None)
            for optimizer in chained_optimizers:
                if hasattr(optimizer, 'model_chunks'):
                    for model_chunk in optimizer.model_chunks:
                        if model_chunk not in self.model_chunks:
                            self.model_chunks.append(model_chunk)
                assert self.config == getattr(optimizer, 'config', None)
            self.is_stub_optimizer = False
        else:
            self.is_stub_optimizer = True
        self.chained_optimizers = chained_optimizers

    @property
    def param_groups(self) -> List[dict]:
        """Get param_groups aggregated over underlying optimizers."""
        param_groups = []
        for optimizer in self.chained_optimizers:
            param_groups += optimizer.param_groups
        return param_groups

    @property
    def state(self) -> ProxyDict:
        """
        Return optimizer state with tuple keys, where the first element is the
        index of the optimizer in the list of chained optimizers.
        """
        return ProxyDict([opt.state for opt in self.chained_optimizers])

    def zero_grad(self, set_to_none=True):
        for optimizer in self.chained_optimizers:
            optimizer.zero_grad(set_to_none)

    def get_loss_scale(self):
        if self.chained_optimizers:
            return self.chained_optimizers[0].get_loss_scale()
        else:
            return torch.tensor([1.0], dtype=torch.float32, device=torch.cuda.current_device())

    def reload_model_params(self):
        for optimizer in self.chained_optimizers:
            optimizer.reload_model_params()

    def state_dict(self):
        return [optimizer.state_dict() for optimizer in self.chained_optimizers]

    def sharded_state_dict(
        self, model_sharded_state_dict: ShardedStateDict, is_loading: bool = False, **kwargs
    ):
        sharded_state_dict = {}
        for optimizer_idx, optimizer in enumerate(self.chained_optimizers):
            optim_state_dict = optimizer.sharded_state_dict(
                model_sharded_state_dict, is_loading, **kwargs
            )
            add_prefix_for_sharding(optim_state_dict, f'chained_{optimizer_idx}.')
            sharded_state_dict[optimizer_idx] = optim_state_dict
        return sharded_state_dict

    def load_state_dict(self, state_dict):
        if len(self.chained_optimizers) != len(state_dict):
            raise RuntimeError(
                f'Expected {len(self.chained_optimizers)} entries'
                f' in state dict, but got {len(state_dict)}.'
            )
        if isinstance(state_dict, dict):
            state_dict = (v for k, v in sorted(state_dict.items()))
        for optimizer, state in zip(self.chained_optimizers, state_dict):
            optimizer.load_state_dict(state)

    @torch.no_grad()
    def prepare_grads(self) -> bool:
        """Pre-processing gradients before the optimizer step, returns whether inf/nan is found."""
        found_inf_flag = False
        for optimizer in self.chained_optimizers:
            found_inf_flag |= optimizer.prepare_grads()

        return found_inf_flag

    @torch.no_grad()
    def step_with_ready_grads(self) -> bool:
        """Step the optimizer with ready gradients, return successful."""
        success = True
        for optimizer_idx, optimizer in enumerate(self.chained_optimizers):
            success &= optimizer.step_with_ready_grads()
            if self.config.overlap_param_gather_with_optimizer_step and optimizer_idx == 0:
                assert success
                assert len(optimizer.model_chunks) == 1
                optimizer.model_chunks[0].start_param_sync(force_dispatch=True)

        return success

    @torch.no_grad()
    def step(self):
        """ChainedOptimizer will step all optimizers one by one."""
        if self.is_stub_optimizer:
            return True, 0.0, 0
        found_inf_flag = self.prepare_grads()
        if found_inf_flag:
            return False, None, None

        # Get grad norm.
        grad_norms = []
        for optimizer in self.chained_optimizers:
            _grad_norm = optimizer.get_grad_norm()
            grad_norms += [_grad_norm if _grad_norm else 0.0]
        grad_norm = math.sqrt(sum([x**2 for x in grad_norms]))

        # Clip gradients.
        for optimizer in self.chained_optimizers:
            if optimizer.config.clip_grad > 0.0:
                clip_grad_by_total_norm_fp32(
                    optimizer.get_parameters(),
                    max_norm=optimizer.config.clip_grad,
                    total_norm=grad_norm,
                    use_decoupled_grad=optimizer.config.use_precision_aware_optimizer,
                )

        # Count the zeros in the grads.
        num_zeros_in_grad = 0
        for optimizer in self.chained_optimizers:
            num_zeros_in_grad += (
                optimizer.count_zeros() if optimizer.config.log_num_zeros_in_grad else 0
            )

        update_successful = self.step_with_ready_grads()

        return update_successful, grad_norm, num_zeros_in_grad

    def save_parameter_state(self, filename: str):
        """Save the distributed parameter states of all optimizers to a file.

        Args:
            filename (str): path to save parameter state to.
        """
        save_states = False
        states = []
        for optimizer in self.chained_optimizers:
            if hasattr(optimizer, 'get_parameter_state_dp_zero'):
                state_dict = optimizer.get_parameter_state_dp_zero()

                # Save checkpoint economically, only when DP rank = 0, state dict
                # needs to be saved.
                if torch.distributed.get_rank(optimizer.data_parallel_group) == 0:
                    states.append(state_dict)
                    save_states = True
                else:
                    states.append(None)
            else:
                states.append(None)

        if save_states:
            torch.save(states, filename)

    def load_parameter_state(self, filename: str, *, update_legacy_format: bool = False):
        """Load the distributed parameter states of all optimizers from a file.

        Args:
            filename (str): path to load parameter state from.
        """
        states = None
        for idx, optimizer in enumerate(self.chained_optimizers):
            if not hasattr(optimizer, 'load_parameter_state_from_dp_zero'):
                continue

            # Lazy loading checkpoint, state dict is needed only when DP rank = 0.
            if torch.distributed.get_rank(optimizer.data_parallel_group) == 0 and states is None:
                states = torch.load(filename)

            state_dict = states[idx] if states else None
            optimizer.load_parameter_state_from_dp_zero(
                state_dict, update_legacy_format=update_legacy_format
            )
