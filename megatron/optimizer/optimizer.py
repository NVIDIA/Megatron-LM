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

from .clip_grads import clip_grad_norm_fp32, count_zeros_fp32

# >>>
from lutil import pax, tp
# <<<


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
    """Use multi-tensor-applier to copy values from one list to another.
    We don't have a blfoat16 implementation so for now if the overflow_buf
    is not provided, we default back to simple loop copy to be compatible
    with bfloat16."""
    if overflow_buf:
        overflow_buf.fill_(0)
        # Scaling with factor `1.0` is equivalent to copy.
        multi_tensor_applier(amp_C.multi_tensor_scale,
                             overflow_buf,
                             [this, that],
                             1.0)
    else:
        for this_, that_ in zip(this, that):
            that_.copy_(this_)



class MegatronOptimizer(ABC):


    def __init__(self, optimizer, clip_grad,
                 log_num_zeros_in_grad,
                 params_have_main_grad,
                 use_contiguous_buffers_in_local_ddp):

        """Input optimizer is the base optimizer for example Adam."""
        self.optimizer = optimizer
        assert self.optimizer, 'no optimizer is provided.'
        # Set gradient clipping and logging params.
        self.clip_grad = clip_grad
        self.log_num_zeros_in_grad = log_num_zeros_in_grad
        self.params_have_main_grad = params_have_main_grad
        self.use_contiguous_buffers_in_local_ddp = use_contiguous_buffers_in_local_ddp

        if self.use_contiguous_buffers_in_local_ddp:
            assert self.params_have_main_grad, \
                "use of contiguous buffer requires that params have main grad"

    def get_parameters(self):
        params = []
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                params.append(param)
        return params


    def clip_grad_norm(self, clip_grad):
        params = self.get_parameters()
        # >>>
        # pax(0, {
        #     "clip_grad" : clip_grad,
        #     "params": [ (p.tensor_model_parallel, tp(p)) for p in params ],
        #     "grads" : [ p.grad for p in params ],
        # })
        # <<<
        return clip_grad_norm_fp32(params, clip_grad)


    def count_zeros(self):
        params = self.get_parameters()
        return count_zeros_fp32(params)


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
    def reduce_gradients(self):
        pass


    @abstractmethod
    def step(self):
        pass


    @abstractmethod
    def gather_params(self):
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


class BaseFloat16Optimizer(MegatronOptimizer):

    def __init__(self, optimizer, clip_grad, log_num_zeros_in_grad,
                 params_have_main_grad, use_contiguous_buffers_in_local_ddp,
                 bf16, grad_scaler,
                 models):

        super().__init__(
            optimizer, clip_grad, log_num_zeros_in_grad,
            params_have_main_grad, use_contiguous_buffers_in_local_ddp)

        # >>>
        self.models = models
        # <<<
        self.bf16 = bf16
        self.grad_scaler = grad_scaler
        # None grad scaler is only supported for bf16.
        if self.grad_scaler is None:
            assert self.bf16, 'fp16 expects a grad scaler.'

        # Tensor used to determine if a nan/if has happend.
        # Any non-zero value indicates inf/nan.
        # Note that we keep this for the cases that grad scaler is none.
        # We still record nan/inf if we have a bfloat16 with a grad scaler.
        if self.grad_scaler:
            self.found_inf = torch.cuda.FloatTensor([0.0])

        # Dummy tensor needed for apex multi-apply tensor.
        # For bfloat, we don't have multi-tensor apply and for now
        # we set it to none so the multi-tensor apply gets ignored.
        if bf16:
            self._dummy_overflow_buf = None
        else:
            self._dummy_overflow_buf = torch.cuda.IntTensor([0])

        # In case grad scaler is not passed, define the unity scale.
        if self.grad_scaler is None:
            self._scale_one = torch.cuda.FloatTensor([1.0])


    def get_loss_scale(self):
        if self.grad_scaler is None:
            return self._scale_one
        return self.grad_scaler.scale


    def _unscale_main_grads_and_check_for_nan(self):

        # Collect main grads.
        main_grads = self._collect_main_grad_data_for_unscaling()
        # pax(1, {"main_grads": main_grads})

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

        # raise Exception("hi.")

        return found_inf_flag


    @torch.no_grad()
    def step(self):

        timers = get_timers()

        # Copy gradients from model params to main params.
        timers('optimizer-copy-to-main-grad').start()
        self._copy_model_grads_to_main_grads()
        timers('optimizer-copy-to-main-grad').stop()

        # Do unscale, check for inf, and update grad scaler only for
        # the case that grad scaler is provided.
        if self.grad_scaler:

            # Unscale and check for inf/nan.
            timers('optimizer-unscale-and-check-inf').start()
            found_inf_flag = self._unscale_main_grads_and_check_for_nan()
            timers('optimizer-unscale-and-check-inf').stop()

            # We are done with scaling gradients
            # so we can update the loss scale.
            self.grad_scaler.update(found_inf_flag)

            # If we found inf/nan, skip the update.
            if found_inf_flag:
                return False, None, None

        # Clip the main gradients.
        timers('optimizer-clip-main-grad').start()
        grad_norm = None
        if self.clip_grad > 0.0:
            grad_norm = self.clip_grad_norm(self.clip_grad)
        timers('optimizer-clip-main-grad').stop()

        # count the zeros in the grads
        num_zeros_in_grad = self.count_zeros() if \
                            self.log_num_zeros_in_grad else None

        # Step the optimizer.
        self.optimizer.step()

        # >>>
        # from lutil import pax, tp
        # pax(0, {
        #     "optimizer / state" :
        #     { hash(k):tp(v) for k,v in self.optimizer.state.items() },
        #     "optimizer / state / len" : len(self.optimizer.state),
        #     "optimizer / state / 0" : list(self.optimizer.state.values())[0],
        # })
        # <<<

        # Update params from main params.
        timers('optimizer-copy-main-to-model-params').start()
        self._copy_main_params_to_model_params()
        timers('optimizer-copy-main-to-model-params').stop()

        # Successful update.
        return True, grad_norm, num_zeros_in_grad


# class Float16OptimizerWithFloat16Params(MegatronOptimizer):
class Float16OptimizerWithFloat16Params(BaseFloat16Optimizer):
    """Float16 optimizer for fp16 and bf16 data types.

    Arguments:
        optimizer: base optimizer such as Adam or SGD
        clip_grad: clip gradeints with this global L2 norm. Note
            that clipping is ignored if clip_grad == 0
        log_num_zeros_in_grad: return number of zeros in the gradients.
        params_have_main_grad: flag indicating if parameters have
            a `main_grad` field. If this is set, we are assuming
            that the model parameters are store in the `main_grad`
            field instead of the typical `grad` field. This happens
            for the DDP cases where there is a continuous buffer
            holding the gradients. For example for bfloat16, we want
            to do gradient accumulation and all-reduces in float32
            and as a result we store those gradients in the main_grad.
            Note that main grad is not necessarily in float32.
        bf16: if true, the model is running in bfloat16.
        grad_scaler: used for scaling gradients. Note that this can be
            None. This case happens when `bf16 = True` and we don't
            use any loss scale. Note that for `bf16 = True`, we can have
            a constnat gradient scaler. Also for `bf16 = False`, we
            always require a grad scaler.
    """

    def __init__(self, optimizer, clip_grad, log_num_zeros_in_grad,
                 params_have_main_grad, use_contiguous_buffers_in_local_ddp,
                 bf16, grad_scaler, models):

        super().__init__(
            optimizer, clip_grad, log_num_zeros_in_grad,
            params_have_main_grad, use_contiguous_buffers_in_local_ddp,
            bf16, grad_scaler, models)

        # ======================
        # main parameter stuff
        # ======================

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
                    if param.type() in ['torch.cuda.HalfTensor',
                                        'torch.cuda.BFloat16Tensor']:
                        float16_params_this_group.append(param)
                        # Create a copy
                        main_param = param.detach().clone().float()
                        # Copy tensor model parallel attributes.
                        mpu.copy_tensor_model_parallel_attributes(main_param,
                                                                  param)
                        if hasattr(param, 'shared'):
                            main_param.shared = param.shared
                        # Replace the optimizer params with the new fp32 copy.
                        param_group['params'][i] = main_param

                        fp32_from_float16_params_this_group.append(main_param)
                        # Reset existing state dict key to the new main param.
                        if param in self.optimizer.state:
                            self.optimizer.state[main_param] \
                                = self.optimizer.state.pop(param)

                    # fp32 params.
                    elif param.type() == 'torch.cuda.FloatTensor':
                        # >>>
                        pax(0, {"param": param})
                        # <<<
                        fp32_params_this_group.append(param)
                        param_group['params'][i] = param

                    else:
                        raise TypeError('Wrapped parameters must be one of '
                                        'torch.cuda.FloatTensor,  '
                                        'torch.cuda.HalfTensor, or '
                                        'torch.cuda.BFloat16Tensor. '
                                        'Received {}'.format(param.type()))

            self.float16_groups.append(float16_params_this_group)
            self.fp32_from_float16_groups.append(
                fp32_from_float16_params_this_group)
            self.fp32_from_fp32_groups.append(fp32_params_this_group)

        # Leverage state_dict() and load_state_dict() to
        # recast preexisting per-param state tensors
        self.optimizer.load_state_dict(self.optimizer.state_dict())

        # >>>
        # from megatron.mpu.layers import param_is_not_tensor_parallel_duplicate
        # params = self.get_parameters()
        # pax(0, {
        #     # "params / 0" : params[0],
        #     "params" : [ (p.tensor_model_parallel, tp(p)) for p in params ],
        #     "grads" : [ (param_is_not_tensor_parallel_duplicate(p.grad), tp(p.grad)) for p in params ],
        # })
        # <<<


    def zero_grad(self, set_to_none=True):
        """We only need to zero the model related parameters, i.e.,
        float16_groups & fp32_from_fp32_groups. We additionally zero
        fp32_from_float16_groups as a memory optimization to reduce
        fragmentation; in the case of set_to_none==True, the space
        used by this field can be safely deallocated at this point."""
        for group in self.float16_groups:
            _zero_grad_group_helper(group, set_to_none)
        for group in self.fp32_from_float16_groups:
            _zero_grad_group_helper(group, set_to_none)
        for group in self.fp32_from_fp32_groups:
            _zero_grad_group_helper(group, set_to_none)


    # >>>
    def reduce_gradients(self, model):

        # >>>
        from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

        from megatron import get_args
        from megatron import get_timers
        from megatron.model import DistributedDataParallel as LocalDDP
        from megatron.model import Float16Module
        from megatron.utils import unwrap_model

        args = get_args()
        timers = get_timers()
        # <<<

        # >>>
        # if not args.use_distributed_optimizer:

        # All-reduce if needed.
        # >>>
        # if args.DDP_impl == 'local' and not args.use_distributed_optimizer:
        if args.DDP_impl == 'local':
        # <<<
            timers('backward-params-all-reduce').start()
            for model_module in model:
                # >>>
                # from lutil import pax, tp
                # pax(0, {
                #     "model" : model,
                #     "model_module" : model_module,
                # })
                # <<<
                # >>>
                # e.g., grad_shard = optimizer.get_grad_shard()
                # <<<
                model_module.allreduce_gradients()
            timers('backward-params-all-reduce').stop()

        # All-reduce word_embeddings' grad across first and last stages to ensure
        # that word_embeddings parameters stay in sync.
        # This should only run for models that support pipelined model parallelism
        # (BERT and GPT-2).
        timers('backward-embedding-all-reduce').start()
        if mpu.is_rank_in_embedding_group(ignore_virtual=True) and \
                mpu.get_pipeline_model_parallel_world_size() > 1:
            if mpu.is_pipeline_first_stage(ignore_virtual=True):
                unwrapped_model = model[0]
            elif mpu.is_pipeline_last_stage(ignore_virtual=True):
                unwrapped_model = model[-1]
            else:  # We do not support the interleaved schedule for T5 yet.
                unwrapped_model = model[0]
            unwrapped_model = unwrap_model(
                unwrapped_model, (torchDDP, LocalDDP, Float16Module))

            if unwrapped_model.share_word_embeddings:
                word_embeddings_weight = unwrapped_model.word_embeddings_weight()
                # >>>
                if args.DDP_impl == 'local':
                    grad = word_embeddings_weight.main_grad
                else:
                    grad = word_embeddings_weight.grad
                torch.distributed.all_reduce(grad, group=mpu.get_embedding_group())
                # +++
                # grad_shard = optimizer.get_grad_shard(word_embeddings)
                # torch.distributed.all_reduce(grad_shard,
                #                              group=mpu.get_embedding_group())
                # <<<

        # All-reduce position_embeddings grad across first (encoder) and split (decoder) 
        # stages to ensure that position embeddings parameters stay in sync.
        # This should only run for T5 models with pipeline parallelism
        if mpu.is_rank_in_position_embedding_group() and \
                mpu.get_pipeline_model_parallel_world_size() > 1 and \
                args.pipeline_model_parallel_split_rank is not None:
            unwrapped_model = model[0]
            unwrapped_model = unwrap_model(
                unwrapped_model, (torchDDP, LocalDDP, Float16Module))
            assert args.DDP_impl == 'local', \
                'T5 model is only supported with local DDP mode'
            # >>>
            grad = unwrapped_model.language_model.embedding.position_embeddings.weight.main_grad
            torch.distributed.all_reduce(grad, group=mpu.get_position_embedding_group())
            # +++
            # grad_shard = optimizer.get_grad_shard(
            #     unwrapped_model.language_model.embedding.position_embeddings.weight)
            # torch.distributed.all_reduce(grad_shard,
            #                              group=mpu.get_position_embedding_group())
            # <<<
        timers('backward-embedding-all-reduce').stop()

    def gather_params(self):

        raise Exception("hi.")

    def _copy_model_grads_to_main_grads(self):
        # This only needs to be done for the float16 group.
        for model_group, main_group in zip(self.float16_groups,
                                           self.fp32_from_float16_groups):
            for model_param, main_param in zip(model_group, main_group):
                if self.params_have_main_grad and hasattr(model_param, 'main_grad'):
                    main_param.grad = model_param.main_grad.float()
                else:
                    if model_param.grad is not None:
                        main_param.grad = model_param.grad.float()

                # Safe to deallocate model's grad/main_grad after copying.
                # (If using contiguous buffers, main_grad's memory should
                # persist and therefore should not be deallocated.)
                model_param.grad = None
                if self.params_have_main_grad and \
                   not self.use_contiguous_buffers_in_local_ddp:
                    model_param.main_grad = None

        # For fp32 grads, we need to reset the grads to main grad.
        if self.params_have_main_grad:
            for model_group in self.fp32_from_fp32_groups:
                for model_param in model_group:
                    model_param.grad = model_param.main_grad

                    # Safe to de-reference model's main_grad after copying.
                    # (If using contiguous buffers, main_grad's memory should
                    # persist and therefore should not be deallocated.)
                    if not self.use_contiguous_buffers_in_local_ddp:
                        model_param.main_grad = None

    def _collect_main_grad_data_for_unscaling(self):

        main_grads = []

        # fp32 params from float16 ones.
        for main_group in self.fp32_from_float16_groups:
            for main_param in main_group:
                if main_param.grad is not None:
                    main_grads.append(main_param.grad.data)

        # pax(1, {"main_grads": main_grads})

        # Append fp32 parameters.
        for main_group in self.fp32_from_fp32_groups:
            for main_param in main_group:
                if main_param.grad is not None:
                    main_grads.append(main_param.grad.data)
        
        # >>>
        # from megatron.mpu.layers import param_is_not_tensor_parallel_duplicate
        # pax(1, {"main_grads": [ (param_is_not_tensor_parallel_duplicate(t), tp(t)) for t in main_grads ]})
        # <<<

        return main_grads


    def _get_model_and_main_params_data_float16(self):
        model_data = []
        main_data = []
        for model_group, main_group in zip(self.float16_groups,
                                           self.fp32_from_float16_groups):
            for model_param, main_param in zip(model_group, main_group):
                model_data.append(model_param.data)
                main_data.append(main_param.data)
        return model_data, main_data


    def _copy_main_params_to_model_params(self):
        # Only needed for the float16 params.
        model_data, main_data = self._get_model_and_main_params_data_float16()
        _multi_tensor_copy_this_to_that(this=main_data, that=model_data,
                                        overflow_buf=self._dummy_overflow_buf)


    def _copy_model_params_to_main_params(self):
        # Only needed for the float16 params.
        model_data, main_data = self._get_model_and_main_params_data_float16()
        _multi_tensor_copy_this_to_that(this=model_data, that=main_data,
                                        overflow_buf=self._dummy_overflow_buf)


    def reload_model_params(self):
        self._copy_model_params_to_main_params()


    def state_dict(self):
        state_dict = {}
        state_dict['optimizer'] = self.optimizer.state_dict()
        if self.grad_scaler:
            state_dict['grad_scaler'] = self.grad_scaler.state_dict()
        state_dict['fp32_from_fp16_params'] = self.fp32_from_float16_groups
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
            if self.grad_scaler:
                self.grad_scaler.load_state_dict(state_dict['grad_scaler'])
            else:
                print_rank_0('***WARNING*** fould the grad scaler in the '
                             'checkpoint but it is None in the class. '
                             'Skipping loading grad scaler ...')

        # Copy data for the main params.
        fp32_from_float16_params_key = 'fp32_from_fp16_params'
        if fp32_from_float16_params_key not in state_dict:
            fp32_from_float16_params_key = 'fp32_from_fp16'
        for current_group, saved_group in zip(
                self.fp32_from_float16_groups,
                state_dict[fp32_from_float16_params_key]):
            for current_param, saved_param in zip(current_group, saved_group):
                current_param.data.copy_(saved_param.data)


# >>>
import math

# from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron import get_args
# from megatron import get_timers
# from megatron.model import DistributedDataParallel as LocalDDP
# from megatron.model import Float16Module
# from megatron.utils import unwrap_model

# class ShardIndex:
class Shard:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.size = end - start
    def normalize(self, start = 0):
        return Shard(start, start + self.size)
    def __str__(self):
        return "%d,%d [%d]" % (self.start, self.end, self.size)

# class Float16DistributedOptimizer(Float16OptimizerWithFloat16Params):
# class Float16DistributedOptimizer(MegatronOptimizer):
class Float16DistributedOptimizer(BaseFloat16Optimizer):

    # >>>
    @classmethod
    def test_reduce_scatter(cls):

        torch.manual_seed(mpu.get_data_parallel_rank())
        size = (20,)
        dtype = torch.float
        device = torch.cuda.current_device()
        data_parallel_world_size = mpu.get_data_parallel_world_size()
        data_parallel_group = mpu.get_data_parallel_group()

        input_list = [
            # torch.randn(size, dtype = dtype, device = device)
            5 * torch.randint(low = 1, high = 3, size = size, dtype = dtype, device = device)
            for _ in range(data_parallel_world_size)
        ]
        output = torch.empty(size, dtype = dtype, device = device)

        torch.distributed.reduce_scatter(
            output,
            input_list,
            group = data_parallel_group,
        )

        if torch.distributed.get_rank() == 0:
            print(output)
        pax(0, {
            "data_parallel_world_size" : data_parallel_world_size,
            "data_parallel_group" : data_parallel_group,
            "input_list" : input_list,
            "output" : tp(output),
        })
    # <<<

    # def __init__(self, *_args):
    #     super().__init__(*_args)
    # def __init__(self, optimizer, clip_grad, log_num_zeros_in_grad,
    #              params_have_main_grad, use_contiguous_buffers_in_local_ddp,
    #              bf16, grad_scaler):

    #     super().__init__(
    #         optimizer, clip_grad, log_num_zeros_in_grad,
    #         params_have_main_grad, use_contiguous_buffers_in_local_ddp,
    #         bf16, grad_scaler)

    #     # >>>
    #     # self.test_reduce_scatter()
    #     # <<<

    #     # >>>
    #     args = get_args()
    #     # <<<

    #     # Data parallel info.
    #     self.data_parallel_group = mpu.get_data_parallel_group()
    #     self.data_parallel_rank = mpu.get_data_parallel_rank()
    #     self.data_parallel_world_size = mpu.get_data_parallel_world_size()

    #     # Total trainable param count.
    #     # self.total_param_size = sum(
    #     #     p.numel()
    #     #     for g in self.param_groups
    #     #     for p in g["params"]
    #     #     # if p .requires_grad ???
    #     # )

    #     # Model params: group sizes, group offset maps.
    #     # self.model_params = []
    #     # self.model_param_group_sizes = []
    #     # self.model_param_group_offset_maps = []
    #     self.model_param_groups = []
    #     for param_group in self.optimizer.param_groups:
    #         param_group_offset = 0
    #         param_group_offset_map = {}
    #         for param in param_group['params']:
    #             if not param.requires_grad:
    #                 continue
    #             # self.model_params.append(param)
    #             param_group_offset_map[param] = {
    #                 "start" : param_group_offset,
    #                 "end" : param_group_offset + param.numel(),
    #             }
    #             param_group_offset += param.numel()
    #         # self.model_param_group_sizes.append(param_group_offset)
    #         # self.model_param_group_offset_maps.append(param_group_offset_map)
    #         self.model_param_groups.append({
    #             "size" : param_group_offset,
    #             "offset_map" : param_group_offset_map,
    #         })

    #     # pax(0, {
    #     #     "model_params" : model_params,
    #     #     "model_param_group_sizes" : model_param_group_sizes,
    #     #     "model_param_group_offset_maps" : model_param_group_offset_maps,
    #     # })

    #     # Shard allocator.
    #     # ** torch.nn.Parameter ??
    #     # ** MemoryBuffer ??
    #     allocate_shard = lambda shard_size, dtype : torch.empty(
    #         (shard_size,),
    #         dtype = dtype,
    #         device = torch.cuda.current_device(),
    #         requires_grad = True)

    #     # Allocate shards.
    #     # (Also, collect world DP shard info.)
    #     # model_main_dtypes = set([ args.params_dtype, torch.float ])
    #     model_main_dtypes = set([ torch.float ]) # fp32 only, for now
    #     # self.world_shard_info_groups = [] # world_group_shard_infos ?
    #     # self.main_param_shard_groups = []
    #     self.world_shard_infos = [{"groups": []} for _ in self.model_param_groups]
    #     for group_index, model_param_group in enumerate(self.model_param_groups):

    #         # Max world shard size.
    #         model_param_size = model_param_group["size"]
    #         max_world_shard_size = int(math.ceil(model_param_size /
    #                                              self.data_parallel_world_size))

    #         # DP world shard infos.
    #         # world_shard_infos = []
    #         for r in range(self.data_parallel_world_size):
    #             shard_start_index = r * max_world_shard_size
    #             shard_end_index = min(model_param_size,
    #                                   shard_start_index + max_world_shard_size)
    #             # world_shard_infos.append({
    #             self.world_shard_infos[r]["groups"].append({
    #                 "start" : shard_start_index,
    #                 "end" : shard_end_index,
    #                 "size" : shard_end_index - shard_start_index,
    #             })
    #         # self.world_shard_info_groups.append(world_shard_infos)
    #         # self.world_shard_infos[group_index].append(world_shard_infos)

    #         # DP local rank's shard info.
    #         # local_shard_info = world_shard_infos[self.data_parallel_rank]
    #         local_shard_info = \
    #             self.world_shard_infos[self.data_parallel_rank]["groups"][-1]
    #         local_shard_start_index = local_shard_info["start"]
    #         local_shard_end_index = local_shard_info["end"]
    #         local_shard_size = local_shard_info["size"]

    #         # Local shard's param 'slice' index map.
    #         local_shard_info["param_slice_index_map"] = {}
    #         for param, offset_dict in model_param_group["offset_map"].items():
    #             # param_start_index = offset_dict["start"]
    #             # param_end_index = offset_dict["end"]
    #             # param_shard_start_index = max(local_shard_start_index,
    #             #                               param_start_index)
    #             # param_shard_end_index = min(local_shard_end_index,
    #             #                             param_end_index)
    #             orig_start_index = offset_dict["start"]
    #             orig_end_index = offset_dict["end"]
    #             shard_start_index = max(
    #                 0,
    #                 orig_start_index - local_shard_start_index)
    #             shard_end_index = min(
    #                 local_shard_end_index,
    #                 orig_end_index - local_shard_start_index)

    #             # if param_shard_end_index > param_shard_start_index:
    #             #     # Indexes are relative to local shard start index.
    #             #     # local_shard_info["param_index_map"][param] = {
    #             #     #     "param" : (
    #             #     #         param_shard_start_index,
    #             #     #         param_shard_end_index,
    #             #     #     ),
    #             #     #     "shard" : (
    #             #     #         param_shard_start_index - local_shard_start_index,
    #             #     #         param_shard_end_index - local_shard_start_index,
    #             #     #     ),
    #             #     # }
    #             #     local_shard_info["param_slice_index_map"][param] = {
    #             #         "param_start" :
    #             #         param_shard_start_index,
    #             #         "shard_start" :
    #             #         param_shard_start_index - local_shard_start_index,
    #             #         "size":
    #             #         param_shard_end_index - param_shard_start_index,
    #             #     }
    #             if shard_end_index > shard_start_index:
    #                 local_shard_info["param_slice_index_map"][param] = {
    #                     "orig_start" : orig_start_index,
    #                     "shard_start" : shard_start_index,
    #                     "size" : shard_end_index - shard_start_index,
    #                 }

    #             # pax(0, {
    #             #     "local index" : "%d, %d" % (
    #             #         local_shard_start_index,
    #             #         local_shard_end_index,
    #             #     ),
    #             #     "param index" : "%s, %d" % (
    #             #         param_start_index,
    #             #         param_end_index,
    #             #     ),
    #             #     "param" : tp(param),
    #             #     "shard_param_index_map" : shard_param_index_map,
    #             #     "local_shard_info" : local_shard_info,
    #             # })

    #         # pax(2, {
    #         #     "data_parallel_rank" : self.data_parallel_rank,
    #         #     "local_shard_info" : local_shard_info,
    #         #     "param_index_map " : [
    #         #         (str(p.shape), i)
    #         #         for p, i in local_shard_info["param_index_map"].items()
    #         #     ],
    #         # })

    #         # Allocate shards.
    #         # (Non-fp32 shards are for convenience; e.g., intermediaries
    #         # between model params and main fp32 shard. Necessary???)
    #         # main_param_shards = {
    #         #     ty : allocate_shard(local_shard_size, ty)
    #         #     for ty in model_main_dtypes}
    #         main_param_shards = {}
    #         for dtype in model_main_dtypes:
    #             main_param = allocate_shard(local_shard_size, dtype)
    #             main_param.grad = allocate_shard(local_shard_size, dtype)
    #             # pax(0, {"main_param": main_param})
    #             main_param_shards[dtype] = main_param
    #         # self.main_param_shard_groups.append(main_param_shards)
    #         local_shard_info["data"] = main_param_shards

    #         # Update optimizer group.
    #         self.optimizer.param_groups[group_index]["params"] = \
    #             [ main_param_shards[torch.float] ]

    #         # pax(0, {
    #         #     "param_groups" : self.optimizer.param_groups,
    #         #     "params" : self.optimizer.param_groups[group_index]["params"],
    #         # })

    #     # Add world start/end indexes, for reduce/gather steps.
    #     offset = 0
    #     for r in self.world_shard_infos:
    #         r["start_index"] = offset
    #         offset += sum(g["size"] for g in r["groups"])
    #         r["end_index"] = offset

    #     # Leverage state_dict() and load_state_dict() to
    #     # recast preexisting per-param state tensors
    #     self.optimizer.load_state_dict(self.optimizer.state_dict())

    #     # >>>
    #     # pax(0, {
    #     #     "world_shard_infos" : self.world_shard_infos,
    #     #     **{
    #     #         "world_shard_infos / %d" % i : r
    #     #         for i, r in enumerate(self.world_shard_infos)
    #     #     },
    #     # })
    #     # <<<
    @classmethod
    # def get_ddp_gbuf_param_shards(cls, model, dtype, gbuf_start):
    # def get_ddp_gbuf_param_shard_map(cls, model, dtype, gbuf_start):
    # def get_model_gbuf_param_shard_index_map(cls,model,dtype,gbuf_world_index):
    def get_model_gbuf_param_shard_map(cls, model, dtype, gbuf_world_shard):

        # Param shard map.
        param_world_index_map = model._grad_buffer_param_index_map[dtype]
        param_shard_map = {}
        for param, param_world_indexes in param_world_index_map.items():

            # Shard range.
            param_world_start, param_world_end = param_world_indexes
            param_local_start = max(
                0,
                param_world_start - gbuf_world_shard.start)
            param_local_end = min(
                gbuf_world_shard.size,
                param_world_end - gbuf_world_shard.start)

            # Add shard, if within range.
            if param_local_end > param_local_start:
                param_local_shard = Shard(param_local_start, param_local_end)
                param_world_shard = param_local_shard.normalize(param_world_start)
                param_shard_map[param] = {
                    "local" : param_local_shard,
                    "world" : param_world_shard,
                }

        # pax(0, {"param_shard_map": [ str((str(p.shape), s)) for p,s in param_shard_map.items() ]})

        return param_shard_map

    @classmethod
    # def get_ddp_gbuf_shard(cls, model, dtype):
    # def get_model_gbuf_shard(cls, model, dtype):
    # def get_model_gbuf_shard_index(cls, model, dtype):
    def get_model_gbuf_shard(cls, model, dtype):

        data_parallel_rank = mpu.get_data_parallel_rank()
        data_parallel_world_size = mpu.get_data_parallel_world_size()

        # Grad buffer shard.
        grad_buffer = model._grad_buffers[dtype]
        gbuf_size = grad_buffer.numel
        max_gbuf_shard_size = int(math.ceil(gbuf_size / data_parallel_world_size))

        gbuf_world_all_shards = []
        for r in range(data_parallel_world_size):
            gbuf_world_start = r * max_gbuf_shard_size
            gbuf_world_end = min(gbuf_size, gbuf_world_start+max_gbuf_shard_size)
            gbuf_world_shard = Shard(gbuf_world_start, gbuf_world_end)
            gbuf_world_all_shards.append(gbuf_world_shard)
        gbuf_world_shard = gbuf_world_all_shards[data_parallel_rank]
        gbuf_local_shard = gbuf_world_shard.normalize()
        # gbuf_local_shard = Shard(0, gbuf_world_index.size)

        # Param shards.
        param_shard_map = cls.get_model_gbuf_param_shard_map(model,
                                                             dtype,
                                                             gbuf_world_shard)

        # Altogether.
        data = {
            "local" : gbuf_local_shard,
            "world" : gbuf_world_shard,
            "world_all" : gbuf_world_all_shards,
            "param_map" : param_shard_map,
        }

        # pax(0, {"data": data})

        return data

    @classmethod
    # def get_ddp_gbuf_shards(cls, model):
    # def get_ddp_gbuf_shard_map(cls, model):
    # def get_model_gbuf_shard_map(cls, model):
    # def get_model_gbuf_shard_index_map(cls, model):
    def get_model_gbuf_shard_map(cls, model):

        # shard_index_map = {
        shard_map = {
            dtype : cls.get_model_gbuf_shard(model, dtype)
            for dtype in model._grad_buffers
        }

        # pax(0, {"shard_map": shard_map})

        return shard_map

    @classmethod
    # def get_param_size_map(cls, model_gbuf_shards):
    # def get_param_model_gbuf_map(cls, model_gbuf_shards):
    def get_param_gbuf_map(cls, model_gbuf_shards):

        # param_size_map = {}
        param_gbuf_map = {}
        for model_index, model_gbuf_shard_map in enumerate(model_gbuf_shards):
            for dtype, gbuf_shard_map in model_gbuf_shard_map.items():
                for param, param_shard_map in gbuf_shard_map["param_map"].items():
                    # assert param not in param_size_map
                    # param_size_map[param] = param_shard_map["local"].size
                    param_gbuf_map[param] = (model_index, dtype)
                    # pax(0, {
                    #     "dtype" : dtype,
                    #     "gbuf_shard_map" : gbuf_shard_map,
                    #     "param" : tp(param),
                    #     "param_shard_map" : param_shard_map,
                    # })

        # pax(0, {
        #     "model_gbuf_shards" : model_gbuf_shards,
        #     # "param_size_map" :
        #     # [ (str(p.shape), s) for p, s in param_size_map.items() ],
        #     "param_gbuf_map" : param_gbuf_map,
        # })

        # return param_size_map
        return param_gbuf_map

    @classmethod
    def get_optimizer_group_shards(cls, param_groups, model_gbuf_shards):

        num_groups = len(param_groups)

        # Param group map.
        param_group_map = {}
        for group_index, group in enumerate(param_groups):
            for param in group["params"]:
                assert param.requires_grad
                param_group_map[param] = group_index

        # Optimizer group shards.
        group_shards = [ {"size": 0, "param_map": {}} for _ in param_groups ]
        for model_gbuf_shard_map in model_gbuf_shards:
            for dtype, gbuf_shard_map in model_gbuf_shard_map.items():
                for param in gbuf_shard_map["param_map"]:
                    
                    group_index = param_group_map[param]
                    group_shard = group_shards[group_index]
                    param_size = gbuf_shard_map["param_map"][param]["local"].size

                    param_group_start = group_shard["size"]
                    param_group_end = param_group_start + param_size
                    param_group_shard = Shard(param_group_start, param_group_end)

                    group_shard["size"] += param_size
                    group_shard["param_map"][param] = param_group_shard

                    # raise Exception("hi.")

        # pax(0, {"param_group_map": [
        #     (g, str(p.shape))
        #     for p, g in param_group_map.items()
        # ]})
        # pax(0, {"group_shards": group_shards})

        return group_shards

    def __init__(self, optimizer, clip_grad, log_num_zeros_in_grad,
                 params_have_main_grad, use_contiguous_buffers_in_local_ddp,
                 bf16, grad_scaler, models):

        super().__init__(
            optimizer, clip_grad, log_num_zeros_in_grad,
            params_have_main_grad, use_contiguous_buffers_in_local_ddp,
            bf16, grad_scaler, models)

        # >>>
        args = get_args()
        assert args.use_contiguous_buffers_in_local_ddp # already checked in args
        # <<<

        # pax(1, {"models": models})

        # # Data parallel info.
        # self.data_parallel_group = mpu.get_data_parallel_group()
        # self.data_parallel_rank = mpu.get_data_parallel_rank()
        # self.data_parallel_world_size = mpu.get_data_parallel_world_size()

        # Model grad buffer shards.
        self.model_gbuf_shards = []
        for model_index, model in enumerate(self.models):
            self.model_gbuf_shards.append(self.get_model_gbuf_shard_map(model))
        self.param_gbuf_map = self.get_param_gbuf_map(self.model_gbuf_shards)

        # Optimizer shards.
        self.opt_group_shards = self.get_optimizer_group_shards(
            self.optimizer.param_groups,
            self.model_gbuf_shards)

        # pax(0, {"opt_group_shards": self.opt_group_shards})

        # Allocate main param/grad shard.
        # ** torch.nn.Parameter ??
        # ** MemoryBuffer ??
        allocate_shard = lambda shard_size, dtype : torch.empty(
            (shard_size,),
            dtype = dtype,
            device = torch.cuda.current_device(),
            requires_grad = True)

        self.main_param_shards = []
        for group_index, group_shard in enumerate(self.opt_group_shards):

            group_size = group_shard["size"]

            # ** todo: for dtype in model_main_dtypes ........ **

            # Allocate shard.
            main_param = allocate_shard(group_size, torch.float)
            main_param.grad = allocate_shard(group_size, torch.float)
            self.main_param_shards.append(main_param)
            mpu.set_tensor_model_parallel_attributes(main_param, True, 0, 1)

            # Update optimizer group.
            self.optimizer.param_groups[group_index]["params"] = [ main_param ]

        # Leverage state_dict() and load_state_dict() to
        # recast preexisting per-param state tensors
        self.optimizer.load_state_dict(self.optimizer.state_dict())

        # >>>
        # pax(0, {
        #     "model_gbuf_shards" : self.model_gbuf_shards,
        #     "opt_group_shards" : self.opt_group_shards,
        #     "main_param_shards" : self.main_param_shards,
        # })
        # <<<

    # def get_loss_scale(self):
    #     if self.grad_scaler is None:
    #         return self._scale_one
    #     return self.grad_scaler.scale

    def load_state_dict(self):
        raise Exception("hi.")
    def reload_model_params(self):
        raise Exception("hi.")
    def state_dict(self):
        raise Exception("hi.")

    # def zero_grad(self, set_to_none=True):

    #     params = []
    #     for model_param_group in self.model_param_groups:
    #         params.extend(model_param_group["offset_map"].keys())
    #     for main_group in self.optimizer.param_groups:
    #         params.extend(main_group["params"])

    #     # _zero_grad_group_helper(params, set_to_none)
    #     _zero_grad_group_helper(params, set_to_none = False)

    #     # pax(0, {
    #     #     "model_param_groups" : self.model_param_groups,
    #     #     "params" : params,
    #     # })
    def zero_grad(self, set_to_none=True):

        model_params = []
        for model in self.models:
            for dtype, param_map in model._grad_buffer_param_index_map.items():
                model_params.extend(param_map.keys())
        # main_params = []
        # for main_group in self.optimizer.param_groups:
        #     main_params.extend(main_group["params"])

        _zero_grad_group_helper(model_params, set_to_none)
        # _zero_grad_group_helper(params, set_to_none = False)

        # pax(0, {"params": params})

    # def reduce_gradients(self, model):

    #     # >>>
    #     # pax(0, {"main param" : self.world_shard_info_groups[0][self.data_parallel_rank]["data"][torch.float]})
    #     # <<<

    #     # >>>
    #     args = get_args()
    #     # timers = get_timers()
    #     # <<<

    #     # >>> [ temporary requirement ... and already checked in arguments.py ]
    #     assert args.use_contiguous_buffers_in_local_ddp
    #     # <<<

    #     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #     # Map param to virtual model.
    #     # ** ideally, this should happen once, during construction.
    #     param_model_map = {}
    #     for vmodel in model:
    #         for dtype, param_index_map in \
    #             vmodel._grad_buffer_param_index_map.items():
    #             for param in param_index_map:
    #                 param_model_map[param] = {
    #                     "dtype" : dtype,
    #                     "model" : vmodel,
    #                 }

    #     # pax(0, {
    #     #     "param_model_map" : [
    #     #         (str(tuple(p.shape)), m)
    #     #         for p, m in param_model_map.items()
    #     #     ],
    #     # })

    #     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #     # Copy model grads to main shard.
    #     local_shard_info_groups = [g[self.data_parallel_rank]
    #                                for g in self.world_shard_info_groups]
    #     for group_index, local_shard_info in enumerate(local_shard_info_groups):
            
    #         # model_param_index_map = 
    #         # shard_param_index_map = local_shard_info["param_index_map"]
    #         # main_index_map = local_shard_info["param_index_map"]
    #         main_slice_index_map = local_shard_info["param_slice_index_map"]
    #         for param, main_slice_indexes in main_slice_index_map.items():

    #             main_slice_orig_start_index = main_slice_indexes["orig_start"]
    #             main_slice_shard_start_index = main_slice_indexes["shard_start"]
    #             main_slice_size = main_slice_indexes["size"]

    #             dtype_model_dict = param_model_map[param]
    #             dtype = dtype_model_dict["dtype"]
    #             vmodel = dtype_model_dict["model"]
    #             model_grad_buffer = vmodel._grad_buffers[dtype].data
    #             model_grad_buffer_start_index = \
    #                 vmodel._grad_buffer_param_index_map[dtype][param][0] + \
    #                 main_slice_orig_start_index
                
    #             main_grad_view = local_shard_info["data"][torch.float].grad[
    #                 main_slice_shard_start_index:
    #                 main_slice_shard_start_index + main_slice_size
    #             ]
    #             model_grad_view = model_grad_buffer[
    #                 model_grad_buffer_start_index:
    #                 model_grad_buffer_start_index + main_slice_size
    #             ]

    #             main_grad_view.detach().copy_(model_grad_view)

    #             # pax(0, {
    #             #     # "local_shard_info" : local_shard_info,
    #             #     "main_slice_orig_start_index" : main_slice_orig_start_index,
    #             #     "main_slice_shard_start_index" : main_slice_shard_start_index,
    #             #     "main_slice_size" : main_slice_size,
    #             #     "model_grad_buffer_start_index" :
    #             #     model_grad_buffer_start_index,
    #             #     "main_grad_view" : tp(main_grad_view),
    #             #     "main_grad_view / detach" : tp(main_grad_view.detach()),
    #             #     "model_grad_view" : tp(model_grad_view),
    #             # })

    #         # pax(0, {
    #         #     "group_index" : group_index,
    #         #     "local_shard_info" : local_shard_info,
    #         #     "shard_param_index_map" : shard_param_index_map,
    #         #     "param" : tp(param),
    #         #     "shard_indexes" : shard_indexes,
    #         #     "grad_buffer_indexes" : grad_buffer_indexes,
    #         # })

    #     pax(0, {
    #         # "world_shard_info_groups" : self.world_shard_info_groups,
    #         # **{"world_shard_info_groups / %d" % i : v
    #         #    for i, v in enumerate(self.world_shard_info_groups)},
    #         # "local_shard_info_groups" : local_shard_info_groups,
    #         "local_shard_info_groups" : [ g["data"] for g in local_shard_info_groups ],
    #     })

    def get_model_grad_buffer_dp_views(self):

        # ** only contiguous grad buffer supported, for now [ TEMPORARY ] **
        args = get_args()
        assert args.use_contiguous_buffers_in_local_ddp

        # Grad buffer views.
        gbuf_view_items = []
        for model_index, model in enumerate(self.models):
            for dtype, gbuf_shard in self.model_gbuf_shards[model_index].items():
                world_shards = gbuf_shard["world_all"]

                gbuf = model._grad_buffers[dtype]
                gbuf_views = []
                for shard in world_shards:
                    gbuf_views.append(gbuf.data[shard.start:shard.end])

                gbuf_view_items.append((model_index, dtype, gbuf_views))

        # pax(0, {"gbuf_view_items": gbuf_view_items})

        return gbuf_view_items

    def reduce_gradients(self, model):

        # >>>
        args = get_args()
        # timers = get_timers()
        # <<<

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Sync word embedding params.

        # ... todo ...

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Sync T5 position embedding params.

        # ... todo ...

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Reduce-scatter.

        # # ** only contiguous grad buffer supported, for now [ TEMPORARY ] **
        # assert args.use_contiguous_buffers_in_local_ddp

        # data_parallel_rank = mpu.get_data_parallel_rank()
        # data_parallel_group = mpu.get_data_parallel_group()
        # for model_index, model in enumerate(self.models):
        #     for dtype, gbuf_shard in self.model_gbuf_shards[model_index].items():
        #         world_shards = gbuf_shard["world_all"]

        #         gbuf = model._grad_buffers[dtype]
        #         gbuf_views = []
        #         for shard in world_shards:
        #             gbuf_views.append(gbuf.data[shard.start:shard.end])

        #         torch.distributed.reduce_scatter(
        #             gbuf_views[data_parallel_rank],
        #             gbuf_views,
        #             group = data_parallel_group,
        #         )

        #         # pax(0, {
        #         #     "model_index" : model_index,
        #         #     "model" : model,
        #         #     "dtype" : str(dtype),
        #         #     "gbuf_shard" : gbuf_shard,
        #         #     "world_shards" : world_shards,
        #         #     "gbuf_views" : gbuf_views,
        #         # })

        data_parallel_rank = mpu.get_data_parallel_rank()
        data_parallel_group = mpu.get_data_parallel_group()

        gbuf_view_items = self.get_model_grad_buffer_dp_views()

        for model_index, dtype, gbuf_views in gbuf_view_items:
            torch.distributed.reduce_scatter(
                gbuf_views[data_parallel_rank],
                gbuf_views,
                group = data_parallel_group,
            )
            
        # pax(0, {"gbuf_view_items": gbuf_view_items})

    def gather_params(self):

        data_parallel_rank = mpu.get_data_parallel_rank()
        data_parallel_group = mpu.get_data_parallel_group()

        gbuf_view_items = self.get_model_grad_buffer_dp_views()

        for model_index, dtype, gbuf_views in gbuf_view_items:
            torch.distributed.all_gather(
                gbuf_views,
                gbuf_views[data_parallel_rank],
                group = data_parallel_group,
            )

        # for param, (model_index, dtype) in self.param_gbuf_map.items():
        #     gbuf = self.model_gbuf_shards[model_index][dtype]

        #     pax(0, {
        #         "param" : tp(param),
        #         "model_index" : model_index,
        #         "dtype" : str(dtype),
        #         "gbuf" : gbuf,
        #     })
        for param in self.param_gbuf_map:
            param.detach().copy_(param.main_grad)
            # pax(0, {
            #     "param" : tp(param),
            #     "main_grad" : tp(param.main_grad),
            #     # "grad" : tp(param.grad),
            # })

        # pax(0, {
        #     "gbuf_view_items" : gbuf_view_items,
        #     "param_gbuf_map" : [
        #         (str(tuple(p.shape)), d)
        #         for p, d in self.param_gbuf_map.items()
        #     ],
        # })

    # def step(self):

    #     raise Exception("step.")

    def _collect_main_grad_data_for_unscaling(self):
        # pax(1, {
        #     "main_param_shards" : self.main_param_shards,
        #     "grads" : [ p.grad.data for p in self.main_param_shards ],
        # })
        return [ p.grad.data for p in self.main_param_shards ]

    def _copy_model_grads_to_main_grads(self):

        for group_index, group_shard in enumerate(self.opt_group_shards):
            for param, main_shard in group_shard["param_map"].items():

                model_index, gbuf_dtype = self.param_gbuf_map[param]
                model_shard = self.model_gbuf_shards \
                    [model_index][gbuf_dtype]["param_map"][param]["world"]

                assert main_shard.size == model_shard.size

                model_grad_tensor = \
                    self.models[model_index]._grad_buffers[gbuf_dtype].data
                main_grad_tensor = \
                    self.main_param_shards[group_index].grad

                model_grad_view = \
                    model_grad_tensor[model_shard.start:model_shard.end]
                main_grad_view = \
                    main_grad_tensor[main_shard.start:main_shard.end]

                main_grad_view.detach().copy_(model_grad_view)

                # pax(0, {
                #     "group_index" : group_index,
                #     "group_shard" : group_shard,
                #     "param" : tp(param),
                #     "model_index" : model_index,
                #     "gbuf_dtype" : str(gbuf_dtype),
                #     "model_grad_tensor" : tp(model_grad_tensor),
                #     "main_grad_tensor" : tp(main_grad_tensor),
                #     "model_grad_view" : tp(model_grad_view),
                #     "main_grad_view" : tp(main_grad_view),
                #     "model_shard" : str(model_shard),
                #     "main_shard" : str(main_shard),
                # })

        # pax(0, {
        #     "model_gbuf_shards" : self.model_gbuf_shards,
        #     "opt_group_shards" : self.opt_group_shards,
        # })


    def _copy_main_params_to_model_params(self):

        for group_index, group_shard in enumerate(self.opt_group_shards):
            for param, main_shard in group_shard["param_map"].items():

                model_index, gbuf_dtype = self.param_gbuf_map[param]
                model_shard = self.model_gbuf_shards \
                    [model_index][gbuf_dtype]["param_map"][param]["world"]

                assert main_shard.size == model_shard.size

                # Use DDP's contiguous buffer to temporarily hold params.
                model_tensor = \
                    self.models[model_index]._grad_buffers[gbuf_dtype].data
                main_tensor = self.main_param_shards[group_index]

                # Copy sub-range within tensor.
                model_view = model_tensor[model_shard.start:model_shard.end]
                main_view = main_tensor[main_shard.start:main_shard.end]

                model_view.detach().copy_(main_view)

                # Debug.
                # pax(0, {
                #     "group_index" : group_index,
                #     "group_shard" : group_shard,
                #     "param" : tp(param),
                #     "model_index" : model_index,
                #     "gbuf_dtype" : str(gbuf_dtype),
                #     "model_grad_tensor" : tp(model_grad_tensor),
                #     "main_grad_tensor" : tp(main_grad_tensor),
                #     "model_grad_view" : tp(model_grad_view),
                #     "main_grad_view" : tp(main_grad_view),
                #     "model_shard" : str(model_shard),
                #     "main_shard" : str(main_shard),
                # })

        # pax(0, {
        #     "model_gbuf_shards" : self.model_gbuf_shards,
        #     "opt_group_shards" : self.opt_group_shards,
        # })

# <<<


class FP32Optimizer(MegatronOptimizer):

    def __init__(self, optimizer, clip_grad,
                 log_num_zeros_in_grad,
                 params_have_main_grad,
                 use_contiguous_buffers_in_local_ddp):

        super(FP32Optimizer, self).__init__(
            optimizer, clip_grad, log_num_zeros_in_grad,
            params_have_main_grad, use_contiguous_buffers_in_local_ddp)

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

        # Copy main_grads to grads.
        if self.params_have_main_grad:
            for param_group in self.optimizer.param_groups:
                for param in param_group['params']:
                    param.grad = param.main_grad

                    # Safe to de-reference model's main_grad after copying.
                    # (If using contiguous buffers, main_grad's memory should
                    # persist and therefore should not be deallocated.)
                    if not self.use_contiguous_buffers_in_local_ddp:
                        param.main_grad = None

        # Clip gradients.
        grad_norm = None
        if self.clip_grad > 0.0:
            grad_norm = self.clip_grad_norm(self.clip_grad)

        # count the zeros in the grads
        num_zeros_in_grad = self.count_zeros() if \
                            self.log_num_zeros_in_grad else None

        # Update parameters.
        self.optimizer.step()

        # No overflow for FP32 optimizer.
        return True, grad_norm, num_zeros_in_grad


    def reload_model_params(self):
        pass


    def state_dict(self):
        return self.optimizer.state_dict()


    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
