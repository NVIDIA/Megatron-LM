# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Megatron optimizer."""

from abc import ABC
from abc import abstractmethod
from apex.multi_tensor_apply import multi_tensor_applier
import amp_C
import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from megatron import get_timers
from megatron import print_rank_0
from megatron.core import mpu, tensor_parallel
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model import Float16Module
from megatron.model.module import param_is_not_shared
from megatron.utils import unwrap_model

from .clip_grads import clip_grad_norm_fp32, count_zeros_fp32


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
                 use_contiguous_buffers_in_local_ddp,
                 models):

        """Input optimizer is the base optimizer for example Adam."""
        self.optimizer = optimizer
        assert self.optimizer, 'no optimizer is provided.'
        # Set gradient clipping and logging params.
        self.clip_grad = clip_grad
        self.log_num_zeros_in_grad = log_num_zeros_in_grad
        self.params_have_main_grad = params_have_main_grad
        self.use_contiguous_buffers_in_local_ddp = use_contiguous_buffers_in_local_ddp

        # 'models' are retained for access to the contiguous grad buffers.
        # (see distributed optimizer)
        self.models = models

        if self.use_contiguous_buffers_in_local_ddp:
            assert self.params_have_main_grad, \
                "use of contiguous buffer requires that params have main grad"


    def get_parameters(self):
        params = []
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                params.append(param)
        return params


    def get_main_grads_for_grad_norm(self):

        # Filter parameters based on:
        #   - grad should not be none
        #   - parameter should not be shared
        #   - should not be a replica due to tensor model parallelism
        params = self.get_parameters()
        grads_for_norm = []
        for param in params:
            grad = param.grad
            grad_not_none = grad is not None
            is_not_shared = param_is_not_shared(param)
            is_not_tp_duplicate = tensor_parallel.param_is_not_tensor_parallel_duplicate(param)
            if grad_not_none and is_not_shared and is_not_tp_duplicate:
                grads_for_norm.append(grad)

        return grads_for_norm


    def get_model_parallel_group(self):
        """Default returned here, but the distributed optimizer overrides this."""
        return mpu.get_model_parallel_group()


    def clip_grad_norm(self, clip_grad):
        params = self.get_parameters()
        grads_for_norm = self.get_main_grads_for_grad_norm()
        return clip_grad_norm_fp32(
            params, grads_for_norm, clip_grad,
            model_parallel_group=self.get_model_parallel_group())


    def count_zeros(self):
        params = self.get_parameters()
        return count_zeros_fp32(params,
                                model_parallel_group=self.get_model_parallel_group())


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


    @abstractmethod
    def step(self, args, timers):
        pass


    def gather_model_params(self, args, timers):
        """
        For the case of a non-distributed-optimizer, there is nothing to
        do here.
        """
        pass


    def allreduce_word_embedding_grads(self, args):
        """
        All-reduce word embedding grads.

        Reduce grads across first and last stages to ensure that word_embeddings
        parameters stay in sync. This should only run for models that support
        pipelined model parallelism (BERT and GPT-2).
        """

        if mpu.is_rank_in_embedding_group(ignore_virtual=True) and \
                mpu.get_pipeline_model_parallel_world_size() > 1:
            if mpu.is_pipeline_first_stage(ignore_virtual=True):
                unwrapped_model = self.models[0]
            elif mpu.is_pipeline_last_stage(ignore_virtual=True):
                unwrapped_model = self.models[-1]
            else:  # We do not support the interleaved schedule for T5 yet.
                unwrapped_model = self.models[0]
            unwrapped_model = unwrap_model(
                unwrapped_model, (torchDDP, LocalDDP, Float16Module))

            if unwrapped_model.share_word_embeddings:
                word_embeddings_weight = unwrapped_model.word_embeddings_weight()
                if args.DDP_impl == 'local':
                    grad = word_embeddings_weight.main_grad
                else:
                    grad = word_embeddings_weight.grad
                torch.distributed.all_reduce(grad, group=mpu.get_embedding_group())


    def allreduce_position_embedding_grads(self, args):
        """
        All-reduce position_embeddings grad across first (encoder) and
        split (decoder) stages to ensure that position embeddings parameters
        stay in sync. This should only run for T5 models with pipeline
        parallelism.
        """
        if mpu.is_rank_in_position_embedding_group() and \
                mpu.get_pipeline_model_parallel_world_size() > 1 and \
                args.pipeline_model_parallel_split_rank is not None:
            unwrapped_model = self.models[0]
            unwrapped_model = unwrap_model(
                unwrapped_model, (torchDDP, LocalDDP, Float16Module))
            assert args.DDP_impl == 'local', \
                'T5 model is only supported with local DDP mode'
            grad = unwrapped_model.language_model.embedding.position_embeddings.weight.main_grad
            torch.distributed.all_reduce(grad, group=mpu.get_position_embedding_group())


    def allreduce_embedding_grads(self, args):
        """All-reduce both word and position embeddings."""
        self.allreduce_word_embedding_grads(args)
        self.allreduce_position_embedding_grads(args)


    def allreduce_layernorm_grads(self, args):
        """All-reduce layernorm grads (for sequence parallelism)."""

        # All-reduce layernorm parameters across model parallel nodes
        # when sequence parallelism is used
        if mpu.get_tensor_model_parallel_world_size() > 1 and \
                args.sequence_parallel:
            grads = []
            for model_module in self.models:
                unwrapped_model = unwrap_model( 
                    model_module, (torchDDP, LocalDDP, Float16Module))
                for param in unwrapped_model.parameters():
                    if getattr(param, 'sequence_parallel', False):
                        grad = param.main_grad if args.DDP_impl == 'local' else param.grad
                        grads.append(grad.data)
            coalesced = _flatten_dense_tensors(grads)
            torch.distributed.all_reduce(
                coalesced, group=mpu.get_tensor_model_parallel_group())
            for buf, synced in zip(grads, _unflatten_dense_tensors(
                    coalesced, grads)):
                buf.copy_(synced)


    def reduce_model_grads(self, args, timers):
        """All-reduce all grads, and all-reduce embeddings."""

        # All-reduce layer-norm grads (for sequence parallelism).
        timers('layernorm-grads-all-reduce', log_level=1).start(
            barrier=args.barrier_with_L1_time)
        self.allreduce_layernorm_grads(args)
        timers('layernorm-grads-all-reduce').stop()

        # All-reduce if needed.
        if args.DDP_impl == 'local':
            timers('grads-all-reduce', log_level=1).start(
                barrier=args.barrier_with_L1_time)
            for model in self.models:
                model.allreduce_gradients()
            timers('grads-all-reduce').stop()

        # All-reduce embedding grads.
        timers('embedding-grads-all-reduce', log_level=1).start(
            barrier=args.barrier_with_L1_time)
        self.allreduce_embedding_grads(args)
        timers('embedding-grads-all-reduce').stop()


class MixedPrecisionOptimizer(MegatronOptimizer):
    """Base class for both the float-16 and the distributed optimizer.

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
        use_contiguous_buffers_in_local_ddp: if true, the local DDP model
            is using a contiguous buffer to hold the model grads.
        fp16: if true, the model is running in fp16.
        bf16: if true, the model is running in bfloat16.
        params_dtype: used by distributed optimizer.
        grad_scaler: used for scaling gradients. Note that this can be
            None. This case happens when `bf16 = True` and we don't
            use any loss scale. Note that for `bf16 = True`, we can have
            a constnat gradient scaler. Also for `bf16 = False`, we
            always require a grad scaler.
        models: list of models (i.e., the virtual pipelining models). This
            is used by the distributed optimizer for mapping parameters.
    """

    def __init__(self, optimizer, clip_grad, log_num_zeros_in_grad,
                 params_have_main_grad, use_contiguous_buffers_in_local_ddp,
                 fp16, bf16, params_dtype, grad_scaler,
                 models):

        super().__init__(
            optimizer, clip_grad, log_num_zeros_in_grad,
            params_have_main_grad, use_contiguous_buffers_in_local_ddp,
            models)

        self.fp16 = fp16
        self.bf16 = bf16
        self.params_dtype = params_dtype
        self.grad_scaler = grad_scaler

        # None grad scaler is only supported for bf16.
        if self.grad_scaler is None:
            assert not self.fp16, 'fp16 expects a grad scaler.'

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


    def reload_model_params(self):
        self._copy_model_params_to_main_params()


    def _unscale_main_grads_and_check_for_nan(self):

        # Collect main grads.
        main_grads = self._collect_main_grad_data_for_unscaling()

        # Reset found inf.
        self.found_inf.fill_(0.0)

        # Unscale and set found inf/nan
        torch._amp_foreach_non_finite_check_and_unscale_(
            main_grads, self.found_inf, self.grad_scaler.inv_scale)

        # Update across all model parallel instances.
        torch.distributed.all_reduce(self.found_inf,
                                     op=torch.distributed.ReduceOp.MAX,
                                     group=self.get_model_parallel_group())

        # Check for nan.
        found_inf_flag = (self.found_inf.item() > 0)

        return found_inf_flag


    @torch.no_grad()
    def step(self, args, timers):

        # Copy gradients from model params to main params.
        timers('optimizer-copy-to-main-grad', log_level=1).start(
            barrier=args.barrier_with_L1_time)
        self._copy_model_grads_to_main_grads()
        timers('optimizer-copy-to-main-grad').stop()

        # Do unscale, check for inf, and update grad scaler only for
        # the case that grad scaler is provided.
        if self.grad_scaler:

            # Unscale and check for inf/nan.
            timers('optimizer-unscale-and-check-inf', log_level=1).start(
                barrier=args.barrier_with_L1_time)
            found_inf_flag = self._unscale_main_grads_and_check_for_nan()
            timers('optimizer-unscale-and-check-inf').stop()

            # We are done with scaling gradients
            # so we can update the loss scale.
            self.grad_scaler.update(found_inf_flag)

            # If we found inf/nan, skip the update.
            if found_inf_flag:
                return False, None, None

        # Clip the main gradients.
        timers('optimizer-clip-main-grad', log_level=1).start(
            barrier=args.barrier_with_L1_time)
        grad_norm = None
        if self.clip_grad > 0.0:
            grad_norm = self.clip_grad_norm(self.clip_grad)
        timers('optimizer-clip-main-grad').stop()

        # Count the zeros in the grads.
        timers('optimizer-count-zeros', log_level=1).start(
            barrier=args.barrier_with_L1_time)
        num_zeros_in_grad = self.count_zeros() if \
                            self.log_num_zeros_in_grad else None
        timers('optimizer-count-zeros').stop()

        # Step the optimizer.
        timers('optimizer-inner-step', log_level=1).start(
            barrier=args.barrier_with_L1_time)
        self.optimizer.step()
        timers('optimizer-inner-step').stop()

        # Update params from main params.
        timers('optimizer-copy-main-to-model-params', log_level=1).start(
            barrier=args.barrier_with_L1_time)
        self._copy_main_params_to_model_params()
        timers('optimizer-copy-main-to-model-params').stop()

        # Successful update.
        return True, grad_norm, num_zeros_in_grad


class Float16OptimizerWithFloat16Params(MixedPrecisionOptimizer):
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
        use_contiguous_buffers_in_local_ddp: if true, the local DDP model
            is using a contiguous buffer to hold the model grads.
        fp16: if true, the model is running in fp16.
        bf16: if true, the model is running in bfloat16.
        grad_scaler: used for scaling gradients. Note that this can be
            None. This case happens when `bf16 = True` and we don't
            use any loss scale. Note that for `bf16 = True`, we can have
            a constnat gradient scaler. Also for `bf16 = False`, we
            always require a grad scaler.
        models: list of models (i.e., the virtual pipelining models). This
            is used by the distributed optimizer for mapping parameters.
    """

    def __init__(self, optimizer, clip_grad, log_num_zeros_in_grad,
                 params_have_main_grad, use_contiguous_buffers_in_local_ddp,
                 fp16, bf16, params_dtype, grad_scaler, models):

        super().__init__(
            optimizer, clip_grad, log_num_zeros_in_grad,
            params_have_main_grad, use_contiguous_buffers_in_local_ddp,
            fp16, bf16, params_dtype, grad_scaler, models)

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
                        tensor_parallel.copy_tensor_model_parallel_attributes(main_param,
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


    def _collect_main_grad_data_for_unscaling(self):

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
        for model_group, main_group in zip(self.float16_groups,
                                           self.fp32_from_float16_groups):
            for model_param, main_param in zip(model_group, main_group):
                model_data.append(model_param.data)
                main_data.append(main_param.data)
        return model_data, main_data


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
            if self.fp16:
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


class FP32Optimizer(MegatronOptimizer):

    def __init__(self, optimizer, clip_grad,
                 log_num_zeros_in_grad,
                 params_have_main_grad,
                 use_contiguous_buffers_in_local_ddp,
                 models):

        super(FP32Optimizer, self).__init__(
            optimizer, clip_grad, log_num_zeros_in_grad,
            params_have_main_grad, use_contiguous_buffers_in_local_ddp,
            models)

        self._scale = torch.cuda.FloatTensor([1.0])


    def zero_grad(self, set_to_none=True):
        """Copied from torch.optim.optimizer"""
        for group in self.optimizer.param_groups:
            _zero_grad_group_helper(group['params'], set_to_none)


    def get_loss_scale(self):
        """FP32 optimizer does not do any scaling."""
        return self._scale


    @torch.no_grad()
    def step(self, args, timers):
        """Clip gradients (if needed) and step the base optimizer.
        Always return successful since there is no overflow."""

        # Copy main_grads to grads.
        timers('optimizer-copy-to-main-grad', log_level=1).start(
            barrier=args.barrier_with_L1_time)
        if self.params_have_main_grad:
            for param_group in self.optimizer.param_groups:
                for param in param_group['params']:
                    param.grad = param.main_grad

                    # Safe to de-reference model's main_grad after copying.
                    # (If using contiguous buffers, main_grad's memory should
                    # persist and therefore should not be deallocated.)
                    if not self.use_contiguous_buffers_in_local_ddp:
                        param.main_grad = None
        timers('optimizer-copy-to-main-grad').stop()

        # Clip gradients.
        timers('optimizer-clip-main-grad', log_level=1).start(
            barrier=args.barrier_with_L1_time)
        grad_norm = None
        if self.clip_grad > 0.0:
            grad_norm = self.clip_grad_norm(self.clip_grad)
        timers('optimizer-clip-main-grad').stop()

        # count the zeros in the grads
        timers('optimizer-count-zeros', log_level=1).start(
            barrier=args.barrier_with_L1_time)
        num_zeros_in_grad = self.count_zeros() if \
                            self.log_num_zeros_in_grad else None
        timers('optimizer-count-zeros').stop()

        # Update parameters.
        timers('optimizer-inner-step', log_level=1).start(
            barrier=args.barrier_with_L1_time)
        self.optimizer.step()
        timers('optimizer-inner-step').stop()

        # No overflow for FP32 optimizer.
        return True, grad_norm, num_zeros_in_grad


    def reload_model_params(self):
        pass


    def state_dict(self):
        return self.optimizer.state_dict()


    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
