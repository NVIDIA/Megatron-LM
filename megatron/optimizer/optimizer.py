# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Megatron optimizer."""
import functools
from abc import ABC
from abc import abstractmethod
from apex.multi_tensor_apply import multi_tensor_applier
import amp_C
import torch

from megatron import get_timers, get_args
from megatron import print_rank_0
from megatron.core import mpu, tensor_parallel, parallel_state
# from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model import Float16Module
from megatron.model.module import param_is_not_shared, local_binary_reduction
from megatron.optimizer.optimizer_helper import rollback_optimizer_step
from megatron.utils import unwrap_model, nvtx_profile

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
                 check_for_nan_in_grad,
                 params_have_main_grad,
                 models):

        """Input optimizer is the base optimizer for example Adam."""
        self.optimizer = optimizer
        assert self.optimizer, 'no optimizer is provided.'
        # Set gradient clipping and logging params.
        self.clip_grad = clip_grad
        self.log_num_zeros_in_grad = log_num_zeros_in_grad
        self.check_for_nan_in_grad = check_for_nan_in_grad
        self.params_have_main_grad = params_have_main_grad

        # 'models' are retained for access to the contiguous grad buffers.
        # (see distributed optimizer)
        self.models = models

        self.partial_reduced_total_norm = torch.FloatTensor([0])
        self.local_total_norm = None
        self.dummy_overflow_buf = torch.cuda.IntTensor([0])
        self.zero_float_tensor = torch.cuda.FloatTensor([0])
        self.parameters_backup = None
        self.do_prev_step = False
        self.do_this_step = False
        self.send_next_reqs = []
        self.send_prev_reqs = []
        self.grad_norm_no_clip_recorder = 0
        self.post_validation_enabled = False

    def record_grad_norm(self, grad_norm):
        if self.post_validation_enabled:
            return
        if self.clip_grad > 0.0:
            if grad_norm is None or grad_norm > self.clip_grad:
                self.grad_norm_no_clip_recorder = 0
            else:
                self.grad_norm_no_clip_recorder += 1
            if self.grad_norm_no_clip_recorder >= 10:
                rank = parallel_state.get_pipeline_model_parallel_rank()
                print(f"{rank}: enable optimizer post validation")
                self.post_validation_enabled = True
        else:
            if grad_norm is not None:
                # optimizer state update successfully
                rank = parallel_state.get_pipeline_model_parallel_rank()
                print(f"{rank}: enable optimizer post validation")
                self.post_validation_enabled = True

    @torch.no_grad()
    def save_parameters_backup(self):
        parameters = self.get_parameters()
        backups = []
        for param in parameters:
            p = param.detach().clone()
            s1 = self.optimizer.state[param]["exp_avg"].detach().clone()
            s2 = self.optimizer.state[param]["exp_avg_sq"].detach().clone()
            backups.append((p, s1, s2))
        self.parameters_backup = backups

    @torch.no_grad()
    def rollback_parameters(self):
        parameters = self.get_parameters()
        for param, (backup, s1, s2) in zip(parameters, self.parameters_backup):
            param.copy_(backup)
            self.optimizer.state[param]["exp_avg"] = s1
            self.optimizer.state[param]["exp_avg_sq"] = s2
        self.parameters_backup = None

    def calc_local_grad_norm(self):
        grads_for_norm = self.get_main_grads_for_grad_norm()
        return self.do_clac_local_grad_norm(
            grads_for_norm,
            tensor_parallel_group=parallel_state.get_tensor_model_parallel_group())

    def get_clip_coeff_and_grad_norm(self, max_norm, norm_type=2):
        _total_norm = self.partial_reduced_total_norm
        if norm_type == torch.inf:
            _total_norm = _total_norm[0].item()
        else:
            _total_norm = _total_norm.item() ** (1.0 / norm_type)
        _clip_coeff = max_norm / (_total_norm + 1.0e-6)
        return _clip_coeff, _total_norm

    def do_clac_local_grad_norm(
        self, grads_for_norm, norm_type=2,
        tensor_parallel_group=None
    ):
        if isinstance(grads_for_norm, torch.Tensor):
            grads_for_norm = [grads_for_norm]

        # Norm parameters.
        norm_type = float(norm_type)
        total_norm = 0.0

        # Calculate norm.
        if norm_type == torch.inf:
            total_norm = max(grad.abs().max() for grad in grads_for_norm)
            total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])
            # Take max across all model-parallel GPUs.
            torch.distributed.all_reduce(total_norm_cuda,
                                         op=torch.distributed.ReduceOp.MAX,
                                         group=tensor_parallel_group)
            total_norm = total_norm_cuda
            # total_norm = total_norm_cuda[0].item()

        else:
            if norm_type == 2.0:
                self.dummy_overflow_buf.fill_(0)
                # Use apex's multi-tensor applier for efficiency reasons.
                # Multi-tensor applier takes a function and a list of list
                # and performs the operation on that list all in one kernel.
                if grads_for_norm:
                    grad_norm, _ = multi_tensor_applier(
                        amp_C.multi_tensor_l2norm,
                        self.dummy_overflow_buf,
                        [grads_for_norm],
                        False  # no per-parameter norm
                    )
                else:
                    self.zero_float_tensor.fill_(0)
                    grad_norm = self.zero_float_tensor
                # Since we will be summing across data parallel groups,
                # we need the pow(norm-type).
                total_norm = grad_norm ** norm_type

            else:
                for grad in grads_for_norm:
                    grad_norm = torch.norm(grad, norm_type)
                    total_norm += grad_norm ** norm_type

            # Sum across all model-parallel GPUs.
            torch.distributed.all_reduce(total_norm,
                                         op=torch.distributed.ReduceOp.SUM,
                                         group=tensor_parallel_group)
            # total_norm = total_norm.item() ** (1.0 / norm_type)

        self.local_total_norm = total_norm.cpu()
        return total_norm

    def partially_reduce_local_total_norm(self, clip_grad):
        return self.do_partially_reduce_local_total_norm(clip_grad)

    def do_partially_reduce_local_total_norm(self, max_norm, norm_type=2):
        # recv value from prev pipeline stage
        # self.partial_reduced_total_norm = self.recv_one(self.partial_reduced_total_norm)
        prev_clip_coeff, prev_grad_norm = self.get_clip_coeff_and_grad_norm(max_norm, norm_type)

        # reduce
        if norm_type == torch.inf:
            self.partial_reduced_total_norm = torch.maximum(self.partial_reduced_total_norm, self.local_total_norm)
        else:
            self.partial_reduced_total_norm = self.partial_reduced_total_norm + self.local_total_norm

        this_clip_coeff, grad_norm = self.get_clip_coeff_and_grad_norm(max_norm, norm_type)
        # rank = parallel_state.get_pipeline_model_parallel_rank()
        return prev_clip_coeff, this_clip_coeff, grad_norm

    def downscale_gradient(self, clip_coeff):
        assert clip_coeff < 1.0
        parameters = self.get_parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        # Grads.
        grads = []
        for param in parameters:
            if param.grad is not None:
                assert param.grad.type() == 'torch.cuda.FloatTensor'
                grads.append(param.grad.detach())
        self.dummy_overflow_buf.fill_(0)
        multi_tensor_applier(amp_C.multi_tensor_scale,
                             self.dummy_overflow_buf,
                             [grads, grads],
                             clip_coeff)

    def get_reduced_global_states(self):
        return [self.partial_reduced_total_norm]

    def send_all(self, to_next=True):
        need_send = False
        dst = None
        if to_next and not parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            need_send = True
            dst = parallel_state.get_pipeline_model_parallel_next_rank()
        if not to_next and not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            need_send = True
            dst = parallel_state.get_pipeline_model_parallel_prev_rank()
        if need_send:
            for global_state in self.get_reduced_global_states():
                send_req = torch.distributed.isend(
                    tensor=global_state,
                    dst=dst,
                    group=parallel_state.get_pipeline_model_parallel_group(),
                )
                if to_next:
                    self.send_next_reqs.append(send_req)
                else:
                    self.send_prev_reqs.append(send_req)

    def recv_all(self, from_prev=True, init_values=None):
        if from_prev:
            for req in self.send_prev_reqs:
                req.wait()
            self.send_prev_reqs = []
        else:
            for req in self.send_next_reqs:
                req.wait()
            self.send_next_reqs = []
        all_global_states = self.get_reduced_global_states()
        if init_values is None:
            init_values = [0.0] * len(all_global_states)
        for global_state, init_value in zip(all_global_states, init_values):
            self.recv_one(global_state, from_prev=from_prev, init_value=init_value)

    def recv_one(self, global_state, from_prev=True, init_value=0.0):
        if from_prev:
            if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                global_state.fill_(init_value)
            else:
                req = torch.distributed.irecv(
                    tensor=global_state,
                    src=parallel_state.get_pipeline_model_parallel_prev_rank(),
                    group=parallel_state.get_pipeline_model_parallel_group(),
                )
                req.wait()
        else:
            if not parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                req = torch.distributed.irecv(
                    tensor=global_state,
                    src=parallel_state.get_pipeline_model_parallel_next_rank(),
                    group=parallel_state.get_pipeline_model_parallel_group(),
                )
                req.wait()
        return global_state


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


    def clip_grad_norm(self, clip_grad, check_for_nan_in_grad):
        params = self.get_parameters()
        grads_for_norm = self.get_main_grads_for_grad_norm()
        return clip_grad_norm_fp32(
            params, grads_for_norm, clip_grad,
            check_for_nan_in_grad,
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



class MixedPrecisionOptimizer(MegatronOptimizer):
    """Base class for both the float-16 and the distributed optimizer.

    Arguments:
        optimizer: base optimizer such as Adam or SGD
        clip_grad: clip gradeints with this global L2 norm. Note
            that clipping is ignored if clip_grad == 0
        log_num_zeros_in_grad: return number of zeros in the gradients.
        check_for_nan_in_grad: check if gradients have a NaN.
        params_have_main_grad: flag indicating if parameters have
            a `main_grad` field. If this is set, we are assuming
            that the model parameters are store in the `main_grad`
            field instead of the typical `grad` field. This happens
            for the DDP cases where there is a continuous buffer
            holding the gradients. For example for bfloat16, we want
            to do gradient accumulation and all-reduces in float32
            and as a result we store those gradients in the main_grad.
            Note that main grad is not necessarily in float32.
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
                 check_for_nan_in_grad, params_have_main_grad,
                 fp16, bf16, params_dtype, grad_scaler, models):

        super().__init__(
            optimizer, clip_grad, log_num_zeros_in_grad,
            check_for_nan_in_grad, params_have_main_grad,
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
            self.partial_reduced_found_inf = torch.FloatTensor([0.0])
        self.fully_reduced_global_states = None

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

    def get_reduced_global_states(self):
        reduced_global_states = []
        if self.grad_scaler:
            reduced_global_states.append(self.partial_reduced_found_inf)
        reduced_global_states.extend(super().get_reduced_global_states())
        return reduced_global_states

    def get_found_inf_flag(self):
        return self.partial_reduced_found_inf.item() > 0

    def _local_unscale_main_grads_and_check_for_nan(self):
        # Collect main grads.
        main_grads = self._collect_main_grad_data_for_unscaling()
        # Reset found inf.
        self.found_inf.fill_(0.0)
        # Unscale and set found inf/nan
        torch._amp_foreach_non_finite_check_and_unscale_(
            main_grads, self.found_inf, self.grad_scaler.inv_scale)

    def partially_reduce_local_found_inf(self):
        # self.partial_reduced_found_inf = self.recv_one(self.partial_reduced_found_inf)
        # check for nan in previous rank
        prev_found_inf_flag = self.get_found_inf_flag()
        self.partial_reduced_found_inf = torch.maximum(self.partial_reduced_found_inf, self.found_inf.cpu())
        # Check for nan.
        this_found_inf_flag = self.get_found_inf_flag()
        return prev_found_inf_flag, this_found_inf_flag

    @functools.partial(nvtx_profile, name="recv_pre_step")
    @torch.no_grad()
    def recv_pre_step(self):
        # recv global states to prev rank
        self.recv_all()

    @functools.partial(nvtx_profile, name="send_pre_step")
    @torch.no_grad()
    def send_pre_step(self):
        # send global states to next rank
        self.send_all()

    @functools.partial(nvtx_profile, name="pre_step")
    @torch.no_grad()
    def pre_step(self, args, timers):
        # Copy gradients from model params to main params.
        timers('optimizer-copy-to-main-grad', log_level=1).start(
            barrier=args.barrier_with_L1_time)
        self._copy_model_grads_to_main_grads()
        timers('optimizer-copy-to-main-grad').stop()

        rank = parallel_state.get_pipeline_model_parallel_rank()

        if self.grad_scaler:
            self._local_unscale_main_grads_and_check_for_nan()
        if self.clip_grad > 0.0:
            local_norm = self.calc_local_grad_norm()

        # recv global states to prev rank
        # self.recv_all()
        self.recv_pre_step()
        prev_found_inf_flag, this_found_inf_flag = False, False
        if self.grad_scaler:
            # Unscale and check for inf/nan.
            timers('optimizer-unscale-and-check-inf', log_level=1).start(
                barrier=args.barrier_with_L1_time)
            prev_found_inf_flag, this_found_inf_flag = self.partially_reduce_local_found_inf()
            timers('optimizer-unscale-and-check-inf').stop()

        # Clip the main gradients.
        timers('optimizer-reduce-grad-norm', log_level=1).start(
            barrier=args.barrier_with_L1_time)
        grad_norm = None
        prev_clip_coeff, this_clip_coeff = 2.0, 2.0
        if self.clip_grad > 0.0:
            prev_clip_coeff, this_clip_coeff, grad_norm = self.partially_reduce_local_total_norm(self.clip_grad)
        timers('optimizer-reduce-grad-norm').stop()

        # send global states to next rank
        # self.send_all()
        self.send_pre_step()

        def can_local_step(found_inf_flag, clip_coeff):
            if self.grad_scaler:
                if found_inf_flag:
                    return False
            if self.clip_grad > 0.0:
                is_nan = clip_coeff == float('inf') or \
                         clip_coeff == -float('inf') or \
                         clip_coeff != clip_coeff
                assert not is_nan
                if is_nan or clip_coeff < 1.0:
                    return False
            return True
        self.do_prev_step = can_local_step(prev_found_inf_flag, prev_clip_coeff)
        self.do_this_step = can_local_step(this_found_inf_flag, this_clip_coeff)
        # print(f"{rank} pre_step: {prev_found_inf_flag}, {prev_clip_coeff} -> {self.do_prev_step} | {this_found_inf_flag}, {this_clip_coeff} -> {self.do_this_step}")
        timers('optimizer-local-step', log_level=1).start(
            barrier=args.barrier_with_L1_time)
        if self.do_this_step:
            # Step the optimizer.
            if args.enable_exactly_numeric_match:
                self.save_parameters_backup()  # for exactly match
            self.optimizer.step()
        timers('optimizer-local-step').stop()

        # Update params from main params.
        timers('optimizer-copy-main-to-model-params', log_level=1).start(
            barrier=args.barrier_with_L1_time)
        if self.do_this_step:
            self._copy_main_params_to_model_params()
        timers('optimizer-copy-main-to-model-params').stop()

    def prepare_fully_reduced_global_states(self):
        self.fully_reduced_global_states = {}
        if self.grad_scaler:
            found_inf_flag = self.get_found_inf_flag()
            self.fully_reduced_global_states["found_inf_flag"] = found_inf_flag
        if self.clip_grad > 0.0:
            clip_coeff, grad_norm = self.get_clip_coeff_and_grad_norm(self.clip_grad)
            self.fully_reduced_global_states["clip_coeff"] = clip_coeff
            self.fully_reduced_global_states["grad_norm"] = grad_norm

    @functools.partial(nvtx_profile, name="recv_post_validation")
    @torch.no_grad()
    def recv_post_validation(self):
        self.recv_all(from_prev=False)
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            self.prepare_fully_reduced_global_states()

    @functools.partial(nvtx_profile, name="send_post_validation")
    @torch.no_grad()
    def send_post_validation(self):
        if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            self.prepare_fully_reduced_global_states()
        self.send_all(to_next=False)

    @functools.partial(nvtx_profile, name="post_validation")
    @torch.no_grad()
    def post_validation(self):
        rank = parallel_state.get_pipeline_model_parallel_rank()
        if self.grad_scaler:
            # found_inf_flag = self.get_found_inf_flag()
            found_inf_flag = self.fully_reduced_global_states["found_inf_flag"]
            self.grad_scaler.update(found_inf_flag)
            if found_inf_flag:
                if self.do_this_step:
                    print("found inf rollback")
                    rollback_optimizer_step(self.optimizer)
                    if get_args().enable_exactly_numeric_match:
                        self.rollback_parameters()  # for exactly match
                    self._copy_main_params_to_model_params()
                return False, None, self.do_this_step, False
        succeed = True
        grad_norm = None
        if self.clip_grad > 0.0:
            # clip_coeff, grad_norm = self.get_clip_coeff_and_grad_norm(self.clip_grad)
            clip_coeff, grad_norm = self.fully_reduced_global_states["clip_coeff"], self.fully_reduced_global_states["grad_norm"]
            is_nan = clip_coeff == float('inf') or \
                     clip_coeff == -float('inf') or \
                     clip_coeff != clip_coeff
            assert not is_nan
            if clip_coeff < 1.0:
                if self.do_this_step:
                    print(f"{rank} grad rollback")
                    rollback_optimizer_step(self.optimizer)
                    if get_args().enable_exactly_numeric_match:
                        self.rollback_parameters()  # for exactly match
                if get_args().enable_exactly_numeric_match:
                    clip_coeff = round(clip_coeff, 4)  # for exactly match
                self.downscale_gradient(clip_coeff)
                self.optimizer.step()
                self._copy_main_params_to_model_params()
                succeed = False
            else:
                assert self.do_this_step
        else:
            assert self.do_this_step

        # updated, grad_norm, rollback, succeed
        return True, grad_norm, not succeed and self.do_this_step, succeed

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
            grad_norm = self.clip_grad_norm(self.clip_grad,
                                            self.check_for_nan_in_grad)
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
        check_for_nan_in_grad: check if gradients have a NaN.
        params_have_main_grad: flag indicating if parameters have
            a `main_grad` field. If this is set, we are assuming
            that the model parameters are store in the `main_grad`
            field instead of the typical `grad` field. This happens
            for the DDP cases where there is a continuous buffer
            holding the gradients. For example for bfloat16, we want
            to do gradient accumulation and all-reduces in float32
            and as a result we store those gradients in the main_grad.
            Note that main grad is not necessarily in float32.
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
                 check_for_nan_in_grad, params_have_main_grad, fp16, bf16,
                 params_dtype, grad_scaler, models):

        super().__init__(
            optimizer, clip_grad, log_num_zeros_in_grad,
            check_for_nan_in_grad, params_have_main_grad,
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

        # For fp32 grads, we need to reset the grads to main grad.
        if self.params_have_main_grad:
            for model_group in self.fp32_from_fp32_groups:
                for model_param in model_group:
                    model_param.grad = model_param.main_grad


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
                 check_for_nan_in_grad,
                 params_have_main_grad,
                 models):

        super(FP32Optimizer, self).__init__(
            optimizer, clip_grad, log_num_zeros_in_grad,
            check_for_nan_in_grad, params_have_main_grad,
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

        timers('optimizer-copy-to-main-grad').stop()

        # Clip gradients.
        timers('optimizer-clip-main-grad', log_level=1).start(
            barrier=args.barrier_with_L1_time)
        grad_norm = None
        if self.clip_grad > 0.0:
            grad_norm = self.clip_grad_norm(self.clip_grad,
                                            self.check_for_nan_in_grad)
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
