# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import logging
from contextlib import contextmanager
from typing import Optional

import torch

from .. import parallel_state
from ..config_logger import has_config_logger_enabled, log_config_to_disk
from ..fp8_utils import is_float8tensor
from ..process_groups_config import GradCommProcessGroups, ModelCommProcessGroups
from ..transformer.cuda_graphs import is_graph_capturing
from ..transformer.transformer_config import TransformerConfig
from ..utils import log_single_rank
from .data_parallel_base import _BaseDataParallel
from .distributed_data_parallel_config import DistributedDataParallelConfig
from .param_and_grad_buffer import _ParamAndGradBuffer, partition_buckets

logger = logging.getLogger(__name__)


class DistributedDataParallel(_BaseDataParallel):
    """
    DDP wrapper which stores grads in contiguous buffers. Also has option of overlapping
    communication with backprop computation by breaking up full model's gradients into smaller
    buckets and running all-reduce / reduce-scatter on each bucket asynchronously. This class
    also provides the option to do the gradient accumulation in a type other than the param type
    (e.g., fp32 for a bf16 model).

    Args:
        config: Transformer config object.
        ddp_config: DistributedDataParallel config object.
        module: Underlying model.
        disable_bucketing: If true, force assign all parameters to a single bucket. If false,
            use standard bucketing policy: assign parameters to smaller buckets and all-reduce
            per bucket _if_ overlap_grad_reduce is True and pp_rank is 0.
        grad_comm_pgs: Optional gradient communication process groups.
        model_comm_pgs: Optional model parallel communication process groups.

    """

    def __init__(
        self,
        config: TransformerConfig,
        ddp_config: DistributedDataParallelConfig,
        module: torch.nn.Module,
        disable_bucketing: bool = False,
        grad_comm_pgs: Optional[GradCommProcessGroups] = None,
        model_comm_pgs: Optional[ModelCommProcessGroups] = None,
    ):
        super().__init__(config=config, module=module)
        if has_config_logger_enabled(config):
            log_config_to_disk(config, locals(), prefix=type(self).__name__)

        self.module = module

        # If bucket_size is not provided as an input, use sane default.
        # If using very large dp_sizes, make buckets larger to ensure that chunks used in NCCL
        # ring-reduce implementations are large enough to remain bandwidth-bound rather than
        # latency-bound.
        if ddp_config.bucket_size is None:
            ddp_config.bucket_size = max(
                40000000, 1000000 * parallel_state.get_data_parallel_world_size()
            )
        # Set bucket_size to infinity if overlap_grad_reduce is False.
        if not ddp_config.overlap_grad_reduce:
            ddp_config.bucket_size = None

        self.ddp_config = ddp_config
        log_single_rank(
            logger,
            logging.INFO,
            f'Setting up DistributedDataParallel with config {self.ddp_config}',
        )
        if grad_comm_pgs is None and model_comm_pgs is None:
            self.dp_group = parallel_state.get_data_parallel_group(
                with_context_parallel=False, partial_data_parallel=False
            )
            self.dp_cp_group = parallel_state.get_data_parallel_group(
                with_context_parallel=True, partial_data_parallel=False
            )
            self.intra_dp_cp_group = parallel_state.get_data_parallel_group(
                with_context_parallel=True, partial_data_parallel=True
            )
            self.expt_dp_group = parallel_state.get_expert_data_parallel_group()
            self.intra_expt_dp_group = parallel_state.get_expert_data_parallel_group(
                partial_expert_data_parallel=True
            )
            if self.ddp_config.num_distributed_optimizer_instances > 1:
                self.inter_dist_opt_group = (
                    parallel_state.get_inter_distributed_optimizer_instance_group()
                )

            self.pp_group = parallel_state.get_pipeline_model_parallel_group()
            self.ep_group = parallel_state.get_expert_model_parallel_group()
        elif grad_comm_pgs is not None and model_comm_pgs is not None:
            # 1. dp group - this is always required
            if not hasattr(grad_comm_pgs, 'dp'):
                raise ValueError("dp process group is required but not provided in grad_comm_pgs")
            self.dp_group = grad_comm_pgs.dp

            # 2. dp_cp group:
            # - If provided in grad_comm_pgs, use it
            # - Otherwise check context_parallel_size
            #   - If cp_size is 1, use same as dp
            #   - If cp_size > 1, raise error as dp_cp is needed
            if hasattr(grad_comm_pgs, 'dp_cp'):
                self.dp_cp_group = grad_comm_pgs.dp_cp
            else:
                cp_size = getattr(config, 'context_parallel_size', 1)
                if cp_size == 1:
                    # If no context parallelism, dp_cp is same as dp
                    self.dp_cp_group = self.dp_group
                else:
                    raise ValueError(
                        "dp_cp process group is required when context_parallel_size > 1 "
                        "but not provided in grad_comm_pgs"
                    )

            # 3. Handle expert data parallel group
            if hasattr(grad_comm_pgs, 'expt_dp'):
                self.expt_dp_group = grad_comm_pgs.expt_dp
            else:
                # Create a new group with just the current rank
                log_single_rank(
                    logger,
                    logging.WARNING,
                    "No expert data parallel group provided in grad_comm_pgs, "
                    "creating a new one with just the current rank",
                )
                # Ideally we dont want any expt_dp_group if not using expt_dp
                # but downstream code expects.
                # this is used to check size and calculate scaling factor.
                self.expt_dp_group = torch.distributed.new_group(
                    ranks=[torch.distributed.get_rank()]
                )

            # 4. Handle intra_dp_cp, intra_expt_dp, and inter_dist_opt
            #    based on optimizer instances:
            if self.ddp_config.num_distributed_optimizer_instances == 1:
                # With a single optimizer instance:
                # - intra_dp_cp is same as dp_cp
                # - intra_expt_dp is same as expt_dp
                # - inter_dist_opt is not needed
                self.intra_dp_cp_group = self.dp_cp_group
                self.intra_expt_dp_group = self.expt_dp_group
            else:
                # With multiple optimizer instances, both groups must be provided
                if not (
                    hasattr(grad_comm_pgs, 'intra_dp_cp')
                    and hasattr(grad_comm_pgs, 'intra_expt_dp')
                    and hasattr(grad_comm_pgs, 'inter_dist_opt')
                ):
                    raise ValueError(
                        "intra_dp_cp, intra_expt_dp, and inter_dist_opt "
                        "process groups are required when using multiple optimizer "
                        "instances (>1) but not provided in grad_comm_pgs"
                    )
                self.intra_dp_cp_group = grad_comm_pgs.intra_dp_cp
                self.intra_expt_dp_group = grad_comm_pgs.intra_expt_dp
                self.inter_dist_opt_group = grad_comm_pgs.inter_dist_opt

            # 5. pp and ep group
            if not all([hasattr(model_comm_pgs, 'pp'), hasattr(model_comm_pgs, 'ep')]):
                raise ValueError(
                    "pp and ep process groups are required but not provided in model_comm_pgs"
                )
            self.pp_group = model_comm_pgs.pp
            self.ep_group = model_comm_pgs.ep

        else:
            raise ValueError(
                "Grad and model comm process groups must be provided or both must be None"
            )

        # Turn off bucketing if we are on a pipeline stage that is not the first (since
        # data-parallel communication on these stages is not on the critical path), or if
        # disable_bucketing is True (e.g., we might not want to break up model parameters
        # into buckets for model chunks after the first in the interleaved schedule).
        self.bucket_size = self.ddp_config.bucket_size
        if isinstance(self.pp_group, list):
            pp_rank = self.pp_group[0].rank()
        else:
            pp_rank = self.pp_group.rank()
        if pp_rank > 0:
            self.bucket_size = None
        if disable_bucketing:
            self.bucket_size = None

        self.param_to_bucket_group = {}

        # Group parameters by their gradient type.
        param_to_name = {}
        dense_params = []
        expert_parallel_params = []
        self.params_with_grad = []
        for name, param in self.module.named_parameters():
            if not param.requires_grad:
                continue

            # Track params with grad to enable direct setting
            # of param.grad_added_to_main_grad
            self.params_with_grad.append(param)

            param.grad_added_to_main_grad = False
            param_to_name[param] = name

            if getattr(param, 'allreduce', True):
                dense_params.append(param)
            else:
                expert_parallel_params.append(param)

        def _allocate_buffers_for_parameters(
            input_params, data_parallel_group, gradient_scaling_factor
        ):
            param_and_grad_dtype_to_params = {}
            param_and_grad_dtype_to_offsets = {}
            param_and_grad_dtype_to_indices = {}

            # Group parameters by their gradient type.
            for param in input_params:
                assert param.requires_grad

                param_dtype = param.dtype
                if is_float8tensor(param):
                    # Currently TE's Float8Tensor is a wrapper of torch.Tensor. It has a "fake"
                    # dtype (usually a higher precision dtype such as bfloat16), but its actual
                    # data is stored in the form of a torch uint8 tensor within the Float8Tensor's
                    # ".data" attribute. Therefore, when creating the param buffer for fp8 params,
                    # it is necessary to use torch.uint8, not the "fake" dtype got from
                    # "param.dtype".
                    param_dtype = torch.uint8
                grad_dtype = torch.float if self.ddp_config.grad_reduce_in_fp32 else param.dtype

                params = param_and_grad_dtype_to_params.get((param_dtype, grad_dtype), [])
                params.append(param)
                param_and_grad_dtype_to_params[(param_dtype, grad_dtype)] = params

                # Get the index of each param among the params with same dtype, if a param is fp8,
                # use its "fake" high precision dtype to find which params have same dtype with it.
                # For example:
                #     Case 1:
                #         params = [p1(bf16), p2(bf16), p3(bf16), p4(bf16)]
                #         param_and_grad_dtype_to_indices = {
                #             (torch.bfloat16, torch.float32): [0, 1, 2, 3],
                #         }
                #     Case 2:
                #         params = [p1(bf16), p2(fp8), p3(fp8), p4(bf16)]
                #         param_and_grad_dtype_to_indices = {
                #             (torch.bfloat16, torch.float32): [0, 3],
                #             (torch.uint8, torch.float32): [1, 2],
                #         }
                # We need these indices to load a non-native-fp8 checkpoint in native-fp8 mode.
                offset = param_and_grad_dtype_to_offsets.get((param.dtype, grad_dtype), 0)
                param_and_grad_dtype_to_offsets[(param.dtype, grad_dtype)] = offset + 1
                indices = param_and_grad_dtype_to_indices.get((param_dtype, grad_dtype), [])
                indices.append(offset)
                param_and_grad_dtype_to_indices[(param_dtype, grad_dtype)] = indices

            if not config.calculate_per_token_loss:
                target_gradient_scaling_factor = 1.0 / self.dp_cp_group.size()
                if self.ddp_config.average_in_collective:
                    if self.ddp_config.num_distributed_optimizer_instances == 1:
                        # Collective is averaging gradients in collective with data_parallel_group.
                        assert (
                            gradient_scaling_factor / data_parallel_group.size()
                            == target_gradient_scaling_factor
                        )
                    else:
                        # For non-expert parameters, gradient_scaling_factor is 1.
                        # For expert parameters, gradient_scaling_factor is edp_size/dp_size.
                        assert (gradient_scaling_factor == 1) or (
                            gradient_scaling_factor
                            == (self.expt_dp_group.size() / self.dp_cp_group.size())
                        )
                else:
                    assert gradient_scaling_factor == target_gradient_scaling_factor

            # Allocate the grad buffers and map the grads.
            buffers = []
            for (param_dtype, grad_dtype), params in param_and_grad_dtype_to_params.items():
                buffers.append(
                    _ParamAndGradBuffer(
                        self.ddp_config,
                        param_dtype,
                        grad_dtype,
                        params,
                        data_parallel_group,
                        self.bucket_size,
                        param_to_name,
                        gradient_scaling_factor,
                        param_and_grad_dtype_to_indices[(param_dtype, grad_dtype)],
                        self.ddp_config.nccl_ub,
                    )
                )

            # In some scenarios, we want to put buckets from different buffers into a group so that
            # their communication can be aggregated. For example, when there are both fp8 buffers
            # and bf16 buffers in the model and vpp is enabled, each model chunk will have an fp8
            # bucket and a bf16 bucket, which doubles the number of communication kernels, and
            # because of the use of CUDA_DEVICE_MAX_CONNECTIONS=1, having multiple back-to-back
            # communications will prevent the overlap of the communication kernels with computation
            # kernels.
            # If bucketing is explicitly disabled, then put all buckets in a buffer into a single
            # bucket group.
            bucket_groups = partition_buckets(buffers, force_single_bucket_group=disable_bucketing)

            if self.ddp_config.num_distributed_optimizer_instances > 1:
                assert (
                    self.ddp_config.use_distributed_optimizer
                ), 'Partial DistOpt cannot be used without DistOpt'
                communication_stream = torch.cuda.Stream(device=torch.cuda.current_device())
                for bucket_group in bucket_groups:
                    bucket_group.inter_distributed_optimizer_instance_group = (
                        self.inter_dist_opt_group
                    )
                    bucket_group.communication_stream = communication_stream

            # Set `next_param_gather_bucket_group` for different bucket groups by iterating through
            # buckets in reverse order (since all-gathers happen in reverse order of buckets).
            if self.ddp_config.use_distributed_optimizer and self.ddp_config.overlap_param_gather:
                num_bucket_groups = len(bucket_groups)
                for i in range(1, num_bucket_groups):
                    bucket_groups[num_bucket_groups - i].next_param_gather_bucket_group = (
                        bucket_groups[num_bucket_groups - i - 1]
                    )

            # Create map from param to bucket group, used in pre_hook.
            for bucket_group in bucket_groups:
                for bucket in bucket_group.buckets:
                    for param in bucket.params_list:
                        self.param_to_bucket_group[param] = bucket_group

            return buffers, bucket_groups

        if config.calculate_per_token_loss:
            assert (
                not self.ddp_config.average_in_collective
            ), "Cannot average in collective when calculating per-token loss!"
            gradient_scaling_factor = 1.0
            expert_gradient_scaling_factor = 1.0
        else:
            # The goal is to scale reduced gradients by 1/dp_size.
            # This can be achieved in two ways:
            #
            # Case 1: average_in_collective=True
            # - Non-expert parameters:
            #   1. No pre-scaling (gradient_scaling_factor=1.0)
            #   2. Do average reduction over dp group (equals to sum then divide by dp_size)
            #   3. Final result is scaled by 1/dp_size as desired
            #
            # - Expert parameters:
            #   1. Scale by edp_size/dp_size before reduction
            #   2. Do average reduction over edp group (equals to sum then divide by edp_size)
            #   3. Resulted scaling: (edp_size/dp_size) * (1/edp_size) = 1/dp_size as desired
            #   (edp_size = expert data parallel world size)
            #
            # Case 2: average_in_collective=False
            # - Both expert and non-expert parameters:
            #   1. Scale gradients by 1/dp_size before reduction
            #   2. Do sum reduction across data parallel ranks
            #   3. Final result is scaled by 1/dp_size as desired
            if self.ddp_config.average_in_collective:
                gradient_scaling_factor = 1.0
                expert_gradient_scaling_factor = self.expt_dp_group.size() / self.dp_cp_group.size()
            else:
                data_parallel_world_size = self.dp_cp_group.size()

                gradient_scaling_factor = 1.0 / data_parallel_world_size
                expert_gradient_scaling_factor = 1.0 / data_parallel_world_size

        # Allocate the param+grad buffers for dense params' grads.
        self.buffers, self.bucket_groups = _allocate_buffers_for_parameters(
            dense_params, self.intra_dp_cp_group, gradient_scaling_factor=gradient_scaling_factor
        )

        # Allocate separate param+grad buffers for expert parallel params' grads.
        self.expert_parallel_buffers, self.expert_parallel_bucket_groups = (
            _allocate_buffers_for_parameters(
                expert_parallel_params,
                self.intra_expt_dp_group,
                gradient_scaling_factor=expert_gradient_scaling_factor,
            )
        )

        # Delete references to weight_tensor if they exist since we don't want two parameter copies
        # if we re-mapped parameters (which happens when we use the distributed optimizer).
        # This is a temporary workaround around a TE bug that is fixed with
        # https://github.com/NVIDIA/TransformerEngine/pull/719.
        if self.ddp_config.use_distributed_optimizer:

            @torch.no_grad()
            def unmap_weight_tensor(m):
                if hasattr(m, 'weight_tensor'):
                    m.weight_tensor = None

            self.module.apply(unmap_weight_tensor)

        # Register backward hook.
        # Accumulation function for the gradients need to be stored so they
        # don't go out of scope.
        self.grad_accs = []
        for param in self.module.parameters():
            if param.requires_grad:
                # Expand so we get access to grad_fn.
                param_tmp = param.expand_as(param)
                # Get the gradient accumulator function.
                grad_acc = param_tmp.grad_fn.next_functions[0][0]
                grad_acc.register_hook(self._make_backward_post_hook(param))
                self.grad_accs.append(grad_acc)

        self.use_forward_hook = (
            self.ddp_config.use_distributed_optimizer and self.ddp_config.overlap_param_gather
        )
        self.remove_forward_pre_hook_handles = {}
        if self.use_forward_hook:
            self.enable_forward_pre_hook()
        self.overlap_param_gather_with_optimizer_step = False

    def enable_forward_pre_hook(self):
        """
        Enable forward pre-hooks needed for param all-gather overlap with forward compute.
        """
        assert self.use_forward_hook
        assert len(self.remove_forward_pre_hook_handles) == 0
        # Register forward pre-hook for all sub-modules.
        for module in self.module.modules():
            self.remove_forward_pre_hook_handles[module] = module.register_forward_pre_hook(
                self._make_forward_pre_hook()
            )

    def disable_forward_pre_hook(self, param_sync: bool = True):
        """
        Disable forward pre-hooks needed for param all-gather overlap with forward compute.
        Skip synchronous param all-gather if `param_sync` is False.
        """
        assert self.use_forward_hook
        # De-register forward pre-hook for all sub-modules.
        for module in self.module.modules():
            assert self.remove_forward_pre_hook_handles[module] is not None
            self.remove_forward_pre_hook_handles[module].remove()
            del self.remove_forward_pre_hook_handles[module]
        assert len(self.remove_forward_pre_hook_handles) == 0

        # Force synchronize parameters.
        if param_sync:
            self.start_param_sync(force_sync=True)

    def _make_forward_pre_hook(self):
        """
        Create a forward pre-hook to wait on all-gather handles when necessary (i.e.,
        when a module uses a parameter in a bucket with a still incomplete all-gather).
        """

        def hook(module, *unused):
            assert (
                self.use_forward_hook
            ), "Should use pre-hook only when overlap_param_gather is True"

            if is_graph_capturing():
                return

            # Make sure all parameters in this module have been all-gathered as necessary.
            for param in module.parameters(recurse=False):
                # Skip parameters without an associated buffer (such parameters have a
                # .requires_grad field equal to False).
                if param not in self.param_to_bucket_group:
                    continue
                assert param.requires_grad

                # If aligning param all-gather across pipeline stages, all-gather is dispatched
                # by start_param_sync calls in core/pipeline_parallelism/schedules.py.
                # If overlapping param all-gather with optimizer step, then all-gather has
                # already been dispatched in optimizer step.
                skip_next_bucket_dispatch = (
                    self.ddp_config.align_param_gather
                    or self.overlap_param_gather_with_optimizer_step
                )
                self.param_to_bucket_group[param].finish_param_sync(
                    skip_next_bucket_dispatch=skip_next_bucket_dispatch
                )

        return hook

    def _make_backward_post_hook(self, param: torch.nn.Parameter):
        """
        Creates a backward post-hook to dispatch an all-reduce / reduce-scatter when
        ready (i.e., when all grads in a bucket have been computed in all microbatches
        in a batch).
        """

        def hook(*unused):
            if is_graph_capturing():
                return

            if param in self.param_to_bucket_group:
                assert param.requires_grad
                if self.ddp_config.overlap_grad_reduce:
                    assert (
                        param.grad is not None
                    ), 'param.grad being None is not safe when overlap_grad_reduce is True'
                if param.grad is not None and (
                    not param.grad_added_to_main_grad or getattr(param, 'zero_out_wgrad', False)
                ):
                    param.main_grad.add_(param.grad.data)
                param.grad = None

                if self.ddp_config.overlap_grad_reduce:
                    self.param_to_bucket_group[param].register_grad_ready(param)

        return hook

    @contextmanager
    def no_sync(self):
        """
        Context manager that turns off gradient synchronization.
        """
        for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
            bucket_group.is_last_microbatch = False
        try:
            yield
        finally:
            for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
                bucket_group.is_last_microbatch = True

    def start_param_sync(self, *unused, force_sync: bool = False, force_dispatch: bool = False):
        """
        Initiates param sync (all-gather) communication operations for all model parameters.

        By default, when overlap_param_gather is set to True, dispatches asynchronous communication
        calls; when overlap_param_gather is set to False, calls synchronous communication
        ops. Can override this default behavior using flags below.

        Args:
            force_sync (bool, optional): force synchronous collective regardless of
                other settings.
            force_dispatch (bool, optional): force dispatch regardless of other settings.
        """
        if not force_sync:
            # If overlapping param AG with optimizer step, AG should not be dispatched again
            # in forward_backward_step.
            if self.overlap_param_gather_with_optimizer_step and not force_dispatch:
                return

        for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
            bucket_group.start_param_sync(force_sync=force_sync)
            # For MXFP8 params, we need to copy the all-gathered param data from the buffer to
            # the param.data, since param buffer is not mapped to model params for MXFP8 case.
            # The paramaters are cast from bf16 to MXFP8 during copy.
            if self.ddp_config.reuse_grad_buf_for_mxfp8_param_ag:
                assert (
                    not self.ddp_config.overlap_param_gather
                ), "MXFP8 param currently does not support DP AG overlap."
                for bucket in bucket_group.buckets:
                    for param in bucket.params:
                        param_start, param_end = bucket.param_to_index[param]
                        param_slice = bucket.param_data.view(-1)[param_start:param_end]
                        param.data.copy_(param_slice.view(param.data.shape))
                    # All-gathered params are not needed after being copied to param.data.
                    # Zero out the grad buffer (shared with param buffer) for gradient accumulation.
                    bucket.grad_data.zero_()

    def start_grad_sync(self, *unused):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operations
        for all model gradients.

        When overlap_grad_reduce is set to True, dispatches asynchronous communication
        calls. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
            bucket_group.start_grad_sync()

    def finish_grad_sync(self):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operations
        for all model gradients.

        When overlap_grad_reduce is set to True, waits for asynchronous communication
        calls to complete. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
            bucket_group.finish_grad_sync()

    def scale_gradients(self, scaling_factor: float):
        """Scale all gradients inside the buffers by `scaling_factor`."""
        for buffer in self.buffers + self.expert_parallel_buffers:
            buffer.scale_gradients(scaling_factor)

    def zero_grad_buffer(self):
        """
        Zeros out all grad buffers. Needs to be called at the beginning of each
        training iteration.
        """
        if not getattr(self.config, 'external_cuda_graph', False):
            # Don't reset grad_added_to_main_grad when CUDA Graph is used.
            # Because in CUDA Graph it no longer has the opportunity to set it back
            # to True, and there will be a double-GA.
            for param in self.params_with_grad:
                param.grad_added_to_main_grad = False
        for buffer in self.buffers + self.expert_parallel_buffers:
            buffer.reset()
        for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
            bucket_group.reset()

    def broadcast_params(self):
        """
        Syncs parameters across all DP ranks.
        """
        for param in self.module.parameters():
            is_expert_parallel = not getattr(param, 'allreduce', True)

            if is_expert_parallel:
                data_parallel_group = self.expt_dp_group
            else:
                data_parallel_group = self.dp_cp_group
            torch.distributed.broadcast(
                param.data,
                src=torch.distributed.get_global_rank(data_parallel_group, 0),
                group=data_parallel_group,
            )
