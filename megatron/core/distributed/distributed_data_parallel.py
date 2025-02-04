# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import functools
import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Tuple

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

from megatron.core.transformer.transformer_layer import TransformerLayer

from .. import parallel_state
from ..config_logger import has_config_logger_enabled, log_config_to_disk
from ..transformer.module import MegatronModule
from ..transformer.transformer_config import TransformerConfig
from ..utils import is_float8tensor, log_single_rank
from .distributed_data_parallel_config import DistributedDataParallelConfig
from .param_and_grad_buffer import _ParamAndGradBuffer, partition_buckets
from .fsdp.param_and_grad_buffer import ParamAndGradBuffer, BucketingPolicy, GradReducePipeline, AllGatherPipeline, PrefetchOrder

logger = logging.getLogger(__name__)


class DistributedDataParallel(MegatronModule):
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

    """

    def __init__(
        self,
        config: TransformerConfig,
        ddp_config: DistributedDataParallelConfig,
        module: torch.nn.Module,
        fsdp_modules: List[torch.nn.Module] = [TransformerLayer],
        disable_bucketing: bool = False,
    ):
        super().__init__(config=config)
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

        # Turn off bucketing if we are on a pipeline stage that is not the first (since
        # data-parallel communication on these stages is not on the critical path), or if
        # disable_bucketing is True (e.g., we might not want to break up model parameters
        # into buckets for model chunks after the first in the interleaved schedule).
        self.bucket_size = self.ddp_config.bucket_size
        if parallel_state.get_pipeline_model_parallel_rank() > 0:
            self.bucket_size = None
        if disable_bucketing:
            self.bucket_size = None

        self.param_to_bucket_group = {}

        if config.calculate_per_token_loss:
            gradient_scaling_factor = 1.0
            expert_gradient_scaling_factor = 1.0
        else:
            if self.ddp_config.average_in_collective:
                gradient_scaling_factor = 1.0
                expert_gradient_scaling_factor = (
                    1.0 / parallel_state.get_expert_model_parallel_world_size()
                )
            else:
                data_parallel_world_size = parallel_state.get_data_parallel_world_size(
                    with_context_parallel=True
                )
                gradient_scaling_factor = 1.0 / data_parallel_world_size
                expert_gradient_scaling_factor = 1.0 / data_parallel_world_size

        if self.ddp_config.with_megatron_fsdp_code_path:
            self.fsdp_modules = fsdp_modules
            self.master_weights = True
            self.data_parallel_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
            self.expert_data_parallel_group = parallel_state.get_data_modulo_expert_parallel_group(with_context_parallel=True)
            if self.ddp_config.data_parallel_sharding_strategy == "MODEL_AND_OPTIMIZER_STATES":
                assert self.ddp_config.overlap_param_gather
            if self.ddp_config.data_parallel_sharding_strategy in [
                "OPTIMIZER_STATES_AND_GRADS",
                "MODEL_AND_OPTIMIZER_STATES",
            ]:
                assert self.ddp_config.overlap_grad_reduce
            self._init_fsdp_param_and_grad_buffer(
                gradient_scaling_factor, expert_gradient_scaling_factor
            )
            self.register_forward_backward_hooks()
        else:
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
                    target_gradient_scaling_factor = 1.0 / parallel_state.get_data_parallel_world_size(
                        with_context_parallel=True
                    )
                    if self.ddp_config.average_in_collective:
                        # Collective is averaging gradients in collective with data_parallel_group.
                        assert (
                            gradient_scaling_factor
                            / torch.distributed.get_world_size(group=data_parallel_group)
                            == target_gradient_scaling_factor
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

            # Allocate the param+grad buffers for dense params' grads.
            self.buffers, self.bucket_groups = _allocate_buffers_for_parameters(
                dense_params,
                parallel_state.get_data_parallel_group(with_context_parallel=True),
                gradient_scaling_factor=gradient_scaling_factor,
            )

            # Allocate separate param+grad buffers for expert parallel params' grads.
            self.expert_parallel_buffers, self.expert_parallel_bucket_groups = (
                _allocate_buffers_for_parameters(
                    expert_parallel_params,
                    parallel_state.get_data_modulo_expert_parallel_group(with_context_parallel=True),
                    gradient_scaling_factor=expert_gradient_scaling_factor,
                )
            )

        # Delete references to weight_tensor if they exist since we don't want two parameter copies
        # if we re-mapped parameters (which happens when we use the distributed optimizer).
        # This is a temporary workaround around a TE bug that is fixed with
        # https://github.com/NVIDIA/TransformerEngine/pull/719.
        if self.ddp_config.use_distributed_optimizer or (self.ddp_config.with_megatron_fsdp_code_path and self.ddp_config.data_parallel_sharding_strategy in [
            "OPTIMIZER_STATES",
            "MODEL_AND_OPTIMIZER_STATES",
            "OPTIMIZER_STATES_AND_GRADS",
        ]):

            @torch.no_grad()
            def unmap_weight_tensor(m):
                if hasattr(m, 'weight_tensor'):
                    m.weight_tensor = None

            self.module.apply(unmap_weight_tensor)

        if self.ddp_config.with_megatron_fsdp_code_path:
            pass
        else:
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

    def _init_fsdp_param_and_grad_buffer(self, gradient_scaling_factor, expert_gradient_scaling_factor):
        # Initialize the param and grad buffer.
        self.data_parallel_sharding_strategy = self.ddp_config.data_parallel_sharding_strategy
        self.param_to_name = {p: name for name, p in self.module.named_parameters()}
        if self.config.average_wgrad_by_dgrad_scale:
            gradient_scaling_factor = None
            expert_gradient_scaling_factor = None
        else:
            gradient_scaling_factor = 1. / torch.distributed.get_world_size(group=self.data_parallel_group)
            expert_gradient_scaling_factor = 1. / torch.distributed.get_world_size(group=self.expert_data_parallel_group)
        self.param_and_grad_buffer = ParamAndGradBuffer(
            self.ddp_config,
            self.module,
            params=list(self.module.parameters()),
            bucketing_policy=BucketingPolicy(
                guide_bucket_size=self.bucket_size,
                fsdp_modules=self.fsdp_modules
                if self.data_parallel_sharding_strategy == "MODEL_AND_OPTIMIZER_STATES"
                else [],
                data_parallel_sharding_strategy=self.data_parallel_sharding_strategy,
            ),
            data_parallel_group=self.data_parallel_group,
            expert_data_parallel_group=self.expert_data_parallel_group,
            param_to_name=self.param_to_name,
            preserve_fp32_weights=self.master_weights,
            grad_reduce_in_fp32=self.ddp_config.grad_reduce_in_fp32,
            gradient_scaling_factor=gradient_scaling_factor,
            expert_gradient_scaling_factor=expert_gradient_scaling_factor,
            device=torch.cuda.current_device(),
        )
        self.buffers = [self.param_and_grad_buffer]

        self.grad_reduce_cuda_stream = torch.cuda.Stream()
        self.all_gather_cuda_stream = torch.cuda.Stream()

        # Initialize the reduce-scatter pipeline.
        self.grad_reduce_pipeline = GradReducePipeline(
            self.param_and_grad_buffer, cuda_stream=self.grad_reduce_cuda_stream
        )

        # Initialize the all-gather pipeline.
        self.all_gather_pipeline = AllGatherPipeline(
            self.param_and_grad_buffer, cuda_stream=self.all_gather_cuda_stream
        )

        # Determine if we should delay the gradient reduction.
        self.is_delay_grad_reduce = self.data_parallel_sharding_strategy in [
            "NO_OP",
            "OPTIMIZER_STATES",
        ]

        self.reduce_scatter_queue_capacity = 400_000_000
        if self.ddp_config.fp8_param_gather:
            self.all_gather_prefetch_capacity = 600_000_000
        else:
            self.all_gather_prefetch_capacity = 400_000_000

    def register_forward_backward_hooks(self):
        self.forward_pre_hooks = {}
        self.forward_hooks = {}
        self.backward_pre_hooks = {}

        first_fsdp_module = None
        last_fsdp_module = None
        for name, module in self.module.named_modules():
            if isinstance(module, tuple(self.fsdp_modules)):
                if not first_fsdp_module:
                    first_fsdp_module = module
                last_fsdp_module = module

        # Register forward-backward hook for each module.
        def release_module_parameters(module, *unused):
            for param in module.parameters():
                self.all_gather_pipeline.release_item(param)

        def all_gather_module_parameters(
            module, *unused, prefetch=True, prefetch_order=PrefetchOrder.AFTER, wait_bucket_ready=True
        ):
            wait_list = []
            ag_pipeline = self.all_gather_pipeline
            for param in module.parameters():
                bucket_id = self.param_and_grad_buffer.param_to_param_group[param]
                ag_pipeline.queue_bucket(
                    bucket_id, prefetch=prefetch, prefetch_order=prefetch_order,
                    prefetch_capacity=self.all_gather_prefetch_capacity,
                )
                wait_list.append(bucket_id)
            for bucket_id in wait_list:
                ag_pipeline.wait_bucket_ready(bucket_id)

        def parameter_all_gather_forward_pre_hook_closure():
            def parameter_all_gather_forward_pre_hook(
                module, args: Tuple[Any, ...], kwargs: Dict[str, Any]
            ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
                if self.ddp_config.data_parallel_sharding_strategy == "MODEL_AND_OPTIMIZER_STATES":
                    # All-gather the parameters in every forward pass for FSDP.
                    ag_pipeline = self.all_gather_pipeline
                    fsdp_forward_prefetch = True
                    for param in module.parameters(recurse=False):
                        bucket_id = self.param_and_grad_buffer.param_to_param_group[param]
                        ag_pipeline.queue_bucket(
                            bucket_id,
                            prefetch=fsdp_forward_prefetch,
                            prefetch_capacity=self.all_gather_prefetch_capacity,
                        )
                        ag_pipeline.wait_bucket_ready(bucket_id)

                    if not torch.is_grad_enabled():
                        return args, kwargs

                    # Register the backward function to release the parameters.
                    if isinstance(module, tuple(self.fsdp_modules)):
                        fsdp_cache_edge_layers = True
                        if fsdp_cache_edge_layers and module is first_fsdp_module:
                            return args, kwargs

                        args_list, args_spec = tree_flatten(args)
                        kwargs_list, kwargs_spec = tree_flatten(kwargs)
                        args_kwargs_list = list(args_list) + list(kwargs_list)
                        inp_tensor_indices: List[int] = []
                        inp_tensors: List[torch.Tensor] = []
                        for i, obj in enumerate(args_kwargs_list):
                            if torch.is_tensor(obj) and obj.requires_grad:
                                inp_tensor_indices.append(i)
                                inp_tensors.append(obj)
                        assert inp_tensors, "No tensors that require gradients"

                        inp_tensors = RegisterFSDPBackwardFunction.apply(
                            functools.partial(release_module_parameters, module), *inp_tensors
                        )

                        for inp_tensor_idx, inp_tensor in zip(inp_tensor_indices, inp_tensors):
                            args_kwargs_list[inp_tensor_idx] = inp_tensor
                        args_list = args_kwargs_list[: len(args_list)]
                        kwargs_list = args_kwargs_list[len(args_list) :]
                        args = tree_unflatten(args_list, args_spec)
                        kwargs = tree_unflatten(kwargs_list, kwargs_spec)

                        return args, kwargs
                else:
                    ag_pipeline = self.all_gather_pipeline
                    for param in module.parameters(recurse=False):
                        bucket_id = self.param_and_grad_buffer.param_to_param_group[param]
                        ag_pipeline.wait_bucket_ready(bucket_id, empty_ok=True)

            return parameter_all_gather_forward_pre_hook

        if self.ddp_config.overlap_param_gather:
            forward_pre_hook = parameter_all_gather_forward_pre_hook_closure()
            for name, module in self.module.named_modules():
                self.forward_pre_hooks[
                    f'module {name} parameter all-gather'
                ] = module.register_forward_pre_hook(forward_pre_hook, with_kwargs=True)

        if self.data_parallel_sharding_strategy == "MODEL_AND_OPTIMIZER_STATES":
            for name, module in self.module.named_modules():
                if isinstance(module, tuple(self.fsdp_modules)):
                    fsdp_cache_edge_layers = True
                    if fsdp_cache_edge_layers and module is last_fsdp_module:
                        pass
                    else:
                        self.forward_hooks[
                            f"release module {name} parameters"
                        ] = module.register_forward_hook(
                            functools.partial(release_module_parameters, module)
                        )
                    # Register all-gather backward-pre hook for each fsdp module. NOTE: We need to
                    # reverse prefetch order in backward pass.
                    self.backward_pre_hooks[
                        f"all-gather module {name} parameters"
                    ] = module.register_full_backward_pre_hook(functools.partial(
                        all_gather_module_parameters, prefetch_order=PrefetchOrder.BEFORE
                    ))

        def _make_param_hook(
            param: torch.nn.Parameter,
        ):
            """
            Creates the all-reduce / reduce-scatter hook for backprop.
            """

            wait_previous_grad_reduce = self.data_parallel_sharding_strategy in [
                "MODEL_AND_OPTIMIZER_STATES",
                "OPTIMIZER_STATES_AND_GRADS",
            ]

            def param_hook(*unused):
                if param.requires_grad:
                    if self.ddp_config.overlap_grad_reduce:
                        assert (
                            param.grad is not None
                        ), 'param.grad being None is not safe when overlap_grad_reduce is True'

                    if param.grad is not None and (
                        not param.grad_added_to_main_grad or getattr(param, 'zero_out_wgrad', False)
                    ):
                        if self.is_delay_grad_reduce:
                            param.main_grad.add_(param.grad.data)
                        else:
                            bucket_id = self.param_and_grad_buffer.param_to_param_group[param]
                            self.grad_reduce_pipeline.place_the_bucket(bucket_id)
                            param.main_grad.copy_(param.grad.data)
                    param.grad = None

                    if self.ddp_config.overlap_grad_reduce and (
                        self.data_parallel_sharding_strategy
                        in ["OPTIMIZER_STATES_AND_GRADS", "MODEL_AND_OPTIMIZER_STATES",]
                        or self.is_last_microbatch
                    ):
                        rs_pipeline = self.grad_reduce_pipeline
                        bucket_id = self.param_and_grad_buffer.param_to_param_group[param]
                        self.grad_reduce_pipeline.place_the_bucket(bucket_id)
                        go_rs = rs_pipeline.mark_item_ready(param, async_rs=True)
                        if go_rs and wait_previous_grad_reduce:
                            rs_pipeline.wait_for_previous_grad_reduce(
                                recommeded_queue_capacity=self.reduce_scatter_queue_capacity
                            )

            return param_hook

        # Register backward gradient accumulation hook for each parameter.
        self.grad_accs = []
        for param in self.module.parameters():
            bucket_id = self.param_and_grad_buffer.param_to_param_group[param]
            wbuf = self.param_and_grad_buffer.parameter_groups[bucket_id].model_weight_buffer
            if param.requires_grad:
                if wbuf.is_data_distributed:
                    wbuf.fetch_the_bucket()

                # Expand so we get access to grad_fn.
                param_tmp = param.expand_as(param)
                # Get the gradient accumulator function.
                grad_acc = param_tmp.grad_fn.next_functions[0][0]
                grad_acc.register_hook(_make_param_hook(param))
                self.grad_accs.append(grad_acc)

                if wbuf.is_data_distributed:
                    wbuf.free_the_bucket_storage()

    def enable_forward_pre_hook(self):
        """
        Enable forward pre-hooks needed for param all-gather overlap with forward compute.
        """
        if self.ddp_config.with_megatron_fsdp_code_path:
            return
        assert self.use_forward_hook
        assert len(self.remove_forward_pre_hook_handles) == 0
        # Register forward pre-hook for all sub-modules.
        for module in self.module.modules():
            self.remove_forward_pre_hook_handles[module] = module.register_forward_pre_hook(
                self._make_forward_pre_hook()
            )

    def disable_forward_pre_hook(self):
        """
        Disable forward pre-hooks needed for param all-gather overlap with forward compute.
        """
        if self.ddp_config.with_megatron_fsdp_code_path:
            return
        assert self.use_forward_hook
        # De-register forward pre-hook for all sub-modules.
        for module in self.module.modules():
            assert self.remove_forward_pre_hook_handles[module] is not None
            self.remove_forward_pre_hook_handles[module].remove()
            del self.remove_forward_pre_hook_handles[module]
        assert len(self.remove_forward_pre_hook_handles) == 0

        # Force synchronize parameters.
        self.start_param_sync(force_sync=True)

    def forward(self, *inputs, **kwargs):
        """
        Calls the wrapped module's forward() method.
        """
        return self.module(*inputs, **kwargs)

    def _make_forward_pre_hook(self):
        """
        Create a forward pre-hook to wait on all-gather handles when necessary (i.e.,
        when a module uses a parameter in a bucket with a still incomplete all-gather).
        """

        def hook(module, *unused):
            assert (
                self.use_forward_hook
            ), "Should use pre-hook only when overlap_param_gather is True"

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
        if self.ddp_config.with_megatron_fsdp_code_path:
            self.is_last_microbatch = False
            try:
                yield
            finally:
                self.is_last_microbatch = True
        else:
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
        if self.ddp_config.with_megatron_fsdp_code_path:
            if self.data_parallel_sharding_strategy in [
                "OPTIMIZER_STATES",
                "OPTIMIZER_STATES_AND_GRADS",
            ]:
                # All-gather the parameters.
                if self.ddp_config.overlap_param_gather:
                    ag_pipeline = self.all_gather_pipeline
                    for bucket_id in range(ag_pipeline.buffer.num_buckets):
                        ag_pipeline.all_gather_bucket_and_set_items(bucket_id, async_op=True)
                else:
                    self.param_and_grad_buffer.all_gather_parameters()
            return

        if not force_sync:
            # If overlapping param AG with optimizer step, AG should not be dispatched again
            # in forward_backward_step.
            if self.overlap_param_gather_with_optimizer_step and not force_dispatch:
                return

        for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
            bucket_group.start_param_sync(force_sync=force_sync)

    def start_grad_sync(self, *unused):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operations
        for all model gradients.

        When overlap_grad_reduce is set to True, dispatches asynchronous communication
        calls. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        if self.ddp_config.with_megatron_fsdp_code_path:
            if self.ddp_config.overlap_grad_reduce:
                for bucket_id, pg in enumerate(self.param_and_grad_buffer.parameter_groups):
                    self.grad_reduce_pipeline.place_the_bucket(bucket_id)
                    for param in pg.params:
                        self.grad_reduce_pipeline.mark_item_ready(param, async_rs=True)
                        if self.data_parallel_sharding_strategy in [
                            "OPTIMIZER_STATES_AND_GRADS",
                            "MODEL_AND_OPTIMIZER_STATES",
                        ]:
                            self.grad_reduce_pipeline.wait_for_previous_grad_reduce(
                                recommeded_queue_capacity=self.reduce_scatter_queue_capacity
                            )
            else:
                if self.data_parallel_sharding_strategy == "NO_OP":
                    self.param_and_grad_buffer.all_reduce_gradients(
                        async_op=self.ddp_config.overlap_grad_reduce,
                    )
                else:
                    self.param_and_grad_buffer.reduce_scatter_gradients()
        else:
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
        if self.ddp_config.with_megatron_fsdp_code_path:
            if self.ddp_config.overlap_grad_reduce:
                self.grad_reduce_pipeline.wait_for_previous_grad_reduce(0)
                self.grad_reduce_pipeline.reset()
            else:
                self.start_grad_sync()

            if self.ddp_config.overlap_param_gather:
                self.all_gather_pipeline.reset()

            # reset the attribute of the parameters for optimizer.
            for _, param in self.optimizer_named_parameters():
                param.reset_attribute()
        else:
            for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
                bucket_group.finish_grad_sync()

    def optimizer_named_parameters(self):
        assert len(self.buffers) == 1
        return self.buffers[0].optimizer_named_parameters

    def scale_gradients(self, scaling_factor: float):
        """Scale all gradients inside the buffers by `scaling_factor`."""
        for buffer in self.buffers + self.expert_parallel_buffers:
            buffer.scale_gradients(scaling_factor)

    def zero_grad_buffer(self):
        """
        Zeros out all grad buffers. Needs to be called at the beginning of each
        training iteration.
        """
        if self.ddp_config.with_megatron_fsdp_code_path:
            for param in self.module.parameters():
                if param.requires_grad:
                    param.grad_added_to_main_grad = False
            for buffer in self.buffers:
                buffer.zero_grad()
        else:
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
                data_parallel_group = parallel_state.get_data_modulo_expert_parallel_group(
                    with_context_parallel=True
                )
            else:
                data_parallel_group = parallel_state.get_data_parallel_group(
                    with_context_parallel=True
                )
            torch.distributed.broadcast(
                param.data,
                src=torch.distributed.get_global_rank(data_parallel_group, 0),
                group=data_parallel_group,
            )

    def state_dict(self, prefix='', keep_vars=False):
        """
        Returns a dictionary containing references to the whole state of the
        wrapped module.

        Both parameters and persistent buffers (e.g. running averages) are included.
        Keys are corresponding parameter and buffer names. Parameters and buffers
        set to None are not included.
        """
        return self.module.state_dict(prefix=prefix, keep_vars=keep_vars)

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        """
        Returns wrapped module's state_dict for checkpoint saving.
        """
        return self.module.state_dict_for_save_checkpoint(prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        """
        Copies parameters and buffers from state_dict into the wrapped module and its
        descendants. If strict is True, then the keys of state_dict must exactly match
        the keys returned by this module’s state_dict() function.
        """
        if self.ddp_config.with_megatron_fsdp_code_path and self.ddp_config.data_parallel_sharding_strategy == "MODEL_AND_OPTIMIZER_STATES":
            # make a copy of the state_dict to avoid modifying the input state_dict
            state_dict = state_dict.copy()
            state_dict_extra_states = {}
            for key in list(state_dict.keys()):
                if key.endswith("_extra_state"):
                    state_dict_extra_states[key] = state_dict[key]
                    del state_dict[key]
            self.module.load_state_dict(state_dict_extra_states, strict=False)

            prefix = "module."
            for buffer in self.buffers:
                for param_groups in buffer.parameter_groups:
                    wbuf = param_groups.model_weight_buffer
                    for model_param in wbuf.params:
                        if is_float8tensor(model_param):
                            fp8_meta = model_param._fp8_meta['scaling_fwd']
                            fp8_meta_index = model_param._fp8_meta_index
                            model_param._scale_inv.copy_(fp8_meta.scale_inv[fp8_meta_index])

                        param_name = f"{buffer.param_to_name[model_param]}"[len(prefix):]
                        if param_name in state_dict:
                            if wbuf.is_data_distributed:
                                model_param.fully_shard_param_local_shard.data.copy_(state_dict[param_name])
                            else:
                                model_param.data.copy_(state_dict[param_name])
                            del state_dict[param_name]
            self.module.load_state_dict(state_dict, strict=False)
            return
        self.module.load_state_dict(state_dict, strict=strict)

class RegisterFSDPBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, post_forward, *inputs: torch.Tensor):
        ctx.post_forward = post_forward
        return inputs

    @staticmethod
    def backward(ctx, *grads: torch.Tensor):
        ctx.post_forward()
        return (None,) + grads
