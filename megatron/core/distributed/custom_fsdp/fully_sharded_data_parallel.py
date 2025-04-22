# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import functools
import logging
from contextlib import contextmanager
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils._pytree import tree_flatten, tree_unflatten

from megatron.core import parallel_state
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.distributed.custom_fsdp.param_and_grad_buffer import (
    AllGatherPipeline,
    BucketingPolicy,
    GradReducePipeline,
    ParamAndGradBuffer,
    PrefetchOrder,
)
from megatron.core.distributed.data_parallel_base import _BaseDataParallel
from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.fp8_utils import is_float8tensor
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import is_submodule, log_single_rank

logger = logging.getLogger(__name__)


class TrainingState(Enum):
    """States of a FSDP parameter group, which are coupled with
    the sharding activity of parameters and gradients during training."""

    # From pre-forward before post-forward, where parameters should be unsharded
    FORWARD = auto()
    # Prior to backward computation, where parameters should be unsharded
    PRE_BACKWARD = auto()
    # After backward computation, where gradients should be re-sharded
    POST_BACKWARD = auto()
    # Before and after module forward computaton or before pre-backward and
    # after post-backward states, where no un/sharding activity happens
    IDLE = auto()


class FullyShardedDataParallel(_BaseDataParallel):
    """Fully Sharded Data Parallel training for MCore models.

    A distributed training wrapper that shards model parameters, gradients and optimizer
    states across data parallel workers. Integrates seamlessly with MCore's tensor
    and expert parallelism features.

    We supports following modes:
    - no_shard: Traditional data parallel training without parameter sharding.
    - optim: Shards optimizer states, this is conceptually close to "ZeRO-1", and
        main weights for mixed precision training, meanwhile the following `optim_grads`
        and `optim_grads_params` will also sharding main weights
        during mixed-precision training, omitted without detailed notation.
    - optim_grads: Shards gradients and optimizer states, this is conceptually close to "ZeRO-2".
    - optim_grads_params: Shards parameters, gradients and optimizer states, this
        is conceptually close to "ZeRO-3".

    Key Features:
    - Compatible with MCore's tensor, context and expert parallelism
    - Automatic mixed precision training (BF16/FP8)
    - Gradient accumulation and bucketing
    - Optimized activation recompute with shard-aware communication: When recomputing
        a whole Transformer layer, gather parameters once for both the recomputation
        and backward computation
    - Compatible with MCore's distributed checkpointing

    Args:
        config: Transformer config object.
        ddp_config: FullyShardedDataParallel config object.
        module: Underlying model.
        fsdp_unit_modules: List of modules that should be treated as FSDP Unit,
            i.e., the minimum releasable model unit. If not provided, defaults to
            [TransformerLayer, LanguageModelEmbedding] for GPT-like models.
        disable_bucketing: If true, force assign all parameters to a single bucket. If false,
            use standard bucketing policy: assign parameters to smaller buckets and all-reduce
            per bucket.
    Examples:
        >>> model = GPTModel(config)
        >>> model = FullyShardedDataParallel(
        ...     config,
        ...     model,
        ...     ddp_config,
        ...     fsdp_unit_modules = [TransformerLayer, LanguageModelEmbedding],
        ... )
    """

    # TODO: add hybrid FSDP (shard model states in a partial DP domain)
    def __init__(
        self,
        config: TransformerConfig,
        ddp_config: DistributedDataParallelConfig,
        module: torch.nn.Module,
        fsdp_unit_modules: Optional[List[torch.nn.Module]] = None,
        disable_bucketing: bool = False,
        device: Optional[torch.device] = None,
    ):
        super().__init__(config=config, module=module)
        if has_config_logger_enabled(config):
            log_config_to_disk(config, locals(), prefix=type(self).__name__)

        self.module = module
        self.ddp_config = ddp_config
        log_single_rank(
            logger,
            logging.INFO,
            f'Setting up DistributedDataParallel with config {self.ddp_config}',
        )

        self.bucket_size = self.ddp_config.bucket_size
        if disable_bucketing:
            self.bucket_size = None
        self.device = device if device else torch.cuda.current_device()

        self.param_to_bucket_group = {}

        if fsdp_unit_modules is not None:
            self.fsdp_unit_modules = fsdp_unit_modules
        else:
            self.fsdp_unit_modules = [TransformerLayer]
            if not getattr(self.module, "share_embeddings_and_output_weights", False):
                self.fsdp_unit_modules.append(LanguageModelEmbedding)
        self.main_weights = True
        self.data_parallel_group = parallel_state.get_data_parallel_group(
            with_context_parallel=True
        )
        self.expert_data_parallel_group = parallel_state.get_expert_data_parallel_group()

        # Determine if we should delay the gradient reduction.
        self.is_delay_grad_reduce = self.ddp_config.data_parallel_sharding_strategy in [
            "no_shard",
            "optim",
        ]

        if self.ddp_config.data_parallel_sharding_strategy == "optim_grads_params":
            assert self.ddp_config.overlap_param_gather
        if not self.is_delay_grad_reduce:
            assert self.ddp_config.overlap_grad_reduce
        self._init_fsdp_param_and_grad_buffer()
        self._register_fsdp_hooks(self.module)

        # Delete references to weight_tensor if they exist since we don't want two parameter copies
        # if we re-mapped parameters (which happens when we use the distributed optimizer).
        # This is a temporary workaround around a TE bug that is fixed with
        # https://github.com/NVIDIA/TransformerEngine/pull/719.
        @torch.no_grad()
        def unmap_weight_tensor(m):
            if hasattr(m, 'weight_tensor'):
                m.weight_tensor = None

        self.module.apply(unmap_weight_tensor)

    def _init_fsdp_param_and_grad_buffer(self):
        if self.config.calculate_per_token_loss:
            # We don't need to scale the gradients in this case.
            gradient_scaling_factor = None
            expert_gradient_scaling_factor = None
        else:
            if self.ddp_config.average_in_collective:
                # FIXME(@jianbinc): Will fix this issue based on Parallel Folding's EDP patch MR.
                raise Exception("Not supported")
            else:
                data_parallel_world_size = parallel_state.get_data_parallel_world_size(
                    with_context_parallel=True
                )
                gradient_scaling_factor = 1.0 / data_parallel_world_size
                expert_gradient_scaling_factor = 1.0 / data_parallel_world_size

        # Initialize the param and grad buffer.
        self.data_parallel_sharding_strategy = self.ddp_config.data_parallel_sharding_strategy
        self.param_to_name = {p: name for name, p in self.module.named_parameters()}
        self.param_and_grad_buffer = ParamAndGradBuffer(
            self.ddp_config,
            self.module,
            bucketing_policy=BucketingPolicy(
                suggested_bucket_size=self.bucket_size,
                fsdp_unit_modules=(
                    # Only when model weights need to be sharded, we need to
                    # identify the minimum releasable model unit, which is the
                    # FSDP Unit Module.
                    self.fsdp_unit_modules
                    if self.data_parallel_sharding_strategy == "optim_grads_params"
                    else []
                ),
                data_parallel_sharding_strategy=self.data_parallel_sharding_strategy,
            ),
            data_parallel_group=self.data_parallel_group,
            expert_data_parallel_group=self.expert_data_parallel_group,
            preserve_fp32_weights=self.ddp_config.preserve_fp32_weights,
            grad_reduce_in_fp32=self.ddp_config.grad_reduce_in_fp32,
            gradient_scaling_factor=gradient_scaling_factor,
            expert_gradient_scaling_factor=expert_gradient_scaling_factor,
            device=self.device,
            reset_parameters_for_meta_device_init_module=self.config.init_model_with_meta_device,
        )
        self.param_and_grad_buffer

        self.side_stream_for_buffer_copy_and_grad_accum = torch.cuda.Stream()

        # Initialize the reduce-scatter pipeline.
        self.grad_reduce_pipeline = GradReducePipeline(
            self.param_and_grad_buffer, cuda_stream=self.side_stream_for_buffer_copy_and_grad_accum
        )

        # Initialize the all-gather pipeline.
        self.all_gather_pipeline = AllGatherPipeline(self.param_and_grad_buffer)

        self.suggested_RS_queue_capacity = self.ddp_config.suggested_communication_unit_size
        self.suggested_AG_prefetch_size = self.ddp_config.suggested_communication_unit_size

    def _register_fsdp_hooks(self, root_module):
        """Register necessary hooks for Fully Sharded Data Parallel (FSDP) execution on the model.

        This function sets up various hooks required for FSDP operations, including parameter
        resharding/unsharding and gradient handling. The registered hooks are:
            - Pre-forward hook: Unshards parameters before forward pass
            - Post-forward hook: Reshards parameters after forward pass
            - Pre-backward hook: Unshards parameters before backward pass
            - Post-backward hook: Reshards parameters after backward pass
            - Gradient accumulation hook: Handles gradient accumulation and reduction across devices

        Args:
            root_module: The PyTorch module to register FSDP hooks on

        Note:
            These hooks are essential for FSDP's memory efficiency as they manage:
            1. Dynamic parameter sharding/unsharding to reduce memory footprint
            2. Proper gradient synchronization across distributed processes
            3. Gradient accumulation for large batch training

        Returns:
            None
        """

        # Initialize module training state.
        for m in root_module.modules():
            setattr(m, "_training_state", TrainingState.IDLE)

        self.forward_pre_hooks = {}
        self.forward_hooks = {}
        self.backward_pre_hooks = {}

        """
        An FSDP unit is a module designed to manage the lifecycle of model parameters
        in Fully Sharded Data Parallel (FSDP) training. It ensures that parameters
        are only used within the module and are released immediately after
        the forward and backward computations are completed.
        This approach is crucial for efficient memory management, as releasing
        parameters too early can lead to issues if other computations depend on them.

        `optim` and `optim_grads` do not require FSDP units because they do not
        shard model parameters.
        """
        if self.data_parallel_sharding_strategy != "optim_grads_params":
            fsdp_unit_modules = []
        else:
            fsdp_unit_modules = self.fsdp_unit_modules

        def release_module_parameters(module, *unused):
            for param in module.parameters():
                bucket_id = self.param_and_grad_buffer.param_to_param_group[param]
                self.all_gather_pipeline.release_bucket(bucket_id)

            if not self.ddp_config.keep_fp8_transpose_cache_when_using_custom_fsdp:
                release_params_fp8_transpose_cache(module.parameters())

        def release_params_fp8_transpose_cache(params):
            for param in params:
                if is_float8tensor(param):
                    param._transpose_invalid = True
                    param._transpose = None

        def all_gather_module_parameters(
            module,
            *unused,
            prefetch=True,
            prefetch_order=PrefetchOrder.FORWARD_PASS_ORDER,
            wait_bucket_ready=True,
        ):
            wait_list = []
            ag_pipeline = self.all_gather_pipeline
            for param in module.parameters():
                bucket_id = self.param_and_grad_buffer.param_to_param_group[param]
                ag_pipeline.queue_bucket_to_all_gather(
                    bucket_id,
                    prefetch=prefetch,
                    prefetch_order=prefetch_order,
                    suggested_AG_prefetch_size=self.suggested_AG_prefetch_size,
                )
                wait_list.append(bucket_id)

            if wait_bucket_ready:
                for bucket_id in wait_list:
                    ag_pipeline.wait_bucket_ready(bucket_id)

        def _post_backward(module, *unused):
            release_module_parameters(module)
            module._training_state = TrainingState.IDLE

        def _pre_forward(module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
            input_training_state = module._training_state
            fsdp_forward_prefetch = True
            if input_training_state == TrainingState.PRE_BACKWARD:
                # In activation recomputation case, we need to cancel forward prefetch.
                fsdp_forward_prefetch = False
            else:
                module._training_state = TrainingState.FORWARD

            if isinstance(module, tuple(fsdp_unit_modules)):
                wait_list = []
                for param in module.parameters():
                    bucket_id = self.param_and_grad_buffer.param_to_param_group[param]
                    self.all_gather_pipeline.queue_bucket_to_all_gather(
                        bucket_id,
                        prefetch=fsdp_forward_prefetch,
                        suggested_AG_prefetch_size=self.suggested_AG_prefetch_size,
                    )
                    wait_list.append(bucket_id)
                for bucket_id in wait_list:
                    self.all_gather_pipeline.wait_bucket_ready(bucket_id)

                if not torch.is_grad_enabled():
                    return args, kwargs

                # Register the backward function to release the parameters.
                args_list, args_spec = tree_flatten(args)
                kwargs_list, kwargs_spec = tree_flatten(kwargs)
                args_kwargs_list = list(args_list) + list(kwargs_list)
                inp_tensor_indices: List[int] = []
                inp_tensors: List[torch.Tensor] = []
                for i, obj in enumerate(args_kwargs_list):
                    if torch.is_tensor(obj) and obj.requires_grad:
                        inp_tensor_indices.append(i)
                        inp_tensors.append(obj)
                if len(inp_tensors) == 0:
                    return args, kwargs
                inp_tensors = RegisterFSDPBackwardFunction.apply(
                    functools.partial(_post_backward, module), *inp_tensors
                )
                for inp_tensor_idx, inp_tensor in zip(inp_tensor_indices, inp_tensors):
                    args_kwargs_list[inp_tensor_idx] = inp_tensor
                args_list = args_kwargs_list[: len(args_list)]
                kwargs_list = args_kwargs_list[len(args_list) :]
                args = tree_unflatten(args_list, args_spec)
                kwargs = tree_unflatten(kwargs_list, kwargs_spec)

                return args, kwargs
            else:
                # All-gather the parameters in every forward pass for FSDP.
                for param in module.parameters(recurse=False):
                    bucket_id = self.param_and_grad_buffer.param_to_param_group[param]
                    self.all_gather_pipeline.queue_bucket_to_all_gather(
                        bucket_id,
                        prefetch=fsdp_forward_prefetch,
                        suggested_AG_prefetch_size=self.suggested_AG_prefetch_size,
                    )
                for param in module.parameters(recurse=False):
                    bucket_id = self.param_and_grad_buffer.param_to_param_group[param]
                    self.all_gather_pipeline.wait_bucket_ready(bucket_id)

            return args, kwargs

        if self.ddp_config.overlap_param_gather:
            fsdp_modules = []
            for name, module in root_module.named_modules():
                if self.ddp_config.data_parallel_sharding_strategy == "optim_grads_params":
                    if any(is_submodule(module, fsdp_module) for fsdp_module in fsdp_modules):
                        continue

                    if isinstance(module, tuple(fsdp_unit_modules)):
                        fsdp_modules.append(module)

                self.forward_pre_hooks[f'module {name} parameter all-gather'] = (
                    module.register_forward_pre_hook(_pre_forward, prepend=True, with_kwargs=True)
                )

        def _pre_backward(module: nn.Module, *unused):
            module._training_state = TrainingState.PRE_BACKWARD
            if isinstance(module, tuple(fsdp_unit_modules)):
                all_gather_module_parameters(
                    module, prefetch_order=PrefetchOrder.BACKWARD_PASS_ORDER
                )

        def _root_pre_backward(module: nn.Module, *unused):
            """Marks the module's training state as 'pre_backward' before the
            backprop, this function is registered on the root module.

            This marking enables us to determine whether forward pass needs to
            perform reshard/unshard operations in activation recomputation
            scenarios.
            """
            for module in root_module.modules():
                if isinstance(module, tuple(fsdp_unit_modules)):
                    module._training_state = TrainingState.PRE_BACKWARD
                    for param in module.parameters():
                        bucket_id = self.param_and_grad_buffer.param_to_param_group[param]
                        self.all_gather_pipeline.wait_bucket_ready(bucket_id, empty_ok=True)
                        self.all_gather_pipeline.release_bucket(bucket_id)

        def _post_forward(module: nn.Module, input: Any, output: Any):
            # When composing with module-hook-based activation checkpointing, the
            # post-backward hook is responsible for the reshard
            if module._training_state == TrainingState.PRE_BACKWARD:
                return output

            release_module_parameters(module)
            module._training_state = TrainingState.IDLE

            return output

        def _release_module_fp8_transpose_cache(module: nn.Module, *unused):
            release_params_fp8_transpose_cache(module.parameters(recurse=False))

        if self.data_parallel_sharding_strategy == "optim_grads_params":
            fsdp_modules = []
            for name, module in root_module.named_modules():
                if any(is_submodule(module, fsdp_module) for fsdp_module in fsdp_modules):
                    continue

                if isinstance(module, tuple(fsdp_unit_modules)):
                    fsdp_modules.append(module)
                    self.forward_hooks[f"release module {name} parameters"] = (
                        module.register_forward_hook(_post_forward, prepend=False)
                    )
                    self.backward_pre_hooks[f"all-gather module {name} parameters"] = (
                        module.register_full_backward_pre_hook(_pre_backward)
                    )
                elif not self.ddp_config.keep_fp8_transpose_cache_when_using_custom_fsdp:
                    self.forward_hooks[f"remove module {name} fp8 transpose cache"] = (
                        module.register_forward_hook(
                            _release_module_fp8_transpose_cache, prepend=False
                        )
                    )
            self._root_pre_backward_hook_handle = root_module.register_full_backward_pre_hook(
                _root_pre_backward
            )

        def _make_param_hook(param: torch.nn.Parameter):
            """
            Creates the all-reduce / reduce-scatter hook for backprop.
            """

            wait_previous_grad_reduce = not self.is_delay_grad_reduce

            # FIXME: Use insert forward op to replace grad acc hook, which will
            # be lost after parameter data movement. For example, module.cuda()
            # will cause the registered grad acc hook to be lost.
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
                            param.main_grad.copy_(param.grad.data)
                    param.grad = None

                    if self.ddp_config.overlap_grad_reduce and (
                        not self.is_delay_grad_reduce or self.is_last_microbatch
                    ):
                        gr_pipeline = self.grad_reduce_pipeline
                        bucket_id = self.param_and_grad_buffer.param_to_param_group[param]
                        gr_pipeline.place_bucket(bucket_id)
                        go_rs = gr_pipeline.mark_item_ready(param, async_rs=True)
                        if go_rs and wait_previous_grad_reduce:
                            gr_pipeline.wait_for_previous_grad_reduce(
                                recommeded_queue_capacity=self.suggested_RS_queue_capacity
                            )

            return param_hook

        # Register backward gradient accumulation hook for each parameter.
        self.grad_accs = []
        for param in root_module.parameters():
            bucket_id = self.param_and_grad_buffer.param_to_param_group[param]
            wbuf = self.param_and_grad_buffer.parameter_groups[bucket_id].model_weight_buffer
            if param.requires_grad:
                if wbuf and wbuf.is_data_distributed:
                    wbuf.fetch_bucket(and_allocate_params_data=True)

                # Expand so we get access to grad_fn.
                param_tmp = param.expand_as(param)
                # Get the gradient accumulator function.
                grad_acc = param_tmp.grad_fn.next_functions[0][0]
                grad_acc.register_hook(_make_param_hook(param))
                self.grad_accs.append(grad_acc)

                if wbuf and wbuf.is_data_distributed:
                    wbuf.free_bucket_storage()

    @contextmanager
    def no_sync(self):
        """
        Context manager that turns off gradient synchronization.
        For grads shard mode there will actually always be gradient sync happening.
        """
        # FIXME: Better handling of grads shard mode and no_sync in the training loop so that
        # the code doesn't bog down developers.
        self.is_last_microbatch = False
        try:
            yield
        finally:
            self.is_last_microbatch = True

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
        if not force_sync and self.ddp_config.overlap_param_gather:
            # All-gather the first bucket before the forward pass.
            self.all_gather_pipeline.queue_bucket_to_all_gather(bucket_id=0, prefetch=False)
        else:
            self.all_gather_pipeline.reset()
            for bucket_id in range(self.all_gather_pipeline.num_buckets):
                self.all_gather_pipeline.all_gather_bucket_and_set_items(
                    bucket_id=bucket_id, async_op=True
                )
                group = self.param_and_grad_buffer.parameter_groups[bucket_id]
                if group.model_weight_buffer is None:
                    continue

                if group.model_weight_buffer.is_data_distributed:
                    # If model weight is sharded, we wait for the all-gather to complete and
                    # then release the bucket immediately to save memory usage.
                    self.all_gather_pipeline.wait_bucket_ready(bucket_id)
            for bucket_id in range(self.all_gather_pipeline.num_buckets):
                self.all_gather_pipeline.wait_bucket_ready(bucket_id)

    def start_grad_sync(self, *unused):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operations
        for all model gradients.

        When overlap_grad_reduce is set to True, dispatches asynchronous communication
        calls. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        if not self.ddp_config.overlap_grad_reduce:
            if self.data_parallel_sharding_strategy == "no_shard":
                self.param_and_grad_buffer.all_reduce_gradients(
                    async_op=self.ddp_config.overlap_grad_reduce
                )
            else:
                self.param_and_grad_buffer.reduce_scatter_gradients()

    def finish_grad_sync(self):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operations
        for all model gradients.

        When overlap_grad_reduce is set to True, waits for asynchronous communication
        calls to complete. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        if self.ddp_config.overlap_grad_reduce:
            self.grad_reduce_pipeline.wait_for_previous_grad_reduce(0)
            self.grad_reduce_pipeline.reset()
        else:
            self.start_grad_sync()

        self.param_and_grad_buffer.update_main_grads()

        if self.ddp_config.overlap_param_gather:
            self.all_gather_pipeline.reset()

    def optimizer_named_parameters(self) -> List[Tuple[str, torch.Tensor]]:
        """
        Returns a list of tuples containing the main weights and their corresponding names
        for mixed-precision training, to be used by the optimizer for updates.

        Returns:
            List[Tuple[str, torch.Tensor]]: A list of tuples, where each tuple
                contains a main weight tensor and its corresponding name.
        """
        return self.param_and_grad_buffer.optimizer_named_parameters

    def scale_gradients(self, scaling_factor: float):
        """Scale all gradients inside the buffers by `scaling_factor`."""
        self.param_and_grad_buffer.scale_gradients(scaling_factor)

    def zero_grad_buffer(self):
        """
        Zeros out all grad buffers. Needs to be called at the beginning of each
        training iteration.
        """
        for param in self.module.parameters():
            if param.requires_grad:
                param.grad_added_to_main_grad = False
        self.param_and_grad_buffer.zero_grad()

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

    def load_state_dict(self, state_dict, strict=True):
        """
        Copies parameters and buffers from state_dict into the wrapped module and its
        descendants. If strict is True, then the keys of state_dict must exactly match
        the keys returned by this module’s state_dict() function.
        """
        if self.ddp_config.data_parallel_sharding_strategy == "optim_grads_params":
            # make a copy of the state_dict to avoid modifying the input state_dict
            state_dict = state_dict.copy()
            state_dict_extra_states = {}
            for key in list(state_dict.keys()):
                if key.endswith("_extra_state"):
                    state_dict_extra_states[key] = state_dict[key]
                    del state_dict[key]
            self.module.load_state_dict(state_dict_extra_states, strict=False)

            prefix = "module."
            buffer = self.param_and_grad_buffer
            for param_groups in buffer.parameter_groups:
                wbuf = param_groups.model_weight_buffer
                for model_param in wbuf.params:
                    if is_float8tensor(model_param):
                        fp8_meta = model_param._fp8_meta['scaling_fwd']
                        fp8_meta_index = model_param._fp8_meta_index
                        model_param._scale_inv.copy_(fp8_meta.scale_inv[fp8_meta_index])

                    param_name = f"{buffer.param_to_name[model_param]}"[len(prefix) :]
                    if param_name in state_dict:
                        if wbuf and wbuf.is_data_distributed:
                            model_param.fully_shard_param_local_shard.data.copy_(
                                state_dict[param_name]
                            )
                        else:
                            model_param.data.copy_(state_dict[param_name])
                        del state_dict[param_name]
            self.module.load_state_dict(state_dict, strict=False)
            return
        self.module.load_state_dict(state_dict, strict=strict)


class RegisterFSDPBackwardFunction(torch.autograd.Function):
    """
    Register a backward function that will be called after the backward pass
    of the model. This function is used to release the parameters after the
    backward pass.
    """

    @staticmethod
    def forward(ctx, post_backward, *inputs: torch.Tensor):
        """
        Forward pass of the RegisterFSDPBackwardFunction function.
        """
        ctx.post_backward = post_backward
        return inputs

    @staticmethod
    def backward(ctx, *grads: torch.Tensor):
        """
        Backward pass of the RegisterFSDPBackwardFunction function.
        """
        ctx.post_backward()
        return (None,) + grads
