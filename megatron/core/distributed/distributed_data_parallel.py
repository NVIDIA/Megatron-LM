# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import logging
from contextlib import contextmanager
from typing import Dict

import torch

from .. import parallel_state
from ..config_logger import has_config_logger_enabled, log_config_to_disk
from ..transformer.module import MegatronModule
from ..transformer.transformer_config import TransformerConfig
from ..utils import log_single_rank
from .distributed_data_parallel_config import DistributedDataParallelConfig
from .param_and_grad_buffer import ParamAndGradBuffer

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

        self.module = module
        self.param_to_buffer = {}

        # Group parameters by their gradient type.
        param_to_name = {}
        dense_params = []
        expert_parallel_params = []
        for name, param in self.module.named_parameters():
            if not param.requires_grad:
                continue

            param.grad_added_to_main_grad = False
            param_to_name[param] = name

            if getattr(param, 'allreduce', True):
                dense_params.append(param)
            else:
                expert_parallel_params.append(param)

        def allocate_buffers_for_parameters(
            input_params, data_parallel_group, gradient_scaling_factor
        ):
            param_and_grad_dtype_to_params = {}

            # Group parameters by their gradient type.
            for param in input_params:
                if not param.requires_grad:
                    continue

                param_dtype = param.dtype
                grad_dtype = torch.float if self.ddp_config.grad_reduce_in_fp32 else param.dtype

                params = param_and_grad_dtype_to_params.get((param_dtype, grad_dtype), [])
                params.append(param)
                param_and_grad_dtype_to_params[(param_dtype, grad_dtype)] = params

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
                    ParamAndGradBuffer(
                        self.ddp_config,
                        param_dtype,
                        grad_dtype,
                        params,
                        data_parallel_group,
                        self.bucket_size,
                        param_to_name,
                        gradient_scaling_factor,
                    )
                )
                for param in params:
                    self.param_to_buffer[param] = buffers[-1]

            return buffers

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

        # Allocate the param+grad buffers for dense params' grads.
        self.buffers = allocate_buffers_for_parameters(
            dense_params,
            parallel_state.get_data_parallel_group(with_context_parallel=True),
            gradient_scaling_factor=gradient_scaling_factor,
        )

        # Allocate separate param+grad buffers for expert parallel params' grads.
        self.expert_parallel_buffers = allocate_buffers_for_parameters(
            expert_parallel_params,
            parallel_state.get_data_modulo_expert_parallel_group(with_context_parallel=True),
            gradient_scaling_factor=expert_gradient_scaling_factor,
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
                grad_acc.register_hook(self._make_param_hook(param, self.param_to_buffer))
                self.grad_accs.append(grad_acc)

    def forward(self, *inputs, **kwargs):
        """
        Calls the wrapped module's forward() method.
        """
        return self.module(*inputs, **kwargs)

    def _make_param_hook(
        self,
        param: torch.nn.Parameter,
        param_to_buffer: Dict[torch.nn.Parameter, ParamAndGradBuffer],
    ):
        """
        Creates the all-reduce / reduce-scatter hook for backprop.
        """

        def param_hook(*unused):
            if param.requires_grad:
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
                    param_to_buffer[param].register_grad_ready(param)

        return param_hook

    @contextmanager
    def no_sync(self):
        """
        Context manager that turns off gradient synchronization.
        """
        for buffer in self.buffers + self.expert_parallel_buffers:
            buffer.is_last_microbatch = False
        try:
            yield
        finally:
            for buffer in self.buffers + self.expert_parallel_buffers:
                buffer.is_last_microbatch = True

    def start_grad_sync(self, *unused):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operations
        for all model gradients.

        When overlap_grad_reduce is set to True, dispatches asynchronous communication
        calls. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        for buffer in self.buffers + self.expert_parallel_buffers:
            buffer.start_grad_sync()

    def scale_gradients(self, scaling_factor: float) -> None:
        """Scale all gradients inside the buffers by `scaling_factor`."""
        for buffer in self.buffers + self.expert_parallel_buffers:
            buffer.scale_gradients(scaling_factor)

    def finish_grad_sync(self):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operations
        for all model gradients.

        When overlap_grad_reduce is set to True, waits for asynchronous communication
        calls to complete. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        for buffer in self.buffers + self.expert_parallel_buffers:
            buffer.finish_grad_sync()

    def zero_grad_buffer(self):
        """
        Zeros out all grad buffers. Needs to be called at the beginning of each
        training iteration.
        """
        for param in self.module.parameters():
            if param.requires_grad:
                param.grad_added_to_main_grad = False
        for buffer in self.buffers + self.expert_parallel_buffers:
            buffer.reset()

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
        self.module.load_state_dict(state_dict, strict=strict)
