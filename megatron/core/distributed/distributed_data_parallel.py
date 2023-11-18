# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from contextlib import contextmanager
from typing import Dict

import torch

from .. import parallel_state
from ..transformer.module import MegatronModule
from ..transformer.transformer_config import TransformerConfig
from .grad_buffer import GradBuffer


class DistributedDataParallel(MegatronModule):
    """
    DDP wrapper which stores grads in contiguous buffers. Also has option of overlapping
    communication with backprop computation by breaking up full model's gradients into smaller
    buckets and running all-reduce / reduce-scatter on each bucket asynchronously. This class
    also provides the option to do the gradient accumulation in a type other than the param type
    (e.g., fp32 for a bf16 model).

    Arguments:
        config: Transformer config object.
        module: Underlying model.
        data_parallel_group: Data-parallel process group.
        accumulate_allreduce_grads_in_fp32: If true, do the gradient accumulation and
            communication in fp32.
        overlap_grad_reduce: If true, overlap communication with backprop computation by
            breaking up grads into buckets. If false, single synchronous communication call
            is used instead.
        use_distributed_optimizer: If true, issue reduce-scatter communication calls as part
            of distributed optimizer. If false, issue all-reduce communication calls.
        disable_bucketing: If true, force assign all parameters to a single bucket. If false,
            use standard bucketing policy: assign parameters to smaller buckets and all-reduce
            per bucket _if_ overlap_grad_reduce is True and pp_rank is 0.

    """

    def __init__(
        self,
        config: TransformerConfig,
        module: torch.nn.Module,
        data_parallel_group: torch.distributed.ProcessGroup,
        accumulate_allreduce_grads_in_fp32: bool,
        overlap_grad_reduce: bool,
        use_distributed_optimizer: bool,
        disable_bucketing: bool = False,
        bucket_size: int = 40000000,
    ):
        super().__init__(config=config)
        self.module = module

        # Set bucket_size to infinity if overlap_grad_reduce is False.
        self.overlap_grad_reduce = overlap_grad_reduce
        self.use_distributed_optimizer = use_distributed_optimizer

        # Turn off bucketing if overlap_grad_reduce is False, if we are on a pipeline stage
        # that is not the first (since data-parallel communication on these stages is not on
        # the critical path), or if disable_bucketing is True (e.g., we might not want to
        # break up model parameters into buckets for model chunks after the first
        # in the interleaved schedule).
        if not self.overlap_grad_reduce:
            bucket_size = None
        if parallel_state.get_pipeline_model_parallel_rank() > 0:
            bucket_size = None
        if disable_bucketing:
            bucket_size = None
        self.bucket_size = bucket_size

        self.module = module
        self.grad_buffers = {}
        self.expert_grads = []
        self.grad_buffer_param_index_map = {}
        self.param_to_grad_buffer = {}

        # Group parameters by their gradient type.
        grad_dtype_to_params = {}
        param_to_name = {}
        for name, param in self.module.named_parameters():
            if param.requires_grad and getattr(param, 'allreduce', True):
                param.grad_added_to_main_grad = False
                param_to_name[param] = name
                dtype = torch.float if accumulate_allreduce_grads_in_fp32 else param.dtype

                params = grad_dtype_to_params.get(dtype, [])
                params.append(param)
                grad_dtype_to_params[dtype] = params

        # Allocate the grad buffers and map the grads.
        # The grad buffer under the hood creates buckets as appropriate based on bucket_size.
        self.data_parallel_world_size = torch.distributed.get_world_size(group=data_parallel_group)
        for dtype, params in grad_dtype_to_params.items():
            self.grad_buffers[dtype] = GradBuffer(
                dtype,
                params,
                data_parallel_group,
                bucket_size,
                param_to_name,
                self.overlap_grad_reduce,
                self.use_distributed_optimizer,
            )
            self.grad_buffer_param_index_map[dtype] = self.grad_buffers[dtype].param_index_map
            for param in params:
                self.param_to_grad_buffer[param] = self.grad_buffers[dtype]

        # Allocate separate buffer for MoE params' grads.
        for param in self.module.parameters():
            if param.requires_grad and not getattr(param, 'allreduce', True):
                param.grad_added_to_main_grad = False
                dtype = torch.float if accumulate_allreduce_grads_in_fp32 else param.dtype
                param.main_grad = torch.zeros(
                    param.data.shape,
                    dtype=dtype,
                    device=torch.cuda.current_device(),
                    requires_grad=False,
                )
                self.expert_grads.append(param.main_grad)

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
                grad_acc.register_hook(self._make_param_hook(param, self.param_to_grad_buffer))
                self.grad_accs.append(grad_acc)

    def forward(self, *inputs, **kwargs):
        """
        Calls the wrapped module's forward() method.
        """
        return self.module(*inputs, **kwargs)

    def _make_param_hook(
        self, param: torch.nn.Parameter, param_to_grad_buffer: Dict[torch.nn.Parameter, GradBuffer]
    ):
        """
        Creates the all-reduce / reduce-scatter hook for backprop.
        """

        def param_hook(*unused):
            if param.requires_grad:
                if self.overlap_grad_reduce:
                    assert (
                        param.grad is not None
                    ), 'param.grad being None is not safe when overlap_grad_reduce is True'
                if param.grad is not None and not param.grad_added_to_main_grad:
                    param.main_grad.add_(param.grad.data)
                param.grad = None
                if self.overlap_grad_reduce:
                    param_to_grad_buffer[param].register_grad_ready(param)

        return param_hook

    @contextmanager
    def no_sync(self):
        """
        Context manager that turns off gradient synchronization.
        """
        for grad_buffer in self.grad_buffers.values():
            grad_buffer.is_last_microbatch = False
        try:
            yield
        finally:
            for grad_buffer in self.grad_buffers.values():
                grad_buffer.is_last_microbatch = True

    def start_grad_sync(self, *unused):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operations
        for all model gradients.

        When overlap_grad_reduce is set to True, dispatches asynchronous communication
        calls. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        for grad_buffer in self.grad_buffers.values():
            grad_buffer.start_grad_sync()

    def finish_grad_sync(self):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operations
        for all model gradients.

        When overlap_grad_reduce is set to True, waits for asynchronous communication
        calls to complete. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        for grad_buffer in self.grad_buffers.values():
            grad_buffer.finish_grad_sync()

        for expert_grad in self.expert_grads:
            expert_grad /= self.data_parallel_world_size

    def zero_grad_buffer(self, zero_buffer):
        """
        Zeros out all grad buffers. Needs to be called at the beginning of each
        training iteration.

        When zero_buffer is set to True, the underlying grad buffer is zeroed out.
        """
        for param in self.module.parameters():
            if param.requires_grad:
                param.grad_added_to_main_grad = False
        for grad_buffer in self.grad_buffers.values():
            grad_buffer.reset(zero_buffer)
        for expert_grad in self.expert_grads:
            expert_grad.zero_()

    def broadcast_params(self):
        """
        Syncs parameters across all DP ranks.
        """
        for param in self.module.parameters():
            torch.distributed.broadcast(
                param.data,
                src=parallel_state.get_data_parallel_src_rank(with_context_parallel=True),
                group=parallel_state.get_data_parallel_group(with_context_parallel=True),
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
