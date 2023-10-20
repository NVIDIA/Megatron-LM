# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import math
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Dict, List

import torch

from megatron.core import mpu

from .module import MegatronModule


class MemoryBuffer:
    def __init__(self, numel: int, numel_padded: int, dtype: torch.dtype):
        self.numel = numel
        self.numel_padded = numel_padded
        self.dtype = dtype
        self.data = torch.zeros(
            self.numel_padded,
            dtype=self.dtype,
            device=torch.cuda.current_device(),
            requires_grad=False,
        )

    def zero(self):
        """Reset the buffer to zero."""
        self.data.zero_()

    def get(self, shape: torch.Size, start_index: int) -> torch.Tensor:
        """Return a tensor with the input `shape` as a view into the
        1-D data starting at `start_index`."""
        end_index = start_index + shape.numel()
        assert end_index <= self.numel, 'Requested tensor is out of buffer range'
        buffer_tensor = self.data[start_index:end_index]
        buffer_tensor = buffer_tensor.view(shape)
        return buffer_tensor


class Bucket:
    """
    Bucket to all-reduce / reduce-scatter gradients for a set of parameters asynchronously.
    Provides functionality to register when params in the bucket have grads available, and
    automatically launches an asynchronous communication call when _all_ params in the bucket
    have grads available.
    """

    def __init__(
        self,
        params: List[torch.nn.Parameter],
        data: torch.Tensor,
        offset: int,
        data_parallel_group: torch.distributed.ProcessGroup,
        overlap_grad_reduce: bool,
        use_distributed_optimizer: bool,
    ):
        # State for bookkeeping: params is the set of parameters this bucket is
        # responsible for, params_with_grad is the set of parameters with grads
        # available. When overlap_grad_reduce is True, communication (all-reduce
        # or reduce-scatter) is issued when params_with_grad equals params.
        self.params_list = params
        self.params = set(params)
        self.params_with_grad = set()
        self.data = data
        self.offset = offset  # Needed by distributed optimizer to keep track of this bucket's offset within the full grad_buffer.
        self.data_parallel_group = data_parallel_group
        self.overlap_grad_reduce = overlap_grad_reduce
        self.use_distributed_optimizer = use_distributed_optimizer

        self.data_parallel_world_size = torch.distributed.get_world_size(group=data_parallel_group)
        self.data_parallel_rank = torch.distributed.get_rank(group=data_parallel_group)

        self.reset()

    def reset(self):
        self.params_with_grad = set()
        self.communication_handle = None
        self.communication_issued = False

    def communicate(self):
        assert (
            self.communication_handle is None and not self.communication_issued
        ), 'Should not have multiple communication calls in flight at once'

        self.data /= self.data_parallel_world_size
        # Use async_op only when overlap_grad_reduce is True.
        if self.use_distributed_optimizer:
            # TODO: Move this import to top of file.
            # Import is here for now because of circular import errors.
            from megatron.optimizer.utils import shard_buffer
            local_data_view = shard_buffer(self.data)[self.data_parallel_rank]
            self.communication_handle = torch.distributed._reduce_scatter_base(
                local_data_view,
                self.data,
                group=self.data_parallel_group,
                async_op=self.overlap_grad_reduce,
            )
        else:
            self.communication_handle = torch.distributed.all_reduce(
                self.data, group=self.data_parallel_group, async_op=self.overlap_grad_reduce
            )
        self.communication_issued = True

    def set(self, param: torch.nn.Parameter):
        assert param in self.params, 'Param is not in the bucket'
        assert param not in self.params_with_grad, 'Cannot set grad twice'
        assert self.overlap_grad_reduce, 'set() should be called only when overlapping grad reduce'
        self.params_with_grad.add(param)
        # If all params in bucket have grads available, issue communication call.
        if len(self.params_with_grad) == len(self.params):
            self.communicate()

    def done(self):
        # If not overlapping grad reduce, issue synchronous communication call here.
        if not self.overlap_grad_reduce:
            self.communicate()
            return
        assert self.communication_handle is not None and self.communication_issued, (
            f'Communication call has not been issued for this bucket '
            f'({len(self.params_with_grad)}/{len(self.params)} params have grad available)'
        )
        self.communication_handle.wait()


class GradBuffer(MemoryBuffer):
    """
    Groups gradients into a contiguous buffer, and then breaks them into buckets with
    roughly bucket_size parameters each.
    """

    def __init__(
        self,
        numel: int,
        numel_padded: int,
        dtype: torch.dtype,
        params: List[torch.nn.Parameter],
        data_parallel_group: torch.distributed.ProcessGroup,
        bucket_size: int,
        param_to_name: Dict[torch.nn.Parameter, str],
        overlap_grad_reduce: bool,
        use_distributed_optimizer: bool,
    ):
        super(GradBuffer, self).__init__(numel, numel_padded, dtype)

        self.buckets = []
        self.param_to_bucket = {}
        self.param_to_bucket_index = {}
        self.overlap_grad_reduce = overlap_grad_reduce
        self.use_distributed_optimizer = use_distributed_optimizer

        self.is_last_microbatch = True

        # Check that params are unique.
        unique_params = set()
        for param in params:
            assert param not in unique_params
            unique_params.add(param)
        del unique_params

        # Helper function to create new bucket, add it to list of buckets, and
        # also update param->bucket mapping.
        def set_bucket_(
            bucket_params: List[torch.nn.Parameter], data_start_index: int, data_end_index: int
        ):

            # Get appropriate view into global GradBuffer.
            bucket_data = self.get(
                torch.Size([data_end_index - data_start_index]), data_start_index
            )
            bucket = Bucket(
                bucket_params,
                bucket_data,
                data_start_index,
                data_parallel_group,
                self.overlap_grad_reduce,
                self.use_distributed_optimizer,
            )
            self.buckets.append(bucket)
            for bucket_param in bucket_params:
                assert bucket_param not in self.param_to_bucket
                assert bucket_param not in self.param_to_bucket_index
                self.param_to_bucket[bucket_param] = bucket
                self.param_to_bucket_index[bucket_param] = len(self.buckets) - 1

        # Map the grads to the buffer and bucket them.
        data_start_index = 0
        bucket_data_start_index = data_start_index
        bucket_params = set()

        # Iterate through parameters in reverse order to roughly follow backprop order.
        for param in params[::-1]:
            # Skip parameters that don't require gradients.
            if not param.requires_grad:
                continue
            this_numel = param.data.nelement()
            data_end_index = data_start_index + this_numel
            param.main_grad = self.get(param.data.shape, data_start_index)
            bucket_params.add(param)

            # If we have enough elements already, form a new buffer.
            # If bucket_size is None, accumulate everything into a single bucket.
            if bucket_size is not None:
                if (data_end_index - bucket_data_start_index) >= bucket_size:
                    set_bucket_(bucket_params, bucket_data_start_index, data_end_index)
                    bucket_data_start_index = data_end_index
                    bucket_params = set()
            data_start_index = data_end_index

        # Add remaining params to a new bucket.
        if len(bucket_params) > 0:
            set_bucket_(bucket_params, bucket_data_start_index, data_end_index)

        if not overlap_grad_reduce:
            assert len(bucket_params) == len(
                params
            ), 'All params should be in one bucket when overlap_grad_reduce is False'

        # Print buckets.
        if torch.distributed.get_rank() == 0:
            print('> buckets for gradient all-reduce / reduce-scatter:')
            for index, bucket in enumerate(self.buckets):
                print(f'    params for bucket {index+1}')
                numel = 0
                for param in bucket.params:
                    numel += param.data.nelement()
                    print(f'      {param_to_name[param]}')
                print(f'     total number of elements: {numel}')

    def reset(self):
        """Set the data to zero and reset all buckets."""
        self.zero()
        for bucket in self.buckets:
            bucket.reset()
        self.is_last_microbatch = True

    def done(self):
        """Wait for all buckets' communication calls to complete."""
        for bucket in self.buckets:
            bucket.done()

    def grad_sync(self):
        """Synchronize grads."""
        for bucket in self.buckets:
            bucket.communicate()

    def mark_grad_as_done(self, param: torch.nn.Parameter):
        """
        When the number of microbatches is greater than 1, we only want
        to register grads when processing the last microbatch and
        overlap_grad_reduce is True.
        """
        assert (
            self.overlap_grad_reduce
        ), 'mark_grad_as_done() should only be called when overlap_grad_reduce is True'
        if self.is_last_microbatch:
            bucket = self.param_to_bucket[param]
            bucket.set(param)


class DistributedDataParallelBase(MegatronModule, ABC):
    """Abstract class for DDP."""

    def __init__(self, module):
        super(DistributedDataParallelBase, self).__init__()
        # Keep a pointer to the model.
        self.module = module

    @abstractmethod
    def sync_gradients(self):
        pass

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def state_dict(self, prefix='', keep_vars=False):
        return self.module.state_dict(prefix=prefix, keep_vars=keep_vars)

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        return self.module.state_dict_for_save_checkpoint(prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        self.module.load_state_dict(state_dict, strict=strict)


class DistributedDataParallel(DistributedDataParallelBase):
    """
    DDP wrapper which stores grads in contiguous buffers. Also has option of
    overlapping communication with backprop computation by breaking up full model's
    gradients into smaller buckets and running all-reduce / reduce-scatter
    on each bucket asynchronously.
    This class:
        - has the potential to reduce memory fragmentation.
        - provides the option to do the gradient accumulation
          in a type other than the params type (e.g., fp32).

    Arguments:
        module: input model.
        data_parallel_group: data-parallel group.
        accumulate_allreduce_grads_in_fp32: if true do the gradient accumulation
            and communication in float32.
        overlap_grad_reduce: if true, overlap communication with backprop
            computation by breaking up grads into buckets. If false, single
            synchronous communication call is used instead.
        use_distributed_optimizer: if true, issue reduce-scatter communication
            calls as part of distributed optimizer. If false, issue all-reducde
            communication calls.

    """

    def __init__(
        self,
        module: torch.nn.Module,
        data_parallel_group: torch.distributed.ProcessGroup,
        accumulate_allreduce_grads_in_fp32: bool,
        overlap_grad_reduce: bool,
        use_distributed_optimizer: bool,
        bucket_size: int = 40000000,
    ):
        super(DistributedDataParallel, self).__init__(module)

        # Set bucket_size to infinity if overlap_grad_reduce is False.
        self.overlap_grad_reduce = overlap_grad_reduce
        self.use_distributed_optimizer = use_distributed_optimizer

        if not self.overlap_grad_reduce:
            bucket_size = None
        self.bucket_size = bucket_size

        self.module = module
        self.grad_buffers = {}
        self.expert_grads = []
        self.grad_buffer_param_index_map = {}
        self.param_to_grad_buffer = {}

        # Group parameters by their gradient type.
        grad_dtype_to_params = {}
        grad_dtype_to_numel = {}
        param_to_name = {}
        for name, param in self.module.named_parameters():
            if param.requires_grad and getattr(param, 'allreduce', True):
                param.grad_added_to_main_grad = False
                param_to_name[param] = name
                dtype = torch.float if accumulate_allreduce_grads_in_fp32 else param.dtype

                params = grad_dtype_to_params.get(dtype, [])
                params.append(param)
                grad_dtype_to_params[dtype] = params

                # Calculate number of elements per dtype.
                grad_dtype_to_numel[dtype] = (
                    grad_dtype_to_numel.get(dtype, 0) + param.data.nelement()
                )

        # Allocate the grad buffers and map the grads.
        # The grad buffer under the hood creates buckets as appropriate, depending on
        # whether overlap_grad_reduce is True or not.
        data_parallel_world_size = torch.distributed.get_world_size(group=data_parallel_group)
        for dtype, params in grad_dtype_to_params.items():
            # Pad so size is divisible by the data parallel size.
            numel = grad_dtype_to_numel[dtype]
            numel_padded = (
                int(math.ceil(numel / data_parallel_world_size)) * data_parallel_world_size
            )

            self.grad_buffers[dtype] = GradBuffer(
                numel,
                numel_padded,
                dtype,
                params,
                data_parallel_group,
                bucket_size,
                param_to_name,
                self.overlap_grad_reduce,
                self.use_distributed_optimizer,
            )

            # Parameters are laid out in the corresponding grad_buffer in reverse
            # order, so count indices from the back.
            index = grad_dtype_to_numel[dtype]
            for param in params:
                self.param_to_grad_buffer[param] = self.grad_buffers[dtype]
                if dtype not in self.grad_buffer_param_index_map:
                    self.grad_buffer_param_index_map[dtype] = {}

                index -= param.data.nelement()
                # Store the indices / bucket of each param.
                self.grad_buffer_param_index_map[dtype][param] = (
                    index,
                    index + param.data.nelement(),
                    self.grad_buffers[dtype].param_to_bucket_index[param],
                )

        # Allocate discreate buffer for MoE params' grads
        for param in self.module.parameters():
            if param.requires_grad and not getattr(param, 'allreduce', True):
                dtype = torch.float if accumulate_allreduce_grads_in_fp32 else param.dtype
                param.main_grad = \
                    torch.zeros(param.data.shape,
                                dtype=dtype,
                                device=torch.cuda.current_device(),
                                requires_grad=False)
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

    def _make_param_hook(
        self, param: torch.nn.Parameter, param_to_grad_buffer: Dict[torch.nn.Parameter, GradBuffer]
    ):
        """Create the all-reduce / reduce-scatter hook for backprop."""

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
                    param_to_grad_buffer[param].mark_grad_as_done(param)

        return param_hook

    @contextmanager
    def no_sync(self):
        """Context manager that turns off gradient synchronization."""
        for grad_buffer in self.grad_buffers.values():
            grad_buffer.is_last_microbatch = False
        try:
            yield
        finally:
            for grad_buffer in self.grad_buffers.values():
                grad_buffer.is_last_microbatch = True

    def grad_sync(self, *unused):
        """Method to dispatch grad sync operations."""
        for grad_buffer in self.grad_buffers.values():
            grad_buffer.grad_sync()

    def zero_grad_buffer(self):
        """Set the grad buffer data to zero. Needs to be called at the
        begining of each iteration."""
        for param in self.module.parameters():
            if param.requires_grad:
                param.grad_added_to_main_grad = False
        for grad_buffer in self.grad_buffers.values():
            grad_buffer.reset()
        for expert_grad in self.expert_grads:
            expert_grad.zero_()

    def broadcast_params(self):
        """Sync params across all DP ranks."""
        for param in self.module.parameters():
            torch.distributed.broadcast(
                param.data,
                src=mpu.get_data_parallel_src_rank(),
                group=mpu.get_data_parallel_group(),
            )

    def sync_gradients(self):
        """
        Reduce gradients across data-parallel ranks.
        When overlap_grad_reduce is set to True, waits for asynchronous
        communication calls to complete.
        When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        for grad_buffer in self.grad_buffers.values():
            grad_buffer.done()
