# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC
from abc import abstractmethod
import math

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from contextlib import contextmanager

from megatron import get_args
from megatron.core import mpu
from .module import MegatronModule


class MemoryBuffer:

    def __init__(self, numel, numel_padded, dtype):
        self.numel = numel
        self.numel_padded = numel_padded
        self.dtype = dtype
        self.data = torch.zeros(self.numel_padded,
                                dtype=self.dtype,
                                device=torch.cuda.current_device(),
                                requires_grad=False)


    def zero(self):
        """Reset the buffer to zero."""
        self.data.zero_()


    def get(self, shape, start_index):
        """Return a tensor with the input `shape` as a view into the
        1-D data starting at `start_index`."""
        end_index = start_index + shape.numel()
        assert end_index <= self.numel, \
            'requested tensor is out of the buffer range.'
        buffer_tensor = self.data[start_index:end_index]
        buffer_tensor = buffer_tensor.view(shape)
        return buffer_tensor



class Bucket:
    """
    Bucket to all-reduce gradients for a set of parameters asynchronously. Provides
    functionality to register when params in the bucket have grads available, and
    automatically launches an asynchronous all_reduce when _all_ params in the bucket
    have grads available.
    """

    def __init__(self, params, data, data_parallel_group, overlap_grad_reduce):
        # State for bookkeeping: params is the set of parameters this bucket is
        # responsible for, params_with_grad is the set of parameters with grads
        # available.
        self.params = set(params)
        self.params_with_grad = set()
        self.data = data
        self.data_parallel_group = data_parallel_group
        self.overlap_grad_reduce = overlap_grad_reduce
        
        self.one_over_data_parallel_size = 1.0 / \
            torch.distributed.get_world_size(group=data_parallel_group)

        self.reset()


    def reset(self):
        self.params_with_grad = set()
        self.allreduce_handle = None
        self.allreduce_issued = False


    def all_reduce(self):
        assert self.allreduce_handle is None, 'allreduce handle is not None'
        assert not self.allreduce_issued, 'allreduce is already issued'
        self.data.mul_(self.one_over_data_parallel_size)
        self.allreduce_handle = torch.distributed.all_reduce(
            self.data, group=self.data_parallel_group,
            async_op=self.overlap_grad_reduce)  # Use async_op only when overlap_grad_reduce is True.
        self.allreduce_issued = True
        

    def set(self, param):
        assert param in self.params, 'param is not in the bucket'
        assert param not in self.params_with_grad, 'cannot set grad twice'
        self.params_with_grad.add(param)
        if self.overlap_grad_reduce and len(self.params_with_grad) == len(self.params):
            self.all_reduce()


    def done(self):
        if not self.overlap_grad_reduce:
            self.all_reduce()
            return
        assert self.allreduce_issued, 'allreduce is not issued for this bucket'
        if self.allreduce_handle is not None:
            self.allreduce_handle.wait()
        self.addreduce_handle = None
        self.allreduce_issued = False
    
    

class GradBuffer(MemoryBuffer):
    """
    Groups gradients into a contiguous buffer, and then breaks them into buckets with
    roughly bucket_size parameters each.
    """
    
    def __init__(self, numel, numel_padded, dtype, params, data_parallel_group,
                 bucket_size, param_to_name, overlap_grad_reduce):
        super(GradBuffer, self).__init__(numel, numel_padded, dtype)

        self.buckets = []
        self.param_to_bucket = {}

        self.is_last_microbatch = False
        
        # Check that params are unique.
        unique_params = set()
        for param in params:
            assert param not in unique_params
            unique_params.add(param)
        del unique_params

        # Map the grads to the buffer and bucket them.
        def set_bucket_(bucket_params, data_start_index, data_end_index):
            bucket_data = self.get(torch.Size([data_end_index - data_start_index]),
                                   data_start_index)
            bucket = Bucket(bucket_params, bucket_data, data_parallel_group, overlap_grad_reduce)
            self.buckets.append(bucket)
            for bucket_param in bucket_params:
                self.param_to_bucket[bucket_param] = bucket

        data_start_index = 0
        bucket_data_start_index = data_start_index
        bucket_params = set()
        for param in params:
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

        # Print buckets.
        if torch.distributed.get_rank() == 0:
            print('> buckets for gradient all-reduce:')
            for index, bucket in enumerate(self.buckets):
                print('    params for bucket {}'.format(index + 1))
                numel = 0
                for param in bucket.params:
                    numel += param.data.nelement()
                    print('      {}'.format(param_to_name[param]))
                print('     total number of elements: {}'.format(numel))


    def reset(self):
        # Set the data to zero and reset all the buckets.
        self.zero()
        for bucket in self.buckets:
            bucket.reset()
        self.is_last_microbatch = False


    def done(self):
        # Wait for all buckets' all-reductions to complete.
        for bucket in self.buckets:
            bucket.done()
        

    def mark_grad_as_done(self, param):
        # Note that when the number of microbatches is greater than 1,
        # we only want to register grads when processing the last microbatch.
        # This method is called from the backward hook.
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
    def allreduce_gradients(self):
        pass


    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)


    def state_dict(self, prefix='', keep_vars=False):
        return self.module.state_dict(prefix=prefix, keep_vars=keep_vars)


    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        return self.module.state_dict_for_save_checkpoint(prefix=prefix,
                                                          keep_vars=keep_vars)


    def load_state_dict(self, state_dict, strict=True):
        self.module.load_state_dict(state_dict, strict=strict)



class OverlappingDistributedDataParallel(DistributedDataParallelBase):
    """
    DDP wrapper which stores grads in contiguous buffers. Also has option of
    overlapping all-reduce with computation by breaking up full model's
    gradients into smaller buckets and running all-reduce on each bucket
    asynchronously.
    This class:
        - has the potential to reduce memory fragmentation.
        - provides the option to do the gradient accumulation
          in a type other than the params type (e.g., fp32).

    Arguments:
        module: input model.
        data_parallel_group: data-parallel group.
        accumulate_allreduce_grads_in_fp32: if true do the gradient accumulation
            and the gradient all-reduce all in in float32.
        overlap_grad_reduce: if true, overlap all-reduce with computation by
            breaking up grads into buckets. If false, single synchronous all-reduce
            is used instead.

    """

    def __init__(self, module, data_parallel_group,
                 accumulate_allreduce_grads_in_fp32,
                 overlap_grad_reduce):
        super(OverlappingDistributedDataParallel, self).__init__(module)        

        # Set bucket_size to infinity if overlap_grad_reduce is False.
        bucket_size = None
        if overlap_grad_reduce:
            bucket_size = 40000000
        
        self.module = module
        self.grad_dtype_to_grad_buffer = {}
        self.param_to_grad_buffer = {}

        # Group parameters by their gradient type.
        grad_dtype_to_param = {}
        grad_dtype_to_numel = {}
        param_to_name = {}
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                param.grad_added_to_main_grad = False
                param_to_name[param] = name
                dtype = torch.float if accumulate_allreduce_grads_in_fp32 else param.dtype

                params = grad_dtype_to_param.get(dtype, [])
                params.append(param)
                grad_dtype_to_param[dtype] = params

                # Calculate number of elements per dtype.
                grad_dtype_to_numel[dtype] = grad_dtype_to_numel.get(dtype, 0) + param.data.nelement()

        # Allocate the grad buffers and map the grads. Make sure parameters are reversed
        # so they are in approximately in the order of backprop.
        # The grad buffer under the hood creates buckets as appropriate, depending on
        # whether overlap_grad_reduce is True or not.
        data_parallel_size = torch.distributed.get_world_size(
            group=data_parallel_group)
        for dtype, params in grad_dtype_to_param.items():
            params.reverse()

            # Pad so size is divisible by the data parallel size.
            numel = grad_dtype_to_numel[dtype]
            numel_padded = int(math.ceil(numel / data_parallel_size)) * data_parallel_size

            self.grad_dtype_to_grad_buffer[dtype] = GradBuffer(
                numel, numel_padded, dtype, params, data_parallel_group,
                bucket_size, param_to_name, overlap_grad_reduce)
            for param in params:
                self.param_to_grad_buffer[param] = self.grad_dtype_to_grad_buffer[dtype]

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
                grad_acc.register_hook(self._make_param_hook(
                    param, self.param_to_grad_buffer))
                self.grad_accs.append(grad_acc)


    def _make_param_hook(self, param, param_to_grad_buffer):
        """Create the all-reduce hook for backprop."""

        def param_hook(*unused):
            if param.requires_grad:
                # Make sure no none values are returned.
                assert param.grad is not None
                if not param.grad_added_to_main_grad:
                    param.main_grad.add_(param.grad.data)
                param.grad = None
                param_to_grad_buffer[param].mark_grad_as_done(param)

        return param_hook


    @contextmanager
    def no_sync(self):
        """Context manager that turns off gradient synchronization."""
        for grad_buffer in self.grad_dtype_to_grad_buffer.values():
            grad_buffer.is_last_microbatch = False
        try:
            yield
        finally:
            for grad_buffer in self.grad_dtype_to_grad_buffer.values():
                grad_buffer.is_last_microbatch = True


    def zero_grad_buffer(self):
        """Set the grad buffer data to zero. Needs to be called at the
        begining of each iteration."""
        for param in self.module.parameters():
            if param.requires_grad:
                param.grad_added_to_main_grad = False
        for grad_buffer in self.grad_dtype_to_grad_buffer.values():
            grad_buffer.reset()


    def broadcast_params(self):
        """
        Sync params across all DP ranks.
        """
        for param in self.module.parameters():
            torch.distributed.broadcast(param.data,
                                        src=mpu.get_data_parallel_src_rank(),
                                        group=mpu.get_data_parallel_group())


    def allreduce_gradients(self):
        """
        Reduce gradients across data parallel ranks.
        When overlap_grad_reduce is set to True, waits for asynchronous all-reduces
        to complete.
        When overlap_grad_reduce is set to False, calls synchronous
        all-reduce.
        """
        for grad_buffer in self.grad_dtype_to_grad_buffer.values():
            grad_buffer.done()


    
class DistributedDataParallel(DistributedDataParallelBase):
    """DDP with contiguous buffers options to store and accumulate gradients.
    This class:
        - has the potential to reduce memory fragmentation.
        - provides the option to do the gradient accumulation
          in a type other than the params type (for example fp32)

    Arguments:
        module: input model.
        accumulate_allreduce_grads_in_fp32: if true do the gradient accumulation
            and the gradient all-reduce all in in float32. If this option is
            true, we require `use_contiguous_buffers` to be true too.
        use_contiguous_buffers: if true, use a contiguous buffer to store the
            gradients.
    """

    def __init__(self, module,
                 accumulate_allreduce_grads_in_fp32,
                 use_contiguous_buffers):

        super(DistributedDataParallel, self).__init__(module)

        self.accumulate_allreduce_grads_in_fp32 \
            = accumulate_allreduce_grads_in_fp32
        self.use_contiguous_buffers = use_contiguous_buffers
        # If we are using fp32-accumulate-allreduce explicitly
        # this means we need main grads in a continous buffer.
        if self.accumulate_allreduce_grads_in_fp32:
            assert self.use_contiguous_buffers

        # ===================================
        # Rest of this part applies only to
        # the case we use continuous buffers.
        # ===================================
        self._grad_buffers = None
        self._grad_buffer_param_index_map = None
        if self.use_contiguous_buffers:
            self._grad_buffers = {}
            self._grad_buffer_param_index_map = {}
            data_parallel_world_size = mpu.get_data_parallel_world_size()

            # Simple function to define buffer type.
            def _get_buffer_type(param):
                return torch.float if \
                    self.accumulate_allreduce_grads_in_fp32 else param.dtype

            # First calculate total number of elements per type.
            type_num_elements = {}
            for param in self.module.parameters():
                if param.requires_grad:
                    dtype = _get_buffer_type(param)
                    type_num_elements[dtype] = type_num_elements.get(dtype, 0) \
                                               + param.data.nelement()

            # Allocate the buffer.
            for dtype, num_elements in type_num_elements.items():

                # If using distributed optimizer, pad memory buffer to be
                # multiple of data_parallel_world_size. (This padding is done
                # due to a constraint with the reduce_scatter op, which requires
                # all tensors have equal size. See: optimizer.py.)
                num_elements_padded = data_parallel_world_size * \
                    int(math.ceil(num_elements / data_parallel_world_size))

                # Allocate grad buffer.
                self._grad_buffers[dtype] = MemoryBuffer(num_elements,
                                                         num_elements_padded,
                                                         dtype)

            # Assume the back prop order is reverse the params order,
            # store the start index for the gradients.
            for param in self.module.parameters():
                if param.requires_grad:
                    dtype = _get_buffer_type(param)
                    type_num_elements[dtype] -= param.data.nelement()
                    param.main_grad = self._grad_buffers[dtype].get(
                        param.data.shape, type_num_elements[dtype])
                    if dtype not in self._grad_buffer_param_index_map:
                        self._grad_buffer_param_index_map[dtype] = {}
                    self._grad_buffer_param_index_map[dtype][param] = (
                        type_num_elements[dtype],
                        type_num_elements[dtype] + param.data.nelement(),
                    )

            # Backward hook.
            # Accumalation function for the gradients. We need
            # to store them so they don't go out of scope.
            self.grad_accs = []
            # Loop over all the parameters in the model.
            for param in self.module.parameters():
                if param.requires_grad:
                    # Expand so we get access to grad_fn.
                    param_tmp = param.expand_as(param)
                    # Get the gradient accumulator functtion.
                    grad_acc = param_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_param_hook(param))
                    self.grad_accs.append(grad_acc)


    def _make_param_hook(self, param):
        """Create the all-reduce hook for backprop."""
        # Hook used for back-prop.
        def param_hook(*unused):
            # Add the gradient to the buffer.
            if param.grad is not None:
                # The gradient function of linear layers is fused with GEMMs
                param.main_grad.add_(param.grad.data)
                # Now we can deallocate grad memory.
                param.grad = None
        return param_hook


    def zero_grad_buffer(self):
        """Set the grad buffer data to zero. Needs to be called at the
        begining of each iteration."""
        assert self._grad_buffers is not None, 'buffers are not initialized.'
        for _, buffer_ in self._grad_buffers.items():
            buffer_.zero()


    def broadcast_params(self):
        for param in self.module.parameters():
            torch.distributed.broadcast(param.data,
                                        src=mpu.get_data_parallel_src_rank(),
                                        group=mpu.get_data_parallel_group())


    def allreduce_gradients(self):
        """Reduce gradients across data parallel ranks."""
        # If we have buffers, simply reduce the data in the buffer.
        if self._grad_buffers is not None:
            for _, buffer_ in self._grad_buffers.items():
                buffer_.data /= mpu.get_data_parallel_world_size()
                torch.distributed.all_reduce(
                    buffer_.data, group=mpu.get_data_parallel_group())
        else:
            # Otherwise, bucketize and all-reduce
            buckets = {}
            # Pack the buckets.
            for param in self.module.parameters():
                if param.requires_grad and param.grad is not None:
                    tp = param.data.type()
                    if tp not in buckets:
                        buckets[tp] = []
                    buckets[tp].append(param)
                    param.main_grad = param.grad

            # For each bucket, all-reduce and copy all-reduced grads.
            for tp in buckets:
                bucket = buckets[tp]
                grads = [param.grad.data for param in bucket]
                coalesced = _flatten_dense_tensors(grads)
                coalesced /= mpu.get_data_parallel_world_size()
                torch.distributed.all_reduce(
                    coalesced, group=mpu.get_data_parallel_group())
                for buf, synced in zip(grads, _unflatten_dense_tensors(
                        coalesced, grads)):
                    buf.copy_(synced)
