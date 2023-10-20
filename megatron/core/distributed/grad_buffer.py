# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from logging import getLogger
from typing import Dict, List

import torch

from .. import parallel_state

logger = getLogger(__name__)


def shard_buffer(buffer: torch.Tensor):
    """
    Shard buffer into dp_size chunks of equal size.
    """
    data_parallel_world_size = parallel_state.get_data_parallel_world_size()
    assert buffer.numel() % data_parallel_world_size == 0
    shard_size = buffer.numel() // data_parallel_world_size
    sharded_buffer = [
        buffer[(r * shard_size) : ((r + 1) * shard_size)] for r in range(data_parallel_world_size)
    ]
    return sharded_buffer


class Bucket:
    """
    Bucket to keep track of a subset of the model's gradients. Provides functionality to register
    when params in the bucket have grads ready to be synced; an asynchronous communication call
    is automatically launched when _all_ params in the bucket have grads ready.

    Arguments:
        params: List of parameters whose gradients are collated in this bucket.
        data: View in larger GradBuffer that this bucket is responsible for.
        offset: Offset of this bucket's view in the larger GradBuffer.
        data_parallel_group: Data-parallel process group.
        overlap_grad_reduce: If true, overlap communication with backprop computation by
            breaking up grads into buckets. If false, single synchronous communication call
            is used instead.
        use_distributed_optimizer: If true, issue reduce-scatter communication calls as part
            of distributed optimizer. If false, issue all-reduce communication calls.
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
        # The distributed optimizer needs to keep track of this bucket's offset
        # within the full grad_buffer.
        self.offset = offset
        self.data_parallel_group = data_parallel_group
        self.overlap_grad_reduce = overlap_grad_reduce
        self.use_distributed_optimizer = use_distributed_optimizer

        self.data_parallel_world_size = torch.distributed.get_world_size(group=data_parallel_group)
        self.data_parallel_rank = torch.distributed.get_rank(group=data_parallel_group)

        self.reset()

    def reset(self):
        """
        Reset metadata in bucket in preparation for the next iteration of training.
        """
        self.params_with_grad = set()
        self.communication_handle = None
        self.communication_issued = False

    def start_grad_sync(self):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operation
        for this bucket.

        When overlap_grad_reduce is set to True, dispatches an asynchronous
        communication call. When overlap_grad_reduce is set to False, makes
        synchronous call.
        """
        assert (
            self.communication_handle is None and not self.communication_issued
        ), 'Should not have multiple communication calls in flight at once'

        self.data /= self.data_parallel_world_size
        # Use async_op only when overlap_grad_reduce is True.
        if self.use_distributed_optimizer:
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

    def finish_grad_sync(self):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operation
        for this bucket.

        When overlap_grad_reduce is set to True, waits for asynchronous communication
        call to complete. When overlap_grad_reduce is set to False, makes synchronous call.
        """
        # If overlap_grad_reduce is False, start (and finish) synchronous communication call here.
        if not self.overlap_grad_reduce:
            self.start_grad_sync()
            return
        assert self.communication_handle is not None and self.communication_issued, (
            f'Communication call has not been issued for this bucket '
            f'({len(self.params_with_grad)}/{len(self.params)} params have grad available)'
        )
        self.communication_handle.wait()

    def register_grad_ready(self, param: torch.nn.Parameter):
        """
        Registers grads for the passed-in param to be "ready" for grad sync.

        When the number of microbatches is greater than 1, we only want to register
        grads as ready when processing the last microbatch and overlap_grad_reduce is True.
        """
        assert param in self.params, 'Param is not in the bucket'
        assert param not in self.params_with_grad, 'Cannot set grad twice'
        assert (
            self.overlap_grad_reduce
        ), 'register_grad_ready() should be called only when overlapping grad reduce'
        self.params_with_grad.add(param)
        # If all params in bucket have grads available, issue communication call.
        if len(self.params_with_grad) == len(self.params):
            self.start_grad_sync()


class GradBuffer:
    """
    Groups gradients into a contiguous buffer, and then breaks the buffer into buckets with
    roughly `bucket_size` parameters each.

    Arguments:
        numel: True number of elements.
        numel_padded: Number of elements in underlying tensor.
        dtype: Type of underlying tensor.
        params: List of parameters whose gradients are collated in the underlying tensor.
        data_parallel_group: Data-parallel process group.
        bucket_size: The rough size of each bucket in terms of number of parameters.
        param_to_name: Mapping from `torch.nn.Parameter` to name (for logging purposes).
        overlap_grad_reduce: If true, overlap communication with backprop computation by
            breaking up grads into buckets. If false, single synchronous communication call
            is used instead.
        use_distributed_optimizer: If true, issue reduce-scatter communication calls as part
            of distributed optimizer. If false, issue all-reduce communication calls.
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
        self.numel = numel
        self.numel_padded = numel_padded
        self.dtype = dtype
        self.data = torch.zeros(
            self.numel_padded,
            dtype=self.dtype,
            device=torch.cuda.current_device(),
            requires_grad=False,
        )

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
        def _set_bucket(
            bucket_params: List[torch.nn.Parameter], data_start_index: int, data_end_index: int
        ):

            # Get appropriate view into global GradBuffer.
            bucket_data = self._get(
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
            param.main_grad = self._get(param.data.shape, data_start_index)
            bucket_params.add(param)

            # If we have enough elements already, form a new buffer.
            # If bucket_size is None, accumulate everything into a single bucket.
            if bucket_size is not None:
                if (data_end_index - bucket_data_start_index) >= bucket_size:
                    _set_bucket(bucket_params, bucket_data_start_index, data_end_index)
                    bucket_data_start_index = data_end_index
                    bucket_params = set()
            data_start_index = data_end_index

        # Add remaining params to a new bucket.
        if len(bucket_params) > 0:
            _set_bucket(bucket_params, bucket_data_start_index, data_end_index)

        if not overlap_grad_reduce:
            assert len(bucket_params) == len(
                params
            ), 'All params should be in one bucket when overlap_grad_reduce is False'

        # Print buckets.
        if torch.distributed.get_rank() == 0:
            logger.info(
                f'Number of buckets for gradient all-reduce / reduce-scatter: {len(self.buckets)}'
            )
            for index, bucket in enumerate(self.buckets):
                numel = 0
                for param in bucket.params:
                    numel += param.data.nelement()
                logger.info(f'Params for bucket {index+1} ({numel} elements):')
                for param in bucket.params:
                    logger.info(f'    {param_to_name[param]}')

    def _get(self, shape: torch.Size, start_index: int) -> torch.Tensor:
        """
        Return a tensor with the input `shape` as a view into the 1-D data starting at
        `start_index`.
        """
        end_index = start_index + shape.numel()
        assert end_index <= self.numel, 'Requested tensor is out of buffer range'
        buffer_tensor = self.data[start_index:end_index]
        buffer_tensor = buffer_tensor.view(shape)
        return buffer_tensor

    def reset(self):
        """
        Zero out the underlying buffer and reset all buckets in preparation for the next
        iteration of training.
        """
        self.data.zero_()
        for bucket in self.buckets:
            bucket.reset()
        self.is_last_microbatch = True

    def start_grad_sync(self):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operations
        for all buckets in the grad buffer.

        When overlap_grad_reduce is set to True, dispatches asynchronous communication
        calls. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        for bucket in self.buckets:
            bucket.start_grad_sync()

    def finish_grad_sync(self):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operations
        for all buckets in the grad buffer.

        When overlap_grad_reduce is set to True, waits for asynchronous communication
        calls to complete. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        for bucket in self.buckets:
            bucket.finish_grad_sync()

    def register_grad_ready(self, param: torch.nn.Parameter):
        """
        Registers grads for the passed-in param to be "ready" for grad sync.

        When the number of microbatches is greater than 1, we only want to register
        grads as ready when processing the last microbatch and overlap_grad_reduce is True.
        """
        assert (
            self.overlap_grad_reduce
        ), 'register_grad_ready() should only be called when overlap_grad_reduce is True'
        if self.is_last_microbatch:
            bucket = self.param_to_bucket[param]
            bucket.register_grad_ready(param)
