# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

# This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”). All Bytedance's Modifications are Copyright 2024 Bytedance Ltd. and/or its affiliates. 

import math
from logging import getLogger
from typing import Dict, List
from functools import reduce

import torch

from .. import parallel_state
from ...quantization_helper import QuantizationHelper

logger = getLogger(__name__)


def shard_buffer(buffer: torch.Tensor, data_parallel_world_size: int):
    """
    Shard buffer into data_parallel_world_size chunks of equal size.
    """
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
        numel_unpadded: Number of unpadded elements in bucket.
        data_parallel_group: Data-parallel process group.
        data_parallel_world_size: World size using the data-parallel group group.
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
        numel_unpadded: int,
        data_parallel_group: torch.distributed.ProcessGroup,
        data_parallel_world_size: int,
        overlap_grad_reduce: bool,
        grad_reduce_stream: torch.cuda.Stream,
        use_distributed_optimizer: bool,
        quantization_helper: QuantizationHelper,
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
        self.numel_unpadded = numel_unpadded
        self.data_parallel_group = data_parallel_group
        self.data_parallel_world_size = data_parallel_world_size
        self.data_parallel_rank = torch.distributed.get_rank(group=data_parallel_group)
        self.overlap_grad_reduce = overlap_grad_reduce
        self.use_distributed_optimizer = use_distributed_optimizer
        self.quantization_helper = quantization_helper
        if self.overlap_grad_reduce:
            assert isinstance(grad_reduce_stream, torch.cuda.Stream)
            self.comm_stream = grad_reduce_stream
        else:
            self.comm_stream = torch.cuda.default_stream()
        self.reset()

    def reset(self):
        """
        Reset metadata in bucket in preparation for the next iteration of training.
        """
        self.params_with_grad = set()
        self.communication_event: torch.cuda.Event = None
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
            self.communication_event is None and not self.communication_issued
        ), 'Should not have multiple communication calls in flight at once'

        stream = self.comm_stream
        
        event = torch.cuda.Event()
        self.communication_event = event
        self.communication_issued = True
        self.data /= self.data_parallel_world_size
        # Use async_op only when overlap_grad_reduce is True.
        if self.use_distributed_optimizer:
            local_data_view = shard_buffer(self.data, self.data_parallel_world_size)[
                self.data_parallel_rank
            ]
            if self.quantization_helper and self.quantization_helper.quantized_gradients:
                stream.wait_stream(torch.cuda.default_stream())
                with torch.cuda.stream(stream):
                    self.quantization_helper.quantize_reduce_gradients(self.data, local_data_view)
                    event.record()
            else:
                stream.wait_stream(torch.cuda.default_stream())
                with torch.cuda.stream(stream):
                    torch.distributed._reduce_scatter_base(
                        local_data_view,
                        self.data,
                        group=self.data_parallel_group,
                        async_op=False,
                    )
                    event.record()
        else:
            stream.wait_stream(torch.cuda.default_stream())
            with torch.cuda.stream(stream):
                torch.distributed.all_reduce(
                    self.data, group=self.data_parallel_group, async_op=False
                )
                event.record()
        if not self.overlap_grad_reduce:
            torch.cuda.current_stream().wait_stream(self.comm_stream)
            self.communication_event = None

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
        assert self.communication_event is not None and self.communication_issued, (
            f'Communication call has not been issued for this bucket '
            f'({len(self.params_with_grad)}/{len(self.params)} params have grad available)'
        )
        torch.cuda.current_stream().wait_stream(self.comm_stream)
        self.communication_event = None

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
        dtype: torch.dtype,
        params: List[torch.nn.Parameter],
        data_parallel_group: torch.distributed.ProcessGroup,
        bucket_size: int,
        param_to_name: Dict[torch.nn.Parameter, str],
        overlap_grad_reduce: bool,
        use_distributed_optimizer: bool,
        quantization_helper = QuantizationHelper
    ):

        # Check that params are unique.
        unique_params = set()
        for param in params:
            assert param not in unique_params
            unique_params.add(param)
        del unique_params

        # Store attributes that will be needed later.
        self.dtype = dtype
        self.data_parallel_group = data_parallel_group
        self.data_parallel_world_size = torch.distributed.get_world_size(
            group=self.data_parallel_group
        )
        self.overlap_grad_reduce = overlap_grad_reduce
        self.grad_reduce_stream = None
        if self.overlap_grad_reduce:
            self.grad_reduce_stream = torch.cuda.Stream()
        self.use_distributed_optimizer = use_distributed_optimizer
        self.quantization_helper = quantization_helper
        self.is_last_microbatch = True

        # Data structures to store underlying buckets and relevant indexing data.
        self.buckets = []
        self.param_to_bucket = {}  # Param -> bucket mapping.
        self.param_index_map = {}  # Param -> location in buffer mapping (used in dist. optimizer).



        def _pad_if_needed(data_index: int):
            bucket_size_divisible_by = 1
            if use_distributed_optimizer and quantization_helper is not None and (quantization_helper.quantized_weights or quantization_helper.quantized_gradients):
                def least_common_multiple(divisors):
                    """Find least common multiple of a list of numbers."""
                    lcm_value = reduce(lambda x, y: x * y // math.gcd(x, y), divisors)
                    return lcm_value
                weight_quantization_pad = 1
                gradient_quantization_pad = 1
                if quantization_helper.quantized_weights:
                    weight_quantization_pad = quantization_helper.wq_group_size
                if quantization_helper.quantized_gradients:
                    gradient_quantization_pad = least_common_multiple([quantization_helper.gq_group_size_intra * quantization_helper.gradient_alltoall_pipeline, quantization_helper.gq_group_size_inter])
                bucket_size_divisible_by = least_common_multiple([weight_quantization_pad, gradient_quantization_pad]) * self.data_parallel_world_size
                return (
                    int(math.ceil(data_index / bucket_size_divisible_by))
                    * bucket_size_divisible_by
                )
            """Pads data indices if using distributed optimizer (to ensure uniform sharding)."""
            if use_distributed_optimizer:
                return (
                    int(math.ceil(data_index / self.data_parallel_world_size))
                    * self.data_parallel_world_size
                )
            return data_index

        # First, figure out how many elements should be in the underlying buffer storage.
        # Note that if we need to split the buffer into smaller buckets, each of these
        # might need to be padded as well (if using the distributed optimizer).
        data_start_index = 0
        bucket_data_start_index = data_start_index
        bucket_params = set()
        self.bucket_indices = []
        per_bucket_numel_unpadded = []
        bucket_id = 0
        for param in params[::-1]:
            # Iterate through parameters in reverse order to roughly follow backprop order,
            # and skip parameters that don't require gradients.
            if not param.requires_grad:
                continue
            this_numel = param.data.nelement()
            data_end_index = data_start_index + this_numel
            self.param_index_map[param] = (
                data_start_index,
                data_end_index,
                bucket_id,
            )
            bucket_params.add(param)

            # If we have enough elements already, form a new bucket.
            # If bucket_size is None, accumulate everything into a single bucket.

            # TODO: Remove len(bucket_params) > 1 when the final head that transforms token
            # representations from hidden space to vocabulary space is in a PyTorch module
            # whose forward method is called. If it is not and a bucket contains only this
            # one parameter, we get incorrect behavior (i.e., higher losses) since we do not
            # call the wait function on the bucket's all_gather_handle (we use forward pre-
            # hooks on PyTorch modules to do this when --overlap-param-gather is used).
            # As a temporary workaround, we make sure that no bucket has only one parameter.
            if bucket_size is not None:
                if (data_end_index - bucket_data_start_index) >= bucket_size and len(
                    bucket_params
                ) > 1:
                    per_bucket_numel_unpadded.append(data_end_index - bucket_data_start_index)
                    data_end_index = _pad_if_needed(data_end_index)
                    self.bucket_indices.append((bucket_data_start_index, data_end_index))
                    bucket_data_start_index = data_end_index
                    bucket_params = set()
                    bucket_id += 1
            data_start_index = data_end_index

        # Add remaining params to a new bucket.
        if len(bucket_params) > 0:
            per_bucket_numel_unpadded.append(data_end_index - bucket_data_start_index)
            data_end_index = _pad_if_needed(data_end_index)
            self.bucket_indices.append((bucket_data_start_index, data_end_index))

        # Next, create underlying storage for buffer (with numel elements that includes
        # padding as necessary).
        self.numel = data_end_index
        if use_distributed_optimizer:
            assert self.numel % self.data_parallel_world_size == 0
        self.data = torch.zeros(
            self.numel, dtype=self.dtype, device=torch.cuda.current_device(), requires_grad=False,
        )

        # Finally, map main_grad fields for each parameter with a .grad field.
        bucket_params = set()
        bucket_data_start_index = 0
        cur_bucket_id = 0
        for param in params[::-1]:
            if not param.requires_grad:
                continue
            data_start_index, data_end_index, bucket_id = self.param_index_map[param]
            param.main_grad = self._get(param.data.shape, data_start_index)
            if bucket_id != cur_bucket_id:
                bucket_data_end_index = _pad_if_needed(data_start_index)
                self._set_bucket(
                    bucket_params=bucket_params,
                    start_index=bucket_data_start_index,
                    end_index=bucket_data_end_index,
                    numel_unpadded=per_bucket_numel_unpadded[cur_bucket_id],
                    bucket_id=cur_bucket_id,
                )
                bucket_data_start_index = bucket_data_end_index
                bucket_params = set()
                assert cur_bucket_id + 1 == len(self.buckets)
                assert bucket_id == cur_bucket_id + 1
                cur_bucket_id = bucket_id
            bucket_params.add(param)

        # Add remaining params to a new bucket.
        if len(bucket_params) > 0:
            bucket_data_end_index = _pad_if_needed(data_end_index)
            self._set_bucket(
                bucket_params=bucket_params,
                start_index=bucket_data_start_index,
                end_index=bucket_data_end_index,
                numel_unpadded=per_bucket_numel_unpadded[cur_bucket_id],
                bucket_id=cur_bucket_id,
            )

        if not overlap_grad_reduce:
            assert len(bucket_params) == len(
                params
            ), 'All params should be in one bucket when overlap_grad_reduce is False'

        # Log buckets for all PP stages.
        if (
            parallel_state.get_data_parallel_rank(with_context_parallel=True) == 0
            and parallel_state.get_tensor_model_parallel_rank() == 0
        ):
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

    def _set_bucket(
        self,
        bucket_params: List[torch.nn.Parameter],
        start_index: int,
        end_index: int,
        numel_unpadded: int,
        bucket_id: int,
    ):
        """
        Helper function to create new bucket, add it to list of buckets, and
        also update param->bucket mapping.
        """

        # Assert that indices are correctly padded (if needed), and that bucket
        # position is same as originally computed.
        if self.use_distributed_optimizer:
            assert start_index % self.data_parallel_world_size == 0
            assert end_index % self.data_parallel_world_size == 0
        assert (start_index, end_index) == self.bucket_indices[bucket_id]

        # Get appropriate view into global GradBuffer.
        bucket_data = self._get(torch.Size([end_index - start_index]), start_index)
        bucket = Bucket(
            params=bucket_params,
            data=bucket_data,
            offset=start_index,
            numel_unpadded=numel_unpadded,
            data_parallel_group=self.data_parallel_group,
            data_parallel_world_size=self.data_parallel_world_size,
            overlap_grad_reduce=self.overlap_grad_reduce,
            grad_reduce_stream=self.grad_reduce_stream,
            use_distributed_optimizer=self.use_distributed_optimizer,
            quantization_helper=self.quantization_helper,
        )
        self.buckets.append(bucket)
        for bucket_param in bucket_params:
            assert bucket_param not in self.param_to_bucket
            self.param_to_bucket[bucket_param] = bucket

    def reset(self, zero_buffer):
        """
        Zero out the underlying buffer and reset all buckets in preparation for the next
        iteration of training.

        When zero_buffer is set to True, the underlying buffer is zeroed out.
        """
        if zero_buffer:
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
