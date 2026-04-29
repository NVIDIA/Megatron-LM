# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import fnmatch
import functools
import logging
import math
import warnings
from contextlib import nullcontext
from enum import Enum
from functools import partial
from typing import Dict, List, Optional, Tuple

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.distributed import _coalescing_manager

import megatron.core.nccl_allocator as nccl_allocator
from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.utils import log_single_rank

from ..fp4_utils import get_nvfp4_rowwise_packed_shape, is_nvfp4tensor
from ..fp8_utils import (
    is_float8tensor,
    is_mxfp8tensor,
    modify_underlying_storage,
    post_all_gather_processing,
)
from ..optimizer.param_layout import pad_bucket_end, pad_param_start
from ..utils import is_torch_min_version, log_on_each_pipeline_stage
from .distributed_data_parallel_config import DistributedDataParallelConfig
from .reduce_scatter_with_fp32_accumulation import reduce_scatter_with_fp32_accumulation

logger = logging.getLogger(__name__)

try:
    if is_torch_min_version("1.13.0"):
        dist_all_gather_func = torch.distributed.all_gather_into_tensor
        dist_reduce_scatter_func = torch.distributed.reduce_scatter_tensor
    else:
        dist_all_gather_func = torch.distributed._all_gather_base
        dist_reduce_scatter_func = torch.distributed._reduce_scatter_base
except:
    dist_all_gather_func = torch.distributed._all_gather_base
    dist_reduce_scatter_func = torch.distributed._reduce_scatter_base

import megatron.core.nccl_allocator as nccl_allocator


class BufferType(Enum):
    """
    Enumeration for buffer type.
    """

    PARAM = 1
    GRAD = 2


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


class _ParamAndGradBucket:
    """
    Bucket to keep track of a subset of the model's parameters and gradients.

    Args:
        params: List of parameters whose gradients are collated in this bucket.
        param_data: View in _ParamAndGradBuffer.param_data that this bucket is responsible for.
        grad_data: View in _ParamAndGradBuffer.grad_data that this bucket is responsible for.
        offset: Offset of this bucket's view in the larger _ParamAndGradBuffer.
        numel_unpadded: Number of unpadded elements in bucket.
        gradient_scaling_factor: This factor is utilized to scale gradients prior to their
            communication. Its application is twofold: it facilitates the averaging of gradients
            and the scaling of gradients in the context of the Mixture of Experts (MoE) model.
        bucket_id: Index of bucket in buffer.
        param_index_map: Mapping from param to (start, end, bucket_id) in the global buffer.
            Used to derive bucket-local offsets for param_to_index.
        params_with_extra_main_grads: List of parameters in this bucket that require a
            separate higher-precision main_grad tensor for local gradient accumulation.
    """

    def __init__(
        self,
        params: List[torch.nn.Parameter],
        param_data: Optional[torch.Tensor],
        grad_data: torch.Tensor,
        offset: int,
        numel_unpadded: int,
        gradient_scaling_factor: float,
        bucket_id: int,
        param_index_map: Dict[torch.nn.Parameter, tuple],
        params_with_extra_main_grads: List[torch.nn.Parameter],
    ):
        self.params_list = params
        self.params = set(params)
        # Make sure there are no duplicate params.
        assert len(self.params_list) == len(self.params)
        self.param_data = param_data
        self.grad_data = grad_data
        # The distributed optimizer needs to keep track of this bucket's offset
        # within the full grad_buffer.
        self.offset = offset
        self.numel_unpadded = numel_unpadded
        self.gradient_scaling_factor = gradient_scaling_factor
        self.bucket_id = bucket_id
        # Derive bucket-local param offsets from the global param_index_map.
        self.param_to_index = {}
        for param in params:
            global_start, global_end, _ = param_index_map[param]
            self.param_to_index[param] = (global_start - offset, global_end - offset)
        self.params_with_extra_main_grads = params_with_extra_main_grads

        # Layer-wise optimizer attributes for async param gather.
        self.layerwise_params_list = None
        self.layerwise_param_flat_sizes = None
        self.layerwise_gather_list = None

    def set_layerwise_params_list(self, layerwise_params_list: List[List[torch.nn.Parameter]]):
        """Set per-rank parameter lists for layer-wise async all-gather.

        Args:
            layerwise_params_list: List of param lists, one per rank in the DP group.
                Each inner list contains the parameters owned by that rank's
                layer-wise optimizer that also belong to this bucket.
        """
        self.layerwise_params_list = layerwise_params_list
        self.layerwise_param_flat_sizes = [
            sum([p.numel() for p in param_list]) for param_list in layerwise_params_list
        ]


class _LayerwiseAllGatherHandle:
    """Handle wrapping multiple async all-gather work objects.

    NCCL guarantees in-order completion on the same communicator, so waiting
    on only the last handle is sufficient.
    """

    def __init__(self, handles):
        self.handles = handles

    def wait(self):
        """Wait on the last handle and clear all handles."""
        if self.handles:
            self.handles[-1].wait()
        self.handles = None


class _ParamAndGradBucketGroup:
    """
    Put multiple buckets into a group so that their communications can be aggregated together.
    Provides functionality to register when params in the bucket group have grads ready to be
    synced; an asynchronous communication call is automatically launched when _all_ params in
    the bucket group have grads ready.

    Args:
        buckets: A list of buckets.
        ddp_config: DistributedDataParallel config object.
        collective_group: intra_distributed_optimizer_instance_group if using distributed
            optimizer, data_parallel_group if not.
        collective_group_size: World size using the intra data-parallel group.
    """

    def __init__(
        self,
        buckets: List[_ParamAndGradBucket],
        ddp_config: DistributedDataParallelConfig,
        collective_group: torch.distributed.ProcessGroup,
        collective_group_size: int,
    ):
        self.buckets = buckets
        self.ddp_config = ddp_config

        # overlap_param_gather covers the layer-wise optimizer case, which sets
        # overlap_param_gather=True without use_distributed_optimizer.
        if self.ddp_config.use_distributed_optimizer or self.ddp_config.overlap_param_gather:
            self.intra_distributed_optimizer_instance_group = collective_group
            self.intra_distributed_optimizer_instance_size = collective_group_size
            self.intra_distributed_optimizer_instance_rank = collective_group.rank()
        if not self.ddp_config.use_distributed_optimizer:
            self.data_parallel_group = collective_group

        # State for bookkeeping: params is the set of parameters this bucket group is
        # responsible for, param_to_bucket maps params to the corresponding bucket.
        self.param_to_bucket = {}
        self.params = set()
        for bucket in self.buckets:
            for param in bucket.params_list:
                self.param_to_bucket[param] = bucket
                self.params.add(param)

        self.next_param_gather_bucket_group = None

        if self.ddp_config.num_distributed_optimizer_instances > 1:
            self.inter_distributed_optimizer_instance_group = None
            self.communication_stream = None
            assert (
                not self.ddp_config.reduce_scatter_with_fp32_accumulation
            ), "RS w/ FP32 accumulation not supported with num_distributed_optimizer_instances > 1"

        global dist_reduce_scatter_func
        if self.ddp_config.reduce_scatter_with_fp32_accumulation:
            dist_reduce_scatter_func = reduce_scatter_with_fp32_accumulation
            log_single_rank(
                logger,
                logging.INFO,
                "Using reduce_scatter_with_fp32_accumulation as reduce-scatter implementation",
            )

        # per_param_grad_ready_counts is a dict mapping parameters to number of times
        # `register_grad_ready` is called for that parameter *when
        # self.is_last_microbatch is True*. Should be 1 for most params but could be greater
        # than 1 if control flow passes through the same parameter multiple times. We lazily
        # populate this in the first batch, hence the .is_first_batch attribute.
        # When overlap_grad_reduce is True, communication (all-reduce or reduce-scatter)
        # is issued when per_param_grad_ready_counts equals golden_per_param_grad_ready_counts.
        # In other words, communication is dispatched as soon as all gradients in this bucket
        # are *ready*, as marked by the backward hook.
        # The set of keys in per_param_grad_ready_counts should be equal to `params`.
        self.golden_per_param_grad_ready_counts = {}
        self.per_param_grad_ready_counts = {}
        self.is_last_microbatch = True
        self.is_first_batch = True

        # Other metadata to keep track of collectives.
        self.param_gather_handle = None
        self.param_gather_dispatched = False
        self.grad_reduce_handle = None

        # Each time a local shard is created from bucket.param_data or bucket.grad_data, it
        # introduces some CPU overheads. We use these two lists to cache the created local
        # shards to avoid unnecessary CPU operations. This does not increase GPU memory usage
        # because it only saves a slice view, which shares the same memory with bucket.param_data
        # or bucket.grad_data.
        self.cached_param_buffer_shard_list = [None] * len(self.buckets)
        self.cached_grad_buffer_shard_list = [None] * len(self.buckets)

    def reset(self):
        """
        Reset metadata in bucket group in preparation for the next iteration of training.
        """
        if self.is_first_batch and len(self.per_param_grad_ready_counts) > 0:
            # Record golden per_param_grad_ready_counts.
            assert len(self.per_param_grad_ready_counts) == len(self.params)
            self.golden_per_param_grad_ready_counts = self.per_param_grad_ready_counts
            self.is_first_batch = False
        self.per_param_grad_ready_counts = {}
        self.is_last_microbatch = True

    def check_grads(self, check_for_nan_or_inf, check_for_large):
        """
        Make sure norm of grads in bucket are not NaN prior to data-parallel
        all-reduce / reduce-scatter.
        """
        rerun_state_machine = get_rerun_state_machine()
        for i in range(len(self.buckets)):
            grad_norm = self.buckets[i].grad_data.norm(p=2)
            # check for NaN, Inf and unexpectedly large grads
            if check_for_nan_or_inf:
                rerun_state_machine.validate_result(
                    result=grad_norm,
                    rejection_func=torch.isnan,
                    message=f"found NaN in local grad norm for bucket #{i} "
                    f"in backward pass before data-parallel communication collective",
                    tolerance=0.001,  # 0.1% tolerance to account for non-deterministic FA backward
                    fatal=True,
                )
                rerun_state_machine.validate_result(
                    result=grad_norm,
                    rejection_func=torch.isinf,
                    message=f"found Inf in local grad norm for bucket #{i} "
                    f"in backward pass before data-parallel communication collective",
                    tolerance=0.001,  # 0.1% tolerance to account for non-deterministic FA backward
                    fatal=True,
                )
            if check_for_large:
                rerun_state_machine.validate_result(
                    result=grad_norm,
                    rejection_func=partial(
                        rerun_state_machine.is_unexpectedly_large, threshold=10, context="grads"
                    ),
                    message=f"found unexpected large grads in bucket #{i} "
                    f"in backward pass before data-parallel communication collective",
                    tolerance=0.001,  # 0.1% tolerance to account for non-deterministic FA backward
                    fatal=False,
                )

    def start_param_sync(self, force_sync: bool = False):
        """
        Initiates all necessary param all-gathers for this bucket.

        When ddp_config.overlap_param_gather is set to True, dispatches an asynchronous
        communication call (unless force_sync is True). When ddp_config.overlap_param_gather
        is set to False, makes synchronous call.

        Args:
            force_sync (bool, optional): force synchronous collective regardless of
                other settings if true.
        """
        # overlap_param_gather covers the layer-wise optimizer case, which sets
        # overlap_param_gather=True without use_distributed_optimizer.
        assert self.ddp_config.use_distributed_optimizer or self.ddp_config.overlap_param_gather

        if force_sync:
            if self.param_gather_handle is not None:
                self.param_gather_handle.wait()
                self.param_gather_handle = None
                return
        else:
            assert self.param_gather_handle is None

        async_op = self.ddp_config.overlap_param_gather and not force_sync

        if not self.ddp_config.use_distributed_optimizer:
            # Layer-wise optimizer path: use all_gather for variable-size
            # param gather.
            #
            # Each rank may own a different number of params per bucket, so
            # layerwise_param_flat_sizes can vary across ranks.  PyTorch's NCCL
            # backend handles uneven tensor sizes in torch.distributed.all_gather
            # (falling back to grouped send/recv internally when sizes differ),
            # so no manual padding is needed.
            dp_size = self.intra_distributed_optimizer_instance_size
            if dp_size == 1:
                # Single-rank group (e.g., expt_dp_size == 1): no all-gather needed.
                self.param_gather_dispatched = True
                return
            local_rank = self.intra_distributed_optimizer_instance_rank
            group = self.intra_distributed_optimizer_instance_group
            layerwise_work_handles = []
            for bucket in self.buckets:
                # Use param dtype (e.g., bf16), NOT grad dtype (which may be
                # fp32 when grad_reduce_in_fp32 is enabled).
                param_dtype = bucket.params_list[0].dtype

                if max(bucket.layerwise_param_flat_sizes) == 0:
                    bucket.layerwise_gather_list = None
                    continue

                local_size = bucket.layerwise_param_flat_sizes[local_rank]
                total_gather_size = sum(bucket.layerwise_param_flat_sizes)

                # Reuse grad_data as the all_gather receive buffer; it is idle
                # during forward and grad_dtype.element_size >= param_dtype.
                reuse_buf = bucket.grad_data.view(param_dtype)
                assert reuse_buf.numel() >= total_gather_size

                # Partition reuse_buf into contiguous per-rank receive slices.
                gather_list = []
                offset = 0
                for i in range(dp_size):
                    size = bucket.layerwise_param_flat_sizes[i]
                    gather_list.append(reuse_buf[offset : offset + size])
                    offset += size
                local_slot_view = gather_list[local_rank]

                # Flatten local params and copy into the local rank's slot.
                # Detach from autograd since start_param_sync may be called
                # during the forward pass where autograd is active.
                if local_size > 0:
                    flat_local_params = _flatten_dense_tensors(
                        bucket.layerwise_params_list[local_rank]
                    ).detach()
                    local_slot_view.copy_(flat_local_params)
                bucket.layerwise_gather_list = gather_list

                work = torch.distributed.all_gather(
                    gather_list, local_slot_view, group=group, async_op=async_op
                )
                if async_op and work is not None:
                    layerwise_work_handles.append(work)

            if async_op:
                self.param_gather_handle = _LayerwiseAllGatherHandle(layerwise_work_handles)
            else:
                # Synchronous: unflatten and copy gathered params immediately.
                for bucket in self.buckets:
                    if bucket.layerwise_gather_list is None:
                        continue
                    for idx, params in enumerate(bucket.layerwise_params_list):
                        if len(params) == 0 or idx == local_rank:
                            continue
                        updated_params = _unflatten_dense_tensors(
                            bucket.layerwise_gather_list[idx], params
                        )
                        for updated_p, model_p in zip(updated_params, params):
                            model_p.data.copy_(updated_p)
                    bucket.layerwise_gather_list = None
                self.param_gather_handle = None
        else:
            # Standard distributed optimizer path: use _coalescing_manager.
            # all_gather_into_tensor writes directly into a contiguous output buffer and
            # does not need a copy-back step, so coalescing works correctly.
            with _coalescing_manager(
                self.intra_distributed_optimizer_instance_group, async_ops=async_op
            ) as cm:
                for idx, bucket in enumerate(self.buckets):
                    if self.cached_param_buffer_shard_list[idx] is None:
                        self.cached_param_buffer_shard_list[idx] = shard_buffer(
                            bucket.param_data, self.intra_distributed_optimizer_instance_size
                        )
                    local_data_view = self.cached_param_buffer_shard_list[idx][
                        self.intra_distributed_optimizer_instance_rank
                    ]
                    dist_all_gather_func(
                        bucket.param_data,
                        local_data_view,
                        group=self.intra_distributed_optimizer_instance_group,
                        async_op=async_op,
                    )
            if async_op:
                self.param_gather_handle = cm
            else:
                # When using `_coalescing_manager`, even if a synchronous op
                # (async_op=False) is used, `cm` is not None. Manually set to None for
                # consistency with prior code.
                self.param_gather_handle = None
        self.param_gather_dispatched = True

    def finish_param_sync(self, skip_next_bucket_dispatch: bool = False):
        """
        Finishes param sync communication operation for this bucket. Dispatches
        next bucket's param sync if available, unless skip_next_bucket_dispatch
        is True.

        When ddp_config.overlap_param_gather is set to True, waits for asynchronous
        communication call to complete (and dispatches one if one is not already
        outstanding). Throws assertion error if ddp_config.overlap_param_gather is set to
        False.

        Args:
            skip_next_bucket_dispatch (bool, optional): if true, dispatch next
                bucket's communication if available.
        """
        assert self.ddp_config.overlap_param_gather

        # If current bucket's param AG has not been dispatched, dispatch it now (e.g., first
        # AG bucket in first model chunk if ddp_config.align_param_gather is False).
        if not self.param_gather_dispatched:
            self.start_param_sync()

        if self.param_gather_handle is not None:
            self.param_gather_handle.wait()
            self.param_gather_handle = None
            # Dispatch next bucket's asynchronous param AG only if it has not been dispatched yet.
            if self.next_param_gather_bucket_group is not None and not skip_next_bucket_dispatch:
                if self.next_param_gather_bucket_group.param_gather_dispatched:
                    warnings.warn(
                        "The next bucket's parameter all-gather operation has already been "
                        "dispatched. This may be caused by a mismatch between the order of "
                        "parameter registration and forward pass execution, which will "
                        "hurt the communication-computation overlap performance."
                    )
                else:
                    self.next_param_gather_bucket_group.start_param_sync()

            # For the mxfp8_param with "reuse_grad_buf_for_mxfp8_param_ag=True",
            # we need to copy the param_data from the shared_param/grad_buffer to param.data
            # after the param all-gather.
            if self.ddp_config.reuse_grad_buf_for_mxfp8_param_ag:
                for bucket in self.buckets:
                    is_bf16_weight_bucket = False
                    for param in bucket.params:
                        # Skip copying since bf16 weights in the mxfp8 model
                        # are already mapped to param.data.
                        if not is_float8tensor(param):
                            is_bf16_weight_bucket = True
                            break
                        param_start, param_end = bucket.param_to_index[param]
                        param_slice = bucket.param_data.view(-1)[param_start:param_end]
                        param.data.copy_(param_slice.view(param.data.shape))
                    if is_bf16_weight_bucket:
                        continue
                    # All-gathered params are not needed after being copied to param.data.
                    # Zero out the param buffer (shared with grad buffer) for gradient accumulation.
                    # We cannot zero out the entire grad buffer because one grad buffer may
                    # correspond to multiple param buffers. If we zero out the entire grad buffer,
                    # it would clear the data of those param buffers that have not yet completed AG.
                    bucket.param_data.zero_()
            elif not self.ddp_config.use_distributed_optimizer:
                for bucket in self.buckets:
                    if bucket.layerwise_gather_list is None:
                        continue
                    # Unflatten and copy gathered params for each rank.
                    for idx, params in enumerate(bucket.layerwise_params_list):
                        # Skip local params and empty tensors.
                        if (
                            len(params) == 0
                            or idx == self.intra_distributed_optimizer_instance_rank
                        ):
                            continue
                        updated_params = _unflatten_dense_tensors(
                            bucket.layerwise_gather_list[idx], params
                        )
                        for updated_p, model_p in zip(updated_params, params):
                            model_p.data.copy_(updated_p)
                    bucket.layerwise_gather_list = None
            else:
                fp8_params = []
                for bucket in self.buckets:
                    for param in bucket.params:
                        if is_float8tensor(param):
                            fp8_params.append(param)
                if len(fp8_params) > 0:
                    post_all_gather_processing(fp8_params)

    def start_grad_sync(self, force_all_reduce: Optional[bool] = False):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operations
        for all buckets in the bucket group.

        When ddp_config.overlap_grad_reduce is set to True, dispatches an asynchronous
        communication call. When ddp_config.overlap_grad_reduce is set to False, makes
        synchronous call.
        """
        if self.is_first_batch and self.grad_reduce_handle is not None:
            # Make this start_grad_sync call a no-op if in first batch and collective has
            # already been dispatched.
            return

        assert (
            self.grad_reduce_handle is None
        ), "Should not have multiple communication calls outstanding at once"

        # Copy accumulated .main_grad into communication buffer before collective if
        # .main_grad is not in .grad_data already (e.g., because we want to do local
        # gradient accumulation in a higher precision).
        for bucket in self.buckets:
            for param in bucket.params_with_extra_main_grads:
                if getattr(param, 'main_grad_copy_in_grad_buffer', None) is not None:
                    param.main_grad_copy_in_grad_buffer.copy_(param.main_grad)

        if self.ddp_config.check_for_nan_in_grad or self.ddp_config.check_for_large_grads:
            self.check_grads(
                check_for_nan_or_inf=self.ddp_config.check_for_nan_in_grad,
                check_for_large=self.ddp_config.check_for_large_grads,
            )

        # gradient_scaling_factor already takes into account whether we are computing
        # an average or sum in the data-parallel collective.
        for bucket in self.buckets:
            if bucket.gradient_scaling_factor != 1.0:
                bucket.grad_data *= bucket.gradient_scaling_factor

        # Decide reduce_op.
        reduce_op = torch.distributed.ReduceOp.SUM
        if self.ddp_config.average_in_collective:
            reduce_op = torch.distributed.ReduceOp.AVG

        # We use the following stream synchronization for the gradient reduction
        # within and across DistOpt instances.

        # Compute Stream: -------------Gradient compute-------------------
        # Comm. Stream:   ------(wait for NCCL)-----(wait for NCCL)-------
        # NCCL Stream:          -------RS------     -------AR------

        # Use async communications only when overlap_grad_reduce is True.
        async_op = (
            self.ddp_config.overlap_grad_reduce
            and self.ddp_config.num_distributed_optimizer_instances == 1
        )
        if (
            self.ddp_config.num_distributed_optimizer_instances > 1
            and self.ddp_config.overlap_grad_reduce
        ):
            # Assign a communication stream if we have multiple DistOpt instances and we
            # need to overlap communication.
            stream_context = torch.cuda.stream(self.communication_stream)

            # The RS/AR communication stream needs to wait for the current stream
            # to complete its gradient computation before launching the next
            # gradient reduction collective.
            self.communication_stream.wait_stream(torch.cuda.current_stream())
        else:
            stream_context = nullcontext()

        if self.ddp_config.use_distributed_optimizer:
            communication_group = self.intra_distributed_optimizer_instance_group
        else:
            communication_group = self.data_parallel_group

        # Coalesce communication kernels across buckets in the bucket group.
        grad_reduce_handle = None
        with stream_context, _coalescing_manager(communication_group, async_ops=async_op) as cm:
            for idx, bucket in enumerate(self.buckets):
                if self.ddp_config.use_distributed_optimizer and not force_all_reduce:
                    if self.cached_grad_buffer_shard_list[idx] is None:
                        self.cached_grad_buffer_shard_list[idx] = shard_buffer(
                            bucket.grad_data, self.intra_distributed_optimizer_instance_size
                        )
                    local_data_view = self.cached_grad_buffer_shard_list[idx][
                        self.intra_distributed_optimizer_instance_rank
                    ]
                    grad_reduce_handle = dist_reduce_scatter_func(
                        local_data_view,
                        bucket.grad_data,
                        op=reduce_op,
                        group=communication_group,
                        async_op=async_op,
                    )
                else:
                    if torch.distributed.get_rank() == 0 and force_all_reduce:
                        logger.info(
                            f"Performing reduction using all_reduce because {force_all_reduce=}"
                        )
                    torch.distributed.all_reduce(
                        bucket.grad_data, op=reduce_op, group=communication_group, async_op=async_op
                    )

        # With multiple DistOpt instances, we need to all-reduce across instances.
        if (
            self.ddp_config.use_distributed_optimizer
            and self.ddp_config.num_distributed_optimizer_instances > 1
        ):
            assert self.inter_distributed_optimizer_instance_group is not None
            # Create a new coalescing manager for the inter-instance all-reduce.
            with (
                stream_context,
                _coalescing_manager(
                    self.inter_distributed_optimizer_instance_group, async_ops=async_op
                ) as cm,
            ):
                for idx, bucket in enumerate(self.buckets):
                    if self.cached_grad_buffer_shard_list[idx] is None:
                        self.cached_grad_buffer_shard_list[idx] = shard_buffer(
                            bucket.grad_data, self.intra_distributed_optimizer_instance_size
                        )
                    local_data_view = self.cached_grad_buffer_shard_list[idx][
                        self.intra_distributed_optimizer_instance_rank
                    ]

                    torch.distributed.all_reduce(
                        local_data_view,
                        op=reduce_op,
                        group=self.inter_distributed_optimizer_instance_group,
                        async_op=async_op,
                    )

        if async_op:
            if self.ddp_config.reduce_scatter_with_fp32_accumulation and not force_all_reduce:
                assert (
                    len(self.buckets) == 1
                ), "Only 1 bucket supported with reduce_scatter_with_fp32_accumulation=True"
                # torch.distributed._coalescing_manager does not correctly handle calling our custom
                # collective handle's .wait() method, so we take matters into our own hands here.
                assert grad_reduce_handle is not None
                self.grad_reduce_handle = grad_reduce_handle
            else:
                self.grad_reduce_handle = cm
        else:
            # When using `_coalescing_manager`, even if a synchronous op (async_op=False) is used,
            # `cm` is not None, which is different from when `_coalescing_manager` is not used in
            # which case the torch.distributed._reduce_scatter_base() will return None. In order to
            # maintain consistency with prior code, we need to manually set communication handle to
            # None.
            self.grad_reduce_handle = None

    def finish_grad_sync(self, force_all_reduce: Optional[bool] = False):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operations
        for all buckets in the bucket group.

        When ddp_config.overlap_grad_reduce is set to True, waits for asynchronous
        communication call to complete. When ddp_config.overlap_grad_reduce is set to False,
        makes synchronous call.
        """
        self.param_gather_dispatched = False
        # If overlap_grad_reduce is False, start (and finish) synchronous communication call here.
        if not self.ddp_config.overlap_grad_reduce:
            self.start_grad_sync(force_all_reduce=force_all_reduce)
            self._copy_back_extra_main_grads()
            return
        # If first batch, start asynchronous communication here. register_grad_ready() launches
        # asynchronous communication only once self.golden_per_param_grad_ready_counts is
        # populated at the end of this first batch.
        if self.is_first_batch:
            self.start_grad_sync(force_all_reduce=force_all_reduce)
        # When using multiple DistOpt instances, we don't need to sync here as we launch
        # communications on a separate communication stream.
        if self.ddp_config.num_distributed_optimizer_instances > 1:
            torch.cuda.current_stream().wait_stream(self.communication_stream)
            self._copy_back_extra_main_grads()
            return
        assert self.grad_reduce_handle is not None, (
            f"Communication call has not been issued for this bucket "
            f"({len(self.per_param_grad_ready_counts)}/{len(self.params)} "
            "params have grad available)"
        )
        self.grad_reduce_handle.wait()
        self.grad_reduce_handle = None
        self._copy_back_extra_main_grads()

    def free_overlap_buffers(self):
        """Free GPU buffers used by overlap param gather.

        Waits on any pending param all-gather handle, then releases the
        per-bucket temporary buffers so that the CUDA memory allocator can
        reclaim them.  Called before async checkpoint saves to avoid OOM in
        the persistent checkpoint worker process.
        """
        if self.param_gather_handle is not None:
            self.param_gather_handle.wait()
            self.param_gather_handle = None
        for bucket in self.buckets:
            bucket.layerwise_gather_list = None

    def _copy_back_extra_main_grads(self):
        """
        Copy reduced gradients from the communication buffer back to .main_grad for
        params that have a separate higher-precision .main_grad tensor.

        This is needed because the optimizer reads from .main_grad to get the reduced
        gradients, but for params with extra main_grads, .main_grad points to the local
        FP32 accumulation tensor rather than the communication buffer where the reduced
        gradients are stored.
        """
        for bucket in self.buckets:
            for param in bucket.params_with_extra_main_grads:
                if getattr(param, 'main_grad_copy_in_grad_buffer', None) is not None:
                    param.main_grad.copy_(param.main_grad_copy_in_grad_buffer)

    def register_grad_ready(
        self, param: torch.nn.Parameter, force_all_reduce: Optional[bool] = False
    ):
        """
        Registers grads for the passed-in param to be "ready" for grad sync.

        When the number of microbatches is greater than 1, we only want to register
        grads as ready when processing the last microbatch and ddp_config.overlap_grad_reduce
        is True.
        """
        assert (
            self.ddp_config.overlap_grad_reduce
        ), "register_grad_ready() should only be called when overlap_grad_reduce is True"
        if self.is_last_microbatch:
            assert param in self.param_to_bucket, "Param is not in the bucket group"
            if param not in self.per_param_grad_ready_counts:
                self.per_param_grad_ready_counts[param] = 0
            self.per_param_grad_ready_counts[param] += 1
            # If all params in bucket group have grads available, issue communication call.
            if not self.is_first_batch:
                if self.per_param_grad_ready_counts == self.golden_per_param_grad_ready_counts:
                    assert len(self.per_param_grad_ready_counts) == len(self.params)
                    self.start_grad_sync(force_all_reduce=force_all_reduce)


def group_params_for_buffers(
    params: List[torch.nn.Parameter], grad_reduce_in_fp32: bool
) -> Dict['BufferKey', Tuple[List[torch.nn.Parameter], List[int]]]:
    """Group parameters by buffer identity for buffer allocation.

    Each distinct buffer is identified by a BufferKey with four dimensions:
    - param_dtype: storage dtype (torch.uint8 for FP8/NVFP4 parameters, else param.dtype).
    - grad_dtype: gradient reduction dtype (torch.float if grad_reduce_in_fp32, else param.dtype).
    - is_expert_parallel: whether the parameter is expert-parallel (param.allreduce == False),
      which requires a separate buffer with a different data-parallel group.
    - use_layerwise_distributed_optimizer: whether the parameter uses the layer-wise distributed
      optimizer, which requires shard-aligned layouts.

    The param_indices track each parameter's position among same-dtype params (using
    the "fake" high-precision dtype for FP8/NVFP4 params), needed for loading non-native-fp8
    checkpoints in native-fp8 mode.

    Args:
        params: List of parameters to group.
        grad_reduce_in_fp32: Whether gradients are reduced in FP32.

    Returns:
        Dict mapping BufferKey to (params_list, param_indices).
    """
    from ..optimizer.param_layout import BufferKey

    key_to_params = {}
    dtype_to_offsets = {}
    key_to_indices = {}

    for param in params:
        assert param.requires_grad

        param_dtype = param.dtype
        if is_float8tensor(param) or is_nvfp4tensor(param):
            param_dtype = torch.uint8
        grad_dtype = torch.float if grad_reduce_in_fp32 else param.dtype
        is_expert_parallel = not getattr(param, 'allreduce', True)
        use_layerwise_distributed_optimizer = getattr(
            param, 'use_layerwise_distributed_optimizer', False
        )

        key = BufferKey(
            param_dtype, grad_dtype, is_expert_parallel, use_layerwise_distributed_optimizer
        )
        param_list = key_to_params.get(key, [])
        param_list.append(param)
        key_to_params[key] = param_list

        # Use param.dtype (not param_dtype) so FP8/NVFP4 params share offsets with their
        # logical high-precision dtype, needed for checkpoint compatibility.
        offset_key = BufferKey(
            param.dtype, grad_dtype, is_expert_parallel, use_layerwise_distributed_optimizer
        )
        offset = dtype_to_offsets.get(offset_key, 0)
        dtype_to_offsets[offset_key] = offset + 1
        indices = key_to_indices.get(key, [])
        indices.append(offset)
        key_to_indices[key] = indices

    result = {}
    for key, param_list in key_to_params.items():
        result[key] = (param_list, key_to_indices[key])
    return result


def _compute_default_per_buffer_param_layout(
    params: List[torch.nn.Parameter], bucket_size: Optional[int]
) -> 'PerBufferParamLayout':
    """Compute parameter layout for the non-distributed-optimizer case.

    No padding is applied. Parameters are iterated in reverse order (backprop order)
    and grouped into buckets of approximately `bucket_size` elements.

    Args:
        params: List of parameters to lay out.
        bucket_size: Approximate number of elements per bucket, or None for a single bucket.

    Returns:
        PerBufferParamLayout with the computed mapping.
    """
    from ..optimizer.param_layout import PerBufferParamLayout

    param_index_map = {}
    bucket_indices = []
    per_bucket_numel_unpadded = []

    param_start_index = 0
    bucket_start_index = 0
    bucket_params = set()
    bucket_id = 0

    for param in params[::-1]:
        this_numel = param.data.nelement()
        param_end_index = param_start_index + this_numel
        param_index_map[param] = (param_start_index, param_end_index, bucket_id)
        bucket_params.add(param)

        if bucket_size is not None and (param_end_index - bucket_start_index) >= bucket_size:
            per_bucket_numel_unpadded.append(param_end_index - bucket_start_index)
            bucket_indices.append((bucket_start_index, param_end_index))
            bucket_start_index = param_end_index
            bucket_params = set()
            bucket_id += 1
        param_start_index = param_end_index

    if len(bucket_params) > 0:
        per_bucket_numel_unpadded.append(param_end_index - bucket_start_index)
        bucket_indices.append((bucket_start_index, param_end_index))

    return PerBufferParamLayout(
        param_index_map=param_index_map,
        bucket_indices=bucket_indices,
        per_bucket_numel_unpadded=per_bucket_numel_unpadded,
    )


class _ParamAndGradBuffer:
    """
    Groups parameters and gradients into a contiguous buffer, and then breaks the buffer into
    buckets with roughly `bucket_size` parameters each.

    Args:
        ddp_config: DistributedDataParallel config object.
        param_dtype: Type of param tensor.
        grad_dtype: Type of grad tensor.
        params: List of parameters whose parameters and gradients are collated in the underlying
            tensor.
        data_parallel_group: Data-parallel process group.
        bucket_size: The rough size of each bucket in terms of number of parameters.
        param_to_name: Mapping from `torch.nn.Parameter` to name (for logging purposes).
        gradient_scaling_factor: This factor is utilized to scale gradients prior to their
            communication. Its application is twofold: it facilitates the averaging of gradients
            and the scaling of gradients in the context of the Mixture of Experts (MoE) model.
        param_indices: The index of each param among the params with same dtype, if a param is fp8,
            use its "fake" high precision dtype to determine which params have same dtype with it.
            These indices are needed when loading a non-native-fp8 checkpoint in native-fp8 mode.
    """

    def __init__(
        self,
        ddp_config: DistributedDataParallelConfig,
        param_dtype: torch.dtype,
        grad_dtype: torch.dtype,
        params_with_names: List[Tuple[torch.nn.Parameter, str]],
        data_parallel_group: torch.distributed.ProcessGroup,
        bucket_size: int,
        param_to_name: Dict[torch.nn.Parameter, str],
        gradient_scaling_factor: float,
        param_indices: List[int],
        nccl_ub: bool,
        pg_collection: Optional[ProcessGroupCollection] = None,
        param_layout: Optional['PerBufferParamLayout'] = None,
    ):

        if pg_collection is None:
            self.dp_cp_group = parallel_state.get_data_and_context_parallel_group(
                with_context_parallel=True
            )
            self.tp_group = parallel_state.get_tensor_model_parallel_group()
        else:
            assert hasattr(pg_collection, 'tp') and hasattr(pg_collection, 'dp_cp')
            self.dp_cp_group = pg_collection.dp_cp
            self.tp_group = pg_collection.tp

        self.ddp_config = ddp_config
        self.params = [param for (param, _) in params_with_names]
        self.param_indices = param_indices

        # Check that params are unique.
        unique_params = set()
        for param, _ in params_with_names:
            assert param not in unique_params
            unique_params.add(param)
        del unique_params

        # Store attributes that will be needed later.
        self.param_dtype = param_dtype
        self.grad_dtype = grad_dtype
        self.data_parallel_group = data_parallel_group
        self.data_parallel_world_size = self.data_parallel_group.size()
        self.gradient_scaling_factor = gradient_scaling_factor
        self.nccl_ub = nccl_ub

        # Data structures to store underlying buckets and relevant indexing data.
        self.buckets = []
        self.param_to_bucket = {}  # Param -> bucket mapping.

        # Use the provided layout if given, otherwise compute the default (no-padding) layout.
        if param_layout is None:
            param_layout = _compute_default_per_buffer_param_layout(self.params, bucket_size)
        self.param_index_map = param_layout.param_index_map
        self.bucket_indices = param_layout.bucket_indices
        per_bucket_numel_unpadded = param_layout.per_bucket_numel_unpadded

        # Check if this buffer contains NVFP4 params.
        #
        # NVFP4 uses a dual-buffer layout: the param buffer stores packed bytes (half the
        # logical numel) while the grad buffer uses the full numel. This is because NVFP4
        # packs two FP4 values into a single uint8 byte for storage/communication, but
        # gradients are computed and reduced in BF16 at full element count.
        #
        #   Logical view:  [v0, v1, v2, v3, ...]   numel = N
        #
        #   Param buffer  (uint8):      [byte0, byte1, ...]      numel = N // 2
        #                                 ^^^^^ packs v0+v1
        #
        #   Grad buffer:  [g0, g1, g2, g3, ...]   numel = N
        #
        # We therefore maintain two index maps:
        #   - param_index_map:              offsets using full numel (from pre-computed layout).
        #   - nvfp4_packed_param_index_map: offsets into the packed param buffer (numel // 2).
        #
        # The packed index map is derived from param_index_map by iterating through
        # the already-computed layout and halving numel for NVFP4 tensors.
        #
        self.has_nvfp4_params = any(is_nvfp4tensor(p) for p in self.params)
        self.nvfp4_packed_param_index_map = None
        self.nvfp4_packed_bucket_indices = None
        if self.has_nvfp4_params:
            self._compute_nvfp4_packed_layout(params_with_names)

        # Next, create underlying storage for buffer (with numel elements that includes
        # padding as necessary).
        self.numel = self.bucket_indices[-1][1]
        self.numel_unpadded = sum(per_bucket_numel_unpadded)
        if self.has_nvfp4_params:
            self.nvfp4_packed_numel = self.nvfp4_packed_bucket_indices[-1][1]
            # nvfp4_packed_numel_unpadded is already set by _compute_nvfp4_packed_layout.

        assert self.numel_unpadded <= self.numel
        if self.has_nvfp4_params:
            assert self.nvfp4_packed_numel_unpadded <= self.nvfp4_packed_numel
        if self.ddp_config.use_distributed_optimizer:
            assert self.numel % self.data_parallel_world_size == 0
            if self.has_nvfp4_params:
                assert self.nvfp4_packed_numel % self.data_parallel_world_size == 0
        else:
            assert self.numel == self.numel_unpadded

        self.param_data = None
        self.grad_data = None
        self.extra_main_grads = []

        if self.nccl_ub:
            # If nccl_ub is True, use nccl_allocator to allocate memory for param_data/grad_data.
            nccl_allocator.init()
            pool = nccl_allocator.create_nccl_mem_pool(
                symmetric=not self.ddp_config.disable_symmetric_registration
            )
            mem_alloc_context = functools.partial(
                nccl_allocator.nccl_mem,
                pool,
                group=self.data_parallel_group,
                symmetric=not self.ddp_config.disable_symmetric_registration,
            )
            # Since nccl communicator group is created lazily, we need to perform a warmup call to
            # initialize NCCL comm buffers for this dp_group before doing buffer registration.
            torch.distributed.barrier()
            tmp_warmup_tensor = torch.zeros([1], device="cuda")
            torch.distributed.all_reduce(tmp_warmup_tensor, group=self.data_parallel_group)
            torch.distributed.barrier()
        else:
            # If nccl_ub is False, mem_alloc_context is nullcontext.
            mem_alloc_context = nullcontext

        with mem_alloc_context():
            # For MXFP8 param: Create a shared buffer for param AG and grad RS for memory efficiency
            # The buffer is mapped to weight gradients whose dtype is either bf16 or FP32.
            # It can be temporarily reused by param AG.
            if self.ddp_config.use_distributed_optimizer and any(
                is_mxfp8tensor(p) for p in self.params
            ):
                self.shared_buffer = torch.zeros(
                    self.numel,
                    dtype=self.grad_dtype,
                    device=torch.cuda.current_device(),
                    requires_grad=False,
                )
                # For FP32 weight grads, only half of the buffer is used to store params in bf16.
                if self.grad_dtype == torch.float32:
                    self.param_data = self.shared_buffer[: math.ceil(self.numel / 2)].view(
                        torch.bfloat16
                    )
                else:
                    self.param_data = self.shared_buffer
                self.grad_data = self.shared_buffer
            else:
                # Only re-map param tensors if using distributed optimizer.
                if self.ddp_config.use_distributed_optimizer:
                    numel = self.nvfp4_packed_numel if self.has_nvfp4_params else self.numel
                    self.param_data = torch.zeros(
                        numel,
                        dtype=self.param_dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                self.grad_data = torch.zeros(
                    self.numel,
                    dtype=self.grad_dtype,
                    device=torch.cuda.current_device(),
                    requires_grad=False,
                )

        self.grad_data_size = 0
        self.param_data_size = 0
        self.param_data_cpu = None

        # Finally, map param.data and param.main_grad fields to buffers.
        def _create_bucket(bucket_id, bucket_params, bucket_params_with_extra_main_grads):
            """
            Look up precomputed bucket indices and create a new bucket.

            Args:
                bucket_id: ID of the bucket to create.
                bucket_params: List of parameters in this bucket.
                bucket_params_with_extra_main_grads: List of parameters with
                    extra FP32 main_grads.

            Returns:
                A new _ParamAndGradBucket instance.
            """
            bucket_start_index, bucket_end_index = self.bucket_indices[bucket_id]
            if self.has_nvfp4_params:
                nvfp4_packed_start_index, nvfp4_packed_end_index = self.nvfp4_packed_bucket_indices[
                    bucket_id
                ]
            else:
                nvfp4_packed_start_index, nvfp4_packed_end_index = None, None
            return self._new_bucket(
                bucket_params=bucket_params,
                start_index=bucket_start_index,
                end_index=bucket_end_index,
                numel_unpadded=per_bucket_numel_unpadded[bucket_id],
                bucket_id=bucket_id,
                nvfp4_packed_start_index=nvfp4_packed_start_index,
                nvfp4_packed_end_index=nvfp4_packed_end_index,
                bucket_params_with_extra_main_grads=bucket_params_with_extra_main_grads,
            )

        bucket_params = []
        bucket_params_with_extra_main_grads = []
        cur_bucket_id = 0
        for param, param_name in params_with_names[::-1]:
            # Get parameter indices computed in previous loop.
            param_start_index, param_end_index, bucket_id = self.param_index_map[param]
            nvfp4_packed_param_start_index = None
            if self.has_nvfp4_params:
                nvfp4_packed_param_start_index, _, _ = self.nvfp4_packed_param_index_map[param]
            # For MXFP8 param:
            # we only need to map bf16 weights (layernorm, embedding, etc) to the buffer.
            if not self.ddp_config.reuse_grad_buf_for_mxfp8_param_ag or not is_mxfp8tensor(param):
                if self.param_data is not None:
                    if is_nvfp4tensor(param):
                        # Remap the NVFP4 tensor's internal rowwise uint8 storage so it
                        # points into the contiguous DDP param buffer. This enables the
                        # all-gather to communicate packed NVFP4 bytes directly.
                        from ..fp4_utils import modify_nvfp4_rowwise_storage

                        packed_shape = get_nvfp4_rowwise_packed_shape(param.data.shape)
                        rowwise_bytes_view = self._get(
                            packed_shape,
                            nvfp4_packed_param_start_index,
                            buffer_type=BufferType.PARAM,
                        )
                        modify_nvfp4_rowwise_storage(param, rowwise_bytes_view)
                    elif is_float8tensor(param):
                        new_param_data = self._get(
                            param.data.shape,
                            (
                                nvfp4_packed_param_start_index
                                if self.has_nvfp4_params
                                else param_start_index
                            ),
                            buffer_type=BufferType.PARAM,
                        )
                        modify_underlying_storage(param, new_param_data)
                    else:
                        new_param_data = self._get(
                            param.data.shape,
                            (
                                nvfp4_packed_param_start_index
                                if self.has_nvfp4_params
                                else param_start_index
                            ),
                            buffer_type=BufferType.PARAM,
                        )
                        old_param_data = param.data
                        param.data = new_param_data
                        assert old_param_data._base is None
                        # Copy tensor values (from initialization or checkpoint).
                        param.data.detach().copy_(old_param_data)
                        del old_param_data

            # Grad buffer always uses full-numel offsets from param_index_map.
            param.main_grad = self._get(
                param.data.shape, param_start_index, buffer_type=BufferType.GRAD
            )
            # Create FP32 copy of .main_grads if necessary.
            promote_main_grads_to_higher_precision = False
            for param_name_pattern in ddp_config.param_name_patterns_for_fp32_local_accumulation:
                if fnmatch.fnmatch(param_name, param_name_pattern) or param_name_pattern == 'all':
                    log_on_each_pipeline_stage(
                        logger,
                        logging.INFO,
                        (
                            f"Matched {param_name} with '{param_name_pattern}'; promoting "
                            f"main_grad.type from {param.main_grad.dtype} to torch.float32!"
                        ),
                        tp_group=self.tp_group,
                        dp_cp_group=self.dp_cp_group,
                    )
                    promote_main_grads_to_higher_precision = True
                    break
            if promote_main_grads_to_higher_precision:
                param.main_grad_copy_in_grad_buffer = (
                    param.main_grad
                )  # Slice into .grad_data buffer.
                param.main_grad = torch.empty_like(param.main_grad, dtype=torch.float32)
                self.extra_main_grads.append(param.main_grad)

            if bucket_id != cur_bucket_id:
                self.buckets.append(
                    _create_bucket(
                        cur_bucket_id, bucket_params, bucket_params_with_extra_main_grads
                    )
                )
                bucket_params = []
                bucket_params_with_extra_main_grads = []
                assert cur_bucket_id + 1 == len(self.buckets)
                assert bucket_id == cur_bucket_id + 1
                cur_bucket_id = bucket_id

            bucket_params.append(param)
            if promote_main_grads_to_higher_precision:
                bucket_params_with_extra_main_grads.append(param)

        # Add remaining params to a new bucket.
        if len(bucket_params) > 0:
            self.buckets.append(
                _create_bucket(cur_bucket_id, bucket_params, bucket_params_with_extra_main_grads)
            )
        # Log buckets for all PP stages.
        log_strs = []
        log_strs.append(
            f"Number of buckets for gradient all-reduce / reduce-scatter: {len(self.buckets)}"
        )
        for index, bucket in enumerate(self.buckets):
            numel = 0
            for param in bucket.params_list:
                numel += param.data.nelement()
            log_strs.append(
                f"Params for bucket {index + 1} ({numel} elements, "
                f"{bucket.grad_data.nelement()} padded size, "
                f"{len(bucket.params_with_extra_main_grads)} param(s) with extra main_grads):"
            )
            for param in bucket.params_list:
                log_strs.append(f"\t{param_to_name[param]} ({param.main_grad.dtype=})")
        log_on_each_pipeline_stage(
            logger,
            logging.INFO,
            "\n".join(log_strs),
            tp_group=self.tp_group,
            dp_cp_group=self.dp_cp_group,
        )

    def _compute_nvfp4_packed_layout(self, params_with_names):
        """Derive packed NVFP4 index map and bucket indices from the primary layout.

        The primary layout (self.param_index_map, self.bucket_indices) uses full numel
        for all params. NVFP4 tensors pack two FP4 values into one byte, so the param
        buffer needs a separate "packed" index map where NVFP4 params occupy half the
        space. Non-NVFP4 params keep their full numel in the packed space.

        The same padding rules used by the primary layout are applied here:
        - 64-element alignment at the start of each param.
        - Bucket-end padding for DP-divisibility (when using distributed optimizer).

        Sets:
            self.nvfp4_packed_param_index_map: param -> (start, end, bucket_id)
            self.nvfp4_packed_bucket_indices: list of (start, end) per bucket
            self.nvfp4_packed_numel_unpadded: total unpadded elements across all buckets
        """

        def _pad_start_of_param(param_start_index: int) -> int:
            if self.ddp_config.use_distributed_optimizer:
                return pad_param_start(param_start_index)
            return param_start_index

        def _pad_end_of_bucket(bucket_end_index: int) -> int:
            if self.ddp_config.use_distributed_optimizer:
                return pad_bucket_end(
                    bucket_end_index,
                    self.data_parallel_world_size,
                    self.ddp_config.pad_buckets_for_high_nccl_busbw,
                )
            return bucket_end_index

        self.nvfp4_packed_param_index_map = {}
        self.nvfp4_packed_bucket_indices = []
        nvfp4_packed_per_bucket_numel_unpadded = []

        packed_param_start = 0
        packed_bucket_start = 0
        cur_bucket_id = 0

        for param, _ in params_with_names[::-1]:
            _, _, bucket_id = self.param_index_map[param]
            param_numel = param.data.nelement()

            packed_param_start = _pad_start_of_param(packed_param_start)

            # Finalize previous bucket if we've moved to a new one.
            if bucket_id != cur_bucket_id:
                # Record unpadded numel, then pad the bucket end.
                nvfp4_packed_per_bucket_numel_unpadded.append(
                    packed_param_start - packed_bucket_start
                )
                packed_bucket_end = _pad_end_of_bucket(packed_param_start)
                self.nvfp4_packed_bucket_indices.append((packed_bucket_start, packed_bucket_end))
                packed_bucket_start = packed_bucket_end
                packed_param_start = packed_bucket_start
                cur_bucket_id = bucket_id

            # NVFP4 tensors use half the numel in the packed param buffer.
            if is_nvfp4tensor(param):
                assert (
                    param_numel % 2 == 0
                ), f"NVFP4 requires even numel for packing, got {param_numel}"
                packed_numel = param_numel // 2
            else:
                packed_numel = param_numel

            packed_param_end = packed_param_start + packed_numel
            self.nvfp4_packed_param_index_map[param] = (
                packed_param_start,
                packed_param_end,
                bucket_id,
            )
            packed_param_start = packed_param_end

        # Finalize last bucket.
        if packed_param_start > packed_bucket_start:
            nvfp4_packed_per_bucket_numel_unpadded.append(packed_param_start - packed_bucket_start)
            packed_bucket_end = _pad_end_of_bucket(packed_param_start)
            self.nvfp4_packed_bucket_indices.append((packed_bucket_start, packed_bucket_end))

        assert len(self.nvfp4_packed_bucket_indices) == len(self.bucket_indices), (
            f"Packed bucket count ({len(self.nvfp4_packed_bucket_indices)}) != "
            f"primary bucket count ({len(self.bucket_indices)})"
        )
        self.nvfp4_packed_numel_unpadded = sum(nvfp4_packed_per_bucket_numel_unpadded)

    def scale_gradients(self, scaling_factor: float) -> None:
        """Scale the gradient data by `scaling_factor`."""
        self.grad_data *= scaling_factor
        for grad in self.extra_main_grads:
            grad *= scaling_factor

    def _get(self, shape: torch.Size, start_index: int, buffer_type: BufferType) -> torch.Tensor:
        """
        Return a tensor with the input `shape` as a view into the 1-D data starting at
        `start_index`.
        """
        end_index = start_index + shape.numel()
        if buffer_type == BufferType.PARAM:
            numel = self.nvfp4_packed_numel if self.has_nvfp4_params else self.numel
            assert end_index <= numel, "Requested tensor is out of param buffer range"
            assert self.param_data is not None
            buffer_tensor = self.param_data[start_index:end_index]
        elif buffer_type == BufferType.GRAD:
            assert end_index <= self.numel, "Requested tensor is out of grad buffer range"
            buffer_tensor = self.grad_data[start_index:end_index]
        else:
            raise Exception("Illegal buffer type provided to GradBuffer._get() function")
        buffer_tensor = buffer_tensor.view(shape)
        return buffer_tensor

    def _new_bucket(
        self,
        bucket_params: List[torch.nn.Parameter],
        start_index: int,
        end_index: int,
        numel_unpadded: int,
        bucket_id: int,
        bucket_params_with_extra_main_grads: List[torch.Tensor],
        nvfp4_packed_start_index: int = None,
        nvfp4_packed_end_index: int = None,
    ) -> _ParamAndGradBucket:
        """
        Helper function that creates a new bucket. Also updates param->bucket mapping.

        For NVFP4 buffers, nvfp4_packed_start_index and nvfp4_packed_end_index
        are provided separately because the param buffer uses packed numel while
        the grad buffer uses full numel.
        """

        # Assert that indices are correctly padded (if needed), and that bucket
        # position is same as originally computed.

        if self.ddp_config.use_distributed_optimizer:
            assert start_index % self.data_parallel_world_size == 0
            assert end_index % self.data_parallel_world_size == 0
        assert (start_index, end_index) == self.bucket_indices[bucket_id]
        if nvfp4_packed_start_index is not None:
            assert (
                nvfp4_packed_start_index,
                nvfp4_packed_end_index,
            ) == self.nvfp4_packed_bucket_indices[bucket_id]

        # Get appropriate view into global _ParamAndGradBuffer.
        # For NVFP4, param buffer uses packed offsets; otherwise same as start/end.
        bucketed_param_data = None
        if self.param_data is not None:
            if nvfp4_packed_start_index is not None:
                assert nvfp4_packed_end_index is not None
                bucketed_param_data = self._get(
                    torch.Size([nvfp4_packed_end_index - nvfp4_packed_start_index]),
                    nvfp4_packed_start_index,
                    buffer_type=BufferType.PARAM,
                )
            else:
                bucketed_param_data = self._get(
                    torch.Size([end_index - start_index]), start_index, buffer_type=BufferType.PARAM
                )
        # Grad buffer always uses full-numel offsets.
        bucketed_grad_data = self._get(
            torch.Size([end_index - start_index]), start_index, buffer_type=BufferType.GRAD
        )
        bucket = _ParamAndGradBucket(
            params=bucket_params,
            param_data=bucketed_param_data,
            grad_data=bucketed_grad_data,
            offset=start_index,
            numel_unpadded=numel_unpadded,
            gradient_scaling_factor=self.gradient_scaling_factor,
            bucket_id=bucket_id,
            param_index_map=self.param_index_map,
            params_with_extra_main_grads=bucket_params_with_extra_main_grads,
        )
        for bucket_param in bucket_params:
            assert bucket_param not in self.param_to_bucket
            self.param_to_bucket[bucket_param] = bucket

        return bucket

    def reset(self):
        """
        Zero out the underlying grad_buffer.
        """
        self.grad_data.zero_()
        for grad in self.extra_main_grads:
            grad.zero_()

    def offload_to_cpu(self, move_params: bool = True, move_grads: bool = True) -> None:
        """
        Offload the buffers to CPU.
        """
        if move_grads and self.grad_data is not None and self.grad_data.storage().size() > 0:
            self.grad_data_size = self.grad_data.storage().size()
            self.grad_data.storage().resize_(0)
        if move_params and self.param_data is not None and self.param_data.storage().size() > 0:
            self.param_data_size = self.param_data.storage().size()
            if self.param_data_cpu is not None:
                self.param_data_cpu.copy_(self.param_data, non_blocking=True)
            else:
                self.param_data_cpu = self.param_data.cpu().pin_memory()
            self.param_data.storage().resize_(0)

    def reload_from_cpu(self, move_params: bool = True, move_grads: bool = True):
        """
        Reload the buffers from CPU.
        """
        if (
            move_params
            and self.param_data is not None
            and self.param_data_cpu is not None
            and self.param_data.storage().size() == 0
        ):
            self.param_data.storage().resize_(self.param_data_size)
            self.param_data.copy_(self.param_data_cpu, non_blocking=True)
        if move_grads and self.grad_data is not None and self.grad_data_size > 0:
            self.grad_data.storage().resize_(self.grad_data_size)
            self.grad_data.zero_()
            self.grad_data_size = 0


def partition_buckets(
    buffers: List[_ParamAndGradBuffer],
    force_single_bucket_group: bool = False,
    reduce_scatter_with_fp32_accumulation: bool = False,
) -> List[_ParamAndGradBucketGroup]:
    """
    Automatically regroup the buckets of input buffers and return a list of bucket groups.

    In some scenarios, we need to put buckets from different buffers into a group so that their
    communication can be aggregated.

    For example, when there are both fp8 weights and bf16 biases in the model and virtual
    pipeline parallelism is enabled, each model chunk will have an fp8 bucket and a bf16 bucket,
    which doubles the number of communication kernels, and because of the use of
    CUDA_DEVICE_MAX_CONNECTIONS=1, having multiple back-to-back communications will prevent the
    overlap of communication kernels with computation kernels.

    The grouping strategy is:
    1. If force_single_bucket_group is True, put all buckets across all buffers into a single
       bucket group.
    2. If force_single_bucket_group is False, when there is no fp8 buffer in the input buffers,
       let each bucket group have only one bucket.
    3. If force_single_bucket_group is False, when using fp8 params, merge all non-fp8 buckets
       into the last fp8 bucket group.
       - Since the non-fp8 parameters (typically the biases of various layers) are relatively
         small, they are likely to be grouped into a single non-fp8 bucket.
       - The fp8 buckets start from the end of the model, i.e., the first bucket corresponds to
         the end of the model, while the last bucket corresponds to the beginning.
       - If we combine the non-fp8 bucket with the first fp8 bucket, we cannot initiate the
         reduce-scatter to synchronize gradients after the backward pass at the end of the model
         has completed. This is because we need to wait for the non-fp8 params from the beginning
         layers to obtain their gradients.
       - Combining the non-fp8 bucket with the last fp8 bucket can help avoid this issue.

    Args:
        buffers (list): list of input buffers.
        single_bucket_group_per_buffer (bool, optional): force group all buckets in each buffer
            into a single bucket group.
    """

    if len(buffers) == 0:
        return []

    dtype_to_buffer_map = {}
    for buffer in buffers:
        dtype = buffer.param_dtype
        # Make sure that the param_dtype of any two buffers is different.
        assert dtype not in dtype_to_buffer_map
        dtype_to_buffer_map[dtype] = buffer

    # Case 1: Put all buckets into a single bucket group if force_single_bucket_group is True.
    if force_single_bucket_group:
        buckets = []
        ddp_config = buffers[0].ddp_config
        data_parallel_group = buffers[0].data_parallel_group
        data_parallel_world_size = buffers[0].data_parallel_world_size
        for buffer in buffers:
            assert ddp_config == buffer.ddp_config
            assert data_parallel_group == buffer.data_parallel_group
            assert data_parallel_world_size == buffer.data_parallel_world_size
            buckets.extend(buffer.buckets)

        bucket_group = _ParamAndGradBucketGroup(
            buckets, ddp_config, data_parallel_group, data_parallel_world_size
        )
        return [bucket_group]

    if torch.uint8 not in dtype_to_buffer_map:
        # Case 2: When there is no fp8 buffer in the input buffers, let each bucket group have
        #         only one bucket.
        bucket_groups = []
        for buffer in buffers:
            for bucket in buffer.buckets:
                bucket_groups.append(
                    _ParamAndGradBucketGroup(
                        [bucket],
                        buffer.ddp_config,
                        buffer.data_parallel_group,
                        buffer.data_parallel_world_size,
                    )
                )
        return bucket_groups
    else:
        # Case 3: When using fp8 params, merge all non-fp8 buckets into the last fp8 bucket group.
        non_fp8_buckets = []
        for buffer in buffers:
            if buffer.param_dtype != torch.uint8:
                for bucket in buffer.buckets:
                    non_fp8_buckets.append(bucket)

        bucket_groups = []
        fp8_buffer = dtype_to_buffer_map[torch.uint8]
        for bucket in fp8_buffer.buckets:
            if len(bucket_groups) == len(fp8_buffer.buckets) - 1:
                # reduce_scatter_with_fp32_accumulation requires exactly one bucket
                # per group (see assert in _ParamAndGradBucketGroup.reduce_scatter).
                # Without this flag the non-FP8 buckets would be merged into the last
                # FP8 group, violating that constraint. So we split them out into
                # their own individual groups instead.
                if reduce_scatter_with_fp32_accumulation:
                    bucket_groups.append(
                        _ParamAndGradBucketGroup(
                            [bucket],
                            buffer.ddp_config,
                            buffer.data_parallel_group,
                            buffer.data_parallel_world_size,
                        )
                    )
                    if non_fp8_buckets:
                        for non_fp8_bucket in non_fp8_buckets:
                            bucket_groups.append(
                                _ParamAndGradBucketGroup(
                                    [non_fp8_bucket],
                                    buffer.ddp_config,
                                    buffer.data_parallel_group,
                                    buffer.data_parallel_world_size,
                                )
                            )

                    continue  # Skip the default bucket group creation below
                else:
                    group_buckets = [bucket] + non_fp8_buckets
            else:
                # The first N-1 bucket groups.
                group_buckets = [bucket]
            bucket_groups.append(
                _ParamAndGradBucketGroup(
                    group_buckets,
                    buffer.ddp_config,
                    buffer.data_parallel_group,
                    buffer.data_parallel_world_size,
                )
            )
        return bucket_groups
