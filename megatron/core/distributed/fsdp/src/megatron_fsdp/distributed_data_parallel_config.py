# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Optional


@dataclass
class DistributedDataParallelConfig:
    """
    Megatron-FSDP `fully_shard` API sub-configuration
    derived from Megatron-Core DistributedDataParallel.
    """

    overlap_grad_reduce: bool = False
    """If true, overlap grad all-reduce / reduce-scatter with backward compute."""

    overlap_param_gather: bool = False
    """If true, overlap param all-gather with forward compute."""

    check_for_nan_in_grad: bool = False
    """
    If True, check for NaNs and Infs in gradients _before_ communication collective.
    Invoked by `start_grad_sync` such as in the Megatron-LM DDP training API.
    """

    bucket_size: Optional[int] = None
    """Maximum number of parameters in each bucket. If unspecified, MCore uses a default
       value of max(40000000, 1000000 * dp_size) parameters (larger DP sizes need larger
       buckets to ensure collectives do not become latency-bound)."""

    average_in_collective: bool = False
    """If true, compute average in collective directly, as opposed to dividing by the
       dp_size first and then computing sum in the collective."""

    fp8_param_gather: bool = False
    """If true, keep the compute param in fp8 (do not use any other intermediate dtype) and
       perform the param all-gather in fp8."""

    data_parallel_sharding_strategy: str = 'no_shard'
    """Sharding strategy for FSDP. Valid values are 'no_shard', 'optim',
      'optim_grads', 'optim_grads_params'."""

    gradient_reduce_div_fusion: bool = True
    """If true, perform gradient reduce and division fusion."""

    suggested_communication_unit_size: int = None
    """Specifies the number of elements to communicate at once during
      FSDP (Fully Sharded Data Parallel) operations. 
      This flag also affects FSDP all-gather prefetch behavior. Setting a larger
      value increases the communication buffer size, while a smaller value
      disables prefetching and may degrade performance. Adjust this value
      based on your system's memory and performance requirements."""

    keep_fp8_transpose_cache: bool = False
    """If true, keep the fp8 transpose cache when using Megatron FSDP."""

    nccl_ub: bool = False
    """If true, allocate and register NCCL userbuffer for param and grad buffer.
      This flag enables SM efficient nccl algorithm that could improve the performance
      of FSDP and DP with comm_overlap. This flag will be much more effective when used
      together with sharp. 
      The follwoing will be the expected number of SM usage for various cases.
      (Note that this is just a reference number and the number of SM usage could vary 
      on message size, communication domain size and nccl version.)
      ----------------------------------------------------------
      | Communication domain | use_sharp | SM usage of "AG/RS" |
      |----------------------|-----------|---------------------|
      | NVL                  | N/A       | 4 / 5               |
      | NVL+IB               | False     | 16 / 16             |
      | NVL+IB               | True      | 6 / 6               |
      | IB                   | False     | 1 / 4               |
      | IB                   | True      | 1 / 1               |
      ----------------------------------------------------------
    """

    fsdp_double_buffer: bool = False
    """If true, use persistently allocated double buffers for the 
      temporary memory needed in the Megatron FSDP communications.
      This option will cause additional memory overhead, however, it is necessary for
      to register user buffer (nccl_ub=True) for the Megatron FSDP. 
      This option will be automatically set to True when nccl_ub=True.
    """

    fsdp_db_use_persist_buf_on_alloc_fail: bool = False
    """Whether to fall back to persistent buffer when a bucket does not
       fit FSDP double buffer size. If true, FSDP will use the persistently 
       allocated buffer for the bucket that does not fit, it will enable NCCL 
       user buffer with the cost of more memory usage. If false, FSDP will use
       Dynamic memory allocator, NCCL user buffer won't not enabled, which 
       usually leads to low performance. 
    """

    outer_dp_sharding_strategy: str = 'no_shard'
    """
    Sharding strategy for outer data parallel group in Hybrid Sharded Data Parallel (HSDP) mode.
    Valid values are 'no_shard', 'optim'. This option is only effective when Hybrid FSDP is enabled.
    """

    disable_symmetric_registration: bool = False
    """If true, disable symmetric (window) registration for NCCL userbuffer registration.
      This option will force to use conventional (local) userbuffer registration 
      when nccl_ub is set.
    """

    fsdp_manual_registration: bool = False
    """If true, manually register the FSDP communication buffers to NCCL user buffer.
      This option is only effective when use_megatron_fsdp and nccl_ub is set.
      For symmetric registration with large models, the registration itself can take 
      a significant amount of time. This option minimizes the number of registration calls
      to minimize the registration time.
    """

    def __post_init__(self):
        import os

        """Check the validity of the config."""
        if self.nccl_ub:
            if 'expandable_segments:True' in os.getenv('PYTORCH_CUDA_ALLOC_CONF', '').split(','):
                raise ValueError(
                    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is currently not supported "
                    "with nccl_ub due to compatibility issue with torch.cuda.MemPool API."
                )
