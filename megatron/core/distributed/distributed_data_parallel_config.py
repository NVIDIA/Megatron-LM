# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Optional, Tuple

from ..utils import is_torch_min_version


@dataclass
class DistributedDataParallelConfig:
    """Configuration for DistributedDataParallel and its distributed optimizer."""

    grad_reduce_in_fp32: bool = False
    """If true, reduce grads in fp32."""

    overlap_grad_reduce: bool = False
    """If true, overlap grad all-reduce / reduce-scatter with backward compute."""

    overlap_param_gather: bool = False
    """If true, overlap param all-gather with forward compute."""

    align_param_gather: bool = False
    """If true, all PP stages will launch param all-gathers simultaneously. Otherwise, each
    PP stage will independently launch as needed.
    """

    use_distributed_optimizer: bool = False
    """If true, issue reduce-scatter collectives to aggregate gradients and clean up
       originally allocated model parameters, otherwise issue all-reduce collectives.
    """

    num_distributed_optimizer_instances: int = 1
    """Sets the factor by which the DP domain is sharded to have the partial DistOpt
       enabled. Defaults to 1, which means DistOpt is across entire DP domain.
    """

    check_for_nan_in_grad: bool = False
    """
    If true, check for NaNs and Infs in gradients _before_ communication collective.
    Invoked by `start_grad_sync` such as in the Megatron-LM DDP training API.
    """

    check_for_large_grads: bool = False
    """If true, check for unexpectedly large gradients _before_ communication collective."""

    bucket_size: Optional[int] = None
    """Maximum number of parameters in each bucket. If unspecified, MCore uses a default
       value of max(40000000, 1000000 * dp_size) parameters (larger DP sizes need larger
       buckets to ensure collectives do not become latency-bound)."""

    num_buckets: Optional[int] = None
    """Number of buckets for data-parallel communication. Should only specify one of
       `bucket_size` and `num_buckets`. If `num_buckets` is specified, `bucket_size`
       will be determined at runtime."""

    pad_buckets_for_high_nccl_busbw: bool = False
    """If true, make sure the bucket size is divisible by a large power of 2 (2^16) to
       ensure NCCL collectives have high bus bandwidth at large DP counts, since NCCL
       message size (which for ring algorithms is bucket_size / dp_size) apparently needs
       to be divisible by a power of 2 for high busbw."""

    reduce_scatter_with_fp32_accumulation: bool = False
    """If true, use a reduce-scatter implementation which sends lower-precision values
       over the wire (using an all-to-all to keep total communication overhead in line
       with the standard ring implementation) but performs accumulation locally in FP32."""

    param_name_patterns_for_fp32_local_accumulation: Tuple[str, ...] = ()
    """List of param_name patterns (in Python's fnmatch format) to match against to do
       local gradient accumulation in FP32. The special pattern 'all' matches every
       parameter. Do not specify when grad_reduce_in_fp32 is already True."""

    average_in_collective: bool = False
    """If true, compute average in collective directly, as opposed to dividing by the
       dp_size first and then computing sum in the collective."""

    fp8_param_gather: bool = False
    """If true, keep the compute param in fp8 (do not use any other intermediate dtype) and
       perform the param all-gather in fp8."""

    fp4_param_gather: bool = False
    """If true, keep the compute param in fp4 (do not use any other intermediate dtype) and
       perform the param all-gather in fp4."""

    reuse_grad_buf_for_mxfp8_param_ag: bool = False
    """If true, reuse the grad buffer for param AG when using mxfp8 recipe. Should be 
       set to True only when fp8_recipe is mxfp8 and fp8_param_gather is True."""

    nccl_ub: bool = False
    """If true, allocate and register NCCL userbuffer for param and grad buffer.
      This flag enables SM efficient nccl algorithm that could improve the performance
      of FSDP and DP with comm_overlap. This flag will be much more effective when used
      together with sharp. 
      The follwoing will be the expected number of SM usage for various cases.
      (Note that this is just a reference number and the number of SM usage could vary 
      on message size, communication domain size and nccl version.)
      | Communication domain | use_sharp | SM usage of "AG/RS" |
      |----------------------|-----------|---------------------|
      | NVL                  | N/A       | 4 / 5               |
      | NVL+IB               | False     | 16 / 16             |
      | NVL+IB               | True      | 6 / 6               |
      | IB                   | False     | 1 / 4               |
      | IB                   | True      | 1 / 1               |
    """

    grad_buffer_offload: bool = False
    """If true, gradient buffers may be released between training phases.

    With ``nccl_ub``, gradient storage is allocated in a separate NCCL memory
    pool so that it can be deregistered, released, recreated, and registered
    again without disturbing persistent parameter storage.
    """

    disable_symmetric_registration: bool = False
    """If true, disable symmetric (window) registration for NCCL userbuffer registration.
      This option will force to use conventional (local) userbuffer registration 
      when nccl_ub is set.
    """

    delay_wgrad_compute: bool = False
    """Delay the weight gradient computation to improve batch-level communication overlapping"""

    @property
    def param_sync_via_bucket_group(self) -> bool:
        """Whether DP parameter synchronization is dispatched through DDP bucket groups.

        ``True``:
        ``_ParamAndGradBucketGroup.start_param_sync()`` is the collective entry point. This is
        the standard DistributedOptimizer path. LayerWise also uses it when overlap needs
        bucket-level prefetch, or when MXFP8 reuse needs ``bucket.grad_data`` as the gather
        buffer. With overlap, the current bucket's forward pre-hook waits for its gather and
        dispatches the next bucket's gather before computation starts, allowing communication
        for the next bucket to overlap with the current computation.

        ``False``:
        DDP bucket groups do not launch parameter all-gather. Then either:

        - No gather is needed because every DP rank runs the same optimizer update.
        - Another component performs the gather. With ``LayerWiseDistributedOptimizer``, this
          happens when ``use_layer_wise_param_layout=False``, parameter-gather overlap is disabled,
          and MXFP8 grad-buffer reuse is disabled. Each rank updates only its assigned parameters,
          then the optimizer calls ``allgather_params()`` synchronously after the step.
        """
        if self.use_distributed_optimizer:
            return True

        # When standard DistOpt is disabled, these flags select the legacy LayerWise
        # bucket-group path: forward-scheduled overlap or synchronous grad-buffer reuse.
        return self.overlap_param_gather or self.reuse_grad_buf_for_mxfp8_param_ag

    def __post_init__(self):
        import os

        """Check the validity of the config."""
        if self.reuse_grad_buf_for_mxfp8_param_ag:
            assert self.fp8_param_gather, "Reuse grad buffer only when keeping params in MXFP8."

        if self.grad_buffer_offload and self.reuse_grad_buf_for_mxfp8_param_ag:
            raise ValueError(
                "Gradient-buffer offload is incompatible with "
                "reuse_grad_buf_for_mxfp8_param_ag because parameters share gradient storage."
            )

        if self.nccl_ub and not is_torch_min_version("2.11.0a0"):
            if 'expandable_segments:True' in os.getenv('PYTORCH_CUDA_ALLOC_CONF', '').split(','):
                raise ValueError(
                    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is currently not supported "
                    "with nccl_ub due to compatibility issue with torch.cuda.MemPool API."
                )

        if len(self.param_name_patterns_for_fp32_local_accumulation) > 0:
            assert not self.grad_reduce_in_fp32, (
                "Only need to explicitly specify param_name patterns for FP32 local accumulation "
                "if .main_grads aren't already in FP32"
            )

        if self.num_buckets is not None:
            assert self.bucket_size is None, "Cannot specify both num_buckets and bucket_size"
            assert self.num_buckets > 0, "num_buckets must be greater than 0"
