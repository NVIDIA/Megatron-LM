# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Optional


@dataclass
class DistributedDataParallelConfig:
    """Configuration for DistributedDataParallel."""

    grad_reduce_in_fp32: bool = False
    """If true, reduce grads in fp32."""

    overlap_grad_reduce: bool = False
    """If true, overlap grad all-reduce / reduce-scatter with backward compute."""

    use_distributed_optimizer: bool = False
    """If true, issue reduce-scatter collectives to aggregate gradients and clean up
       originally allocated model parameters, otherwise issue all-reduce collectives.
    """

    check_for_nan_in_grad: bool = False
    """ If true, check for NaNs in gradients _before_ communication collective."""

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
