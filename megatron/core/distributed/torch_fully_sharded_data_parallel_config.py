# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Union

from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig


@dataclass
class TorchFullyShardedDataParallelConfig(DistributedDataParallelConfig):
    """Configuration for TorchFullyShardedDataParallel."""

    reshard_after_forward: Union[bool, int] = True
    """
    Controls the parameter behavior after forward.

    See PyTorch for complete documentation:
    https://github.com/pytorch/pytorch/blob/ac8ddf115065106f038865389a07f2d0c9ed5e11/torch/distributed/fsdp/_fully_shard/_fully_shard.py#L97C31-L97C49 # pylint: disable=line-too-long 
    """
