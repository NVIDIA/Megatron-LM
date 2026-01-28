# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

try:
    from packaging.version import Version
except ImportError:
    pass

from .distributed_data_parallel import DistributedDataParallel
from .distributed_data_parallel_config import DistributedDataParallelConfig
from .finalize_model_grads import finalize_model_grads
from .fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from .grad_buffer_offload import (
    GradBufferOffloadState,
    offload_grad_data,
    onload_grad_data,
    get_grad_buffer_memory_usage,
)
from .torch_fully_sharded_data_parallel import TorchFullyShardedDataParallel
from .torch_fully_sharded_data_parallel_config import TorchFullyShardedDataParallelConfig
