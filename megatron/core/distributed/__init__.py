# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

try:
    from packaging.version import Version
except ImportError:
    pass

from .distributed_data_parallel import DistributedDataParallel
from .distributed_data_parallel_config import DistributedDataParallelConfig
from .finalize_model_grads import finalize_model_grads
from .fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from .torch_fully_sharded_data_parallel import TorchFullyShardedDataParallel
from .torch_fully_sharded_data_parallel_config import TorchFullyShardedDataParallelConfig

# Backward compatibility patch for FSDP module reorganization
import sys
import importlib.util

spec = importlib.util.find_spec('megatron.core.distributed.fsdp.src.megatron_fsdp')
if spec:
    custom_fsdp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom_fsdp)
    sys.modules['megatron.core.distributed.custom_fsdp'] = custom_fsdp
    if hasattr(custom_fsdp, 'MegatronFSDP'):
        custom_fsdp.FullyShardedDataParallel = custom_fsdp.MegatronFSDP
