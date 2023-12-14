import megatron.core.tensor_parallel
import megatron.core.utils
import megatron.core.cpu_offload
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel
from megatron.core.inference_params import InferenceParams
from megatron.core.model_parallel_config import ModelParallelConfig

# Alias parallel_state as mpu, its legacy name
mpu = parallel_state

__all__ = [
    "parallel_state",
    "tensor_parallel",
    "utils",
    "cpu_offload",
    "DistributedDataParallel",
    "InferenceParams",
    "ModelParallelConfig",
]
