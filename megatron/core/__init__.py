import megatron.core.parallel_state
import megatron.core.tensor_parallel
import megatron.core.utils

from .inference_params import InferenceParams
from .model_parallel_config import ModelParallelConfig

# Alias parallel_state as mpu, its legacy name
mpu = parallel_state

__all__ = ["parallel_state", "tensor_parallel", "utils", "InferenceParams", "ModelParallelConfig"]
