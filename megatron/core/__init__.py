from .parallel_state import (
    initialize_model_parallel,
    get_tensor_model_parallel_world_size,
    get_pipeline_model_parallel_world_size,
    get_data_parallel_world_size,
)
from megatron.core import tensor_parallel
