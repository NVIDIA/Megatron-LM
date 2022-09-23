from .parallel_state import (
    initialize_model_parallel,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_pipeline_model_parallel_rank,
    get_virtual_pipeline_model_parallel_rank, set_virtual_pipeline_model_parallel_rank,
    get_data_parallel_world_size,
    get_data_parallel_rank,
    get_global_memory_buffer,
    get_num_layers,
)
from megatron.core import tensor_parallel
