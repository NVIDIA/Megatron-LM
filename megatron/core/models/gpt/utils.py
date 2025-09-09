from megatron.core.parallel_state import (
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size
)

def offloading_checker(tensor):
    return hasattr(tensor, "offloading_activation") and tensor.offloading_activation
