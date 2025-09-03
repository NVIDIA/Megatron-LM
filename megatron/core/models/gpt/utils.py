from megatron.core.parallel_state import (
    get_pipeline_model_parallel_rank,
    get_virtual_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size
)

def offloading_checker(tensor):
    return hasattr(tensor, "offloading_activation") and tensor.offloading_activation

def get_first_layer_index(config, num_layers_per_pipeline_rank):
    if 'core_attn' in config.offload_modules:
        return 0
    pp_rank = get_pipeline_model_parallel_rank()
    pp_size = get_pipeline_model_parallel_world_size()
    vpp_rank = get_virtual_pipeline_model_parallel_rank()
    layer_index_start = num_layers_per_pipeline_rank * (pp_size * vpp_rank + pp_rank)
    if config.first_k_dense_replace > layer_index_start:
        return config.first_k_dense_replace - layer_index_start
    else:
        return 0