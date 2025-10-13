from tests.unit_tests.pipeline_parallel.test_multimodule_schedules import create_hypercomm_grid, _get_pg_collection_with_embedding_groups
import torch.distributed as dist
from contextlib import contextmanager
from megatron.core.distributed.finalize_model_grads import finalize_model_grads as _finalize_model_grads

def is_current_rank_in_grid(grid) -> bool:
        """Check if the current rank is in the grid."""
        return grid.rank_offset <= dist.get_rank() < (grid.rank_offset + grid.size)


def get_module_to_grid_tuple(mimo_model, vision_module_grid, language_module_grid):
    return_tuple = [(mimo_model.modality_submodules['images'], vision_module_grid), (mimo_model.language_model, language_module_grid)]
    return return_tuple



@contextmanager
def multimodule_no_sync(module_to_grid_tuple):
    contexts = []
    for module, grid in module_to_grid_tuple:
        if module is not None and is_current_rank_in_grid(grid):
            contexts.append(module.no_sync())
    
    # Enter all contexts
    for ctx in contexts:
        ctx.__enter__()
    
    try:
        yield
    finally:
        # Exit all contexts in reverse order
        for ctx in reversed(contexts):
            ctx.__exit__(None, None, None)


def get_pg_collections_for_rank(module_to_grid_map):
    """Get pg_collections for modules that should be initialized on the current rank."""
    pg_collections = []
    for _ , grid_name in module_to_grid_map.items():
        if is_current_rank_in_grid(grid_name):
            pg_collections.append(_get_pg_collection_with_embedding_groups(grid_name))
    return pg_collections

def finalize_model_grads(model, num_tokens=None, pg_collection=None, *, module_to_grid_tuple):
    """Wrapper to call finalize_model_grads for each module in its respective grid.
    
    Args:
        model: Model list (passed by scheduler, but not used - we use module_to_grid_tuple instead)
        num_tokens: Number of tokens for gradient scaling
        pg_collection: Process group collection (passed by scheduler, but not used - we use grid-specific PGs)
        module_to_grid_tuple: Tuple of (module, grid) pairs to finalize grads for each module in its grid
    """
    for module, grid in module_to_grid_tuple:
        if module is not None and is_current_rank_in_grid(grid):
            _finalize_model_grads([module], num_tokens=num_tokens, pg_collection=_get_pg_collection_with_embedding_groups(grid))


def zero_grad_buffer_for_multimodule(module_to_grid_tuple):
    """Reset gradient buffers for all DDP-wrapped modules in their respective grids.
        
    Args:
        module_to_grid_tuple: Tuple of (module, grid) pairs to reset grads for each module
    """
    for module, grid in module_to_grid_tuple:
        if module is not None and is_current_rank_in_grid(grid):
            module.zero_grad_buffer()