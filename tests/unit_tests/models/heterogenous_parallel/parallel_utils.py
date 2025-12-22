from tests.unit_tests.pipeline_parallel.test_multimodule_schedules import create_hypercomm_grid, _get_pg_collection_with_embedding_groups
import torch.distributed as dist
from contextlib import contextmanager
from megatron.core.distributed.finalize_model_grads import finalize_model_grads as _finalize_model_grads
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.hyper_comm_grid import HyperCommGrid
from typing import Optional
import torch

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
        pg_collection: Process group collection
        module_to_grid_tuple: Tuple of (module, grid) pairs to finalize grads for each module in its grid
    """
    for module, grid in module_to_grid_tuple:
        if module is not None and is_current_rank_in_grid(grid):
            _finalize_model_grads([module], num_tokens=num_tokens, pg_collection=pg_collection)


def zero_grad_buffer_for_multimodule(module_to_grid_tuple):
    """Reset gradient buffers for all DDP-wrapped modules in their respective grids.
        
    Args:
        module_to_grid_tuple: Tuple of (module, grid) pairs to reset grads for each module
    """
    for module, grid in module_to_grid_tuple:
        if module is not None and is_current_rank_in_grid(grid):
            module.zero_grad_buffer()


def _create_pg_collection(
    tp_size: int, pp_size: int, cp_size: int, ep_size: int, num_distributed_optimizer_instances: int, rank_offset: int, world_size: int
) -> ProcessGroupCollection:
    """Create all process groups via HyperCommGrid and return a ProcessGroupCollection."""
    # world_size = torch.distributed.get_world_size()
    model_size = tp_size * pp_size * cp_size
    if world_size % model_size != 0:
        raise RuntimeError(f"world_size ({world_size}) is not divisible by {model_size}")
    dp_size = world_size // model_size

    grid = HyperCommGrid(
        shape=[tp_size, cp_size, dp_size, pp_size],
        dim_names=["tp", "cp", "dp", "pp"],
        rank_offset=rank_offset,
        backend="nccl",
    )
    # Core groups
    tp_pg = grid.create_pg(["tp"])
    cp_pg = grid.create_pg(["cp"])
    pp_pg = grid.create_pg(["pp"])
    dp_pg = grid.create_pg(["dp"])
    mp_pg = grid.create_pg(["tp", "pp"])
    tp_cp_pg = grid.create_pg(["tp", "cp"])
    tp_dp_cp_pg = grid.create_pg(["tp", "dp", "cp"])
    dp_cp_pg = grid.create_pg(["dp", "cp"])

    # Expert/MoE related groups (refer to original parallel_state.initialize_model_parallel)
    expert_tp_size = 1 # TODO: add expert_tp_size as input argument
    # Expert data-parallel size folds CP into DP (as in original expert rank generator)
    expt_model_block = expert_tp_size * ep_size * pp_size
    if world_size % expt_model_block != 0:
        raise RuntimeError(
            f"world_size ({world_size}) is not divisible by expert_tensor_model_pipeline size ({expt_model_block})"
        )
    expt_dp_size = world_size // expt_model_block
    use_optimizer_instance_groups = num_distributed_optimizer_instances > 1
    inner_dp_dim: Optional[str] = None
    outer_dp_dim: Optional[str] = None
    if use_optimizer_instance_groups:
        assert expt_dp_size % num_distributed_optimizer_instances == 0, (
            "Expert DP size must be divisible by the number of optimizer instances."
        )
        inner_expt_dp_size = expt_dp_size // num_distributed_optimizer_instances
        expert_grid = HyperCommGrid(
            shape=[expert_tp_size, ep_size, inner_expt_dp_size, num_distributed_optimizer_instances, pp_size],
            dim_names=["tp", "ep", "inner_dp", "outer_dp", "pp"],
            rank_offset=rank_offset,
            backend="nccl",
        )
        dp_group_dims: list[str] = ["inner_dp", "outer_dp"]
        inner_dp_dim = "inner_dp"
        outer_dp_dim = "outer_dp"
    else:
        expert_grid = HyperCommGrid(
            shape=[expert_tp_size, ep_size, expt_dp_size, pp_size],
            dim_names=["tp", "ep", "dp", "pp"],
            rank_offset=rank_offset,
            backend="nccl",
        )
        dp_group_dims = ["dp"]
    ep_pg = expert_grid.create_pg(["ep"])
    expt_tp_pg = expert_grid.create_pg(["tp"])
    tp_ep_pg = expert_grid.create_pg(["tp", "ep"])
    tp_ep_pp_pg = expert_grid.create_pg(["tp", "ep", "pp"])
    expt_dp_pg = expert_grid.create_pg(dp_group_dims)

    # Embedding and position-embedding groups
    embd_pg = None
    pos_embd_pg = None
    # Enumerate ranks per PP group
    pp_rank_lists = grid._gen_rank_enum(["pp"])
    # Determine embedding ranks for each pp group
    embedding_rank_lists: list[list[int]] = []
    pos_embedding_rank_lists: list[list[int]] = []
    for ranks in pp_rank_lists:
        if not ranks:
            continue
        # embedding_ranks: first and last pp stage (or only one if pp_size==1)
        embedding_rank_lists.append([ranks[0]] if len(ranks) == 1 else [ranks[0], ranks[-1]])
        pos_embedding_rank_lists.append([ranks[0]])
    if embedding_rank_lists:
        embd_pg, _ = torch.distributed.new_subgroups_by_enumeration(embedding_rank_lists, backend="nccl")
    if pos_embedding_rank_lists:
        pos_embd_pg, _ = torch.distributed.new_subgroups_by_enumeration(pos_embedding_rank_lists, backend="nccl")

    # Build Partial-Distributed-Optimizer groups for Expert DP when multiple instances are used.
    intra_expt_dp_pg = None
    inter_dist_opt_pg = None
    intra_dist_opt_pg = None
    if inner_dp_dim is not None and outer_dp_dim is not None:
        intra_expt_dp_pg = expert_grid.create_pg([inner_dp_dim])
        inter_dist_opt_pg = expert_grid.create_pg([outer_dp_dim])
        # Match distributed optimizer instance grouping from parallel_state:
        # combine tp-ep-pp ranks across the intra-partial DP slice.
        intra_dist_opt_pg = expert_grid.create_pg(["tp", "ep", inner_dp_dim, "pp"])

    # Build ProcessGroupCollection with available groups.
    pg_collection = ProcessGroupCollection(
        tp=tp_pg,
        pp=pp_pg,
        mp=mp_pg,
        embd=embd_pg,
        pos_embd=pos_embd_pg,
        cp=cp_pg,
        tp_cp=tp_cp_pg,
        hcp=None,
        ep=ep_pg,
        expt_tp=expt_tp_pg,
        tp_ep=tp_ep_pg,
        tp_ep_pp=tp_ep_pp_pg,
        tp_dp_cp=tp_dp_cp_pg,
        dp=dp_pg,
        dp_cp=dp_cp_pg,
        expt_dp=expt_dp_pg,
        intra_dp_cp=dp_cp_pg,
        intra_expt_dp=intra_expt_dp_pg if intra_expt_dp_pg is not None else expt_dp_pg,
        inter_dist_opt=inter_dist_opt_pg,
        intra_dist_opt=intra_dist_opt_pg,
    )
    return pg_collection, grid, expert_grid
