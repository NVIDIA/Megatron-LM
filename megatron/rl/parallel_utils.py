# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""
Utilities for building process groups for RL inference models with custom parallelism.
"""

from typing import Optional

import torch.distributed

from megatron.core import mpu
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.process_groups_config import ProcessGroupCollection


def build_inference_pg_collection(
    tp_size: int,
    world_size: int,
    use_tp_pp_dp_mapping: bool = False,
    cp_size: Optional[int] = None,
    pp_size: Optional[int] = None,
    ep_size: Optional[int] = None,
) -> ProcessGroupCollection:
    """
    Build a ProcessGroupCollection for an RL inference model with custom parallelism settings.

    This is used when the inference model needs a different tensor parallel size than the
    training model. The function creates all necessary process groups (TP, CP, PP, EP, DP,
    and composite groups) using HyperCommGrid.

    Args:
        tp_size: Tensor model parallel size for the inference model.
        world_size: Total world size (number of ranks).
        use_tp_pp_dp_mapping: If True, use 'tp-cp-ep-pp-dp' order; otherwise 'tp-cp-ep-dp-pp'.
        cp_size: Context parallel size. Defaults to current MPU value if None.
        pp_size: Pipeline parallel size. Defaults to current MPU value if None.
        ep_size: Expert parallel size. Defaults to current MPU value if None.

    Returns:
        ProcessGroupCollection configured for the inference model.

    Raises:
        AssertionError: If world size is not divisible by tp*cp*pp.
    """
    # Use current MPU values as defaults
    if cp_size is None:
        cp_size = mpu.get_context_parallel_world_size()
    if pp_size is None:
        pp_size = mpu.get_pipeline_model_parallel_world_size()
    if ep_size is None:
        ep_size = mpu.get_expert_model_parallel_world_size()

    dp_size = world_size // (tp_size * cp_size * pp_size)
    assert dp_size >= 1 and (tp_size * cp_size * pp_size * dp_size) == world_size, \
        "World size must be divisible by tp*cp*pp for inference PG layout"

    # Build process group grid with appropriate dimension ordering
    if use_tp_pp_dp_mapping:
        # Order: tp-cp-ep-pp-dp (pp before dp)
        grid = HyperCommGrid(
            [tp_size, cp_size, ep_size, pp_size, dp_size],
            ["tp", "cp", "ep", "pp", "dp"]
        )
    else:
        # Order: tp-cp-ep-dp-pp (dp before pp) - this is the default
        grid = HyperCommGrid(
            [tp_size, cp_size, ep_size, dp_size, pp_size],
            ["tp", "cp", "ep", "dp", "pp"]
        )

    # Create base process groups
    tp_group = grid.create_pg("tp")
    cp_group = grid.create_pg("cp")
    pp_group = grid.create_pg("pp")
    ep_group = grid.create_pg("ep")
    dp_group = grid.create_pg("dp")

    # Create composite groups required by MoE/router and other utilities
    tp_cp_group = grid.create_pg(["tp", "cp"])
    mp_group = grid.create_pg(["tp", "cp", "ep", "pp"])
    tp_ep_group = grid.create_pg(["tp", "ep"])
    tp_ep_pp_group = grid.create_pg(["tp", "ep", "pp"])
    dp_cp_group = grid.create_pg(["cp", "dp"])
    tp_dp_cp_group = grid.create_pg(["tp", "cp", "dp"])

    # Create embedding groups
    embd_group_ranks = mpu.default_embedding_ranks(
        torch.distributed.get_process_group_ranks(pp_group)
    )
    embd_group = torch.distributed.new_group(ranks=embd_group_ranks)

    pos_embd_group_ranks = mpu.default_position_embedding_ranks(
        torch.distributed.get_process_group_ranks(pp_group)
    )
    pos_embd_group = torch.distributed.new_group(ranks=pos_embd_group_ranks)

    return ProcessGroupCollection(
        tp=tp_group,
        cp=cp_group,
        pp=pp_group,
        ep=ep_group,
        embd=embd_group,
        pos_embd=pos_embd_group,
        dp=dp_group,
        tp_cp=tp_cp_group,
        mp=mp_group,
        expt_tp=tp_group,
        expt_dp=dp_group,
        tp_ep=tp_ep_group,
        tp_ep_pp=tp_ep_pp_group,
        dp_cp=dp_cp_group,
        tp_dp_cp=tp_dp_cp_group,
    )
