# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""
Utilities for building process groups for RL inference models with custom parallelism.
"""

from typing import Optional

import torch.distributed as dist

from megatron.core import mpu
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.process_groups_config import ProcessGroupCollection


def build_inference_pg_collection(
    world_size: int,
    tp_size: Optional[int] = None,
    pp_size: Optional[int] = None,
    cp_size: Optional[int] = None,
    ep_size: Optional[int] = None,
    expt_tp_size: Optional[int] = None,
    use_tp_pp_dp_mapping: bool = False,
) -> ProcessGroupCollection:
    """
    Build a ProcessGroupCollection for an RL inference model with custom parallelism.

    Uses two HyperCommGrids matching the structure of mpu:
    - decoder_grid: for dense/attention layers (tp, cp, dp, pp)
    - expert_grid: for MoE expert layers (expt_tp, ep, expt_dp, pp)

    Args:
        world_size: Total world size (number of ranks).
        tp_size: Tensor model parallel size. Defaults to training's TP size.
        pp_size: Pipeline parallel size. Defaults to training's PP size.
        cp_size: Context parallel size. Defaults to training's CP size.
        ep_size: Expert parallel size. Defaults to training's EP size.
        expt_tp_size: Expert tensor parallel size. Defaults to training's expert TP size.
        use_tp_pp_dp_mapping: If True, use 'tp-pp-dp' order; otherwise 'tp-dp-pp'.

    Returns:
        ProcessGroupCollection configured for the inference model.
    """
    # Use current MPU values as defaults
    if tp_size is None:
        tp_size = mpu.get_tensor_model_parallel_world_size()
    if cp_size is None:
        cp_size = mpu.get_context_parallel_world_size()
    if pp_size is None:
        pp_size = mpu.get_pipeline_model_parallel_world_size()
    if ep_size is None:
        ep_size = mpu.get_expert_model_parallel_world_size()
    if expt_tp_size is None:
        expt_tp_size = mpu.get_expert_tensor_parallel_world_size()


    # Compute DP size for dense layers (same formula as mpu)
    # world = tp × cp × dp × pp
    dp_size = world_size // (tp_size * cp_size * pp_size)
    assert dp_size >= 1 and (tp_size * cp_size * dp_size * pp_size) == world_size, (
        f"World size ({world_size}) must be divisible by tp*cp*pp ({tp_size * cp_size * pp_size})"
    )

    # Compute expert DP size (same formula as mpu)
    # world = expt_tp × ep × expt_dp × pp
    expt_dp_size = world_size // (expt_tp_size * ep_size * pp_size)
    assert expt_dp_size >= 1 and (expt_tp_size * ep_size * expt_dp_size * pp_size) == world_size, (
        f"World size ({world_size}) must be divisible by expt_tp*ep*pp ({expt_tp_size * ep_size * pp_size})"
    )

    rank = dist.get_rank()

    # ====================
    # Create decoder grid for dense/attention layers
    # Matches mpu's decoder_rank_generator with ep=1
    # ====================
    if use_tp_pp_dp_mapping:
        # Order: tp-cp-pp-dp
        decoder_grid = HyperCommGrid(
            [tp_size, cp_size, pp_size, dp_size],
            ["tp", "cp", "pp", "dp"]
        )
    else:
        # Order: tp-cp-dp-pp (default)
        decoder_grid = HyperCommGrid(
            [tp_size, cp_size, dp_size, pp_size],
            ["tp", "cp", "dp", "pp"]
        )

    # Create dense layer groups from decoder_grid
    tp_group = decoder_grid.create_pg("tp")
    cp_group = decoder_grid.create_pg("cp")
    pp_group = decoder_grid.create_pg("pp")
    dp_group = decoder_grid.create_pg("dp")
    mp_group = decoder_grid.create_pg(["tp", "pp"])
    tp_cp_group = decoder_grid.create_pg(["tp", "cp"])
    dp_cp_group = decoder_grid.create_pg(["cp", "dp"])
    tp_dp_cp_group = decoder_grid.create_pg(["tp", "cp", "dp"])

    # ====================
    # Create expert grid for MoE expert layers
    # Matches mpu's expert_decoder_rank_generator with cp=1
    # ====================
    if use_tp_pp_dp_mapping:
        # Order: tp-ep-pp-dp
        expert_grid = HyperCommGrid(
            [expt_tp_size, ep_size, pp_size, expt_dp_size],
            ["tp", "ep", "pp", "dp"]
        )
    else:
        # Order: tp-ep-dp-pp (default)
        expert_grid = HyperCommGrid(
            [expt_tp_size, ep_size, expt_dp_size, pp_size],
            ["tp", "ep", "dp", "pp"]
        )

    # Verify PP groups match between decoder and expert grids (required by mpu)
    decoder_pp_enum = decoder_grid.get_rank_enum("pp")
    expert_pp_enum = expert_grid.get_rank_enum("pp")
    assert decoder_pp_enum == expert_pp_enum, (
        f"PP groups must match between decoder and expert grids. "
        f"Decoder: {decoder_pp_enum}, Expert: {expert_pp_enum}"
    )

    # Create expert layer groups from expert_grid
    ep_group = expert_grid.create_pg("ep")
    expt_tp_group = expert_grid.create_pg("tp")
    expt_dp_group = expert_grid.create_pg("dp")
    tp_ep_group = expert_grid.create_pg(["tp", "ep"])
    tp_ep_pp_group = expert_grid.create_pg(["tp", "ep", "pp"])

    # ====================
    # Embedding groups (derived from PP groups)
    # ====================
    embd_group = None
    pos_embd_group = None

    pp_rank_enum = decoder_grid.get_rank_enum("pp")
    for pp_ranks in pp_rank_enum:
        # Embedding is on first and last PP stage
        if len(pp_ranks) == 1:
            embd_ranks = [pp_ranks[0]]
        else:
            embd_ranks = [pp_ranks[0], pp_ranks[-1]]
        group = dist.new_group(ranks=embd_ranks)
        if rank in embd_ranks:
            embd_group = group

        # Position embedding is only on first PP stage
        pos_embd_ranks = [pp_ranks[0]]
        group = dist.new_group(ranks=pos_embd_ranks)
        if rank in pos_embd_ranks:
            pos_embd_group = group

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
        expt_tp=expt_tp_group,
        expt_dp=expt_dp_group,
        tp_ep=tp_ep_group,
        tp_ep_pp=tp_ep_pp_group,
        dp_cp=dp_cp_group,
        tp_dp_cp=tp_dp_cp_group,
    )
