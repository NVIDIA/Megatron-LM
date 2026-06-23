# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Framework-agnostic primitives for heterogeneous inference sharding: build a
``ProcessGroupCollection`` per shard, each over a contiguous rank window at its
own parallelism."""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import torch.distributed as dist

from megatron.core import mpu
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.inference.shards_spec import InferenceShardSpec, normalize_shard_specs
from megatron.core.process_groups_config import ProcessGroupCollection


def build_inference_pg_collection(
    world_size: int,
    tp_size: Optional[int] = None,
    pp_size: Optional[int] = None,
    cp_size: Optional[int] = None,
    ep_size: Optional[int] = None,
    expt_tp_size: Optional[int] = None,
    use_tp_pp_dp_mapping: bool = False,
    rank_offset: int = 0,
) -> ProcessGroupCollection:
    """Build a ProcessGroupCollection for one inference model.

    Uses two HyperCommGrids matching mpu:
    - decoder_grid for dense/attention layers (tp, cp, dp, pp)
    - expert_grid for MoE expert layers (expt_tp, ep, expt_dp, pp)

    Args:
        world_size: Number of ranks in this inference window.
        tp_size: Tensor model parallel size. Defaults to training's TP size.
        pp_size: Pipeline parallel size. Defaults to training's PP size.
        cp_size: Context parallel size. Defaults to training's CP size.
        ep_size: Expert parallel size. Defaults to training's EP size.
        expt_tp_size: Expert tensor parallel size. Defaults to training's
            expert TP size.
        use_tp_pp_dp_mapping: If True, use 'tp-pp-dp' order; otherwise
            'tp-dp-pp'.
        rank_offset: Starting global rank of the window. Use ``0`` for
            collocated inference (shares ranks with training); use a non-zero
            offset for non-collocated setups where inference ranks are disjoint
            from training ranks.

    Returns:
        ProcessGroupCollection configured for the inference model. On ranks
        outside the ``[rank_offset, rank_offset + world_size)`` window every
        process-group field is a non-member sentinel returned by
        :func:`torch.distributed.new_subgroups_by_enumeration` — callers should
        not use that instance; see
        :func:`build_inference_pg_collections_for_shards` for the right way to
        filter.
    """
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

    # Dense layer DP size (world = tp * cp * dp * pp)
    dp_size = world_size // (tp_size * cp_size * pp_size)
    assert dp_size >= 1 and (tp_size * cp_size * dp_size * pp_size) == world_size, (
        f"World size ({world_size}) must be divisible by tp*cp*pp "
        f"({tp_size * cp_size * pp_size})"
    )

    # Expert DP size (world = expt_tp * ep * expt_dp * pp)
    expt_dp_size = world_size // (expt_tp_size * ep_size * pp_size)
    assert expt_dp_size >= 1 and (expt_tp_size * ep_size * expt_dp_size * pp_size) == world_size, (
        f"World size ({world_size}) must be divisible by expt_tp*ep*pp "
        f"({expt_tp_size * ep_size * pp_size})"
    )

    rank = dist.get_rank()

    if use_tp_pp_dp_mapping:
        decoder_grid = HyperCommGrid(
            [tp_size, cp_size, pp_size, dp_size], ["tp", "cp", "pp", "dp"], rank_offset=rank_offset
        )
    else:
        decoder_grid = HyperCommGrid(
            [tp_size, cp_size, dp_size, pp_size], ["tp", "cp", "dp", "pp"], rank_offset=rank_offset
        )

    tp_group = decoder_grid.create_pg("tp")
    cp_group = decoder_grid.create_pg("cp")
    pp_group = decoder_grid.create_pg("pp")
    dp_group = decoder_grid.create_pg("dp")
    mp_group = decoder_grid.create_pg(["tp", "pp"])
    tp_cp_group = decoder_grid.create_pg(["tp", "cp"])
    dp_cp_group = decoder_grid.create_pg(["cp", "dp"])
    tp_dp_cp_group = decoder_grid.create_pg(["tp", "cp", "dp"])

    if use_tp_pp_dp_mapping:
        expert_grid = HyperCommGrid(
            [expt_tp_size, ep_size, pp_size, expt_dp_size],
            ["tp", "ep", "pp", "dp"],
            rank_offset=rank_offset,
        )
    else:
        expert_grid = HyperCommGrid(
            [expt_tp_size, ep_size, expt_dp_size, pp_size],
            ["tp", "ep", "dp", "pp"],
            rank_offset=rank_offset,
        )

    decoder_pp_enum = decoder_grid.get_rank_enum("pp")
    expert_pp_enum = expert_grid.get_rank_enum("pp")
    assert decoder_pp_enum == expert_pp_enum, (
        f"PP groups must match between decoder and expert grids. "
        f"Decoder: {decoder_pp_enum}, Expert: {expert_pp_enum}"
    )

    ep_group = expert_grid.create_pg("ep")
    expt_tp_group = expert_grid.create_pg("tp")
    expt_dp_group = expert_grid.create_pg("dp")
    tp_ep_group = expert_grid.create_pg(["tp", "ep"])
    tp_ep_pp_group = expert_grid.create_pg(["tp", "ep", "pp"])

    embd_group = None
    pos_embd_group = None
    pp_rank_enum = decoder_grid.get_rank_enum("pp")
    for pp_ranks in pp_rank_enum:
        if len(pp_ranks) == 1:
            embd_ranks = [pp_ranks[0]]
        else:
            embd_ranks = [pp_ranks[0], pp_ranks[-1]]
        group = dist.new_group(ranks=embd_ranks)
        if rank in embd_ranks:
            embd_group = group
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


@dataclass
class InferenceShard:
    """One shard in a multi-shard inference layout: its identity, its rank
    window, and this rank's process groups for it.

    Attributes:
        spec: This shard's :class:`~megatron.core.inference.shards_spec.InferenceShardSpec`
            (``tp``/``pp``/``ep``/``expt_tp``/``dp`` and optional
            ``role`` = ``prefill``/``decode``).
        rank_offset: First global rank belonging to this shard.
        world_size: Number of ranks in this shard (tp*pp*dp).
        pg_collection: The shard's ProcessGroupCollection if the current rank
            is a member of this shard, else ``None`` -- the ``is not None`` check
            is how a rank finds its own shard. Every rank still participates in
            the collective process-group creation for every shard
            (``dist.new_group`` is world-collective); only members get a usable
            handle.
    """

    spec: InferenceShardSpec
    rank_offset: int
    world_size: int
    pg_collection: Optional[ProcessGroupCollection]


def build_inference_pg_collections_for_shards(
    total_world_size: int,
    shards: Union[str, Sequence[InferenceShardSpec], Sequence[dict]],
    use_tp_pp_dp_mapping: bool = False,
) -> List[InferenceShard]:
    """Build one ProcessGroupCollection per heterogeneous inference shard.

    Partitions global ranks into contiguous non-overlapping windows, one per
    shard. Shard ``i`` owns ranks
    ``[offset_i, offset_i + tp_i*pp_i*dp_i)``.

    Every rank must call this function so the collective ``dist.new_group``
    calls inside :func:`build_inference_pg_collection` succeed for every shard.
    The returned ``pg_collection`` is populated only on ranks belonging to
    that shard; others see ``None``.

    Args:
        total_world_size: Full world size across training + all inference
            shards.
        shards: Shard layout in any form ``normalize_shard_specs`` accepts -- a
            spec string, a list of :class:`InferenceShardSpec`, or a list of
            raw dicts. Normalized internally to validated specs.
        use_tp_pp_dp_mapping: Passed through to ``build_inference_pg_collection``.

    Returns:
        List of :class:`InferenceShard`, one per input spec.
    """
    specs = normalize_shard_specs(shards, total_world_size)
    rank = dist.get_rank()
    results: List[InferenceShard] = []
    offset = 0
    for i, spec in enumerate(specs):
        tp, pp, ep, expt_tp, dp = spec.tp, spec.pp, spec.ep, spec.expt_tp, spec.dp
        shard_world = tp * pp * dp
        assert offset + shard_world <= total_world_size, (
            f"Shard {i} ({spec}) runs out of ranks: needs "
            f"[{offset}, {offset + shard_world}), total_world_size={total_world_size}."
        )
        pgc = build_inference_pg_collection(
            world_size=shard_world,
            tp_size=tp,
            pp_size=pp,
            # Inference shards don't context-parallelize; the spec validates cp == 1.
            cp_size=spec.cp,
            ep_size=ep,
            expt_tp_size=expt_tp,
            use_tp_pp_dp_mapping=use_tp_pp_dp_mapping,
            rank_offset=offset,
        )
        in_shard = offset <= rank < offset + shard_world
        results.append(
            InferenceShard(
                spec=spec,
                rank_offset=offset,
                world_size=shard_world,
                pg_collection=pgc if in_shard else None,
            )
        )
        offset += shard_world
    return results
