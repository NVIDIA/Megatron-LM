# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Disaggregated prefill/decode rollouts for RL."""

import copy

import torch.distributed as dist

from megatron.core.inference.shards import build_inference_pg_collections_for_shards
from megatron.core.inference.shards_spec import (
    resolve_inference_shard,
    spec_declares_disaggregation,
)


def is_disagg_rollout(args) -> bool:
    """Whether RL rollouts should run through a prefill/decode split."""
    return spec_declares_disaggregation(args.inference_shards)


def disagg_refit_pools(
    inference_shards: str | None, world_size: int, rank: int | None = None
) -> tuple[int, int]:
    """Return the refit pool count and this rank's pool index."""
    if rank is None:
        rank = dist.get_rank()
    if not spec_declares_disaggregation(inference_shards):
        return 1, 0
    assignment = resolve_inference_shard(inference_shards, world_size, rank)
    return assignment.shard_count, assignment.index


def build_disagg_inference_model(
    args,
    model_type,
    transformer_config,
    *,
    cfg_container,
    model_alloc_ctx,
):
    """Build this rank's disaggregated RL inference shard."""
    if not args.inference_dynamic_batching_enable_prefix_caching:
        raise ValueError("disaggregated RL rollouts require prefix caching")
    shards = build_inference_pg_collections_for_shards(
        total_world_size=args.world_size,
        shards=args.inference_shards,
        use_tp_pp_dp_mapping=args.use_tp_pp_dp_mapping,
    )
    local_shards = [shard for shard in shards if shard.pg_collection is not None]
    if len(local_shards) != 1:
        raise RuntimeError(
            f"rank {dist.get_rank()} belongs to {len(local_shards)} inference shards; expected one"
        )
    my_shard = local_shards[0]
    my_pg = my_shard.pg_collection
    my_spec = my_shard.spec

    # RL inference shards use CP=1.
    cfg = copy.deepcopy(transformer_config)
    cfg.tensor_model_parallel_size = my_spec.tp
    cfg.pipeline_model_parallel_size = my_spec.pp
    cfg.context_parallel_size = 1
    cfg.expert_model_parallel_size = my_spec.ep
    if my_spec.expt_tp is not None:
        cfg.expert_tensor_parallel_size = my_spec.expt_tp

    with model_alloc_ctx:
        model_config = copy.deepcopy(cfg_container.model)
        model_config.transformer = cfg
        model = model_config.get_builder_cls()(model_config).build_distributed_models(
            pg_collection=my_pg, wrap_with_ddp=False, model_type=model_type
        )
    model[0].eval()
    return model
