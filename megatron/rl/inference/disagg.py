# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Disaggregated prefill/decode rollouts for RL."""

import copy

import torch.distributed as dist

from megatron.core.inference.shards import build_inference_pg_collection
from megatron.core.inference.shards_spec import (
    parse_inference_shards_spec,
    spec_declares_disaggregation,
)


def is_disagg_rollout(args) -> bool:
    """Whether RL rollouts should run through a prefill/decode split."""
    return spec_declares_disaggregation(args.inference_shards)


def disagg_refit_pools(inference_shards, world_size: int, rank: int | None = None) -> tuple[int, int]:
    """Return the refit pool count and this rank's pool index."""
    if rank is None:
        rank = dist.get_rank()
    if not spec_declares_disaggregation(inference_shards):
        return 1, 0
    specs = parse_inference_shards_spec(inference_shards, world_size)
    offset = 0
    for index, spec in enumerate(specs):
        if offset <= rank < offset + spec.world_size:
            return len(specs), index
        offset += spec.world_size
    raise RuntimeError(f"rank {rank} not in any disaggregated shard")


def _iter_shard_windows(specs, rank):
    """Yield each shard's rank offset and local-membership flag."""
    offset = 0
    for spec in specs:
        yield offset, spec, (offset <= rank < offset + spec.world_size)
        offset += spec.world_size


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
    rank = dist.get_rank()

    my_pg = None
    my_spec = None
    specs = parse_inference_shards_spec(args.inference_shards, args.world_size)
    for offset, spec, is_local in _iter_shard_windows(specs, rank):
        pg = build_inference_pg_collection(
            world_size=spec.world_size,
            tp_size=spec.tp,
            pp_size=spec.pp,
            cp_size=1,
            ep_size=spec.ep,
            expt_tp_size=spec.expt_tp,
            rank_offset=offset,
            use_tp_pp_dp_mapping=args.use_tp_pp_dp_mapping,
        )
        if is_local:
            my_pg, my_spec = pg, spec
    assert my_pg is not None and my_spec is not None, f"rank {rank} not in any disagg shard window"

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
