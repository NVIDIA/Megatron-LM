# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Disaggregated (prefill/decode) rollouts for RL.

Hooks the coordinator-native prefill/decode split into RL's existing separate-
inference-model lifecycle: build this rank's shard model on its shard groups
(``build_disagg_inference_model``) and configure the shared coordinator on the
engine that wraps it (``configure_disagg_engine``). Refit into the shard pools
goes through the core ``swap_model_weights`` (one pass per pool, driven by
``disagg_refit_pools``). All gated on ``--inference-shards`` declaring ``role=``
tags; off otherwise."""

import copy
from contextlib import nullcontext

import torch.distributed as dist

from megatron.core.inference.disaggregation.coordinator_setup import (
    configure_prebuilt_disagg_engine,
)
from megatron.core.inference.shards_spec import (
    parse_inference_shards_spec,
    spec_declares_disaggregation,
)
from megatron.core.inference.shards import build_inference_pg_collection


def is_disagg_rollout(args) -> bool:
    """Whether RL rollouts should run through a prefill/decode split."""
    spec = getattr(args, "inference_shards", None)
    return bool(spec) and spec_declares_disaggregation(spec)


def _specs(args):
    return parse_inference_shards_spec(args.inference_shards, args.world_size)


def _iter_shard_windows(specs, rank):
    """Yield ``(offset, spec, is_mine)`` for each contiguous shard window, in
    order -- the same windowing ``configure_prebuilt_disagg_engine`` uses."""
    offset = 0
    for s in specs:
        yield offset, s, (offset <= rank < offset + s.world_size)
        offset += s.world_size


def build_disagg_inference_model(
    args,
    model_provider,
    model_type,
    base_config,
    get_model,
    *,
    cfg_container=None,
    model_alloc_ctx=None,
):
    """Build this rank's prefill/decode shard inference model (or ``None`` if
    ``--inference-shards`` doesn't declare disaggregation).

    Every rank builds *every* shard's process groups -- ``new_group`` is
    collective, so non-members must call it too -- but instantiates the model
    only on its own shard's groups, at that shard's parallelism. The result is
    the refit target (``swap_model_weights`` driven by ``disagg_refit_pools``)
    and what the engine wraps (see :func:`configure_disagg_engine`); placing it
    on the shard groups is the only difference from RL's existing separate-
    inference-model build.

    ``model_alloc_ctx`` is the weight-allocation context (UVM / saver region for
    idle-offload); the caller supplies the same one the colocated path uses so
    ``--rl-offload-inference-model-weights-when-idle`` works here too.
    """
    if not is_disagg_rollout(args):
        return None
    assert args.inference_dynamic_batching_enable_prefix_caching, (
        "disaggregated RL rollouts (--inference-shards with role=) require "
        "--inference-dynamic-batching-enable-prefix-caching: the decode side "
        "admits the handed-off KV via a prefix-cache hit."
    )
    rank = dist.get_rank()

    my_pg = None
    my_spec = None
    for offset, s, mine in _iter_shard_windows(_specs(args), rank):
        pg = build_inference_pg_collection(
            world_size=s.world_size,
            tp_size=s.tp,
            pp_size=s.pp,
            cp_size=1,
            ep_size=s.ep,
            expt_tp_size=s.expt_tp,
            rank_offset=offset,
            use_tp_pp_dp_mapping=args.use_tp_pp_dp_mapping,
        )
        if mine:
            my_pg, my_spec = pg, s
    assert my_pg is not None, f"rank {rank} not in any disagg shard window"

    # Shard parallelism. CP is forced to 1: spec.world_size = tp*pp*dp carries no
    # CP factor (the pg is built cp_size=1), so the config must agree.
    cfg = copy.deepcopy(base_config)
    cfg.tensor_model_parallel_size = my_spec.tp
    cfg.pipeline_model_parallel_size = my_spec.pp
    cfg.context_parallel_size = 1
    cfg.expert_model_parallel_size = my_spec.ep
    if my_spec.expt_tp is not None:
        cfg.expert_tensor_parallel_size = my_spec.expt_tp

    with model_alloc_ctx or nullcontext():
        if cfg_container is not None:
            model_config = copy.deepcopy(cfg_container.model)
            model_config.transformer = cfg
            model = model_config.get_builder_cls()(model_config).build_distributed_models(
                pg_collection=my_pg, wrap_with_ddp=False, model_type=model_type
            )
        else:
            model = get_model(
                model_provider, model_type, wrap_with_ddp=False, pg_collection=my_pg, config=cfg
            )
    model[0].eval()
    return model


def configure_disagg_engine(engine, *, disagg_router="round_robin"):
    """Set the disagg role on `engine` and spawn the shared coordinator."""
    from megatron.training.global_vars import get_args

    args = get_args()
    configure_prebuilt_disagg_engine(
        engine, _specs(args), disagg_router=disagg_router,
        kv_transport_backend=args.disagg_kv_transport_backend,
    )
