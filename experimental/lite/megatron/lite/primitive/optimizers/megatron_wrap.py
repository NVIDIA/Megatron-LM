# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Megatron-Core optimizer wrap backend for Megatron Lite."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch  # pyright: ignore[reportMissingImports]
import torch.nn as nn  # pyright: ignore[reportMissingImports]

from megatron.lite.primitive.protocols import ExpertClassifierFn, default_expert_classifier


def validate_mc_config(engine_cfg) -> None:
    """Validate dist_opt constraints owned by this optimizer primitive."""
    p = engine_cfg.parallel
    if p.vpp > 1 and p.pp == 1:
        raise ValueError("dist_opt requires pp>1 when vpp>1.")


# Legacy alias — kept for compat shim path
validate_mc_session = validate_mc_config


def _effective_etp(parallel) -> int:
    return int(parallel.etp if parallel.etp is not None else 1)


def _ensure_mc_mpu_parallel_state(engine_cfg) -> None:
    """Initialize Megatron-Core mpu globals when MC fallback groups are used."""

    from megatron.core import parallel_state as mpu  # pyright: ignore[reportMissingImports]

    p = engine_cfg.parallel
    expected = (int(p.tp), int(p.ep), _effective_etp(p), int(p.pp), int(p.cp))
    if mpu.is_initialized():
        current = (
            int(mpu.get_tensor_model_parallel_world_size()),
            int(mpu.get_expert_model_parallel_world_size()),
            int(mpu.get_expert_tensor_parallel_world_size() or 1),
            int(mpu.get_pipeline_model_parallel_world_size()),
            int(mpu.get_context_parallel_world_size()),
        )
        if current != expected:
            raise RuntimeError(
                "dist_opt found an incompatible existing Megatron-Core parallel state: "
                f"current={current}, expected={expected}."
            )
        return

    mpu.initialize_model_parallel(
        tensor_model_parallel_size=p.tp,
        pipeline_model_parallel_size=p.pp,
        virtual_pipeline_model_parallel_size=None if int(p.vpp or 1) <= 1 else p.vpp,
        context_parallel_size=p.cp,
        expert_model_parallel_size=p.ep,
        expert_tensor_parallel_size=_effective_etp(p),
        create_gloo_process_groups=bool(getattr(engine_cfg, "deterministic", False)),
    )


def build_mc_optimizer_config(opt, *, override_optimizer_config: dict[str, Any] | None = None):
    """Build MC OptimizerConfig from user's OptimizerConfig (duck-typed).

    Single source of truth for Megatron Lite's Megatron-Core optimizer stack.

    Works on either `runtime.contracts.config.OptimizerConfig` (real dataclass)
    or a `SimpleNamespace` with the same field names (legacy lite path).
    """
    from megatron.core.optimizer.optimizer_config import (
        OptimizerConfig as MCOptimizerConfig,  # pyright: ignore[reportMissingImports]
    )

    offload = getattr(opt, "offload_fraction", None) or 0.0
    args: dict[str, Any] = {
        "optimizer": opt.optimizer,
        "lr": opt.lr,
        "min_lr": getattr(opt, "min_lr", 0.0),
        "weight_decay": opt.weight_decay,
        "clip_grad": opt.clip_grad,
        "use_distributed_optimizer": True,
        "bf16": True,
        "params_dtype": torch.bfloat16,
    }
    if offload > 0:
        args["optimizer_offload_fraction"] = offload
        args["overlap_cpu_optimizer_d2h_h2d"] = True
        args["optimizer_cpu_offload"] = True
    if getattr(opt, "adam_beta1", None) is not None:
        args["adam_beta1"] = opt.adam_beta1
    if getattr(opt, "adam_beta2", None) is not None:
        args["adam_beta2"] = opt.adam_beta2
    if getattr(opt, "adam_eps", None) is not None:
        args["adam_eps"] = opt.adam_eps
    if getattr(opt, "use_precision_aware_optimizer", None) is not None:
        args["use_precision_aware_optimizer"] = opt.use_precision_aware_optimizer
    if getattr(opt, "decoupled_weight_decay", None) is not None:
        args["decoupled_weight_decay"] = opt.decoupled_weight_decay
    if override_optimizer_config:
        args.update(override_optimizer_config)
    return MCOptimizerConfig(**args)


def build_mc_stack(
    model_chunks: list[nn.Module],
    *,
    model_cfg,
    engine_cfg,
    ps,
    is_expert: ExpertClassifierFn | None = None,
    proto=None,
    skip_ddp_wrap: bool = False,
):
    """Wrap ML model chunks with MC DDP and build the matching MC optimizer.

    Args:
        skip_ddp_wrap: when True, ``model_chunks`` are assumed to already be
            MC ``DistributedDataParallel``-wrapped; we skip our own wrapping
            and feed them directly to the optimizer. The bucket layout
            influences optimizer master-grad sharding, so callers that prewrap
            chunks own the DDP config compatibility.
    """
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads
    from megatron.core.optimizer import get_megatron_optimizer
    from megatron.core.transformer.enums import ModelType

    validate_mc_config(engine_cfg)

    p = engine_cfg.parallel
    opt = engine_cfg.optimizer

    mc_transformer_cfg = _build_transformer_config(model_cfg, engine_cfg)
    mc_transformer_cfg.finalize_model_grads_func = finalize_model_grads
    if is_expert is not None:
        is_expert_param = is_expert
    elif proto is not None and hasattr(proto, "EXPERT_CLASSIFIER"):
        is_expert_param = proto.EXPERT_CLASSIFIER
    else:
        is_expert_param = default_expert_classifier
    use_mpu_groups = bool(getattr(engine_cfg, "deterministic", False))
    if use_mpu_groups:
        _ensure_mc_mpu_parallel_state(engine_cfg)
    pg_collection = None if use_mpu_groups else _build_pg_collection(ps, engine_cfg)

    if skip_ddp_wrap:
        # Caller already wrapped and marked every param. Our helper setting
        # `param.allreduce` on dense params could clash with MC code paths that
        # distinguish `hasattr(param,'allreduce')` from `getattr(..., True)`.
        wrapped_chunks = list(model_chunks)
    else:
        ddp_config = DistributedDataParallelConfig(
            use_distributed_optimizer=True, overlap_grad_reduce=False, grad_reduce_in_fp32=True
        )
        wrapped_chunks = []
        for chunk_idx, chunk in enumerate(model_chunks):
            chunk.model_type = ModelType.encoder_or_decoder
            _mark_mc_parallel_attrs(chunk, is_expert_param, tp_size=p.tp)
            ddp_kwargs = {}
            if pg_collection is not None:
                ddp_kwargs["pg_collection"] = pg_collection
            wrapped_chunks.append(
                DistributedDataParallel(
                    mc_transformer_cfg,
                    ddp_config,
                    chunk,
                    disable_bucketing=(chunk_idx > 0),
                    **ddp_kwargs,
                )
            )

    # Single-source-of-truth OptimizerConfig construction for native lite
    # model protocols.
    opt_config = build_mc_optimizer_config(opt)

    # This branch falls back to MC mpu globals for the optimizer's process
    # groups. Long term, this primitive should always pass its own
    # `pg_collection`.
    if skip_ddp_wrap or use_mpu_groups:
        optimizer = get_megatron_optimizer(config=opt_config, model_chunks=wrapped_chunks)
        optimizer._mc_pg_collection = None  # pyright: ignore[reportAttributeAccessIssue]
    else:
        optimizer = get_megatron_optimizer(
            config=opt_config,
            model_chunks=wrapped_chunks,
            use_gloo_process_groups=False,
            pg_collection=pg_collection,
        )
        optimizer._mc_pg_collection = pg_collection  # pyright: ignore[reportAttributeAccessIssue]
    return wrapped_chunks, optimizer


def build_mc_training_optimizer(
    model_chunks: list[nn.Module],
    *,
    model_cfg,
    impl_cfg,
    ps,
    model_name: str,
    is_expert: ExpertClassifierFn | None = None,
    skip_ddp_wrap: bool = False,
    deterministic: bool | None = None,
):
    """Build the MC DDP+optimizer stack from a Megatron Lite model ImplConfig."""

    opt = impl_cfg.optimizer_config
    if opt is None:
        opt = SimpleNamespace(
            optimizer="adam",
            lr=1e-4,
            weight_decay=0.01,
            clip_grad=1.0,
            offload_fraction=None,
            adam_beta1=None,
            adam_beta2=None,
            adam_eps=None,
        )
    if deterministic is None:
        from megatron.lite.primitive.deterministic import deterministic_requested

        deterministic = deterministic_requested()

    engine_cfg = SimpleNamespace(
        model_name=model_name,
        parallel=impl_cfg.parallel,
        optimizer=opt,
        deterministic=bool(deterministic),
    )
    model_chunks[:], optimizer = build_mc_stack(
        model_chunks,
        model_cfg=model_cfg,
        engine_cfg=engine_cfg,
        ps=ps,
        is_expert=is_expert,
        skip_ddp_wrap=skip_ddp_wrap,
    )

    def finalize_grads() -> None:
        finalize_mc_grads(model_chunks, optimizer)

    return optimizer, finalize_grads


def finalize_mc_grads(model_chunks: list[nn.Module], optimizer) -> None:
    """Run MC gradient finalization to match the optimizer's expected contract."""
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads

    finalize_model_grads(model_chunks, pg_collection=optimizer._mc_pg_collection)


def _build_transformer_config(model_cfg, engine_cfg):
    from megatron.core.transformer.transformer_config import TransformerConfig

    p = engine_cfg.parallel
    kwargs = dict(
        num_layers=max(getattr(model_cfg, "num_hidden_layers", 1), 1),
        hidden_size=max(getattr(model_cfg, "hidden_size", 1), 1),
        num_attention_heads=max(getattr(model_cfg, "num_attention_heads", 1), 1),
        num_query_groups=getattr(model_cfg, "num_key_value_heads", None),
        num_moe_experts=getattr(model_cfg, "num_experts", None),
        moe_ffn_hidden_size=getattr(model_cfg, "moe_intermediate_size", None),
        tensor_model_parallel_size=p.tp,
        pipeline_model_parallel_size=p.pp,
        context_parallel_size=p.cp,
        expert_model_parallel_size=p.ep,
        expert_tensor_parallel_size=p.etp if p.etp is not None else 1,
        sequence_parallel=p.tp > 1,
        bf16=True,
        params_dtype=torch.bfloat16,
    )
    if hasattr(model_cfg, "add_bias_linear"):
        kwargs["add_bias_linear"] = bool(model_cfg.add_bias_linear)
    elif kwargs["num_moe_experts"] is not None and kwargs["expert_tensor_parallel_size"] > 1:
        kwargs["add_bias_linear"] = False
    if p.pp > 1:
        kwargs["pipeline_dtype"] = torch.bfloat16
    return TransformerConfig(**kwargs)


def _mark_mc_parallel_attrs(
    model: nn.Module, is_expert_param: ExpertClassifierFn, *, tp_size: int
) -> None:
    """Mark per-param MC metadata (allreduce / tensor_model_parallel / sequence_parallel).

    IMPORTANT: respect attrs that are already set. Prewrapped MC models may
    mark these correctly per-param (e.g. `moe.router.weight` is 2D but
    TP-replicated, and must NOT have `tensor_model_parallel=True`). Blind
    override would cause MC grad-norm to over-count replicated params.
    """
    sp_param_ids = {id(param) for param in getattr(model, "sp_params", [])}
    for name, param in model.named_parameters():
        # MC uses `allreduce=False` to route expert params into expert-DP buffers.
        if not hasattr(param, "allreduce"):
            param.allreduce = not is_expert_param(name)
        if tp_size > 1 and id(param) not in sp_param_ids and param.ndim > 1:
            # vision params are replicated across TP (AVG all-reduce, not TP-split).
            # tensor_model_parallel=True would cause MC to wrong-account their grad-norm.
            if getattr(param, "average_gradients_across_tp_domain", False):
                continue
            # Skip params already marked sequence_parallel=True: they are TP-replicated
            # with SP-sharded input (e.g. shared_experts.gate_weight, RMSNorm weights).
            # Stacking tensor_model_parallel=True on top would cause double all-reduce.
            if getattr(param, "sequence_parallel", False):
                continue
            # MC excludes TP replicas from grad-norm accounting via this metadata.
            if not hasattr(param, "tensor_model_parallel"):
                param.tensor_model_parallel = True

    for param in getattr(model, "sp_params", []):
        if not hasattr(param, "sequence_parallel"):
            param.sequence_parallel = True
        param.allreduce = True
        param.tensor_model_parallel = False


def _build_pg_collection(ps, engine_cfg):
    import torch.distributed as dist  # pyright: ignore[reportMissingImports]

    from megatron.core.process_groups_config import ProcessGroupCollection

    if ps.pp_group is None:
        raise ValueError("dist_opt requires a local pp_group.")

    def _dense_rank(tp_i: int, cp_i: int, dp_i: int, pp_i: int) -> int:
        return ((pp_i * ps.dp_size + dp_i) * ps.cp_size + cp_i) * ps.tp_size + tp_i

    def _expert_rank(etp_i: int, ep_i: int, edp_i: int, pp_i: int) -> int:
        return ((pp_i * ps.expert_dp_size + edp_i) * ps.ep_size + ep_i) * ps.etp_size + etp_i

    rank = dist.get_rank()
    world = dist.get_world_size()

    singleton_group = None
    for singleton_rank in range(world):
        group = dist.new_group([singleton_rank])
        if rank == singleton_rank:
            singleton_group = group
    if singleton_group is None:
        raise RuntimeError(
            "Failed to construct singleton process group for optional MC reductions."
        )

    if engine_cfg.parallel.pp == 1:
        mp_group = ps.tp_group
        tp_ep_pp_group = ps.tp_ep_group
    else:
        mp_group = None
        for dp_idx in range(ps.dp_size):
            for cp_idx in range(ps.cp_size):
                ranks = [
                    _dense_rank(tp_idx, cp_idx, dp_idx, pp_idx)
                    for pp_idx in range(ps.pp_size)
                    for tp_idx in range(ps.tp_size)
                ]
                group = dist.new_group(ranks)
                if rank in ranks:
                    mp_group = group

        tp_ep_pp_group = None
        for expert_dp_idx in range(ps.expert_dp_size):
            ranks = [
                _expert_rank(etp_idx, ep_idx, expert_dp_idx, pp_idx)
                for pp_idx in range(ps.pp_size)
                for ep_idx in range(ps.ep_size)
                for etp_idx in range(ps.etp_size)
            ]
            group = dist.new_group(ranks)
            if rank in ranks:
                tp_ep_pp_group = group

        if mp_group is None or tp_ep_pp_group is None:
            raise RuntimeError("Failed to construct mc pipeline-aware process groups.")

    return ProcessGroupCollection(
        tp=ps.tp_group,
        cp=ps.cp_group,
        pp=ps.pp_group,
        ep=ps.ep_group,
        mp=mp_group,
        dp=ps.dp_group,
        dp_cp=ps.dp_cp_group,
        expt_dp=ps.ep_dp_group,
        expt_tp=ps.etp_group,
        tp_ep=ps.tp_ep_group,
        tp_ep_pp=tp_ep_pp_group,
        # For MC distributed optimizer, grad stats are reduced over the full optimizer instance.
        # With a single dist-opt instance in this benchmark proof, that is the global world group.
        intra_dist_opt=dist.group.WORLD,
        # ML models do not expose MC's embedding/position-embedding sharing surface.
        # Use singleton groups so MC's optional embedding reductions become no-ops
        # without falling back to the global MCore embedding group.
        embd=singleton_group,
        pos_embd=singleton_group,
    )


# ---------------------------------------------------------------------------
# Backend adapter (consumed by runtime/session.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MCBackend:
    name: str = "mc"
    runtime_backend: str = "mc"

    def zero_grad(self, optimizer: Any) -> None:
        optimizer.zero_grad()

    def finish_grad_sync(self, optimizer: Any) -> None:
        if hasattr(optimizer, "finish_grad_sync"):
            optimizer.finish_grad_sync()

    def clip_grad_norm(self, optimizer: Any):
        if hasattr(optimizer, "clip_grad_norm"):
            return optimizer.clip_grad_norm()
        return None

    def step(self, optimizer: Any):
        return optimizer.step()

    def state_dict(self, optimizer: Any) -> dict:
        return optimizer.state_dict()

    def load_state_dict(self, optimizer: Any, state_dict: dict) -> None:
        optimizer.load_state_dict(state_dict)

    def finalize_grads(self, finalize_fn, model_chunks: list[Any], optimizer: Any) -> None:
        finalize_fn(model_chunks, optimizer)


BACKEND = MCBackend()

__all__ = [
    "BACKEND",
    "MCBackend",
    "build_mc_stack",
    "build_mc_training_optimizer",
    "finalize_mc_grads",
    "validate_mc_config",
    "validate_mc_session",
]
