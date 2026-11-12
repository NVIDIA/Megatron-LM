# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pluggable arms for the MoE-layer comparison (``compare_arms.py``).

An arm is one way of executing the same logical MoE layer: a token dispatcher, a load
balancer, or a MoEScheduler planner. Every arm starts from the same shared configuration
(``base_config``), so identical geometry, seeds, precision and expert implementation are
guaranteed by construction; an arm only adds the ``TransformerConfig`` overrides that select
its method. An arm that needs something this checkout or node does not provide is reported
as unavailable with the reason, and is never silently replaced by another arm.

This module is pure Python and imports nothing from Megatron at import time, so the registry,
the configuration rules and the availability logic are unit-tested on CPU.
"""

from __future__ import annotations

import dataclasses
import importlib.util
from dataclasses import dataclass
from typing import Any, Mapping

# Requirement tokens. ``probe_requirements`` evaluates them on the current node.
HYBRIDEP = "hybridep"
HYBRIDEP_DENSE_ROUTING = "hybridep_dense_routing"
MOE_SCHEDULER = "moe_scheduler"
TE_OP_FUSER = "te_op_fuser"
ULTRAEP = "ultraep"

FAMILIES = ("dispatcher", "virtual_experts", "moe_scheduler")
SCHEDULER_TRANSPORTS = ("replica_nccl", "replica_peer_tma")


@dataclass(frozen=True)
class Geometry:
    """One MoE layer's shape, shared by every arm of a run."""

    ep: int = 2
    experts: int = 8
    topk: int = 2
    tokens: int = 512
    hidden: int = 128
    ffn: int = 256

    def validate(self) -> None:
        """Reject shapes that some arm cannot run, so every arm runs the same shape."""
        if not 2 <= self.ep <= 64 or not self.ep <= self.experts <= 8192:
            raise ValueError("requires 2 <= EP <= experts <= 8192, EP <= 64")
        if self.experts % self.ep or not 1 <= self.topk <= min(32, self.experts):
            raise ValueError("experts must divide EP evenly and 1 <= topk <= min(32, experts)")
        if (
            self.tokens <= 0
            or self.hidden <= 0
            or self.hidden % 128
            or self.ffn <= 0
            or self.ffn % 128
        ):
            raise ValueError("positive token count and hidden/FFN sizes divisible by 128 required")


@dataclass(frozen=True)
class Arm:
    """One execution method: config overrides on top of ``base_config`` and what it needs."""

    name: str
    family: str
    overrides: Mapping[str, Any]
    requires: tuple[str, ...] = ()
    source: str = ""
    # MoEScheduler replica slots across the EP group: "per_rank" (one per EP rank) or
    # "experts" (one per logical expert, which the moon_ep planner requires).
    idle_experts: str | None = None

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unknown arm family {self.family!r}")
        if (self.family == "moe_scheduler") != (self.idle_experts is not None):
            raise ValueError("idle_experts is required for, and only for, MoEScheduler arms")
        if self.idle_experts not in (None, "per_rank", "experts"):
            raise ValueError(f"unknown idle_experts rule {self.idle_experts!r}")


_HYBRIDEP_DISPATCH = {
    "moe_token_dispatcher_type": "flex",
    "moe_flex_dispatcher_backend": "hybridep",
}


def _scheduler(planner: str, idle: str, extra: tuple[str, ...] = ()) -> Arm:
    return Arm(
        name=f"scheduler_{planner}",
        family="moe_scheduler",
        overrides={
            **_HYBRIDEP_DISPATCH,
            "moe_enable_scheduler": True,
            "moe_scheduler_planner_type": planner,
        },
        requires=(HYBRIDEP, HYBRIDEP_DENSE_ROUTING, MOE_SCHEDULER, TE_OP_FUSER) + extra,
        source="MoEScheduler (NVIDIA/Megatron-LM#8500, unmerged)",
        idle_experts=idle,
    )


ARMS: dict[str, Arm] = {}


def register(arm: Arm) -> Arm:
    """Add an arm. Names are unique; registering a different arm under a used name fails."""
    if arm.name in ARMS and ARMS[arm.name] != arm:
        raise ValueError(f"arm {arm.name!r} is already registered differently")
    ARMS[arm.name] = arm
    return arm


for _arm in (
    Arm(
        name="alltoall",
        family="dispatcher",
        overrides={"moe_token_dispatcher_type": "alltoall"},
        source="MoEAlltoAllTokenDispatcher",
    ),
    Arm(
        name="hybridep",
        family="dispatcher",
        overrides=dict(_HYBRIDEP_DISPATCH),
        requires=(HYBRIDEP,),
        source="flex dispatcher, HybridEP backend",
    ),
    Arm(
        name="moonep",
        family="virtual_experts",
        overrides={**_HYBRIDEP_DISPATCH, "moe_virtual_expert_load_balance": True},
        requires=(HYBRIDEP, HYBRIDEP_DENSE_ROUTING),
        source="VirtualExpertLoadBalancer over HybridEP (MoonEP, NVIDIA/Megatron-LM#7450)",
    ),
    _scheduler("echo", "per_rank"),
    _scheduler("eplb", "per_rank"),
    _scheduler("moon_ep", "experts"),
    _scheduler("ultra_ep", "per_rank", (ULTRAEP,)),
):
    register(_arm)

DEFAULT_ARMS = ("alltoall", "hybridep", "moonep")


def base_config(geometry: Geometry, *, te_op_fuser: bool = False) -> dict[str, Any]:
    """``TransformerConfig`` keyword arguments shared by every arm (BF16, TP=ETP=1, TE grouped
    experts, FP32 router, no aux loss, routing replay on)."""
    return {
        "num_layers": 1,
        "hidden_size": geometry.hidden,
        "ffn_hidden_size": geometry.ffn,
        "moe_ffn_hidden_size": geometry.ffn,
        "num_attention_heads": 4,
        "num_moe_experts": geometry.experts,
        "expert_model_parallel_size": geometry.ep,
        "expert_tensor_parallel_size": 1,
        "moe_router_topk": geometry.topk,
        "moe_grouped_gemm": True,
        "moe_use_grouped_tensor": True,
        "use_transformer_engine_op_fuser": te_op_fuser,
        "gradient_accumulation_fusion": True,
        "moe_router_dtype": "fp32",
        "moe_router_load_balancing_type": "none",
        "moe_aux_loss_coeff": 0.0,
        "moe_enable_routing_replay": True,
        "moe_router_fusion": False,
        "add_bias_linear": False,
        "gated_linear_unit": True,
        "bf16": True,
    }


def scheduler_idle_experts(arm: Arm, geometry: Geometry) -> int:
    """Replica slots across the EP group for a MoEScheduler arm."""
    if arm.idle_experts == "experts":
        return geometry.experts
    return geometry.ep


def config_for(
    arm: Arm,
    geometry: Geometry,
    *,
    te_op_fuser: bool = False,
    scheduler_transport: str = "replica_nccl",
) -> dict[str, Any]:
    """Full ``TransformerConfig`` keyword arguments for ``arm`` (without the activation and
    dtype objects, which the adapter adds)."""
    if scheduler_transport not in SCHEDULER_TRANSPORTS:
        raise ValueError(f"unknown MoEScheduler transport {scheduler_transport!r}")
    shared = base_config(geometry, te_op_fuser=te_op_fuser)
    clash = sorted(set(arm.overrides) & set(shared))
    if clash:
        # A fair comparison needs every arm to keep the shared settings.
        raise ValueError(f"arm {arm.name!r} overrides shared settings {clash}")
    config = {**shared, **arm.overrides}
    if arm.family == "moe_scheduler":
        config["moe_scheduler_expert_dispatcher_type"] = scheduler_transport
        config["moe_scheduler_num_idle_experts"] = scheduler_idle_experts(arm, geometry)
    return config


def resolve(names: str | list[str] | tuple[str, ...]) -> list[Arm]:
    """Arms named in a comma-separated list (or sequence), in order, without duplicates."""
    if isinstance(names, str):
        names = [n.strip() for n in names.split(",") if n.strip()]
    if not names:
        raise ValueError("no arms requested")
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate arm in {list(names)}")
    unknown = [n for n in names if n not in ARMS]
    if unknown:
        raise ValueError(f"unknown arm(s) {unknown}; known: {sorted(ARMS)}")
    return [ARMS[n] for n in names]


def run_order(arms: list[Arm], index: int, order: str = "rotate") -> list[Arm]:
    """Arm order for the ``index``-th workload: rotated so no arm always runs first."""
    if order == "fixed":
        return list(arms)
    if order != "rotate":
        raise ValueError(f"unknown order {order!r}")
    shift = index % len(arms)
    return list(arms[shift:]) + list(arms[:shift])


def _transformer_config_fields() -> set[str]:
    from megatron.core.transformer.transformer_config import TransformerConfig

    return {f.name for f in dataclasses.fields(TransformerConfig)}


def probe_requirements(*, te_op_fuser: bool) -> dict[str, tuple[bool, str]]:
    """Evaluate every requirement token on this node and checkout: ``{token: (ok, reason)}``."""
    probes: dict[str, tuple[bool, str]] = {}
    try:
        from megatron.core.transformer.moe import fused_a2a

        have, dense = fused_a2a.HAVE_HYBRIDEP, fused_a2a.HAVE_HYBRIDEP_DENSE_ROUTING
    except Exception as exc:  # pragma: no cover - depends on the environment
        have = dense = False
        probes[HYBRIDEP] = (False, f"importing fused_a2a failed: {exc}")
    probes.setdefault(HYBRIDEP, (have, "" if have else "deep_ep.HybridEPBuffer is not importable"))
    probes[HYBRIDEP_DENSE_ROUTING] = (
        dense,
        "" if dense else "the installed HybridEP lacks dense top-k routing (DeepEP@10d4dd7)",
    )
    try:
        fields = _transformer_config_fields()
        scheduler = "moe_enable_scheduler" in fields
    except Exception as exc:  # pragma: no cover - depends on the environment
        scheduler, fields = False, set()
        probes[MOE_SCHEDULER] = (False, f"importing TransformerConfig failed: {exc}")
    probes.setdefault(
        MOE_SCHEDULER,
        (
            scheduler,
            "" if scheduler else "TransformerConfig has no moe_enable_scheduler (needs #8500)",
        ),
    )
    probes[TE_OP_FUSER] = (
        te_op_fuser,
        (
            ""
            if te_op_fuser
            else "MoEScheduler requires use_transformer_engine_op_fuser; drop "
            "--no-te-op-fuser to run every arm with it"
        ),
    )
    ultra = importlib.util.find_spec("ultra_ep") is not None
    probes[ULTRAEP] = (ultra, "" if ultra else "the UltraEP package (ultra_ep) is not installed")
    return probes


def availability(arm: Arm, probes: Mapping[str, tuple[bool, str]]) -> tuple[bool, list[str]]:
    """Whether ``arm`` can run given ``probes``, and every unmet requirement's reason."""
    missing = []
    for token in arm.requires:
        ok, reason = probes.get(token, (False, f"requirement {token!r} was not probed"))
        if not ok:
            missing.append(f"{token}: {reason}")
    return not missing, missing


def availability_table(
    arms: list[Arm], probes: Mapping[str, tuple[bool, str]]
) -> list[dict[str, Any]]:
    """One record per arm, for the run log."""
    rows = []
    for arm in arms:
        ok, reasons = availability(arm, probes)
        rows.append(
            {
                "arm": arm.name,
                "family": arm.family,
                "source": arm.source,
                "available": ok,
                "reasons": reasons,
            }
        )
    return rows
