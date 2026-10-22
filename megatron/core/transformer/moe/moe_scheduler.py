# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Backend-neutral interfaces for MoE expert and token rerouting.

MoEScheduler is intended to run after router output is available and before the
normal MoE token dispatcher starts.  This module only defines the shared
contracts:

* planners own logical placement and lower it to physical copy/exchange requests;
* replica dispatch starts immediately, while home dispatch stages a deferred plan;
* rerouting always uses the currently committed placement;
* backend-specific lowering is kept inside the concrete expert dispatcher.

Concrete Echo, EPLB, UltraEP, and MoonEP planners should produce an opaque
``MoEPlacementResult`` between their placement and reroute phases.
Concrete expert dispatchers consume physical source-slot tables, never logical mappings.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch

from megatron.core.transformer.moe.replica_weight_transport import (
    REPLICA_EXPERT_DISPATCHER_TYPES,
    HomeExpertPlacement,
)

if TYPE_CHECKING:
    from megatron.core.optimizer.optimizer import MegatronOptimizer
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.moe.home_expert_dispatch import HomeExpertDispatch
    from megatron.core.transformer.transformer_config import TransformerConfig

logger = logging.getLogger(__name__)

ECHO_BACKEND = "echo"
EPLB_BACKEND = "eplb"
ULTRA_EP_BACKEND = "ultra_ep"
MOON_EP_BACKEND = "moon_ep"


def validate_scheduler_environment(
    config: TransformerConfig, pg_collection: ProcessGroupCollection
) -> None:
    """Check execution dependencies before allocating a layer's expert weights."""
    if not torch.cuda.is_available():
        raise RuntimeError("MoEScheduler requires CUDA.")
    group = getattr(pg_collection, "ep", None)
    if group is None or not torch.distributed.is_initialized():
        raise RuntimeError("MoEScheduler requires an initialized explicit EP process group.")
    if torch.distributed.get_backend(group) != "nccl":
        raise ValueError("MoEScheduler requires a NCCL EP process group.")
    if group.size() != config.expert_model_parallel_size:
        raise ValueError("MoEScheduler EP process group does not match expert_model_parallel_size.")
    from megatron.core.transformer.moe.fused_a2a import HAVE_HYBRIDEP
    from megatron.core.utils import is_te_min_version

    if not HAVE_HYBRIDEP:
        raise RuntimeError("MoEScheduler requires the HybridEP token-dispatch package.")
    try:
        import triton  # noqa: F401
        from transformer_engine.pytorch import ops
    except ImportError as exc:
        raise RuntimeError(
            "MoEScheduler requires Triton and Transformer Engine grouped ops."
        ) from exc
    if not is_te_min_version("2.14.0") or not all(
        hasattr(ops, name) for name in ("GroupedLinear", "ScaledSwiGLU")
    ):
        raise RuntimeError("MoEScheduler requires TE >= 2.14 with GroupedLinear/ScaledSwiGLU.")
    if (
        config.activation_func_clamp_value is not None
        or config.activation_func.__name__ == "quick_gelu"
    ):
        if not hasattr(ops, "ScaledClampedQGeGLU"):
            raise RuntimeError("MoEScheduler clamped/quick-GeGLU requires TE ScaledClampedQGeGLU.")
        if config.activation_func.__name__ == "silu" and not is_te_min_version("2.17.0.dev0"):
            raise RuntimeError("MoEScheduler clamped SwiGLU requires TE >= 2.17.0.dev0.")
    if config.activation_func.__name__ == "squared_relu" and not hasattr(ops, "ScaledSReLU"):
        raise RuntimeError("MoEScheduler weighted squared-ReLU requires TE ScaledSReLU.")
    if int(os.environ.get("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "0")) <= 0:
        raise RuntimeError("MoEScheduler requires NVTE_CUTEDSL_FUSED_GROUPED_MLP=1.")
    if config.moe_scheduler_planner_type == "moon_ep" and group.size() > 1:
        from megatron.core.transformer.moe.moonep_moe_scheduler import get_planner_workspace

        # Allocate/validate the actual histogram window now, not on the first
        # forward. This also checks native NCCL symmetric-memory availability.
        get_planner_workspace(
            device=torch.device("cuda", torch.cuda.current_device()),
            num_experts=config.num_moe_experts,
            group=group,
        )


def _rank0_info(message: str) -> None:
    try:
        is_rank0 = (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        )
    except RuntimeError:
        is_rank0 = True
    if is_rank0:
        logger.info("%s", message)


def _tensor_shape(tensor: torch.Tensor | None) -> tuple[int, ...] | None:
    return None if tensor is None else tuple(tensor.shape)


def _validate_2d_tensor(name: str, tensor: torch.Tensor) -> None:
    if tensor.dim() != 2:
        raise ValueError(f"Expected {name} to be 2D, got shape {tuple(tensor.shape)}")


@dataclass(frozen=True)
class SchedulerContext:
    """Static and per-forward context shared by planners and dispatchers."""

    layer_number: int | None
    num_logical_experts: int
    num_local_experts: int
    local_expert_indices: tuple[int, ...]
    ep_size: int
    ep_rank: int
    router_topk: int
    training: bool
    config: TransformerConfig | None = None
    pg_collection: ProcessGroupCollection | None = None

    def __post_init__(self) -> None:
        if self.num_logical_experts <= 0:
            raise ValueError("num_logical_experts must be positive.")
        if self.num_local_experts <= 0:
            raise ValueError("num_local_experts must be positive.")
        if self.ep_size <= 0:
            raise ValueError("ep_size must be positive.")
        if self.ep_rank < 0 or self.ep_rank >= self.ep_size:
            raise ValueError(f"ep_rank must be in [0, {self.ep_size}), got {self.ep_rank}.")
        if self.router_topk <= 0:
            raise ValueError("router_topk must be positive.")
        if len(self.local_expert_indices) != self.num_local_experts:
            raise ValueError(
                "Expected local_expert_indices to match num_local_experts, "
                f"got {len(self.local_expert_indices)} and {self.num_local_experts}"
            )


class MoEPlacementResult:
    """Opaque planner state passed from ``update_placement`` to ``reroute``.

    The scheduler does not inspect planner-private state. Concrete planners
    use subclasses to retain the allocation metadata needed to reroute tokens
    without recomputing placement or storing implicit per-forward state.

    ``max_dispatch_routes_per_rank`` is an optional public capacity guarantee:
    after rerouting, no rank receives more logical routes than this Python
    integer. It excludes per-expert transport padding. None makes no promise.
    ``max_dispatch_padded_tokens_per_rank`` optionally bounds the full aligned
    transport buffer instead; dispatch must not add expert padding a second time.
    """

    max_dispatch_routes_per_rank: int | None = None
    max_dispatch_padded_tokens_per_rank: int | None = None


def _validate_reroute_output(
    num_physical_experts: int, routing_map: torch.Tensor, probs: torch.Tensor
) -> None:
    _validate_2d_tensor("routing_map", routing_map)
    _validate_2d_tensor("probs", probs)
    if routing_map.dtype != torch.bool:
        raise ValueError(f"Expected bool routing_map, got {routing_map.dtype}")
    if probs.shape != routing_map.shape:
        raise ValueError(
            "Expected probs and routing_map to have the same shape, "
            f"got {tuple(probs.shape)} and {tuple(routing_map.shape)}"
        )
    if num_physical_experts != routing_map.size(1):
        raise ValueError(
            "Expected physical expert count to match routing_map expert dimension, "
            f"got {num_physical_experts} and {routing_map.size(1)}"
        )


class MoELoadPlanner(torch.nn.Module, ABC):
    """Base class for backend-neutral MoE load planners."""

    planner_name: ClassVar[str] = "abstract"

    @abstractmethod
    def update_placement(
        self,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        context: SchedulerContext,
        *,
        tokens_per_expert: torch.Tensor | None = None,
    ) -> tuple[HomeExpertPlacement | None, torch.Tensor | None, MoEPlacementResult]:
        """Return optional home exchange, replica copies, and private reroute state."""

    def step(self, completed_version: int | None = None) -> None:
        """Commit completed home placement after a successful optimizer update."""
        if completed_version is not None:
            raise RuntimeError("This planner has no pending home exchange.")

    def begin_logical_checkpoint_load(self) -> None:
        """Select identity-owned initialization, if supported by this planner."""
        raise NotImplementedError(f"{self.planner_name} does not support home checkpoints.")

    def finish_logical_checkpoint_load(self) -> None:
        """Finish identity-owned initialization after checkpoint loading."""
        raise NotImplementedError(f"{self.planner_name} does not support home checkpoints.")

    def logical_export_placement(self) -> HomeExpertPlacement | None:
        """Return physical sources restoring logical order, ignoring pending proposals."""
        raise NotImplementedError(f"{self.planner_name} does not support logical export.")

    @abstractmethod
    def reroute(
        self,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        placement_result: MoEPlacementResult,
        context: SchedulerContext,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return dense physical ``routing_map`` and ``probs`` tensors."""


class ExpertDispatch(torch.nn.Module, ABC):
    """Base class for ReplicaHybridEP and future placement materializers."""

    dispatcher_name: ClassVar[str] = "abstract"

    def supports(self, expert_placement: torch.Tensor, context: SchedulerContext) -> bool:
        """Return whether this dispatcher can materialize the given physical layout."""
        del expert_placement, context
        return True

    @abstractmethod
    def dispatch(
        self, experts: torch.nn.Module, expert_placement: torch.Tensor, context: SchedulerContext
    ) -> None:
        """Materialize the planned expert placement before token dispatch begins."""

    def finalize(self, context: SchedulerContext) -> None:
        """Release transient dispatch state after the MoE forward finishes."""
        del context

    def assert_idle(self) -> None:
        """Check that no outstanding dispatch can consume home state being migrated."""

    def bind_experts(self, experts: torch.nn.Module) -> None:
        """Bind expert parameters for dispatchers that own persistent runtime weights."""
        del experts

    def bind_home_dispatch(self, home: HomeExpertDispatch) -> None:
        """Provide a compatible transport after binding expert weights."""
        raise NotImplementedError(f"{self.dispatcher_name} does not support home migration.")

    def bind_home_optimizer(
        self, home: HomeExpertDispatch, optimizer: MegatronOptimizer | torch.optim.Optimizer
    ) -> None:
        """Provide authoritative state without exposing dispatcher runtime internals."""
        raise NotImplementedError(f"{self.dispatcher_name} does not support home migration.")

    def wrap_layer_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Attach lifecycle work that must run after all layer-input consumers."""
        return hidden_states

    def before_token_dispatch(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply a backend-specific autograd boundary before token dispatch."""
        return hidden_states

    def after_token_dispatch(self, dispatched_hidden: torch.Tensor) -> torch.Tensor:
        """Apply a backend-specific autograd boundary after token dispatch."""
        return dispatched_hidden

    def before_token_combine(self, expert_output: torch.Tensor) -> torch.Tensor:
        """Apply a backend-specific autograd boundary before token combine."""
        return expert_output

    def after_token_combine(self, combined_hidden: torch.Tensor) -> torch.Tensor:
        """Apply a backend-specific autograd boundary after token combine."""
        return combined_hidden


class MoEScheduler(torch.nn.Module):
    """Small orchestrator that connects a planner to an expert dispatcher."""

    _logged_config_signatures: ClassVar[set[tuple[str, str, int]]] = set()
    _logged_runtime_summary: ClassVar[bool] = False

    def __init__(
        self,
        planner: MoELoadPlanner,
        expert_dispatch: ExpertDispatch,
        home_expert_dispatch: HomeExpertDispatch | None = None,
    ) -> None:
        super().__init__()
        self.planner = planner
        self.expert_dispatch = expert_dispatch
        self.home_expert_dispatch = home_expert_dispatch
        self.max_dispatch_routes_per_rank: int | None = None
        self.max_dispatch_padded_tokens_per_rank: int | None = None

    @classmethod
    def from_config(
        cls, config: TransformerConfig, pg_collection: ProcessGroupCollection
    ) -> "MoEScheduler":
        """Build the configured MoEScheduler backend stack."""
        planner_type = getattr(config, "moe_scheduler_planner_type", None)
        expert_dispatcher_type = getattr(config, "moe_scheduler_expert_dispatcher_type", None)
        if expert_dispatcher_type == "replica_hybridep":
            raise ValueError(
                "replica_hybridep is not implemented; use replica_peer_tma or replica_nccl."
            )
        if planner_type not in ("echo", "eplb", "moon_ep"):
            raise ValueError(f"Unsupported MoEScheduler planner: {planner_type}")
        if expert_dispatcher_type not in REPLICA_EXPERT_DISPATCHER_TYPES:
            raise ValueError(
                f"Unsupported MoEScheduler expert dispatcher: {expert_dispatcher_type}"
            )

        num_idle_experts = getattr(config, "moe_scheduler_num_idle_experts", None)
        if num_idle_experts is None:
            raise ValueError(
                "moe_scheduler_num_idle_experts must be set when MoEScheduler is enabled."
            )

        if planner_type == "echo":
            from megatron.core.transformer.moe.echo_moe_scheduler import EchoLoadPlanner

            planner = EchoLoadPlanner(num_idle_experts)
        elif planner_type == "eplb":
            from megatron.core.transformer.moe.eplb_moe_scheduler import EPLBLoadPlanner

            planner = EPLBLoadPlanner(
                num_redundant_experts=num_idle_experts,
                home_update_interval=getattr(config, "moe_scheduler_home_update_interval", 0),
                num_logical_experts=config.num_moe_experts,
                pg_collection=pg_collection,
            )
        else:
            from megatron.core.transformer.moe.moe_utils import get_align_size_for_quantization
            from megatron.core.transformer.moe.moonep_moe_scheduler import MoonEPLoadPlanner

            ep_size = getattr(config, "expert_model_parallel_size", 1)

            planner = MoonEPLoadPlanner(
                num_redundant_experts=num_idle_experts // ep_size,
                token_padding=max(1, get_align_size_for_quantization(config)),
            )
        if num_idle_experts == 0:
            from megatron.core.transformer.moe.home_only_expert_dispatch import (
                HomeOnlyExpertDispatch,
            )

            expert_dispatch = HomeOnlyExpertDispatch(config=config, pg_collection=pg_collection)
        else:
            from megatron.core.transformer.moe.replica_expert_dispatch import ReplicaExpertDispatch

            expert_dispatch = ReplicaExpertDispatch(config=config, pg_collection=pg_collection)
        config_signature = (str(planner_type), str(expert_dispatcher_type), int(num_idle_experts))
        if config_signature not in cls._logged_config_signatures:
            cls._logged_config_signatures.add(config_signature)
            _rank0_info(
                "configured "
                f"planner={planner_type} "
                f"expert_dispatcher={expert_dispatcher_type} "
                f"num_idle_experts={num_idle_experts}"
            )
        home_dispatch = None
        if getattr(config, "moe_scheduler_home_update_interval", 0):
            from megatron.core.transformer.moe.home_expert_dispatch import HomeExpertDispatch

            home_dispatch = HomeExpertDispatch(
                buffer_size_bytes=getattr(config, "moe_scheduler_home_exchange_buffer_mb", 256)
                * 1024
                * 1024
            )
        return cls(
            planner=planner, expert_dispatch=expert_dispatch, home_expert_dispatch=home_dispatch
        )

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Delegate checkpoint ownership to concrete components."""
        from megatron.core.transformer.utils import sharded_state_dict_default

        self._check_checkpoint_boundary()
        state = {}
        for name, module in self.named_children():
            state.update(
                sharded_state_dict_default(module, prefix + name + ".", sharded_offsets, metadata)
            )
        return state

    def _check_checkpoint_boundary(self) -> None:
        if self.home_expert_dispatch is not None:
            self.expert_dispatch.assert_idle()
            self.home_expert_dispatch.assert_healthy()

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        self._check_checkpoint_boundary()
        super()._save_to_state_dict(destination, prefix, keep_vars)

    def bind_experts(self, experts: torch.nn.Module) -> None:
        """Bind the layer's native experts to the configured dispatch backend."""
        self.expert_dispatch.bind_experts(experts)
        if self.home_expert_dispatch is not None:
            self.expert_dispatch.bind_home_dispatch(self.home_expert_dispatch)

    def bind_optimizer(self, optimizer: MegatronOptimizer | torch.optim.Optimizer) -> None:
        """Bind authoritative state after DDP/FSDP and optimizer construction."""
        if self.home_expert_dispatch is not None:
            self.expert_dispatch.bind_home_optimizer(self.home_expert_dispatch, optimizer)

    @property
    def has_pending_home_exchange(self) -> bool:
        """Whether the next successful update needs a migration window."""
        return self.home_expert_dispatch is not None and self.home_expert_dispatch.has_pending

    def prepare_home_exchange(self) -> None:
        """Validate drained, bound home state before deferring optimizer gathers."""
        self.expert_dispatch.assert_idle()
        if self.home_expert_dispatch is not None:
            self.home_expert_dispatch.assert_ready()

    def begin_logical_checkpoint_load(self) -> None:
        """Select logical-weight initialization through the planner contract."""
        self.planner.begin_logical_checkpoint_load()

    def finish_logical_checkpoint_load(self) -> None:
        """Restore strict checkpoint requirements after logical initialization."""
        self.planner.finish_logical_checkpoint_load()

    def canonicalize_home_for_export(self) -> None:
        """Restore committed homes without executing pending training proposals."""
        self.prepare_home_exchange()
        if self.home_expert_dispatch is None:
            raise RuntimeError("Logical export requires home migration.")
        placement = self.planner.logical_export_placement()
        if placement is not None:
            self.home_expert_dispatch.export_placement(placement)

    def step(self) -> None:
        """Migrate updated home state, then publish mapping, at a drained step boundary."""
        version = None
        if self.home_expert_dispatch is not None:
            self.expert_dispatch.assert_idle()
            version = self.home_expert_dispatch.step()
        self.planner.step(version)

    def wrap_layer_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Attach dispatcher work before routing and shared-expert computation."""
        return self.expert_dispatch.wrap_layer_input(hidden_states)

    def _log_first_schedule(
        self,
        input_routing_map: torch.Tensor,
        num_physical_experts: int,
        output_routing_map: torch.Tensor,
        context: SchedulerContext,
        *,
        dispatch_materialized: bool,
    ) -> None:
        if MoEScheduler._logged_runtime_summary:
            return
        MoEScheduler._logged_runtime_summary = True
        expert_backend = self.expert_dispatch.dispatcher_name
        num_transfers = max(0, num_physical_experts - context.num_logical_experts)
        assignment_backend = self.planner.planner_name
        reroute_backend = "planner"
        _rank0_info(
            "first schedule completed "
            f"layer={context.layer_number} "
            f"training={context.training} "
            f"planner={self.planner.planner_name} "
            f"dispatcher={self.expert_dispatch.dispatcher_name} "
            f"input_routing_map_shape={_tensor_shape(input_routing_map)} "
            f"output_routing_map_shape={_tensor_shape(output_routing_map)} "
            f"expert_backend={expert_backend} "
            f"num_physical_experts={num_physical_experts} "
            f"num_transfers={num_transfers} "
            f"assignment_backend={assignment_backend} "
            f"reroute_backend={reroute_backend} "
            f"dispatch_materialized={dispatch_materialized}"
        )

    def schedule(
        self,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        experts: torch.nn.Module,
        context: SchedulerContext,
        *,
        tokens_per_expert: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Plan token/expert rerouting and materialize the expert side."""
        # Bounds belong to this invocation, never a previous placement/shape.
        self.max_dispatch_routes_per_rank = None
        self.max_dispatch_padded_tokens_per_rank = None
        home_expert_placement, replica_expert_placement, placement_result = (
            self.planner.update_placement(
                probs, routing_map, context, tokens_per_expert=tokens_per_expert
            )
        )
        if home_expert_placement is not None:
            if self.home_expert_dispatch is None:
                raise ValueError("Planner requested home exchange without HomeExpertDispatch.")
            self.home_expert_dispatch.dispatch(experts, home_expert_placement, context)
        if replica_expert_placement is not None:
            if (
                replica_expert_placement.ndim != 2
                or replica_expert_placement.shape[0] != context.ep_size
                or replica_expert_placement.dtype not in (torch.int32, torch.int64)
            ):
                raise ValueError("Expected integer replica placement [EP, S].")
            if not self.expert_dispatch.supports(replica_expert_placement, context):
                raise ValueError(
                    "Expert dispatcher does not support planner output replica source-slot table."
                )
            self.expert_dispatch.dispatch(experts, replica_expert_placement, context)
        rerouted_routing_map, rerouted_probs = self.planner.reroute(
            probs, routing_map, placement_result, context
        )
        num_physical_experts = context.num_logical_experts + (
            replica_expert_placement.numel() if replica_expert_placement is not None else 0
        )
        _validate_reroute_output(num_physical_experts, rerouted_routing_map, rerouted_probs)
        self.max_dispatch_routes_per_rank = placement_result.max_dispatch_routes_per_rank
        self.max_dispatch_padded_tokens_per_rank = (
            placement_result.max_dispatch_padded_tokens_per_rank
        )
        self._log_first_schedule(
            routing_map,
            num_physical_experts,
            rerouted_routing_map,
            context,
            dispatch_materialized=replica_expert_placement is not None,
        )
        return rerouted_probs, rerouted_routing_map

    def finalize(self, context: SchedulerContext) -> None:
        """Finalize the expert-dispatch portion of a scheduled forward."""
        self.expert_dispatch.finalize(context)

    def before_token_dispatch(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the configured dispatcher's pre-dispatch autograd boundary."""
        return self.expert_dispatch.before_token_dispatch(hidden_states)

    def after_token_dispatch(self, dispatched_hidden: torch.Tensor) -> torch.Tensor:
        """Run the configured dispatcher's post-dispatch autograd boundary."""
        return self.expert_dispatch.after_token_dispatch(dispatched_hidden)

    def before_token_combine(self, expert_output: torch.Tensor) -> torch.Tensor:
        """Run the configured dispatcher's pre-combine autograd boundary."""
        return self.expert_dispatch.before_token_combine(expert_output)

    def after_token_combine(self, combined_hidden: torch.Tensor) -> torch.Tensor:
        """Run the configured dispatcher's post-combine autograd boundary."""
        return self.expert_dispatch.after_token_combine(combined_hidden)
