# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Native expert execution with deferred NCCL home migration and no replicas."""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from megatron.core.transformer.moe.home_expert_state import OptimizerHomeExpertState
from megatron.core.transformer.moe.moe_scheduler import ExpertDispatch, SchedulerContext
from megatron.core.transformer.moe.replica_expert_runtime import _collect_replica_projection_specs
from megatron.core.transformer.moe.replica_nccl_transport import NcclP2PTransport
from megatron.core.transformer.moe.replica_weight_transport import (
    ReplicaTransportConfig,
    register_replica_transport_finalizer,
)

if TYPE_CHECKING:
    from megatron.core.optimizer.optimizer import MegatronOptimizer
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.moe.home_expert_dispatch import HomeExpertDispatch
    from megatron.core.transformer.transformer_config import TransformerConfig


class _NativeBackwardLifetime(torch.autograd.Function):
    """Keep ownership fixed until the entire native backward graph is drained."""

    @staticmethod
    def forward(ctx, output, dispatcher, token):
        """Retain the pending native backward token without changing the activation."""
        ctx.dispatcher, ctx.token = dispatcher, token
        return output

    @staticmethod
    def backward(ctx, grad):
        # A combine-output hook runs before expert backward. Release only at
        # graph completion, after native expert and router/DP hooks have run.
        """Release the token after all native parameter and router hooks complete."""
        torch.autograd.Variable._execution_engine.queue_callback(
            lambda: ctx.dispatcher._pending_backwards.remove(ctx.token)
        )
        return grad, None, None


_home_only_dispatchers = weakref.WeakSet()


def _finalize_home_only_dispatchers() -> None:
    for dispatcher in list(_home_only_dispatchers):
        dispatcher.destroy()


class HomeOnlyExpertDispatch(ExpertDispatch):
    """Bind canonical optimizer state without replacing native expert parameters.

    The transport owns no replica weights, gradients, or native wgrad staging.
    Existing training-step glue fences native work, migrates optimizer state,
    refreshes model weights, and only then permits the next forward.
    """

    dispatcher_name = "home_nccl"

    def __init__(self, *, config: TransformerConfig, pg_collection: ProcessGroupCollection) -> None:
        super().__init__()
        if not (
            config.moe_scheduler_num_idle_experts == 0
            and config.moe_scheduler_planner_type == "eplb"
            and config.moe_scheduler_expert_dispatcher_type == "replica_nccl"
            and config.moe_scheduler_home_update_interval > 0
        ):
            raise ValueError(
                "Home-only execution requires EPLB, NCCL, zero replicas "
                "and a positive home interval."
            )
        self.config = config
        self.group = pg_collection.ep
        self.transport: NcclP2PTransport | None = None
        self.source_parameters: tuple[torch.nn.Parameter, ...] = ()
        self._pending_backwards: set[object] = set()

    def bind_experts(self, experts: torch.nn.Module) -> None:
        """Borrow native FC1/FC2 parameters; never install a replica runtime."""
        if self.transport is not None:
            raise RuntimeError("Home-only experts were already bound.")
        world_size = dist.get_world_size(self.group)
        if world_size != self.config.expert_model_parallel_size:
            raise ValueError("Home-only EP size does not match its process group.")
        if self.config.num_moe_experts % world_size:
            raise ValueError("Home-only execution requires uniform home expert counts.")
        h = self.config.num_moe_experts // world_size
        specs, device = _collect_replica_projection_specs(
            experts, num_local_experts=h, backend_name="Home-only"
        )
        mxfp8 = specs[0].weight_format == "mxfp8"
        config = ReplicaTransportConfig(
            group=self.group,
            device=device,
            world_size=world_size,
            num_local_home_experts=h,
            num_local_replica_slots=0,
            member_shapes=tuple(spec.member_shape for spec in specs),
            weight_format=specs[0].weight_format,
            rowwise_scale_shapes=(
                tuple(spec.rowwise_scale_shape for spec in specs) if mxfp8 else None
            ),
            columnwise_scale_shapes=(
                tuple(spec.columnwise_scale_shape for spec in specs) if mxfp8 else None
            ),
            grad_dtype=torch.float32,
            num_sms=None,
        )
        self.transport = NcclP2PTransport(config)
        self.source_parameters = tuple(parameter for spec in specs for parameter in spec.parameters)
        _home_only_dispatchers.add(self)
        register_replica_transport_finalizer("home_only", _finalize_home_only_dispatchers)

    def bind_home_dispatch(self, home: HomeExpertDispatch) -> None:
        """Give home migration its NCCL transport without a replica runtime."""
        if self.transport is None:
            raise RuntimeError("Bind native experts before home dispatch.")
        home.bind_transport(self.transport)

    def bind_home_optimizer(
        self, home: HomeExpertDispatch, optimizer: MegatronOptimizer | torch.optim.Optimizer
    ) -> None:
        """Resolve authoritative optimizer state through existing shard adapters."""
        if self.transport is None:
            raise RuntimeError("Bind native experts before the optimizer.")
        home.bind_state_adapter(
            OptimizerHomeExpertState(optimizer, self.source_parameters, self.transport.config)
        )

    def supports(self, expert_placement: torch.Tensor, context: SchedulerContext) -> bool:
        """The planner must return None for replica placement, including empty tables."""
        return False

    def dispatch(
        self, experts: torch.nn.Module, expert_placement: torch.Tensor, context: SchedulerContext
    ) -> None:
        """Reject accidental replica work on the native execution path."""
        raise RuntimeError("Home-only execution cannot dispatch replica weights.")

    def after_token_combine(self, combined_hidden: torch.Tensor) -> torch.Tensor:
        """Track only native graph lifetime; do not copy weights or reduce gradients."""
        if torch.is_grad_enabled() and combined_hidden.requires_grad:
            token = object()
            self._pending_backwards.add(token)
            return _NativeBackwardLifetime.apply(combined_hidden, self, token)
        return combined_hidden

    def assert_idle(self) -> None:
        """Reject home mutation while a native microbatch still needs backward."""
        if self._pending_backwards:
            raise RuntimeError("Home exchange requires all native forwards/backwards to finish.")

    def destroy(self) -> None:
        """Drain home communication while retaining the borrowed EP process group."""
        if self.transport is not None:
            self.transport.destroy()
        _home_only_dispatchers.discard(self)
