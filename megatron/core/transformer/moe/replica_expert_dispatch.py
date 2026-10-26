# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Replica expert runtime adapter for the backend-neutral MoEScheduler contract."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch

from megatron.core.transformer.moe.moe_scheduler import ExpertDispatch, SchedulerContext
from megatron.core.transformer.moe.replica_expert_runtime import ReplicaExpertRuntime, ReplicaPlan
from megatron.core.transformer.moe.replica_weight_transport import create_replica_weight_transport

if TYPE_CHECKING:
    from megatron.core.optimizer.optimizer import MegatronOptimizer
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.moe.home_expert_dispatch import HomeExpertDispatch
    from megatron.core.transformer.transformer_config import TransformerConfig

try:
    from transformer_engine.pytorch.module.base import get_dummy_wgrad
except ImportError:
    get_dummy_wgrad = None


@dataclass
class _ReplicaPlanSlot:
    """Stable placement storage retained through one forward/backward lifetime."""

    experts_to_copy: torch.Tensor
    plan: ReplicaPlan | None = None
    in_use: bool = False
    lifetime_tracked: bool = False


class _ReplicaPlanLifetime(torch.autograd.Function):
    """Release a placement slot after dispatch backward has consumed the plan."""

    @staticmethod
    def forward(ctx, hidden_states, *args):
        """Retain the placement context until dispatch backward completes."""
        ctx.dispatcher, ctx.context = args[-2:]
        ctx.num_source_parameters = len(args) - 2
        return hidden_states

    @staticmethod
    def backward(ctx, grad_hidden_states):
        """Release the placement slot after its final backward consumer."""
        slot = ctx.context.slot
        if slot is None:
            raise RuntimeError("Replica layer input completed without a plan slot.")
        ctx.dispatcher._release_plan_slot(slot)
        return grad_hidden_states, *([None] * (ctx.num_source_parameters + 2))


class _ReplicaBackwardHook(torch.autograd.Function):
    """Run a dispatcher lifecycle boundary while passing its gradient through."""

    @staticmethod
    def forward(ctx, tensor, hook):
        """Attach a backward lifecycle callback without changing the tensor."""
        ctx.hook = hook
        return tensor

    @staticmethod
    def backward(ctx, grad):
        """Run the lifecycle callback and pass through the incoming gradient."""
        ctx.hook()
        return grad, None


class _ReplicaWaitGradReduce(torch.autograd.Function):
    """Publish replica gradients after every layer-input consumer ran backward."""

    @staticmethod
    def forward(ctx, hidden_states, *args):
        """Retain source parameters and the plan until the layer-input boundary."""
        runtime, context = args[-2:]
        ctx.runtime = runtime
        ctx.context = context
        ctx.num_source_parameters = len(args) - 2
        return hidden_states

    @staticmethod
    def backward(ctx, grad_hidden_states):
        """Wait for replica reduction and trigger the original parameter hooks."""
        source_grads = ctx.runtime.wait_grad_reduce(ctx.context.plan)
        if len(source_grads) != ctx.num_source_parameters:
            raise RuntimeError(
                "Replica reduction returned a different number of wgrads than source parameters."
            )

        autograd_grads = []
        for parameter, source_grad in zip(ctx.runtime.source_parameters, source_grads):
            if source_grad is None or getattr(parameter, "is_gtp_weight_remat", False):
                autograd_grads.append(source_grad)
                continue

            main_grad = getattr(parameter, "main_grad", None)
            if main_grad is None or not hasattr(parameter, "grad_added_to_main_grad"):
                # AccumulateGrad may retain parameter.grad, so it needs storage
                # independent from the caller-owned fused gradient destination.
                autograd_grads.append(source_grad.clone())
                continue

            if get_dummy_wgrad is None:
                raise RuntimeError("Replica fused wgrad accumulation requires Transformer Engine.")
            # FSDP's unsharded communication bucket is not zero-initialized:
            # each microbatch overwrites it before reduce-scatter accumulates
            # into the persistent shard. DDP instead accumulates locally.
            if source_grad.data_ptr() != main_grad.data_ptr():
                raise RuntimeError(
                    "Replica native gradient no longer aliases its owner's main_grad."
                )
            # GEMM and transport already wrote the owner; only trigger its hook.
            parameter.grad_added_to_main_grad = True
            autograd_grads.append(
                get_dummy_wgrad(
                    list(parameter.shape),
                    parameter.dtype,
                    zero=getattr(parameter, "zero_out_wgrad", False),
                )
            )

        ctx.runtime.release_workspace()
        return grad_hidden_states, *autograd_grads, None, None


class ReplicaExpertDispatch(ExpertDispatch):
    """Materialize an ``E + R`` layout with a transport-backed replica runtime.

    Placement is a physical source-home table ``[EP, S]``. Source IDs encode
    ``rank * H + slot`` and carry no logical identity. The runtime retains each
    table through backward so replica gradients return to the same source slot.
    The configured expert-dispatch type selects the runtime's weight transport.
    """

    dispatcher_name = "replica"

    def __init__(self, *, config: TransformerConfig, pg_collection: ProcessGroupCollection) -> None:
        super().__init__()
        self.config = config
        self.dispatcher_name = str(config.moe_scheduler_expert_dispatcher_type)
        self.group = pg_collection.ep
        self.num_experts = int(config.num_moe_experts)
        self.ep_size = int(config.expert_model_parallel_size)
        self.num_replica_slots = int(config.moe_scheduler_num_idle_experts)
        self.num_local_home_experts = self.num_experts // self.ep_size
        self.num_local_replica_slots = self.num_replica_slots // self.ep_size
        self.num_local_runtime_experts = self.num_local_home_experts + self.num_local_replica_slots
        self.runtime: ReplicaExpertRuntime | None = None
        self._plan_slots: list[_ReplicaPlanSlot] = []
        self._active_plan_slot: _ReplicaPlanSlot | None = None
        self._active_plan: ReplicaPlan | None = None
        self._forward_context = None
        self._placement_version = 0

    def bind_experts(self, experts: torch.nn.Module) -> None:
        """Bind native expert parameters before the first scheduled forward."""
        if self.runtime is not None:
            raise RuntimeError("Replica experts were already bound.")
        self.runtime = ReplicaExpertRuntime(
            experts=experts,
            group=self.group,
            num_experts=self.num_experts,
            num_local_home_experts=self.num_local_home_experts,
            num_local_replica_slots=self.num_local_replica_slots,
            transport_factory=functools.partial(
                create_replica_weight_transport, self.config.moe_scheduler_expert_dispatcher_type
            ),
            grad_dtype=torch.bfloat16 if self.config.grad_reduce_in_bf16 else torch.float32,
            num_sms=self.config.moe_flex_dispatcher_num_sms,
            replica_storage=self.config.moe_scheduler_replica_storage,
            replica_mxfp8_storage=self.config.moe_scheduler_replica_mxfp8_storage,
            replica_exchange_buffer_bytes=getattr(
                self.config, "moe_scheduler_replica_exchange_buffer_mb", 256
            )
            * 1024**2,
        )
        experts.set_replica_expert_runtime(self.runtime)

    def bind_home_dispatch(self, home: HomeExpertDispatch) -> None:
        """Share the transport while keeping runtime details inside this backend."""
        if self.runtime is None:
            raise RuntimeError("Replica experts must be bound before home dispatch.")
        home.bind_transport(self.runtime.transport)

    def bind_home_optimizer(
        self, home: HomeExpertDispatch, optimizer: MegatronOptimizer | torch.optim.Optimizer
    ) -> None:
        """Bind the optimizer adapter to this backend's native source parameters."""
        from megatron.core.transformer.moe.home_expert_state import OptimizerHomeExpertState

        if self.runtime is None:
            raise RuntimeError("Replica experts must be bound before the optimizer.")
        home.bind_state_adapter(
            OptimizerHomeExpertState(
                optimizer, self.runtime.source_parameters, self.runtime.transport.config
            )
        )

    def assert_idle(self) -> None:
        """Reject home mutation while any microbatch can still use the old weights."""
        if self._forward_context is not None or any(slot.in_use for slot in self._plan_slots):
            raise RuntimeError("Home exchange requires all replica forwards/backwards to finish.")

    def supports(self, replica_expert_placement: torch.Tensor, context: SchedulerContext) -> bool:
        return (
            replica_expert_placement.dim() == 2
            and context.num_logical_experts == self.num_experts
            and context.ep_size == self.ep_size
            and tuple(replica_expert_placement.shape)
            == (self.ep_size, self.num_local_replica_slots)
            and replica_expert_placement.dtype in (torch.int32, torch.int64)
        )

    def _acquire_plan_slot(self, device: torch.device) -> _ReplicaPlanSlot:
        for slot in self._plan_slots:
            if not slot.in_use:
                slot.in_use = True
                slot.lifetime_tracked = False
                return slot
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Replica needs another in-flight placement slot during CUDA graph "
                "capture. Warm up the same number of outstanding forwards before capture."
            )
        slot = _ReplicaPlanSlot(
            experts_to_copy=torch.empty(
                (self.ep_size, self.num_local_replica_slots), dtype=torch.int32, device=device
            ),
            in_use=True,
        )
        self._plan_slots.append(slot)
        return slot

    def _release_plan_slot(self, slot: _ReplicaPlanSlot) -> None:
        if not slot.in_use:
            raise RuntimeError("Replica placement slot was released twice.")
        slot.plan = None
        slot.in_use = False
        slot.lifetime_tracked = False

    def dispatch(
        self,
        experts: torch.nn.Module,
        replica_expert_placement: torch.Tensor,
        context: SchedulerContext,
    ) -> None:
        """Start asynchronous weight prefetch for the common physical layout."""
        del experts
        if self.runtime is None:
            raise RuntimeError("Replica experts must be bound before dispatch.")
        if self._active_plan is not None or self._active_plan_slot is not None:
            raise RuntimeError(
                "Replica requires the previous token combine to finish before dispatch."
            )
        if not self.supports(replica_expert_placement, context):
            raise ValueError(
                "ReplicaExpertRuntime requires a physical [EP, S] source-slot table matching its "
                "configured replica slots."
            )

        slot = self._acquire_plan_slot(replica_expert_placement.device)
        slot.experts_to_copy.copy_(replica_expert_placement)
        self._placement_version += 1
        plan = ReplicaPlan(
            virtual_experts=None,
            experts_to_copy=slot.experts_to_copy,
            version=self._placement_version,
        )
        slot.plan = plan
        if self._forward_context is None:
            raise RuntimeError("Replica layer input must be wrapped before planning.")
        self._forward_context.plan = plan
        self._forward_context.slot = slot
        self._active_plan_slot = slot
        self._active_plan = plan
        try:
            self.runtime.acquire_workspace()
            self.runtime.last_plan = plan
            self.runtime.start_prefetch(plan)
        except Exception:
            self.runtime.release_workspace()
            self._forward_context.plan = None
            self._forward_context.slot = None
            self._forward_context = None
            self._active_plan_slot = None
            self._active_plan = None
            self._release_plan_slot(slot)
            raise

    def wrap_layer_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Attach the final gradient wait outside router/shared-expert backward."""
        if self.runtime is None:
            raise RuntimeError("Replica experts must be bound before forward.")
        if self._forward_context is not None:
            raise RuntimeError("Replica layer input was wrapped twice without combine.")
        context = self._forward_context = SimpleNamespace(plan=None, slot=None)
        if not torch.is_grad_enabled():
            return hidden_states
        hidden_states = _ReplicaPlanLifetime.apply(
            hidden_states, *self.runtime.source_parameters, self, context
        )
        return _ReplicaWaitGradReduce.apply(
            hidden_states, *self.runtime.source_parameters, self.runtime, context
        )

    def before_token_dispatch(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Start pending reductions after dispatch backward has completed."""
        if self._active_plan is None or not torch.is_grad_enabled():
            return hidden_states
        if self._active_plan_slot is None or self._active_plan_slot.plan is not self._active_plan:
            raise RuntimeError("Replica lost its active placement slot.")
        self._active_plan_slot.lifetime_tracked = hidden_states.requires_grad
        return _ReplicaBackwardHook.apply(
            hidden_states,
            functools.partial(self.runtime.start_pending_grad_reduces, self._active_plan),
        )

    def after_token_dispatch(self, dispatched_hidden: torch.Tensor) -> torch.Tensor:
        """No boundary is needed between dispatch and expert compute."""
        return dispatched_hidden

    def before_token_combine(self, expert_output: torch.Tensor) -> torch.Tensor:
        """Wait for backward-direction weights immediately before expert backward."""
        if self._active_plan is None:
            return expert_output
        return _ReplicaBackwardHook.apply(
            expert_output,
            functools.partial(self.runtime.wait_prefetch_for_backward, self._active_plan),
        )

    def after_token_combine(self, combined_hidden: torch.Tensor) -> torch.Tensor:
        """Start backward weight prefetch and finish the forward plan scope."""
        plan = self._active_plan
        slot = self._active_plan_slot
        if plan is None or slot is None or slot.plan is not plan:
            return combined_hidden
        if torch.is_grad_enabled() and combined_hidden.requires_grad:
            combined_hidden = _ReplicaBackwardHook.apply(
                combined_hidden, functools.partial(self.runtime.start_backward_prefetch, plan)
            )
        self.runtime.release_workspace()
        self._active_plan = None
        self._active_plan_slot = None
        self._forward_context = None
        if not slot.lifetime_tracked:
            self._release_plan_slot(slot)
        return combined_hidden
