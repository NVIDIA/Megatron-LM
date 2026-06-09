# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from megatron.core.inference.async_transaction import (
    AsyncDecodePlan,
    AsyncDecodeTransaction,
    AsyncLayoutSnapshot,
    AsyncRowMapPolicy,
    AsyncTxnState,
)
from megatron.core.inference.ep_async_protocol import EPAsyncHandoffDecision, EPStepBeginDecision


class AsyncDecodeContextOps(Protocol):
    """Context operations used by the async decode coordinator."""

    def build_prepared_state(self) -> object | None:
        """Build and return a prepared decode state."""
        ...

    def publish_prepared_state(self) -> None:
        """Publish a prepared decode state to the live context."""
        ...

    def copy_prepared_input_ids(
        self,
        sampled_tokens_cuda: Tensor,
        sampled_mtp_tokens_cuda: Tensor | None,
        *,
        num_speculative_tokens: int,
    ) -> bool:
        """Copy sampled tokens into prepared decode input rows."""
        ...

    def current_layout(self, *, tokens_per_request: int) -> AsyncLayoutSnapshot:
        """Return the current context layout."""
        ...

    def queue_h2d_transfer(self) -> torch.cuda.Event | None:
        """Queue the prepared bookkeeping H2D transfer."""
        ...

    def current_input_and_position_ids(self) -> tuple[Tensor, Tensor]:
        """Return current input and position ids."""
        ...

    def row_mapped_reuse_allowed(self, *, row_mapped: bool) -> bool:
        """Return whether row-mapped reuse can be consumed by this context state."""
        ...

    def register_active_ledger(self, ledger: object) -> None:
        """Register the active borrowed resource ledger."""
        ...

    def clear_active_ledger(self, ledger: object) -> None:
        """Clear the active borrowed resource ledger."""
        ...


class AsyncDecodeAllocatorOps(Protocol):
    """Allocator operations used only by transaction-owned ledgers."""

    def release_kv_blocks(self, blocks: Tensor) -> None:
        """Release KV blocks."""
        ...

    def release_mamba_slots(self, slots: Tensor) -> None:
        """Release Mamba slots."""
        ...

    def record_deferred_kv_release(self, count: int) -> None:
        """Record released deferred KV blocks."""
        ...

    def record_deferred_mamba_release(self, count: int) -> None:
        """Record released deferred Mamba slots."""
        ...


class AsyncDecodeEPOps(Protocol):
    """EP step-ordering operations used by the coordinator."""

    def begin_step(
        self,
        *,
        has_real_work: bool,
        has_pending_forward: bool,
        pending_forward_reusable: bool,
        pending_forward_row_mapped: bool,
    ) -> EPStepBeginDecision:
        """Resolve EP step-begin state."""
        ...

    def async_handoff(
        self, *, has_real_work: bool, can_launch_async_handoff: bool
    ) -> EPAsyncHandoffDecision:
        """Resolve EP async-handoff state."""
        ...

    def ensure_handoff_decided(self, *, has_real_work: bool) -> None:
        """Ensure the handoff phase has a decision."""
        ...

    def diagnostics(self) -> object | None:
        """Return EP diagnostics."""
        ...


class AsyncDecodeDiagnosticsOps(Protocol):
    """Write-only async diagnostics operations."""

    def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment a scalar async counter."""
        ...

    def record_disable_reason(self, reason: str) -> None:
        """Record one async disable reason."""
        ...

    def record_eligibility(self, reason: str | None) -> None:
        """Record one eligibility result."""
        ...


class AsyncDecodeModelCallbacks(Protocol):
    """Model/controller callbacks needed by async launch orchestration."""

    def launch_prepared_forward(self, input_ids: Tensor, position_ids: Tensor) -> None:
        """Launch the prepared forward pass."""
        ...

    def speculative_token_count(self) -> int:
        """Return configured speculative-token count."""
        ...

    def row_map_policy(self) -> AsyncRowMapPolicy:
        """Return the row-map policy."""
        ...

    def requires_logprobs(self) -> bool:
        """Return whether active requests need logprob results."""
        ...

    def requires_mtp(self) -> bool:
        """Return whether MTP state is required."""
        ...

    def classify_eligibility(self, *, allow_mtp: bool = False) -> str | None:
        """Return an async scheduling disable reason, or None."""
        ...


@dataclass(slots=True)
class AsyncCoordinatorStepState:
    """Per-step EP state owned by the coordinator."""

    ep_step_begin_decision: EPStepBeginDecision | None = None
    ep_handoff_decision: EPAsyncHandoffDecision | None = None
    handoff_decided: bool = False


class AsyncDecodeCoordinator:
    """Coordinates async dynamic decode generation for a text generation controller."""

    def __init__(
        self,
        *,
        context_ops: AsyncDecodeContextOps,
        allocator_ops: AsyncDecodeAllocatorOps,
        ep_ops: AsyncDecodeEPOps,
        diagnostics_ops: AsyncDecodeDiagnosticsOps,
        model_callbacks: AsyncDecodeModelCallbacks,
    ) -> None:
        self._context_ops = context_ops
        self._allocator_ops = allocator_ops
        self._ep_ops = ep_ops
        self._diagnostics_ops = diagnostics_ops
        self._model_callbacks = model_callbacks
        self._prepared_state = None
        self._pending_transaction: AsyncDecodeTransaction | None = None
        self._next_step_id = 0
        self._step_state = AsyncCoordinatorStepState()

    def pending_transaction(self) -> AsyncDecodeTransaction | None:
        """Return the active transaction if it still owns pending async work."""
        transaction = self._pending_transaction
        if transaction is not None and transaction.is_in_flight:
            return transaction
        return None

    def has_pending_forward(self) -> bool:
        """Return whether a transaction owns a pending async forward."""
        return self.pending_transaction() is not None

    def begin_transaction(
        self,
        *,
        snapshot: AsyncLayoutSnapshot,
        plan: AsyncDecodePlan | None = None,
        state: AsyncTxnState = AsyncTxnState.PREPARED,
    ) -> AsyncDecodeTransaction:
        """Create and register the transaction for a just-prepared async forward."""
        transaction = AsyncDecodeTransaction(
            step_id=self._next_step_id,
            state=state,
            snapshot=snapshot,
            plan=plan,
        )
        self._pending_transaction = transaction
        self._next_step_id += 1
        return transaction

    def retire_transaction(self) -> None:
        """Retire and clear the active transaction."""
        transaction = self._pending_transaction
        if transaction is not None:
            transaction.mark_retired()
        self._pending_transaction = None

    def begin_step(self, *, has_real_work: bool) -> EPStepBeginDecision:
        """Preview pending forward status and resolve EP step-begin state."""
        self._step_state = AsyncCoordinatorStepState()
        pending_forward_reusable = True
        pending_forward_row_mapped = False
        transaction = self.pending_transaction()
        has_pending_forward = transaction is not None
        if transaction is not None:
            current_layout = self._context_ops.current_layout(
                tokens_per_request=transaction.snapshot.graph_shape.tokens_per_request
            )
            assert transaction.plan is not None
            decision = transaction.plan.resolve_pending_forward(
                current_layout, row_map_policy=self._model_callbacks.row_map_policy()
            )
            pending_forward_reusable = decision.reusable
            pending_forward_row_mapped = decision.row_mapped

        decision = self._ep_ops.begin_step(
            has_real_work=has_real_work,
            has_pending_forward=has_pending_forward,
            pending_forward_reusable=pending_forward_reusable,
            pending_forward_row_mapped=pending_forward_row_mapped,
        )
        self._step_state.ep_step_begin_decision = decision
        return decision

    def decide_handoff(
        self, *, has_real_work: bool, can_launch_async_handoff: bool
    ) -> EPAsyncHandoffDecision:
        """Resolve or return the cached EP async-handoff decision."""
        if self._step_state.ep_handoff_decision is not None:
            return self._step_state.ep_handoff_decision
        self._step_state.handoff_decided = True
        decision = self._ep_ops.async_handoff(
            has_real_work=has_real_work,
            can_launch_async_handoff=can_launch_async_handoff,
        )
        self._step_state.ep_handoff_decision = decision
        return decision

    def ensure_handoff_decided(self, *, has_real_work: bool) -> None:
        """Publish an explicit EP async handoff skip when this step did not attempt one."""
        if self._step_state.handoff_decided:
            return
        self.decide_handoff(has_real_work=has_real_work, can_launch_async_handoff=False)

    def _finalize_transaction(self) -> None:
        """Compatibility placeholder for the single transaction finalization path."""
        self.retire_transaction()

    def _discard_prepared_state(self) -> None:
        """Compatibility placeholder for prepared-state discard."""
        self._prepared_state = None

    def _build_transaction_participants(self) -> tuple[object, ...]:
        """Compatibility placeholder for single participant construction."""
        return ()

    def diagnostics(self) -> object | None:
        """Return EP diagnostics through the write-only protocol boundary."""
        return self._ep_ops.diagnostics()


class _DynamicContextOps:
    """Mechanical adapter from DynamicInferenceContext to AsyncDecodeContextOps."""

    def __init__(self, context: object) -> None:
        self._context = context

    def build_prepared_state(self) -> object | None:
        return None

    def publish_prepared_state(self) -> None:
        self._context.publish_async_prepared_decode_plan()

    def copy_prepared_input_ids(
        self,
        sampled_tokens_cuda: Tensor,
        sampled_mtp_tokens_cuda: Tensor | None,
        *,
        num_speculative_tokens: int,
    ) -> bool:
        return self._context.copy_async_prepared_decode_input_ids_from_samples(
            sampled_tokens_cuda,
            sampled_mtp_tokens_cuda,
            num_speculative_tokens=num_speculative_tokens,
        )

    def current_layout(self, *, tokens_per_request: int) -> AsyncLayoutSnapshot:
        return AsyncLayoutSnapshot.from_context_current(
            self._context, tokens_per_request=tokens_per_request
        )

    def queue_h2d_transfer(self) -> torch.cuda.Event | None:
        return self._context.transfer_bookkeeping_to_gpu(
            include_token_to_input_ids=False,
            refresh_request_staging=False,
            record_done_event=True,
        )

    def current_input_and_position_ids(self) -> tuple[Tensor, Tensor]:
        return self._context.current_input_and_position_ids()

    def row_mapped_reuse_allowed(self, *, row_mapped: bool) -> bool:
        if not row_mapped:
            return True
        return bool(
            self._context.config.materialize_only_last_token_logits or self._context.is_decode_only()
        )

    def register_active_ledger(self, ledger: object) -> None:
        if hasattr(self._context, "register_active_async_ledger"):
            self._context.register_active_async_ledger(ledger)

    def clear_active_ledger(self, ledger: object) -> None:
        if hasattr(self._context, "clear_active_async_ledger"):
            self._context.clear_active_async_ledger(ledger)


class _DynamicContextAllocatorOps:
    """Mechanical adapter from DynamicInferenceContext allocators to ledger ops."""

    def __init__(self, context: object) -> None:
        self._context = context

    def release_kv_blocks(self, blocks: Tensor) -> None:
        self._context.kv_block_allocator.release_memory_blocks(blocks)

    def release_mamba_slots(self, slots: Tensor) -> None:
        if getattr(self._context, "is_hybrid_model", False):
            self._context.mamba_metadata.free_slot_ids(slots)

    def record_deferred_kv_release(self, count: int) -> None:
        self._context.async_kv_deferred_release_count += count

    def record_deferred_mamba_release(self, count: int) -> None:
        self._context.async_mamba_deferred_release_count += count


class _ControllerEPOps:
    """Mechanical adapter for optional EP async protocol decisions."""

    def __init__(self) -> None:
        self._protocol = None

    def set_protocol(self, protocol: object) -> None:
        self._protocol = protocol

    def begin_step(
        self,
        *,
        has_real_work: bool,
        has_pending_forward: bool,
        pending_forward_reusable: bool,
        pending_forward_row_mapped: bool,
    ) -> EPStepBeginDecision:
        protocol = self._protocol
        if protocol is not None and protocol.enabled:
            begin_step = getattr(protocol, "begin_step", None)
            if begin_step is None:
                begin_step = protocol.decide_step_begin
            return begin_step(
                has_real_work=has_real_work,
                has_pending_forward=has_pending_forward,
                pending_forward_reusable=pending_forward_reusable,
                pending_forward_row_mapped=pending_forward_row_mapped,
            )

        return EPStepBeginDecision(
            step_id=-1,
            has_real_work=has_real_work,
            reuse_pending_forward=bool(has_pending_forward and pending_forward_reusable),
            discard_pending_forward=bool(has_pending_forward and not pending_forward_reusable),
            row_mapped_forward=bool(has_pending_forward and pending_forward_row_mapped),
        )

    def async_handoff(
        self, *, has_real_work: bool, can_launch_async_handoff: bool
    ) -> EPAsyncHandoffDecision:
        protocol = self._protocol
        if protocol is not None and protocol.enabled:
            decide_launch = getattr(protocol, "decide_launch", None)
            if decide_launch is None:
                decide_launch = protocol.decide_async_handoff
            return decide_launch(
                has_real_work=has_real_work,
                can_launch_async_handoff=can_launch_async_handoff,
            )

        return EPAsyncHandoffDecision(
            step_id=-1,
            has_real_work=has_real_work,
            launch_async_forward=can_launch_async_handoff,
            skip_async_forward=not can_launch_async_handoff,
            any_launch_request=can_launch_async_handoff,
            any_skip_request=not can_launch_async_handoff,
        )

    def ensure_handoff_decided(self, *, has_real_work: bool) -> None:
        return None

    def diagnostics(self) -> object | None:
        protocol = self._protocol
        if protocol is None:
            return None
        return protocol.diagnostics()


class _ControllerDiagnosticsOps:
    """Mechanical write-only adapter for controller async diagnostics."""

    def __init__(self, controller: object) -> None:
        self._controller = controller

    def increment_counter(self, name: str, value: int = 1) -> None:
        self._controller._increment_async_counter(name, value)

    def record_disable_reason(self, reason: str) -> None:
        self._controller._record_async_disable_reason(reason)

    def record_eligibility(self, reason: str | None) -> None:
        self._controller._record_async_eligibility_result(reason)


class _ControllerModelCallbacks:
    """Mechanical adapter for controller model callbacks."""

    def __init__(self, controller: object) -> None:
        self._controller = controller

    def launch_prepared_forward(self, input_ids: Tensor, position_ids: Tensor) -> None:
        self._controller._dynamic_step_forward_logits(input_ids, position_ids)

    def speculative_token_count(self) -> int:
        return self._controller.num_speculative_tokens

    def row_map_policy(self) -> AsyncRowMapPolicy:
        return getattr(self._controller, "_async_row_map_policy", AsyncRowMapPolicy.REUSE)

    def requires_logprobs(self) -> bool:
        return self._controller._active_requests_need_logprob_results()

    def requires_mtp(self) -> bool:
        return self._controller.num_speculative_tokens > 0

    def classify_eligibility(self, *, allow_mtp: bool = False) -> str | None:
        return self._controller._async_scheduling_disabled_reason(allow_mtp=allow_mtp)


def build_async_decode_coordinator(controller: object) -> AsyncDecodeCoordinator:
    """Construct the async decode coordinator from narrow protocol adapters."""
    context = controller.inference_wrapped_model.inference_context
    ep_ops = _ControllerEPOps()
    return AsyncDecodeCoordinator(
        context_ops=_DynamicContextOps(context),
        allocator_ops=_DynamicContextAllocatorOps(context),
        ep_ops=ep_ops,
        diagnostics_ops=_ControllerDiagnosticsOps(controller),
        model_callbacks=_ControllerModelCallbacks(controller),
    )
