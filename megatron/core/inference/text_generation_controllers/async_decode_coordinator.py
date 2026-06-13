# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import Protocol

import torch
from torch import Tensor

from megatron.core.inference.async_transaction import (
    AsyncCoordinatorStepState,
    AsyncDecodeLayout,
    AsyncDecodePlan,
    AsyncDecodeTransaction,
    AsyncLogprobMTPParticipant,
    AsyncMambaStateParticipant,
    AsyncPendingForwardUse,
    AsyncPreparedDecodeState,
    AsyncResourceParticipant,
    AsyncRowMapPolicy,
    AsyncSampleReadbackParticipant,
    AsyncSampleTicket,
    AsyncTxnState,
)
from megatron.core.inference.ep_async_protocol import EPAsyncHandoffDecision, EPStepBeginDecision


class AsyncDecodeContextOps(Protocol):
    """Context operations used by the async decode coordinator."""

    def build_prepared_state(
        self, *, pre_sampling: bool = False
    ) -> AsyncPreparedDecodeState | None:
        """Build and return a prepared decode state."""
        ...

    def publish_prepared_state(self, state: AsyncPreparedDecodeState) -> None:
        """Publish a prepared decode state to the live context."""
        ...

    def copy_prepared_input_ids(
        self,
        state: AsyncPreparedDecodeState,
        sampled_tokens_cuda: Tensor,
        sampled_mtp_tokens_cuda: Tensor | None,
        *,
        num_speculative_tokens: int,
    ) -> bool:
        """Copy sampled tokens into prepared decode input rows."""
        ...

    def current_layout(self, *, tokens_per_request: int) -> AsyncDecodeLayout:
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
        self._prepared_state: AsyncPreparedDecodeState | None = None
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

    def pending_graph_request_count(self) -> int | None:
        """Return the pending transaction's CUDA graph request count, if any."""
        transaction = self.pending_transaction()
        if transaction is None:
            return None
        return transaction.plan.graph_shape.padded_active_request_count

    def begin_transaction(
        self,
        *,
        plan: AsyncDecodePlan,
        state: AsyncTxnState = AsyncTxnState.PREPARED,
    ) -> AsyncDecodeTransaction:
        """Create and register the transaction for a just-prepared async forward."""
        transaction = AsyncDecodeTransaction(
            step_id=self._next_step_id,
            state=state,
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

    def prepare_next(self, *, pre_sampling: bool = False) -> bool:
        """Build and own one prepared async decode state."""
        self._discard_prepared_state()
        state = self._context_ops.build_prepared_state(pre_sampling=pre_sampling)
        if state is None:
            return False
        self._prepared_state = state
        return True

    def prepared_state(self) -> AsyncPreparedDecodeState | None:
        """Return the coordinator-owned prepared decode state, if any."""
        return self._prepared_state

    def consume_prepared_state(self) -> AsyncPreparedDecodeState | None:
        """Return and clear the prepared state after launch ownership transfers."""
        state = self._prepared_state
        self._prepared_state = None
        return state

    def discard_prepared_state(self) -> None:
        """Discard the prepared state and release prepared-only resources."""
        self._discard_prepared_state()

    def publish_prepared_state(self) -> None:
        """Publish the currently prepared state to the live context."""
        state = self._prepared_state
        if state is not None:
            self._context_ops.publish_prepared_state(state)

    def launch_prepared(
        self, sample_ticket: object | None = None
    ) -> tuple[bool, torch.cuda.Event | None, int | None]:
        """Publish, transfer, launch, and register one prepared async transaction."""
        state = self._prepared_state
        if state is None:
            return False, None, None

        self._context_ops.publish_prepared_state(state)
        h2d_done_event = self._context_ops.queue_h2d_transfer()
        input_ids, position_ids = self._context_ops.current_input_and_position_ids()
        self._model_callbacks.launch_prepared_forward(input_ids, position_ids)
        self._diagnostics_ops.increment_counter("_async_forward_launch_count")
        self._diagnostics_ops.increment_counter("_async_launched_forward_count")

        transaction = self.begin_transaction(plan=state.plan, state=AsyncTxnState.PREPARED)
        ledger = state.resource_ledger
        ledger.in_flight = True
        self._context_ops.register_active_ledger(ledger)
        participants = self._build_transaction_participants(
            resource_ledger=ledger, sample_ticket=sample_ticket
        )
        if participants:
            transaction.add_participants(*participants)
            transaction.prepare_participants()
        transaction.mark_launched(
            sample_ticket=sample_ticket,
            resource_ledger=ledger,
            h2d_done_event=h2d_done_event,
        )
        self._prepared_state = None
        return True, h2d_done_event, state.plan.graph_shape.padded_active_request_count

    def copy_samples_to_prepared_inputs(
        self,
        sampled_tokens_cuda: Tensor,
        sampled_mtp_tokens_cuda: Tensor | None,
    ) -> bool:
        """Copy sampled ids through the coordinator-owned prepared state."""
        state = self._prepared_state
        if state is None:
            return False
        return self._context_ops.copy_prepared_input_ids(
            state,
            sampled_tokens_cuda,
            sampled_mtp_tokens_cuda,
            num_speculative_tokens=self._model_callbacks.speculative_token_count(),
        )

    def pending_forward_row_status(self) -> tuple[bool, bool]:
        """Return whether the pending forward is reusable and row-mapped."""
        transaction = self.pending_transaction()
        if transaction is None:
            return True, False
        current_layout = self._context_ops.current_layout(
            tokens_per_request=transaction.plan.graph_shape.tokens_per_request
        )
        decision = transaction.plan.resolve_pending_forward(
            current_layout, row_map_policy=self._model_callbacks.row_map_policy()
        )
        if not decision.reusable:
            return False, False
        return True, decision.row_mapped

    def consume_pending_forward(self) -> tuple[bool, Tensor | None, bool]:
        """Resolve, commit or rollback, and retire the pending forward."""
        transaction = self.pending_transaction()
        if transaction is None:
            return False, None, False

        current_layout = self._context_ops.current_layout(
            tokens_per_request=transaction.plan.graph_shape.tokens_per_request
        )
        decision = transaction.plan.resolve_pending_forward(
            current_layout, row_map_policy=self._model_callbacks.row_map_policy()
        )
        graph_request_count = transaction.plan.graph_shape.padded_active_request_count
        if graph_request_count is None:
            graph_request_count = transaction.plan.graph_shape.active_request_count
        transaction.resolution = AsyncPendingForwardUse(
            reused=decision.reusable,
            row_indices=decision.row_map,
            row_mapped=decision.row_mapped,
            graph_request_count=graph_request_count,
        )

        if not decision.reusable:
            if not decision.graph_compatible:
                self._diagnostics_ops.increment_counter("_async_graph_mismatch_discard_count")
            else:
                self._diagnostics_ops.increment_counter("_async_layout_mismatch_discard_count")
            self._finalize_transaction(
                transaction, reused=False, reason=decision.reason or "pending forward not reusable"
            )
            return False, None, False

        if decision.row_mapped and not self._context_ops.row_mapped_reuse_allowed(row_mapped=True):
            transaction.resolution = AsyncPendingForwardUse(
                reused=False,
                row_indices=decision.row_map,
                row_mapped=True,
                graph_request_count=graph_request_count,
            )
            self._finalize_transaction(
                transaction, reused=False, reason="row-mapped non-decode forward not reusable"
            )
            return False, None, False

        if decision.row_mapped:
            self._diagnostics_ops.increment_counter("_async_row_mapped_forward_count")
        else:
            self._diagnostics_ops.increment_counter("_async_identity_forward_count")
        self._diagnostics_ops.increment_counter("_async_reused_forward_count")
        row_indices = (
            decision.row_map.to(device=torch.cuda.current_device(), dtype=torch.long)
            if decision.row_mapped and decision.row_map is not None
            else None
        )
        self._finalize_transaction(transaction, reused=True, reason=None)
        return True, row_indices, decision.row_mapped

    def discard_pending(self, reason: str = "discarded before step begin") -> None:
        """Rollback and retire the pending forward, if one exists."""
        transaction = self.pending_transaction()
        if transaction is None:
            return
        self._finalize_transaction(transaction, reused=False, reason=reason)

    def begin_step(self, *, has_real_work: bool) -> EPStepBeginDecision:
        """Preview pending forward status and resolve EP step-begin state."""
        self._step_state = AsyncCoordinatorStepState()
        pending_forward_reusable = True
        pending_forward_row_mapped = False
        transaction = self.pending_transaction()
        has_pending_forward = transaction is not None
        if transaction is not None:
            current_layout = self._context_ops.current_layout(
                tokens_per_request=transaction.plan.graph_shape.tokens_per_request
            )
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

    def _finalize_transaction(
        self, transaction: AsyncDecodeTransaction, *, reused: bool, reason: str | None
    ) -> None:
        """Commit or rollback transaction-owned effects and retire the transaction."""
        ledger = transaction.resource_ledger
        if reused:
            transaction.mark_committed()
            if ledger is not None:
                ledger.release_deferred(self._allocator_ops)
                self._context_ops.clear_active_ledger(ledger)
            self._diagnostics_ops.increment_counter("_async_committed_forward_count")
        else:
            transaction.rollback(reason or "pending forward not reused")
            if ledger is not None:
                ledger.drain(self._allocator_ops)
                self._context_ops.clear_active_ledger(ledger)
            self._diagnostics_ops.increment_counter("_async_discarded_forward_count")
            self._diagnostics_ops.increment_counter("_async_rolled_back_forward_count")
        transaction.mark_retired()
        if self._pending_transaction is transaction:
            self._pending_transaction = None

    def _discard_prepared_state(self) -> None:
        """Release resources held by a prepared state that was never launched."""
        state = self._prepared_state
        if state is not None:
            ledger = state.resource_ledger
            if ledger.reservation_count > 0:
                self._allocator_ops.release_kv_blocks(ledger.reserved_block_ids_tensor())
                ledger.clear_reservations()
            self._context_ops.clear_active_ledger(ledger)
        self._prepared_state = None

    def _build_transaction_participants(
        self, *, resource_ledger: object | None, sample_ticket: object | None
    ) -> tuple[object, ...]:
        """Build the participants for a launched transaction in one place."""
        context = getattr(self._context_ops, "_context", None)
        participants = []
        if context is not None and getattr(context, "is_hybrid_model", False):
            participants.append(AsyncMambaStateParticipant(context))
        if isinstance(sample_ticket, AsyncSampleTicket):
            participants.append(AsyncSampleReadbackParticipant(sample_ticket))

        requires_logprobs = self._model_callbacks.requires_logprobs()
        requires_mtp = self._model_callbacks.requires_mtp()
        if requires_logprobs or requires_mtp:
            participants.append(
                AsyncLogprobMTPParticipant(
                    requires_logprobs=requires_logprobs,
                    requires_mtp=requires_mtp,
                )
            )

        has_resource_work = bool(
            getattr(resource_ledger, "reservation_count", 0)
            or getattr(resource_ledger, "deferred_kv_blocks", ())
            or getattr(resource_ledger, "deferred_mamba_slots", ())
        )
        if context is not None and has_resource_work and hasattr(resource_ledger, "release_deferred"):
            participants.append(AsyncResourceParticipant(resource_ledger, context))
        return tuple(participants)

    def diagnostics(self) -> object | None:
        """Return EP diagnostics through the write-only protocol boundary."""
        return self._ep_ops.diagnostics()


class _DynamicContextOps:
    """Mechanical adapter from DynamicInferenceContext to AsyncDecodeContextOps."""

    def __init__(self, context: object) -> None:
        self._context = context

    def build_prepared_state(
        self, *, pre_sampling: bool = False
    ) -> AsyncPreparedDecodeState | None:
        return self._context.prepare_async_decode_next_step(pre_sampling=pre_sampling)

    def publish_prepared_state(self, state: AsyncPreparedDecodeState) -> None:
        self._context.publish_async_prepared_decode_state(state)

    def copy_prepared_input_ids(
        self,
        state: AsyncPreparedDecodeState,
        sampled_tokens_cuda: Tensor,
        sampled_mtp_tokens_cuda: Tensor | None,
        *,
        num_speculative_tokens: int,
    ) -> bool:
        return self._context.copy_async_prepared_decode_input_ids_from_samples(
            state,
            sampled_tokens_cuda,
            sampled_mtp_tokens_cuda,
            num_speculative_tokens=num_speculative_tokens,
        )

    def current_layout(self, *, tokens_per_request: int) -> AsyncDecodeLayout:
        return AsyncDecodeLayout.from_context_current(
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
        config = getattr(self._context, "config", None)
        is_decode_only = getattr(self._context, "is_decode_only", None)
        if config is None or is_decode_only is None:
            return True
        return bool(
            config.materialize_only_last_token_logits or is_decode_only()
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
        if not getattr(self._controller, "_async_logprob_requests_seen", False):
            return False
        try:
            return self._controller._active_requests_need_logprob_results()
        except (AttributeError, KeyError):
            return True

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
