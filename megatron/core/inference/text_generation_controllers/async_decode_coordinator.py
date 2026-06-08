# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import Dict, Optional

from megatron.core.inference.async_transaction import (
    AsyncDecodePlan,
    AsyncDecodeTransaction,
    AsyncLayoutSnapshot,
    AsyncTxnState,
)


class AsyncDecodeCoordinator:
    """Coordinates async dynamic decode generation for a text generation controller."""

    def __init__(self, controller: object) -> None:
        self.controller = controller

    def pending_transaction(self) -> AsyncDecodeTransaction | None:
        """Return the active transaction if it still owns pending async work."""
        transaction = getattr(self.controller, "_async_step_transaction", None)
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
        step_id = getattr(self.controller, "_async_transaction_next_step_id", 0)
        transaction = AsyncDecodeTransaction(
            step_id=step_id,
            state=state,
            snapshot=snapshot,
            plan=plan,
        )
        self.controller._async_step_transaction = transaction
        self.controller._async_transaction_next_step_id = step_id + 1
        return transaction

    def retire_transaction(self) -> None:
        """Retire and clear the controller's active transaction."""
        transaction = getattr(self.controller, "_async_step_transaction", None)
        if transaction is not None:
            transaction.mark_retired()
        self.controller._async_step_transaction = None

    async def async_generate_output_tokens_dynamic_batch(
        self, *, skip_bookkeeping: Optional[bool] = False
    ) -> Optional[Dict]:
        """Run one async dynamic decode step through the controller primitives."""
        return await self.controller._async_generate_output_tokens_dynamic_batch_impl(
            skip_bookkeeping=skip_bookkeeping
        )
