# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Asynchronous D2H output copies for dynamic inference steps."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass(frozen=True, kw_only=True)
class AsyncStepOutputResult:
    """CPU views produced by an event-gated dynamic-step output copy."""

    sampled_tokens_cpu: torch.Tensor
    sampled_mtp_tokens_cpu: Optional[torch.Tensor] = None
    accepted_tokens_cpu: Optional[torch.Tensor] = None
    accepted_token_counts_cpu: Optional[torch.Tensor] = None
    log_probs: Optional[object] = None
    top_n_logprobs: Optional[object] = None
    routing_indices_per_request: Optional[object] = None
    output_ready_event: Optional[torch.cuda.Event] = None


class _AsyncStepOutputSlot:
    """Pinned CPU buffers owned by one output-copy slot."""

    def __init__(self, max_requests: int, num_speculative_tokens: int):
        self.in_use = False
        self.sampled_tokens_cpu = torch.empty(
            max_requests, dtype=torch.int64, device="cpu", pin_memory=True
        )
        self.sampled_mtp_tokens_cpu = (
            torch.empty(
                (num_speculative_tokens, max_requests),
                dtype=torch.int64,
                device="cpu",
                pin_memory=True,
            )
            if num_speculative_tokens > 0
            else None
        )
        self.accepted_tokens_cpu = (
            torch.empty(
                (max_requests, num_speculative_tokens),
                dtype=torch.int64,
                device="cpu",
                pin_memory=True,
            )
            if num_speculative_tokens > 0
            else None
        )
        self.accepted_token_counts_cpu = (
            torch.empty(max_requests, dtype=torch.int64, device="cpu", pin_memory=True)
            if num_speculative_tokens > 0
            else None
        )


class AsyncStepOutputPool:
    """Pinned CPU output buffers plus the dedicated dynamic-output D2H stream."""

    def __init__(self, max_requests: int, num_speculative_tokens: int, depth: int = 2):
        if max_requests <= 0:
            raise ValueError(f"max_requests must be > 0, got {max_requests}")
        if num_speculative_tokens < 0:
            raise ValueError(
                f"num_speculative_tokens must be >= 0, got {num_speculative_tokens}"
            )
        if depth <= 0:
            raise ValueError(f"depth must be > 0, got {depth}")

        self.max_requests = max_requests
        self.num_speculative_tokens = num_speculative_tokens
        self.output_stream = torch.cuda.Stream()
        self._slots = [
            _AsyncStepOutputSlot(max_requests, num_speculative_tokens) for _ in range(depth)
        ]

    def acquire(self) -> _AsyncStepOutputSlot:
        """Acquire a free pinned-buffer slot."""
        for slot in self._slots:
            if not slot.in_use:
                slot.in_use = True
                return slot
        raise RuntimeError("No free async output slots; retire an output before acquiring more")

    def release(self, slot: _AsyncStepOutputSlot) -> None:
        """Release a previously acquired pinned-buffer slot."""
        slot.in_use = False


class AsyncStepOutput:
    """In-flight output copy for one dynamic inference step."""

    def __init__(
        self,
        *,
        pool: AsyncStepOutputPool,
        slot: _AsyncStepOutputSlot,
        active_request_count: int,
        output_ready_event: torch.cuda.Event,
        sampled_mtp_copied: bool,
        accepted_tokens_copied: bool,
        accepted_token_counts_copied: bool,
        sampled_tokens_source: torch.Tensor,
        sampled_mtp_tokens_source: Optional[torch.Tensor],
        accepted_tokens_source: Optional[torch.Tensor],
        accepted_token_counts_source: Optional[torch.Tensor],
        log_probs: Optional[object],
        top_n_logprobs: Optional[object],
        routing_indices_per_request_cpu: Optional[object],
        routing_indices_sources: Optional[Dict[int, torch.Tensor]],
    ):
        self._pool = pool
        self._slot = slot
        self.active_request_count = active_request_count
        self.output_ready_event = output_ready_event
        self._sampled_mtp_copied = sampled_mtp_copied
        self._accepted_tokens_copied = accepted_tokens_copied
        self._accepted_token_counts_copied = accepted_token_counts_copied
        self._result: Optional[AsyncStepOutputResult] = None
        self._released = False

        # Keep GPU sources alive until the output-ready event has completed.
        self._sampled_tokens_source = sampled_tokens_source
        self._sampled_mtp_tokens_source = sampled_mtp_tokens_source
        self._accepted_tokens_source = accepted_tokens_source
        self._accepted_token_counts_source = accepted_token_counts_source
        self._log_probs = log_probs
        self._top_n_logprobs = top_n_logprobs
        self._routing_indices_per_request_cpu = routing_indices_per_request_cpu
        self._routing_indices_sources = routing_indices_sources

    @classmethod
    def begin_copy(
        cls,
        *,
        pool: AsyncStepOutputPool,
        active_request_count: int,
        sampled_tokens_cuda: torch.Tensor,
        sampled_mtp_tokens_cuda: Optional[torch.Tensor] = None,
        accepted_tokens_cuda: Optional[torch.Tensor] = None,
        accepted_token_counts_cuda: Optional[torch.Tensor] = None,
        log_probs: Optional[object] = None,
        top_n_logprobs: Optional[object] = None,
        routing_indices_per_request: Optional[Dict[int, torch.Tensor]] = None,
        routing_step_id: Optional[int] = None,
        compute_stream: Optional[torch.cuda.Stream] = None,
        compute_done_event: Optional[torch.cuda.Event] = None,
    ) -> "AsyncStepOutput":
        """Enqueue nonblocking output copies and record an output-ready event."""
        if active_request_count < 0:
            raise ValueError(f"active_request_count must be >= 0, got {active_request_count}")
        if active_request_count > pool.max_requests:
            raise ValueError(
                f"active_request_count {active_request_count} exceeds pool max_requests "
                f"{pool.max_requests}"
            )

        slot = pool.acquire()
        output_stream = pool.output_stream
        if compute_done_event is not None:
            output_stream.wait_event(compute_done_event)
        elif compute_stream is not None:
            output_stream.wait_stream(compute_stream)
        else:
            output_stream.wait_stream(torch.cuda.current_stream())

        sampled_mtp_copied = sampled_mtp_tokens_cuda is not None
        accepted_tokens_copied = accepted_tokens_cuda is not None
        accepted_token_counts_copied = accepted_token_counts_cuda is not None
        log_probs = copy.deepcopy(log_probs)
        top_n_logprobs = cls._clone_top_n_payload(top_n_logprobs)
        routing_indices_cpu = None
        routing_indices_sources = None
        if routing_indices_per_request is not None:
            routing_indices_cpu = {}
            routing_indices_sources = {}

        with torch.cuda.stream(output_stream):
            slot.sampled_tokens_cpu[:active_request_count].copy_(
                sampled_tokens_cuda[:active_request_count], non_blocking=True
            )
            if sampled_mtp_copied:
                if slot.sampled_mtp_tokens_cpu is None:
                    raise RuntimeError("sampled MTP output requested without MTP output buffers")
                slot.sampled_mtp_tokens_cpu[:, :active_request_count].copy_(
                    sampled_mtp_tokens_cuda[:, :active_request_count], non_blocking=True
                )
            if accepted_tokens_copied:
                if slot.accepted_tokens_cpu is None:
                    raise RuntimeError("accepted-token output requested without accepted buffers")
                slot.accepted_tokens_cpu[:active_request_count, :].copy_(
                    accepted_tokens_cuda[:active_request_count, :], non_blocking=True
                )
            if accepted_token_counts_copied:
                if slot.accepted_token_counts_cpu is None:
                    raise RuntimeError(
                        "accepted-token-count output requested without accepted-count buffers"
                )
                slot.accepted_token_counts_cpu[:active_request_count].copy_(
                    accepted_token_counts_cuda[:active_request_count], non_blocking=True
                )
            if routing_indices_per_request is not None:
                for request_id, routing_indices in routing_indices_per_request.items():
                    request_id = int(request_id)
                    if routing_indices.is_cuda:
                        routing_indices_sources[request_id] = routing_indices
                        routing_indices_cpu[request_id] = torch.empty(
                            routing_indices.shape,
                            dtype=routing_indices.dtype,
                            device="cpu",
                            pin_memory=True,
                        )
                        routing_indices_cpu[request_id].copy_(
                            routing_indices, non_blocking=True
                        )
                    else:
                        routing_indices_cpu[request_id] = routing_indices.detach().clone()
            output_ready_event = torch.cuda.Event()
            output_ready_event.record(output_stream)

        if routing_indices_cpu is not None:
            routing_indices_cpu = {
                "step_id": None if routing_step_id is None else int(routing_step_id),
                "by_request_id": routing_indices_cpu,
            }

        return cls(
            pool=pool,
            slot=slot,
            active_request_count=active_request_count,
            output_ready_event=output_ready_event,
            sampled_mtp_copied=sampled_mtp_copied,
            accepted_tokens_copied=accepted_tokens_copied,
            accepted_token_counts_copied=accepted_token_counts_copied,
            sampled_tokens_source=sampled_tokens_cuda,
            sampled_mtp_tokens_source=sampled_mtp_tokens_cuda,
            accepted_tokens_source=accepted_tokens_cuda,
            accepted_token_counts_source=accepted_token_counts_cuda,
            log_probs=log_probs,
            top_n_logprobs=top_n_logprobs,
            routing_indices_per_request_cpu=routing_indices_cpu,
            routing_indices_sources=routing_indices_sources,
        )

    @staticmethod
    def _clone_top_n_payload(top_n_logprobs: Optional[object]) -> Optional[object]:
        """Clone top-n payload tensors so retirement owns stable CPU objects."""
        if top_n_logprobs is None:
            return None
        cloned = {}
        for request_id, values in top_n_logprobs.items():
            cloned[int(request_id)] = [
                (top_values.detach().cpu().clone(), top_indices.detach().cpu().clone())
                for top_values, top_indices in values
            ]
        return cloned

    def is_ready(self) -> bool:
        """Return whether the output-ready event has completed."""
        return self.output_ready_event.query()

    @property
    def sampled_tokens_cpu(self) -> torch.Tensor:
        """Return sampled tokens only after ``result()`` has synchronized the event."""
        if self._result is None:
            raise RuntimeError("AsyncStepOutput.result() must be called before reading outputs")
        return self._result.sampled_tokens_cpu

    def result(self) -> AsyncStepOutputResult:
        """Synchronize the output event and return CPU views."""
        if self._result is None:
            self.output_ready_event.synchronize()
            active_slice = slice(0, self.active_request_count)
            sampled_tokens_cpu = self._slot.sampled_tokens_cpu[active_slice]
            sampled_mtp_tokens_cpu = (
                self._slot.sampled_mtp_tokens_cpu[:, active_slice]
                if self._sampled_mtp_copied
                else None
            )
            accepted_tokens_cpu = (
                self._slot.accepted_tokens_cpu[active_slice, :]
                if self._accepted_tokens_copied
                else None
            )
            accepted_token_counts_cpu = (
                self._slot.accepted_token_counts_cpu[active_slice]
                if self._accepted_token_counts_copied
                else None
            )
            self._result = AsyncStepOutputResult(
                sampled_tokens_cpu=sampled_tokens_cpu,
                sampled_mtp_tokens_cpu=sampled_mtp_tokens_cpu,
                accepted_tokens_cpu=accepted_tokens_cpu,
                accepted_token_counts_cpu=accepted_token_counts_cpu,
                log_probs=self._log_probs,
                top_n_logprobs=self._top_n_logprobs,
                routing_indices_per_request=self._routing_indices_per_request_cpu,
                output_ready_event=self.output_ready_event,
            )
            self._release_slot()
        return self._result

    def _release_slot(self) -> None:
        if not self._released:
            self._pool.release(self._slot)
            self._released = True
