# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Async step output pool for the dynamic text generation controller.

Owns the dedicated ``d2h_output`` CUDA stream and the preallocated pinned
destination buffers that back every per-step CPU-visible D2H copy
(sampled tokens and, if speculative decoding is enabled, sampled MTP
tokens).

Design follows v3 plan §2.6 (stream layout) and §2.3 (typed objects). See
``lawrence/reports/20260429-context-cpu-async-schedule-claude-v3.md``.

This module is wired in commit 4. The existing serial controller path
resolves each ``AsyncStepOutput`` synchronously via ``result()`` immediately
after enqueueing the copy, so behavior is bit-identical to today's blocking
``.cpu()``. Commit 5 introduces the retirement service that consumes these
outputs in step-id order and lets queue depth exceed 1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    # Module-level import would pull in ``engines/__init__.py`` and trigger a
    # circular import via ``data_parallel_inference_coordinator`` →
    # ``text_generation_controller`` → this module.
    from megatron.core.inference.engines.async_pipeline_types import AsyncStepOutput


class AsyncStepOutputPool:
    """Per-controller pool of pinned destination buffers and the dedicated
    ``d2h_output`` CUDA stream.

    A single set of pinned buffers is sufficient at queue depth 1: the serial
    wrapper resolves each ``AsyncStepOutput`` before the next step enqueues a
    new copy. Commit 18 raises queue depth and sizes the pinned-buffer pool
    accordingly.
    """

    SAMPLED_TOKENS_KEY = "sampled_tokens"
    SAMPLED_MTP_TOKENS_KEY = "sampled_mtp_tokens"
    LOGPROBS_KEY = "logprobs"
    TOP_N_LOGPROBS_KEY = "top_n_logprobs"
    ROUTING_INDICES_KEY = "routing_indices_per_request"

    def __init__(
        self,
        max_requests: int,
        num_speculative_tokens: int,
        device: torch.device,
    ) -> None:
        self._device = device
        self._num_speculative_tokens = num_speculative_tokens
        self._d2h_stream = torch.cuda.Stream(device=device)
        self._pinned_sampled_tokens = torch.empty(
            max_requests, dtype=torch.int64, device="cpu", pin_memory=True
        )
        if num_speculative_tokens > 0:
            self._pinned_sampled_mtp_tokens: Optional[torch.Tensor] = torch.empty(
                [num_speculative_tokens, max_requests],
                dtype=torch.int64,
                device="cpu",
                pin_memory=True,
            )
        else:
            self._pinned_sampled_mtp_tokens = None

    def ensure_mtp_buffer(self, num_speculative_tokens: int, max_requests: int) -> None:
        """Late-init the MTP pinned destination once ``num_speculative_tokens``
        is known. Called from ``_init_mtp_sampling_tensor`` so the pool stays
        in sync with the controller's GPU MTP buffer.
        """
        if num_speculative_tokens <= 0:
            return
        if (
            self._pinned_sampled_mtp_tokens is not None
            and self._pinned_sampled_mtp_tokens.shape == (num_speculative_tokens, max_requests)
        ):
            return
        self._num_speculative_tokens = num_speculative_tokens
        self._pinned_sampled_mtp_tokens = torch.empty(
            [num_speculative_tokens, max_requests],
            dtype=torch.int64,
            device="cpu",
            pin_memory=True,
        )

    @property
    def d2h_stream(self) -> torch.cuda.Stream:
        return self._d2h_stream

    def begin_copy(
        self,
        step_id: int,
        sampled_tokens_gpu: torch.Tensor,
        sampled_mtp_tokens_gpu: Optional[torch.Tensor],
    ) -> "AsyncStepOutput":
        """Enqueue an asynchronous D2H of the per-step sampled tensors on the
        dedicated ``d2h_output`` stream. Returns an ``AsyncStepOutput`` whose
        ``d2h_done_event`` fires once every copy in the bundle completes.

        Source-tensor lifetime is bound to the returned output via
        ``source_gpu_tensors`` (kept alive until the caller is done with the
        bundle), and pinned destinations live for the lifetime of the pool.
        """
        # Lazy import to break circular import via engines/__init__.py.
        from megatron.core.inference.engines.async_pipeline_types import AsyncStepOutput

        active_request_count = sampled_tokens_gpu.shape[0]

        # Today the sampling kernels run on the controller's current (compute)
        # stream. Wait on that stream so the D2H reads the produced tokens
        # rather than racing with the kernel that wrote them. Commit 18
        # tightens this to a per-step ``sample_done_event``.
        self._d2h_stream.wait_stream(torch.cuda.current_stream())

        sampled_dest = self._pinned_sampled_tokens[:active_request_count]
        mtp_dest: Optional[torch.Tensor] = None

        with torch.cuda.stream(self._d2h_stream):
            sampled_dest.copy_(sampled_tokens_gpu, non_blocking=True)
            if sampled_mtp_tokens_gpu is not None:
                assert self._pinned_sampled_mtp_tokens is not None, (
                    "MTP source tensor present but pool has no MTP pinned destination; "
                    "call ensure_mtp_buffer() after enabling speculative decoding."
                )
                mtp_dest = self._pinned_sampled_mtp_tokens[:, :active_request_count]
                mtp_dest.copy_(sampled_mtp_tokens_gpu, non_blocking=True)
            d2h_done = torch.cuda.Event()
            d2h_done.record(self._d2h_stream)

        output = AsyncStepOutput(step_id=step_id, d2h_done_event=d2h_done)
        output.source_gpu_tensors[self.SAMPLED_TOKENS_KEY] = sampled_tokens_gpu
        output.pinned_destinations[self.SAMPLED_TOKENS_KEY] = sampled_dest
        output.payload_metadata[self.SAMPLED_TOKENS_KEY] = True
        if sampled_mtp_tokens_gpu is not None:
            output.source_gpu_tensors[self.SAMPLED_MTP_TOKENS_KEY] = sampled_mtp_tokens_gpu
            output.pinned_destinations[self.SAMPLED_MTP_TOKENS_KEY] = mtp_dest
            output.payload_metadata[self.SAMPLED_MTP_TOKENS_KEY] = True
        return output

    def add_payload(
        self,
        output: "AsyncStepOutput",
        name: str,
        source_gpu: torch.Tensor,
    ) -> torch.Tensor:
        """v3 plan §commit 21 — add a CPU-visible payload (logprobs,
        top-n logprobs, routing records) to an existing AsyncStepOutput.

        Allocates a fresh pinned destination tensor sized to ``source_gpu``,
        enqueues the D2H on the dedicated d2h_output stream, and re-records
        ``d2h_done_event`` on that stream so consumers waiting on the
        bundle's event also see the new payload's copy. Returns the pinned
        destination view (the caller may keep a handle for downstream
        retirement-service emission).
        """
        # Pinned destinations for these payloads are sized per call rather
        # than from a fixed pool because logprob shapes vary with request
        # count + vocab subset. The cost (~MB-scale per step) is negligible
        # vs. the throughput gain from overlapping with forward.
        pinned = torch.empty(
            source_gpu.shape, dtype=source_gpu.dtype, device="cpu", pin_memory=True
        )
        with torch.cuda.stream(self._d2h_stream):
            pinned.copy_(source_gpu, non_blocking=True)
            new_event = torch.cuda.Event()
            new_event.record(self._d2h_stream)
        output.source_gpu_tensors[name] = source_gpu
        output.pinned_destinations[name] = pinned
        output.payload_metadata[name] = True
        output.d2h_done_event = new_event
        return pinned
