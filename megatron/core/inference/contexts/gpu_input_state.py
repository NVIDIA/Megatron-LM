# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""GPU-resident sampled-token state for async dynamic input preparation."""

from __future__ import annotations

from typing import Optional

import torch


class GpuSampledTokenState:
    """Owns GPU-side sampled-token buffers and the event that makes them usable."""

    def __init__(
        self,
        *,
        max_requests: int,
        num_speculative_tokens: int,
        device,
        stream: torch.cuda.Stream,
    ):
        if max_requests <= 0:
            raise ValueError(f"max_requests must be > 0, got {max_requests}")
        if num_speculative_tokens < 0:
            raise ValueError(
                f"num_speculative_tokens must be >= 0, got {num_speculative_tokens}"
            )

        self.max_requests = int(max_requests)
        self.num_speculative_tokens = int(num_speculative_tokens)
        self.stream = stream
        self.sampled_tokens = torch.empty(self.max_requests, dtype=torch.int64, device=device)
        self.sampled_mtp_tokens = (
            torch.empty(
                (self.num_speculative_tokens, self.max_requests),
                dtype=torch.int64,
                device=device,
            )
            if self.num_speculative_tokens > 0
            else None
        )
        self.accepted_token_counts = torch.empty(
            self.max_requests, dtype=torch.int64, device=device
        )
        self.sample_gpu_ready_event = torch.cuda.Event()
        self.last_active_request_count = 0

    def record(
        self,
        *,
        sampled_tokens: torch.Tensor,
        active_request_count: int,
        source_stream: Optional[torch.cuda.Stream] = None,
        sampled_mtp_tokens: Optional[torch.Tensor] = None,
        accepted_token_counts: Optional[torch.Tensor] = None,
    ) -> torch.cuda.Event:
        """Enqueue GPU-to-GPU sampled-token handoff and record a ready event."""
        active_request_count = int(active_request_count)
        if active_request_count < 0:
            raise ValueError(
                f"active_request_count must be >= 0, got {active_request_count}"
            )
        if active_request_count > self.max_requests:
            raise ValueError(
                f"active_request_count {active_request_count} exceeds max_requests "
                f"{self.max_requests}"
            )

        if source_stream is None:
            source_stream = torch.cuda.current_stream()
        self.stream.wait_stream(source_stream)

        active_slice = slice(0, active_request_count)
        with torch.cuda.stream(self.stream):
            self.sampled_tokens[active_slice].copy_(
                sampled_tokens[active_slice], non_blocking=True
            )
            if self.sampled_mtp_tokens is not None:
                if sampled_mtp_tokens is None:
                    self.sampled_mtp_tokens[:, active_slice].fill_(-1)
                else:
                    self.sampled_mtp_tokens[:, active_slice].copy_(
                        sampled_mtp_tokens[:, active_slice], non_blocking=True
                    )
            if accepted_token_counts is None:
                self.accepted_token_counts[active_slice].fill_(1)
            else:
                self.accepted_token_counts[active_slice].copy_(
                    accepted_token_counts[active_slice], non_blocking=True
                )
            self.sample_gpu_ready_event.record(self.stream)

        self.last_active_request_count = active_request_count
        return self.sample_gpu_ready_event

    def debug_compare_cpu(
        self,
        *,
        sampled_tokens_cpu: torch.Tensor,
        active_request_count: int,
        sampled_mtp_tokens_cpu: Optional[torch.Tensor] = None,
        accepted_token_counts_cpu: Optional[torch.Tensor] = None,
    ) -> None:
        """Synchronize in debug mode and compare GPU state with CPU output copies."""
        active_request_count = int(active_request_count)
        active_slice = slice(0, active_request_count)
        self.sample_gpu_ready_event.synchronize()

        sampled_tokens = self.sampled_tokens[active_slice].cpu()
        if not torch.equal(sampled_tokens, sampled_tokens_cpu[active_slice]):
            raise RuntimeError("GPU sampled-token state diverged from CPU sampled output")

        if sampled_mtp_tokens_cpu is not None:
            if self.sampled_mtp_tokens is None:
                raise RuntimeError("CPU sampled MTP output exists without GPU MTP state")
            sampled_mtp_tokens = self.sampled_mtp_tokens[:, active_slice].cpu()
            if not torch.equal(sampled_mtp_tokens, sampled_mtp_tokens_cpu[:, active_slice]):
                raise RuntimeError("GPU sampled-MTP state diverged from CPU sampled output")

        if accepted_token_counts_cpu is not None:
            accepted_counts = self.accepted_token_counts[active_slice].cpu()
            if not torch.equal(accepted_counts, accepted_token_counts_cpu[active_slice]):
                raise RuntimeError("GPU accepted-token counts diverged from CPU output")
