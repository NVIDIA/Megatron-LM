# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Common interface implemented by MoE megakernel backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn


class MegakernelBackend(nn.Module, ABC):
    """Backend-neutral interface for replacing an MCore MoE layer."""

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Execute the MoE workload using MCore's authoritative routing result.

        ``padding_mask`` (batch-first ``[bsz, seq_length]``, True = padding) marks tokens that
        must get no routes: the backend neither dispatches nor computes them, so their routed
        output and their probs gradient are zero.
        """

    def refresh_weight_views(self) -> None:
        """Update weight-derived state before a CUDA graph that contains this backend replays.

        A TE CUDA graph whose scope covers the whole MoE (``cuda_graph_modules`` containing
        ``moe``) replays the backend without running Python. The layer calls this eagerly before
        every replay, after the parameter all-gather hooks and on the replay stream, so a backend
        that converts its weights once per optimizer step does that here. No-op by default.
        """

    def cuda_graphs_deleted(self) -> None:
        """Release state pinned for the CUDA graphs that contained this backend.

        The layer calls this after ``TECudaGraphHelper.delete_cuda_graphs`` dropped its graphs, so
        no graph references the backend's buffers anymore (for example, MOK's weight views stop
        blocking parameter reloads and moves). No-op by default.
        """
