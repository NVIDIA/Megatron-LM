# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Common interface implemented by MoE megakernel backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class MegakernelBackend(nn.Module, ABC):
    """Backend-neutral interface for replacing an MCore MoE layer."""

    handles_shared_expert: bool = False
    supports_cuda_graphs: bool = False

    def cuda_graph_capture_prepare(self):
        """Bind stable runtime storage before a local runner captures this module."""
        raise NotImplementedError

    def cuda_graph_prepare_replay(self, is_first_microbatch, grad_enabled):
        """Prepare an invocation after the runner's native DDP pre-forward hooks."""
        raise NotImplementedError

    def cuda_graph_prepare_backward(self):
        """Validate borrowed operands before any captured backward work launches."""
        raise NotImplementedError

    def cuda_graph_finish_replay(self):
        """Complete runtime ownership after the runner joins backward execution."""
        raise NotImplementedError

    def cuda_graph_release(self):
        """Release the runner's runtime binding when its graphs are deleted."""
        raise NotImplementedError

    @abstractmethod
    def forward(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor
    ) -> torch.Tensor:
        """Execute the MoE workload using MCore's authoritative routing result."""
