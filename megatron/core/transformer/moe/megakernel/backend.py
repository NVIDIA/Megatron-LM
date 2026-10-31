# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Common interface implemented by MoE megakernel backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager

import torch
from torch import nn


class MegakernelBackend(nn.Module, ABC):
    """Backend-neutral interface for replacing an MCore MoE layer."""

    handles_shared_expert: bool = False
    supports_cuda_graphs: bool = False

    def cuda_graph_key(self):
        """Return the immutable execution binding for the current native schedule."""
        return None

    @contextmanager
    def cuda_graph_capture_scope(self, graph_key):
        """Bind one runner while its warmup and forward graph are created."""
        self.cuda_graph_capture_prepare()
        yield

    def cuda_graph_capture_prepare(self):
        """Bind stable runtime storage before a local runner captures this module."""
        raise NotImplementedError

    def cuda_graph_prepare_replay(self, is_first_microbatch, grad_enabled, graph_key=None):
        """Prepare an invocation after native DDP hooks; return its completion token."""
        raise NotImplementedError

    def cuda_graph_prepare_backward(self, call):
        """Validate borrowed operands before any captured backward work launches."""
        raise NotImplementedError

    def cuda_graph_finish_replay(self, call):
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
