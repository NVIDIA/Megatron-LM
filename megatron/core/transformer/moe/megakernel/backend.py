# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Common interface implemented by MoE megakernel backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class MegakernelBackend(nn.Module, ABC):
    """Backend-neutral interface for replacing an MCore MoE layer."""

    @abstractmethod
    def forward(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor
    ) -> torch.Tensor:
        """Execute the MoE workload using MCore's authoritative routing result."""
