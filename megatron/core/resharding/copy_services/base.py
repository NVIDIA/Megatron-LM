# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class CopyService(ABC):
    """Abstract interface for submitting and executing batched P2P copy operations."""

    @abstractmethod
    def submit_send(self, src_tensor: torch.Tensor, dest_rank: int):
        """Register a tensor send from the current rank to ``dest_rank``."""
        ...

    @abstractmethod
    def submit_recv(self, dest_tensor: torch.Tensor, src_rank: int):
        """Register a tensor receive into ``dest_tensor`` from ``src_rank``."""
        ...

    @abstractmethod
    def run(self):
        """Execute all previously submitted send/recv operations as a single batch."""
        ...
