# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch


class CopyService(ABC):
    """Abstract interface for submitting and executing batched P2P copy operations.

    All backends accept an optional *task_id* on submit calls.  The task_id is
    a globally unique identifier shared between the matching send and recv for
    the same transfer.  It is required for local (same-rank) copy matching and
    for the NVSHMEM backend's scheduling.  Backends that do not need it for
    remote transfers simply ignore it.
    """

    @abstractmethod
    def submit_send(self, src_tensor: torch.Tensor, dest_rank: int, task_id: Optional[int] = None):
        """Register a tensor send from the current rank to ``dest_rank``."""
        ...

    @abstractmethod
    def submit_recv(self, dest_tensor: torch.Tensor, src_rank: int, task_id: Optional[int] = None):
        """Register a tensor receive into ``dest_tensor`` from ``src_rank``."""
        ...

    @abstractmethod
    def run(self):
        """Execute all previously submitted send/recv operations as a single batch."""
        ...
