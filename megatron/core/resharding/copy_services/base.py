from __future__ import annotations

from abc import ABC, abstractmethod
import torch


class CopyService(ABC):
    @abstractmethod
    def submit_send(self, src_tensor: torch.Tensor, dest_rank: int):
        ...

    @abstractmethod
    def submit_recv(self, dest_tensor: torch.Tensor, src_rank: int):
        ...

    @abstractmethod
    def run(self):
        ...


