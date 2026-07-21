# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Input and output contracts for ``Runtime.forward_backward``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


class Batch:
    """Base contract for a model-agnostic runtime batch."""

    def __len__(self) -> int:
        """Return the number of sequences in the batch."""
        raise NotImplementedError

    def sizes(self) -> torch.Tensor:
        """Return per-sequence token counts."""
        raise NotImplementedError


@dataclass(slots=True)
class PackedBatch(Batch):
    """Variable-length sequences packed without padding."""

    input_ids: torch.Tensor
    labels: torch.Tensor
    seq_lens: torch.Tensor
    loss_mask: torch.Tensor | None = None
    position_ids: torch.Tensor | None = None
    routed_experts: torch.Tensor | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.seq_lens)

    def sizes(self) -> torch.Tensor:
        return self.seq_lens

    @property
    def cu_seqlens(self) -> torch.Tensor:
        """Return cumulative sequence lengths in int32."""
        return torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=self.seq_lens.device),
                self.seq_lens.cumsum(0).to(torch.int32),
            ]
        )

    @property
    def total_tokens(self) -> int:
        return int(self.seq_lens.sum())

    def make_position_ids(self) -> torch.Tensor:
        """Return explicit or generated per-token position IDs."""
        if self.position_ids is not None:
            return self.position_ids
        return torch.cat(
            [
                torch.arange(length, device=self.seq_lens.device)
                for length in self.seq_lens.tolist()
            ]
        )


@dataclass(slots=True)
class TrainBatch:
    """Legacy fixed-shape batch contract."""

    input_ids: torch.Tensor
    labels: torch.Tensor
    loss_mask: torch.Tensor | None = None
    position_ids: torch.Tensor | None = None
    routed_experts: torch.Tensor | None = None
    cp_size: int | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ModelOutputs:
    """Model outputs understood by runtime integrations."""

    loss: torch.Tensor | None = None
    vocab_parallel_logits: torch.Tensor | None = None
    log_probs: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None
    values: torch.Tensor | None = None
    mtp_logits: torch.Tensor | None = None
    mtp_loss: torch.Tensor | None = None
    routed_experts: torch.Tensor | None = None


@dataclass(slots=True)
class ForwardResult:
    """Result of one logical runtime forward/backward call."""

    model_output: ModelOutputs = field(default_factory=ModelOutputs)
    metrics: dict[str, Any] = field(default_factory=dict)


__all__ = ["Batch", "ForwardResult", "ModelOutputs", "PackedBatch", "TrainBatch"]
