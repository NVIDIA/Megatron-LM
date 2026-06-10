"""Data contracts — forward_backward input/output types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


class Batch:
    """Protocol for data passed to Runtime.forward_backward.

    Sequences are packed without padding.  ``sizes()`` gives per-sequence
    lengths so the runtime can reconstruct ``cu_seqlens`` / ``position_ids``
    for THD attention.

    Subclass this to carry domain-specific fields (loss_mask, routed_experts,
    etc.) while keeping the runtime interface uniform.
    """

    def __len__(self) -> int:
        """Number of sequences in this batch."""
        raise NotImplementedError

    def sizes(self) -> torch.Tensor:
        """Per-sequence token counts.  Shape ``[num_seqs]``."""
        raise NotImplementedError

    def __getitem__(self, key: str) -> Any:
        """Dict-like field access (``batch["input_ids"]``)."""
        raise NotImplementedError


@dataclass(slots=True)
class PackedBatch(Batch):
    """Variable-length packed batch — no padding.

    All token-level tensors are 1-D with length ``sum(seq_lens)``.
    ``cu_seqlens`` and ``position_ids`` are derived automatically.
    """

    input_ids: torch.Tensor  # [total_tokens]
    labels: torch.Tensor  # [total_tokens]
    seq_lens: torch.Tensor  # [num_seqs]
    loss_mask: torch.Tensor | None = None  # [total_tokens]
    position_ids: torch.Tensor | None = None  # [total_tokens], auto if None
    routed_experts: torch.Tensor | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.seq_lens)

    def sizes(self) -> torch.Tensor:
        return self.seq_lens

    def __getitem__(self, key: str) -> Any:
        if key in self.__slots__:
            return getattr(self, key)
        return self.extras[key]

    @property
    def cu_seqlens(self) -> torch.Tensor:
        """Cumulative sequence lengths for THD attention.  Shape ``[num_seqs+1]``."""
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
        """Generate per-token position_ids from seq_lens."""
        if self.position_ids is not None:
            return self.position_ids
        return torch.cat(
            [torch.arange(s, device=self.seq_lens.device) for s in self.seq_lens.tolist()]
        )


@dataclass(slots=True)
class TrainBatch:
    """Legacy fixed-shape batch (padded).  Use PackedBatch for new code."""

    input_ids: torch.Tensor
    labels: torch.Tensor
    loss_mask: torch.Tensor | None = None
    position_ids: torch.Tensor | None = None
    routed_experts: torch.Tensor | None = None
    cp_size: int | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ModelOutputs:
    """Model forward output."""

    loss: torch.Tensor | None = None
    vocab_parallel_logits: torch.Tensor | None = None
    log_probs: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None
    values: torch.Tensor | None = None
    # MTP
    mtp_logits: torch.Tensor | None = None
    mtp_loss: torch.Tensor | None = None
    # Router Replay: recorded routing decisions
    routed_experts: torch.Tensor | None = None


@dataclass(slots=True)
class ForwardResult:
    """Output of forward_backward."""

    model_output: ModelOutputs = field(default_factory=ModelOutputs)
    metrics: dict[str, Any] = field(default_factory=dict)


__all__ = ["Batch", "ForwardResult", "ModelOutputs", "PackedBatch", "TrainBatch"]
