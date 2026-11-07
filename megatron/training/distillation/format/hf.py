# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Reference consumer of a planned corpus blend for Hugging Face-style training (torch only).

- :class:`CorpusStream` yields one data-parallel rank's samples of a manifest (see
  :mod:`.blend`) in Megatron's order, so each lane is read by one rank when the lane
  count is a multiple of world size × micro batch size.
- :func:`collate` pads samples into ``input_ids``, ``attention_mask``, ``position_ids``,
  ``labels`` and the teacher's top-K targets.
- :func:`topk_kl` is the single-device sparse KL that Megatron's offline KD loss
  computes, with the stored tail as the teacher's residual ("ghost") bucket.

Labels are stored already shifted: ``labels[t]`` is the token that ``logits[t]`` should
predict, and the teacher row at ``t`` comes from the teacher's logits at ``t``. Compare
both with the student's logits at the same position; a model's built-in loss, which
shifts labels again, does not apply. Packed samples carry positions that restart at
every segment, as variable-length attention kernels expect.
"""

from typing import Any, Iterator

import torch

from ..v3_replay import CorpusReader
from .contract import LOGPROB_SENTINEL


def rank_positions(
    total: int, *, rank: int, world_size: int, micro_batch_size: int, start: int = 0
) -> Iterator[int]:
    """Global positions of one rank, step by step, as Megatron's sampler assigns them."""
    step = world_size * micro_batch_size
    for base in range(start - start % step, total - step + 1, step):
        for position in range(base + rank * micro_batch_size, base + (rank + 1) * micro_batch_size):
            if position >= start:
                yield position


class CorpusStream(torch.utils.data.IterableDataset):
    """One rank's samples of a planned blend, in order (resumable from a global position)."""

    def __init__(
        self,
        root: str,
        manifest: dict,
        *,
        rank: int = 0,
        world_size: int = 1,
        micro_batch_size: int = 1,
        start: int = 0,
        **reader_options: Any,
    ):
        self.root, self.manifest = root, manifest
        self.rank, self.world_size, self.micro_batch_size = rank, world_size, micro_batch_size
        self.start = start
        self.reader_options = reader_options

    def __iter__(self) -> Iterator[dict]:
        reader = CorpusReader(self.root, self.manifest, **self.reader_options)
        total = self.manifest["lanes"] * self.manifest["lane_samples"]
        try:
            for position in rank_positions(
                total,
                rank=self.rank,
                world_size=self.world_size,
                micro_batch_size=self.micro_batch_size,
                start=self.start,
            ):
                yield reader.samples([position], position + 1)[0]
        finally:
            reader.close()


def collate(samples: list[dict], *, pad_token_id: int = 0) -> dict[str, torch.Tensor]:
    """Right-pad samples to the longest one; padding has zero loss weight and no targets."""
    length = max(sample["tokens"].numel() for sample in samples)
    k = samples[0]["teacher_values"].shape[-1] if "teacher_values" in samples[0] else 0
    batch = {
        "input_ids": torch.full((len(samples), length), pad_token_id, dtype=torch.long),
        "attention_mask": torch.zeros(len(samples), length, dtype=torch.long),
        "position_ids": torch.zeros(len(samples), length, dtype=torch.long),
        "labels": torch.full((len(samples), length), -100, dtype=torch.long),
        "loss_weights": torch.zeros(len(samples), length),
    }
    if k:
        dtype = samples[0]["teacher_values"].dtype
        batch["teacher_values"] = torch.full(
            (len(samples), length, k), LOGPROB_SENTINEL, dtype=dtype
        )
        batch["teacher_indices"] = torch.zeros(len(samples), length, k, dtype=torch.long)
        batch["teacher_tail"] = torch.full((len(samples), length), LOGPROB_SENTINEL)
    for row, sample in enumerate(samples):
        n = sample["tokens"].numel()
        weights = sample["loss_mask"].float()
        batch["input_ids"][row, :n] = sample["tokens"].long()
        batch["attention_mask"][row, :n] = 1
        batch["position_ids"][row, :n] = sample["position_ids"].long()
        batch["loss_weights"][row, :n] = weights
        batch["labels"][row, :n] = torch.where(weights != 0, sample["labels"].long(), -100)
        if k:
            scored = torch.nonzero(weights != 0).reshape(-1)
            batch["teacher_values"][row, scored] = sample["teacher_values"]
            batch["teacher_indices"][row, scored] = sample["teacher_indices"].long()
            batch["teacher_tail"][row, scored] = sample["teacher_tail"].float()
    return batch


def topk_kl(
    student_logits: torch.Tensor,
    teacher_values: torch.Tensor,
    teacher_indices: torch.Tensor,
    teacher_tail: torch.Tensor,
    loss_weights: torch.Tensor,
) -> torch.Tensor:
    """Forward KL(teacher ‖ student) over the retained entries plus the residual bucket.

    Shapes: logits ``[B, T, V]``, targets ``[B, T, K]``, tail and weights ``[B, T]``.
    Sentinel entries are skipped; the result is averaged over the total loss weight.
    """
    logprobs = student_logits.float().log_softmax(-1)
    valid = teacher_values != LOGPROB_SENTINEL
    teacher = teacher_values.float()
    student = logprobs.gather(-1, teacher_indices.clamp(0, logprobs.shape[-1] - 1))
    student_tail = torch.log((1.0 - (student.exp() * valid).sum(-1)).clamp(min=1e-8))
    kl = (valid * teacher.exp() * (teacher - student)).sum(-1)
    kl = kl + teacher_tail.exp() * (teacher_tail - student_tail)
    return (kl * loss_weights).sum() / loss_weights.sum().clamp(min=1.0)


__all__ = ["CorpusStream", "collate", "rank_positions", "topk_kl"]
