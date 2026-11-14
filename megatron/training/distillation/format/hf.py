# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Reference consumer of a planned corpus blend for Hugging Face-style training (torch only).

- :func:`check_student` rejects a student tokenizer that differs from the teacher's.
- :class:`CorpusStream` yields one data-parallel rank's samples of a plan (see
  :mod:`.blend`) in Megatron's order, so each lane is read by one rank when the lane
  count is a multiple of world size × micro batch size. It keeps only that rank's part
  of the plan (:func:`.blend.rank_plan`).
- :func:`collate` pads samples into ``input_ids``, ``attention_mask``, ``position_ids``,
  ``labels`` and the teacher's top-K targets.
- :func:`topk_kl` is the single-device sparse KL that Megatron's offline KD loss
  computes, with the stored tail as the teacher's residual ("ghost") bucket.

Labels are stored already shifted: ``labels[t]`` is the token that ``logits[t]`` should
predict, and the teacher row at ``t`` comes from the teacher's logits at ``t``. Compare
both with the student's logits at the same position; a model's built-in loss, which
shifts labels again, does not apply. Packed and EOD-reset samples use an explicit 4D
additive attention mask: select a model/backend that honors it (eager or SDPA standard
attention). FlashAttention and recurrent/linear attention need their own boundary adapter.
"""

from typing import Any, Iterator

import torch

from ..v3_replay import CorpusReader
from .blend import rank_plan
from .contract import LOGPROB_SENTINEL, hf_tokenizer_identity


def check_student(manifest: dict, tokenizer: Any, *, vocab_size: int | None = None) -> None:
    """Require the student's Hugging Face tokenizer to be the teacher's.

    ``vocab_size``, when given, is the student's output (logit) width; it must cover
    every real token.
    """
    if hf_tokenizer_identity(tokenizer)["sha256"] != manifest["contract"]["tokenizer"]["sha256"]:
        raise ValueError("Offline KD student tokenizer differs from the teacher's")
    if vocab_size is not None and vocab_size < len(tokenizer):
        raise ValueError(
            f"Student vocabulary ({vocab_size}) is smaller than the tokenizer ({len(tokenizer)})"
        )


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
    """One rank's samples, resumable from a global position; at most one loader worker.

    ``tokenizer`` is the student's Hugging Face tokenizer, checked with
    :func:`check_student`; ``model_vocab_size``, the model's output width, is checked
    too when given. Every stored target index must be a token of the tokenizer.
    This stream already assigns DP ranks. Do not apply another distributed dataset
    sharder (for example Trainer/Accelerate's automatic sharding) on top of it.
    ``manifest`` is a whole :func:`.blend.plan` (or this rank's :func:`.blend.rank_plan`);
    the stream keeps only this rank's lanes.
    """

    def __init__(
        self,
        root: str,
        manifest: dict,
        *,
        tokenizer: Any,
        rank: int = 0,
        world_size: int = 1,
        micro_batch_size: int = 1,
        start: int = 0,
        model_vocab_size: int | None = None,
        **reader_options: Any,
    ):
        check_student(manifest, tokenizer, vocab_size=model_vocab_size)
        reader_options["vocab_size"] = len(tokenizer)
        layout = {
            "dp_rank": rank,
            "dp_size": world_size,
            "micro_batch_size": micro_batch_size,
            "start": start,
        }
        if manifest.get("rank") is None:
            manifest = rank_plan(manifest, **layout)
        elif manifest["rank"] != layout:
            raise ValueError(f"This corpus plan is for {manifest['rank']}, not {layout}")
        self.root, self.manifest = root, manifest
        self.rank, self.world_size, self.micro_batch_size = rank, world_size, micro_batch_size
        self.start = start
        self.reader_options = reader_options

    def __iter__(self) -> Iterator[dict]:
        worker = torch.utils.data.get_worker_info()
        if worker is not None and worker.num_workers > 1:
            raise ValueError(
                "CorpusStream requires DataLoader num_workers=0 or 1 to preserve replay order; "
                "multiple workers would duplicate samples"
            )
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


def _segments(sample: dict) -> torch.Tensor:
    """A sample's attention segments; a sample without any is one causal segment."""
    n = sample["tokens"].numel()
    segments = sample.get("segments")
    if segments is None:
        return torch.tensor([n] if n else [], dtype=torch.long)
    segments = segments.long()
    if (segments <= 0).any() or int(segments.sum()) != n:
        raise ValueError("HF offline KD segments must cover the stored tokens exactly")
    return segments


def _attention_mask(samples: list[dict], length: int) -> torch.Tensor:
    """Causal attention within each stored segment, independent of the scoring mask."""
    segments = [_segments(sample) for sample in samples]
    if all(parts.numel() <= 1 for parts in segments):
        mask = torch.zeros(len(samples), length, dtype=torch.long)
        for row, sample in enumerate(samples):
            mask[row, : sample["tokens"].numel()] = 1
        return mask
    mask = torch.full((len(samples), 1, length, length), torch.finfo(torch.float32).min)
    for row, parts in enumerate(segments):
        n = int(parts.sum())
        segment_ids = torch.repeat_interleave(torch.arange(parts.numel()), parts)
        positions = torch.arange(n)
        allowed = (positions[:, None] >= positions[None, :]) & (
            segment_ids[:, None] == segment_ids[None, :]
        )
        mask[row, 0, :n, :n].masked_fill_(allowed, 0.0)
    return mask


def collate(samples: list[dict], *, pad_token_id: int = 0) -> dict[str, torch.Tensor]:
    """Right-pad samples and preserve their contexts; pad top-K widths with sentinels.

    Attention follows each sample's ``segments`` (packed sequences or EOD-reset
    documents). Samples with several segments get a 4D additive block-causal mask for
    eager/SDPA attention; otherwise the usual 2D padding mask is returned.
    """
    if not samples:
        raise ValueError("Cannot collate an empty offline KD microbatch")
    length = max(sample["tokens"].numel() for sample in samples)
    if length == 0:
        raise ValueError("Cannot collate an offline KD microbatch with no input tokens")
    with_targets = "teacher_values" in samples[0]
    if any(("teacher_values" in sample) != with_targets for sample in samples):
        raise ValueError("Cannot mix offline KD samples with and without targets")
    k = max(sample["teacher_values"].shape[-1] for sample in samples) if with_targets else 0
    batch = {
        "input_ids": torch.full((len(samples), length), pad_token_id, dtype=torch.long),
        "attention_mask": _attention_mask(samples, length),
        "position_ids": torch.zeros(len(samples), length, dtype=torch.long),
        "labels": torch.full((len(samples), length), -100, dtype=torch.long),
        "loss_weights": torch.zeros(len(samples), length),
    }
    if k:
        dtype = samples[0]["teacher_values"].dtype
        for sample in samples[1:]:
            dtype = torch.promote_types(dtype, sample["teacher_values"].dtype)
        batch["teacher_values"] = torch.full(
            (len(samples), length, k), LOGPROB_SENTINEL, dtype=dtype
        )
        batch["teacher_indices"] = torch.zeros(len(samples), length, k, dtype=torch.long)
        batch["teacher_tail"] = torch.full((len(samples), length), LOGPROB_SENTINEL)
    for row, sample in enumerate(samples):
        n = sample["tokens"].numel()
        weights = sample["loss_mask"].float()
        batch["input_ids"][row, :n] = sample["tokens"].long()
        batch["position_ids"][row, :n] = sample["position_ids"].long()
        batch["loss_weights"][row, :n] = weights
        batch["labels"][row, :n] = torch.where(weights != 0, sample["labels"].long(), -100)
        if k:
            scored = torch.nonzero(weights != 0).reshape(-1)
            sample_k = sample["teacher_values"].shape[-1]
            batch["teacher_values"][row, scored, :sample_k] = sample["teacher_values"].to(dtype)
            batch["teacher_indices"][row, scored, :sample_k] = sample["teacher_indices"].long()
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
    Sentinel entries are skipped; target IDs outside the student's output vocabulary
    join the residual bucket. The result is averaged over the total loss weight.
    """
    logprobs = student_logits.float().log_softmax(-1)
    valid = teacher_values != LOGPROB_SENTINEL
    teacher = teacher_values.float()
    if (valid & (teacher_indices < 0)).any():
        raise ValueError("HF offline KD has an unmasked negative target index")
    unknown = valid & (teacher_indices >= logprobs.shape[-1])
    dropped = teacher.masked_fill(~unknown, -torch.inf).logsumexp(-1)
    teacher_tail = torch.logaddexp(teacher_tail.float(), dropped)
    valid = valid & ~unknown
    teacher = teacher.masked_fill(~valid, LOGPROB_SENTINEL)
    student = logprobs.gather(-1, teacher_indices.masked_fill(~valid, 0))
    student_tail = torch.log((1.0 - (student.exp() * valid).sum(-1)).clamp(min=1e-8))
    kl = (valid * teacher.exp() * (teacher - student)).sum(-1)
    kl = kl + teacher_tail.exp() * (teacher_tail - student_tail)
    return (kl * loss_weights).sum() / loss_weights.sum().clamp(min=1.0)


__all__ = ["CorpusStream", "check_student", "collate", "rank_positions", "topk_kl"]
