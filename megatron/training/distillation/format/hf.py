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
shifts labels again, does not apply. Packed and EOD-reset samples use an explicit 4D
additive attention mask: select a model/backend that honors it (eager or SDPA standard
attention). FlashAttention and recurrent/linear attention need their own boundary adapter.
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
    """One rank's samples, resumable from a global position; at most one loader worker.

    This stream already assigns DP ranks. Do not apply another distributed dataset
    sharder (for example Trainer/Accelerate's automatic sharding) on top of it.
    """

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


def _attention_mask(
    samples: list[dict], length: int, modes: list[str], eod_ids: list[int | None]
) -> torch.Tensor:
    """Preserve real-token attention independently of the scoring mask."""
    if all(mode == "causal" for mode in modes):
        mask = torch.zeros(len(samples), length, dtype=torch.long)
        for row, sample in enumerate(samples):
            mask[row, : sample["tokens"].numel()] = 1
        return mask
    mask = torch.full((len(samples), 1, length, length), torch.finfo(torch.float32).min)
    for row, (sample, mode, eod_id) in enumerate(zip(samples, modes, eod_ids)):
        n = sample["tokens"].numel()
        if mode == "isolated_segments":
            if "segments" not in sample:
                raise ValueError("HF offline KD isolated_segments requires segment lengths")
            segments = sample["segments"].long()
            if (segments <= 0).any() or int(segments.sum()) != n:
                raise ValueError("HF offline KD segments must cover the stored tokens exactly")
            segment_ids = torch.repeat_interleave(torch.arange(segments.numel()), segments)
        elif mode == "causal_eod_reset":
            if type(eod_id) is not int:
                raise ValueError("HF offline KD causal_eod_reset requires an eod_id")
            # EOD belongs to the preceding context; the following token starts a new one.
            segment_ids = torch.zeros(n, dtype=torch.long)
            segment_ids[1:] = (sample["tokens"][:-1] == eod_id).long().cumsum(0)
        else:
            segment_ids = torch.zeros(n, dtype=torch.long)
        positions = torch.arange(n)
        allowed = (positions[:, None] >= positions[None, :]) & (
            segment_ids[:, None] == segment_ids[None, :]
        )
        mask[row, 0, :n, :n].masked_fill_(allowed, 0.0)
    return mask


def collate(
    samples: list[dict],
    *,
    pad_token_id: int = 0,
    attention_mode: str | None = None,
    eod_id: int | None = None,
) -> dict[str, torch.Tensor]:
    """Right-pad samples and preserve their contexts; pad top-K widths with sentinels.

    Readers attach attention metadata to each sample. Hand-built samples can supply
    it through the keyword arguments. Non-causal modes return a 4D additive mask for
    eager/SDPA attention; plain causal samples use the usual 2D padding mask.
    """
    if not samples:
        raise ValueError("Cannot collate an empty offline KD microbatch")
    length = max(sample["tokens"].numel() for sample in samples)
    if length == 0:
        raise ValueError("Cannot collate an offline KD microbatch with no input tokens")
    modes, eod_ids = [], []
    for sample in samples:
        mode = sample.get("attention_mode", attention_mode or "causal")
        if mode not in ("causal", "causal_eod_reset", "isolated_segments"):
            raise ValueError(f"Unsupported HF offline KD attention mode {mode!r}")
        if attention_mode is not None and mode != attention_mode:
            raise ValueError("HF collator attention_mode differs from the stored sample")
        if (
            "attention_mode" not in sample
            and attention_mode is None
            and "segments" in sample
            and sample["segments"].numel() > 1
        ):
            raise ValueError("HF collator needs an explicit attention_mode for multiple segments")
        stored_eod = sample.get("eod_id")
        if (
            mode == "causal_eod_reset"
            and eod_id is not None
            and stored_eod is not None
            and stored_eod != eod_id
        ):
            raise ValueError("HF collator eod_id differs from the stored sample")
        modes.append(mode)
        eod_ids.append(stored_eod if stored_eod is not None else eod_id)
    with_targets = "teacher_values" in samples[0]
    if any(("teacher_values" in sample) != with_targets for sample in samples):
        raise ValueError("Cannot mix offline KD samples with and without targets")
    k = max(sample["teacher_values"].shape[-1] for sample in samples) if with_targets else 0
    batch = {
        "input_ids": torch.full((len(samples), length), pad_token_id, dtype=torch.long),
        "attention_mask": _attention_mask(samples, length, modes, eod_ids),
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


__all__ = ["CorpusStream", "collate", "rank_positions", "topk_kl"]
