# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Per-iteration packed SFT statistics shared by training and multimodal callers."""

from typing import Dict, Optional

import torch

from megatron.core import mpu

# Optional per-iteration packed SFT statistics. Individual sample lengths are
# retained so the median for the global batch is exact.
_packed_sequence_lengths_in_iteration: list[torch.Tensor] = []
_packed_sequence_trained_tokens_in_iteration: Optional[torch.Tensor] = None
_packed_sequence_stats_active: bool = False


def update_packed_sequence_stats(sample_lengths, loss_mask):
    """Accumulate original-sample lengths and trained tokens for one microbatch."""
    global _packed_sequence_lengths_in_iteration
    global _packed_sequence_trained_tokens_in_iteration
    global _packed_sequence_stats_active

    if sample_lengths is None or loss_mask is None:
        return

    # Contribute once per data-parallel replica. All model-parallel peers still
    # participate in the global collectives when the accumulator is consumed.
    if torch.distributed.is_initialized() and mpu.model_parallel_is_initialized():
        if (
            not mpu.is_pipeline_last_stage(ignore_virtual=True)
            or mpu.get_tensor_model_parallel_rank() != 0
            or mpu.get_context_parallel_rank() != 0
        ):
            return

    device = (
        torch.device(f"cuda:{torch.cuda.current_device()}")
        if torch.cuda.is_available()
        else sample_lengths.device
    )
    lengths = sample_lengths.detach().reshape(-1)
    lengths = lengths[lengths > 0]
    if lengths.numel() == 0:
        return

    _packed_sequence_lengths_in_iteration.append(lengths.to(device=device, dtype=torch.float64))
    trained_tokens = loss_mask.detach().to(device=device, dtype=torch.float64).sum()
    if _packed_sequence_trained_tokens_in_iteration is None:
        _packed_sequence_trained_tokens_in_iteration = torch.zeros(
            (), dtype=torch.float64, device=device
        )
    _packed_sequence_trained_tokens_in_iteration += trained_tokens
    _packed_sequence_stats_active = True


def _reset_packed_sequence_stats_in_iteration():
    """Clear the per-iteration packed statistics accumulator."""
    global _packed_sequence_lengths_in_iteration
    global _packed_sequence_trained_tokens_in_iteration
    global _packed_sequence_stats_active
    _packed_sequence_lengths_in_iteration = []
    _packed_sequence_trained_tokens_in_iteration = None
    _packed_sequence_stats_active = False


def consume_packed_sequence_stats_in_iteration() -> Optional[Dict[str, float]]:
    """Read, reset, and globally gather packed SFT stats for this iteration."""
    device = (
        torch.device(f"cuda:{torch.cuda.current_device()}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if _packed_sequence_lengths_in_iteration:
        local_lengths = torch.cat(_packed_sequence_lengths_in_iteration).to(
            device=device, dtype=torch.float64
        )
    else:
        local_lengths = torch.empty(0, dtype=torch.float64, device=device)

    if _packed_sequence_trained_tokens_in_iteration is None:
        trained_tokens = torch.zeros((), dtype=torch.float64, device=device)
    else:
        trained_tokens = _packed_sequence_trained_tokens_in_iteration.to(
            device=device, dtype=torch.float64
        )

    if torch.distributed.is_initialized():
        local_count = torch.tensor([local_lengths.numel()], dtype=torch.int64, device=device)
        counts = [torch.empty_like(local_count) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(counts, local_count)
        counts = torch.cat(counts)
        torch.distributed.all_reduce(trained_tokens, op=torch.distributed.ReduceOp.SUM)

        if counts.sum().item() == 0:
            _reset_packed_sequence_stats_in_iteration()
            return None

        padded_lengths = torch.zeros(int(counts.max().item()), dtype=torch.float64, device=device)
        if local_lengths.numel() > 0:
            padded_lengths[: local_lengths.numel()] = local_lengths
        gathered_lengths = [torch.empty_like(padded_lengths) for _ in range(counts.numel())]
        torch.distributed.all_gather(gathered_lengths, padded_lengths)
        lengths = torch.cat(
            [
                gathered[: int(count.item())]
                for gathered, count in zip(gathered_lengths, counts)
                if count.item() > 0
            ]
        )
    else:
        if not _packed_sequence_stats_active:
            _reset_packed_sequence_stats_in_iteration()
            return None
        lengths = local_lengths

    if lengths.numel() == 0:
        _reset_packed_sequence_stats_in_iteration()
        return None

    stats = {
        "packed_sequence/total_tokens": lengths.sum().item(),
        "packed_sequence/trained_tokens": trained_tokens.item(),
        "packed_sequence/original_samples": float(lengths.numel()),
        "packed_sequence/original_sample_length_min": lengths.min().item(),
        "packed_sequence/original_sample_length_mean": lengths.mean().item(),
        "packed_sequence/original_sample_length_max": lengths.max().item(),
        "packed_sequence/original_sample_length_median": torch.quantile(lengths, 0.5).item(),
        "packed_sequence/original_sample_length_stdv": lengths.std(unbiased=False).item(),
    }
    _reset_packed_sequence_stats_in_iteration()
    return stats
