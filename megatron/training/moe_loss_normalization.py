# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Independent input-token normalization of MoE losses for masked SFT."""

from contextlib import contextmanager
from typing import Iterator

import torch


def prepare_moe_loss_normalization(
    data_iterator: Iterator | None,
    num_microbatches: int,
    *,
    pad_token_id: int,
    tp_group: torch.distributed.ProcessGroup,
    dp_group: torch.distributed.ProcessGroup,
) -> tuple[Iterator | None, torch.Tensor]:
    """Buffer a step's SFT batches and return its global scored/input-token ratio.

    Call on every TP/DP/CP rank before executing the training schedule. Only TP
    rank zero consumes the iterator. Batches here must precede CP slicing: CP
    replicas see the same complete sequences, so reduce across DP, not DP x CP.
    The caller must pass ``tokens == pad_token_id`` as the router padding mask.
    Buffering CPU batch references does not retain GPU activations.

    A step-wide ratio is required: per-microbatch ratios incorrectly weight
    auxiliary losses when scored-token fractions differ between microbatches.
    """
    if num_microbatches < 1:
        raise ValueError("num_microbatches must be positive.")
    if tp_group is None or dp_group is None:
        raise ValueError("Explicit TP and DP process groups are required.")
    counts = torch.zeros(2, dtype=torch.int64, device="cuda")
    buffered = None
    if torch.distributed.get_rank(tp_group) == 0:
        if data_iterator is None:
            raise ValueError("Input-token MoE normalization requires an SFT data iterator.")
        buffered = [next(data_iterator) for _ in range(num_microbatches)]
        # Accumulate on the data device (normally CPU), with just one transfer.
        local_counts = torch.stack(
            [
                torch.stack(
                    [
                        batch["loss_mask"].sum().to(torch.int64),
                        batch["tokens"].ne(pad_token_id).sum(),
                    ]
                )
                for batch in buffered
            ]
        ).sum(dim=0)
        counts.copy_(local_counts)
    torch.distributed.broadcast(
        counts, src=torch.distributed.get_global_rank(tp_group, 0), group=tp_group
    )
    torch.distributed.all_reduce(counts, group=dp_group)
    # finalize_model_grads also clamps an empty scored-token denominator to one.
    ratio = counts[0].clamp_min(1).float() / counts[1].clamp_min(1)
    return (iter(buffered) if buffered is not None else None), ratio


@contextmanager
def input_token_moe_loss_scaling(config, scored_to_input_ratio: torch.Tensor):
    """Compose the correction with AMP or an existing custom MoE scaling hook.

    Router backward seeds a summed input-token auxiliary gradient. Finalization
    divides *all* gradients by the global scored-token count. Multiplying only
    the auxiliary seed by scored/input cancels that shared denominator while
    leaving the main-loss gradient unchanged. Keep the ratio constant throughout
    the schedule, including interleaved forward/backward and recomputation.
    """
    original = config.moe_grad_scale_func

    def scale():
        if original is not None:
            value = original()
            if torch.is_tensor(value):
                value = value.to(device=scored_to_input_ratio.device)
            return value * scored_to_input_ratio
        if config.grad_scale_func is not None:
            return config.grad_scale_func(torch.ones_like(scored_to_input_ratio)) * (
                scored_to_input_ratio
            )
        return scored_to_input_ratio

    config.moe_grad_scale_func = scale
    try:
        yield
    finally:
        config.moe_grad_scale_func = original
