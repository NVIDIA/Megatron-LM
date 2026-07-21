# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Forward step and per-token loss for MIMO training."""

from __future__ import annotations

from functools import partial

import torch

from megatron.core.packed_seq_params import PackedSeqParams


def loss_func(output_tensor: torch.Tensor, *, loss_mask: torch.Tensor):
    """Return summed per-token loss, integer local token count, and logging tensors."""
    if not isinstance(output_tensor, torch.Tensor):
        raise TypeError(
            "loss_func expects the terminal language stage to return a per-token loss tensor, "
            f"got {type(output_tensor).__name__}"
        )

    if not isinstance(loss_mask, torch.Tensor) or output_tensor.shape != loss_mask.shape:
        raise RuntimeError(
            "MIMO per-token loss requires a loss_mask with the same shape as the model output"
        )

    output = output_tensor.float()
    mask = loss_mask.float()
    masked = output * mask
    num_tokens = mask.sum().to(torch.int)
    loss_sum = masked.sum()
    return (
        loss_sum,
        num_tokens,
        {"lm loss": torch.stack((loss_sum.detach(), num_tokens.detach().float()))},
    )


def mimo_forward_step(data_iterator, model):
    """Run a MIMO microbatch for the pipeline schedule.

    On the last pipeline stage, the schedule passes ``output_tensor`` to the returned loss closure.
    """
    batch = next(data_iterator) if data_iterator is not None else {"input_ids": None}
    batch = move_batch_to_cuda(batch)

    output_tensor, loss_mask = model(**batch)
    return output_tensor, partial(loss_func, loss_mask=loss_mask)


def move_batch_to_cuda(value):
    """Move tensor leaves, including PackedSeqParams tensor fields, to CUDA."""
    if isinstance(value, torch.Tensor):
        return value.cuda(non_blocking=True)
    if isinstance(value, dict):
        return {key: move_batch_to_cuda(item) for key, item in value.items()}
    if isinstance(value, list):
        return [move_batch_to_cuda(item) for item in value]
    if isinstance(value, tuple):
        return tuple(move_batch_to_cuda(item) for item in value)

    if isinstance(value, PackedSeqParams):
        for attr in (
            "cu_seqlens_q",
            "cu_seqlens_kv",
            "cu_seqlens_q_padded",
            "cu_seqlens_kv_padded",
            "max_seqlen_q",
            "max_seqlen_kv",
        ):
            sub = getattr(value, attr, None)
            if isinstance(sub, torch.Tensor) and not sub.is_cuda:
                setattr(value, attr, sub.cuda(non_blocking=True))
        return value
    return value
