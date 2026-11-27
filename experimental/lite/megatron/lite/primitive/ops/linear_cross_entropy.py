# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Linear + vocab-parallel cross entropy helpers.

The fast path delegates to VERL's Triton-backed fused kernel when available.
The fallback keeps the same contract and is used by unit/smoke tests that do
not have the fused extension on ``PYTHONPATH``.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from megatron.lite.primitive.ops.cross_entropy import vocab_parallel_cross_entropy


def _all_reduce_if_needed(tensor: torch.Tensor, group, op=dist.ReduceOp.SUM) -> torch.Tensor:
    if group is not None and dist.is_initialized() and dist.get_world_size(group) > 1:
        dist.all_reduce(tensor, op=op, group=group)
    return tensor


def _vocab_parallel_entropy(logits: torch.Tensor, tp_group=None) -> torch.Tensor:
    logits = logits.float()
    logits_max = logits.max(dim=-1).values
    _all_reduce_if_needed(logits_max, tp_group, op=dist.ReduceOp.MAX)

    shifted = logits - logits_max.unsqueeze(-1)
    exp_logits = torch.exp(shifted)
    sum_exp = exp_logits.sum(dim=-1)
    _all_reduce_if_needed(sum_exp, tp_group)

    weighted_logits = (exp_logits * logits).sum(dim=-1)
    _all_reduce_if_needed(weighted_logits, tp_group)
    expected_logits = weighted_logits / sum_exp
    return torch.log(sum_exp) + logits_max - expected_logits


def _reshape_like_labels(values: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if values.shape != labels.shape and values.numel() == labels.numel():
        return values.reshape(labels.shape)
    return values


_VERL_LCE = None  # None: unresolved; False: VERL's fused kernel is unavailable.


def _verl_linear_cross_entropy():
    """VERL's fused kernel, resolved once.

    Importing VERL pulls in libraries that draw from the global RNGs at import
    time (rich's style ids). A first import inside a training step would advance
    the streams a restored checkpoint just set, so resumed training would diverge
    from continuous training; the import leaves them unchanged.
    """
    global _VERL_LCE
    if _VERL_LCE is None:
        import random

        import numpy as np

        states = random.getstate(), np.random.get_state(), torch.get_rng_state()
        try:
            from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy as fused
        except Exception:
            fused = False
        finally:
            random.setstate(states[0])
            np.random.set_state(states[1])
            torch.set_rng_state(states[2])
        _VERL_LCE = fused
    return _VERL_LCE or None


def linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    tp_group=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return token log-probs and entropy without changing VERL's fused API."""
    _verl_lce = _verl_linear_cross_entropy()
    if _verl_lce is not None and hidden.is_cuda:
        log_probs, entropy = _verl_lce(hidden, weight, labels, float(temperature), "none", tp_group)
        return _reshape_like_labels(log_probs, labels), _reshape_like_labels(entropy, labels)

    logits = torch.matmul(hidden, weight.t())
    if temperature != 1.0:
        logits = logits / float(temperature)
    loss = vocab_parallel_cross_entropy(logits.clone(), labels, tp_group)
    entropy = _vocab_parallel_entropy(logits, tp_group)
    return _reshape_like_labels(-loss, labels), _reshape_like_labels(entropy, labels)


__all__ = ["linear_cross_entropy"]
