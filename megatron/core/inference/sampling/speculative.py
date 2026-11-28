# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Reference (PyTorch) chain speculative sampling.

Standard speculative / rejection sampling (Leviathan et al. 2023, Chen et al. 2023) for a chain of
`D` draft tokens per request. The interface mirrors `flashinfer.sampling.chain_speculative_sampling`
so the FlashInfer backend can swap in the fused kernel.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor


def sample_from_probs_torch(probs: Tensor, generator: Optional[torch.Generator]) -> Tensor:
    """Draw one token per row of an (optionally unnormalized) probability matrix.

    Uses the exponential race (`argmax(p / E)`, `E ~ Exp(1)`), the same Gumbel-max form as
    `TorchSampling.sample_from_logits`: it is exact for unnormalized non-negative weights and
    never forms a cumulative sum over the vocabulary.

    Args:
        probs (Tensor): `[N, V]` non-negative weights; each row must have positive mass.
        generator (Optional[torch.Generator]): RNG for the exponential draws.

    Returns:
        Tensor: `[N]` int64 sampled token ids.
    """
    race = torch.empty_like(probs, dtype=torch.float32)
    race.exponential_(generator=generator)
    return (probs.float() / race).argmax(dim=-1)


def chain_speculative_sampling_torch(
    draft_probs: Tensor,
    draft_token_ids: Tensor,
    target_probs: Tensor,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Tensor]:
    """Verify a chain of draft tokens per request by speculative (rejection) sampling.

    For each row and k = 1..D in order, draft `x_k ~ q_k` is accepted with probability
    `min(1, p_k(x_k) / q_k(x_k))`. At the first rejection a replacement token is drawn from
    `normalize(max(0, p_k - q_k))` and the chain stops; if every draft is accepted a bonus token is
    drawn from `p_{D+1}`. The emitted sequence is then exactly distributed as the target model's
    autoregressive samples.

    Args:
        draft_probs (Tensor): `[B, D, V]` draft distributions q_1..q_D (after filtering).
        draft_token_ids (Tensor): `[B, D]` drafted tokens x_1..x_D.
        target_probs (Tensor): `[B, D + 1, V]` target distributions p_1..p_{D+1} (after
            filtering), where p_k is the target's distribution for the position of x_k.
        generator (Optional[torch.Generator]): RNG for the uniforms and the replacement draw.

    Returns:
        Tuple[Tensor, Tensor]:
            output_token_ids `[B, D + 1]` int64: accepted drafts, then the replacement or bonus
                token at index `num_accepted`, then -1 padding.
            num_accepted `[B]` int64: number of accepted drafts (excludes the emitted token).
    """
    batch, num_drafts, _ = draft_probs.shape
    assert target_probs.shape[0] == batch and target_probs.shape[1] == num_drafts + 1
    device = draft_probs.device
    draft_token_ids = draft_token_ids.long()
    rows = torch.arange(batch, device=device)

    p_x = target_probs[:, :num_drafts].gather(2, draft_token_ids.unsqueeze(-1)).squeeze(-1)
    q_x = draft_probs.gather(2, draft_token_ids.unsqueeze(-1)).squeeze(-1)
    uniform = torch.rand((batch, num_drafts), device=device, generator=generator)
    # u < p / q  <=>  u * q < p (no division; q(x) > 0 for a token drawn from q).
    accept = uniform * q_x.float() < p_x.float()
    num_accepted = accept.long().cumprod(dim=1).sum(dim=1)

    # Distribution of the emitted token: the residual at the first rejected position, or the
    # bonus distribution p_{D+1} when every draft was accepted.
    all_accepted = num_accepted == num_drafts
    p_row = target_probs[rows, num_accepted].float()
    q_row = draft_probs[rows, num_accepted.clamp(max=num_drafts - 1)].float()
    residual = (p_row - q_row).clamp_(min=0.0)
    emit_dist = torch.where(all_accepted.unsqueeze(1), p_row, residual)
    # A rejection implies p(x) < q(x), so the residual has positive mass unless p == q up to
    # rounding; fall back to p in that measure-zero case.
    degenerate = emit_dist.sum(dim=-1) <= 0
    emit_dist = torch.where(degenerate.unsqueeze(1), p_row, emit_dist)
    emitted = sample_from_probs_torch(emit_dist, generator)

    positions = torch.arange(num_drafts + 1, device=device).unsqueeze(0)
    padded_drafts = torch.cat(
        [draft_token_ids, torch.full((batch, 1), -1, device=device, dtype=torch.long)], dim=1
    )
    output = torch.where(positions < num_accepted.unsqueeze(1), padded_drafts, -1)
    output = torch.where(positions == num_accepted.unsqueeze(1), emitted.unsqueeze(1), output)
    return output, num_accepted
