from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _rollout_selected_log_probs(
    logits: torch.Tensor, labels: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Evaluate selected tokens with the same reduction used by rollout."""
    from vllm.v1.worker.gpu.sample.logprob import compute_token_logprobs

    rollout_logits = logits if temperature == 1.0 else logits.float() / temperature
    return compute_token_logprobs(
        rollout_logits, labels.unsqueeze(-1)
    ).squeeze(-1)


def _differentiable_log_probs_and_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    *,
    calculate_entropy: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    scaled_logits = logits.float()
    if temperature != 1.0:
        scaled_logits = scaled_logits / temperature
    log_probs = F.log_softmax(scaled_logits, dim=-1)
    selected = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    entropy = None
    if calculate_entropy:
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    return selected, entropy


def aligned_selected_log_probs(
    hidden_states: torch.Tensor,
    lm_head: nn.Module,
    labels: torch.Tensor,
    temperature: float,
    chunk_size: int,
    *,
    calculate_entropy: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Use the rollout value and the same BF16 LM-head for its training VJP.

    vLLM's selected-logprob Triton kernel intentionally has no autograd.  The
    visible value is evaluated by that exact kernel.  The differentiable path
    is derived from the very same BF16 logits.  Non-reentrant checkpointing
    drops each full-vocabulary chunk and recomputes it during backward.
    """
    if chunk_size <= 0:
        raise ValueError("logprob_chunk_size must be positive")

    selected_chunks = []
    entropy_chunks = []
    grad_enabled = torch.is_grad_enabled()
    # Keep both forward and backward full-vocabulary storage bounded.  Training
    # recomputes one chunk at a time, so the only difference from a one-shot
    # head is floating-point accumulation order for the shared weight gradient.
    for start in range(0, hidden_states.shape[0], chunk_size):
        stop = min(start + chunk_size, hidden_states.shape[0])
        chunk_labels = labels[start:stop]

        def chunk_forward(
            chunk_hidden: torch.Tensor, chunk_labels=chunk_labels
        ):
            logits = lm_head(chunk_hidden)
            differentiable, entropy = _differentiable_log_probs_and_entropy(
                logits,
                chunk_labels,
                temperature,
                calculate_entropy=calculate_entropy,
            )
            with torch.no_grad():
                visible = _rollout_selected_log_probs(
                    logits, chunk_labels, temperature
                )
            selected = visible + (differentiable - differentiable.detach())
            if calculate_entropy:
                assert entropy is not None
                return selected, entropy
            return selected

        chunk_hidden = hidden_states[start:stop]
        if grad_enabled:
            chunk_result = checkpoint(
                chunk_forward, chunk_hidden, use_reentrant=False
            )
        else:
            with torch.no_grad():
                logits = lm_head(chunk_hidden)
                selected = _rollout_selected_log_probs(
                    logits, chunk_labels, temperature
                )
                entropy = None
                if calculate_entropy:
                    _, entropy = _differentiable_log_probs_and_entropy(
                        logits,
                        chunk_labels,
                        temperature,
                        calculate_entropy=True,
                    )
                chunk_result = (selected, entropy) if calculate_entropy else selected

        if calculate_entropy:
            selected, entropy = chunk_result
            selected_chunks.append(selected)
            entropy_chunks.append(entropy)
        else:
            selected_chunks.append(chunk_result)

    selected = torch.cat(selected_chunks, dim=0)
    entropy = torch.cat(entropy_chunks, dim=0) if calculate_entropy else None
    return selected, entropy
