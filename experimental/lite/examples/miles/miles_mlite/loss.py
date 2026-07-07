# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""External loss functions for the miles MLite actor."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import torch
from megatron.lite.runtime.contracts.loss import LossContext


def _nested_to_list(output: torch.Tensor, seq_lens: torch.Tensor) -> list[torch.Tensor]:
    if getattr(output, "is_nested", False):
        return [x.reshape(-1) for x in output.unbind()]
    tensor = output
    if tensor.dim() == 2 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)
    pieces = []
    offset = 0
    for length_t in seq_lens:
        length = int(length_t.item())
        pieces.append(tensor.narrow(0, offset, length).reshape(-1))
        offset += length
    return pieces


def _unpack_output(handle, runtime_batch, output: torch.Tensor) -> list[torch.Tensor]:
    proto = handle._extras.get("protocol")
    unpack = getattr(proto, "unpack_forward_output", None)
    if unpack is None:
        raise ValueError("Model protocol must expose unpack_forward_output for miles losses.")
    model = handle._model[0] if isinstance(handle._model, list | tuple) else handle._model
    return _nested_to_list(unpack(model, runtime_batch, output), runtime_batch.seq_lens)


def extract_response_log_probs(raw_output: dict[str, torch.Tensor], runtime_batch, source_batch, handle):
    log_probs = raw_output.get("log_probs")
    if log_probs is None:
        raise ValueError("Megatron Lite model output must contain token log_probs.")
    full_log_probs = _unpack_output(handle, runtime_batch, log_probs)
    response_log_probs = []
    for values, mask in zip(full_log_probs, source_batch["aligned_loss_masks"], strict=True):
        active = mask.to(device=values.device).bool()
        if values.numel() != active.numel():
            raise ValueError(
                f"log_probs length {values.numel()} does not match aligned mask length {active.numel()}."
            )
        response_log_probs.append(values[active])
    return response_log_probs


def _masked_sample_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=values.device, dtype=values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def _mean_scalars(values: list[torch.Tensor], device) -> torch.Tensor:
    if not values:
        return torch.zeros((), device=device, dtype=torch.float32)
    return torch.stack([v.float() for v in values]).mean()


def _policy_loss(
    args,
    current: list[torch.Tensor],
    source_batch: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    old = source_batch.get("rollout_log_probs") if getattr(args, "use_rollout_logprobs", False) else None
    if old is None:
        old = source_batch.get("log_probs")
    if old is None:
        raise ValueError("policy_loss requires actor log_probs or rollout_log_probs.")
    advantages = source_batch.get("advantages")
    if advantages is None:
        raise ValueError("policy_loss requires advantages. Did GRPO preprocessing run?")

    eps_clip = float(getattr(args, "eps_clip", 0.2))
    eps_clip_high = float(getattr(args, "eps_clip_high", eps_clip))
    losses = []
    clipfracs = []
    kls = []
    for cur, old_lp, adv, mask in zip(current, old, advantages, source_batch["loss_masks"], strict=True):
        cur = cur.float()
        old_lp = old_lp.to(device=cur.device, dtype=cur.dtype)
        adv = adv.to(device=cur.device, dtype=cur.dtype)
        if cur.numel() != old_lp.numel() or cur.numel() != adv.numel():
            raise ValueError(
                "policy_loss tensor length mismatch: "
                f"current={cur.numel()} old={old_lp.numel()} advantages={adv.numel()}."
            )
        active = mask.to(device=cur.device, dtype=cur.dtype)
        ppo_kl = old_lp - cur
        ratio = (-ppo_kl).exp()
        pg_losses1 = -ratio * adv
        pg_losses2 = -ratio.clamp(1.0 - eps_clip, 1.0 + eps_clip_high) * adv
        pg_losses = torch.maximum(pg_losses1, pg_losses2)
        losses.append(_masked_sample_mean(pg_losses, active))
        clipfracs.append(_masked_sample_mean((pg_losses2 > pg_losses1).float(), active))
        kls.append(_masked_sample_mean(ppo_kl, active))

    loss = _mean_scalars(losses, current[0].device)
    entropy_loss = torch.zeros_like(loss)
    if getattr(args, "entropy_coef", 0.0):
        # MLite model protocols can return entropy, but qwen3_moe keeps this optional.
        entropy_loss = torch.zeros_like(loss)
        loss = loss - float(args.entropy_coef) * entropy_loss

    metrics = {
        "loss": loss.detach(),
        "pg_loss": loss.detach(),
        "pg_clipfrac": _mean_scalars(clipfracs, current[0].device).detach(),
        "ppo_kl": _mean_scalars(kls, current[0].device).detach(),
        "entropy_loss": entropy_loss.detach(),
    }
    return loss, metrics


def _sft_loss(current: list[torch.Tensor], source_batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    losses = []
    for log_probs, mask in zip(current, source_batch["loss_masks"], strict=True):
        mask = mask.to(device=log_probs.device, dtype=log_probs.dtype)
        if log_probs.numel() != mask.numel():
            raise ValueError(
                f"SFT log_probs length {log_probs.numel()} does not match response mask length {mask.numel()}."
            )
        losses.append(-_masked_sample_mean(log_probs.float(), mask))
    loss = _mean_scalars(losses, current[0].device)
    return loss, {"loss": loss.detach()}


def make_runtime_loss_fn(
    args,
    handle,
    *,
    forward_store: list[dict[str, list[torch.Tensor]]] | None = None,
    loss_context_iter: Iterator[LossContext] | None = None,
) -> Callable:
    """Build a Megatron Lite runtime loss_fn for SFT, forward-only logprob, or policy loss."""

    def _next_loss_context() -> LossContext | None:
        if loss_context_iter is None:
            return None
        try:
            return next(loss_context_iter)
        except StopIteration as exc:
            raise RuntimeError("MLite miles loss context iterator ended before loss_fn calls.") from exc

    def _loss_fn(raw_output: dict[str, torch.Tensor], runtime_batch, loss_context=None):
        if loss_context is None:
            loss_context = _next_loss_context()
        if loss_context is None or loss_context.source_batch is None:
            raise ValueError("MLite miles loss_fn requires a LossContext with source_batch.")
        source_batch = loss_context.source_batch
        current = extract_response_log_probs(raw_output, runtime_batch, source_batch, handle)
        if forward_store is not None:
            forward_store.append({"log_probs": [x.detach() for x in current]})
            zero = torch.zeros((), device=current[0].device, dtype=torch.float32)
            return zero, {"loss": zero.detach()}

        loss_type = getattr(args, "loss_type", "sft_loss")
        if loss_type == "sft_loss":
            return _sft_loss(current, source_batch)
        if loss_type == "policy_loss":
            return _policy_loss(args, current, source_batch)
        raise NotImplementedError(f"Megatron Lite miles actor does not support loss_type={loss_type!r}.")

    return _loss_fn
