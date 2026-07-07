# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Rollout-data conversion for the miles MLite actor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from megatron.lite.runtime.contracts.data import PackedBatch
from megatron.lite.runtime.contracts.loss import LossContext


@dataclass(slots=True)
class RuntimeMicroBatch:
    runtime_batch: PackedBatch
    source_batch: dict[str, Any]

    def loss_context(self) -> LossContext:
        return LossContext(
            temperature=float(self.source_batch.get("temperature", 1.0)),
            calculate_entropy=bool(self.source_batch.get("calculate_entropy", False)),
            return_log_probs=True,
            source_batch=self.source_batch,
        )

    def as_runtime_item(self):
        return self.runtime_batch


def _as_tensor_list(values, *, dtype, device) -> list[torch.Tensor]:
    return [torch.as_tensor(v, dtype=dtype, device=device).reshape(-1) for v in values]


def _optional_tensor_list(data: dict[str, Any], key: str, *, dtype, device) -> list[torch.Tensor] | None:
    if key not in data or data[key] is None:
        return None
    return _as_tensor_list(data[key], dtype=dtype, device=device)


def _group_microbatches(
    total_lengths: list[int],
    *,
    micro_batch_size: int,
    use_dynamic_batch_size: bool,
    max_tokens_per_gpu: int,
) -> list[list[int]]:
    num_samples = len(total_lengths)
    if num_samples == 0:
        return []
    if not use_dynamic_batch_size:
        mbs = max(int(micro_batch_size), 1)
        return [list(range(i, min(i + mbs, num_samples))) for i in range(0, num_samples, mbs)]

    budget = max(int(max_tokens_per_gpu), 1)
    groups: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0
    for idx, length in enumerate(total_lengths):
        length = int(length)
        if current and current_tokens + length > budget:
            groups.append(current)
            current = []
            current_tokens = 0
        current.append(idx)
        current_tokens += length
    if current:
        groups.append(current)
    return groups


def response_mask_to_full(loss_mask: torch.Tensor, total_length: int, response_length: int) -> torch.Tensor:
    prompt_length = total_length - response_length
    if loss_mask.numel() != response_length:
        raise ValueError(
            f"response loss mask length {loss_mask.numel()} does not match response_length={response_length}."
        )
    return F.pad(loss_mask.float(), (prompt_length, 0), value=0.0)


def response_mask_to_aligned_full(
    loss_mask: torch.Tensor, total_length: int, response_length: int
) -> torch.Tensor:
    prompt_length = total_length - response_length
    left_pad = max(prompt_length - 1, 0)
    right_pad = total_length - response_length - left_pad
    return F.pad(loss_mask.float(), (left_pad, right_pad), value=0.0)


def _select(values: list[Any] | None, group: list[int]) -> list[Any] | None:
    if values is None:
        return None
    return [values[i] for i in group]


def _put_if_present(batch: dict[str, Any], key: str, values: list[Any] | None, group: list[int]) -> None:
    selected = _select(values, group)
    if selected is not None:
        batch[key] = selected


def build_runtime_microbatches(
    rollout_data: dict[str, Any],
    *,
    micro_batch_size: int,
    use_dynamic_batch_size: bool,
    max_tokens_per_gpu: int,
    calculate_entropy: bool = False,
    temperature: float = 1.0,
) -> list[RuntimeMicroBatch]:
    """Build model-agnostic PackedBatch items plus loss-side source metadata."""
    device = torch.device("cuda", torch.cuda.current_device())
    tokens = _as_tensor_list(rollout_data["tokens"], dtype=torch.long, device=device)
    response_loss_masks = _as_tensor_list(rollout_data["loss_masks"], dtype=torch.float32, device=device)
    rollout_log_probs = _optional_tensor_list(rollout_data, "rollout_log_probs", dtype=torch.float32, device=device)
    old_log_probs = _optional_tensor_list(rollout_data, "log_probs", dtype=torch.float32, device=device)
    ref_log_probs = _optional_tensor_list(rollout_data, "ref_log_probs", dtype=torch.float32, device=device)
    advantages = _optional_tensor_list(rollout_data, "advantages", dtype=torch.float32, device=device)
    returns = _optional_tensor_list(rollout_data, "returns", dtype=torch.float32, device=device)

    total_lengths = [int(x) for x in rollout_data["total_lengths"]]
    response_lengths = [int(x) for x in rollout_data["response_lengths"]]
    rewards = rollout_data.get("rewards")

    groups = _group_microbatches(
        total_lengths,
        micro_batch_size=micro_batch_size,
        use_dynamic_batch_size=use_dynamic_batch_size,
        max_tokens_per_gpu=max_tokens_per_gpu,
    )

    microbatches: list[RuntimeMicroBatch] = []
    for group in groups:
        group_tokens = [tokens[i] for i in group]
        group_total_lengths = [total_lengths[i] for i in group]
        group_response_lengths = [response_lengths[i] for i in group]
        group_response_masks = [response_loss_masks[i] for i in group]
        full_masks = [
            response_mask_to_full(mask, total, response)
            for mask, total, response in zip(
                group_response_masks, group_total_lengths, group_response_lengths, strict=True
            )
        ]
        aligned_masks = [
            response_mask_to_aligned_full(mask, total, response)
            for mask, total, response in zip(
                group_response_masks, group_total_lengths, group_response_lengths, strict=True
            )
        ]

        flat_tokens = torch.cat(group_tokens, dim=0).contiguous()
        runtime_batch = PackedBatch(
            input_ids=flat_tokens,
            labels=flat_tokens,
            loss_mask=torch.cat(full_masks, dim=0).contiguous(),
            seq_lens=torch.tensor(group_total_lengths, dtype=torch.int64, device=device),
        )

        source_batch: dict[str, Any] = {
            "unconcat_tokens": group_tokens,
            "total_lengths": group_total_lengths,
            "response_lengths": group_response_lengths,
            "loss_masks": group_response_masks,
            "full_loss_masks": full_masks,
            "aligned_loss_masks": aligned_masks,
            "temperature": temperature,
            "calculate_entropy": calculate_entropy,
        }
        _put_if_present(source_batch, "rollout_log_probs", rollout_log_probs, group)
        _put_if_present(source_batch, "log_probs", old_log_probs, group)
        _put_if_present(source_batch, "ref_log_probs", ref_log_probs, group)
        _put_if_present(source_batch, "advantages", advantages, group)
        _put_if_present(source_batch, "returns", returns, group)
        if rewards is not None:
            source_batch["rewards"] = [rewards[i] for i in group]

        microbatches.append(RuntimeMicroBatch(runtime_batch, source_batch))

    return microbatches
