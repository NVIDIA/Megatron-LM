# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Dependency-light MoE router replay state machine."""

from __future__ import annotations

from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


class RouterReplayAction(Enum):
    RECORD = "record"
    REPLAY_FORWARD = "replay_forward"
    REPLAY_BACKWARD = "replay_backward"


class RouterReplay:
    """Replay expert indices while gathering scores from the live router."""

    global_router_replay_instances: list["RouterReplay"] = []

    def __init__(self) -> None:
        self.target_topk_idx: torch.Tensor | None = None
        self.target_replay_mask: torch.Tensor | None = None
        self.recorded_topk_idx: torch.Tensor | None = None
        self.router_replay_action: RouterReplayAction | None = None
        self.replay_backward_list: list[torch.Tensor] = []
        self.replay_backward_mask_list: list[torch.Tensor | None] = []
        RouterReplay.global_router_replay_instances.append(self)

    @staticmethod
    def clear_global_router_replay_instances() -> None:
        RouterReplay.global_router_replay_instances.clear()

    @staticmethod
    def set_replay_data(
        all_layers_topk_indices: list[torch.Tensor],
        replay_mask: torch.Tensor | None = None,
    ) -> None:
        instances = RouterReplay.global_router_replay_instances
        if len(all_layers_topk_indices) != len(instances):
            raise ValueError(
                f"router replay expects {len(instances)} per-layer tensors, "
                f"got {len(all_layers_topk_indices)}."
            )
        for instance, indices in zip(instances, all_layers_topk_indices, strict=True):
            instance.target_topk_idx = indices
            instance.target_replay_mask = replay_mask
            # Activation checkpoint recomputation can happen after later PP
            # micro-batches have replaced the forward target.  Preserve the
            # per-microbatch sequence for REPLAY_BACKWARD, matching Megatron's
            # pipeline schedule contract.
            instance.replay_backward_list.append(indices)
            instance.replay_backward_mask_list.append(replay_mask)

    @staticmethod
    def get_recorded_data() -> list[torch.Tensor | None]:
        return [
            instance.recorded_topk_idx
            for instance in RouterReplay.global_router_replay_instances
        ]

    @staticmethod
    def set_global_router_replay_action(action: RouterReplayAction) -> None:
        for instance in RouterReplay.global_router_replay_instances:
            instance.router_replay_action = action

    @staticmethod
    def clear_global_state() -> None:
        for instance in RouterReplay.global_router_replay_instances:
            instance.target_topk_idx = None
            instance.target_replay_mask = None
            instance.recorded_topk_idx = None
            instance.router_replay_action = None
            instance.replay_backward_list.clear()
            instance.replay_backward_mask_list.clear()

    def apply(
        self,
        probs_dense: torch.Tensor,
        topk_scores: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        selected = self.select_indices(topk_indices)
        if selected is topk_indices:
            return topk_scores, topk_indices
        return probs_dense.gather(-1, selected).to(topk_scores.dtype), selected

    def select_indices(self, native_indices: torch.Tensor) -> torch.Tensor:
        """Return replayed/native indices according to action and causal mask."""

        action = self.router_replay_action
        if action == RouterReplayAction.RECORD:
            self.recorded_topk_idx = native_indices
            return native_indices
        if action == RouterReplayAction.REPLAY_FORWARD:
            target = self.target_topk_idx
            mask = self.target_replay_mask
        elif action == RouterReplayAction.REPLAY_BACKWARD:
            if not self.replay_backward_list:
                raise RuntimeError(
                    "router replay backward is active but its target queue is empty."
                )
            target = self.replay_backward_list.pop(0)
            mask = self.replay_backward_mask_list.pop(0)
        else:
            return native_indices
        if target is None:
            raise RuntimeError("router replay is active but no target indices were set.")

        target = target.to(device=native_indices.device, dtype=torch.long)
        if target.shape != native_indices.shape:
            raise ValueError(
                "router replay target shape does not match live routing: "
                f"target={tuple(target.shape)} live={tuple(native_indices.shape)}."
            )
        if mask is None:
            return target
        mask = mask.to(device=native_indices.device, dtype=torch.bool).reshape(-1, 1)
        if mask.size(0) != target.size(0):
            raise ValueError(
                "router replay mask length does not match routing rows: "
                f"mask={mask.size(0)} rows={target.size(0)}."
            )
        return torch.where(mask, target, native_indices)


def attach_router_replay(model: nn.Module, *, reset: bool = True) -> int:
    if reset:
        RouterReplay.clear_global_router_replay_instances()
    count = 0
    for module in model.modules():
        if hasattr(module, "router_replay"):
            module.router_replay = RouterReplay()
            count += 1
    return count


def detach_router_replay(model: nn.Module) -> None:
    for module in model.modules():
        if hasattr(module, "router_replay"):
            module.router_replay = None


def gather_replayed_router_scores(
    logits: torch.Tensor,
    indices: torch.Tensor,
    *,
    score_function: str,
    use_pre_softmax: bool = False,
    scaling_factor: float | None = None,
) -> torch.Tensor:
    """Recompute live gate weights for externally selected expert indices."""

    if score_function == "softmax":
        if use_pre_softmax:
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32).gather(
                -1, indices
            )
        else:
            scores = torch.softmax(
                logits.gather(-1, indices), dim=-1, dtype=torch.float32
            )
    elif score_function in ("sigmoid", "sqrtsoftplus"):
        dense = (
            logits.float().sigmoid()
            if score_function == "sigmoid"
            else F.softplus(logits.float()).sqrt()
        )
        scores = dense.gather(-1, indices)
        if indices.size(-1) > 1:
            scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
    else:
        raise ValueError(f"unsupported router replay score function {score_function!r}")
    if scaling_factor:
        scores = scores * scaling_factor
    return scores


__all__ = [
    "RouterReplay",
    "RouterReplayAction",
    "attach_router_replay",
    "detach_router_replay",
    "gather_replayed_router_scores",
]
