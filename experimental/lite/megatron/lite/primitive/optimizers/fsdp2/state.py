"""Optimizer state movement helpers for the FSDP2 primitive."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from megatron.lite.primitive.optimizers.fsdp2.adamw import (
    dtensor_from_local,
    is_dtensor_like,
    iter_torch_optimizers,
)


@dataclass
class OffloadedStateEntry:
    device: torch.device
    is_dtensor: bool = False
    device_mesh: Any | None = None
    placements: Any | None = None


def move_optimizer_state_to_cpu(
    optimizer: Any,
    offloaded: dict[tuple[int, str], OffloadedStateEntry],
    *,
    include_dtensor_state: bool,
) -> None:
    for child in iter_torch_optimizers(optimizer):
        state = getattr(child, "state", None)
        if not isinstance(state, dict):
            continue
        for param, param_state in state.items():
            if not isinstance(param_state, dict):
                continue
            for key, value in list(param_state.items()):
                if is_dtensor_like(value):
                    if not include_dtensor_state:
                        continue
                    local_value = value.to_local()
                    if not isinstance(local_value, torch.Tensor) or not local_value.is_cuda:
                        continue
                    offloaded[(id(param), key)] = OffloadedStateEntry(
                        device=local_value.device,
                        is_dtensor=True,
                        device_mesh=value.device_mesh,
                        placements=value.placements,
                    )
                    param_state[key] = local_value.detach().to("cpu")
                    continue

                if not isinstance(value, torch.Tensor) or not value.is_cuda:
                    continue
                offloaded[(id(param), key)] = OffloadedStateEntry(device=value.device)
                param_state[key] = value.detach().to("cpu")


def move_offloaded_optimizer_state_to_device(
    optimizer: Any,
    offloaded: dict[tuple[int, str], OffloadedStateEntry],
) -> None:
    if not offloaded:
        return
    remaining = dict(offloaded)
    for child in iter_torch_optimizers(optimizer):
        state = getattr(child, "state", None)
        if not isinstance(state, dict):
            continue
        for param, param_state in state.items():
            if not isinstance(param_state, dict):
                continue
            for key, entry in list(remaining.items()):
                param_id, state_key = key
                if param_id != id(param) or state_key not in param_state:
                    continue
                value = param_state[state_key]
                if isinstance(value, torch.Tensor) and not is_dtensor_like(value):
                    device_value = value.to(entry.device, non_blocking=True)
                    if entry.is_dtensor:
                        param_state[state_key] = dtensor_from_local(
                            device_value,
                            entry.device_mesh,
                            entry.placements,
                        )
                    else:
                        param_state[state_key] = device_value
                remaining.pop(key, None)
    for key in set(offloaded) - set(remaining):
        offloaded.pop(key, None)


__all__ = [
    "OffloadedStateEntry",
    "move_offloaded_optimizer_state_to_device",
    "move_optimizer_state_to_cpu",
]
