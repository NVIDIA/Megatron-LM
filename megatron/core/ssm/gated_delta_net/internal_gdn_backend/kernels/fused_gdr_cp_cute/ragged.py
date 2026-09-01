# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Derived from flash-linear-attention; see this package's __init__.py for the
# inline MIT license notice and upstream contributor link.

"""Host helper for recurrence-neutral 64-token physical backing.

The non-WS CuTeDSL forward kernel uses an unpredicated, clamped 64-token copy
window.  Its native mask handles arbitrary logical lengths, but ``T < 64``
still needs 64 addressable rows.  This helper supplies that tiny-input backing:

* additive/tensor operands are zero-padded;
* cumulative gate operands repeat their last logical row.

Those values make the added tokens exact recurrence no-ops.  Each operand
family keeps one grow-only flat allocation.  A shorter cover is a contiguous
prefix view of that allocation, so workloads that cycle through many lengths
do not retain one tensor per distinct length.

The warp-specialized CuTeDSL and CuTe C++ production kernels handle their
partial final tiles natively and do not stage full inputs through this helper.
"""

from __future__ import annotations

import torch


class NeutralTailPadder:
    """Cache and populate device buffers padded along dimension one."""

    def __init__(self, chunk_size: int = 64) -> None:
        self.chunk_size = int(chunk_size)
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self._buffers: dict[tuple, torch.Tensor] = {}

    def padded_length(self, length: int) -> int:
        if length <= 0:
            raise ValueError("sequence length must be positive")
        return ((length + self.chunk_size - 1) // self.chunk_size) * self.chunk_size

    def pad(
        self,
        tensor: torch.Tensor,
        *,
        name: str,
        edge: bool = False,
    ) -> torch.Tensor:
        """Return ``tensor`` or a cached, neutrally padded contiguous copy."""
        length = tensor.shape[1]
        padded = self.padded_length(length)
        if padded == length:
            return tensor
        shape = (tensor.shape[0], padded, *tensor.shape[2:])
        key = (
            name,
            tensor.shape[0],
            tensor.shape[2:],
            tensor.dtype,
            tensor.device.type,
            tensor.device.index,
        )
        needed = tensor.shape[0] * padded
        for extent in tensor.shape[2:]:
            needed *= extent
        storage = self._buffers.get(key)
        if storage is None or storage.numel() < needed:
            storage = torch.empty(
                needed, dtype=tensor.dtype, device=tensor.device
            )
            self._buffers[key] = storage
        out = storage[:needed].view(shape)
        out[:, :length].copy_(tensor)
        tail = out[:, length:]
        if edge:
            tail.copy_(tensor[:, length - 1 : length].expand_as(tail))
        else:
            tail.zero_()
        return out


__all__ = ["NeutralTailPadder"]
