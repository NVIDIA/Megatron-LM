# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compact CSA2 candidate blocks for unpacked sequences."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class CSA2CandidateBlocks:
    """Sorted local block IDs; invalid slots are -1 after the valid IDs."""

    indices: Tensor
    block_size: int

    def to_mask(self, num_keys: int) -> Tensor:
        """Expand only while scoring a consumer; the shared state retains block IDs."""
        shape = (*self.indices.shape[:-1], num_keys)
        if num_keys == 0 or self.indices.shape[-1] == 0:
            return torch.zeros(shape, dtype=torch.bool, device=self.indices.device)
        num_blocks = (num_keys + self.block_size - 1) // self.block_size
        flags = torch.zeros(
            (*self.indices.shape[:-1], num_blocks + 1), dtype=torch.bool, device=self.indices.device
        )
        # Invalid slots write to a discarded column instead of overwriting block zero.
        flags.scatter_(-1, self.indices.masked_fill(self.indices < 0, num_blocks).long(), True)
        return flags[..., :num_blocks].repeat_interleave(self.block_size, -1)[..., :num_keys]


@torch.no_grad()
def candidate_blocks_from_scores(
    scores: Tensor, visible: Tensor | int, topk_blocks: int, block_size: int
) -> CSA2CandidateBlocks:
    """Rank causal block maxima, forcing the newest visible block and stable ties.

    ``visible`` broadcasts against score rows with a trailing singleton dimension.
    Earlier block IDs win equal scores, independent of masked future capacity.
    """
    if topk_blocks <= 0 or block_size <= 0:
        raise ValueError("CSA2 candidate block count and size must be positive")
    padded = F.pad(scores, (0, -scores.shape[-1] % block_size), value=-torch.inf)
    num_blocks = padded.shape[-1] // block_size
    block_scores = padded.unflatten(-1, (num_blocks, block_size)).amax(-1)
    newest = (visible - 1) // block_size
    block_scores = block_scores.masked_fill(
        torch.arange(num_blocks, device=scores.device) == newest, torch.inf
    )
    ids = block_scores.argsort(dim=-1, descending=True, stable=True)[
        ..., : min(topk_blocks, num_blocks)
    ]
    valid = block_scores.gather(-1, ids) > -torch.inf
    sentinel = torch.iinfo(torch.int32).max
    ids = ids.masked_fill(~valid, sentinel).sort(-1).values
    return CSA2CandidateBlocks(ids.masked_fill(ids == sentinel, -1).int(), block_size)
