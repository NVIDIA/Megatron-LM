# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Tri Dao, Albert Gu.

# Some of this code was adopted from https://github.com/state-spaces/mamba/
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic-mode helpers for the SSM Triton kernels.

Kernel-config selection used to live here, but it is not specific to SSM: the
same wall-clock autotuning affects Transformer Engine's MoE permutation kernels,
and a model without Mamba needs it just as much. It moved to
:mod:`megatron.core.tuning`, which is also where the policy is installed from.

What remains here is the tile workspaces that let the SSD kernels accumulate in
a fixed order, plus re-exports of the names this module defined before the move
so out-of-tree callers keep working.
"""

import torch

from megatron.core.tuning import autotune_configs, set_deterministic_mode, use_deterministic_mode


def alloc_tile_workspace(base_shape, tile_dim, dtype, device, deterministic, *, zero_init=True):
    """Allocate buffer for deterministic per-program reductions."""
    if base_shape is None:
        return None, 0
    if deterministic:
        factory = torch.zeros if zero_init else torch.empty
        tensor = factory(*base_shape, tile_dim, device=device, dtype=dtype)
        return tensor, tensor.stride(-1)
    return torch.empty(*base_shape, device=device, dtype=dtype), 0


def finalize_tile_workspace(tensor, deterministic):
    """Finalize tile workspace."""
    if tensor is None:
        return None
    if deterministic:
        tensor = tensor.sum(dim=-1)
    return tensor


__all__ = [
    "alloc_tile_workspace",
    "autotune_configs",
    "finalize_tile_workspace",
    "set_deterministic_mode",
    "use_deterministic_mode",
]
