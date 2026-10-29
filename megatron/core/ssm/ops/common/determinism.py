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
a fixed order, the check that batch-invariant mode runs these kernels pinned,
plus re-exports of the names this module defined before the move so out-of-tree
callers keep working.
"""

import torch

from megatron.core.tuning import (
    active_policy,
    autotune_configs,
    pins,
    set_deterministic_mode,
    use_deterministic_mode,
)

# Packages that define the SSM Triton kernels: mamba_ssm's run the training forward,
# the in-tree ones inference.
_SSM_KERNEL_MODULES = ("mamba_ssm", "megatron.core.ssm.ops")


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


def assert_ssm_kernels_pinned():
    """Refuse batch-invariant SSM layers unless the autotune policy pins their kernels.

    An unpinned Triton kernel times its candidate configs for each call shape, so a
    request's tiling, and with it the reduction order, can depend on the rest of its
    batch. Only the installed :mod:`megatron.core.tuning` policy decides that, at launch,
    for every kernel in its scope. Call at layer construction: the training initializer
    installs the policy after the model config is built but before the model is.
    """
    policy = active_policy()
    installed = "none" if policy is None else f"mode={policy.mode!r}, modules={policy.modules}"
    assert pins(*_SSM_KERNEL_MODULES), (
        "Batch invariant mode on SSM (Mamba) layers requires a Triton autotune policy that "
        "pins the mamba_ssm and megatron.core SSM kernels, so none of them chooses its "
        f"config by timing each call shape; installed policy: {installed}. Run with "
        "--deterministic-mode or --triton-autotune-mode pinned, keeping mamba_ssm and "
        "megatron.core in --triton-autotune-modules. Applications that build models "
        "without Megatron's training initializer call megatron.core.tuning.install(..., "
        "deterministic=True) or install(AutotunePolicy(mode='pinned')) before building "
        "them, as megatron/core/tuning/README.md describes. Setting the deterministic_mode "
        "config field, torch.use_deterministic_algorithms(True) or MAMBA_DETERMINISTIC=1 "
        "does not install a policy."
    )


__all__ = [
    "alloc_tile_workspace",
    "assert_ssm_kernels_pinned",
    "autotune_configs",
    "finalize_tile_workspace",
    "set_deterministic_mode",
    "use_deterministic_mode",
]
