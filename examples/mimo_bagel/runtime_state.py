# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Process-local runtime objects that must not enter training checkpoints."""

import random
from typing import Any

import numpy as np
import torch

_DIFFUSION_WRAPPER: Any = None
_MFU_TRACKER: Any = None
_REFERENCE_TRAINING_RNG_SEED: int | None = None


def set_diffusion_wrapper(wrapper: Any) -> None:
    """Store the rank-local diffusion wrapper outside the checkpointed args namespace."""

    global _DIFFUSION_WRAPPER
    _DIFFUSION_WRAPPER = wrapper


def get_diffusion_wrapper() -> Any:
    """Return the diffusion wrapper registered for the current process."""

    return _DIFFUSION_WRAPPER


def set_mfu_tracker(tracker: Any) -> None:
    """Store the MFU tracker outside the checkpointed args namespace."""

    global _MFU_TRACKER
    _MFU_TRACKER = tracker


def get_mfu_tracker() -> Any:
    """Return the MFU tracker registered for the current process."""

    return _MFU_TRACKER


def seed_reference_training_rng_once(seed: int) -> bool:
    """Reset process RNG once, immediately before BAGEL's first batch.

    Native BAGEL derives a rank-local training seed as ``global_seed *
    world_size + rank``. MCore must reset to that value after model construction,
    because constructing its fused representation consumes a different number of
    random values even when both implementations then load the same checkpoint.

    Returns ``True`` only for the call that performs the reset.
    """

    global _REFERENCE_TRAINING_RNG_SEED
    if seed <= 0:
        raise ValueError(f"Reference BAGEL training seed must be positive, got {seed}")
    if _REFERENCE_TRAINING_RNG_SEED is not None:
        if _REFERENCE_TRAINING_RNG_SEED != seed:
            raise RuntimeError(
                "Reference BAGEL training RNG was already seeded with "
                f"{_REFERENCE_TRAINING_RNG_SEED}, cannot reseed with {seed}"
            )
        return False

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    _REFERENCE_TRAINING_RNG_SEED = seed
    return True
