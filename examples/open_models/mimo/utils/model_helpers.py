# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""
Utility helpers for mimo models.
"""

import torch
from megatron.core import dist_checkpointing


def load_submodule_ckpt(module: torch.nn.Module, ckpt_dir: str):
    """Load *ckpt_dir* into *module* using Megatron distributed-checkpointing."""

    # 1) Ask for tensors using a `module.` prefix so they match checkpoint keys.
    sharded_sd_with_prefix = module.sharded_state_dict(prefix="module.")

    # Remove fp8 extra_state tensors – they may not exist in older checkpoints.
    for k in list(sharded_sd_with_prefix.keys()):
        if "extra_state" in k:
            del sharded_sd_with_prefix[k]

    # 2) Wrap it under a root key just as in user snippet; this becomes the state
    #    dict returned by `load` so we can easily strip the prefix afterwards.
    wrapper_sd = dict(state_dict=sharded_sd_with_prefix)
    loaded = dist_checkpointing.load(
        sharded_state_dict=wrapper_sd,
        checkpoint_dir=ckpt_dir,
    )
    # 3) Remove the prefix and push into the module.
    cleaned = {k.removeprefix("module."): v for k, v in loaded["state_dict"].items()}

    incompatible = module.load_state_dict(cleaned, strict=False)
    unexpected = [k for k in incompatible.unexpected_keys if "extra_state" not in k]
    missing = [k for k in incompatible.missing_keys if "extra_state" not in k]
    if unexpected or missing:
        raise RuntimeError(
            f"load_state_dict had unexpected mismatch. Missing: {missing}, Unexpected: {unexpected}"
        )
