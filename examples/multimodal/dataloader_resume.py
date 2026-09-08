# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.


def is_initial_checkpoint_load_without_dataloader_state(args: object) -> bool:
    """Return whether model weights are being loaded to start a fresh training run."""
    return getattr(args, "iteration", None) == 0 and (
        getattr(args, "finetune", False)
        or getattr(args, "pretrained_checkpoint", None) is not None
    )


def should_strictly_load_dataloader_state(args: object) -> bool:
    """Return whether missing or invalid dataloader state should fail the run."""
    return getattr(args, "strict_dataloader_state_load", False)
