# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.


def is_initial_checkpoint_load_without_dataloader_state(train_state: object, checkpoint: object) -> bool:
    """Return whether model weights are being loaded to start a fresh training run: the run is at
    iteration 0 and its primary load applied finetune semantics (--finetune, the
    pretrained-checkpoint fallback, --freeze-all-layers) or a pretrained checkpoint is named."""
    return train_state.iteration == 0 and (
        train_state.effective_finetune or checkpoint.pretrained_checkpoint is not None
    )


def should_strictly_load_dataloader_state(args: object) -> bool:
    """Return whether missing or invalid dataloader state should fail the run."""
    return getattr(args, "strict_dataloader_state_load", False)
