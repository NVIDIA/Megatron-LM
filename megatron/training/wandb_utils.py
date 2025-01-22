# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from pathlib import Path

from megatron.training.global_vars import get_wandb_writer
        

def on_save_checkpoint_success(checkpoint_path: str, tracker_filename: str, save_dir: str, iteration: int) -> None:
    """Function to be called after checkpointing succeeds and checkpoint is persisted for logging it as an artifact in W&B

    Args:
        checkpoint_path (str): path of the saved checkpoint
        tracker_filename (str): path of the tracker filename for the checkpoint iteration
        save_dir (str): path of the root save folder for all checkpoints
        iteration (int): iteration of the checkpoint
    """

    wandb_writer = get_wandb_writer()

    if wandb_writer:
        metadata = {"iteration": iteration}
        artifact = wandb_writer.Artifact(Path(save_dir).stem, type="model", metadata=metadata)
        artifact.add_reference(f"file://{checkpoint_path}", checksum=False)
        artifact.add_file(tracker_filename)
        wandb_writer.run.log_artifact(artifact, aliases=[Path(checkpoint_path).stem])