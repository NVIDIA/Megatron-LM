# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import os
from pathlib import Path
from typing import Tuple

from megatron.training.utils import print_rank_last


wandb_id_prefix = "wandb_id."

def load_wandb_id_from_local(directory):
    if not os.path.exists(directory):
        return None
    for file in os.listdir(directory):
        if file.startswith(wandb_id_prefix):
            return file.replace(wandb_id_prefix, "")

    return None

def save_wandb_id_to_local(directory, run_id):
    if os.path.isdir(directory):
        for file in os.listdir(directory):
            if file.startswith(wandb_id_prefix):
                os.remove(os.path.join(directory, file))

    os.mknod(os.path.join(directory, f"{wandb_id_prefix}{run_id}"))


def _get_wandb_artifact_tracker_filename(save_dir: str) -> Path:
    """Wandb artifact tracker file records the latest artifact wandb entity and project"""
    return Path(save_dir) / "latest_wandb_artifact_path.txt"


def _get_artifact_name_and_version(save_dir: Path, checkpoint_path: Path) -> Tuple[str, str]:
    return save_dir.stem, checkpoint_path.stem


def on_save_checkpoint_success(wandb_writer, checkpoint_path: str, tracker_filename: str, save_dir: str, iteration: int) -> None:
    """Function to be called after checkpointing succeeds and checkpoint is persisted for logging it as an artifact in W&B

    Args:
        checkpoint_path (str): path of the saved checkpoint
        tracker_filename (str): path of the tracker filename for the checkpoint iteration
        save_dir (str): path of the root save folder for all checkpoints
        iteration (int): iteration of the checkpoint
    """
    if wandb_writer:
        try:
            metadata = {"iteration": iteration}
            artifact_name, artifact_version = _get_artifact_name_and_version(Path(save_dir), Path(checkpoint_path))
            artifact = wandb_writer.Artifact(artifact_name, type="model", metadata=metadata)
            # wandb's artifact.add_reference requires absolute paths
            checkpoint_path = str(Path(checkpoint_path).resolve())
            artifact.add_reference(f"file://{checkpoint_path}", checksum=False)
            artifact.add_file(tracker_filename)
            wandb_writer.run.log_artifact(artifact, aliases=[artifact_version])
            wandb_tracker_filename = _get_wandb_artifact_tracker_filename(save_dir)
            wandb_tracker_filename.write_text(f"{wandb_writer.run.entity}/{wandb_writer.run.project}")
        except Exception:
            print_rank_last(f"  failed to save checkpoint {checkpoint_path} in wandb")


def on_load_checkpoint_success(wandb_writer, checkpoint_path: str, load_dir: str) -> None:
    """Function to be called after succesful loading of a checkpoint, for aggregation and logging it to W&B

    Args:
        wandb_writer: W&B writer
        checkpoint_path (str): path of the loaded checkpoint
        load_dir (str): path of the root save folder for all checkpoints
        iteration (int): iteration of the checkpoint
    """

    if wandb_writer:
        try:
            artifact_name, artifact_version = _get_artifact_name_and_version(Path(load_dir), Path(checkpoint_path))
            wandb_tracker_filename = _get_wandb_artifact_tracker_filename(load_dir)
            artifact_path = ""
            if wandb_tracker_filename.is_file():
                artifact_path = wandb_tracker_filename.read_text().strip()
                artifact_path = f"{artifact_path}/"
            wandb_writer.run.use_artifact(f"{artifact_path}{artifact_name}:{artifact_version}")
        except Exception:
            print_rank_last(f"  failed to find checkpoint {checkpoint_path} in wandb")
