# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Checkpoint helpers."""

from megatron.lite.primitive.ckpt.dcp import load_training_checkpoint, save_training_checkpoint
from megatron.lite.primitive.ckpt.hf_weights import HFWeights

__all__ = [
    "HFWeights",
    "attach_model_sharded_state_dict",
    "load_training_checkpoint",
    "save_training_checkpoint",
]


def __getattr__(name: str):
    if name == "attach_model_sharded_state_dict":
        from megatron.lite.primitive.ckpt.distckpt import attach_model_sharded_state_dict

        return attach_model_sharded_state_dict
    raise AttributeError(name)
