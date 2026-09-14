# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Utilities for preserving metadata on replaced parameters."""

import torch


def copy_parameter_metadata(destination: torch.Tensor, source: torch.Tensor) -> None:
    """Copy dynamically attached Megatron metadata between parameters.

    Megatron records sharding and refit metadata as public Python attributes.
    Tensor subclasses use private attributes for their storage and quantization
    implementation details; those must not leak into a replacement tensor.

    Args:
        destination: Tensor receiving the metadata.
        source: Tensor whose metadata should be copied.
    """
    for name, value in vars(source).items():
        if not name.startswith("_"):
            setattr(destination, name, value)
