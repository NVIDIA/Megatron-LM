# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

import os

import torch


def get_config_path(project_dir: str) -> str:
    """Config copy stored within retro project dir."""
    return os.path.join(project_dir, "config.json")


def get_gpt_data_dir(project_dir: str) -> str:
    """Get project-relative directory of GPT bin/idx datasets."""
    return os.path.join(project_dir, "data")


# ** Note ** : Retro's compatibility between cross attention and Flash/Fused
#   Attention is currently a work in progress. We default to returning None for
#   now.
# def get_all_true_mask(size, device):
#     return torch.full(size=size, fill_value=True, dtype=torch.bool, device=device)
def get_all_true_mask(size, device):
    return None
