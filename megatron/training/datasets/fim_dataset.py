# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The FIM dataset lives in Megatron Core (``megatron.core.datasets.gpt_fim_dataset``); this module
keeps the import path it had."""

from megatron.core.datasets.gpt_fim_dataset import GPTFIMDataset, GPTFIMDatasetConfig

__all__ = ["GPTFIMDataset", "GPTFIMDatasetConfig"]
