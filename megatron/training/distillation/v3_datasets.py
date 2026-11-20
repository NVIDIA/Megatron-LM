# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Validation datasets for students whose training inputs come from V3 files."""

import copy
from typing import Any, Callable


class _ValidationConfig:
    """Suppress training while preserving the source config's validation boundaries.

    Core's split_matrix is now a derived property. A read-only view avoids changing
    split weights (which would move validation boundaries) or the run-owned config.
    """

    def __init__(self, source: Any):
        self.source = source

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "source"), name)

    @property
    def split_matrix(self) -> Any:
        matrix = self.source.split_matrix
        return [None, *matrix[1:]] if matrix is not None else None

    @property
    def blend_per_split(self) -> Any:
        blends = self.source.blend_per_split
        return [None, *blends[1:]] if blends is not None else None


def build_validation_datasets(
    sample_counts: list, vp_stage: int | None, built_on_rank: Callable, *, hybrid: bool = False
) -> tuple:
    """Use the native GPT provider with the training split suppressed for V3."""
    from megatron.training.datasets.providers import build_gpt_datasets
    from megatron.training.global_vars import get_run_config

    from .v3_runtime import use_hybrid_layout

    config = get_run_config()
    if hybrid:
        use_hybrid_layout()
    if not config.validation.eval_iters and not config.validation.full_validation:
        return None, None, None
    data = copy.copy(config.data)
    _, dataset, _ = data.builder_inputs()
    if not (dataset.mock or dataset.blend or dataset.blend_per_split):
        raise ValueError("V3 validation requires an explicit validation dataset or --mock-data")
    data.dataset = _ValidationConfig(dataset)
    return build_gpt_datasets(
        [0, *sample_counts[1:]], vp_stage, data_config=data, built_on_rank=built_on_rank
    )
