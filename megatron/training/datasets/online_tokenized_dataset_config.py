# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared configuration for datasets that tokenize and align samples online."""

from dataclasses import dataclass

from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig


@dataclass
class OnlineTokenizedDatasetConfig(BlendedMegatronDatasetConfig):
    """Shared SFT/Varlen inputs; parallel sizes are derived centrally at startup."""

    data_parallel_size: int = 1
    """Data-parallel size used for dynamic-CP sample alignment."""

    context_parallel_size: int | None = None
    """Context-parallel size used for sample alignment."""

    sequence_parallel_size: int = 0
    """Tensor-parallel size when sequence parallelism is enabled, otherwise zero."""

    dynamic_context_parallel: bool = False
    """Whether sample alignment includes the data-parallel size."""

    def validate(self) -> None:
        """Check shared settings and require the tokenizer before building datasets."""
        super().validate()
        assert self.tokenizer is not None, "tokenizer must be bound before building datasets."
