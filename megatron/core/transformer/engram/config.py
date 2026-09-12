# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Configuration and validation for Engram memory."""

from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from megatron.core import parallel_state
from megatron.core.transformer.transformer_config import (
    TransformerConfig,
    validate_engram_layer_ids,
)

_ENGRAM_CHECKPOINT_SCHEMA_VERSION = 4
ENGRAM_TABLE_LR_MULTIPLIER = 5.0
ENGRAM_TABLE_WEIGHT_DECAY = 0.0


def _parallel_world_size(getter: Callable[[], int]) -> int:
    return getter() if parallel_state.model_parallel_is_initialized() else 1


@dataclass
class EngramConfig:
    """Engram memory configuration with Megatron runtime constraints."""

    enabled: bool = False
    hash_table_min_sizes: tuple[int, ...] = ()
    max_ngram_size: int = 3
    embedding_dim_per_ngram: int = 512
    num_hash_heads_per_ngram: int = 8
    layer_ids: tuple[int, ...] = ()
    pad_id: int | None = None
    seed: int = 0
    kernel_size: int = 4
    hidden_size: int = 0
    sequence_parallel: bool = False
    context_parallel_size: int = 1
    calculate_per_token_loss: bool = False
    table_backend: str = "local"
    init_method: Callable | None = None
    params_dtype: torch.dtype = torch.float32
    use_cpu_initialization: bool = True
    perform_initialization: bool = True

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        if self.max_ngram_size < 2:
            raise ValueError("engram_max_ngram_size must be at least 2")
        if len(self.hash_table_min_sizes) != self.max_ngram_size - 1:
            raise ValueError(
                "engram_hash_table_min_sizes must contain one size for every n-gram order "
                "from 2 through engram_max_ngram_size"
            )
        if any(size < 2 for size in self.hash_table_min_sizes):
            raise ValueError("engram_hash_table_min_sizes entries must be at least 2")
        if self.num_hash_heads_per_ngram < 1:
            raise ValueError("engram_num_hash_heads_per_ngram must be positive")
        if (
            self.embedding_dim_per_ngram <= 0
            or self.embedding_dim_per_ngram % self.num_hash_heads_per_ngram != 0
        ):
            raise ValueError(
                "engram_embedding_dim_per_ngram must be positive and divisible by "
                "engram_num_hash_heads_per_ngram"
            )
        if not self.layer_ids or len(set(self.layer_ids)) != len(self.layer_ids):
            raise ValueError("engram_layer_ids must be non-empty and unique")
        if min(self.layer_ids) < 0:
            raise ValueError("engram_layer_ids use stable non-negative memory IDs")
        if self.kernel_size < 1:
            raise ValueError("engram_kernel_size must be positive")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive when Engram is enabled")
        if self.table_backend not in ("local", "row_a2a"):
            raise ValueError(f"Unsupported Engram table backend: {self.table_backend}")
        if self.table_backend == "row_a2a":
            if self.params_dtype not in (torch.float32, torch.bfloat16):
                raise ValueError("Engram row_a2a supports only FP32 and BF16 parameters")

    @property
    def lookup_count(self) -> int:
        """Number of hash lookups per token across n-gram orders and heads."""
        return self.num_hash_heads_per_ngram * (self.max_ngram_size - 1)

    @property
    def embedding_dim(self) -> int:
        """Width of one hash-head value after splitting the n-gram embedding."""
        return self.embedding_dim_per_ngram // self.num_hash_heads_per_ngram

    @property
    def engram_hidden_size(self) -> int:
        """Concatenated Engram value width written back into the residual stream."""
        return (self.max_ngram_size - 1) * self.embedding_dim_per_ngram

    def validate_model(self, tensor_parallel_size: int) -> None:
        """Validate the table partition against the model's tensor parallel degree."""
        if not self.enabled:
            return
        validate_engram_layer_ids(self.layer_ids)
        if self.table_backend == "local" and self.embedding_dim % tensor_parallel_size != 0:
            raise ValueError("Engram embedding_dim must be divisible by tensor model parallel size")

    @classmethod
    def from_transformer_config(
        cls, config: TransformerConfig, pad_id: int | None = None
    ) -> "EngramConfig":
        """Build an EngramConfig from Megatron TransformerConfig attributes."""
        layer_ids: Sequence[int] = getattr(config, "engram_layer_ids", None) or ()
        hash_table_min_sizes = getattr(config, "engram_hash_table_min_sizes", None) or ()
        result = cls(
            enabled=bool(layer_ids),
            hash_table_min_sizes=tuple(hash_table_min_sizes),
            max_ngram_size=getattr(config, "engram_max_ngram_size", 3),
            embedding_dim_per_ngram=getattr(config, "engram_embedding_dim_per_ngram", 512),
            num_hash_heads_per_ngram=getattr(config, "engram_num_hash_heads_per_ngram", 8),
            layer_ids=tuple(layer_ids),
            pad_id=pad_id,
            seed=getattr(config, "engram_seed", 0),
            kernel_size=getattr(config, "engram_kernel_size", 4),
            hidden_size=config.hidden_size,
            sequence_parallel=config.sequence_parallel,
            context_parallel_size=config.context_parallel_size,
            calculate_per_token_loss=config.calculate_per_token_loss,
            table_backend=getattr(config, "engram_table_backend", "local"),
            init_method=config.init_method,
            params_dtype=config.params_dtype,
            use_cpu_initialization=config.use_cpu_initialization,
            perform_initialization=config.perform_initialization,
        )
        result.validate_model(config.tensor_model_parallel_size)
        return result
