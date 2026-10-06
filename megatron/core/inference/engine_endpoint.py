# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Typed description of a running dynamic-inference engine."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from megatron.core.inference.config import PrefixCachingCoordinatorPolicy
from megatron.core.utils import experimental_api

if TYPE_CHECKING:
    from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine


@experimental_api
@dataclass(frozen=True)
class InferenceEngineCapabilities:
    """Static capabilities needed by an external inference control plane.

    This record deliberately contains no framework-specific fields. A serving
    integration can translate it to its own registration type without reaching
    into ``DynamicInferenceContext`` or depending on Megatron's CLI package.
    """

    context_length: int
    kv_cache_block_size: int
    total_kv_blocks: int
    max_num_seqs: int
    max_num_batched_tokens: int
    bos_token_id: int
    enable_prefix_caching: bool
    logical_data_parallel_size: int = 1
    prefix_caching_coordinator_policy: PrefixCachingCoordinatorPolicy | None = None
    """Routing policy for client-side hashing; None supports older endpoint records."""

    def __post_init__(self) -> None:
        integer_fields = (
            "context_length",
            "kv_cache_block_size",
            "total_kv_blocks",
            "max_num_seqs",
            "max_num_batched_tokens",
            "bos_token_id",
            "logical_data_parallel_size",
        )
        for field_name in integer_fields:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
        if not isinstance(self.enable_prefix_caching, bool):
            raise TypeError("enable_prefix_caching must be a boolean")
        if self.prefix_caching_coordinator_policy is not None and not isinstance(
            self.prefix_caching_coordinator_policy, PrefixCachingCoordinatorPolicy
        ):
            raise TypeError("prefix_caching_coordinator_policy must be a routing policy")

        positive_fields = (
            "context_length",
            "kv_cache_block_size",
            "max_num_seqs",
            "max_num_batched_tokens",
            "logical_data_parallel_size",
        )
        for field_name in positive_fields:
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.total_kv_blocks < 0:
            raise ValueError("total_kv_blocks must be non-negative")

    @classmethod
    def from_engine(
        cls, engine: DynamicInferenceEngine, *, logical_data_parallel_size: int = 1
    ) -> "InferenceEngineCapabilities":
        """Inspect a constructed dynamic engine once at registration time."""

        allocator = engine.context.kv_block_allocator
        tokenizer = engine.controller.tokenizer
        bos_token_id = next(
            (
                int(value)
                for name in ("bos", "bos_token_id", "eod")
                if (value := getattr(tokenizer, name, None)) is not None
            ),
            0,
        )
        return cls(
            context_length=int(engine.context.max_sequence_length),
            kv_cache_block_size=int(engine.context.block_size_tokens),
            # Block zero is the allocator's root and cannot hold request KV.
            total_kv_blocks=max(0, int(allocator.pool_size) - 1),
            max_num_seqs=int(engine.context.max_requests),
            max_num_batched_tokens=int(engine.context.max_tokens),
            bos_token_id=bos_token_id,
            enable_prefix_caching=bool(engine.context.enable_prefix_caching),
            logical_data_parallel_size=int(logical_data_parallel_size),
            prefix_caching_coordinator_policy=(
                engine.context.config.prefix_caching_coordinator_policy
            ),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InferenceEngineCapabilities":
        """Deserialize a capabilities mapping received across a process boundary."""

        policy = value.get("prefix_caching_coordinator_policy")
        return cls(
            context_length=value["context_length"],
            kv_cache_block_size=value["kv_cache_block_size"],
            total_kv_blocks=value["total_kv_blocks"],
            max_num_seqs=value["max_num_seqs"],
            max_num_batched_tokens=value["max_num_batched_tokens"],
            bos_token_id=value["bos_token_id"],
            enable_prefix_caching=value["enable_prefix_caching"],
            logical_data_parallel_size=value.get("logical_data_parallel_size", 1),
            prefix_caching_coordinator_policy=(
                PrefixCachingCoordinatorPolicy(policy) if policy is not None else None
            ),
        )

    def to_dict(self) -> dict[str, int | bool | str | None]:
        """Return a serialization-friendly representation."""

        return asdict(self)


@experimental_api
@dataclass(frozen=True)
class InferenceEngineEndpoint:
    """Address and capabilities of an already-running inference engine."""

    coordinator_address: str
    capabilities: InferenceEngineCapabilities

    def __post_init__(self) -> None:
        if not isinstance(self.coordinator_address, str):
            raise TypeError("coordinator_address must be a string")
        if not self.coordinator_address:
            raise ValueError("coordinator_address must be non-empty")
        if not isinstance(self.capabilities, InferenceEngineCapabilities):
            raise TypeError("capabilities must be InferenceEngineCapabilities")

    @classmethod
    def from_engine(
        cls,
        coordinator_address: str,
        engine: DynamicInferenceEngine,
        *,
        logical_data_parallel_size: int = 1,
    ) -> "InferenceEngineEndpoint":
        """Describe a running engine without transferring ownership of it."""

        return cls(
            coordinator_address=coordinator_address,
            capabilities=InferenceEngineCapabilities.from_engine(
                engine, logical_data_parallel_size=logical_data_parallel_size
            ),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InferenceEngineEndpoint":
        """Deserialize an endpoint received across a process boundary."""

        capabilities = value["capabilities"]
        if not isinstance(capabilities, Mapping):
            raise TypeError("capabilities must be a mapping")
        return cls(
            coordinator_address=value["coordinator_address"],
            capabilities=InferenceEngineCapabilities.from_dict(capabilities),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly representation."""

        return asdict(self)
