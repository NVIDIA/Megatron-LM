# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""State records for KV-cache and SSM-state imports awaiting completion."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, List

from megatron.core.inference.inference_request import (
    DynamicInferenceEvent,
    DynamicInferenceEventType,
    DynamicInferenceRequest,
)
from megatron.core.inference.sampling_params import SamplingParams


@dataclass(kw_only=True)
class HandoffRequestMetadata:
    """Request identity and RL history carried across the internal prefill/decode hop."""

    uid: str
    policy_epoch: list[tuple[int, int]] | None
    kv_cache_epoch: list[tuple[int, int]] | None
    eviction_timestamps: list[float]

    @classmethod
    def from_request(cls, request: DynamicInferenceRequest) -> HandoffRequestMetadata:
        """Capture the merged source history, including prefill evictions."""
        return cls(
            uid=request.uid,
            policy_epoch=request.policy_epoch,
            kv_cache_epoch=request.kv_cache_epoch,
            eviction_timestamps=[
                event.timestamp
                for event in request.events
                if event.type == DynamicInferenceEventType.EVICT
            ],
        )

    def restore(self, request: DynamicInferenceRequest, generation_epoch: int | None) -> None:
        """Preserve source history; decode owns positions after the source's sampled token."""
        request.uid = self.uid
        for field_name in ("policy_epoch", "kv_cache_epoch"):
            source = getattr(self, field_name)
            epochs = None if source is None else [tuple(boundary) for boundary in source]
            if epochs and generation_epoch is not None and epochs[-1][1] != generation_epoch:
                # The first generated token was sampled by prefill. Decode first
                # consumes it at position len(prompt), then samples the next token.
                epochs.append((len(request.prompt_tokens), generation_epoch))
            setattr(request, field_name, epochs)
        request.events[:0] = [
            DynamicInferenceEvent(type=DynamicInferenceEventType.EVICT, timestamp=timestamp)
            for timestamp in self.eviction_timestamps
        ]


@dataclass(kw_only=True)
class DeferredKvHandoff:
    """Decode handoff waiting for local cache capacity before transfer starts."""

    request_id: int
    prompt: list
    sampling_params: SamplingParams
    kv_meta: dict
    resume_tokens: List[int]
    resume_log_probs: List[float]
    src_block_ids: List[int]
    hashes: List[int]
    num_blocks: int
    future: asyncio.Future
    offload_params: dict | None = None
    request_metadata: HandoffRequestMetadata | None = None


@dataclass(kw_only=True)
class PendingSSMImport:
    """Exact SSM state being transferred into a reserved live request slot."""

    handles: list[Any]
    live_slot: int


@dataclass(kw_only=True)
class PendingKvImport:
    """Decode request waiting for an asynchronous KV-cache import."""

    request_id: int
    prompt: list
    sampling_params: SamplingParams
    local_blocks: List[int]
    hashes: List[int]
    cached_prefix_block_count: int
    handle: Any
    future: asyncio.Future
    ssm: PendingSSMImport | None = None
    block_hash_salt: str | None = None
    request_metadata: HandoffRequestMetadata | None = None
    offload_params: dict | None = None
    resume_tokens: List[int] = field(default_factory=list)  # Sampled token, then MTP proposals.
    resume_log_probs: List[float] = field(default_factory=list)  # First sampled token only.
    continuation_blocks: List[int] = field(default_factory=list)  # Empty KV for resume writes.
    local_error: Exception | None = None  # Exact local error, if this rank failed.
    destinations_safe: bool = True  # Whether allocated blocks may return to the pool.
    terminal_state_reported: bool = False  # Avoid repeat reports and premature block reuse.
    cancel_requested: bool = False
