# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""State records for KV-cache and SSM-state imports awaiting completion."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, List

from megatron.core.inference.sampling_params import SamplingParams


@dataclass(kw_only=True)
class DeferredKvHandoff:
    """Decode handoff waiting for local cache capacity before transfer starts."""

    request_id: int
    prompt: list
    sampling_params: SamplingParams
    kv_meta: dict
    src_block_ids: List[int]
    hashes: List[int]
    num_blocks: int
    future: asyncio.Future


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
    resume_tokens: List[int] = field(default_factory=list)  # Sampled token, then MTP proposals.
    continuation_blocks: List[int] = field(default_factory=list)  # Empty KV for resume writes.
    local_error: Exception | None = None  # Exact local error, if this rank failed.
    destinations_safe: bool = True  # Whether allocated blocks may return to the pool.
    terminal_state_reported: bool = False  # Avoid repeat reports and premature block reuse.
