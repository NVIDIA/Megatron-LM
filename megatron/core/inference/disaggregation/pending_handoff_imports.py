# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""State records for KV-cache imports awaiting completion."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, List

from megatron.core.inference.sampling_params import SamplingParams


@dataclass(kw_only=True)
class DeferredKvHandoff:
    """Decode handoff waiting for local KV-cache capacity."""

    request_id: int
    prompt: list
    sampling_params: SamplingParams
    kv_meta: dict
    src_block_ids: List[int]
    hashes: List[int]
    num_blocks: int
    future: asyncio.Future


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
    local_error: Exception | None = None  # Exact local error, if this rank failed.
    destinations_safe: bool = True  # Whether allocated blocks may return to the pool.
    terminal_state_reported: bool = False  # This rank sent its completion report.
