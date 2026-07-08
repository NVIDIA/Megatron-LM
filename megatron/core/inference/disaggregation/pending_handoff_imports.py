# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""State records for KV-cache and Mamba-state imports awaiting completion."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, List, Optional

from megatron.core.inference.sampling_params import SamplingParams


@dataclass(kw_only=True)
class PendingMambaImport:
    """Mamba state transfers attached to a pending KV-cache import."""

    handles: dict[str, Any]
    target_blocks: List[int]
    positions: List[int]


@dataclass(kw_only=True)
class PendingKvImport:
    """Decode request waiting for an asynchronous KV-cache import."""

    request_id: int
    prompt: list
    sampling_params: SamplingParams
    local_blocks: List[int]
    hashes: list
    hashes_to_register: int
    handle: Any
    future: asyncio.Future
    mamba: Optional[PendingMambaImport] = None
