# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Wire payload construction and validation for prefill/decode handoff."""

from __future__ import annotations

from typing import Any, Sequence, Tuple


def make_submit_request_with_kv_message(
    header_value: int,
    request_id: int,
    prompt: Any,
    sampling_params: dict,
    kv_meta: dict,
    src_block_ids: list,
) -> list:
    """Build a ``SUBMIT_REQUEST_WITH_KV`` message."""

    return [header_value, int(request_id), prompt, sampling_params, kv_meta, list(src_block_ids)]


def parse_submit_request_with_kv_fields(fields: Sequence[Any]) -> Tuple[Any, ...]:
    """Validate and unpack fields following ``SUBMIT_REQUEST_WITH_KV``."""

    if len(fields) != 5:
        raise ValueError(f"SUBMIT_REQUEST_WITH_KV payload must have 5 fields, got {len(fields)}")
    return tuple(fields)


def make_release_kv_message(header_value: int, request_id: int) -> list:
    """Build a ``RELEASE_KV`` message."""

    return [header_value, int(request_id)]
