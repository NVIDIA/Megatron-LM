# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""HF layer clustering for vLLM layerwise-reload IPC streams."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator
from typing import Any

_LAYER_INDEX_RE = re.compile(
    r"(?:^|\.)"
    r"(?:(?P<prefix>model\.(?:language_model\.)?)?layers\.(?P<layer>\d+))"
    r"(?:\.|$)"
)
_MTP_INDEX_RE = re.compile(r"(?:^|\.)(?:(?:model\.)?mtp\.(?P<mtp>\d+))(?:\.|$)")

_EMBED_NAMES = frozenset(
    {
        "embed.weight",
        "model.embed.weight",
        "model.embed_tokens.weight",
    }
)
_NORM_NAMES = frozenset({"norm.weight", "model.norm.weight"})
_HEAD_NAMES = frozenset(
    {
        "head.weight",
        "lm_head.weight",
        "model.lm_head.weight",
    }
)


def resync_layer_cluster_key(name: str) -> tuple[int, int]:
    """Return a stable sortable key that groups checkpoint tensors by decoder layer.

    Tensors that belong to the same HF decoder layer (weights and their ``.scale``
  companions) must be loaded in one contiguous ``load_weights`` batch while the
    vLLM layerwise-reload lifecycle is active. Otherwise multiple submodules stay
    in the deferred ``online_process_loader`` state and staging memory accumulates
    across layers (observed as the 53-58 GiB receiver peak in r1-r11).
    """
    if name in _EMBED_NAMES:
        return (0, 0)
    layer_match = _LAYER_INDEX_RE.search(name)
    if layer_match is not None:
        return (1, int(layer_match.group("layer")))
    mtp_match = _MTP_INDEX_RE.search(name)
    if mtp_match is not None:
        return (2, int(mtp_match.group("mtp")))
    if name in _NORM_NAMES:
        return (3, 0)
    if name in _HEAD_NAMES:
        return (4, 0)
    return (5, hash(name.split(".", 1)[0]) & 0xFFFF)


class LayerClusterBuffer:
    """Accumulate IPC buckets and flush one HF layer cluster at a time."""

    def __init__(self, load_fn: Callable[[list[tuple[str, Any]]], None]) -> None:
        self._load_fn = load_fn
        self._buffer: list[tuple[str, Any]] = []
        self._current_key: tuple[int, int] | None = None
        self.flush_count = 0

    def ingest_bucket(self, bucket: Iterable[tuple[str, Any]]) -> None:
        for name, tensor in bucket:
            key = resync_layer_cluster_key(name)
            if self._current_key is not None and key != self._current_key:
                self._flush()
            self._current_key = key
            self._buffer.append((name, tensor))

    def finalize(self) -> None:
        self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        self._load_fn(self._buffer)
        self.flush_count += 1
        self._buffer = []


def iter_layer_clustered_weights(
    weights: Iterable[tuple[str, Any]],
) -> Iterator[tuple[str, Any]]:
    """Reorder a flat export stream so each HF layer cluster is contiguous."""
    current_key: tuple[int, int] | None = None
    buffer: list[tuple[str, Any]] = []

    def flush() -> Iterator[tuple[str, Any]]:
        nonlocal buffer, current_key
        if buffer:
            yield from buffer
            buffer = []
            current_key = None

    for name, tensor in weights:
        key = resync_layer_cluster_key(name)
        if current_key is not None and key != current_key:
            yield from flush()
        current_key = key
        buffer.append((name, tensor))
    yield from flush()


__all__ = [
    "LayerClusterBuffer",
    "iter_layer_clustered_weights",
    "resync_layer_cluster_key",
]
