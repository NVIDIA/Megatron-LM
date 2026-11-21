# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Checkpoint save and weight export read parameters outside a forward, so the parameter all-gather
deferred by ``overlap_param_gather`` must be forced first; without overlap there is nothing pending.
"""

from __future__ import annotations

from types import SimpleNamespace

from megatron.lite.primitive.optimizers.megatron_wrap import sync_overlapped_params


class _Chunk:
    def __init__(self, overlap_param_gather: bool | None):
        if overlap_param_gather is not None:
            self.ddp_config = SimpleNamespace(overlap_param_gather=overlap_param_gather)
        self.calls = []

    def start_param_sync(self, *unused, force_sync: bool = False):
        self.calls.append(force_sync)


def test_forces_the_deferred_gather_only_under_overlap():
    overlapped, eager, plain = _Chunk(True), _Chunk(False), _Chunk(None)
    sync_overlapped_params([overlapped, eager, plain])
    assert overlapped.calls == [True]
    assert eager.calls == [] and plain.calls == []
