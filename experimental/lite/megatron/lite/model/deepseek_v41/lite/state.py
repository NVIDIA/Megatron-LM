# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""One-microbatch ownership of DS4.1 compressed KV tensors."""
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any


class KVSourceState:
    """Keep local owner outputs attached to their original autograd graph.

    The caller must keep the scope open through backward/recomputation. A new
    microbatch cannot reuse an old graph, and nested scopes are rejected.
    """

    def __init__(self) -> None:
        self.active = False
        self.values = {}

    @contextmanager
    def microbatch(self) -> Iterator["KVSourceState"]:
        if self.active:
            raise RuntimeError("A KV source microbatch is already active")
        self.active = True
        try:
            yield self
        finally:
            self.values.clear()
            self.active = False

    def resolve(self, layer: int, owner: int, geometry: tuple, produce: Callable[[], Any]) -> Any:
        if not self.active:
            raise RuntimeError("KV reuse requires an active microbatch scope")
        key = (owner, geometry)
        if layer == owner:
            self.values[key] = produce()
        elif key not in self.values:
            raise RuntimeError("KV owner has not produced this layout and grad mode")
        return self.values[key]
