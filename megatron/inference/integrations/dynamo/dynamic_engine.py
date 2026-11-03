# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Dynamo-specific dynamic inference engine type."""

import time
from collections.abc import Callable

from megatron.core.inference.disaggregation.engine import StateHandoffDynamicInferenceEngine
from megatron.core.inference.engines.dynamic_engine import EngineState


class DynamoDynamicInferenceEngine(StateHandoffDynamicInferenceEngine):
    """Dynamic inference engine with Dynamo KV/state handoff support."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._progress_callback: Callable[[], None] | None = None
        self._last_progress_time = 0.0

    def set_progress_callback(self, callback: Callable[[], None]) -> None:
        """Observe scheduling progress from the engine thread, not a timer thread."""
        self._progress_callback = callback

    def schedule_requests(self) -> int:
        count = super().schedule_requests()
        if (
            self.rank == 0
            and self.state == EngineState.RUNNING
            and self._progress_callback is not None
        ):
            now = time.monotonic()
            if now - self._last_progress_time >= 0.5:
                self._progress_callback()
                self._last_progress_time = now
        return count
