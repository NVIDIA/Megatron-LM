# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Opaque model handle exchanged through the runtime interface."""

from __future__ import annotations

from typing import Any


class ModelHandle:
    """Hold backend state while exposing only stable distributed metadata."""

    def __init__(
        self,
        *,
        model: Any,
        optimizer: Any = None,
        lr_scheduler: Any = None,
        parallel_state: Any = None,
        config: Any = None,
        _extras: dict[str, Any] | None = None,
    ):
        self._model = model
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._parallel_state = parallel_state
        self._config = config
        self._extras = _extras or {}

    @property
    def dp_rank(self) -> int:
        return getattr(self._parallel_state, "dp_rank", 0)

    @property
    def dp_size(self) -> int:
        return getattr(self._parallel_state, "dp_size", 1)

    @property
    def dp_group(self):
        return getattr(self._parallel_state, "dp_group", None)

    @property
    def cp_range(self) -> tuple[int, int]:
        return self._extras.get("cp_range", (1, 1))

    @property
    def config(self) -> Any:
        return self._config


__all__ = ["ModelHandle"]
