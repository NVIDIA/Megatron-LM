# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""ModelHandle — opaque handle returned by Runtime.build_model()."""

from __future__ import annotations

from typing import Any


class ModelHandle:
    """Opaque handle returned by Runtime.build_model().

    Only documented properties on this class are part of the public contract.
    Internal helpers may still use ``_model`` / ``_optimizer`` / ``_extras``
    while the higher-level runtime API is being stabilized.
    """

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
        ps = self._parallel_state
        if ps is None:
            return 0
        return getattr(ps, "dp_rank", 0)

    @property
    def dp_size(self) -> int:
        ps = self._parallel_state
        if ps is None:
            return 1
        return getattr(ps, "dp_size", 1)

    @property
    def dp_group(self):
        ps = self._parallel_state
        if ps is None:
            return None
        return getattr(ps, "dp_group", None)

    @property
    def cp_range(self) -> tuple[int, int]:
        return self._extras.get("cp_range", (1, 1))

    @property
    def config(self) -> Any:
        """Backend config captured when this handle was built."""
        return self._config


__all__ = ["ModelHandle"]
