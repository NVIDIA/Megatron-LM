"""Atomic per-run JSON log writer.

One file per run: ``<log_dir>/<run_name>.json``. Rewritten atomically every
update by writing to ``.tmp`` and renaming. Rank > 0 workers are no-ops.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


class JsonLogger:
    def __init__(self, log_dir: str | Path, run_name: str, meta: dict[str, Any]):
        self.log_dir = Path(log_dir)
        self.run_name = run_name
        self.path = self.log_dir / f"{run_name}.json"
        self.tmp_path = self.log_dir / f"{run_name}.json.tmp"
        self.meta: dict[str, Any] = dict(meta)
        self.scalars: dict[str, Any] = {}
        self.series: dict[str, list[Any]] = {}
        self._active = _rank() == 0
        if self._active:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._flush()

    def set(self, **kwargs: Any) -> None:
        """Set top-level scalar fields (n_params, final_val_loss, mfu, ...)."""
        if not self._active:
            return
        self.scalars.update(kwargs)
        self._flush()

    def append(self, **columns: Any) -> None:
        """Append one row across multiple series columns (step, train_loss, ...)."""
        if not self._active:
            return
        for k, v in columns.items():
            self.series.setdefault(k, []).append(v)
        self._flush()

    def _flush(self) -> None:
        payload = {
            **self.meta,
            **self.scalars,
            "series": self.series,
        }
        with self.tmp_path.open("w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        os.replace(self.tmp_path, self.path)
