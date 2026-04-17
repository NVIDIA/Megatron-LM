"""Per-run logger: append-only JSONL series + rewritten meta/scalars sidecar.

One series row per call to ``append`` is written as a single JSON line to
``<log_dir>/<run_name>.jsonl``. That scales O(1) per step rather than the
O(N) of the old "dump the whole history on every call" implementation -- at
~2000 steps with per-layer dicts the old format was writing ~24 MB per step
and stalling rank 0 enough to slow training by ~15%.

Meta, top-level scalars and one-shot state (``phases``) go to a sidecar
``<log_dir>/<run_name>.meta.json`` that is atomically rewritten on every
``set()`` call. ``set()`` is infrequent (a few times per run) so the O(size)
rewrite cost is irrelevant there.

``load_runs.py`` reassembles the old ``{..meta, "series": {col: [..]}}`` shape
by reading the sidecar plus streaming the JSONL, so downstream notebooks keep
working.

Rank > 0 workers are no-ops.
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
        self.meta_path = self.log_dir / f"{run_name}.meta.json"
        self.meta_tmp_path = self.log_dir / f"{run_name}.meta.json.tmp"
        self.series_path = self.log_dir / f"{run_name}.jsonl"
        self.meta: dict[str, Any] = dict(meta)
        self.scalars: dict[str, Any] = {}
        self._active = _rank() == 0
        if self._active:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            # Truncate any stale JSONL from a previous run with the same name.
            self.series_path.write_text("")
            self._flush_meta()

    def set(self, **kwargs: Any) -> None:
        """Set top-level scalar fields (n_params, final_val_loss, phases, ...)."""
        if not self._active:
            return
        self.scalars.update(kwargs)
        self._flush_meta()

    def append(self, **columns: Any) -> None:
        """Append one row to the JSONL series. O(1) per call."""
        if not self._active:
            return
        line = json.dumps(columns, default=str, separators=(",", ":"))
        with self.series_path.open("a") as fh:
            fh.write(line)
            fh.write("\n")

    def _flush_meta(self) -> None:
        payload = {**self.meta, **self.scalars}
        with self.meta_tmp_path.open("w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        os.replace(self.meta_tmp_path, self.meta_path)
