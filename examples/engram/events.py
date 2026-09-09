# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Experiment event records that do not rewind W&B scalar-history steps."""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


class EventJournal:
    """Mirror event metadata into JSONL and TensorBoard, using W&B config while live."""

    def __init__(
        self,
        artifacts: Path,
        checkpoint_directory: Path | None,
        tensorboard: Callable[[], Any],
        wandb: Callable[[], Any],
    ) -> None:
        self.path = artifacts / "events.jsonl"
        self.checkpoints = checkpoint_directory
        self.tensorboard = tensorboard
        self.wandb = wandb
        self.session = uuid.uuid4().hex
        self.seen = set()
        self.known_checkpoints = set(self._completed_checkpoints())

    def emit(self, kind: str, step: int, **details: Any) -> None:
        """Record a named event once per local session without adding scalar-history rows."""
        writer = self.tensorboard()
        if writer is None or (kind, step) in self.seen:
            return
        self.seen.add((kind, step))
        wandb = self.wandb()
        live_wandb = wandb is not None and getattr(wandb, "run", None) is not None
        if wandb is None:
            sink = "disabled"
        elif live_wandb:
            sink = "run_config"
        elif kind == "checkpoint_saved":
            sink = "native_checkpoint_artifact"
        elif kind == "execution_finished":
            sink = "native_run_finish"
        else:
            sink = "not_live"
        record = {
            "event": kind,
            "step": step,
            "session": self.session,
            "observed_utc": datetime.now(timezone.utc).isoformat(),
            "wandb_event_source": sink,
            **details,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        writer.add_text(f"recipe/events/{kind}", json.dumps(record), step)
        if live_wandb:
            wandb.config.update({f"recipe_event_{kind}_{step}": record}, allow_val_change=True)

    def _completed_checkpoints(self) -> dict[int, Path]:
        if self.checkpoints is None:
            return {}
        tracker = self.checkpoints / "latest_checkpointed_iteration.txt"
        if not tracker.exists():
            return {}
        try:
            latest = int(tracker.read_text().strip())
        except ValueError:
            return {}
        result = {}
        for metadata in self.checkpoints.glob("iter_*/metadata.json"):
            try:
                iteration = int(metadata.parent.name.removeprefix("iter_"))
            except ValueError:
                continue
            if iteration <= latest:
                result[iteration] = metadata.parent
        return result

    def observe_checkpoints(self) -> None:
        """Record only completed saves first observed after this process began."""
        for iteration, path in sorted(self._completed_checkpoints().items()):
            if iteration not in self.known_checkpoints:
                self.emit("checkpoint_saved", iteration, checkpoint=str(path))
                self.known_checkpoints.add(iteration)

    def finish(self, step: int, outcome: str, error_type: str | None = None) -> None:
        """Flush final save evidence after the native driver exits or raises."""
        self.observe_checkpoints()
        self.emit("execution_finished", step, outcome=outcome, error_type=error_type)
        writer = self.tensorboard()
        if writer is not None:
            writer.flush()
