# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU event-journal checks, independent of GPU training and W&B connectivity."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from examples.engram.check_events import compare_events
from examples.engram.events import EventJournal


class TestEventJournal(unittest.TestCase):
    """Events must preserve native checkpoint completion and scalar-history ordering."""

    def test_native_artifact_and_finish_need_persisted_evidence(self) -> None:
        records = [
            {
                "session": "a",
                "event": "checkpoint_saved",
                "step": 130,
                "wandb_event_source": "native_checkpoint_artifact",
            },
            {
                "session": "a",
                "event": "execution_finished",
                "step": 130,
                "wandb_event_source": "native_run_finish",
                "outcome": "exited",
            },
        ]
        tb = {(r["session"], r["event"], r["step"]): [r] for r in records}
        self.assertFalse(compare_events(records, tb, {}, [])["passed"])
        segment = {"sessions": ["a"], "checkpoint_steps": [130], "exits": [0]}
        self.assertTrue(compare_events(records, tb, {}, [segment])["passed"])
        segment["checkpoint_steps"] = [128]
        self.assertFalse(compare_events(records, tb, {}, [segment])["passed"])

    def test_events_use_metadata_without_rewinding_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            texts, updates = [], []
            tb = SimpleNamespace(add_text=lambda *args: texts.append(args), flush=lambda: None)
            wb = SimpleNamespace(
                run=True,
                config=SimpleNamespace(update=lambda value, **kwargs: updates.append(value)),
            )
            journal = EventJournal(root, None, lambda: tb, lambda: wb)
            journal.emit("checkpoint_loaded", 128, checkpoint="saved/path")
            journal.emit("checkpoint_loaded", 128, checkpoint="saved/path")
            self.assertEqual(len(texts), 1)
            self.assertEqual(len(updates), 1)
            record = json.loads((root / "events.jsonl").read_text())
            self.assertEqual(record["step"], 128)
            self.assertEqual(record["wandb_event_source"], "run_config")

    def test_last_checkpoint_is_observed_only_after_native_completion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoints = root / "checkpoints"
            old = checkpoints / "iter_0000128"
            old.mkdir(parents=True)
            (old / "metadata.json").write_text("{}")
            tracker = checkpoints / "latest_checkpointed_iteration.txt"
            tracker.write_text("128")
            texts = []
            tb = SimpleNamespace(add_text=lambda *args: texts.append(args), flush=lambda: None)
            wb = SimpleNamespace(run=None)
            journal = EventJournal(root, checkpoints, lambda: tb, lambda: wb)
            new = checkpoints / "iter_0000130"
            new.mkdir()
            (new / "metadata.json").write_text("{}")
            journal.observe_checkpoints()
            self.assertEqual(texts, [])
            tracker.write_text("130")
            journal.finish(130, "exited")
            records = [
                json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [item["event"] for item in records], ["checkpoint_saved", "execution_finished"]
            )
            self.assertEqual(records[0]["step"], 130)
            self.assertEqual(records[0]["wandb_event_source"], "native_checkpoint_artifact")

    def test_non_logging_rank_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            journal = EventJournal(root, None, lambda: None, lambda: None)
            journal.emit("mtp_weight_changed", 33080, weight=0.15)
            self.assertFalse((root / "events.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
