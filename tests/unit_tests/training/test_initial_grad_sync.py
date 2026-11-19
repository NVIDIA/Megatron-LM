# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU checks for the one-time allocator wrapper, without initializing CUDA."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "initial_grad_sync_under_test",
    Path(__file__).resolve().parents[3] / "megatron/training/initial_grad_sync.py",
)
assert SPEC is not None and SPEC.loader is not None
startup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(startup)


class TestInitialGradSync(unittest.TestCase):
    def test_release_once_and_preserve_custom_callback(self):
        events = []
        cuda = SimpleNamespace(
            synchronize=lambda: events.append("sync"), empty_cache=lambda: events.append("release")
        )
        calls = []

        def finalize(models, tokens, *, force=False):
            events.append("finalize")
            calls.append((models, tokens, force))
            return tokens / 2

        wrapped = startup.release_initial_grad_sync_cache(finalize)
        models = object()
        with patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=cuda)}):
            self.assertEqual(wrapped(models, 8, force=True), 4)
            self.assertEqual(wrapped(models, 6), 3)
        self.assertEqual(events, ["sync", "release", "finalize", "finalize"])
        self.assertEqual(calls, [(models, 8, True), (models, 6, False)])
        self.assertIs(wrapped.__wrapped__, finalize)
        self.assertIs(startup.release_initial_grad_sync_cache(wrapped), wrapped)

    def test_original_failure_propagates_without_releasing_again(self):
        events = []
        cuda = SimpleNamespace(
            synchronize=lambda: None, empty_cache=lambda: events.append("release")
        )

        def fail():
            raise RuntimeError("custom finalizer failure")

        wrapped = startup.release_initial_grad_sync_cache(fail)
        with patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=cuda)}):
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, "custom finalizer failure"):
                    wrapped()
        self.assertEqual(events, ["release"])


if __name__ == "__main__":
    unittest.main()
