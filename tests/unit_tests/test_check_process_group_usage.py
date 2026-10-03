# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Stdlib-only tests for the global process-group usage ratchet.

Run directly with ``python tests/unit_tests/test_check_process_group_usage.py`` to avoid the
GPU fixtures used when pytest collects the enclosing unit-test package.
"""

import contextlib
import importlib.util
import io
import json
import pathlib
import tempfile
import unittest
from unittest import mock

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "check_process_group_usage", REPO_ROOT / "tools" / "check_process_group_usage.py"
)
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)

IMPORT = "from megatron.core import parallel_state\n"
CALL = "parallel_state.get_tensor_model_parallel_group()"


class TestDetection(unittest.TestCase):
    """Exercise actual Python source snippets without importing Megatron or torch."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = pathlib.Path(self.directory.name) / "sample.py"

    def hits(self, source):
        self.path.write_text(source, encoding="utf-8")
        return [identity for _, identity in checker._violations_in(self.path)]

    def test_import_aliases(self):
        cases = [
            (IMPORT, CALL),
            (
                "from megatron.core import parallel_state as grid\n",
                "grid.get_tensor_model_parallel_group()",
            ),
            ("from megatron.core import mpu as grid\n", "grid.get_tensor_model_parallel_group()"),
            (
                "import megatron.core.parallel_state as grid\n",
                "grid.get_tensor_model_parallel_group()",
            ),
            (
                "import megatron.core.parallel_state\n",
                "megatron.core.parallel_state.get_tensor_model_parallel_group()",
            ),
            (
                "from megatron import core\n",
                "core.parallel_state.get_tensor_model_parallel_group()",
            ),
            (
                "from megatron.core.parallel_state import get_tensor_model_parallel_group as get_group\n",
                "get_group()",
            ),
            ("from .. import parallel_state as grid\n", "grid.get_tensor_model_parallel_group()"),
            (
                "from ..parallel_state import get_tensor_model_parallel_group as get_group\n",
                "get_group()",
            ),
        ]
        for prefix, call in cases:
            with self.subTest(prefix=prefix):
                self.assertEqual(
                    self.hits(prefix + call),
                    ["<module>:accessor:parallel_state.get_tensor_model_parallel_group"],
                )

    def test_flags_use_mpu_shim(self):
        self.assertEqual(
            self.hits(
                "from megatron.core.process_groups_config import ProcessGroupCollection as PGC\n"
                "PGC.use_mpu_process_groups()\n"
            ),
            ["<module>:shim:use_mpu_process_groups"],
        )

    def test_flags_rank_and_world_size_accessors(self):
        self.assertEqual(
            len(
                self.hits(
                    IMPORT + "parallel_state.get_tensor_model_parallel_rank()\n"
                    "parallel_state.get_data_parallel_world_size()\n"
                )
            ),
            2,
        )

    def test_flags_global_rank_lists(self):
        names = [
            "get_context_parallel_global_ranks",
            "get_gtp_weight_remat_global_ranks",
            "get_expert_gtp_weight_remat_global_ranks",
        ]
        self.assertEqual(
            len(self.hits(IMPORT + "\n".join(f"parallel_state.{name}()" for name in names))), 3
        )

    def test_function_local_import_does_not_hide_a_sibling_call(self):
        self.assertEqual(
            self.hits(
                "from megatron.core import parallel_state as grid\n"
                "def unrelated():\n    from other_backend import grid\n"
                "def model():\n    return grid.get_tensor_model_parallel_group()\n"
            ),
            ["model:accessor:parallel_state.get_tensor_model_parallel_group"],
        )

    def test_function_local_import_does_not_create_a_sibling_violation(self):
        self.assertEqual(
            self.hits(
                "from other_backend import grid\n"
                "def local():\n    from megatron.core import parallel_state as grid\n"
                "def model():\n    return grid.get_tensor_model_parallel_group()\n"
            ),
            [],
        )

    def test_function_local_import_is_detected(self):
        self.assertEqual(
            self.hits(
                "def model():\n    from megatron.core import parallel_state as grid\n"
                "    return grid.get_tensor_model_parallel_group()\n"
            ),
            ["model:accessor:parallel_state.get_tensor_model_parallel_group"],
        )

    def test_parameter_and_assignment_shadow_imported_alias(self):
        for binding in ("def model(grid):\n", "def model(config):\n    grid = config\n"):
            with self.subTest(binding=binding):
                self.assertEqual(
                    self.hits(
                        "from megatron.core import parallel_state as grid\n"
                        + binding
                        + "    return grid.get_tensor_model_parallel_group()\n"
                    ),
                    [],
                )

    def test_class_import_does_not_hide_a_method_global(self):
        self.assertEqual(
            self.hits(
                "from megatron.core import parallel_state as grid\n"
                "class Model:\n    from other_backend import grid\n"
                "    def forward(self):\n        return grid.get_tensor_model_parallel_group()\n"
            ),
            ["Model.forward:accessor:parallel_state.get_tensor_model_parallel_group"],
        )

    def test_ignores_retained_surface(self):
        names = [
            "initialize_model_parallel",
            "destroy_model_parallel",
            "is_initialized",
            "get_virtual_pipeline_model_parallel_rank",
            "get_virtual_pipeline_model_parallel_world_size",
            "get_global_memory_buffer",
            "get_nccl_options",
            "get_all_ranks",
        ]
        self.assertEqual(
            self.hits(IMPORT + "\n".join(f"parallel_state.{name}()" for name in names)), []
        )

    def test_ignores_unrelated_imports_and_getters(self):
        self.assertEqual(
            self.hits(
                "import unrelated.parallel_state as parallel_state\n" + CALL + "\n"
                "from unrelated.parallel_state import get_data_parallel_group\n"
                "get_data_parallel_group()\n"
                "config.get_thing_group()\n"
                "some_object.get_rank()\n"
                "some_object.use_mpu_process_groups()\n"
            ),
            [],
        )

    def test_line_shifts_do_not_change_identity(self):
        source = IMPORT + "def f():\n    " + CALL + "\n"
        self.assertEqual(self.hits(source), self.hits("# inserted comment\n\n" + source))

    def test_enclosing_scopes_distinguish_calls(self):
        self.assertEqual(
            self.hits(IMPORT + "class Model:\n    async def forward(self):\n        " + CALL),
            ["Model.forward:accessor:parallel_state.get_tensor_model_parallel_group"],
        )

    def test_same_line_duplicates_are_counted(self):
        hits = self.hits(IMPORT + CALL + "; " + CALL)
        self.assertEqual(len(hits), 2)
        self.assertEqual(
            checker._difference({"file": hits}, {"file": hits[:1]}), {"file": hits[:1]}
        )

    def test_moving_call_to_another_function_is_new(self):
        before = self.hits(IMPORT + "def old():\n    " + CALL)
        after = self.hits(IMPORT + "def new():\n    " + CALL)
        self.assertEqual(checker._difference({"file": after}, {"file": before}), {"file": after})

    def test_invalid_python_fails_closed(self):
        with self.assertRaises(SyntaxError):
            self.hits(IMPORT + "def broken(:\n")


class TestRatchet(unittest.TestCase):
    """Exercise scan and CLI update behavior against isolated miniature repositories."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = pathlib.Path(self.directory.name)
        self.core = self.root / "megatron" / "core"
        self.core.mkdir(parents=True)
        self.source = self.core / "sample.py"
        self.source.write_text(IMPORT + CALL + "\n", encoding="utf-8")
        self.allowlist = self.root / "allowlist.json"
        patcher = mock.patch.multiple(
            checker, REPO_ROOT=self.root, SCAN_ROOT=self.core, ALLOWLIST=self.allowlist
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.allowlist.write_text(json.dumps({"allowed": checker.scan()}), encoding="utf-8")

    def run_check(self, *argv):
        with contextlib.redirect_stdout(io.StringIO()):
            return checker.main(list(argv))

    def test_matching_baseline_passes(self):
        self.assertEqual(self.run_check(), 0)

    def test_line_shift_passes_without_update(self):
        self.source.write_text("\n# new comment\n" + IMPORT + CALL, encoding="utf-8")
        self.assertEqual(self.run_check(), 0)

    def test_added_call_fails_and_update_leaves_baseline_unchanged(self):
        original = self.allowlist.read_bytes()
        self.source.write_text(IMPORT + CALL + "; " + CALL, encoding="utf-8")
        self.assertEqual(self.run_check(), 1)
        self.assertEqual(self.run_check("--update"), 1)
        self.assertEqual(self.allowlist.read_bytes(), original)

    def test_removal_requires_update_then_passes(self):
        self.source.write_text("", encoding="utf-8")
        self.assertEqual(self.run_check(), 1)
        self.assertEqual(self.run_check("--update"), 0)
        self.assertEqual(json.loads(self.allowlist.read_text())["total"], 0)
        self.assertEqual(self.run_check(), 0)

    def test_new_scope_cannot_spend_a_removed_call(self):
        self.source.write_text(IMPORT + "def other():\n    " + CALL, encoding="utf-8")
        self.assertEqual(self.run_check("--update"), 1)

    def test_missing_allowlist_cannot_bootstrap_new_calls(self):
        self.allowlist.unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_check("--update")

    def test_invalid_python_cannot_shrink_baseline(self):
        original = self.allowlist.read_bytes()
        self.source.write_text("def invalid(:\n", encoding="utf-8")
        with self.assertRaises(SyntaxError):
            self.run_check("--update")
        self.assertEqual(self.allowlist.read_bytes(), original)

    def test_only_core_and_nonexempt_files_are_scanned(self):
        for rel in (*checker.EXEMPT, "megatron/training/example.py", "tests/example.py"):
            path = self.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(IMPORT + CALL, encoding="utf-8")
        self.assertEqual(self.run_check(), 0)

    def test_stats_does_not_hide_new_calls_in_check_mode(self):
        self.source.write_text(IMPORT + CALL + "; " + CALL, encoding="utf-8")
        self.assertEqual(self.run_check("--stats"), 0)
        self.assertEqual(self.run_check(), 1)


class TestCommittedBaseline(unittest.TestCase):
    """Check the production tree and committed baseline together."""

    def test_matches_current_tree(self):
        found = checker.scan()
        allowed = checker._load_allowlist()
        self.assertEqual(checker._difference(found, allowed), {})
        self.assertEqual(checker._difference(allowed, found), {})
        total = json.loads(checker.ALLOWLIST.read_text())["total"]
        self.assertEqual(total, sum(map(len, found.values())))


if __name__ == "__main__":
    unittest.main()
