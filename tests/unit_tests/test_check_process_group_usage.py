# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Stdlib-only tests for the global process-group usage ratchet.

Run directly with ``python tests/unit_tests/test_check_process_group_usage.py`` to avoid the
GPU fixtures used when pytest collects the enclosing unit-test package.
"""

import ast
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
SHIM_IMPORT = "from megatron.core.process_groups_config import ProcessGroupCollection\n"
SHIM_CALL = "ProcessGroupCollection.use_mpu_process_groups()"
GUARDED_IMPORT = "from megatron.core.process_groups_config import resolve_process_groups\n"
GUARDED_ARGS = "(None, owner='Model', required=('tp',))"
GUARDED_CALL = "resolve_process_groups" + GUARDED_ARGS
WARN_IMPORT = "from megatron.core.process_groups_config import warn_global_process_group_fallback\n"
WARN = "warn_global_process_group_fallback('f')"
WARNED_HIT = "f:accessor:parallel_state.get_tensor_model_parallel_group:warned"
UNWARNED_HIT = "f:accessor:parallel_state.get_tensor_model_parallel_group"

# Allowlisted reads that no fallback warning dominates, although each runs only after one: the
# read sits under an ``if`` on a value that only a warned fallback leaves None, which the check
# does not follow. Remove an entry once its read follows the warning in the same block.
UNWARNED_READS = {
    # if dp_group is None: after an if/else whose branches warn for the missing pg_collection
    # or pg_collection.dp_cp_gtp_remat.
    "megatron/core/transformer/moe/moe_logging.py": [
        "MoEMetricsTracker._sync_metrics:accessor:parallel_state.get_data_parallel_group"
    ],
    # if pp_size is None: only the branch that warns for the missing pp_group sets it to None.
    "megatron/core/transformer/multi_token_prediction.py": [
        "mtp_on_this_rank:accessor:parallel_state.get_pipeline_model_parallel_world_size"
    ],
}


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

    def test_flags_resolve_process_groups_as_guarded(self):
        cases = [
            (GUARDED_IMPORT, GUARDED_CALL),
            (
                "from megatron.core import process_groups_config as pgc\n",
                "pgc.resolve_process_groups" + GUARDED_ARGS,
            ),
            (
                "import megatron.core.process_groups_config\n",
                "megatron.core.process_groups_config.resolve_process_groups" + GUARDED_ARGS,
            ),
            (
                "from ..process_groups_config import resolve_process_groups as resolve\n",
                "resolve" + GUARDED_ARGS,
            ),
        ]
        for prefix, call in cases:
            with self.subTest(prefix=prefix):
                self.assertEqual(
                    self.hits(prefix + call), ["<module>:guarded:resolve_process_groups"]
                )

    def test_guarded_calls_are_keyed_by_enclosing_scope(self):
        self.assertEqual(
            self.hits(
                GUARDED_IMPORT
                + "class Model:\n    def __init__(self, pg_collection=None):\n        "
                + GUARDED_CALL
            ),
            ["Model.__init__:guarded:resolve_process_groups"],
        )

    def test_ignores_helpers_that_do_not_read_global_state(self):
        self.assertEqual(
            self.hits(
                "from megatron.core import process_groups_config as pgc\n"
                "pgc.require_process_groups(pg_collection, ('tp',), owner='Model')\n"
                "pgc.warn_global_process_group_fallback('Model')\n"
                "from unrelated import resolve_process_groups\n" + GUARDED_CALL
            ),
            [],
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

    def test_flags_gloo_group_accessors(self):
        for name in ("get_data_parallel_group_gloo", "get_expert_data_parallel_group_gloo"):
            identity = f"<module>:accessor:parallel_state.{name}"
            cases = [
                (IMPORT, f"parallel_state.{name}()"),
                ("from megatron.core import mpu\n", f"mpu.{name}()"),
                (f"from megatron.core.parallel_state import {name}\n", f"{name}()"),
                (f"from ..parallel_state import {name} as get_group\n", "get_group()"),
            ]
            for prefix, call in cases:
                with self.subTest(prefix=prefix, call=call):
                    self.assertEqual(self.hits(prefix + call), [identity])

    def test_reassignment_does_not_hide_an_earlier_imported_call(self):
        self.assertEqual(
            self.hits(
                "def model(config):\n"
                "    from megatron.core import parallel_state as grid\n"
                "    group = grid.get_tensor_model_parallel_group()\n"
                "    grid = config\n"
                "    return group\n"
            ),
            ["model:accessor:parallel_state.get_tensor_model_parallel_group"],
        )
        self.assertEqual(
            self.hits(
                "from megatron.core import parallel_state as grid\n"
                "group = grid.get_tensor_model_parallel_group()\n"
                "grid = None\n"
            ),
            ["<module>:accessor:parallel_state.get_tensor_model_parallel_group"],
        )

    def test_reassignment_keeps_the_import_binding_for_later_calls(self):
        # Conservative: a branch or loop may skip the reassignment, so later calls still count.
        self.assertEqual(
            self.hits(
                "def model(config):\n"
                "    from megatron.core import parallel_state as grid\n"
                "    if config.override:\n"
                "        grid = config\n"
                "    return grid.get_tensor_model_parallel_group()\n"
            ),
            ["model:accessor:parallel_state.get_tensor_model_parallel_group"],
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
            "register_model_parallel_teardown",
            "is_initialized",
            "model_parallel_is_initialized",
            "register_process_groups",
            "build_process_group_collection",
            "get_nccl_options",
        ]
        self.assertEqual(
            self.hits(IMPORT + "\n".join(f"parallel_state.{name}()" for name in names)), []
        )

    def test_registered_collection_is_a_shim_not_an_accessor(self):
        # The registry replaces the accessors outside Megatron Core and is not deprecated, but it
        # returns the same global groups: Megatron Core still takes its groups from callers.
        self.assertFalse(checker._is_deprecated_accessor("get_registered_process_groups"))
        for prefix, call in [
            (IMPORT, "parallel_state.get_registered_process_groups()"),
            ("from megatron.core import mpu\n", "mpu.get_registered_process_groups()"),
            (
                "from ..parallel_state import get_registered_process_groups as registered\n",
                "registered()",
            ),
        ]:
            with self.subTest(call=call):
                self.assertEqual(
                    self.hits(prefix + call),
                    ["<module>:shim:parallel_state.get_registered_process_groups"],
                )

    def test_counts_only_the_deprecated_parallel_state_functions_as_accessors(self):
        # A function that parallel_state does not deprecate is not an accessor or a function
        # entry, even when the accessor naming rule matches it.
        source = (REPO_ROOT / "megatron" / "core" / "parallel_state.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        retained = [
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and not any("_deprecated_accessor" in ast.unparse(d) for d in node.decorator_list)
        ]
        self.assertIn("get_registered_process_groups", retained)
        counted = [
            name
            for name in retained
            if checker._is_deprecated_accessor(name) or name in checker.DEPRECATED_FUNCTIONS
        ]
        self.assertEqual(counted, [])

    def test_forbid_helpers_stub_the_accessors_and_the_registry_getter(self):
        # The test helpers that forbid the global grid stub what _returns_global_groups selects:
        # every deprecated accessor, and the registry getter, which returns the same groups.
        source = (REPO_ROOT / "megatron" / "core" / "parallel_state.py").read_text(encoding="utf-8")
        functions = [
            node.name for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
        ]
        selected = {name for name in functions if checker._returns_global_groups(name)}
        accessors = {name for name in functions if checker._is_deprecated_accessor(name)}
        self.assertIn("get_tensor_model_parallel_group", accessors)
        self.assertEqual(selected - accessors, {"get_registered_process_groups"})
        self.assertEqual(checker.REGISTRY_GETTERS, {"get_registered_process_groups"})

    def test_flags_other_deprecated_functions(self):
        names = [
            "is_pipeline_first_stage",
            "is_pipeline_last_stage",
            "is_rank_in_embedding_group",
            "is_rank_in_position_embedding_group",
            "get_virtual_pipeline_model_parallel_rank",
            "set_virtual_pipeline_model_parallel_rank",
            "get_virtual_pipeline_model_parallel_world_size",
            "set_virtual_pipeline_model_parallel_world_size",
            "get_all_ranks",
            "set_tensor_model_parallel_world_size",
            "set_tensor_model_parallel_rank",
            "set_pipeline_model_parallel_world_size",
            "set_pipeline_model_parallel_rank",
            "set_data_parallel_rank",
            "set_expert_model_parallel_world_size",
            "set_expert_model_parallel_rank",
            "set_expert_tensor_parallel_world_size",
            "set_expert_tensor_parallel_rank",
            "get_dynamic_data_context_parallel_group_map",
            "get_dynamic_tensor_data_context_parallel_group_map",
            "get_global_memory_buffer",
            "destroy_global_memory_buffer",
            "_set_global_memory_buffer",
        ]
        for name in names:
            cases = [
                (IMPORT, f"parallel_state.{name}()"),
                ("from megatron.core import mpu\n", f"mpu.{name}()"),
                (f"from megatron.core.parallel_state import {name}\n", f"{name}()"),
            ]
            for prefix, call in cases:
                with self.subTest(prefix=prefix, call=call):
                    self.assertEqual(
                        self.hits(prefix + call), [f"<module>:function:parallel_state.{name}"]
                    )

    def test_call_after_a_fallback_warning_is_warned_not_guarded(self):
        # Only resolve_process_groups() is guarded: a direct call keeps its kind when
        # warn_global_process_group_fallback() precedes it, and is classified as warned.
        self.assertEqual(
            self.hits(
                "from megatron.core import parallel_state, process_groups_config as pgc\n"
                "def schedule(vp_size=None):\n"
                "    if vp_size is None:\n"
                "        pgc.warn_global_process_group_fallback('schedule', 'vp_size')\n"
                "        vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()\n"
            ),
            [
                "schedule:function:parallel_state.get_virtual_pipeline_model_parallel_world_size"
                ":warned"
            ],
        )

    def test_reads_that_a_fallback_warning_dominates_are_warned(self):
        cases = {
            "later in the same block": f"def f():\n    {WARN}\n    {CALL}\n",
            "in a block nested after the warning": (
                f"def f(group):\n    {WARN}\n    if group is None:\n"
                f"        for _ in range(1):\n            {CALL}\n"
            ),
            "in the same branch": (
                f"def f(group):\n    if group is None:\n        {WARN}\n        {CALL}\n"
            ),
            "in a comprehension": f"def f():\n    {WARN}\n    return [{CALL} for _ in ()]\n",
            "in a global_process_group_fallback block": (
                "from megatron.core.process_groups_config import global_process_group_fallback\n"
                f"def f():\n    with global_process_group_fallback('f'):\n        {CALL}\n"
            ),
            "after a warning through a module alias": (
                "import megatron.core.process_groups_config as pgc\n"
                f"def f():\n    pgc.{WARN}\n    {CALL}\n"
            ),
            "after a function-local relative import": (
                "from other import warn_global_process_group_fallback\n"
                "def f(group):\n    if group is None:\n"
                "        from ..process_groups_config import warn_global_process_group_fallback\n"
                f"        {WARN}\n        {CALL}\n"
            ),
        }
        for case, body in cases.items():
            with self.subTest(case=case):
                self.assertEqual(self.hits(IMPORT + WARN_IMPORT + body), [WARNED_HIT])

    def test_reads_that_no_fallback_warning_dominates_are_unwarned(self):
        cases = {
            "before the warning": f"def f():\n    {CALL}\n    {WARN}\n",
            "after a branch that warns": (
                f"def f(group):\n    if group is None:\n        {WARN}\n    {CALL}\n"
            ),
            "in the other branch": (
                f"def f(group):\n    if group is None:\n        {WARN}\n"
                f"    else:\n        {CALL}\n"
            ),
            "under a test of a value that only the warned branch sets": (
                f"def f(group):\n    size = 1\n    if group is None:\n        {WARN}\n"
                f"        size = None\n    if size is None:\n        {CALL}\n"
            ),
            "in a handler of the block that warns": (
                f"def f():\n    try:\n        {WARN}\n    except ValueError:\n        {CALL}\n"
            ),
            "in a function defined after the warning": (
                f"def f():\n    {WARN}\n    def inner():\n        {CALL}\n"
            ),
            "in a lambda defined after the warning": (
                f"def f():\n    {WARN}\n    return lambda: {CALL}\n"
            ),
            "after a warning in another function": f"def g():\n    {WARN}\ndef f():\n    {CALL}\n",
            "after a warning that is not a statement": (
                f"def f(group):\n    group or {WARN}\n    {CALL}\n"
            ),
            "after an unrelated function of the same name": (
                "from other import warn_global_process_group_fallback\n"
                f"def f():\n    {WARN}\n    {CALL}\n"
            ),
            "in a block that warns about something else": (
                f"def f():\n    with other_fallback('f'):\n        {CALL}\n"
            ),
        }
        for case, body in cases.items():
            with self.subTest(case=case):
                scope = "f.inner" if "def inner" in body else "f"
                self.assertEqual(
                    self.hits(IMPORT + WARN_IMPORT + body), [scope + UNWARNED_HIT.removeprefix("f")]
                )

    def test_guarded_calls_take_no_warned_suffix(self):
        self.assertEqual(
            self.hits(WARN_IMPORT + GUARDED_IMPORT + f"def f():\n    {WARN}\n    {GUARDED_CALL}"),
            ["f:guarded:resolve_process_groups"],
        )

    def test_counts_every_deprecated_parallel_state_function(self):
        # Megatron Core's own calls do not emit ParallelStateDeprecationWarning, so the ratchet is
        # what tracks them: it counts a call of every function that parallel_state deprecates.
        source = (REPO_ROOT / "megatron" / "core" / "parallel_state.py").read_text(encoding="utf-8")
        deprecated = sorted(
            node.name
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef)
            and any("_deprecated_accessor" in ast.unparse(d) for d in node.decorator_list)
        )
        self.assertTrue(deprecated)
        hits = self.hits(IMPORT + "\n".join(f"parallel_state.{name}()" for name in deprecated))
        self.assertEqual(sorted(hit.rpartition(".")[2] for hit in hits), deprecated)

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

    def write_baseline(self, source):
        """Make ``source`` the scanned file and record its reads as the allowlist."""
        self.source.write_text(source, encoding="utf-8")
        self.allowlist.write_text(json.dumps({"allowed": checker.scan()}), encoding="utf-8")

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
            path.write_text(IMPORT + CALL + "\n" + GUARDED_IMPORT + GUARDED_CALL, encoding="utf-8")
        self.assertEqual(self.run_check(), 0)

    def test_update_keeps_a_call_before_a_reassignment(self):
        self.source.write_text(
            "def model(config):\n"
            "    from megatron.core import parallel_state as grid\n"
            "    group = grid.get_tensor_model_parallel_group()\n"
            "    grid = config\n"
            "    return group\n",
            encoding="utf-8",
        )
        self.allowlist.write_text(json.dumps({"allowed": checker.scan()}), encoding="utf-8")
        self.assertEqual(self.run_check("--update"), 0)
        self.assertEqual(json.loads(self.allowlist.read_text())["total"], 1)
        self.assertEqual(self.run_check(), 0)

    def test_new_function_use_points_to_its_own_replacement(self):
        # A new stage predicate, virtual-pipeline or memory-buffer call has no group to accept
        # instead, so its error names the other replacements; a new group read's error does not.
        hint = "A 'function' use has a replacement of its own"
        cases = {
            "parallel_state.set_virtual_pipeline_model_parallel_rank(0)": True,
            "parallel_state.get_global_memory_buffer()": True,
            CALL: False,
        }
        for call, has_hint in cases.items():
            with self.subTest(call=call):
                self.write_baseline("")
                self.source.write_text(IMPORT + call, encoding="utf-8")
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(checker.main([]), 1)
                self.assertIn("1 new use(s) of global parallel_state", output.getvalue())
                self.assertEqual(hint in output.getvalue(), has_hint)

    def test_stats_does_not_hide_new_calls_in_check_mode(self):
        self.source.write_text(IMPORT + CALL + "; " + CALL, encoding="utf-8")
        self.assertEqual(self.run_check("--stats"), 0)
        self.assertEqual(self.run_check(), 1)

    def test_stats_reports_guarded_and_warned_reads_separately(self):
        self.source.write_text(
            IMPORT
            + SHIM_IMPORT
            + GUARDED_IMPORT
            + WARN_IMPORT
            + f"{CALL}\n{GUARDED_CALL}\ndef f():\n    {WARN}\n    {CALL}; {SHIM_CALL}\n",
            encoding="utf-8",
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(checker.main(["--stats"]), 0)
        lines = output.getvalue().splitlines()
        self.assertEqual(
            lines[0], "4 use(s) of global parallel_state across 1 file(s) in megatron/core"
        )
        self.assertEqual(
            [line.split() for line in lines[1:5]],
            [["accessor", "2"], ["function", "0"], ["shim", "1"], ["guarded", "1"]],
        )
        self.assertEqual(
            lines[5],
            "3 accessor, function and shim use(s), by whether a fallback warning precedes them",
        )
        self.assertEqual([line.split() for line in lines[6:]], [["warned", "2"], ["unwarned", "1"]])

    def test_read_that_loses_its_warning_fails(self):
        self.write_baseline(IMPORT + WARN_IMPORT + f"def f():\n    {WARN}\n    {CALL}\n")
        original = self.allowlist.read_bytes()
        self.assertEqual(json.loads(original)["allowed"], {"megatron/core/sample.py": [WARNED_HIT]})
        for case, source in {
            "warning removed": f"def f():\n    {CALL}\n",
            "warning moved into a branch": (
                f"def f(group):\n    if group is None:\n        {WARN}\n    {CALL}\n"
            ),
        }.items():
            with self.subTest(case=case):
                self.source.write_text(IMPORT + WARN_IMPORT + source, encoding="utf-8")
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(checker.main([]), 1)
                self.assertIn(
                    "ERROR: 1 allowlisted use(s) of global parallel_state in megatron/core no "
                    "longer follow a fallback warning",
                    output.getvalue(),
                )
                self.assertIn("megatron/core/sample.py:", output.getvalue())
                self.assertNotIn("new use(s)", output.getvalue())
                self.assertEqual(self.run_check("--update"), 1)
                self.assertEqual(self.allowlist.read_bytes(), original)

    def test_read_that_gains_a_warning_is_recorded(self):
        self.write_baseline(IMPORT + f"def f():\n    {CALL}\n")
        self.source.write_text(
            IMPORT + WARN_IMPORT + f"def f():\n    {WARN}\n    {CALL}\n", encoding="utf-8"
        )
        # The unwarned entry is stale until --update records the warned one instead.
        self.assertEqual(self.run_check(), 1)
        self.assertEqual(self.run_check("--update"), 0)
        self.assertEqual(
            json.loads(self.allowlist.read_text())["allowed"],
            {"megatron/core/sample.py": [WARNED_HIT]},
        )
        self.assertEqual(self.run_check(), 0)

    def test_a_reclassified_read_is_not_spent_by_a_guarded_call(self):
        # The warned read replaces its own unwarned entry, so the guarded call next to it is new.
        self.write_baseline(IMPORT + f"def f():\n    {CALL}\n")
        self.source.write_text(
            IMPORT
            + WARN_IMPORT
            + GUARDED_IMPORT
            + f"def f():\n    {WARN}\n    {CALL}\n    {GUARDED_CALL}\n",
            encoding="utf-8",
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(checker.main(["--update"]), 1)
        self.assertIn("ERROR: 1 new use(s)", output.getvalue())
        self.assertIn("f:guarded:resolve_process_groups", output.getvalue())

    def test_conversion_to_resolve_process_groups_swaps_the_entry(self):
        # One guarded call may replace the shim, or several accessor reads, of its scope.
        for read_import, reads in ((SHIM_IMPORT, SHIM_CALL), (IMPORT, CALL + "; " + CALL)):
            with self.subTest(reads=reads):
                self.write_baseline(read_import + "def build():\n    " + reads)
                self.source.write_text(
                    GUARDED_IMPORT + "def build():\n    " + GUARDED_CALL, encoding="utf-8"
                )
                # The replaced read is stale until --update records the guarded entry instead.
                self.assertEqual(self.run_check(), 1)
                self.assertEqual(self.run_check("--update"), 0)
                allowlist = json.loads(self.allowlist.read_text())
                self.assertEqual(
                    allowlist["allowed"],
                    {"megatron/core/sample.py": ["build:guarded:resolve_process_groups"]},
                )
                self.assertEqual(allowlist["total"], 1)
                self.assertEqual(self.run_check(), 0)

    def test_guarded_call_without_a_replaced_read_is_new(self):
        other = self.core / "other.py"
        read_in_build = IMPORT + "def build():\n    " + CALL
        guarded_in_build = GUARDED_IMPORT + "def build():\n    " + GUARDED_CALL
        cases = {
            "added next to a kept read": (
                read_in_build,
                {self.source: GUARDED_IMPORT + read_in_build + "; " + GUARDED_CALL},
            ),
            "replaces a read in another scope": (
                IMPORT + "def old():\n    " + CALL,
                {self.source: guarded_in_build},
            ),
            "replaces a read in another file": (
                read_in_build,
                {self.source: "", other: guarded_in_build},
            ),
            "two guarded calls replace one read": (
                read_in_build,
                {self.source: guarded_in_build + "; " + GUARDED_CALL},
            ),
            "a read replaces a guarded call": (guarded_in_build, {self.source: read_in_build}),
        }
        for case, (before, after) in cases.items():
            with self.subTest(case=case):
                other.unlink(missing_ok=True)
                self.write_baseline(before)
                original = self.allowlist.read_bytes()
                for path, source in after.items():
                    path.write_text(source, encoding="utf-8")
                self.assertEqual(self.run_check(), 1)
                self.assertEqual(self.run_check("--update"), 1)
                self.assertEqual(self.allowlist.read_bytes(), original)


class TestAllowlist(unittest.TestCase):
    """The committed allowlist grandfathers only reads that warn before they read the grid."""

    def test_allowlisted_reads_are_warned_or_guarded(self):
        path = REPO_ROOT / "tools" / "process_group_usage_allowlist.json"
        allowed = json.loads(path.read_text(encoding="utf-8"))["allowed"]
        unwarned = {}
        for rel, hits in allowed.items():
            for hit in hits:
                if checker._kind(hit) != checker.GUARDED and not hit.endswith(checker.WARNED):
                    unwarned.setdefault(rel, []).append(hit)
        if unwarned == UNWARNED_READS:
            return
        # The locations of the allowlisted files that still exist; the checker reports the others.
        locations = [
            f"{rel}:{lineno}  {identity}"
            for rel, hits in sorted(unwarned.items())
            if (REPO_ROOT / rel).is_file()
            for lineno, identity in checker._violations_in(REPO_ROOT / rel)
            if identity in hits
        ]
        self.assertEqual(
            unwarned,
            UNWARNED_READS,
            "Allowlisted reads that no fallback warning dominates:\n" + "\n".join(locations),
        )


if __name__ == "__main__":
    unittest.main()
