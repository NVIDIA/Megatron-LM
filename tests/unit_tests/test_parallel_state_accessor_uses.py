# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""In-tree code outside Megatron Core calls the deprecated parallel_state functions, and
ProcessGroupCollection.use_mpu_process_groups(), only in a warned fallback or in bootstrap, so it
emits no ParallelStateDeprecationWarning.

A call is allowed inside ``with global_process_group_fallback(...)`` (or a helper of the same
module that returns one), whose ProcessGroupFallbackWarning reports the fallback, and inside a
``with warnings.catch_warnings()`` block that ignores ParallelStateDeprecationWarning, which marks
bootstrap code. A reference in a function or lambda defined in such a block runs later, outside
it, and is not allowed. Megatron Core is exempt at run time and is checked by
tools/check_process_group_usage.py instead; tests use the global grid on purpose. Files listed
in UNIMPORTABLE_FILES, which import a module that does not exist and so cannot run, are skipped.

Stdlib only. Run directly with ``python tests/unit_tests/test_parallel_state_accessor_uses.py`` to
avoid the GPU fixtures used when pytest collects the enclosing unit-test package.
"""

import ast
import importlib.util
import pathlib
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PARALLEL_STATE_FILE = REPO_ROOT / "megatron" / "core" / "parallel_state.py"
PARALLEL_STATE = "megatron.core.parallel_state"
# megatron.core exports parallel_state under its legacy name as well.
PARALLEL_STATE_NAMES = {"parallel_state", "mpu"}
FALLBACK = "global_process_group_fallback"
WARNING = "ParallelStateDeprecationWarning"
# ProcessGroupCollection.use_mpu_process_groups(), deprecated with the parallel_state accessors.
# The name is unique to it, so every attribute of that name counts, whatever its owner.
SHIM = "use_mpu_process_groups"
# megatron/core is scanned by tools/check_process_group_usage.py; tests are not scanned.
SCANNED_FOLDERS = ("megatron", "examples", "tools", "tasks")
EXCLUDED_FOLDER = REPO_ROOT / "megatron" / "core"
# Files that cannot run, and the missing module each imports: tasks/eval_utils.py still imports
# megatron.schedules, which moved to megatron.core.pipeline_parallel in 2023.
UNIMPORTABLE_FILES = {REPO_ROOT / "tasks" / "eval_utils.py": "megatron.schedules"}


def deprecated_functions(source):
    """Return the names of the functions that parallel_state decorates as deprecated."""
    names = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                function = decorator.func if isinstance(decorator, ast.Call) else decorator
                if isinstance(function, ast.Name) and function.id == "_deprecated_accessor":
                    names.add(node.name)
    return names


def _called_name(node):
    """Return the name of the function that ``node`` calls, or None if it is not a call."""
    if not isinstance(node, ast.Call):
        return None
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


def _fallback_helpers(tree):
    """Return the module's functions whose body returns ``global_process_group_fallback(...)``."""
    helpers = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            for statement in node.body:
                if isinstance(statement, ast.Return) and _called_name(statement.value) == FALLBACK:
                    helpers.add(node.name)
    return helpers


def _ignores_the_warning(statement):
    """True for ``warnings.simplefilter("ignore", ParallelStateDeprecationWarning)`` and the
    equivalent ``filterwarnings`` call."""
    if not isinstance(statement, ast.Expr) or _called_name(statement.value) not in (
        "simplefilter",
        "filterwarnings",
    ):
        return False
    arguments = list(statement.value.args) + [keyword.value for keyword in statement.value.keywords]
    return (
        bool(arguments)
        and isinstance(arguments[0], ast.Constant)
        and arguments[0].value == "ignore"
        and any(WARNING in ast.unparse(argument) for argument in arguments[1:])
    )


def uncovered_uses(source, package, deprecated):
    """Return the lines that use a deprecated parallel_state function, or the
    ProcessGroupCollection.use_mpu_process_groups() shim, outside an allowed block.

    ``package`` is the dotted package of the scanned file, used to resolve relative imports. A
    name bound to the parallel_state module, or to one of its deprecated functions, anywhere in
    the file counts in every scope.
    """
    tree = ast.parse(source)
    modules, functions = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(a.asname for a in node.names if a.name == PARALLEL_STATE and a.asname)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                module = importlib.util.resolve_name("." * node.level + module, package)
            for alias in node.names:
                if module == PARALLEL_STATE and alias.name in deprecated:
                    functions.add(alias.asname or alias.name)
                elif module == "megatron.core" and alias.name in PARALLEL_STATE_NAMES:
                    modules.add(alias.asname or alias.name)
    covering = {FALLBACK} | _fallback_helpers(tree)
    lines = []

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.covered = False

        def visit_With(self, node):
            for item in node.items:
                self.visit(item)
            covers = any(_called_name(item.context_expr) in covering for item in node.items)
            covers = covers or (
                any(_called_name(item.context_expr) == "catch_warnings" for item in node.items)
                and any(_ignores_the_warning(statement) for statement in node.body)
            )
            outer, self.covered = self.covered, self.covered or covers
            for statement in node.body:
                self.visit(statement)
            self.covered = outer

        def _visit_deferred(self, node):
            # A function or lambda runs when it is called, which can be after the block ends.
            outer, self.covered = self.covered, False
            self.generic_visit(node)
            self.covered = outer

        visit_FunctionDef = visit_AsyncFunctionDef = visit_Lambda = _visit_deferred

        def visit_Attribute(self, node):
            if not self.covered and node.attr in deprecated:
                owner = node.value
                if (isinstance(owner, ast.Name) and owner.id in modules) or (
                    ast.unparse(owner) == PARALLEL_STATE
                ):
                    lines.append(node.lineno)
            if not self.covered and node.attr == SHIM:
                lines.append(node.lineno)
            self.generic_visit(node)

        def visit_Name(self, node):
            if not self.covered and node.id in functions:
                lines.append(node.lineno)

    Visitor().visit(tree)
    return sorted(lines)


class TestUncoveredUses(unittest.TestCase):
    """The scan finds every spelling of a deprecated call outside an allowed block."""

    PACKAGE = "megatron.training"
    DEPRECATED = {"get_tensor_model_parallel_rank", "set_tensor_model_parallel_rank"}

    def uses(self, source):
        return uncovered_uses(source, self.PACKAGE, self.DEPRECATED)

    def test_finds_uncovered_uses(self):
        cases = [
            ("from megatron.core import mpu\nmpu.get_tensor_model_parallel_rank()\n", [2]),
            (
                "from megatron.core import parallel_state as ps\n"
                "def f():\n    return ps.get_tensor_model_parallel_rank()\n",
                [3],
            ),
            (
                "from megatron.core.parallel_state import get_tensor_model_parallel_rank\n"
                "rank = get_tensor_model_parallel_rank\n",
                [2],
            ),
            (
                "from ..core.parallel_state import set_tensor_model_parallel_rank as set_rank\n"
                "set_rank(0)\n",
                [2],
            ),
            (
                "import megatron.core.parallel_state as grid\n"
                "grid.get_tensor_model_parallel_rank()\n",
                [2],
            ),
            (
                "import megatron.core.parallel_state\n"
                "megatron.core.parallel_state.get_tensor_model_parallel_rank()\n",
                [2],
            ),
            (
                "from megatron.core import mpu\n"
                "with global_process_group_fallback('f'):\n"
                "    is_first = lambda: mpu.get_tensor_model_parallel_rank() == 0\n",
                [3],
            ),
            (
                "import warnings\nfrom megatron.core import mpu\n"
                "with warnings.catch_warnings():\n"
                "    mpu.set_tensor_model_parallel_rank(0)\n",
                [4],
            ),
            (
                "from megatron.core.process_groups_config import ProcessGroupCollection\n"
                "pg_collection = ProcessGroupCollection.use_mpu_process_groups()\n",
                [2],
            ),
            (
                "import megatron.core.process_groups_config as pgc\n"
                "def f():\n"
                "    return pgc.ProcessGroupCollection.use_mpu_process_groups(['tp'])\n",
                [3],
            ),
            (
                "warn_global_process_group_fallback('f')\n"
                "pg_collection = ProcessGroupCollection.use_mpu_process_groups()\n",
                [2],
            ),
            (
                "with global_process_group_fallback('f'):\n"
                "    build = lambda: ProcessGroupCollection.use_mpu_process_groups()\n",
                [2],
            ),
        ]
        for source, lines in cases:
            with self.subTest(source=source):
                self.assertEqual(self.uses(source), lines)

    def test_allows_warned_fallbacks_and_bootstrap(self):
        cases = [
            "from megatron.core import mpu\n"
            "with global_process_group_fallback('f', 'tp_group'):\n"
            "    rank = mpu.get_tensor_model_parallel_rank()\n",
            "from megatron.core import mpu\n"
            "from megatron.core.process_groups_config import global_process_group_fallback\n"
            "def _fallback(owner):\n"
            "    return global_process_group_fallback(owner, deprecated_in='0.21')\n"
            "def f():\n"
            "    with _fallback('f'):\n"
            "        return mpu.get_tensor_model_parallel_rank()\n",
            "import warnings\nfrom megatron.core import mpu\n"
            "from megatron.core.parallel_state import ParallelStateDeprecationWarning\n"
            "with warnings.catch_warnings():\n"
            "    warnings.simplefilter('ignore', ParallelStateDeprecationWarning)\n"
            "    mpu.set_tensor_model_parallel_rank(0)\n",
            "from megatron.core import mpu\nmpu.initialize_model_parallel()\n",
            "from megatron.core import utils\nutils.get_tensor_model_parallel_rank()\n",
            "with global_process_group_fallback('f', deprecated_in='0.21'):\n"
            "    pg_collection = ProcessGroupCollection.use_mpu_process_groups(['tp'])\n",
            "import warnings\n"
            "with warnings.catch_warnings():\n"
            "    warnings.simplefilter('ignore', ParallelStateDeprecationWarning)\n"
            "    pg_collection = ProcessGroupCollection.use_mpu_process_groups()\n",
            "from megatron.core import parallel_state\n"
            "pg_collection = parallel_state.get_registered_process_groups()\n",
        ]
        for source in cases:
            with self.subTest(source=source):
                self.assertEqual(self.uses(source), [])


class TestInTreeUses(unittest.TestCase):
    """Library, training, example, tool and task code outside Megatron Core does not warn."""

    def test_parallel_state_deprecates_functions(self):
        names = deprecated_functions(PARALLEL_STATE_FILE.read_text(encoding="utf-8"))
        self.assertIn("get_tensor_model_parallel_group", names)
        self.assertIn("set_tensor_model_parallel_rank", names)
        self.assertNotIn("initialize_model_parallel", names)

    def test_in_tree_code_outside_core_does_not_warn(self):
        deprecated = deprecated_functions(PARALLEL_STATE_FILE.read_text(encoding="utf-8"))
        paths = [path for folder in SCANNED_FOLDERS for path in (REPO_ROOT / folder).rglob("*.py")]
        paths.extend(REPO_ROOT.glob("*.py"))
        violations = []
        for path in sorted(paths):
            if EXCLUDED_FOLDER in path.parents or path in UNIMPORTABLE_FILES:
                continue
            relative = path.relative_to(REPO_ROOT)
            package = ".".join(relative.parent.parts)
            source = path.read_text(encoding="utf-8")
            for line in uncovered_uses(source, package, deprecated):
                violations.append(f"{relative}:{line}")
        self.assertEqual(violations, [])

    def test_unimportable_files_import_a_missing_module(self):
        """A skipped file still imports a module that does not exist; once it imports, scan it."""
        for path, module in UNIMPORTABLE_FILES.items():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
            self.assertIn(module, imported)
            module_path = REPO_ROOT.joinpath(*module.split("."))
            self.assertFalse(module_path.with_suffix(".py").exists() or module_path.is_dir())


if __name__ == "__main__":
    unittest.main()
