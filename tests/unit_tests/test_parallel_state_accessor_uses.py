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
CORE = "megatron.core"
# megatron.core exports parallel_state under its legacy name as well.
PARALLEL_STATE_NAMES = {"parallel_state", "mpu"}
FALLBACK = "global_process_group_fallback"
WARNING = "ParallelStateDeprecationWarning"
# ProcessGroupCollection.use_mpu_process_groups(), deprecated with the parallel_state accessors.
# The name is unique to it, so every attribute of that name counts, whatever its owner.
SHIM = "use_mpu_process_groups"
# Every top-level folder with Python code except tests/, which uses the global grid on purpose,
# and the CI configuration in .github/ and .gitlab/, whose scripts do not import Megatron.
# megatron/core is scanned by tools/check_process_group_usage.py instead.
SCANNED_FOLDERS = ("megatron", "examples", "tools", "tasks", "scripts", "docs")
EXCLUDED_FOLDER = REPO_ROOT / "megatron" / "core"
# Files that cannot run, and the missing module each imports: tasks/eval_utils.py still imports
# megatron.schedules, which moved to megatron.core.pipeline_parallel in 2023.
UNIMPORTABLE_FILES = {REPO_ROOT / "tasks" / "eval_utils.py": "megatron.schedules"}
# The modules that an expression can refer to, by the dotted name that importlib.import_module()
# or sys.modules looks them up by.
MODULE_NAMES = {
    "megatron": "megatron",
    CORE: CORE,
    PARALLEL_STATE: PARALLEL_STATE,
    f"{CORE}.mpu": PARALLEL_STATE,
}


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


def _attribute(node):
    """Return ``(owner, name)`` for ``owner.name`` or ``getattr(owner, "name"[, default])``."""
    if isinstance(node, ast.Attribute):
        return node.value, node.attr
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) in (2, 3)
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
    ):
        return node.args[0], node.args[1].value
    return None


def _looked_up_module(node):
    """Return the module that ``importlib.import_module("...")`` or ``sys.modules["..."]`` names
    with a constant, if it is one of MODULE_NAMES."""
    if _called_name(node) == "import_module" and node.args:
        name = node.args[0]
    elif isinstance(node, ast.Subscript) and ast.unparse(node.value) == "sys.modules":
        name = node.slice
    else:
        return None
    return MODULE_NAMES.get(name.value) if isinstance(name, ast.Constant) else None


def _assigned_pairs(target, value):
    """Yield ``(target, value)`` for an assignment, pairing the items of a tuple or list target
    with those of a tuple or list value of the same length, such as ``a, b = mpu, 1``."""
    if (
        isinstance(target, (ast.Tuple, ast.List))
        and isinstance(value, (ast.Tuple, ast.List))
        and len(target.elts) == len(value.elts)
        and not any(isinstance(item, ast.Starred) for item in target.elts + value.elts)
    ):
        for item, item_value in zip(target.elts, value.elts):
            yield from _assigned_pairs(item, item_value)
    else:
        yield target, value


def _module_references(tree, package):
    """Return a function that tells which of megatron, megatron.core and parallel_state an
    expression refers to, or None.

    A name or attribute refers to a module when an import anywhere in the file binds it to the
    module, or when an assignment binds it to an expression that refers to the module, such as
    ``grid = parallel_state``, ``self.mpu = megatron.core.parallel_state``,
    ``grid = getattr(megatron.core, "mpu")``, ``grid, size = mpu, 1`` or
    ``grid = importlib.import_module("megatron.core.parallel_state")``. An assignment in a class
    body also binds the attribute through ``self``, ``cls`` and the class name. The name
    ``megatron`` always refers to the package.

    Not followed: a module passed as an argument, returned by a function, stored in a container
    or bound to an attribute of another object, and a module that ``__import__()`` or a name
    built at run time looks up.
    """
    bound = {"megatron": {"megatron"}, CORE: set(), PARALLEL_STATE: set()}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname and alias.name in bound:
                    bound[alias.name].add(alias.asname)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                module = importlib.util.resolve_name("." * node.level + module, package)
            for alias in node.names:
                name = f"{module}.{alias.name}"
                if module == CORE and alias.name in PARALLEL_STATE_NAMES:
                    name = PARALLEL_STATE
                if name in bound:
                    bound[name].add(alias.asname or alias.name)

    def refers_to(node):
        if isinstance(node, (ast.Name, ast.Attribute)):
            spelling = ast.unparse(node)
            for module, spellings in bound.items():
                if spelling in spellings:
                    return module
        looked_up = _looked_up_module(node)
        if looked_up:
            return looked_up
        attribute = _attribute(node)
        if attribute:
            owner, name = refers_to(attribute[0]), attribute[1]
            if owner == "megatron" and name == "core":
                return CORE
            if owner == CORE and name in PARALLEL_STATE_NAMES:
                return PARALLEL_STATE
        return None

    # (target, value, the class whose body makes the assignment or None).
    assignments = []
    class_bodies = {
        id(statement): node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        for statement in node.body
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            pairs = [
                pair for target in node.targets for pair in _assigned_pairs(target, node.value)
            ]
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and node.value is not None:
            pairs = [(node.target, node.value)]
        else:
            continue
        assignments.extend((target, value, class_bodies.get(id(node))) for target, value in pairs)
    # An alias of an alias needs another pass.
    changed = True
    while changed:
        changed = False
        for target, value, owner in assignments:
            module = refers_to(value)
            if module and isinstance(target, (ast.Name, ast.Attribute)):
                spellings = {ast.unparse(target)}
                if owner and isinstance(target, ast.Name):
                    spellings |= {f"{prefix}.{target.id}" for prefix in ("self", "cls", owner)}
                if not spellings <= bound[module]:
                    bound[module] |= spellings
                    changed = True
    return refers_to


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
    name bound to the parallel_state module, by an import or an assignment, or to one of its
    deprecated functions, anywhere in the file counts in every scope. An attribute counts
    through any chain that reaches the module, such as ``megatron.core.mpu``, and so does
    ``getattr()`` with a constant name. _module_references lists the aliases it does not follow.
    """
    tree = ast.parse(source)
    refers_to = _module_references(tree, package)
    functions = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                module = importlib.util.resolve_name("." * node.level + module, package)
            for alias in node.names:
                if module == PARALLEL_STATE and alias.name == "*":
                    functions.update(deprecated)
                elif module == PARALLEL_STATE and alias.name in deprecated:
                    functions.add(alias.asname or alias.name)
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

        def _visit_attribute(self, node):
            owner, name = _attribute(node)
            if not self.covered and (
                name == SHIM or (name in deprecated and refers_to(owner) == PARALLEL_STATE)
            ):
                lines.append(node.lineno)
            self.generic_visit(node)

        visit_Attribute = _visit_attribute

        def visit_Call(self, node):
            if _attribute(node):
                self._visit_attribute(node)
            else:
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

    def test_finds_uses_through_aliases_attribute_chains_and_getattr(self):
        cases = [
            # Module aliases made by assignment, including an alias of an alias.
            (
                "from megatron.core import parallel_state\nps = parallel_state\n"
                "ps.get_tensor_model_parallel_rank()\n",
                [3],
            ),
            (
                "import megatron.core.parallel_state\nmpu = megatron.core.parallel_state\n"
                "def f():\n    return mpu.get_tensor_model_parallel_rank()\n",
                [4],
            ),
            (
                "from megatron.core import mpu\ngrid = mpu\nalias: object = grid\n"
                "alias.set_tensor_model_parallel_rank(0)\n",
                [4],
            ),
            (
                "from megatron.core import mpu\nclass Model:\n    def __init__(self):\n"
                "        self.grid = mpu\n    def rank(self):\n"
                "        return self.grid.get_tensor_model_parallel_rank()\n",
                [6],
            ),
            # Attribute chains through megatron and megatron.core.
            ("import megatron\nmegatron.core.mpu.get_tensor_model_parallel_rank()\n", [2]),
            (
                "from megatron import core\ncore.parallel_state.get_tensor_model_parallel_rank()\n",
                [2],
            ),
            ("import megatron.core as mcore\nmcore.mpu.get_tensor_model_parallel_rank()\n", [2]),
            ("from .. import core as c\nc.parallel_state.set_tensor_model_parallel_rank(0)\n", [2]),
            # getattr with a constant name, with or without a default.
            (
                "from megatron.core import parallel_state\n"
                "getattr(parallel_state, 'get_tensor_model_parallel_rank')()\n",
                [2],
            ),
            (
                "from megatron.core import mpu\n"
                "setter = getattr(mpu, 'set_tensor_model_parallel_rank', None)\n",
                [2],
            ),
            (
                "import megatron.core\ngrid = getattr(megatron.core, 'parallel_state')\n"
                "grid.get_tensor_model_parallel_rank()\n",
                [3],
            ),
            ("getattr(ProcessGroupCollection, 'use_mpu_process_groups')()\n", [1]),
            # A star import binds every deprecated function.
            ("from megatron.core.parallel_state import *\nget_tensor_model_parallel_rank()\n", [2]),
            # Class-body aliases, through self, cls and the class.
            (
                "from megatron.core import mpu\nclass Model:\n    grid = mpu\n"
                "    def rank(self):\n        return self.grid.get_tensor_model_parallel_rank()\n",
                [5],
            ),
            (
                "from megatron.core import mpu\nclass Model:\n    grid: object = mpu\n"
                "Model.grid.set_tensor_model_parallel_rank(0)\n",
                [4],
            ),
            # Tuple unpacking.
            (
                "from megatron.core import mpu\ngrid, size = mpu, 1\n"
                "grid.get_tensor_model_parallel_rank()\n",
                [3],
            ),
            # Modules looked up by name.
            (
                "import importlib\nps = importlib.import_module('megatron.core.parallel_state')\n"
                "ps.get_tensor_model_parallel_rank()\n",
                [3],
            ),
            (
                "from importlib import import_module\n"
                "import_module('megatron.core').mpu.get_tensor_model_parallel_rank()\n",
                [2],
            ),
            (
                "import sys\n"
                "sys.modules['megatron.core.parallel_state'].set_tensor_model_parallel_rank(0)\n",
                [2],
            ),
        ]
        for source, lines in cases:
            with self.subTest(source=source):
                self.assertEqual(self.uses(source), lines)

    def test_ignores_other_modules_and_names(self):
        cases = [
            "from other import parallel_state\nps = parallel_state\n"
            "ps.get_tensor_model_parallel_rank()\n",
            "import megatron.core\nmegatron.core.tensor_parallel.get_tensor_model_parallel_rank()\n",
            "from megatron import training\ntraining.mpu.get_tensor_model_parallel_rank()\n",
            "getattr(config, 'get_tensor_model_parallel_rank')()\n",
            "from megatron.core import parallel_state\n"
            "getattr(parallel_state, 'initialize_model_parallel')()\n",
            "from megatron.core import parallel_state\nps = parallel_state\n"
            "with global_process_group_fallback('f'):\n"
            "    ps.get_tensor_model_parallel_rank()\n",
            "from megatron.core import mpu\nsize, grid = mpu.initialize_model_parallel, 1\n"
            "grid.get_tensor_model_parallel_rank()\n",
            "class Model:\n    grid = other\n"
            "    def rank(self):\n        return self.grid.get_tensor_model_parallel_rank()\n",
            "import importlib\n"
            "importlib.import_module('megatron.core.utils').get_tensor_model_parallel_rank()\n",
            "import sys\nsys.modules['other.parallel_state'].get_tensor_model_parallel_rank()\n",
        ]
        for source in cases:
            with self.subTest(source=source):
                self.assertEqual(self.uses(source), [])

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
    """The code in SCANNED_FOLDERS and at the top level, outside Megatron Core, does not warn."""

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
