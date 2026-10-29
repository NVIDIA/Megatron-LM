# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""In-tree code outside Megatron Core calls the deprecated parallel_state functions, and
ProcessGroupCollection.use_mpu_process_groups(), only in a warned fallback, so a stock run emits
no ParallelStateDeprecationWarning, and it never touches the private state of parallel_state.

A call is allowed inside ``with global_process_group_fallback(...)`` (or a helper of the same
module that returns one), whose ProcessGroupFallbackWarning reports the fallback, and inside a
``with warnings.catch_warnings()`` block that ignores ParallelStateDeprecationWarning after the
code has warned on its own: a statement that is only a call of
``warnings.warn(..., ParallelStateDeprecationWarning)`` (the category by position or keyword) or of
``warn_global_process_group_fallback()`` comes before the ``with``, in the block
that holds it or in a block that contains that block, of the same function (of the module for
module-level code). A warning inside an earlier branch, loop or ``with``, or inside the
``catch_warnings()`` block, does not count. A reference in a function or lambda defined in such a
block runs later, outside it, and is not allowed. A read, write or deletion of a private
(``_``-prefixed) attribute of parallel_state is never allowed; the deprecated
``_set_global_memory_buffer()`` counts as a deprecated function instead. Megatron Core is exempt
at run time and is checked by tools/check_process_group_usage.py instead; tests use the global
grid on purpose. Files listed in UNIMPORTABLE_FILES, which import a module that does not exist
and so cannot run, are skipped.

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
# The fallback warning that a call made before a block that ignores WARNING can report instead.
FALLBACK_WARNING = "warn_global_process_group_fallback"
WARNING = "ParallelStateDeprecationWarning"
# ProcessGroupCollection.use_mpu_process_groups(), deprecated with the parallel_state accessors.
# The name is unique to it, so every attribute of that name counts, whatever its owner.
SHIM = "use_mpu_process_groups"
# Every top-level folder with Python code except tests/, .github/ and .gitlab/, plus the
# functional tests in tests/, which run Megatron end to end like the entry points. The rest of
# tests/ is skipped: the unit tests use the global grid on purpose, and tests/test_utils and
# tests/performance_tests, like the CI configuration in .github/ and .gitlab/, hold scripts that
# do not import Megatron. megatron/core is scanned by tools/check_process_group_usage.py instead.
SCANNED_FOLDERS = (
    "megatron",
    "examples",
    "tools",
    "tasks",
    "scripts",
    "docs",
    "tests/functional_tests",
)
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


def _last_name(node):
    """Return ``name`` for the expression ``name`` or ``owner.name``, or None."""
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def _called_name(node):
    """Return the name of the function that ``node`` calls, or None if it is not a call."""
    return _last_name(node.func) if isinstance(node, ast.Call) else None


def _constant_string(node):
    """Return the value of a string constant, or None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
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


def _attribute(node, functions=("getattr",)):
    """Return ``(owner, name)`` for ``owner.name``, or for a call of one of ``functions`` that
    names the attribute with a constant, such as ``getattr(owner, "name"[, default])``."""
    if isinstance(node, ast.Attribute):
        return node.value, node.attr
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in functions
        and len(node.args) in (2, 3)
        and _constant_string(node.args[1])
    ):
        return node.args[0], node.args[1].value
    return None


def _looked_up_module(node):
    """Return the module that ``importlib.import_module(name[, package])`` or
    ``sys.modules["..."]`` names with constants, if it is one of MODULE_NAMES."""
    if _called_name(node) == "import_module":
        keywords = {keyword.arg: keyword.value for keyword in node.keywords}
        name = _constant_string(node.args[0] if node.args else keywords.get("name"))
        package = node.args[1] if len(node.args) > 1 else keywords.get("package")
        try:
            name = importlib.util.resolve_name(name, _constant_string(package)) if name else None
        except ImportError:
            # A relative name without a package, or beyond the top-level package.
            return None
    elif isinstance(node, ast.Subscript) and ast.unparse(node.value) == "sys.modules":
        name = _constant_string(node.slice)
    else:
        return None
    return MODULE_NAMES.get(name)


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
    ``grid = getattr(megatron.core, "mpu")``, ``grid, size = mpu, 1``,
    ``grid = mpu if lazy else None`` or
    ``grid = importlib.import_module("megatron.core.parallel_state")``. An assignment in a class
    body also binds the attribute through ``self``, ``cls`` and the class name. The name
    ``megatron`` always refers to the package, and after a star import of ``megatron.core`` the
    name ``parallel_state`` refers to the module (``megatron.core.__all__`` does not export
    ``mpu``).

    Not followed: a module passed as an argument (a parameter default included), returned by a
    function, stored in a container, bound by ``for`` or ``with ... as``, or bound to an attribute
    of another object, an import in a class body that methods reach through ``self``, and a module
    that ``__import__()`` or a name built at run time looks up.
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
                if module == CORE and alias.name == "*":
                    bound[PARALLEL_STATE].add("parallel_state")
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
        # Either side of a conditional expression, and any operand of ``and`` and ``or``, can be
        # its value.
        if isinstance(node, (ast.IfExp, ast.BoolOp)):
            options = (node.body, node.orelse) if isinstance(node, ast.IfExp) else node.values
            return next(filter(None, map(refers_to, options)), None)
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


def _is_private(name):
    """True for a private attribute such as ``_DATA_PARALLEL_GROUP``, but not a dunder."""
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def _warns(statement, warn_functions):
    """True for ``warnings.warn(..., ParallelStateDeprecationWarning, ...)``, with the category
    given by position or keyword, and for ``warn_global_process_group_fallback(...)``.

    ``warn_functions`` holds the spellings of ``warnings.warn`` that the file's imports bind, such
    as ``warnings.warn`` or ``warn``; another function named ``warn``, such as a logger's, does
    not emit the warning.
    """
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return False
    call = statement.value
    if _called_name(call) == FALLBACK_WARNING:
        return True
    if ast.unparse(call.func) not in warn_functions:
        return False
    categories = call.args[1:2] + [k.value for k in call.keywords if k.arg == "category"]
    return any(_last_name(category) == WARNING for category in categories)


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
    ProcessGroupCollection.use_mpu_process_groups() shim, outside an allowed block, and those that
    read, write or delete a private attribute of parallel_state anywhere.

    ``package`` is the dotted package of the scanned file, used to resolve relative imports. A
    name bound to the parallel_state module, by an import or an assignment, or to one of its
    deprecated functions, anywhere in the file counts in every scope. An attribute counts
    through any chain that reaches the module, such as ``megatron.core.mpu``, and so does
    ``getattr()``, ``setattr()`` or ``delattr()`` with a constant name. _module_references lists
    the aliases it does not follow.
    """
    tree = ast.parse(source)
    refers_to = _module_references(tree, package)
    functions = set()
    # Names bound to a private attribute of parallel_state.
    private = set()
    # The spellings of warnings.warn that the imports bind.
    warn_functions = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "warnings":
                    warn_functions.add(f"{alias.asname or alias.name}.warn")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                module = importlib.util.resolve_name("." * node.level + module, package)
            for alias in node.names:
                if module == "warnings" and alias.name == "warn":
                    warn_functions.add(alias.asname or alias.name)
                elif module == PARALLEL_STATE and alias.name == "*":
                    # parallel_state has no __all__, so a star import binds its public names only.
                    functions.update(name for name in deprecated if not _is_private(name))
                elif module == PARALLEL_STATE and alias.name in deprecated:
                    # A deprecated function is one even if private: _set_global_memory_buffer.
                    functions.add(alias.asname or alias.name)
                elif module == PARALLEL_STATE and _is_private(alias.name):
                    private.add(alias.asname or alias.name)
    covering = {FALLBACK} | _fallback_helpers(tree)
    lines = []

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.covered = False
            # Whether an earlier statement of the function, in the current block or in a block
            # that contains it, has warned (_warns).
            self.warned = False

        def generic_visit(self, node):
            # Visit each list of statements, such as the branches of an ``if``, as a block.
            for _, value in ast.iter_fields(node):
                if isinstance(value, list):
                    self._visit_block(value)
                elif isinstance(value, ast.AST):
                    self.visit(value)

        def _visit_block(self, statements):
            # A warning covers the statements after it in its block, not those after the block.
            outer = self.warned
            for statement in statements:
                if isinstance(statement, ast.AST):
                    self.visit(statement)
                    self.warned = self.warned or _warns(statement, warn_functions)
            self.warned = outer

        def visit_With(self, node):
            for item in node.items:
                self.visit(item)
            covers = any(_called_name(item.context_expr) in covering for item in node.items)
            # Ignoring the warning hides the read unless the code has warned before the block.
            covers = covers or (
                self.warned
                and any(_called_name(item.context_expr) == "catch_warnings" for item in node.items)
                and any(_ignores_the_warning(statement) for statement in node.body)
            )
            outer, self.covered = self.covered, self.covered or covers
            self._visit_block(node.body)
            self.covered = outer

        def _visit_deferred(self, node):
            # A function or lambda runs when it is called, which can be after the block ends.
            outer = self.covered, self.warned
            self.covered = self.warned = False
            self.generic_visit(node)
            self.covered, self.warned = outer

        visit_FunctionDef = visit_AsyncFunctionDef = visit_Lambda = _visit_deferred

        def _visit_attribute(self, node, attribute):
            owner, name = attribute
            module = refers_to(owner)
            if module == PARALLEL_STATE and _is_private(name) and name not in deprecated:
                # The private state of parallel_state is not an interface, even in a fallback.
                lines.append(node.lineno)
            elif not self.covered and (
                name == SHIM or (name in deprecated and module == PARALLEL_STATE)
            ):
                lines.append(node.lineno)
            self.generic_visit(node)

        def visit_Attribute(self, node):
            self._visit_attribute(node, _attribute(node))

        def visit_Call(self, node):
            # getattr(), setattr() and delattr() with a constant name use the attribute.
            attribute = _attribute(node, ("getattr", "setattr", "delattr"))
            if attribute:
                self._visit_attribute(node, attribute)
            else:
                self.generic_visit(node)

        def visit_Name(self, node):
            if node.id in private or (not self.covered and node.id in functions):
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
            # setattr() and delattr() with a constant name.
            (
                "from megatron.core import mpu\n"
                "setattr(mpu, 'set_tensor_model_parallel_rank', f)\n",
                [2],
            ),
            # A star import binds every public deprecated function, and megatron.core's binds
            # parallel_state.
            ("from megatron.core.parallel_state import *\nget_tensor_model_parallel_rank()\n", [2]),
            ("from megatron.core import *\nparallel_state.get_tensor_model_parallel_rank()\n", [2]),
            # Either side of a conditional expression, and any operand of ``and`` and ``or``.
            (
                "from megatron.core import mpu\ngrid = mpu if lazy else None\n"
                "grid.get_tensor_model_parallel_rank()\n",
                [3],
            ),
            (
                "from megatron.core import mpu\n(None or mpu).set_tensor_model_parallel_rank(0)\n",
                [2],
            ),
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
                "import importlib\nimportlib.import_module(name='megatron.core.parallel_state')"
                ".get_tensor_model_parallel_rank()\n",
                [2],
            ),
            (
                "import importlib\nimportlib.import_module('.parallel_state', 'megatron.core')"
                ".get_tensor_model_parallel_rank()\n",
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
            "import importlib\nimportlib.import_module('.parallel_state', 'other')"
            ".get_tensor_model_parallel_rank()\n",
            "import sys\nsys.modules['other.parallel_state'].get_tensor_model_parallel_rank()\n",
            "from other import *\nparallel_state.get_tensor_model_parallel_rank()\n",
            # megatron.core.__all__ exports parallel_state but not mpu.
            "from megatron.core import *\ndef f(mpu):\n    mpu.get_tensor_model_parallel_rank()\n",
            "from megatron.core import mpu\ngrid = config if lazy else None\n"
            "grid.get_tensor_model_parallel_rank()\n",
        ]
        for source in cases:
            with self.subTest(source=source):
                self.assertEqual(self.uses(source), [])

    def test_star_import_binds_only_public_names(self):
        deprecated = self.DEPRECATED | {"_set_global_memory_buffer"}
        # parallel_state has no __all__, so the star import does not bind the private function.
        source = (
            "from megatron.core.parallel_state import *\n"
            "def f(_set_global_memory_buffer):\n    _set_global_memory_buffer()\n"
        )
        self.assertEqual(uncovered_uses(source, self.PACKAGE, deprecated), [])
        source = "from megatron.core.parallel_state import *\nget_tensor_model_parallel_rank()\n"
        self.assertEqual(uncovered_uses(source, self.PACKAGE, deprecated), [2])

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
            "warnings.warn('lazy init is deprecated', ParallelStateDeprecationWarning)\n"
            "with warnings.catch_warnings():\n"
            "    warnings.simplefilter('ignore', ParallelStateDeprecationWarning)\n"
            "    mpu.set_tensor_model_parallel_rank(0)\n",
            "from megatron.core import mpu\nmpu.initialize_model_parallel()\n",
            "from megatron.core import utils\nutils.get_tensor_model_parallel_rank()\n",
            "with global_process_group_fallback('f', deprecated_in='0.21'):\n"
            "    pg_collection = ProcessGroupCollection.use_mpu_process_groups(['tp'])\n",
            "import warnings\n"
            "warn_global_process_group_fallback('f', deprecated_in='0.21', removed_in='0.23')\n"
            "with warnings.catch_warnings():\n"
            "    warnings.simplefilter('ignore', ParallelStateDeprecationWarning)\n"
            "    pg_collection = ProcessGroupCollection.use_mpu_process_groups()\n",
            "from megatron.core import parallel_state\n"
            "pg_collection = parallel_state.get_registered_process_groups()\n",
        ]
        for source in cases:
            with self.subTest(source=source):
                self.assertEqual(self.uses(source), [])

    def test_bootstrap_block_follows_an_explicit_warning(self):
        # Ignoring ParallelStateDeprecationWarning is allowed only after the code has reported the
        # read itself, earlier in the function, as megatron/training/initialize.py does.
        imports = "import warnings\nfrom megatron.core import mpu\n"
        block = (
            "with warnings.catch_warnings():\n"
            "    warnings.simplefilter('ignore', ParallelStateDeprecationWarning)\n"
            "    mpu.set_tensor_model_parallel_rank(0)\n"
        )
        warn = "warnings.warn('deprecated', ParallelStateDeprecationWarning)\n"

        def body(*statements, depth=1):
            return "".join(
                "    " * depth + line + "\n"
                for statement in statements
                for line in statement.splitlines()
            )

        allowed = {
            "a warning by position": warn.replace(")", ", stacklevel=2)"),
            "a warning by keyword": warn.replace("Parallel", "category=Parallel"),
            "a qualified category": warn.replace("Parallel", "parallel_state.Parallel"),
            "a fallback warning": "warn_global_process_group_fallback('f')\n",
            "warnings imported under another name": (
                "import warnings as w\n" + warn.replace("warnings.", "w.")
            ),
            "warn imported from warnings": (
                "from warnings import warn\n" + warn.replace("warnings.", "")
            ),
        }
        for case, warning in allowed.items():
            with self.subTest(case=case):
                self.assertEqual(self.uses(imports + "def f():\n" + body(warning, block)), [])
        with self.subTest(case="a warning in a block that contains the bootstrap block"):
            self.assertEqual(
                self.uses(
                    imports + "def f(lazy):\n" + body(warn, "if lazy:") + body(block, depth=2)
                ),
                [],
            )
        flagged = {
            "no warning": body(block),
            "a warning after the block": body(block, warn),
            "a warning in a branch before the block": (
                body("if lazy:") + body(warn, depth=2) + body(block)
            ),
            "a warning of another category": body(
                warn.replace("ParallelStateDeprecationWarning", "DeprecationWarning"), block
            ),
            "a category whose name only contains the warning's": body(
                warn.replace("Parallel", "NotParallel"), block
            ),
            "a logger's warn": body(warn.replace("warnings.", "logger."), block),
            "a warn function of another module": body(
                "from other import warn", warn.replace("warnings.", ""), block
            ),
            "a warning that is not a statement": body("lazy or " + warn, block),
            "a warning the block itself silences": body(
                block.replace("    mpu.", "    " + warn + "    mpu.")
            ),
        }
        for case, function_body in flagged.items():
            with self.subTest(case=case):
                source = imports + "def f(lazy):\n" + function_body
                self.assertEqual(len(self.uses(source)), 1, source)
        with self.subTest(case="a warning in another function"):
            self.assertEqual(
                self.uses(imports + "def g():\n" + body(warn) + "def f():\n" + body(block)), [8]
            )
        with self.subTest(case="a function defined after the warning"):
            self.assertEqual(self.uses(imports + warn + "def f():\n" + body(block)), [7])

    def test_finds_private_attributes_anywhere(self):
        # A private attribute of parallel_state is not an interface: no block allows it.
        cases = [
            ("from megatron.core import mpu\nmpu._DATA_PARALLEL_GROUP\n", [2]),
            ("from megatron.core import parallel_state\nparallel_state._MPU_X = 0\n", [2]),
            ("from megatron.core import mpu\ndel mpu._DATA_PARALLEL_GROUP\n", [2]),
            ("from megatron.core import mpu\ngetattr(mpu, '_DATA_PARALLEL_GROUP', None)\n", [2]),
            ("from megatron.core import mpu\nsetattr(mpu, '_DATA_PARALLEL_GROUP', None)\n", [2]),
            ("from megatron.core import parallel_state\ndelattr(parallel_state, '_MPU_X')\n", [2]),
            (
                "import megatron.core\ngrid = megatron.core.parallel_state\n"
                "grid._DATA_PARALLEL_GROUP = None\n",
                [3],
            ),
            (
                "from megatron.core.parallel_state import _DATA_PARALLEL_GROUP as group\n"
                "def f():\n    return group\n",
                [3],
            ),
            (
                "from megatron.core import mpu\n"
                "with global_process_group_fallback('f'):\n"
                "    mpu._MPU_TENSOR_MODEL_PARALLEL_RANK = 0\n",
                [3],
            ),
            (
                "import warnings\nfrom megatron.core import mpu\n"
                "warnings.warn('deprecated', ParallelStateDeprecationWarning)\n"
                "with warnings.catch_warnings():\n"
                "    warnings.simplefilter('ignore', ParallelStateDeprecationWarning)\n"
                "    mpu._MPU_TENSOR_MODEL_PARALLEL_RANK = 0\n",
                [6],
            ),
        ]
        for source, lines in cases:
            with self.subTest(source=source):
                self.assertEqual(self.uses(source), lines)
        for source in (
            "from megatron.core import mpu\nmpu.__name__\n",
            "from megatron.core import utils\nutils._GLOBAL_MEMORY_BUFFER\n",
            "from other import parallel_state\nparallel_state._DATA_PARALLEL_GROUP\n",
            "self._DATA_PARALLEL_GROUP = None\n",
        ):
            with self.subTest(source=source):
                self.assertEqual(self.uses(source), [])

    def test_a_deprecated_private_function_counts_as_deprecated(self):
        # parallel_state deprecates _set_global_memory_buffer() like its public functions, so a
        # fallback block allows it, as tools/check_process_group_usage.py does.
        deprecated = self.DEPRECATED | {"_set_global_memory_buffer"}
        fallback = "with global_process_group_fallback('f'):\n    "
        for imports, call in (
            ("from megatron.core import mpu\n", "mpu._set_global_memory_buffer()\n"),
            (
                "from megatron.core.parallel_state import _set_global_memory_buffer\n",
                "_set_global_memory_buffer()\n",
            ),
        ):
            with self.subTest(call=call):
                self.assertEqual(uncovered_uses(imports + call, self.PACKAGE, deprecated), [2])
                self.assertEqual(
                    uncovered_uses(imports + fallback + call, self.PACKAGE, deprecated), []
                )


class TestInTreeUses(unittest.TestCase):
    """The code in SCANNED_FOLDERS and at the top level, outside Megatron Core, does not warn."""

    def test_parallel_state_deprecates_functions(self):
        names = deprecated_functions(PARALLEL_STATE_FILE.read_text(encoding="utf-8"))
        self.assertIn("get_tensor_model_parallel_group", names)
        self.assertIn("set_tensor_model_parallel_rank", names)
        self.assertIn("_set_global_memory_buffer", names)
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
