# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The package version has not reached a removal version that the code still announces.

The compatibility fallbacks to the global process groups, and the deprecated parallel_state
accessors, name the Megatron Core release that removes them: the ``removed_in`` argument of
``warn_global_process_group_fallback``, ``global_process_group_fallback`` and
``resolve_process_groups``, the default of that argument, and the module constants named
``*_REMOVED_IN``, such as ``_ACCESSOR_REMOVED_IN`` in parallel_state.py. Once the version in
megatron/core/package_info.py reaches one of them, the code it names is due for removal, and this
test fails and lists it. The pre-release suffix does not count: 0.23.0rc0 has reached 0.23.

Stdlib only. Run directly with ``python tests/unit_tests/test_removal_versions.py`` to avoid the
GPU fixtures used when pytest collects the enclosing unit-test package.
"""

import ast
import pathlib
import tempfile
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PACKAGE_INFO = pathlib.Path("megatron") / "core" / "package_info.py"
# The code that ships, or runs Megatron Core: the tests are not scanned.
SCANNED_FOLDERS = ("megatron", "examples", "tools", "tasks", "scripts")
# The functions whose removed_in argument names the release that removes a fallback.
FALLBACKS = (
    "warn_global_process_group_fallback",
    "global_process_group_fallback",
    "resolve_process_groups",
)
ARGUMENT = "removed_in"
CONSTANT_SUFFIX = "_REMOVED_IN"


def package_version(root):
    """Return ``(MAJOR, MINOR, PATCH)`` from package_info.py, without importing it."""
    values = {}
    for node in ast.parse((root / PACKAGE_INFO).read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    values[target.id] = node.value.value
    return values["MAJOR"], values["MINOR"], values["PATCH"]


def parse_version(text):
    """Return ``"0.23"`` as ``(0, 23, 0)``; raise ValueError for anything else."""
    parts = tuple(int(part) for part in text.split("."))
    if not 1 <= len(parts) <= 3:
        raise ValueError(f"not a release version: {text!r}")
    return parts + (0,) * (3 - len(parts))


def _source_paths(root):
    """Return the Python files of SCANNED_FOLDERS and of the top level of ``root``."""
    paths = [path for folder in SCANNED_FOLDERS for path in (root / folder).rglob("*.py")]
    return sorted(paths + list(root.glob("*.py")))


def _module_constants(tree):
    """Return ``{name: (line, value)}`` for the module-level string constants of ``tree``."""
    constants = {}
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else []
        if isinstance(node, ast.AnnAssign):
            targets = [node.target]
        value = getattr(node, "value", None)
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            for target in targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = (node.lineno, value.value)
    return constants


def _called_name(node):
    """Return the name of the function that the call ``node`` calls, without its owner."""
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


def removal_sites(root):
    """Return ``(location, version, description)`` for every removal version in the code.

    ``version`` is the announced string, or, for a ``*_REMOVED_IN`` constant that the scanned
    code does not define, the source of its reference. A ``removed_in`` value that is None or
    comes from elsewhere, such as a parameter that the enclosing function forwards, an attribute
    or an item, is not a site: the code that provides the value is one.
    """
    trees = {path: ast.parse(path.read_text(encoding="utf-8")) for path in _source_paths(root)}
    constants = {path: _module_constants(tree) for path, tree in trees.items()}
    # Constants that other modules reference as an attribute, such as
    # parallel_state._ACCESSOR_REMOVED_IN.
    removal_constants = {
        name: value
        for values in constants.values()
        for name, (_, value) in values.items()
        if name.endswith(CONSTANT_SUFFIX)
    }

    def resolve(path, expression):
        if isinstance(expression, ast.Constant):
            return None if expression.value is None else str(expression.value)
        if isinstance(expression, ast.Name) and expression.id in constants[path]:
            return constants[path][expression.id][1]
        name = getattr(expression, "id", None) or getattr(expression, "attr", None)
        if name in removal_constants:
            return removal_constants[name]
        # A removal constant that the scanned code does not define cannot be checked.
        if isinstance(expression, (ast.Name, ast.Attribute)) and name.endswith(CONSTANT_SUFFIX):
            return ast.unparse(expression)
        return None

    sites, defaults, default_uses = [], {}, []
    for path, tree in trees.items():
        relative = path.relative_to(root).as_posix()
        for name, (line, value) in constants[path].items():
            if name.endswith(CONSTANT_SUFFIX):
                sites.append((f"{relative}:{line}", value, name))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                arguments = node.args
                positional = arguments.posonlyargs + arguments.args
                pairs = list(zip(positional[::-1], arguments.defaults[::-1]))
                pairs += list(zip(arguments.kwonlyargs, arguments.kw_defaults))
                for argument, default in pairs:
                    if argument.arg == ARGUMENT and default is not None:
                        version = resolve(path, default)
                        if version is not None:
                            sites.append(
                                (f"{relative}:{node.lineno}", version, f"default of {node.name}")
                            )
                            if node.name in FALLBACKS:
                                defaults[node.name] = version
            elif isinstance(node, ast.Call):
                keywords = [keyword for keyword in node.keywords if keyword.arg == ARGUMENT]
                for keyword in keywords:
                    version = resolve(path, keyword.value)
                    if version is not None:
                        sites.append(
                            (f"{relative}:{node.lineno}", version, f"{_called_name(node)}()")
                        )
                if not keywords and _called_name(node) in FALLBACKS:
                    default_uses.append((f"{relative}:{node.lineno}", _called_name(node)))
    # A fallback call that omits removed_in announces its function's default.
    for location, name in default_uses:
        if name in defaults:
            sites.append((location, defaults[name], f"{name}() by default"))
    # In file order, by line number.
    return sorted(
        sites, key=lambda site: (site[0].rpartition(":")[0], int(site[0].rpartition(":")[2]))
    )


def due_sites(version, sites):
    """Return the sites whose removal version ``version`` has reached, or that do not parse."""
    due = []
    for site in sites:
        try:
            removal = parse_version(site[1])
        except ValueError:
            due.append(site)
            continue
        if tuple(version) >= removal:
            due.append(site)
    return due


def _report(version, due):
    """Describe the sites in ``due`` for a failure at package version ``version``."""
    lines = [f"  {location}  {ARGUMENT}={removal}  ({what})" for location, removal, what in due]
    return (
        f"Megatron Core {'.'.join(map(str, version))} has reached the removal version of "
        f"{len(due)} site(s); remove the fallbacks or deprecated functions they announce, or "
        "correct a version that does not parse:\n" + "\n".join(lines)
    )


class TestRemovalVersions(unittest.TestCase):
    """The repository announces no removal that its package version has already reached."""

    @classmethod
    def setUpClass(cls):
        cls.sites = removal_sites(REPO_ROOT)

    def test_package_version_has_not_reached_a_removal_version(self):
        version = package_version(REPO_ROOT)
        due = due_sites(version, self.sites)
        if due:
            self.fail(_report(version, due))

    def test_collects_the_announced_removal_versions(self):
        sites = self.sites
        described = {(location.partition(":")[0], what) for location, _, what in sites}
        self.assertIn(("megatron/core/parallel_state.py", "_ACCESSOR_REMOVED_IN"), described)
        for function in FALLBACKS:
            self.assertIn(
                ("megatron/core/process_groups_config.py", f"default of {function}"), described
            )
            self.assertTrue(any(what == f"{function}()" for _, _, what in sites), function)
        for location, removal, what in sites:
            with self.subTest(location=location, what=what):
                parse_version(removal)

    def test_a_fake_version_that_reaches_a_removal_lists_its_sites(self):
        # Fake package versions taken from the announced removal versions, so that the test
        # does not depend on which versions the tree announces.
        sites = self.sites
        earliest = min(parse_version(removal) for _, removal, _ in sites)
        due = due_sites(earliest, sites)
        self.assertEqual(due, [site for site in sites if parse_version(site[1]) == earliest])
        self.assertIn(f"{due[0][0]}  {ARGUMENT}={due[0][1]}", _report(earliest, due))
        before = (earliest[0], earliest[1] - 1, 999) if earliest[1] else (earliest[0] - 1, 999, 0)
        self.assertEqual(due_sites(before, sites), [])
        latest = max(parse_version(removal) for _, removal, _ in sites)
        self.assertEqual(due_sites(latest, sites), sites)


class TestRemovalSites(unittest.TestCase):
    """Collect removal versions from a synthetic tree."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = pathlib.Path(directory.name)
        self.write(
            PACKAGE_INFO.as_posix(), "MAJOR = 0\nMINOR = 22\nPATCH = 0\nPRE_RELEASE = 'rc0'\n"
        )
        self.write(
            "megatron/core/process_groups_config.py",
            "_FALLBACK_REMOVED_IN = '0.22'\n"
            "def warn_global_process_group_fallback(owner, *, removed_in=_FALLBACK_REMOVED_IN):\n"
            "    _warn(owner, removed_in=removed_in)\n",
        )
        self.write(
            "megatron/training/fallbacks.py",
            "from megatron.core import process_groups_config as pgc\n"
            "_GROUP_REMOVED_IN = '0.24'\n"
            "pgc.warn_global_process_group_fallback('a', removed_in='0.23')\n"
            "pgc.warn_global_process_group_fallback('b')\n"
            "pgc.warn_global_process_group_fallback('c', removed_in=_GROUP_REMOVED_IN)\n",
        )
        self.write("tests/unit_tests/test_fallbacks.py", "f(removed_in='0.1')\n")

    def write(self, relative, source):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")

    def test_collects_constants_defaults_arguments_and_default_uses(self):
        self.assertEqual(
            removal_sites(self.root),
            [
                ("megatron/core/process_groups_config.py:1", "0.22", "_FALLBACK_REMOVED_IN"),
                (
                    "megatron/core/process_groups_config.py:2",
                    "0.22",
                    "default of warn_global_process_group_fallback",
                ),
                ("megatron/training/fallbacks.py:2", "0.24", "_GROUP_REMOVED_IN"),
                (
                    "megatron/training/fallbacks.py:3",
                    "0.23",
                    "warn_global_process_group_fallback()",
                ),
                (
                    "megatron/training/fallbacks.py:4",
                    "0.22",
                    "warn_global_process_group_fallback() by default",
                ),
                (
                    "megatron/training/fallbacks.py:5",
                    "0.24",
                    "warn_global_process_group_fallback()",
                ),
            ],
        )

    def test_due_sites_follow_the_package_version(self):
        sites = removal_sites(self.root)
        self.assertEqual(package_version(self.root), (0, 22, 0))
        due = due_sites(package_version(self.root), sites)
        self.assertEqual([site[1] for site in due], ["0.22", "0.22", "0.22"])
        self.assertEqual(due_sites((0, 21, 9), sites), [])
        self.assertEqual(len(due_sites((0, 23, 0), sites)), 4)
        self.assertEqual(due_sites((1, 0, 0), sites), sites)

    def test_a_removal_version_that_does_not_parse_is_due(self):
        self.write(
            "megatron/training/odd.py",
            "def f():\n    warn_global_process_group_fallback('d', removed_in='next')\n"
            "    warn_global_process_group_fallback('e', removed_in=other._LATER_REMOVED_IN)\n",
        )
        due = due_sites((0, 0, 0), removal_sites(self.root))
        self.assertEqual(
            due,
            [
                ("megatron/training/odd.py:2", "next", "warn_global_process_group_fallback()"),
                (
                    "megatron/training/odd.py:3",
                    "other._LATER_REMOVED_IN",
                    "warn_global_process_group_fallback()",
                ),
            ],
        )

    def test_values_from_elsewhere_are_not_sites(self):
        self.write(
            "megatron/training/forwarding.py",
            "def warn_deprecated(owner, removed_in=None):\n"
            "    warn_global_process_group_fallback(owner, removed_in=removed_in)\n"
            "class Fallback:\n"
            "    def warn(self, **kwargs):\n"
            "        warn_global_process_group_fallback('f', removed_in=self.removed_in)\n"
            "        warn_global_process_group_fallback('g', removed_in=kwargs['removed_in'])\n"
            "        warn_global_process_group_fallback('h', removed_in=version())\n",
        )
        sites = removal_sites(self.root)
        self.assertEqual([site for site in sites if "forwarding" in site[0]], [])
        self.assertEqual(due_sites((0, 21, 0), sites), [])


if __name__ == "__main__":
    unittest.main()
