# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""In-tree code takes the global memory buffer from megatron.core.utils, not from parallel_state.

Stdlib only. Run directly with ``python tests/unit_tests/test_global_memory_buffer_imports.py`` to
avoid the GPU fixtures used when pytest collects the enclosing unit-test package.
"""

import ast
import importlib.util
import pathlib
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PARALLEL_STATE = "megatron.core.parallel_state"
# megatron.core exports parallel_state under its legacy name as well.
PARALLEL_STATE_NAMES = {"parallel_state", "mpu"}
BUFFER_FUNCTIONS = {
    "get_global_memory_buffer",
    "destroy_global_memory_buffer",
    "_set_global_memory_buffer",
}
# Tests are not scanned: they exercise the parallel_state aliases on purpose.
SCANNED_FOLDERS = ("megatron", "examples", "tools")


def parallel_state_buffer_uses(source, package):
    """Return the lines that take a buffer function from parallel_state.

    ``package`` is the dotted package of the scanned file, used to resolve relative imports. A
    name bound to the parallel_state module anywhere in the file counts in every scope.
    """
    tree = ast.parse(source)
    modules = set()
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(a.asname for a in node.names if a.name == PARALLEL_STATE and a.asname)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                module = importlib.util.resolve_name("." * node.level + module, package)
            for alias in node.names:
                if module == PARALLEL_STATE and alias.name in BUFFER_FUNCTIONS:
                    lines.append(node.lineno)
                elif module == "megatron.core" and alias.name in PARALLEL_STATE_NAMES:
                    modules.add(alias.asname or alias.name)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in BUFFER_FUNCTIONS:
            owner = node.value
            if (isinstance(owner, ast.Name) and owner.id in modules) or (
                ast.unparse(owner) == PARALLEL_STATE
            ):
                lines.append(node.lineno)
    return sorted(lines)


class TestParallelStateBufferUses(unittest.TestCase):
    """The scan finds every spelling of the parallel_state aliases and nothing else."""

    PACKAGE = "megatron.core.transformer"

    def test_detects_parallel_state_aliases(self):
        cases = [
            ("from megatron.core.parallel_state import get_global_memory_buffer\n", [1]),
            (
                "from megatron.core.parallel_state import (\n"
                "    destroy_model_parallel,\n"
                "    destroy_global_memory_buffer,\n"
                ")\n",
                [1],
            ),
            ("from ..parallel_state import get_global_memory_buffer\n", [1]),
            (
                "from megatron.core import parallel_state\n"
                "parallel_state.get_global_memory_buffer().get_tensor(shape, dtype, 'mpu')\n",
                [2],
            ),
            ("from megatron.core import mpu\nmpu.destroy_global_memory_buffer()\n", [2]),
            ("from .. import parallel_state as ps\nps._set_global_memory_buffer()\n", [2]),
            ("import megatron.core.parallel_state as grid\ngrid.get_global_memory_buffer()\n", [2]),
            (
                "import megatron.core.parallel_state\n"
                "megatron.core.parallel_state.get_global_memory_buffer()\n",
                [2],
            ),
            (
                "from megatron.core import parallel_state\n"
                "def f():\n    return parallel_state.get_global_memory_buffer()\n",
                [3],
            ),
        ]
        for source, lines in cases:
            with self.subTest(source=source):
                self.assertEqual(parallel_state_buffer_uses(source, self.PACKAGE), lines)

    def test_ignores_the_core_buffer_and_unrelated_buffers(self):
        cases = [
            "from megatron.core.utils import get_global_memory_buffer\nget_global_memory_buffer()\n",
            "from megatron.core import utils\nutils.destroy_global_memory_buffer()\n",
            "from ..utils import destroy_global_memory_buffer, get_global_memory_buffer\n",
            "from .utils import get_global_memory_buffer\n",
            "from megatron.core import parallel_state\n"
            "parallel_state.destroy_model_parallel()\n"
            "fsdp_utils.get_global_memory_buffer()\n",
        ]
        for source in cases:
            with self.subTest(source=source):
                self.assertEqual(parallel_state_buffer_uses(source, self.PACKAGE), [])


class TestInTreeBufferUses(unittest.TestCase):
    """Library, example and tool code use megatron.core.utils for the global memory buffer."""

    def test_in_tree_code_uses_the_core_buffer(self):
        paths = [path for folder in SCANNED_FOLDERS for path in (REPO_ROOT / folder).rglob("*.py")]
        paths.extend(REPO_ROOT.glob("*.py"))
        violations = []
        for path in sorted(paths):
            relative = path.relative_to(REPO_ROOT)
            package = ".".join(relative.parent.parts)
            source = path.read_text(encoding="utf-8")
            for line in parallel_state_buffer_uses(source, package):
                violations.append(f"{relative}:{line}")
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
