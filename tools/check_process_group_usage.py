#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Fail on *new* reads of global process-group state in ``megatron/core``.

``megatron.core.parallel_state`` holds the process groups for a single, global parallel grid.
Reading it inside library code is being removed: for a model built on independent parallel grids
it returns the wrong grid, silently. See https://github.com/NVIDIA/Megatron-LM/issues/6307.

This check is a ratchet, not a cleanup. Every existing violation is recorded in an allowlist so
the build stays green; the check fails only when a *new* one appears, or when the allowlist
claims a violation that no longer exists (so the allowlist shrinks as the migration lands).
Entries count each accessor within its enclosing function/class, independently of line numbers.
This is a syntactic check of imported accessors, not data-flow analysis. A call sees the most
recent import of its name that precedes it in the same scope, even if the name is reassigned
later; aliases created through assignments (``grid = parallel_state``) or dynamic attribute
lookup are outside its scope.

Usage::

    python tools/check_process_group_usage.py            # check
    python tools/check_process_group_usage.py --update   # remove stale allowlist entries
    python tools/check_process_group_usage.py --stats    # summarize without failing
"""

import argparse
import ast
import json
import pathlib
import sys
from collections import Counter

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCAN_ROOT = REPO_ROOT / "megatron" / "core"
ALLOWLIST = REPO_ROOT / "tools" / "process_group_usage_allowlist.json"

# Files permitted to read global state: the module itself and the collection that bridges to it.
# Everything else in megatron/core is subject to the ratchet, including existing fallbacks.
EXEMPT = {"megatron/core/parallel_state.py", "megatron/core/process_groups_config.py"}

# Accessors that read the global grid. Deliberately excludes initialize/destroy/is_initialized
# (the intended long-term surface) and the virtual-pipeline and memory-buffer globals, which have
# no replacement yet and are tracked separately.
DEPRECATED_PREFIXES = ("get_",)
# "_gloo" covers the Gloo group accessors (get_data_parallel_group_gloo,
# get_expert_data_parallel_group_gloo), which also return groups of the global grid.
DEPRECATED_SUFFIXES = ("_group", "_groups", "_gloo", "_rank", "_ranks", "_world_size", "_src_rank")
NOT_DEPRECATED = {
    "get_nccl_options",
    "get_all_ranks",
    "get_global_memory_buffer",
    "get_virtual_pipeline_model_parallel_rank",
    "get_virtual_pipeline_model_parallel_world_size",
}


def _is_deprecated_accessor(name: str) -> bool:
    if name in NOT_DEPRECATED:
        return False
    return name.startswith(DEPRECATED_PREFIXES) and name.endswith(DEPRECATED_SUFFIXES)


def _qualified_name(node):
    """Return the dotted spelling of a simple name or attribute expression."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_qualified_name(node.value)}.{node.attr}"
    return ""


def _scope_bindings(node):
    """Collect the binding events of one scope without leaking nested scopes into their parent.

    Returns ``{name: [(position, target), ...]}`` in source order. ``target`` is the dotted path
    a name was imported from, or ``""`` for any other binding (parameter, assignment, nested
    definition).
    """

    class Bindings(ast.NodeVisitor):
        """Collect names bound by the current module, function, or class."""

        def __init__(self):
            self.events = {}

        def bind(self, name, node, target):
            """Record that ``name`` is bound to ``target`` at ``node``'s position."""
            self.events.setdefault(name, []).append(((node.lineno, node.col_offset), target))

        def visit_Import(self, node):
            """Record the module bound by each import name."""
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                self.bind(name, node, alias.name if alias.asname else name)

        def visit_ImportFrom(self, node):
            """Record direct imports, including relative imports of core compatibility modules."""
            module = node.module or ""
            # Relative imports in megatron/core can refer to either of these modules.
            if node.level and module in ("", "parallel_state", "process_groups_config"):
                module = "megatron.core" + (f".{module}" if module else "")
            for alias in node.names:
                self.bind(alias.asname or alias.name, node, f"{module}.{alias.name}")

        def visit_Name(self, node):
            """Keep simple assignments from being mistaken for imported module aliases."""
            if isinstance(node.ctx, ast.Store):
                self.bind(node.id, node, "")

        def visit_FunctionDef(self, node):
            """Bind a nested definition's name without collecting imports from its body."""
            self.bind(node.name, node, "")

        visit_AsyncFunctionDef = visit_FunctionDef
        visit_ClassDef = visit_FunctionDef

        def visit_Lambda(self, node):
            """Do not collect bindings from an anonymous nested scope."""
            pass

        # Comprehension targets and lambda arguments have their own scope.
        visit_ListComp = visit_Lambda
        visit_SetComp = visit_Lambda
        visit_DictComp = visit_Lambda
        visit_GeneratorExp = visit_Lambda

    bindings = Bindings()
    if hasattr(node, "args"):
        args = node.args
        for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg):
            if arg:
                # Parameters are bound before any statement of the body runs.
                bindings.events.setdefault(arg.arg, []).append(((0, -1), ""))
    for statement in node.body:
        bindings.visit(statement)
    return bindings.events


def _binding_at(events, position):
    """Return what a name refers to at ``position`` in the scope that binds it.

    A call sees the most recent import of the name that precedes it, even when the name is
    reassigned afterwards: the accessor has already run against the global grid by then. Other
    bindings never hide an earlier import (a conservative choice for branches and loops), but a
    name bound in the scope without a preceding import is a local value, not a module.
    """
    imported = [target for start, target in events if target and start < position]
    return imported[-1] if imported else ""


def _binding_seen_by_nested_scopes(events):
    """Return what a name in an enclosing scope refers to when a nested scope runs later."""
    imported = [target for _, target in events if target]
    return imported[-1] if imported else ""


def _violations_in(path: pathlib.Path):
    """Return (lineno, scope:kind:detail) pairs for global process-group reads in one file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    class Visitor(ast.NodeVisitor):
        """Attach a stable enclosing scope to each directly imported global-state call."""

        def __init__(self):
            self.scope = []
            self.hits = []
            # One (kind, names visible from enclosing scopes, own binding events) per scope.
            self.bindings = [("module", {}, _scope_bindings(tree))]

        def visit_FunctionDef(self, node):
            """Resolve function-local imports independently of sibling and class bindings."""
            for expression in (*node.decorator_list, *node.args.defaults, *node.args.kw_defaults):
                if expression:
                    self.visit(expression)
            self._visit_scope(node, "function")

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node):
            """Resolve class-body imports without making them lexical bindings in its methods."""
            for expression in (*node.decorator_list, *node.bases, *node.keywords):
                self.visit(expression)
            self._visit_scope(node, "class")

        def _visit_scope(self, node, kind):
            _, outer, events = next(
                binding for binding in reversed(self.bindings) if binding[0] != "class"
            )
            visible = {
                **outer,
                **{name: _binding_seen_by_nested_scopes(seen) for name, seen in events.items()},
            }
            self.bindings.append((kind, visible, _scope_bindings(node)))
            self.scope.append(node.name)
            for statement in node.body:
                self.visit(statement)
            self.scope.pop()
            self.bindings.pop()

        def _expression_bindings(self, names, position=None):
            _, outer, events = next(
                binding for binding in reversed(self.bindings) if binding[0] != "class"
            )
            visible = {
                **outer,
                **{
                    name: (
                        _binding_at(seen, position)
                        if position
                        else _binding_seen_by_nested_scopes(seen)
                    )
                    for name, seen in events.items()
                },
            }
            self.bindings.append(("function", visible, {name: [((0, -1), "")] for name in names}))

        def visit_Lambda(self, node):
            for expression in (*node.args.defaults, *node.args.kw_defaults):
                if expression:
                    self.visit(expression)
            args = node.args
            names = [
                arg.arg
                for arg in (
                    *args.posonlyargs,
                    *args.args,
                    *args.kwonlyargs,
                    args.vararg,
                    args.kwarg,
                )
                if arg
            ]
            self._expression_bindings(names)
            self.visit(node.body)
            self.bindings.pop()

        def visit_ListComp(self, node):
            self.visit(node.generators[0].iter)
            names = [
                target.id
                for generator in node.generators
                for target in ast.walk(generator.target)
                if isinstance(target, ast.Name) and isinstance(target.ctx, ast.Store)
            ]
            # Only generator expressions defer their body until iteration.
            position = (
                None if isinstance(node, ast.GeneratorExp) else (node.lineno, node.col_offset)
            )
            self._expression_bindings(names, position)
            for index, generator in enumerate(node.generators):
                if index:
                    self.visit(generator.iter)
                for condition in generator.ifs:
                    self.visit(condition)
            if isinstance(node, ast.DictComp):
                self.visit(node.key)
                self.visit(node.value)
            else:
                self.visit(node.elt)
            self.bindings.pop()

        visit_SetComp = visit_ListComp
        visit_DictComp = visit_ListComp
        visit_GeneratorExp = visit_ListComp

        def visit_Call(self, node):
            """Record calls resolved through an import, including import aliases."""
            name = _qualified_name(node.func)
            root, separator, suffix = name.partition(".")
            _, outer, events = self.bindings[-1]
            if root in events:
                target = _binding_at(events[root], (node.lineno, node.col_offset))
            else:
                target = outer.get(root, root)
            name = target + separator + suffix
            module, _, accessor = name.rpartition(".")
            identity = None
            if module in ("megatron.core.parallel_state", "megatron.core.mpu"):
                if _is_deprecated_accessor(accessor):
                    identity = f"accessor:parallel_state.{accessor}"
            elif name == (
                "megatron.core.process_groups_config.ProcessGroupCollection.use_mpu_process_groups"
            ):
                identity = "shim:use_mpu_process_groups"
            if identity:
                scope = ".".join(self.scope) or "<module>"
                self.hits.append((node.lineno, f"{scope}:{identity}"))
            self.generic_visit(node)

    visitor = Visitor()
    visitor.visit(tree)
    return visitor.hits


def scan():
    """Return {relative path: sorted list of "scope:kind:detail"}, retaining duplicate calls."""
    found = {}
    for path in sorted(SCAN_ROOT.rglob("*.py")):
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in EXEMPT:
            continue
        hits = [identity for _, identity in _violations_in(path)]
        if hits:
            found[rel] = sorted(hits)
    return found


def _load_allowlist():
    """Load the committed baseline; a missing or malformed file must fail the check."""
    return json.loads(ALLOWLIST.read_text(encoding="utf-8"))["allowed"]


def _counts(found):
    """Count calls by kind for the status output."""
    c = Counter()
    for hits in found.values():
        for h in hits:
            c[h.split(":", 2)[1]] += 1
    return c


def _difference(left, right):
    """Return calls in left exceeding the per-scope counts in right."""
    diff = {}
    for rel, hits in left.items():
        extra = Counter(hits) - Counter(right.get(rel, []))
        if extra:
            diff[rel] = sorted(extra.elements())
    return diff


def main(argv=None) -> int:
    """Check the baseline, or refresh it only after verifying no new calls were introduced."""
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--update", action="store_true", help="remove stale allowlist entries")
    mode.add_argument(
        "--stats", action="store_true", help="summarize without checking the allowlist"
    )
    args = ap.parse_args(argv)

    found = scan()
    counts = _counts(found)
    total = sum(counts.values())

    if args.stats:
        print(f"{total} global process-group read(s) across {len(found)} file(s) in megatron/core")
        for kind, n in counts.most_common():
            print(f"  {kind:10} {n}")
        return 0

    allowed = _load_allowlist()

    added = _difference(found, allowed)
    removed = _difference(allowed, found)

    if added:
        n = sum(len(v) for v in added.values())
        print(f"ERROR: {n} new read(s) of global process-group state in megatron/core:\n")
        for rel, hits in sorted(added.items()):
            # Show current source locations while keeping line numbers out of the baseline.
            for lineno, identity in _violations_in(REPO_ROOT / rel):
                if identity in hits:
                    print(f"  {rel}:{lineno}  {identity}")
        print(
            "\nmegatron/core must not read process groups from parallel_state. Accept a "
            "ProcessGroupCollection or an explicit torch.distributed.ProcessGroup from the "
            "caller and pass it through.\n"
            "Note that ProcessGroupCollection.use_mpu_process_groups() is NOT a valid "
            "replacement -- it reads the same global state.\n"
            "See https://github.com/NVIDIA/Megatron-LM/issues/6307\n"
        )
        return 1

    if args.update:
        ALLOWLIST.write_text(
            json.dumps(
                {
                    "_comment": (
                        "Grandfathered reads of global process-group state in megatron/core. "
                        "Entries are scope:kind:detail; duplicates count separate calls. "
                        "This list must only shrink. Remove stale entries with "
                        "`python tools/check_process_group_usage.py --update`."
                    ),
                    "total": total,
                    "allowed": found,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"Wrote {ALLOWLIST.relative_to(REPO_ROOT)}: {total} grandfathered site(s) "
            f"across {len(found)} file(s)."
        )
        return 0

    if removed:
        n = sum(len(v) for v in removed.values())
        print(
            f"{n} allowlisted site(s) no longer exist -- nice. Refresh the allowlist:\n"
            f"  python tools/check_process_group_usage.py --update\n"
        )
        for rel, hits in sorted(removed.items()):
            for h in hits:
                print(f"  {rel}  {h}")
        return 1

    print(f"OK: no new global process-group reads ({total} grandfathered).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
