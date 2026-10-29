#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Fail on *new* uses of the global ``parallel_state`` in ``megatron/core``.

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

Each entry has one of four kinds:

* ``accessor``: a ``parallel_state`` group, rank or size accessor;
* ``function``: another deprecated ``parallel_state`` function: a pipeline-stage predicate, the
  virtual-pipeline rank or size, ``get_all_ranks``, a rank or size override, a dynamic
  context-parallel group map, or a memory-buffer alias;
* ``shim``: ``ProcessGroupCollection.use_mpu_process_groups()``, which reads the same globals;
* ``guarded``: ``resolve_process_groups()``, the warned fallback in ``process_groups_config.py``.
  It reads the globals only when the caller omits its groups, and warns first.

A guarded call is still a read, so wrapping a fallback in ``resolve_process_groups()`` does not
remove it from the allowlist. A guarded call that replaces a removed accessor, function or shim
read in the same file and scope is a conversion, and ``--update`` swaps the entry. Any other new
guarded call fails like a new read.

An accessor, function or shim read that a fallback warning dominates is *warned*, and its entry
ends with ``:warned``: in the same function, an earlier statement of its block, or of a block
that contains it, calls ``warn_global_process_group_fallback()``, or the read sits in the block
of ``with global_process_group_fallback(...)``. A warning in a branch does not dominate the
statements after the branch, and a function or lambda defined after a warning runs without it.
A read that is warned only through data flow, for example under a later ``if group is None:``,
is *unwarned*. A read that gains its warning is a conversion; an allowlisted warned read that
loses it fails like a new read.

Usage::

    python tools/check_process_group_usage.py            # check
    python tools/check_process_group_usage.py --update   # remove stale entries, record conversions
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

# Group, rank and size accessors that read the global grid. Deliberately excludes
# initialize/destroy/is_initialized (the intended long-term surface).
DEPRECATED_PREFIXES = ("get_",)
# "_gloo" covers the Gloo group accessors (get_data_parallel_group_gloo,
# get_expert_data_parallel_group_gloo), which also return groups of the global grid.
DEPRECATED_SUFFIXES = ("_group", "_groups", "_gloo", "_rank", "_ranks", "_world_size", "_src_rank")
# get_ functions that are not accessors, whether or not the suffixes above match them:
# get_nccl_options is not deprecated, and the others are counted with DEPRECATED_FUNCTIONS. The
# test helpers that forbid the global grid stub the accessors only and expect these to keep working.
NOT_ACCESSORS = {
    "get_nccl_options",
    "get_all_ranks",
    "get_global_memory_buffer",
    "get_virtual_pipeline_model_parallel_rank",
    "get_virtual_pipeline_model_parallel_world_size",
}
# The other deprecated functions of the global grid. A call of a setter writes the grid rather
# than reads it, and counts the same way.
DEPRECATED_FUNCTIONS = {
    # Pipeline-stage predicates.
    "is_pipeline_first_stage",
    "is_pipeline_last_stage",
    "is_rank_in_embedding_group",
    "is_rank_in_position_embedding_group",
    # Virtual-pipeline rank and size.
    "get_virtual_pipeline_model_parallel_rank",
    "set_virtual_pipeline_model_parallel_rank",
    "get_virtual_pipeline_model_parallel_world_size",
    "set_virtual_pipeline_model_parallel_world_size",
    # The ranks of all groups, and the overrides of ranks and sizes.
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
    # Dynamic context-parallel group maps.
    "get_dynamic_data_context_parallel_group_map",
    "get_dynamic_tensor_data_context_parallel_group_map",
    # Aliases of the memory-buffer functions of megatron.core.utils.
    "get_global_memory_buffer",
    "destroy_global_memory_buffer",
    "_set_global_memory_buffer",
}

# Entry kinds in --stats order. "guarded" calls always warn before they read the globals.
KINDS = ("accessor", "function", "shim", "guarded")
GUARDED = "guarded"

# A call of another kind that a fallback warning dominates ends with this suffix: an earlier
# statement of the same block, or of a block that contains it, in the same function, calls
# FALLBACK_WARNING, or the call sits in the block of ``with FALLBACK_BLOCK(...)``.
WARNED = ":warned"
FALLBACK_WARNING = "megatron.core.process_groups_config.warn_global_process_group_fallback"
FALLBACK_BLOCK = "megatron.core.process_groups_config.global_process_group_fallback"


def _is_deprecated_accessor(name: str) -> bool:
    if name in NOT_ACCESSORS:
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
            # Whether a fallback warning dominates the statement being visited.
            self.warned = False

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
            # A function body runs when it is called, not where it is defined; a class body runs
            # in place.
            outer_warned = self.warned
            if kind == "function":
                self.warned = False
            self._visit_block(node.body)
            self.warned = outer_warned
            self.scope.pop()
            self.bindings.pop()

        def generic_visit(self, node):
            """Visit each list of statements, such as the branches of an ``if``, as a block."""
            for _, value in ast.iter_fields(node):
                if isinstance(value, list):
                    self._visit_block(value)
                elif isinstance(value, ast.AST):
                    self.visit(value)

        def _visit_block(self, statements):
            """Visit ``statements`` in order; a fallback warning dominates the ones after it.

            The warning stays within the block: a branch that warns does not dominate the
            statements after the branch.
            """
            outer = self.warned
            for statement in statements:
                if isinstance(statement, ast.AST):
                    self.visit(statement)
                    if isinstance(statement, ast.Expr) and self._calls(
                        statement.value, FALLBACK_WARNING
                    ):
                        self.warned = True
            self.warned = outer

        def visit_With(self, node):
            """A ``with global_process_group_fallback(...)`` block has warned before its body."""
            for item in node.items:
                self.visit(item)
            outer = self.warned
            self.warned = outer or any(
                self._calls(item.context_expr, FALLBACK_BLOCK) for item in node.items
            )
            self._visit_block(node.body)
            self.warned = outer

        visit_AsyncWith = visit_With

        def visit_Lambda(self, node):
            """A lambda runs when it is called, outside the block that defines it."""
            outer, self.warned = self.warned, False
            self.generic_visit(node)
            self.warned = outer

        def _resolve(self, node):
            """Return the dotted path that the function called by ``node`` was imported from."""
            name = _qualified_name(node.func)
            root, separator, suffix = name.partition(".")
            _, outer, events = self.bindings[-1]
            if root in events:
                target = _binding_at(events[root], (node.lineno, node.col_offset))
            else:
                target = outer.get(root, root)
            return target + separator + suffix

        def _calls(self, node, function):
            """True if ``node`` calls ``function``, given as its dotted import path."""
            return isinstance(node, ast.Call) and self._resolve(node) == function

        def visit_Call(self, node):
            """Record calls resolved through an import, including import aliases."""
            name = self._resolve(node)
            module, _, accessor = name.rpartition(".")
            identity = None
            if module in ("megatron.core.parallel_state", "megatron.core.mpu"):
                if _is_deprecated_accessor(accessor):
                    identity = f"accessor:parallel_state.{accessor}"
                elif accessor in DEPRECATED_FUNCTIONS:
                    identity = f"function:parallel_state.{accessor}"
            elif name == (
                "megatron.core.process_groups_config.ProcessGroupCollection.use_mpu_process_groups"
            ):
                identity = "shim:use_mpu_process_groups"
            elif name == "megatron.core.process_groups_config.resolve_process_groups":
                identity = "guarded:resolve_process_groups"
            if identity:
                scope = ".".join(self.scope) or "<module>"
                if self.warned and not identity.startswith(GUARDED + ":"):
                    identity += WARNED
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


def _scope(hit):
    """Return the enclosing scope of an entry ``scope:kind:detail``."""
    return hit.split(":", 2)[0]


def _kind(hit):
    """Return the kind of an entry ``scope:kind:detail``."""
    return hit.split(":", 2)[1]


def _counts(found):
    """Count calls by kind, and the other kinds' calls as warned or unwarned, for the output."""
    c = Counter()
    for hits in found.values():
        for h in hits:
            c[_kind(h)] += 1
            if _kind(h) != GUARDED:
                c["warned" if h.endswith(WARNED) else "unwarned"] += 1
    return c


def _difference(left, right):
    """Return calls in left exceeding the per-scope counts in right."""
    diff = {}
    for rel, hits in left.items():
        extra = Counter(hits) - Counter(right.get(rel, []))
        if extra:
            diff[rel] = sorted(extra.elements())
    return diff


def _match_replacements(added, removed):
    """Match the calls in ``added`` with the removed reads of the same file that they replace.

    Each removed accessor, function or shim read is matched at most once. Returns
    ``(new, converted, lost)``:

    * ``converted``: a read that a fallback warning now dominates, matched with the same read
      without one, and a guarded call, matched with another read of its scope. ``--update``
      records them.
    * ``lost``: a read that a fallback warning no longer dominates, matched with the same read
      with one. It fails like a new read.
    * ``new``: every other call.
    """
    new, converted, lost = {}, {}, {}
    for rel, hits in added.items():
        replaceable = [hit for hit in removed.get(rel, []) if _kind(hit) != GUARDED]
        # Match the reads first: a guarded call must not take the removed read that the same
        # read with a changed classification replaces.
        for hit in sorted(hits, key=lambda hit: _kind(hit) == GUARDED):
            if _kind(hit) == GUARDED:
                match = next((read for read in replaceable if _scope(read) == _scope(hit)), None)
                target = converted
            elif hit.endswith(WARNED):
                match = hit[: -len(WARNED)] if hit[: -len(WARNED)] in replaceable else None
                target = converted
            else:
                match = hit + WARNED if hit + WARNED in replaceable else None
                target = lost
            if match is None:
                target = new
            else:
                replaceable.remove(match)
            target.setdefault(rel, []).append(hit)
    return new, converted, lost


def _print_locations(entries):
    """Print the current source locations of the entries, which keep line numbers out."""
    for rel, hits in sorted(entries.items()):
        for lineno, identity in _violations_in(REPO_ROOT / rel):
            if identity in hits:
                print(f"  {rel}:{lineno}  {identity}")


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
    total = sum(counts[kind] for kind in KINDS)

    if args.stats:
        print(
            f"{total} use(s) of global parallel_state across {len(found)} file(s) in megatron/core"
        )
        for kind in KINDS:
            print(f"  {kind:10} {counts[kind]}")
        print(
            f"{total - counts[GUARDED]} accessor, function and shim use(s), by whether a fallback "
            "warning precedes them"
        )
        for classification in ("warned", "unwarned"):
            print(f"  {classification:10} {counts[classification]}")
        return 0

    allowed = _load_allowlist()

    removed = _difference(allowed, found)
    added, converted, lost = _match_replacements(_difference(found, allowed), removed)

    if lost:
        n = sum(len(v) for v in lost.values())
        print(
            f"ERROR: {n} allowlisted use(s) of global parallel_state in megatron/core no longer "
            "follow a fallback warning:\n"
        )
        _print_locations(lost)
        print(
            "\nA read of the global grid in megatron/core is a compatibility fallback for a "
            "caller that omitted its groups, and warns first: call "
            "warn_global_process_group_fallback() earlier in the same block, or in a block that "
            "contains it, or read inside `with global_process_group_fallback(...)`.\n"
        )
    if added:
        n = sum(len(v) for v in added.values())
        print(f"ERROR: {n} new use(s) of global parallel_state in megatron/core:\n")
        _print_locations(added)
        print(
            "\nmegatron/core must not read process groups from parallel_state. Accept a "
            "ProcessGroupCollection or an explicit torch.distributed.ProcessGroup from the "
            "caller and pass it through."
        )
        if any(hit.split(":", 2)[1] == "function" for hits in added.values() for hit in hits):
            print(
                "A 'function' use has a replacement of its own, which the function's "
                "_deprecated_accessor decorator in parallel_state.py names: for example the "
                "vp_stage of the model chunk, the virtual_pipeline_model_parallel_size of the "
                "model's configuration, is_pp_first_stage() or is_vp_first_stage() and their "
                "last-stage forms in megatron.core.pipeline_parallel.utils, or megatron.core.utils "
                "for the memory buffer."
            )
        print(
            "Note that ProcessGroupCollection.use_mpu_process_groups() is NOT a valid "
            "replacement -- it reads the same global state.\n"
            "resolve_process_groups() is counted too: it may only replace a read in the same "
            "function, for callers that do not pass their groups yet.\n"
            "See https://github.com/NVIDIA/Megatron-LM/issues/6307\n"
        )
    if added or lost:
        return 1

    if args.update:
        ALLOWLIST.write_text(
            json.dumps(
                {
                    "_comment": (
                        "Grandfathered uses of global parallel_state in megatron/core. "
                        "Entries are scope:kind:detail, with a :warned suffix for a read that "
                        "follows a fallback warning; duplicates count separate calls. "
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
        if converted:
            n = sum(len(v) for v in converted.values())
            print(
                f"\n{n} of them became guarded fallback(s) or now follow a fallback warning, "
                "which --update records:\n"
            )
            for rel, hits in sorted(converted.items()):
                for h in hits:
                    print(f"  {rel}  {h}")
        return 1

    print(
        f"OK: no new uses of global parallel_state ({total} grandfathered: "
        f"{counts[GUARDED]} guarded, {counts['warned']} warned, {counts['unwarned']} unwarned)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
