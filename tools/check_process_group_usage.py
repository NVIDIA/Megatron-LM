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
This is a syntactic check, not data-flow analysis. It counts every reference to the functions
and attributes below, called or not, made through ``parallel_state``, ``mpu``, an import
(``import ... as``, ``from ... import``, a star import, ``megatron.core``'s included), an
assigned alias (``grid = parallel_state``, either side of a conditional expression or of ``and``
and ``or``, an assignment expression, a name declared ``global`` or ``nonlocal``, an attribute
such as ``self.grid``, or a class attribute through ``self``, ``cls`` and the class),
``getattr()``, ``setattr()`` or ``delattr()`` with a constant name,
``importlib.import_module()`` with a constant name, relative or not, or ``sys.modules``. A
reference sees the most recent import or alias of its name that precedes it in the same scope,
even if the name is reassigned later. It does not follow a module passed as an argument (a
parameter default included), returned by a function, stored in a container or bound by ``for``
or ``with ... as``, an import in a class body that methods reach through ``self``, nor a name
built at run time; an alias of a function is not followed either, because the reference that
makes it counts. An attribute alias is known by its spelling in the whole file: once one class
sets ``self.grid = parallel_state``, ``self.grid`` counts in every class of the file, which also
covers the subclasses that inherit it.

Each entry has one of five kinds:

* ``accessor``: a ``parallel_state`` group, rank or size accessor;
* ``function``: another deprecated ``parallel_state`` function: a pipeline-stage predicate, the
  virtual-pipeline rank or size, ``get_all_ranks``, a rank or size override, a dynamic
  context-parallel group map, or a memory-buffer alias;
* ``shim``: a collection of the global grid: ``ProcessGroupCollection.use_mpu_process_groups()``,
  which reads the same globals, or ``parallel_state.get_registered_process_groups()``, which
  returns the collection registered for the run. The registry is not deprecated, but Megatron
  Core takes its groups from the caller;
* ``guarded``: a helper of ``process_groups_config.py`` that reads the globals only when the
  caller omits its groups, and warns for that fallback (``GUARDED_HELPERS``):
  ``resolve_process_groups()``, ``resolve_gtp_remat_group()``, and
  ``ProcessGroupCollection.setup_process_groups_for_ddp()`` and
  ``setup_process_groups_for_optimizer()``. ``resolve_gtp_remat_group()`` does not warn when
  its caller passes ``owner=None`` because that caller has warned already. Their own reads are
  in an exempt file, so each call counts, whether or not the argument it forwards can be None;
* ``private``: a read, write or deletion of a private (``_``-prefixed) attribute of
  ``parallel_state``, such as ``_DATA_PARALLEL_GROUP`` or an ``_MPU_*`` override; the deprecated
  ``_set_global_memory_buffer()`` is a ``function``. That state is not an interface, so a
  fallback warning does not make it acceptable: a private entry is never warned.

A guarded call is still a read, so wrapping a fallback in ``resolve_process_groups()`` does not
remove it from the allowlist. A guarded call that replaces a removed read of another kind in the
same file and scope is a conversion, and ``--update`` swaps the entry. Any other new guarded call
fails like a new read.

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
import importlib.util
import json
import pathlib
import re
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
# get_nccl_options is not deprecated, get_registered_process_groups is the registry that replaces
# the accessors outside Megatron Core (REGISTRY_GETTERS), and the others are counted with
# DEPRECATED_FUNCTIONS.
NOT_ACCESSORS = {
    "get_nccl_options",
    "get_registered_process_groups",
    "get_all_ranks",
    "get_global_memory_buffer",
    "get_virtual_pipeline_model_parallel_rank",
    "get_virtual_pipeline_model_parallel_world_size",
}
# Registry getters: not deprecated, but they return the groups of the global grid that the run
# registered, so a core call counts as a shim. The test helpers that forbid the global grid stub
# them with the accessors (_returns_global_groups).
REGISTRY_GETTERS = {"get_registered_process_groups"}
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

# Entry kinds in --stats order. "guarded" calls warn for the fallback when they read the globals,
# apart from resolve_gtp_remat_group(owner=None), whose caller has warned already. "private"
# entries read or write the private state of parallel_state, and are never warned.
KINDS = ("accessor", "function", "shim", "guarded", "private")
GUARDED = "guarded"
PRIVATE = "private"

# The modules whose attributes are the functions and the state of the global grid: megatron.core
# also exports parallel_state as mpu.
PARALLEL_STATE_MODULES = ("megatron.core.parallel_state", "megatron.core.mpu")
SHIM = "megatron.core.process_groups_config.ProcessGroupCollection.use_mpu_process_groups"
# The binding that ``from module import *`` makes, under a name no identifier can have.
STAR = "*"
# The most times a file is visited to follow attribute aliases of attribute aliases.
MAX_ALIAS_PASSES = 10

# The helpers of process_groups_config.py that read the global grid when the caller omits its
# groups, and the detail of their entries. The file is exempt, so their reads count at each call.
# A helper whose fallback is removed leaves this map.
GUARDED_HELPERS = {
    "megatron.core.process_groups_config.resolve_process_groups": "resolve_process_groups",
    "megatron.core.process_groups_config.resolve_gtp_remat_group": "resolve_gtp_remat_group",
    "megatron.core.process_groups_config.ProcessGroupCollection.setup_process_groups_for_ddp": (
        "setup_process_groups_for_ddp"
    ),
    (
        "megatron.core.process_groups_config.ProcessGroupCollection"
        ".setup_process_groups_for_optimizer"
    ): "setup_process_groups_for_optimizer",
}

# The paths an assigned alias is followed to: the modules and the class whose attributes the check
# counts, the packages that contain them, and the module lookups. Any other assignment binds a
# local value, which does not hide an earlier import.
ALIAS_TARGETS = {
    "megatron",
    "megatron.core",
    *PARALLEL_STATE_MODULES,
    "megatron.core.process_groups_config",
    "megatron.core.process_groups_config.ProcessGroupCollection",
    "importlib",
    "importlib.import_module",
    "sys",
    "sys.modules",
}

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


def _returns_global_groups(name: str) -> bool:
    """True for a parallel_state accessor or registry getter: the functions that return groups of
    the global grid, which the test helpers that forbid the global grid stub."""
    return _is_deprecated_accessor(name) or name in REGISTRY_GETTERS


def _is_private(name: str) -> bool:
    """True for a private attribute such as ``_DATA_PARALLEL_GROUP``, but not a dunder."""
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def _classify(path):
    """Return ``kind:detail`` for the dotted path of a function or attribute the check counts."""
    module, _, name = path.rpartition(".")
    if module in PARALLEL_STATE_MODULES:
        if _is_deprecated_accessor(name):
            return f"accessor:parallel_state.{name}"
        if name in DEPRECATED_FUNCTIONS:
            return f"function:parallel_state.{name}"
        if name in REGISTRY_GETTERS:
            return f"shim:parallel_state.{name}"
        if _is_private(name):
            return f"{PRIVATE}:parallel_state.{name}"
    elif path == SHIM:
        return "shim:use_mpu_process_groups"
    elif path in GUARDED_HELPERS:
        return f"{GUARDED}:{GUARDED_HELPERS[path]}"
    return None


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


def _scope_bindings(node):
    """Collect the binding events of one scope without leaking nested scopes into their parent.

    Returns ``({name: [(position, target), ...]}, declared)``. ``target`` is the dotted path a name
    was imported from, the expression an assignment binds it to, which may be a module alias, or
    ``""`` for any other binding (parameter, nested definition). A star import binds ``STAR`` to
    its module. ``declared`` holds the names that the body declares ``global`` or ``nonlocal``,
    whose events belong to another scope (_moved_bindings).
    """

    class Bindings(ast.NodeVisitor):
        """Collect names bound by the current module, function, or class."""

        def __init__(self):
            self.events = {}
            self.declared = set()

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
                if alias.name == "*":
                    self.bind(STAR, node, module)
                else:
                    self.bind(alias.asname or alias.name, node, f"{module}.{alias.name}")

        def visit_Name(self, node):
            """Keep simple assignments from being mistaken for imported module aliases."""
            if isinstance(node.ctx, ast.Store):
                self.bind(node.id, node, "")

        def visit_Assign(self, node):
            """Record the value of each name an assignment binds, which may alias a module."""
            for target in node.targets:
                for item, value in _assigned_pairs(target, node.value):
                    self.assign(item, value, node)
            self.visit(node.value)

        def visit_AnnAssign(self, node):
            """Record an annotated assignment like a plain one."""
            if node.value is None:
                self.visit(node.target)
            else:
                self.assign(node.target, node.value, node)
                self.visit(node.value)

        def visit_NamedExpr(self, node):
            """Record an assignment expression like a plain assignment."""
            self.assign(node.target, node.value, node)
            self.visit(node.value)

        def assign(self, target, value, statement):
            """Bind a name to the expression assigned to it, from the end of ``statement`` on,
            so that the expression itself does not see the binding."""
            if isinstance(target, ast.Name):
                position = (statement.end_lineno, statement.end_col_offset)
                self.events.setdefault(target.id, []).append((position, value))
            else:
                self.visit(target)

        def visit_FunctionDef(self, node):
            """Bind a nested definition's name without collecting imports from its body."""
            self.bind(node.name, node, "")

        visit_AsyncFunctionDef = visit_FunctionDef
        visit_ClassDef = visit_FunctionDef

        def visit_Global(self, node):
            """Record the names that the body binds in another scope."""
            self.declared.update(node.names)

        visit_Nonlocal = visit_Global

        def visit_Lambda(self, node):
            """Do not collect bindings from an anonymous nested scope."""
            pass

        def visit_ListComp(self, node):
            """Bind the target of each assignment expression in a comprehension, which binds it
            in the enclosing scope; its loop targets and lambda arguments have their own scope."""
            nodes = [node]
            while nodes:
                child = nodes.pop()
                if isinstance(child, ast.NamedExpr):
                    self.assign(child.target, child.value, child)
                if not isinstance(child, ast.Lambda):
                    nodes.extend(ast.iter_child_nodes(child))

        visit_SetComp = visit_ListComp
        visit_DictComp = visit_ListComp
        visit_GeneratorExp = visit_ListComp

    bindings = Bindings()
    if hasattr(node, "args"):
        args = node.args
        for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg):
            if arg:
                # Parameters are bound before any statement of the body runs.
                bindings.events.setdefault(arg.arg, []).append(((0, -1), ""))
    for statement in node.body:
        bindings.visit(statement)
    return bindings.events, bindings.declared


def _moved_bindings(tree, source):
    """Return the binding events that ``global`` and ``nonlocal`` declarations move.

    Returns ``{id(scope node): {name: [(position, target), ...]}}``. A function or class body
    that declares a name ``global`` binds it in the module, and one that declares it ``nonlocal``
    binds it in the nearest enclosing function that binds it. The events keep their positions and
    are resolved in the scope they move to.
    """
    moved = {}
    # Only a source with one of the words can declare a name; skip the walk of the others.
    if not re.search(r"\b(?:global|nonlocal)\b", source):
        return moved
    collected = {}

    def bindings(scope):
        if id(scope) not in collected:
            collected[id(scope)] = _scope_bindings(scope)
        return collected[id(scope)]

    def owner(name, enclosing):
        """Return the nearest function of ``enclosing`` that binds ``name`` itself, or None."""
        for scope in reversed(enclosing):
            if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
                events, declared = bindings(scope)
                if name in events and name not in declared:
                    return scope
        return None

    def visit(statements, chain):
        """Visit the declarations among ``statements``, nested blocks included; ``chain`` lists
        the module, function and class nodes that enclose them, innermost last."""
        for statement in statements:
            if isinstance(statement, (ast.Global, ast.Nonlocal)) and len(chain) > 1:
                events = bindings(chain[-1])[0]
                for name in statement.names:
                    if isinstance(statement, ast.Global):
                        target = tree
                    else:
                        target = owner(name, chain[1:-1])
                    if target is not None:
                        moved.setdefault(id(target), {}).setdefault(name, []).extend(
                            events.get(name, [])
                        )
            elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                visit(statement.body, chain + [statement])
            else:
                # The blocks of compound statements, and of except handlers and match cases.
                for field in ("body", "orelse", "finalbody", "handlers", "cases"):
                    visit(getattr(statement, field, ()), chain)

    visit(tree.body, [tree])
    return moved


class _Scope:
    """The binding events of a module, class or function body.

    ``parent`` is the nearest enclosing scope whose names the body sees: a function nested in a
    class sees the names of the scope around the class, not those of the class body. ``moved``
    holds the events that ``global`` and ``nonlocal`` declarations move (_moved_bindings).
    """

    def __init__(self, kind, node, parent, moved):
        self.kind = kind
        self.events, declared = _scope_bindings(node)
        if kind != "module":
            # The body binds and reads these names in the module or an enclosing function.
            for name in declared:
                self.events.pop(name, None)
        for name, events in moved.get(id(node), {}).items():
            self.events.setdefault(name, []).extend(events)
        self.parent = parent
        # What each name refers to when a nested scope runs, once resolved.
        self.seen = {}


def _string(node):
    """Return the value of a string constant, or None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _violations_in(path: pathlib.Path):
    """Return (lineno, scope:kind:detail) pairs for global process-group reads in one file."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    moved = _moved_bindings(tree, source)

    class Visitor(ast.NodeVisitor):
        """Attach a stable enclosing scope to each reference to the global grid."""

        def __init__(self, attribute_aliases):
            self.scope = []
            self.hits = []
            self.scopes = [_Scope("module", tree, None, moved)]
            # What each binding event refers to, once resolved (_target).
            self.targets = {}
            # Whether a fallback warning dominates the statement being visited.
            self.warned = False
            # Attribute spellings, such as "self.grid", that alias a module or class of
            # megatron: those found by the previous pass, and those this pass finds.
            self.attribute_aliases = attribute_aliases
            self.found_aliases = {}

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
            current = self.scopes[-1]
            parent = current.parent if current.kind == "class" else current
            self.scopes.append(_Scope(kind, node, parent, moved))
            self.scope.append(node.name)
            # A function body runs when it is called, not where it is defined; a class body runs
            # in place.
            outer_warned = self.warned
            if kind == "function":
                self.warned = False
            self._visit_block(node.body)
            self.warned = outer_warned
            self.scope.pop()
            self.scopes.pop()

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

        def _target(self, target, scope):
            """Return the dotted path that a binding event of ``scope`` refers to.

            An assignment is an alias only of one of ALIAS_TARGETS. An alias of a counted
            function is not followed: it is a reference, which counts where the assignment is.
            Each event is resolved once: the events of ``x = x.y`` repeated in one scope would
            otherwise resolve each earlier one again, in time exponential in their number.
            """
            if isinstance(target, str):
                return target
            key = (id(target), id(scope))
            if key not in self.targets:
                # An event that its own resolution reaches again refers to nothing.
                self.targets[key] = ""
                path = self._path(target, scope)
                self.targets[key] = path if path in ALIAS_TARGETS else ""
            return self.targets[key]

        def _binding_at(self, events, position, scope):
            """Return what a name refers to at ``position`` in the scope that binds it.

            A reference sees the most recent import or alias of the name that precedes it, even
            when the name is reassigned afterwards: the accessor has already run against the
            global grid by then. Other bindings never hide an earlier import or alias (a
            conservative choice for branches and loops), but a name bound in the scope without a
            preceding one is a local value, not a module.
            """
            for start, target in sorted(events, key=lambda event: event[0], reverse=True):
                if start < position:
                    resolved = self._target(target, scope)
                    if resolved:
                        return resolved
            return ""

        def _seen_by_nested_scopes(self, name, scope):
            """Return what ``name`` of ``scope`` refers to when a nested scope runs later."""
            if name not in scope.seen:
                scope.seen[name] = ""
                events = sorted(scope.events[name], key=lambda event: event[0], reverse=True)
                for _, target in events:
                    resolved = self._target(target, scope)
                    if resolved:
                        scope.seen[name] = resolved
                        break
            return scope.seen[name]

        def _enclosing(self, name, scope):
            """Return what ``name`` refers to in the scopes that enclose ``scope``, or None."""
            enclosing = scope.parent
            while enclosing is not None:
                if name in enclosing.events:
                    return self._seen_by_nested_scopes(name, enclosing)
                enclosing = enclosing.parent
            return None

        def _lookup(self, name, position, scope):
            """Return the dotted path that ``name`` refers to at ``position`` in ``scope``."""
            if name in scope.events:
                return self._binding_at(scope.events[name], position, scope)
            found = self._enclosing(name, scope)
            if found is not None:
                return found
            # A star import binds the public names of its module; follow only those the check
            # counts or follows, so that builtins such as getattr keep their meaning.
            star = self._binding_at(scope.events.get(STAR, []), position, scope)
            star = star or self._enclosing(STAR, scope)
            if star == "megatron.core" and name == "mpu":
                # megatron.core.__all__ exports parallel_state but not mpu.
                return name
            path = f"{star}.{name}"
            if star and not name.startswith("_") and (_classify(path) or path in ALIAS_TARGETS):
                return path
            return name

        def _path(self, node, scope=None):
            """Return the dotted path that the expression ``node`` refers to, or ``""``."""
            scope = scope or self.scopes[-1]
            if isinstance(node, ast.Name):
                return self._lookup(node.id, (node.lineno, node.col_offset), scope)
            if isinstance(node, ast.Attribute):
                if self.attribute_aliases:
                    alias = self.attribute_aliases.get(ast.unparse(node))
                    if alias:
                        return alias
                owner = self._path(node.value, scope)
                return f"{owner}.{node.attr}" if owner else ""
            if isinstance(node, (ast.IfExp, ast.BoolOp)):
                # Either side of a conditional expression, and any operand of ``and`` and ``or``,
                # can be its value: follow the first that is a module or class the check follows.
                options = (node.body, node.orelse) if isinstance(node, ast.IfExp) else node.values
                paths = (self._path(option, scope) for option in options)
                return next((path for path in paths if path in ALIAS_TARGETS), "")
            if isinstance(node, ast.Call):
                return self._named_attribute(node, ("getattr",), scope) or self._imported_module(
                    node, scope
                )
            if (
                isinstance(node, ast.Subscript)
                and _string(node.slice)
                and self._path(node.value, scope) == "sys.modules"
            ):
                return node.slice.value
            return ""

        def _named_attribute(self, node, functions, scope=None):
            """Return the dotted path of the attribute that a call of one of ``functions``, such
            as ``getattr(owner, "name"[, default])``, names with a constant, or ``""``."""
            if (
                isinstance(node, ast.Call)
                and not node.keywords
                and len(node.args) in (2, 3)
                and _string(node.args[1])
                and self._path(node.func, scope) in functions
            ):
                owner = self._path(node.args[0], scope)
                return f"{owner}.{node.args[1].value}" if owner else ""
            return ""

        def _imported_module(self, node, scope):
            """Return the module that ``importlib.import_module(name[, package])`` imports, if
            its name, and the package of a relative name, are constants, or ``""``."""
            keywords = {keyword.arg: keyword.value for keyword in node.keywords}
            name = _string(node.args[0] if node.args else keywords.get("name"))
            if not name or self._path(node.func, scope) != "importlib.import_module":
                return ""
            package = _string(node.args[1] if len(node.args) > 1 else keywords.get("package"))
            try:
                return importlib.util.resolve_name(name, package)
            except ImportError:
                # A relative name without a package, or beyond the top-level package.
                return ""

        def _calls(self, node, function):
            """True if ``node`` calls ``function``, given as its dotted import path."""
            return isinstance(node, ast.Call) and self._path(node.func) == function

        def _reference(self, node):
            """Record ``node`` if it refers to a function or attribute that the check counts, or
            writes or deletes one with ``setattr()`` or ``delattr()``."""
            path = self._path(node)
            if not path and isinstance(node, ast.Call):
                path = self._named_attribute(node, ("setattr", "delattr"))
            identity = _classify(path)
            if identity:
                scope = ".".join(self.scope) or "<module>"
                if self.warned and identity.split(":", 1)[0] not in (GUARDED, PRIVATE):
                    identity += WARNED
                self.hits.append((node.lineno, f"{scope}:{identity}"))

        def visit_Name(self, node):
            """Record a name that refers to a counted function, such as a direct import."""
            if isinstance(node.ctx, ast.Load):
                self._reference(node)

        def visit_Attribute(self, node):
            """Record an attribute of the global grid, whether it is read, written or called."""
            self._reference(node)
            self.generic_visit(node)

        def visit_Call(self, node):
            """Record ``getattr()``, ``setattr()`` or ``delattr()`` of a counted function or
            attribute with a constant name."""
            self._reference(node)
            self.generic_visit(node)

        def visit_Assign(self, node):
            """Record the attribute aliases that an assignment makes, then visit it."""
            for target in node.targets:
                for item, value in _assigned_pairs(target, node.value):
                    self._record_alias(item, value)
            self.generic_visit(node)

        def visit_AnnAssign(self, node):
            """Record an annotated attribute alias like a plain one."""
            if node.value is not None:
                self._record_alias(node.target, node.value)
            self.generic_visit(node)

        def _record_alias(self, target, value):
            """Record ``self.grid = parallel_state``, and ``grid = parallel_state`` in a class
            body, which methods reach through ``self``, ``cls`` and the class."""
            path = self._path(value)
            if path not in ALIAS_TARGETS:
                return
            if isinstance(target, ast.Attribute):
                spellings = [ast.unparse(target)]
            elif isinstance(target, ast.Name) and self.scopes[-1].kind == "class":
                spellings = [f"{owner}.{target.id}" for owner in ("self", "cls", self.scope[-1])]
            else:
                return
            for spelling in spellings:
                self.found_aliases.setdefault(spelling, path)

    # An attribute alias can be assigned after the code that uses it, or alias another one: visit
    # again with the aliases found until no new one appears.
    aliases = {}
    for _ in range(MAX_ALIAS_PASSES):
        visitor = Visitor(aliases)
        visitor.visit(tree)
        if visitor.found_aliases == aliases:
            break
        aliases = visitor.found_aliases
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

    Each removed read of a kind other than guarded is matched at most once. Returns
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
            f"{total - counts[GUARDED]} accessor, function, shim and private use(s), by whether a "
            "fallback warning precedes them"
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
        if any(_kind(hit) == PRIVATE for hits in added.values() for hit in hits):
            print(
                "A 'private' use reads or writes the internal state of parallel_state. That "
                "state is not an interface, so not even a warned fallback may use it."
            )
        print(
            "Note that neither ProcessGroupCollection.use_mpu_process_groups() nor "
            "parallel_state.get_registered_process_groups() is a valid replacement -- they "
            "return the same global groups.\n"
            "resolve_process_groups() and the other guarded helpers of process_groups_config.py "
            "(resolve_gtp_remat_group(), ProcessGroupCollection.setup_process_groups_for_ddp() "
            "and setup_process_groups_for_optimizer()) are counted too: a call may only replace "
            "a read in the same function, for callers that do not pass their groups yet.\n"
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
