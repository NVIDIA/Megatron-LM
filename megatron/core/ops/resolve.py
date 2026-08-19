# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The one place a BackendSpecProvider is built.

Nothing here names an individual backend. Each family under ``megatron.core.ops`` declares its
own operations, its own backend table, and its own default, so adding a backend touches that
family and nothing else.
"""

from __future__ import annotations

import importlib
import warnings
from functools import lru_cache
from types import ModuleType
from typing import Mapping

from megatron.core.ops import determinism
from megatron.core.ops._availability import is_installed, require
from megatron.core.ops.operations import Operation
from megatron.core.ops.options import BackendOptions
from megatron.core.ops.spec_provider import BackendSpecProvider

__all__ = [
    "PRESETS",
    "backends_for",
    "build_spec_provider",
    "find_operation",
    "get_backend",
    "get_backend_spec_provider",
    "operations",
    "validate_backend",
]

#: The families that declare operations. Adding one is a milestone; adding a backend is not.
_FAMILIES = ("norm", "linear", "attention", "moe", "loss")

#: What --transformer-impl accepts. A preset names the backend each family should prefer;
#: a family that does not offer that name keeps its own default.
PRESETS = ("local", "transformer_engine", "inference_optimized")


@lru_cache(maxsize=None)
def _families() -> tuple[ModuleType, ...]:
    return tuple(importlib.import_module(f"megatron.core.ops.{name}") for name in _FAMILIES)


@lru_cache(maxsize=None)
def _family_of(operation: Operation) -> ModuleType:
    for family in _families():
        if family.FAMILY == operation.family:
            return family
    raise ValueError(f"No family declares '{operation.qualified_name}'")


def operations() -> tuple[Operation, ...]:
    """Every operation, in family order."""
    return tuple(op for family in _families() for op in family.OPERATIONS)


def find_operation(name: str) -> Operation:
    """Return the operation called ``name``, accepting ``method`` or ``family.method``."""
    matches = [op for op in operations() if name in (op.method, op.qualified_name)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        choices = ", ".join(op.method for op in operations())
        raise ValueError(f"Unknown operation '{name}'. Valid operations: {choices}")
    choices = ", ".join(op.qualified_name for op in matches)
    raise ValueError(f"Operation '{name}' is ambiguous. Use one of: {choices}")


def backends_for(operation: Operation) -> tuple[str, ...]:
    """Every backend name that can fill one operation."""
    return tuple(_family_of(operation).BACKENDS)


def _instantiate(backend: object, options: BackendOptions) -> object:
    """Build one backend, giving it the options only if it asked for them."""
    from_options = getattr(backend, "from_options", None)
    return from_options(options) if from_options is not None else backend()  # type: ignore[operator]


def _check_backend(
    family: ModuleType, backend_name: str, wanted_by: str, *, deterministic_mode: bool = False
) -> None:
    """The one place a backend name, its dependency, and its determinism are checked."""
    if backend_name not in family.BACKENDS:
        raise ValueError(
            f"Unknown backend '{backend_name}' for {wanted_by}. "
            f"Available backends: {', '.join(family.BACKENDS)}"
        )
    backend = family.BACKENDS[backend_name]

    requires = getattr(backend, "REQUIRES", None)
    if requires is not None and not is_installed(requires):
        raise ValueError(
            f"Backend '{backend_name}' for {wanted_by} requires '{requires}', "
            "which is not installed."
        )

    if not deterministic_mode:
        return
    declared = getattr(backend, "DETERMINISM", None)
    if declared == determinism.NONDETERMINISTIC:
        raise ValueError(
            f"Backend '{backend_name}' for {wanted_by} is not deterministic, and "
            "--deterministic-mode is on. Select a different backend or drop the flag."
        )
    if declared not in determinism.VALUES:
        raise ValueError(
            f"Backend '{backend_name}' for {wanted_by} does not declare DETERMINISM. "
            f"Declare one of: {', '.join(sorted(determinism.VALUES))}."
        )
    if declared == determinism.UNKNOWN:
        warnings.warn(
            f"Backend '{backend_name}' for {wanted_by} has not been established as "
            "deterministic, so --deterministic-mode cannot guarantee bit-exact results "
            "for that operation."
        )


def validate_backend(
    operation: Operation, backend_name: str, *, deterministic_mode: bool = False
) -> None:
    """Raise if ``backend_name`` cannot fill ``operation`` here.

    A backend declares what it needs as ``REQUIRES``, so no family repeats the check. Public
    so the command line can reject a bad ``--op-backend`` while it parses, rather than leaving
    it to fail at model build time.
    """
    _check_backend(
        _family_of(operation),
        backend_name,
        f"operation '{operation.method}'",
        deterministic_mode=deterministic_mode,
    )


def _owner_for(operation: Operation, backend_name: str, options: BackendOptions) -> object:
    validate_backend(operation, backend_name, deterministic_mode=options.deterministic_mode)
    return _instantiate(_family_of(operation).BACKENDS[backend_name], options)


def _preset_owners(options: BackendOptions) -> dict[Operation, object]:
    """Assign every operation from one preset: the family's entry for it, or its default."""
    if options.transformer_impl not in PRESETS:
        raise ValueError(
            f"unknown transformer_impl='{options.transformer_impl}'. "
            f"Valid choices: {', '.join(PRESETS)}"
        )
    owners: dict[Operation, object] = {}
    for family in _families():
        name = (
            options.transformer_impl
            if options.transformer_impl in family.BACKENDS
            else family.DEFAULT
        )
        # A named preset fails clearly when its backend cannot be used here.
        _check_backend(
            family,
            name,
            f"--transformer-impl {options.transformer_impl}",
            deterministic_mode=options.deterministic_mode,
        )
        backend = _instantiate(family.BACKENDS[name], options)
        owners.update(
            {
                operation: backend
                for operation in family.OPERATIONS
                if not operation.optional or callable(getattr(backend, operation.method, None))
            }
        )
    return owners


def _legacy_overrides(options: BackendOptions) -> dict[Operation, str]:
    """Ask each family what the settings older than --op-backend select for it.

    A family that has such settings exposes ``legacy_backends(options)``; the mapping lives
    with the family so its flag values and its backend names can be read together. Nothing
    about any particular family is known here.
    """
    overrides: dict[Operation, str] = {}
    for family in _families():
        translate = getattr(family, "legacy_backends", None)
        if translate is not None:
            overrides.update(translate(options))
    return overrides


def _override_owners(options: BackendOptions) -> dict[Operation, object]:
    """Resolve every operation an explicit selector claims, rejecting two claims on one slot."""
    legacy = _legacy_overrides(options)
    explicit = {
        find_operation(name): backend for name, backend in options.operation_backends.items()
    }

    contested = sorted(op.method for op in set(legacy) & set(explicit))
    if contested:
        raise ValueError(
            "Two settings select a backend for the same operation: "
            + ", ".join(
                f"'{name}' (from an existing setting and from --op-backend)" for name in contested
            )
            + ". Keep only one of them."
        )
    return {
        operation: _owner_for(operation, name, options)
        for operation, name in {**legacy, **explicit}.items()
    }


def _layered_owners(overlay: object, owners: dict[Operation, object]) -> dict[Operation, object]:
    """Give ``overlay`` the slots it implements, and leave the rest where they are.

    A partial backend layered over an assembled provider -- Kitchen is the one in tree --
    forwards the operations it does not own to a fallback. It cannot forward a slot that did
    not exist when it was written, so anything it does not define itself stays with the
    backend that already had it, rather than reaching an unowned stub.
    """
    return {
        operation: (
            overlay
            if getattr(type(overlay), operation.method, None)
            not in (None, getattr(BackendSpecProvider, operation.method, None))
            else owner
        )
        for operation, owner in owners.items()
    }


def build_spec_provider(options: BackendOptions) -> BackendSpecProvider:
    """Build the provider for ``options``.

    Selection order is fixed and total: the preset assigns every operation, Kitchen layers over
    that when enabled, and --op-backend wins last. Every optional dependency a selected backend
    needs is checked here, while the model is being built.
    """
    owners = _preset_owners(options)
    if options.use_kitchen:
        from megatron.core.ops.kitchen import kitchen_backend

        require("nvidia_kitchen")
        owners = _layered_owners(kitchen_backend(BackendSpecProvider(owners), options), owners)
    owners.update(_override_owners(options))
    return BackendSpecProvider(owners)


def get_backend_spec_provider(
    config: object, *, transformer_impl: str | None = None
) -> BackendSpecProvider:
    """Build the provider a TransformerConfig asks for."""
    return build_spec_provider(
        BackendOptions.from_config(config, transformer_impl=transformer_impl)
    )


def get_backend(transformer_impl: str, **options) -> BackendSpecProvider:
    """Build the provider for a preset name, with no config to read."""
    return build_spec_provider(BackendOptions(transformer_impl=transformer_impl, **options))
