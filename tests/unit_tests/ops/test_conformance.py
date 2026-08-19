# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Every registered backend meets its family's contract.

Parametrized off the real family tables, so a backend cannot be added without being covered.
"""

import importlib

import pytest

from megatron.core.ops import BackendOptions, build_spec_provider, operations
from megatron.core.ops.resolve import _FAMILIES, _instantiate

_OPTIONS = BackendOptions(transformer_impl="local")

_REGISTERED = [
    (family_name, backend_name)
    for family_name in _FAMILIES
    for backend_name in importlib.import_module(f"megatron.core.ops.{family_name}").BACKENDS
]


def _family(name):
    return importlib.import_module(f"megatron.core.ops.{name}")


def _build_or_skip(family, backend_name):
    try:
        return _instantiate(family.BACKENDS[backend_name], _OPTIONS)
    except ImportError as error:
        pytest.skip(str(error))


@pytest.mark.parametrize(("family_name", "backend_name"), _REGISTERED)
def test_backend_fills_every_slot_its_family_declares(family_name, backend_name):
    """A backend registered in a family has to answer that family's whole contract."""
    family = _family(family_name)
    backend = _build_or_skip(family, backend_name)

    missing = [
        op.method
        for op in family.OPERATIONS
        if not op.optional and not callable(getattr(backend, op.method, None))
    ]
    assert not missing, f"{family_name}.{backend_name} does not implement {missing}"


@pytest.mark.parametrize(("family_name", "backend_name"), _REGISTERED)
def test_backend_is_selectable_by_name(family_name, backend_name):
    """Every table entry is reachable through --op-backend, and owns its family's slots."""
    family = _family(family_name)
    _build_or_skip(family, backend_name)
    # An optional slot may legitimately be unimplemented, so pick one every backend must fill.
    operation = next(op for op in family.OPERATIONS if not op.optional)

    provider = build_spec_provider(
        BackendOptions(
            transformer_impl="local", operation_backends={operation.method: backend_name}
        )
    )

    owner = provider._owners[operation]
    assert (
        type(owner).__name__ == type(_instantiate(family.BACKENDS[backend_name], _OPTIONS)).__name__
    )


def test_every_family_declares_its_defaults_and_operations():
    """The resolver relies on these three names; a new family must supply all of them."""
    for family_name in _FAMILIES:
        family = _family(family_name)
        assert family.OPERATIONS, f"{family_name} declares no operations"
        assert family.BACKENDS, f"{family_name} declares no backends"
        assert family.DEFAULT in family.BACKENDS, f"{family_name}.DEFAULT is not in BACKENDS"
        assert all(op.family == family.FAMILY for op in family.OPERATIONS)


def test_operation_names_are_unique_across_families():
    """--op-backend takes a bare method name, so two families must not both claim one."""
    names = [operation.method for operation in operations()]
    assert len(names) == len(set(names)), f"duplicate operation names: {names}"


def test_a_family_may_live_in_a_subfolder():
    """Sub-families like ``attention.dsa`` resolve by their dotted module path.

    Pins the mechanism that lets one family split into sub-folders with disjoint backend
    tables, so DSA cannot be offered a backend that only core attention has.
    """
    from megatron.core.ops.operations import Operation
    from megatron.core.ops.resolve import _family_of

    for family_name in _FAMILIES:
        family = _family(family_name)
        assert family.FAMILY == family_name, "FAMILY must be the module path under ops"
        probe = Operation(family_name, family.OPERATIONS[0].method)
        assert _family_of(probe) is family

    dotted = [name for name in _FAMILIES if "." in name]
    for name in dotted:
        assert importlib.import_module(f"megatron.core.ops.{name}") is _family(name)
