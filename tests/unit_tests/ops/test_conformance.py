# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Every registered backend meets its family's contract.

Parametrized off the real family tables, so a backend cannot be added without being covered.
"""

import importlib

import pytest

from megatron.core.ops import PRESETS, BackendOptions, build_spec_provider, operations
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
    from packaging.requirements import Requirement

    from megatron.core.ops._availability import is_installed

    backend = family.BACKENDS[backend_name]
    requires = getattr(backend, "REQUIRES", None)
    if requires is not None and not is_installed(Requirement(requires).name):
        pytest.skip(f"{requires} is not installed")
    try:
        return _instantiate(backend, _OPTIONS)
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
    backend = _build_or_skip(family, backend_name)
    # An optional slot may legitimately be unimplemented -- a fusion fills only the handover
    # point it covers -- so pick one this backend actually fills.
    operation = next(op for op in family.OPERATIONS if callable(getattr(backend, op.method, None)))
    # A fusion is only selectable where the operations it spans can be served, so build it on
    # a preset that can serve them.
    spans = getattr(family.BACKENDS[backend_name], "FUSES", {})
    preset = next((name for name in spans.values() if name in PRESETS), "local")

    provider = build_spec_provider(
        BackendOptions(transformer_impl=preset, operation_backends={operation.method: backend_name})
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


@pytest.mark.parametrize(("family_name", "backend_name"), _REGISTERED)
def test_backend_declares_its_determinism(family_name, backend_name):
    """Every backend states whether it is bit-exact, so 'nobody checked' cannot look like 'safe'."""
    from megatron.core.ops import determinism

    backend = _family(family_name).BACKENDS[backend_name]
    declared = getattr(backend, "DETERMINISM", None)
    assert declared in determinism.VALUES, (
        f"{family_name}.{backend_name} declares DETERMINISM={declared!r}; "
        f"expected one of {sorted(determinism.VALUES)}"
    )


def test_deterministic_mode_rejects_a_nondeterministic_backend():
    """--deterministic-mode must not silently select a backend known not to be bit-exact."""
    from megatron.core.ops import determinism

    options = BackendOptions(
        transformer_impl="local",
        deterministic_mode=True,
        operation_backends={"vocab_parallel_cross_entropy": "megatron_fused"},
    )
    assert _family("loss").BACKENDS["megatron_fused"].DETERMINISM == determinism.NONDETERMINISTIC
    with pytest.raises(ValueError, match="not deterministic"):
        build_spec_provider(options)


def test_deterministic_mode_allows_a_deterministic_backend():
    options = BackendOptions(
        transformer_impl="local",
        deterministic_mode=True,
        operation_backends={"vocab_parallel_cross_entropy": "megatron"},
    )
    assert build_spec_provider(options).vocab_parallel_cross_entropy() is not None


def test_a_version_constraint_in_requires_is_enforced(monkeypatch):
    """REQUIRES may pin a minimum, and the check happens before anything is built."""
    from megatron.core.ops import resolve
    from megatron.core.ops.mlp import BACKENDS as MLP_BACKENDS

    assert MLP_BACKENDS["te_op_fuser"].REQUIRES == "transformer_engine>=1.13.0"

    from packaging.version import Version

    monkeypatch.setattr(resolve, "is_installed", lambda name: True)
    monkeypatch.setattr(resolve, "installed_version", lambda name: Version("1.12.0"))
    with pytest.raises(ValueError, match="requires 'transformer_engine>=1.13.0'; found 1.12"):
        build_spec_provider(
            BackendOptions(
                transformer_impl="local", operation_backends={"mlp_module": "te_op_fuser"}
            )
        )


class TestFusionDeclarations:
    """A backend that reaches across operations says so, and the resolver holds it to that."""

    @staticmethod
    def _op_fuser(**kwargs):
        return BackendOptions(transformer_impl="transformer_engine", use_te_op_fuser=True, **kwargs)

    def test_a_fusion_declares_the_operations_it_spans(self):
        from megatron.core.ops.linear import COLUMN_PARALLEL_LAYER_NORM_LINEAR
        from megatron.core.ops.mlp import BACKENDS as MLP_BACKENDS

        spans = MLP_BACKENDS["te_op_fuser"].FUSES
        assert spans[COLUMN_PARALLEL_LAYER_NORM_LINEAR] == "transformer_engine"

    def test_a_neighbouring_slot_it_cannot_work_with_is_refused(self):
        """Previously this built a spec that died with a type error during construction."""
        with pytest.raises(ValueError, match="spans 'column_parallel_layer_norm_linear'"):
            build_spec_provider(
                self._op_fuser(operation_backends={"column_parallel_layer_norm_linear": "local"})
            )

    def test_a_preset_that_cannot_serve_the_fusion_is_refused(self):
        with pytest.raises(ValueError, match="spans"):
            build_spec_provider(BackendOptions(transformer_impl="local", use_te_op_fuser=True))

    def test_the_matching_preset_is_accepted(self):
        from megatron.core.extensions.transformer_engine import HAVE_TE

        if not HAVE_TE:
            pytest.skip("Transformer Engine is required")
        assert build_spec_provider(self._op_fuser()).mlp_module() is not None
