# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import dataclasses
from functools import partial
from typing import Optional, Protocol, Union

import pytest

from megatron.core.transformer.spec_utils import (
    ModuleSpec,
    build_module,
    get_param,
    get_submodules,
    set_param,
    try_get_constructed_type,
)


def dummy_method(x: int, y: str) -> dict:
    return {"x": x, "y": y}


class ExampleA:
    def __init__(self, x: int, y: str):
        self.x = x
        self.y = y


class TestBuildModule:
    """Unit tests for `build_module` function."""

    def test_build_module_from_function(self):
        """Test building module from a FunctionType."""

        assert build_module(dummy_method) == dummy_method
        assert build_module(ModuleSpec(module=dummy_method)) == dummy_method

    def test_build_module_from_type(self):
        """Test building module from a type."""

        class Empty:
            pass

        assert isinstance(build_module(Empty), Empty)
        assert isinstance(build_module(ModuleSpec(module=Empty)), Empty)

    def test_build_module_by_import(self):
        """Test building module by importing from a string."""
        assert (
            type(
                build_module(
                    ModuleSpec(module=('megatron.core.transformer.identity_op', 'IdentityOp'))
                )
            ).__name__
            == 'IdentityOp'
        )

    def test_build_module_with_params(self):
        """Test building module with parameters."""

        build_time = build_module(ExampleA, 1, 'abc')
        assert isinstance(build_time, ExampleA)
        assert build_time.x == 1
        assert build_time.y == 'abc'

        by_spec = build_module(ModuleSpec(module=ExampleA, params={'x': 2, 'y': 'def'}))
        assert isinstance(by_spec, ExampleA)
        assert by_spec.x == 2
        assert by_spec.y == 'def'

        mixed = build_module(ModuleSpec(module=ExampleA, params={'y': 'ghi'}), 3)
        assert isinstance(mixed, ExampleA)
        assert mixed.x == 3
        assert mixed.y == 'ghi'

    def test_build_module_by_call(self):
        """Test building module by calling a ModuleSpec instance."""

        by_spec = ModuleSpec(module=ExampleA, params={'x': 2, 'y': 'def'})()
        assert isinstance(by_spec, ExampleA)
        assert by_spec.x == 2
        assert by_spec.y == 'def'

        mixed = ModuleSpec(module=ExampleA, params={'y': 'ghi'})(3)
        assert isinstance(mixed, ExampleA)
        assert mixed.x == 3
        assert mixed.y == 'ghi'


class OtherChild:
    def __init__(self, x: int):
        self.x = x


class ABuilder(Protocol):
    def __call__(self, x: int) -> ExampleA: ...


@dataclasses.dataclass
class BSubmodules:
    x: ModuleSpec | type
    y: ABuilder


class ExampleB:
    def __init__(self, submodules: BSubmodules, z: int):
        self.x = build_module(submodules.x, x=10)
        self.y = submodules.y(x=10)
        self.z = z


class TestGetSubmodules:
    """Test that the getter utilities work as expected."""

    def test_get_submodules_missing(self):
        """Test getting submodules from a spec without one."""
        with pytest.raises(ValueError):
            get_submodules(dummy_method)
        with pytest.raises(ValueError):
            get_submodules(ExampleA)
        with pytest.raises(KeyError):
            get_submodules(partial(ExampleA, x=1, y='test'))

    def test_get_submodules_module_spec(self):
        """Test getting submodules from a spec."""
        assert get_submodules(ModuleSpec(module=dummy_method)) is None
        assert get_submodules(ModuleSpec(module=ExampleA)) is None
        submodules = BSubmodules(x=OtherChild, y=partial(ExampleA, y='test'))
        assert (
            get_submodules(ModuleSpec(module=ExampleB, submodules=submodules, params={'z': 123}))
            == submodules
        )

    def test_get_submodules_partial(self):
        """Test getting submodules from a use of partial."""
        submodules = BSubmodules(x=OtherChild, y=partial(ExampleA, y='test'))
        assert get_submodules(partial(ExampleB, submodules=submodules, z=123)) == submodules


class TestGetParam:
    """Test that the `get_param` utility works as expected."""

    def test_get_param_missing(self):
        """Test getting a param from a spec that does not carry one."""
        with pytest.raises(ValueError):
            get_param(dummy_method, 'x')
        with pytest.raises(ValueError):
            get_param(ExampleA, 'x')
        with pytest.raises(KeyError):
            get_param(partial(ExampleA, x=1, y='test'), 'z')
        with pytest.raises(KeyError):
            get_param(ModuleSpec(module=ExampleA, params={'x': 1}), 'y')

    def test_get_param_module_spec(self):
        """Test getting a param from a ModuleSpec."""
        spec = ModuleSpec(module=ExampleA, params={'x': 1, 'y': 'test'})
        assert get_param(spec, 'x') == 1
        assert get_param(spec, 'y') == 'test'

    def test_get_param_partial(self):
        """Test getting a param from a `partial`."""
        p = partial(ExampleA, x=1, y='test')
        assert get_param(p, 'x') == 1
        assert get_param(p, 'y') == 'test'

    def test_get_param_submodules(self):
        """Test that `get_param('submodules', ...)` delegates to `get_submodules`."""
        submodules = BSubmodules(x=OtherChild, y=partial(ExampleA, y='test'))
        spec = ModuleSpec(module=ExampleB, submodules=submodules, params={'z': 123})
        assert get_param(spec, 'submodules') is submodules

        p = partial(ExampleB, submodules=submodules, z=123)
        assert get_param(p, 'submodules') is submodules

        # A spec with no submodules should still raise ValueError via get_submodules.
        with pytest.raises(ValueError):
            get_param(dummy_method, 'submodules')


class TestSetParam:
    """Test that the `set_param` utility works as expected."""

    def test_set_param_module_spec(self):
        """Test setting a param on a ModuleSpec."""
        spec = ModuleSpec(module=ExampleA, params={'x': 1, 'y': 'old'})
        set_param(spec, 'y', 'new')
        assert spec.params == {'x': 1, 'y': 'new'}

        # Also covers inserting a key that wasn't already in `params`.
        set_param(spec, 'z', 99)
        assert spec.params == {'x': 1, 'y': 'new', 'z': 99}

        # Round-trip via get_param.
        assert get_param(spec, 'y') == 'new'
        assert get_param(spec, 'z') == 99

    def test_set_param_partial(self):
        """Test setting a param on a `partial`."""
        p = partial(ExampleA, x=1, y='old')
        set_param(p, 'y', 'new')
        assert p.keywords == {'x': 1, 'y': 'new'}

        # Inserting a previously absent keyword.
        set_param(p, 'extra', 42)
        assert p.keywords['extra'] == 42

        # Round-trip via get_param.
        assert get_param(p, 'y') == 'new'
        assert get_param(p, 'extra') == 42

    def test_set_param_missing(self):
        """Test setting a param on a spec without a `params`/`keywords` slot."""
        with pytest.raises(ValueError):
            set_param(dummy_method, 'x', 1)
        with pytest.raises(ValueError):
            set_param(ExampleA, 'x', 1)

    def test_set_param_submodules_rejected(self):
        """`set_param` must refuse to overwrite `submodules` for any spec type."""
        submodules = BSubmodules(x=OtherChild, y=partial(ExampleA, y='test'))
        spec = ModuleSpec(module=ExampleB, submodules=submodules, params={'z': 123})
        with pytest.raises(ValueError):
            set_param(spec, 'submodules', BSubmodules(x=OtherChild, y=partial(ExampleA, y='x')))
        # The original submodules must be untouched.
        assert spec.submodules is submodules

        p = partial(ExampleB, submodules=submodules, z=123)
        with pytest.raises(ValueError):
            set_param(p, 'submodules', BSubmodules(x=OtherChild, y=partial(ExampleA, y='x')))
        assert p.keywords['submodules'] is submodules

        # Even for unsupported spec types, the submodules guard fires first.
        with pytest.raises(ValueError):
            set_param(dummy_method, 'submodules', submodules)


# Helpers for try_get_constructed_type tests.
def _func_returning_example_a() -> ExampleA:
    return ExampleA(0, 'x')


def _func_no_annotation():
    pass


def _func_returns_dict() -> dict:
    return {}


def _func_returns_string_forward_ref() -> 'ExampleA':  # noqa: F821 - intentional forward ref
    return ExampleA(0, 'x')


def _func_returning_union() -> Union[ExampleA, OtherChild]:
    return ExampleA(0, 'x')


def _func_returning_optional() -> Optional[ExampleA]:
    return None


class _CallableInstanceAnnotated:
    """A callable instance with a concrete return-type annotation on `__call__`."""

    def __call__(self) -> ExampleA:
        return ExampleA(0, 'x')


class _CallableInstanceUnannotated:
    """A callable instance with no return-type annotation on `__call__`."""

    def __call__(self):
        return ExampleA(0, 'x')


class _FactoryHost:
    """Hosts classmethod factories with and without return-type annotations."""

    @classmethod
    def make_annotated(cls) -> ExampleA:
        return ExampleA(0, 'x')

    @classmethod
    def make_unannotated(cls):
        return ExampleA(0, 'x')


class TestTryGetConstructedType:
    """Tests for `try_get_constructed_type`."""

    def test_returns_type_for_plain_class(self):
        """A plain class is returned as the constructed type."""
        assert try_get_constructed_type(ExampleA) is ExampleA

    def test_unwraps_module_spec(self):
        """A `ModuleSpec` wrapping a type unwraps to that type."""
        assert try_get_constructed_type(ModuleSpec(module=ExampleA)) is ExampleA

    def test_unwraps_partial_of_type(self):
        """`partial(SomeType, ...)` unwraps to the underlying type."""
        assert try_get_constructed_type(partial(ExampleA, x=1)) is ExampleA

    def test_unwraps_partial_of_module_spec(self):
        """A `partial` whose func is a `ModuleSpec` unwraps through both layers."""
        spec = ModuleSpec(module=ExampleA, params={'y': 'hi'})
        assert try_get_constructed_type(partial(spec, x=1)) is ExampleA

    def test_unwraps_module_spec_of_partial(self):
        """A `ModuleSpec` whose `module` is a `partial(SomeType)` unwraps to the type."""
        assert try_get_constructed_type(ModuleSpec(module=partial(ExampleA, x=1))) is ExampleA

    def test_unwraps_nested_partials(self):
        """Multiple nested `partial`s unwrap recursively to the inner type."""
        nested = partial(partial(partial(ExampleA, x=1), y='hi'))
        assert try_get_constructed_type(nested) is ExampleA

    def test_uses_return_annotation_for_function(self):
        """A function with a concrete return-type annotation returns that annotation."""
        assert try_get_constructed_type(_func_returning_example_a) is ExampleA

    def test_function_without_annotation_raises(self):
        """A function with no return annotation cannot be introspected → ValueError."""
        with pytest.raises(ValueError, match="return type annotation"):
            try_get_constructed_type(_func_no_annotation)

    def test_function_with_non_type_annotation_returns_it_if_type(self):
        """A function annotated `-> dict` returns `dict` (which IS a type)."""
        assert try_get_constructed_type(_func_returns_dict) is dict

    def test_function_with_string_forward_ref_raises(self):
        """A stringified forward-ref return annotation is not a real type → ValueError."""
        with pytest.raises(ValueError, match="is not a type"):
            try_get_constructed_type(_func_returns_string_forward_ref)

    def test_callable_instance_with_annotation_resolves(self):
        """A callable instance with an annotated `__call__` resolves via its signature."""
        assert try_get_constructed_type(_CallableInstanceAnnotated()) is ExampleA

    def test_callable_instance_without_annotation_raises(self):
        """A callable instance whose `__call__` lacks a return annotation → ValueError."""
        with pytest.raises(ValueError, match="return type annotation"):
            try_get_constructed_type(_CallableInstanceUnannotated())

    def test_plain_non_callable_raises(self):
        """A non-callable, non-type value (e.g. an int) raises."""
        with pytest.raises(ValueError, match="not a callable or a type"):
            try_get_constructed_type(42)  # type: ignore[arg-type]

    def test_lambda_without_annotation_raises(self):
        """A lambda has no return annotation → ValueError."""
        with pytest.raises(ValueError, match="return type annotation"):
            try_get_constructed_type(lambda: ExampleA(0, 'x'))

    def test_builder_protocol_partial_unwraps(self):
        """A realistic builder pattern: `partial(Type, **defaults)` resolves to `Type`."""
        builder = partial(ExampleA, y='default')
        assert try_get_constructed_type(builder) is ExampleA

    def test_union_return_annotation_raises(self):
        """`Union[A, B]` is not a single concrete type and cannot be resolved."""
        with pytest.raises(ValueError, match="is not a type"):
            try_get_constructed_type(_func_returning_union)

    def test_optional_return_annotation_raises(self):
        """`Optional[A]` is `Union[A, None]` — also rejected as not a single concrete type."""
        with pytest.raises(ValueError, match="is not a type"):
            try_get_constructed_type(_func_returning_optional)

    def test_classmethod_factory_with_return_type(self):
        """A classmethod factory annotated with a concrete return type resolves correctly."""
        assert try_get_constructed_type(_FactoryHost.make_annotated) is ExampleA

    def test_classmethod_factory_without_return_type_raises(self):
        """A classmethod factory with no return annotation → ValueError."""
        with pytest.raises(ValueError, match="return type annotation"):
            try_get_constructed_type(_FactoryHost.make_unannotated)

    def test_partial_wrapping_callable_instance(self):
        """`partial` over a callable instance unwraps and resolves via the instance's `__call__`."""
        instance = _CallableInstanceAnnotated()
        # `partial(instance)` is unwrapped to `instance`, which is then introspected.
        assert try_get_constructed_type(partial(instance)) is ExampleA
