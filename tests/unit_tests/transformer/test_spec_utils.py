import dataclasses
from functools import partial
from typing import Protocol

import pytest

from megatron.core.transformer.spec_utils import ModuleSpec, build_module, get_submodules


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
