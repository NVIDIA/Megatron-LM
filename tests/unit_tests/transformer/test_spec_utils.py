# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import dataclasses
import errno
import traceback
from functools import partial
from typing import Protocol

import pytest

from megatron.core.transformer.spec_utils import (
    ModuleSpec,
    build_module,
    get_submodules,
    import_module,
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

    @pytest.mark.parametrize("use_spec", [False, True])
    def test_constructor_preserves_unicode_error(self, use_spec):
        """Keep structured exceptions that cannot be reconstructed from a string."""
        original_errors = []

        class DecodeModule:
            def __init__(self) -> None:
                try:
                    b"\xff".decode("utf-8")
                except UnicodeDecodeError as error:
                    original_errors.append(error)
                    raise

        spec = ModuleSpec(module=DecodeModule) if use_spec else DecodeModule
        with pytest.raises(UnicodeDecodeError) as raised:
            build_module(spec)

        assert raised.value is original_errors[0]
        assert raised.value.object == b"\xff"
        assert raised.value.encoding == "utf-8"
        assert (raised.value.start, raised.value.end) == (0, 1)
        assert "when instantiating DecodeModule" in raised.value.__notes__

    def test_constructor_preserves_file_error(self, tmp_path):
        """Keep the filename and errno produced by an actual file operation."""
        missing = tmp_path / "missing.txt"

        class ReadModule:
            def __init__(self) -> None:
                missing.read_bytes()

        with pytest.raises(FileNotFoundError) as raised:
            build_module(ReadModule)

        assert raised.value.errno == errno.ENOENT
        assert raised.value.filename == str(missing)
        assert "when instantiating ReadModule" in raised.value.__notes__

    def test_nested_constructor_preserves_cause_and_context_notes(self):
        """Annotate each factory without replacing the original exception or cause."""
        cause = ValueError("invalid configuration")
        original = RuntimeError("construction failed")
        original.add_note("existing diagnostic")

        class Inner:
            def __init__(self) -> None:
                raise original from cause

        class Outer:
            def __init__(self) -> None:
                build_module(Inner)

        with pytest.raises(RuntimeError) as raised:
            build_module(Outer)

        assert raised.value is original
        assert raised.value.__cause__ is cause
        assert raised.value.__notes__ == [
            "existing diagnostic",
            "when instantiating Inner",
            "when instantiating Outer",
        ]
        formatted = "".join(traceback.format_exception(raised.value))
        assert "raise original from cause" in formatted
        assert "when instantiating Inner" in formatted
        assert "when instantiating Outer" in formatted


class TestImportModule:
    """Unit tests for dynamic spec imports."""

    def test_missing_module_raises(self):
        with pytest.raises(
            ImportError, match="Could not import module 'megatron.core.models.does_not_exist'"
        ):
            import_module(('megatron.core.models.does_not_exist', 'missing_spec'))

    def test_missing_spec_raises(self):
        with pytest.raises(
            ImportError,
            match=(
                "Could not find spec 'does_not_exist' in module "
                "'megatron.core.transformer.identity_op'"
            ),
        ):
            import_module(('megatron.core.transformer.identity_op', 'does_not_exist'))


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
