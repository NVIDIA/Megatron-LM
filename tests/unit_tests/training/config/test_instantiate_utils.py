# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import enum
import functools
import logging
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from megatron.training.config.instantiate_utils import (
    InstantiationException,
    InstantiationMode,
    _call_target,
    _convert_node,
    _convert_target_to_string,
    _extract_pos_args,
    _is_target,
    _Keys,
    _locate,
    _prepare_input_dict_or_list,
    _resolve_target,
    instantiate,
    instantiate_node,
)


# Test classes and functions for instantiation testing
class TestClass:
    """Test class for instantiation."""

    def __init__(self, arg1=None, arg2=None, **kwargs):
        self.arg1 = arg1
        self.arg2 = arg2
        self.kwargs = kwargs


def test_function(arg1=None, arg2=None, **kwargs):
    """Test function for instantiation."""
    return {"arg1": arg1, "arg2": arg2, "kwargs": kwargs}


class TestInstantiationException:
    """Test InstantiationException class."""

    def test_instantiation_exception_creation(self):
        """Test creating InstantiationException."""
        msg = "Test error message"
        exc = InstantiationException(msg)
        assert str(exc) == msg
        assert isinstance(exc, Exception)


class TestInstantiationMode:
    """Test InstantiationMode enum."""

    def test_instantiation_mode_values(self):
        """Test InstantiationMode enum values."""
        assert InstantiationMode.STRICT.value == "strict"
        assert InstantiationMode.LENIENT.value == "lenient"


class TestKeys:
    """Test _Keys enum."""

    def test_keys_values(self):
        """Test _Keys enum values."""
        assert _Keys.TARGET == "_target_"
        assert _Keys.PARTIAL == "_partial_"
        assert _Keys.CALL == "_call_"
        assert _Keys.ARGS == "_args_"
        assert _Keys.NAME == "_name_"


class TestInstantiate:
    """Test instantiate function."""

    def test_instantiate_none(self):
        """Test instantiate with None config."""
        result = instantiate(None)
        assert result is None

    def test_instantiate_simple_class(self):
        """Test instantiating a simple class."""
        config = {
            "_target_": "tests.unit_tests.utils.test_instantiate_utils.TestClass",
            "arg1": "value1",
            "arg2": "value2",
        }
        result = instantiate(config)
        assert isinstance(result, TestClass)
        assert result.arg1 == "value1"
        assert result.arg2 == "value2"

    def test_instantiate_function(self):
        """Test instantiating a function."""
        config = {
            "_target_": "tests.unit_tests.utils.test_instantiate_utils.test_function",
            "arg1": "value1",
            "arg2": "value2",
        }
        result = instantiate(config)
        expected = {"arg1": "value1", "arg2": "value2", "kwargs": {}}
        assert result == expected

    def test_instantiate_with_args(self):
        """Test instantiate with positional args."""
        config = {
            "_target_": "tests.unit_tests.utils.test_instantiate_utils.test_function",
            "_args_": ["pos1", "pos2"],
        }
        result = instantiate(config)
        expected = {"arg1": "pos1", "arg2": "pos2", "kwargs": {}}
        assert result == expected

    def test_instantiate_with_partial(self):
        """Test instantiate with partial=True."""
        config = {
            "_target_": "tests.unit_tests.utils.test_instantiate_utils.test_function",
            "_partial_": True,
            "arg1": "value1",
        }
        result = instantiate(config)
        assert isinstance(result, functools.partial)
        actual_result = result(arg2="value2")
        expected = {"arg1": "value1", "arg2": "value2", "kwargs": {}}
        assert actual_result == expected

    def test_instantiate_with_call_false(self):
        """Test instantiate with _call_=False."""
        config = {
            "_target_": "tests.unit_tests.utils.test_instantiate_utils.test_function",
            "_call_": False,
        }
        result = instantiate(config)
        assert callable(result)
        assert result == test_function

    def test_instantiate_with_call_false_and_extra_keys(self):
        """Test instantiate with _call_=False and extra keys raises error."""
        config = {
            "_target_": "tests.unit_tests.utils.test_instantiate_utils.test_function",
            "_call_": False,
            "extra_key": "value",
        }
        with pytest.raises(InstantiationException, match="_call_ was set to False"):
            instantiate(config)

    def test_instantiate_with_kwargs_override(self):
        """Test instantiate with kwargs override."""
        config = {
            "_target_": "tests.unit_tests.utils.test_instantiate_utils.test_function",
            "arg1": "original",
        }
        result = instantiate(config, arg1="override", arg2="new")
        expected = {"arg1": "override", "arg2": "new", "kwargs": {}}
        assert result == expected

    def test_instantiate_list_config(self):
        """Test instantiate with list config."""
        config = [
            {
                "_target_": "tests.unit_tests.utils.test_instantiate_utils.test_function",
                "arg1": "item1",
            },
            {
                "_target_": "tests.unit_tests.utils.test_instantiate_utils.test_function",
                "arg1": "item2",
            },
        ]
        result = instantiate(config)
        assert len(result) == 2
        assert result[0] == {"arg1": "item1", "arg2": None, "kwargs": {}}
        assert result[1] == {"arg1": "item2", "arg2": None, "kwargs": {}}

    def test_instantiate_list_with_partial_raises_error(self):
        """Test instantiate list with _partial_=True raises error."""
        config = ["item1", "item2"]
        with pytest.raises(InstantiationException, match="_partial_ keyword is not compatible"):
            instantiate(config, _partial_=True)

    def test_instantiate_invalid_config_type(self):
        """Test instantiate with invalid config type."""
        with pytest.raises(InstantiationException, match="Cannot instantiate config of type"):
            instantiate("invalid_config")

    def test_instantiate_strict_mode_error(self):
        """Test instantiate in strict mode with error."""
        config = {
            "_target_": "tests.unit_tests.utils.test_instantiate_utils.TestClass",
            "nested": {"_target_": "non.existent.module.Class"},
        }
        with pytest.raises(InstantiationException):
            instantiate(config, mode=InstantiationMode.STRICT)

    def test_instantiate_lenient_mode_error(self):
        """In lenient mode, nested resolution errors now propagate (no auto-None)."""
        config = {
            "_target_": "tests.unit_tests.utils.test_instantiate_utils.TestClass",
            "nested": {"_target_": "non.existent.module.Class"},
        }
        with pytest.raises(InstantiationException, match="Error locating target"):
            instantiate(config, mode=InstantiationMode.LENIENT)

    def test_instantiate_with_omegaconf_dict(self):
        """Test instantiate with OmegaConf DictConfig."""
        config = OmegaConf.create(
            {
                "_target_": "tests.unit_tests.utils.test_instantiate_utils.TestClass",
                "arg1": "value1",
            }
        )
        result = instantiate(config)
        assert isinstance(result, TestClass)
        assert result.arg1 == "value1"

    def test_instantiate_with_omegaconf_list(self):
        """Test instantiate with OmegaConf ListConfig."""
        config = OmegaConf.create(
            [
                {
                    "_target_": "tests.unit_tests.utils.test_instantiate_utils.test_function",
                    "arg1": "item1",
                }
            ]
        )
        result = instantiate(config)
        assert len(result) == 1
        assert result[0] == {"arg1": "item1", "arg2": None, "kwargs": {}}


class TestInstantiateNode:
    """Test instantiate_node function."""

    def test_instantiate_node_none(self):
        """Test instantiate_node with None."""
        result = instantiate_node(None)
        assert result is None

    def test_instantiate_node_non_config(self):
        """Test instantiate_node with non-config value."""
        result = instantiate_node("simple_string")
        assert result == "simple_string"

    def test_instantiate_node_dict_without_target(self):
        """Test instantiate_node with dict without _target_."""
        config = OmegaConf.create({"key1": "value1", "key2": "value2"})
        result = instantiate_node(config)
        assert result == {"key1": "value1", "key2": "value2"}

    def test_instantiate_node_list(self):
        """Test instantiate_node with list."""
        config = OmegaConf.create(["item1", "item2"])
        result = instantiate_node(config)
        assert result == ["item1", "item2"]

    def test_instantiate_node_partial_not_bool_raises_error(self):
        """Test instantiate_node with non-bool partial raises error."""
        config = OmegaConf.create({"_partial_": "not_bool"})
        with pytest.raises(TypeError, match="_partial_ flag must be a bool"):
            instantiate_node(config)


class TestLocate:
    """Test _locate function."""

    def test_locate_valid_path(self):
        """Test _locate with valid path."""
        result = _locate("builtins.str")
        assert result == str

    def test_locate_empty_path(self):
        """Test _locate with empty path."""
        with pytest.raises(ImportError, match="Empty path"):
            _locate("")

    def test_locate_invalid_path(self):
        """Test _locate with invalid path."""
        with pytest.raises(ImportError, match="Unable to import any module"):
            _locate("non.existent.module")

    def test_locate_invalid_dotstring(self):
        """Test _locate with invalid dotstring."""
        with pytest.raises(ValueError, match="invalid dotstring"):
            _locate("invalid..path")

    def test_locate_relative_import(self):
        """Test _locate with relative import."""
        with pytest.raises(ValueError, match="Relative imports are not supported"):
            _locate(".relative.import")

    def test_locate_attribute_error(self):
        """Test _locate with attribute that doesn't exist."""
        with pytest.raises(ImportError, match="Are you sure that"):
            _locate("builtins.nonexistent_attribute")


class TestIsTarget:
    """Test _is_target function."""

    def test_is_target_dict_with_target(self):
        """Test _is_target with dict containing _target_."""
        config = {"_target_": "some.target"}
        assert _is_target(config) is True

    def test_is_target_dict_without_target(self):
        """Test _is_target with dict not containing _target_."""
        config = {"other_key": "value"}
        assert _is_target(config) is False

    def test_is_target_omegaconf_with_target(self):
        """Test _is_target with OmegaConf containing _target_."""
        config = OmegaConf.create({"_target_": "some.target"})
        assert _is_target(config) is True

    def test_is_target_omegaconf_without_target(self):
        """Test _is_target with OmegaConf not containing _target_."""
        config = OmegaConf.create({"other_key": "value"})
        assert _is_target(config) is False

    def test_is_target_non_dict(self):
        """Test _is_target with non-dict value."""
        assert _is_target("string") is False
        assert _is_target(123) is False
        assert _is_target([]) is False


class TestCallTarget:
    """Test _call_target function."""

    def test_call_target_normal(self):
        """Test _call_target with normal call."""
        result = _call_target(test_function, False, (), {"arg1": "value1"}, "test_key")
        expected = {"arg1": "value1", "arg2": None, "kwargs": {}}
        assert result == expected

    def test_call_target_partial(self):
        """Test _call_target with partial=True."""
        result = _call_target(test_function, True, (), {"arg1": "value1"}, "test_key")
        assert isinstance(result, functools.partial)

    def test_call_target_with_args(self):
        """Test _call_target with positional args."""
        result = _call_target(test_function, False, ("pos1", "pos2"), {}, "test_key")
        expected = {"arg1": "pos1", "arg2": "pos2", "kwargs": {}}
        assert result == expected

    def test_call_target_error_normal(self):
        """Test _call_target with error in normal call."""

        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(InstantiationException, match="Error in call to target"):
            _call_target(failing_function, False, (), {}, "test_key")

    def test_call_target_error_partial(self):
        """Test _call_target with error in partial creation."""
        # Create a mock that raises an error when used with functools.partial
        mock_target = MagicMock()
        mock_target.__module__ = "test_module"
        mock_target.__qualname__ = "test_function"

        with patch("functools.partial", side_effect=ValueError("Partial error")):
            with pytest.raises(InstantiationException, match="Error in creating partial"):
                _call_target(mock_target, True, (), {}, "test_key")


class TestConvertTargetToString:
    """Test _convert_target_to_string function."""

    def test_convert_callable_to_string(self):
        """Test converting callable to string."""
        result = _convert_target_to_string(test_function)
        assert "test_function" in result

    def test_convert_non_callable_to_string(self):
        """Test converting non-callable to string."""
        result = _convert_target_to_string("already_string")
        assert result == "already_string"


class TestPrepareInputDictOrList:
    """Test _prepare_input_dict_or_list function."""

    def test_prepare_dict(self):
        """Test preparing input dict."""
        input_dict = {"_target_": test_function, "key1": "value1", "nested": {"key2": "value2"}}
        result = _prepare_input_dict_or_list(input_dict)
        assert "_target_" in result
        assert "test_function" in result["_target_"]
        assert result["key1"] == "value1"
        assert result["nested"]["key2"] == "value2"

    def test_prepare_list(self):
        """Test preparing input list."""
        input_list = [{"_target_": test_function, "key1": "value1"}, ["nested_item"]]
        result = _prepare_input_dict_or_list(input_list)
        assert len(result) == 2
        assert "test_function" in result[0]["_target_"]
        assert result[0]["key1"] == "value1"
        assert result[1] == ["nested_item"]


class TestResolveTarget:
    """Test _resolve_target function."""

    def test_resolve_string_target(self):
        """Test resolving string target."""
        result = _resolve_target("builtins.str", "test_key")
        assert result == str

    def test_resolve_callable_target(self):
        """Test resolving already callable target."""
        result = _resolve_target(test_function, "test_key")
        assert result == test_function

    def test_resolve_invalid_string_target(self):
        """Test resolving invalid string target."""
        with pytest.raises(InstantiationException, match="Error locating target"):
            _resolve_target("invalid.target", "test_key")

    def test_resolve_non_callable_target(self):
        """Test resolving non-callable target with check_callable=True."""
        with pytest.raises(InstantiationException, match="Expected a callable target"):
            _resolve_target("builtins.__name__", "test_key", check_callable=True)

    def test_resolve_non_callable_target_no_check(self):
        """Test resolving non-callable target with check_callable=False."""
        result = _resolve_target("builtins.__name__", "test_key", check_callable=False)
        assert result == "builtins"


class TestExtractPosArgs:
    """Test _extract_pos_args function."""

    def test_extract_pos_args_no_input_args(self):
        """Test extracting pos args with no input args."""
        kwargs = {"_args_": ["arg1", "arg2"], "key1": "value1"}
        args, remaining_kwargs = _extract_pos_args((), kwargs)
        assert args == ["arg1", "arg2"]
        assert remaining_kwargs == {"key1": "value1"}

    def test_extract_pos_args_with_input_args(self):
        """Test extracting pos args with input args override."""
        kwargs = {"_args_": ["config_arg1", "config_arg2"], "key1": "value1"}
        input_args = ["input_arg1", "input_arg2"]
        args, remaining_kwargs = _extract_pos_args(input_args, kwargs)
        assert args == input_args
        assert remaining_kwargs == {"key1": "value1"}

    def test_extract_pos_args_no_args_key(self):
        """Test extracting pos args with no _args_ key."""
        kwargs = {"key1": "value1"}
        args, remaining_kwargs = _extract_pos_args((), kwargs)
        assert args == ()
        assert remaining_kwargs == {"key1": "value1"}

    def test_extract_pos_args_invalid_type(self):
        """Test extracting pos args with invalid _args_ type."""
        kwargs = {"_args_": 123}  # Integer is not a sequence
        with pytest.raises(InstantiationException, match="Unsupported _args_ type"):
            _extract_pos_args((), kwargs)


class TestConvertNode:
    """Test _convert_node function."""

    def test_convert_omegaconf_node(self):
        """Test converting OmegaConf node."""
        config = OmegaConf.create({"key1": "value1", "key2": 2})
        result = _convert_node(config)
        assert result == {"key1": "value1", "key2": 2}
        assert not OmegaConf.is_config(result)

    def test_convert_non_config_node(self):
        """Test converting non-config node."""
        value = {"key1": "value1"}
        result = _convert_node(value)
        assert result == value


class TestComplexScenarios:
    """Test complex instantiation scenarios."""

    def test_nested_instantiation(self):
        """Test nested instantiation scenario."""
        config = {
            "_target_": "tests.unit_tests.utils.test_instantiate_utils.TestClass",
            "arg1": {
                "_target_": "tests.unit_tests.utils.test_instantiate_utils.test_function",
                "arg1": "nested_value",
            },
            "arg2": "simple_value",
        }
        result = instantiate(config)
        assert isinstance(result, TestClass)
        assert result.arg1 == {"arg1": "nested_value", "arg2": None, "kwargs": {}}
        assert result.arg2 == "simple_value"

    def test_list_with_nested_targets(self):
        """Test list with nested target instantiation."""
        config = [
            {
                "_target_": "tests.unit_tests.utils.test_instantiate_utils.TestClass",
                "arg1": {
                    "_target_": "tests.unit_tests.utils.test_instantiate_utils.test_function",
                    "arg1": "item1",
                },
            },
            "simple_item",
        ]
        result = instantiate(config)
        assert len(result) == 2
        assert isinstance(result[0], TestClass)
        assert result[0].arg1 == {"arg1": "item1", "arg2": None, "kwargs": {}}
        assert result[1] == "simple_item"

    def test_missing_values_with_partial(self):
        """Test missing values with partial instantiation."""
        config = OmegaConf.create(
            {
                "_target_": "tests.unit_tests.utils.test_instantiate_utils.test_function",
                "_partial_": True,
                "arg1": "value1",
                "missing_arg": "???",  # OmegaConf missing value
            }
        )
        OmegaConf.set_struct(config, True)

        result = instantiate(config)
        assert isinstance(result, functools.partial)
        # The missing value should be skipped in partial mode
        actual_result = result(arg2="value2")
        expected = {"arg1": "value1", "arg2": "value2", "kwargs": {}}
        assert actual_result == expected


class DummyTarget:
    def __init__(self, a: int, b: int = 0) -> None:
        self.a = a
        self.b = b


class KwTarget:
    def __init__(self, **kwargs) -> None:
        self.kwargs = dict(kwargs)


def _target_qualname(obj) -> str:
    return f"{obj.__module__}.{obj.__qualname__}"


def test_drops_unexpected_kwargs_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    config = {
        "_target_": _target_qualname(DummyTarget),
        "a": 10,
        "foo": 123,  # unexpected key that should be dropped
    }

    with caplog.at_level(logging.WARNING):
        obj = instantiate(config)

    assert isinstance(obj, DummyTarget)
    assert obj.a == 10
    # 'foo' is dropped; 'b' remains default
    assert obj.b == 0

    # Ensure a warning was emitted mentioning the dropped key
    warnings = [rec.getMessage() for rec in caplog.records if rec.levelno == logging.WARNING]
    assert any("Dropping unexpected config keys" in m for m in warnings)
    assert any("foo" in m for m in warnings)


def test_allows_kwargs_when_target_accepts_var_kwargs(caplog: pytest.LogCaptureFixture) -> None:
    config = {"_target_": _target_qualname(KwTarget), "foo": 1, "bar": 2}

    with caplog.at_level(logging.WARNING):
        obj = instantiate(config)

    assert isinstance(obj, KwTarget)
    assert obj.kwargs == {"foo": 1, "bar": 2}

    # No warning should be emitted for **kwargs targets
    warnings = [rec.getMessage() for rec in caplog.records if rec.levelno == logging.WARNING]
    assert not any("Dropping unexpected config keys" in m for m in warnings)


def test_raises_on_unexpected_kwargs_in_strict_mode() -> None:
    config = {"_target_": _target_qualname(DummyTarget), "a": 10, "foo": 123}

    with pytest.raises(InstantiationException):
        instantiate(config, mode=InstantiationMode.STRICT)


class TestEnum(enum.Enum):
    A = 1
    B = 2


class TestInstantiateEnum:
    """Test instantiation of Enums."""

    def test_instantiate_enum_with_args(self):
        """Test instantiating an Enum with _args_."""
        config = {
            "_target_": "tests.unit_tests.utils.test_instantiate_utils.TestEnum",
            "_args_": [1],
        }
        result = instantiate(config)
        assert result == TestEnum.A

    def test_instantiate_enum_with_args_lenient(self):
        """Test instantiating an Enum with _args_ in lenient mode (default)."""
        config = {
            "_target_": "tests.unit_tests.utils.test_instantiate_utils.TestEnum",
            "_args_": [2],
        }
        # This previously failed because _args_ was dropped in lenient mode
        result = instantiate(config)
        assert result == TestEnum.B
