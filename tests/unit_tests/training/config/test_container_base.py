# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import copy
import functools
import os
import tempfile
import warnings
from dataclasses import dataclass, field
from unittest.mock import MagicMock, mock_open, patch

import pytest
import torch
from megatron.core.msc_utils import MultiStorageClientFeature

from megatron.training.config.container import ConfigContainerBase
from megatron.training.config.instantiate_utils import InstantiationMode


def _target_qualname(obj) -> str:
    return f"{obj.__module__}.{obj.__qualname__}"


# Test functions for callable testing
def activation_function(x):
    """Test activation function."""
    return x * 2


def loss_function(pred, target, reduction="mean"):
    """Test loss function with parameters."""
    return abs(pred - target)


# Test dataclasses for testing
@dataclass
class SimpleDataclass:
    """Simple dataclass for testing."""

    name: str = "test"
    value: int = 42


@dataclass
class NestedDataclass:
    """Nested dataclass for testing."""

    simple: SimpleDataclass
    description: str = "nested"


@dataclass
class CallableDataclass:
    """Dataclass with callable and partial fields for testing."""

    name: str = "callable_test"
    activation_func: callable = activation_function
    loss_func: callable = functools.partial(loss_function, reduction="sum")
    torch_func: callable = torch.nn.functional.relu
    lambda_func: callable = lambda x: x + 1
    regular_value: int = 100


@dataclass
class TestConfigContainer(ConfigContainerBase):
    """Test configuration container."""

    name: str = "test_config"
    value: int = 100
    description: str = "A test configuration"


@dataclass
class ComplexConfigContainer(ConfigContainerBase):
    """Complex configuration container for testing."""

    simple_config: TestConfigContainer
    nested_data: NestedDataclass
    items: list[str]
    metadata: dict[str, int]


@dataclass
class CallableConfigContainer(ConfigContainerBase):
    """Configuration container with callable fields for testing."""

    name: str = "callable_config"
    callable_data: CallableDataclass = None  # Will be set in tests
    activation: callable = activation_function
    partial_loss: callable = functools.partial(loss_function, reduction="none")
    torch_activation: callable = torch.nn.functional.gelu

    def __post_init__(self):
        """Initialize callable_data if not provided."""
        if self.callable_data is None:
            self.callable_data = CallableDataclass()


class TestConfigContainer_FromDict:
    """Test ConfigContainer.from_dict method."""

    @patch("megatron.training.config.container.instantiate")
    def test_from_dict_basic(self, mock_instantiate):
        """Test basic from_dict functionality."""
        config_dict = {
            "_target_": _target_qualname(TestConfigContainer),
            "name": "from_dict",
            "value": 300,
        }

        expected_config = TestConfigContainer(name="from_dict", value=300)
        mock_instantiate.return_value = expected_config

        result = TestConfigContainer.from_dict(config_dict)

        mock_instantiate.assert_called_once_with(config_dict, mode=InstantiationMode.STRICT)
        assert result.name == "from_dict"
        assert result.value == 300

    @patch("megatron.training.config.container.instantiate")
    def test_from_dict_with_mode(self, mock_instantiate):
        """Test from_dict with different instantiation modes."""
        config_dict = {"_target_": _target_qualname(TestConfigContainer), "name": "lenient"}

        expected_config = TestConfigContainer(name="lenient")
        mock_instantiate.return_value = expected_config

        result = TestConfigContainer.from_dict(config_dict, mode=InstantiationMode.LENIENT)

        mock_instantiate.assert_called_once_with(config_dict, mode=InstantiationMode.LENIENT)
        assert result.name == "lenient"

    def test_from_dict_missing_target(self):
        """Test from_dict raises error when _target_ is missing."""
        config_dict = {"name": "test"}

        with pytest.raises(AssertionError):
            TestConfigContainer.from_dict(config_dict)

    def test_from_dict_extra_keys_strict_mode(self):
        """Test from_dict raises error for extra keys in strict mode."""
        config_dict = {
            "_target_": _target_qualname(TestConfigContainer),
            "name": "test",
            "extra_key": "should_fail",
        }

        with pytest.raises(ValueError, match="Dictionary contains extra keys"):
            TestConfigContainer.from_dict(config_dict, mode=InstantiationMode.STRICT)

    @patch("megatron.training.config.container.instantiate")
    def test_from_dict_extra_keys_lenient_mode(self, mock_instantiate):
        """Test from_dict removes extra keys in lenient mode."""
        config_dict = {
            "_target_": _target_qualname(TestConfigContainer),
            "name": "test",
            "extra_key": "should_be_removed",
        }

        expected_config = TestConfigContainer(name="test")
        mock_instantiate.return_value = expected_config

        TestConfigContainer.from_dict(config_dict, mode=InstantiationMode.LENIENT)

        # Verify that extra_key was removed from the dict passed to instantiate
        called_dict = mock_instantiate.call_args[0][0]
        assert "extra_key" not in called_dict
        assert called_dict["name"] == "test"
        assert called_dict["_target_"] == _target_qualname(TestConfigContainer)

    def test_from_dict_preserves_original(self):
        """Test that from_dict doesn't modify the original dictionary."""
        original_dict = {
            "_target_": _target_qualname(TestConfigContainer),
            "name": "original",
            "extra_key": "should_be_preserved_in_original",
        }

        original_copy = copy.deepcopy(original_dict)

        with pytest.raises(ValueError):  # This will fail in strict mode
            TestConfigContainer.from_dict(original_dict, mode=InstantiationMode.STRICT)

        # Original dict should be unchanged
        assert original_dict == original_copy


class TestConfigContainer_FromYaml:
    """Test ConfigContainer.from_yaml method."""

    def test_from_yaml_file_not_found(self):
        """Test from_yaml raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="YAML file not found"):
            TestConfigContainer.from_yaml("non_existent_file.yaml")

    @patch("megatron.training.config.container.MultiStorageClientFeature.is_enabled")
    @patch("megatron.training.config.container.OmegaConf")
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    def test_from_yaml_success(self, mock_exists, mock_file, mock_omegaconf, mock_msc):
        """Test successful YAML loading."""
        mock_msc.return_value = False
        mock_exists.return_value = True
        yaml_content = f"""
        _target_: {_target_qualname(TestConfigContainer)}
        name: yaml_config
        value: 500
        """
        mock_file.return_value.read.return_value = yaml_content

        # Mock yaml.safe_load to return parsed content
        with patch("yaml.safe_load") as mock_yaml_load:
            config_dict = {
                "_target_": _target_qualname(TestConfigContainer),
                "name": "yaml_config",
                "value": 500,
            }
            mock_yaml_load.return_value = config_dict

            # Mock OmegaConf methods
            mock_conf = MagicMock()
            mock_omegaconf.create.return_value = mock_conf
            mock_omegaconf.to_container.return_value = config_dict

            # Mock the from_dict method
            with patch.object(TestConfigContainer, "from_dict") as mock_from_dict:
                expected_config = TestConfigContainer(name="yaml_config", value=500)
                mock_from_dict.return_value = expected_config

                result = TestConfigContainer.from_yaml("test.yaml")

                mock_exists.assert_called_once_with("test.yaml")
                mock_file.assert_called_once_with("test.yaml", "r")
                mock_yaml_load.assert_called_once()
                mock_omegaconf.create.assert_called_once_with(config_dict)
                mock_omegaconf.to_container.assert_called_once_with(mock_conf, resolve=True)
                mock_from_dict.assert_called_once_with(config_dict, mode=InstantiationMode.LENIENT)

                assert result.name == "yaml_config"
                assert result.value == 500

    @patch("megatron.training.config.container.MultiStorageClientFeature.is_enabled")
    @patch("os.path.exists")
    def test_from_yaml_with_mode(self, mock_exists, mock_msc):
        """Test from_yaml with different instantiation modes."""
        mock_msc.return_value = False
        mock_exists.return_value = True

        with patch("builtins.open", mock_open()):
            with patch("yaml.safe_load", return_value={}):
                with patch("megatron.training.config.container.OmegaConf") as mock_omegaconf:
                    # Mock OmegaConf methods to return expected values
                    mock_conf = MagicMock()
                    mock_omegaconf.create.return_value = mock_conf
                    mock_omegaconf.to_container.return_value = {}  # Return actual empty dict

                    with patch.object(TestConfigContainer, "from_dict") as mock_from_dict:
                        TestConfigContainer.from_yaml("test.yaml", mode=InstantiationMode.STRICT)
                        mock_from_dict.assert_called_once_with({}, mode=InstantiationMode.STRICT)


class TestConfigContainer_ToDict:
    """Test ConfigContainer.to_dict method."""

    def test_to_dict_basic(self):
        """Test basic to_dict functionality."""
        config = TestConfigContainer(name="test", value=123, description="test desc")
        result = config.to_dict()

        expected = {
            "_target_": _target_qualname(TestConfigContainer),
            "name": "test",
            "value": 123,
            "description": "test desc",
        }

        assert result == expected

    def test_to_dict_with_nested_config_container(self):
        """Test to_dict with nested ConfigContainer."""
        simple_config = TestConfigContainer(name="nested", value=456)
        nested_data = NestedDataclass(simple=SimpleDataclass(name="inner", value=789))

        complex_config = ComplexConfigContainer(
            simple_config=simple_config,
            nested_data=nested_data,
            items=["a", "b", "c"],
            metadata={"key1": 1, "key2": 2},
        )

        result = complex_config.to_dict()

        # Check the structure
        assert "_target_" in result
        assert result["_target_"] == _target_qualname(ComplexConfigContainer)

        # Check nested ConfigContainer
        assert result["simple_config"]["_target_"] == _target_qualname(TestConfigContainer)
        assert result["simple_config"]["name"] == "nested"
        assert result["simple_config"]["value"] == 456

        # Check nested regular dataclass
        assert result["nested_data"]["_target_"] == _target_qualname(NestedDataclass)
        assert result["nested_data"]["simple"]["_target_"] == _target_qualname(SimpleDataclass)
        assert result["nested_data"]["simple"]["name"] == "inner"
        assert result["nested_data"]["simple"]["value"] == 789

        # Check lists and dicts
        assert result["items"] == ["a", "b", "c"]
        assert result["metadata"] == {"key1": 1, "key2": 2}

    # TODO (@maanug): reenable after migrating model config+builder
    # def test_convert_serializable_nested_in_config(self):
    #     """Test that a Serializable nested inside a ConfigContainer is serialized via as_dict()."""

    #     class NestedSerializable:
    #         def __init__(self, value):
    #             self.value = value

    #         def as_dict(self) -> dict:
    #             return {"_target_": "my.module.NestedSerializable", "value": self.value}

    #         @classmethod
    #         def from_dict(cls, data):
    #             return cls(data["value"])

    #     @dataclass
    #     class ConfigWithSerializable(ConfigContainerBase):
    #         name: str = "ser_test"
    #         nested: object = None

    #         def __post_init__(self):
    #             if self.nested is None:
    #                 self.nested = NestedSerializable(99)

    #     config = ConfigWithSerializable()
    #     result = config.to_dict()

    #     assert result["name"] == "ser_test"
    #     assert result["nested"] == {"_target_": "my.module.NestedSerializable", "value": 99}

    def test_to_dict_excludes_private_fields(self):
        """Test that to_dict excludes fields starting with underscore."""
        config = TestConfigContainer()
        result = config.to_dict()

        # Should include _target_ but exclude __version__
        assert "_target_" in result
        assert "__version__" not in result


class TestConfigContainer_ConvertValueToDict:
    """Test ConfigContainer._convert_value_to_dict method."""

    def test_convert_config_container(self):
        """Test converting ConfigContainer instance."""
        config = TestConfigContainer(name="convert_test", value=999)
        result = TestConfigContainer._convert_value_to_dict(config)

        expected = {
            "_target_": _target_qualname(TestConfigContainer),
            "name": "convert_test",
            "value": 999,
            "description": "A test configuration",
        }

        assert result == expected

    def test_convert_regular_dataclass(self):
        """Test converting regular dataclass."""
        simple = SimpleDataclass(name="simple_test", value=555)
        result = TestConfigContainer._convert_value_to_dict(simple)

        expected = {
            "_target_": _target_qualname(SimpleDataclass),
            "name": "simple_test",
            "value": 555,
        }

        assert result == expected

    def test_convert_list(self):
        """Test converting list with nested dataclasses."""
        items = [SimpleDataclass(name="item1", value=1), "string_item", 42]
        result = TestConfigContainer._convert_value_to_dict(items)

        assert len(result) == 3
        assert result[0]["_target_"] == _target_qualname(SimpleDataclass)
        assert result[0]["name"] == "item1"
        assert result[1] == "string_item"
        assert result[2] == 42

    def test_convert_tuple(self):
        """Test converting tuple."""
        items = (SimpleDataclass(name="tuple_item"), "string")
        result = TestConfigContainer._convert_value_to_dict(items)

        assert len(result) == 2
        assert result[0]["_target_"] == _target_qualname(SimpleDataclass)
        assert result[1] == "string"

    def test_convert_dict(self):
        """Test converting dictionary with nested dataclasses."""
        data = {
            "config": SimpleDataclass(name="dict_config"),
            "value": 123,
            "nested": {"inner": SimpleDataclass(name="inner_config")},
        }
        result = TestConfigContainer._convert_value_to_dict(data)

        assert result["config"]["_target_"] == _target_qualname(SimpleDataclass)
        assert result["value"] == 123
        assert result["nested"]["inner"]["_target_"] == _target_qualname(SimpleDataclass)

    # TODO (@maanug): reenable after migrating model config+builder
    # def test_convert_serializable(self):
    #     """Test converting a Serializable instance uses as_dict()."""

    #     class MySerializable:
    #         def as_dict(self) -> dict:
    #             return {"_target_": "my.module.MySerializable", "x": 42}

    #         @classmethod
    #         def from_dict(cls, data):
    #             return cls()

    #     obj = MySerializable()
    #     assert isinstance(obj, Serializable)  # runtime_checkable sanity check

    #     result = TestConfigContainer._convert_value_to_dict(obj)

    #     assert result == {"_target_": "my.module.MySerializable", "x": 42}

    def test_convert_primitive_types(self):
        """Test converting primitive types."""
        assert TestConfigContainer._convert_value_to_dict(42) == 42
        assert TestConfigContainer._convert_value_to_dict("string") == "string"
        assert TestConfigContainer._convert_value_to_dict(True) is True
        assert TestConfigContainer._convert_value_to_dict(None) is None
        assert TestConfigContainer._convert_value_to_dict(3.14) == 3.14

    def test_convert_excludes_private_fields_in_dataclass(self):
        """Test that private fields are excluded from dataclass conversion."""

        @dataclass
        class DataclassWithPrivate:
            public_field: str = "public"
            _private_field: str = "private"

        obj = DataclassWithPrivate()
        result = TestConfigContainer._convert_value_to_dict(obj)

        assert "public_field" in result
        assert "_private_field" not in result
        assert "_target_" in result


class TestConfigContainer_ToYaml:
    """Test ConfigContainer.to_yaml method."""

    @patch("megatron.training.config.container.MultiStorageClientFeature.is_enabled")
    @patch("megatron.training.config.container.safe_yaml_representers")
    @patch("yaml.safe_dump")
    @patch("builtins.open", new_callable=mock_open)
    def test_to_yaml_save_to_file(
        self, mock_file, mock_yaml_dump, mock_safe_representers, mock_msc
    ):
        """Test to_yaml saving to file."""
        mock_msc.return_value = False
        config = TestConfigContainer(name="file_test", value=888)
        mock_safe_representers.return_value.__enter__ = MagicMock()
        mock_safe_representers.return_value.__exit__ = MagicMock()

        config.to_yaml("test_output.yaml")

        mock_safe_representers.assert_called_once()
        mock_file.assert_called_once_with("test_output.yaml", "w")
        mock_yaml_dump.assert_called_once()

        # Verify the correct arguments were passed to yaml.safe_dump
        call_args = mock_yaml_dump.call_args
        assert call_args[1]["default_flow_style"] is False

    def test_to_yaml_with_msc_url(self):
        """Test to_yaml with MSC URL."""
        from megatron.training.config.instantiate_utils import target_allowlist

        config = TestConfigContainer(name="msc_test", value=999)

        MultiStorageClientFeature.enable()

        # Verify that the file is created in the temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            config.to_yaml(f"msc://default{temp_dir}/test_output.yaml")
            assert os.path.exists(f"{temp_dir}/test_output.yaml")

            target_allowlist.disable()
            loaded_config = TestConfigContainer.from_yaml(
                f"msc://default{temp_dir}/test_output.yaml"
            )
            target_allowlist.enable()
            assert config.to_dict() == loaded_config.to_dict()


class TestConfigContainer_PrintYaml:
    """Test ConfigContainer.print_yaml method."""

    @patch("megatron.training.config.container.safe_yaml_representers")
    @patch("yaml.safe_dump")
    @patch("builtins.print")
    def test_print_yaml_basic(self, mock_print, mock_yaml_dump, mock_safe_representers):
        """Test print_yaml basic functionality."""
        config = TestConfigContainer(name="print_test", value=555, description="test print")
        mock_yaml_dump.return_value = "printed_yaml_content"
        mock_safe_representers.return_value.__enter__ = MagicMock()
        mock_safe_representers.return_value.__exit__ = MagicMock()

        config.print_yaml()

        # Verify safe_yaml_representers context manager is used
        mock_safe_representers.assert_called_once()

        # Verify yaml.safe_dump is called with correct arguments
        mock_yaml_dump.assert_called_once()
        call_args = mock_yaml_dump.call_args

        # Check the config dict passed to yaml.safe_dump
        config_dict = call_args[0][0]
        assert config_dict["_target_"] == _target_qualname(TestConfigContainer)
        assert config_dict["name"] == "print_test"
        assert config_dict["value"] == 555
        assert config_dict["description"] == "test print"

        # Check yaml.safe_dump options
        assert call_args[1]["default_flow_style"] is False

        # Verify print is called with the YAML content
        mock_print.assert_called_once_with("printed_yaml_content")

    @patch("megatron.training.config.container.safe_yaml_representers")
    @patch("yaml.safe_dump")
    @patch("builtins.print")
    def test_print_yaml_with_complex_config(
        self, mock_print, mock_yaml_dump, mock_safe_representers
    ):
        """Test print_yaml with complex nested configuration."""
        simple_config = TestConfigContainer(name="nested", value=123)
        nested_data = NestedDataclass(simple=SimpleDataclass(name="inner", value=456))

        complex_config = ComplexConfigContainer(
            simple_config=simple_config,
            nested_data=nested_data,
            items=["a", "b", "c"],
            metadata={"key1": 10, "key2": 20},
        )

        mock_yaml_dump.return_value = "complex_yaml_content"
        mock_safe_representers.return_value.__enter__ = MagicMock()
        mock_safe_representers.return_value.__exit__ = MagicMock()

        complex_config.print_yaml()

        # Verify the method was called correctly
        mock_safe_representers.assert_called_once()
        mock_yaml_dump.assert_called_once()
        mock_print.assert_called_once_with("complex_yaml_content")

        # Verify the complex structure in the config dict
        call_args = mock_yaml_dump.call_args
        config_dict = call_args[0][0]

        assert config_dict["_target_"] == _target_qualname(ComplexConfigContainer)
        assert config_dict["simple_config"]["name"] == "nested"
        assert config_dict["nested_data"]["simple"]["value"] == 456
        assert config_dict["items"] == ["a", "b", "c"]
        assert config_dict["metadata"] == {"key1": 10, "key2": 20}

    @patch("megatron.training.config.container.safe_yaml_representers")
    @patch("yaml.safe_dump")
    @patch("builtins.print")
    def test_print_yaml_calls_to_dict(self, mock_print, mock_yaml_dump, mock_safe_representers):
        """Test that print_yaml correctly calls to_dict method."""
        config = TestConfigContainer(name="to_dict_test", value=999)
        mock_yaml_dump.return_value = "yaml_output"
        mock_safe_representers.return_value.__enter__ = MagicMock()
        mock_safe_representers.return_value.__exit__ = MagicMock()

        # Mock to_dict to verify it's called
        with patch.object(config, "to_dict") as mock_to_dict:
            expected_dict = {
                "_target_": _target_qualname(TestConfigContainer),
                "name": "to_dict_test",
                "value": 999,
                "description": "A test configuration",
            }
            mock_to_dict.return_value = expected_dict

            config.print_yaml()

            # Verify to_dict was called
            mock_to_dict.assert_called_once()

            # Verify the returned dict was passed to yaml.safe_dump
            call_args = mock_yaml_dump.call_args
            assert call_args[0][0] == expected_dict


class TestConfigContainer_DeepCopy:
    """Test ConfigContainer.__deepcopy__ method."""

    def test_deepcopy_basic(self):
        """Test basic deep copy functionality."""
        config = TestConfigContainer(name="original", value=100)
        copied_config = copy.deepcopy(config)

        assert copied_config is not config
        assert copied_config.name == config.name
        assert copied_config.value == config.value
        assert copied_config.description == config.description

        # Modify original to verify they're independent
        config.name = "modified"
        assert copied_config.name == "original"

    def test_deepcopy_with_nested_structures(self):
        """Test deep copy with nested dataclasses and containers."""
        simple_config = TestConfigContainer(name="nested", value=456)
        nested_data = NestedDataclass(simple=SimpleDataclass(name="inner", value=789))

        complex_config = ComplexConfigContainer(
            simple_config=simple_config,
            nested_data=nested_data,
            items=["a", "b", "c"],
            metadata={"key1": 1, "key2": 2},
        )

        copied_config = copy.deepcopy(complex_config)

        # Verify it's a deep copy
        assert copied_config is not complex_config
        assert copied_config.simple_config is not complex_config.simple_config
        assert copied_config.nested_data is not complex_config.nested_data
        assert copied_config.items is not complex_config.items
        assert copied_config.metadata is not complex_config.metadata

        # Verify values are preserved
        assert copied_config.simple_config.name == "nested"
        assert copied_config.nested_data.simple.name == "inner"
        assert copied_config.items == ["a", "b", "c"]
        assert copied_config.metadata == {"key1": 1, "key2": 2}

        # Verify independence
        complex_config.simple_config.name = "modified"
        complex_config.items.append("d")

        assert copied_config.simple_config.name == "nested"
        assert len(copied_config.items) == 3


class TestConfigContainer_Integration:
    """Integration tests for ConfigContainer."""

    def test_roundtrip_dict_conversion(self):
        """Test that converting to dict and back preserves data."""
        # Create a complex configuration
        simple_config = TestConfigContainer(name="roundtrip", value=999)
        nested_data = NestedDataclass(
            simple=SimpleDataclass(name="nested", value=888), description="roundtrip test"
        )

        original_config = ComplexConfigContainer(
            simple_config=simple_config,
            nested_data=nested_data,
            items=["x", "y", "z"],
            metadata={"test": 42},
        )

        # Convert to dict
        config_dict = original_config.to_dict()

        # Verify dict structure
        assert "_target_" in config_dict
        assert config_dict["simple_config"]["name"] == "roundtrip"
        assert config_dict["nested_data"]["simple"]["value"] == 888

        # Convert back (would work if instantiate is properly implemented)
        # This tests the dict structure is correct for round-trip
        assert config_dict["simple_config"]["_target_"] == _target_qualname(TestConfigContainer)
        assert config_dict["nested_data"]["_target_"] == _target_qualname(NestedDataclass)

    def test_yaml_roundtrip_structure(self):
        """Test YAML conversion produces expected structure."""
        config = TestConfigContainer(name="yaml_roundtrip", value=1234)

        with patch("megatron.training.config.container.safe_yaml_representers"):
            with patch("yaml.safe_dump") as mock_dump:
                config.print_yaml()

                # Verify the dictionary passed to yaml.safe_dump
                call_args = mock_dump.call_args[0][0]
                assert call_args["_target_"] == _target_qualname(TestConfigContainer)
                assert call_args["name"] == "yaml_roundtrip"
                assert call_args["value"] == 1234

    def test_error_handling_in_complex_scenarios(self):
        """Test error handling with complex nested structures."""
        # Test with missing _target_ in from_dict
        invalid_dict = {"name": "invalid"}

        with pytest.raises(AssertionError):
            TestConfigContainer.from_dict(invalid_dict)

        # Test with extra keys in strict mode
        invalid_strict_dict = {
            "_target_": _target_qualname(TestConfigContainer),
            "name": "test",
            "invalid_field": "should_fail",
        }

        with pytest.raises(ValueError, match="extra keys"):
            TestConfigContainer.from_dict(invalid_strict_dict, mode=InstantiationMode.STRICT)


class TestConfigContainer_EdgeCases:
    """Test edge cases for ConfigContainer."""

    def test_empty_config_container(self):
        """Test ConfigContainer with minimal fields."""

        @dataclass
        class MinimalConfig(ConfigContainerBase):
            pass

        config = MinimalConfig()
        result = config.to_dict()

        assert "_target_" in result
        # The actual path will be generated based on the local class
        assert "MinimalConfig" in result["_target_"]

    def test_config_with_none_values(self):
        """Test ConfigContainer with None values."""

        @dataclass
        class ConfigWithNone(ConfigContainerBase):
            optional_field: str = None
            required_field: str = "required"

        config = ConfigWithNone()
        result = config.to_dict()

        assert result["optional_field"] is None
        assert result["required_field"] == "required"

    def test_config_with_complex_nested_types(self):
        """Test ConfigContainer with complex nested types."""

        @dataclass
        class ComplexConfig(ConfigContainerBase):
            nested_list: list[dict[str, SimpleDataclass]]
            nested_dict: dict[str, list[SimpleDataclass]]

        nested_list = [
            {"item1": SimpleDataclass(name="list_item1", value=1)},
            {"item2": SimpleDataclass(name="list_item2", value=2)},
        ]

        nested_dict = {
            "group1": [SimpleDataclass(name="group1_item1", value=10)],
            "group2": [SimpleDataclass(name="group2_item1", value=20)],
        }

        config = ComplexConfig(nested_list=nested_list, nested_dict=nested_dict)
        result = config.to_dict()

        # Verify complex nested structure conversion
        assert len(result["nested_list"]) == 2
        assert result["nested_list"][0]["item1"]["_target_"] == _target_qualname(SimpleDataclass)
        assert result["nested_dict"]["group1"][0]["name"] == "group1_item1"


class TestConfigContainer_CallablesAndPartials:
    """Test ConfigContainer handling of callables and partial functions."""

    def test_dataclass_with_callables_to_dict(self):
        """Test converting dataclass with callables to dict."""
        callable_data = CallableDataclass()
        result = TestConfigContainer._convert_value_to_dict(callable_data)

        # Verify the structure includes _target_
        assert "_target_" in result
        assert _target_qualname(CallableDataclass) in result["_target_"]

        # Regular fields should be preserved
        assert result["name"] == "callable_test"
        assert result["regular_value"] == 100

        # Callable fields behavior depends on implementation
        # They might be excluded or handled specially
        # Let's check what's actually in the result
        print(f"Callable dataclass to_dict result: {result}")

    def test_config_container_with_callables_to_dict(self):
        """Test ConfigContainer with callable fields converted to dict."""
        config = CallableConfigContainer()
        result = config.to_dict()

        # Verify basic structure
        assert "_target_" in result
        assert result["name"] == "callable_config"

        # Check nested callable dataclass
        assert "callable_data" in result
        assert result["callable_data"]["name"] == "callable_test"
        assert result["callable_data"]["regular_value"] == 100

        # The actual handling of callables will depend on the implementation
        print(f"CallableConfigContainer to_dict result: {result}")

    def test_partial_function_handling(self):
        """Test handling of functools.partial objects."""
        partial_func = functools.partial(loss_function, reduction="sum")
        result = TestConfigContainer._convert_value_to_dict(partial_func)

        # Partial functions are callables and may be excluded or handled specially
        # The exact behavior depends on the implementation
        print(f"Partial function conversion result: {result}")

    def test_various_callable_types(self):
        """Test conversion of various callable types."""
        callables_to_test = [
            activation_function,  # Regular function
            functools.partial(loss_function, reduction="mean"),  # Partial function
            torch.nn.functional.relu,  # Torch function
            lambda x: x * 2,  # Lambda function
            torch.nn.ReLU(),  # Callable class instance
        ]

        for i, callable_obj in enumerate(callables_to_test):
            result = TestConfigContainer._convert_value_to_dict(callable_obj)
            print(f"Callable {i} ({type(callable_obj).__name__}) result: {result}")

    def test_config_with_callables_roundtrip_behavior(self):
        """Test the behavior of configs with callables in roundtrip scenarios."""
        config = CallableConfigContainer(name="roundtrip_test")

        # Convert to dict
        config_dict = config.to_dict()

        # Verify the structure is reasonable for potential reconstruction
        assert "_target_" in config_dict
        assert config_dict["name"] == "roundtrip_test"

        # The exact fields present will depend on how callables are handled
        # This test documents the actual behavior
        print(f"Roundtrip config dict keys: {list(config_dict.keys())}")

        # Verify nested structure
        if "callable_data" in config_dict:
            assert isinstance(config_dict["callable_data"], dict)
            assert "_target_" in config_dict["callable_data"]

    def test_callable_dataclass_field_exclusion(self):
        """Test which fields are excluded when dataclass contains callables."""
        callable_data = CallableDataclass()
        result = TestConfigContainer._convert_value_to_dict(callable_data)

        # Document which fields are preserved vs excluded
        expected_non_callable_fields = ["name", "regular_value"]
        callable_fields = ["activation_func", "loss_func", "torch_func", "lambda_func"]

        # Check if regular fields are preserved
        for field in expected_non_callable_fields:
            assert field in result, f"Regular field {field} should be preserved"

        # Document callable field handling
        for field in callable_fields:
            if field in result:
                print(f"Callable field {field} was preserved with value: {result[field]}")
            else:
                print(f"Callable field {field} was excluded from serialization")

    def test_mixed_container_with_callables_and_regular_data(self):
        """Test container mixing callable and regular data."""

        @dataclass
        class MixedConfig(ConfigContainerBase):
            name: str = "mixed"
            regular_list: list[str] = None
            callable_func: callable = activation_function
            nested_data: SimpleDataclass = None

            def __post_init__(self):
                if self.regular_list is None:
                    self.regular_list = ["a", "b", "c"]
                if self.nested_data is None:
                    self.nested_data = SimpleDataclass(name="nested", value=999)

        config = MixedConfig()
        result = config.to_dict()

        # Verify mixed content handling
        assert result["name"] == "mixed"
        assert result["regular_list"] == ["a", "b", "c"]
        assert result["nested_data"]["name"] == "nested"
        assert result["nested_data"]["value"] == 999

        # Document callable handling
        if "callable_func" in result:
            print(f"Callable func preserved as: {result['callable_func']}")
        else:
            print("Callable func was excluded from serialization")

    def test_deepcopy_with_callables(self):
        """Test deep copying ConfigContainer with callable fields."""
        config = CallableConfigContainer(name="deepcopy_test")

        # Verify original works
        assert config.name == "deepcopy_test"
        assert callable(config.activation)
        assert callable(config.partial_loss)

        # Test deep copy
        copied_config = copy.deepcopy(config)

        # Verify copy independence
        assert copied_config is not config
        assert copied_config.name == "deepcopy_test"

        # Verify callable fields are handled properly
        assert callable(copied_config.activation)
        assert callable(copied_config.partial_loss)

        # Test that functions still work
        assert copied_config.activation(5) == 10  # test_activation_function multiplies by 2

        # Modify original to verify independence
        config.name = "modified"
        assert copied_config.name == "deepcopy_test"


class TestConfigContainerBackwardCompat:
    """Test ConfigContainerBase integration with backward compatibility."""

    def test_from_dict_removes_init_false_fields(self):
        """Test from_dict removes init=False fields before instantiation."""
        # This test verifies the integration point in from_dict
        # Note: We can't directly test with DataclassWithInitFalse as it's not a ConfigContainer,
        # but we can test that the sanitization is called by checking no error is raised
        # when extra fields that would be init=False are present
        from megatron.training.config.instantiate_utils import target_allowlist

        config_dict = {
            "_target_": _target_qualname(TestConfigContainer),
            "name": "test_from_dict",
            "value": 42,
            "description": "test description",
        }

        # This should work without error
        target_allowlist.disable()
        result = TestConfigContainer.from_dict(config_dict, mode=InstantiationMode.STRICT)
        target_allowlist.enable()

        assert result.name == "test_from_dict"
        assert result.value == 42
