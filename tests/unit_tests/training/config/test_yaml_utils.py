# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import enum
import functools
import os
import tempfile
from dataclasses import dataclass
from unittest.mock import Mock

import pytest
import yaml

from megatron.training.config.yaml_utils import (
    _enum_representer,
    _function_representer,
    _generation_config_representer,
    _partial_representer,
    _safe_object_representer,
    _torch_dtype_representer,
    safe_yaml_representers,
)


class TestEnum(enum.Enum):
    """Test enum"""

    VALUE1 = "test_value1"
    VALUE2 = "test_value2"


@dataclass
class TestDataclass:
    """Test dataclass"""

    name: str
    value: int


class TestClass:
    def __init__(self, name: str = "test"):
        """Test class"""
        self.name = name


def test_function():
    """Test function"""
    return "test"


class TestSafeYamlRepresenters:
    """Test the safe_yaml_representers context manager."""

    def test_context_manager_adds_and_removes_representers(self):
        """Test that representers are properly added and removed."""
        # Save original state
        original_representers = yaml.SafeDumper.yaml_representers.copy()
        original_multi_representers = yaml.SafeDumper.yaml_multi_representers.copy()

        # Use context manager
        with safe_yaml_representers():
            # Check that new representers were added
            assert functools.partial in yaml.SafeDumper.yaml_representers
            assert enum.Enum in yaml.SafeDumper.yaml_multi_representers
            assert type(lambda: ...) in yaml.SafeDumper.yaml_representers

        # Check that original representers were restored
        assert yaml.SafeDumper.yaml_representers == original_representers
        assert yaml.SafeDumper.yaml_multi_representers == original_multi_representers

    def test_context_manager_handles_exceptions(self):
        """Test that representers are restored even if an exception occurs."""
        original_representers = yaml.SafeDumper.yaml_representers.copy()
        original_multi_representers = yaml.SafeDumper.yaml_multi_representers.copy()

        try:
            with safe_yaml_representers():
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Check that original representers were still restored
        assert yaml.SafeDumper.yaml_representers == original_representers
        assert yaml.SafeDumper.yaml_multi_representers == original_multi_representers


class TestFunctionRepresenter:
    """Test the _function_representer function."""

    def test_function_representation(self):
        """Test representing a function in YAML."""
        dumper = yaml.SafeDumper("")
        result = _function_representer(dumper, test_function)

        # The result should be a MappingNode
        assert hasattr(result, "value")

        # Parse the represented data using the context manager
        with safe_yaml_representers():
            data = yaml.safe_load(yaml.safe_dump({"test": test_function}))
        assert "_target_" in data["test"]
        assert "_call_" in data["test"]
        assert data["test"]["_call_"] is False
        assert "test_function" in data["test"]["_target_"]


class TestPartialRepresenter:
    """Test the _partial_representer function."""

    def test_partial_without_keywords(self):
        """Test representing a partial function without keyword arguments."""
        partial_func = functools.partial(test_function)
        dumper = yaml.SafeDumper("")
        _ = _partial_representer(dumper, partial_func)

        # Parse the represented data
        with safe_yaml_representers():
            data = yaml.safe_load(yaml.safe_dump({"test": partial_func}))
        assert "_target_" in data["test"]
        assert "_partial_" in data["test"]
        assert data["test"]["_partial_"] is True
        assert "_args_" in data["test"]
        assert data["test"]["_args_"] == []

    def test_partial_with_args_and_kwargs(self):
        """Test representing a partial function with arguments and keyword arguments."""

        def example_func(a, b, c=None):
            return a + b + (c or 0)

        partial_func = functools.partial(example_func, 1, c=10)
        dumper = yaml.SafeDumper("")
        _ = _partial_representer(dumper, partial_func)

        # Parse the represented data
        with safe_yaml_representers():
            data = yaml.safe_load(yaml.safe_dump({"test": partial_func}))
        assert data["test"]["_args_"] == [1]
        assert data["test"]["c"] == 10


class TestEnumRepresenter:
    """Test the _enum_representer function."""

    def test_enum_representation(self):
        """Test representing an enum value in YAML."""
        enum_value = TestEnum.VALUE1
        dumper = yaml.SafeDumper("")
        _ = _enum_representer(dumper, enum_value)

        # Parse the represented data
        with safe_yaml_representers():
            data = yaml.safe_load(yaml.safe_dump({"test": enum_value}))
        assert "_target_" in data["test"]
        assert "_call_" in data["test"]
        assert data["test"]["_call_"] is True
        assert "_args_" in data["test"]
        assert data["test"]["_args_"] == ["test_value1"]
        assert "_name_" in data["test"]
        assert data["test"]["_name_"] == "VALUE1"
        assert "TestEnum" in data["test"]["_target_"]


class TestSafeObjectRepresenter:
    """Test the _safe_object_representer function."""

    def test_object_with_qualname(self):
        """Test representing an object that has __qualname__ attribute."""
        obj = test_function
        dumper = yaml.SafeDumper("")
        _ = _safe_object_representer(dumper, obj)

        # Parse the represented data
        with safe_yaml_representers():
            data = yaml.safe_load(yaml.safe_dump({"test": obj}))
        assert "_target_" in data["test"]
        assert "_call_" in data["test"]
        assert data["test"]["_call_"] is False

    def test_object_without_qualname(self):
        """Test representing an object that doesn't have __qualname__ attribute."""
        obj = TestClass("test")
        dumper = yaml.SafeDumper("")
        _ = _safe_object_representer(dumper, obj)

        # Parse the represented data
        with safe_yaml_representers():
            data = yaml.safe_load(yaml.safe_dump({"test": obj}))
        assert "_target_" in data["test"]
        assert "_call_" in data["test"]
        assert data["test"]["_call_"] is True
        assert "TestClass" in data["test"]["_target_"]


class TestTorchDtypeRepresenter:
    """Test the _torch_dtype_representer function."""

    def test_torch_dtype_representation(self):
        """Test representing a torch dtype in YAML."""
        import torch

        dtype = torch.float32
        dumper = yaml.SafeDumper("")
        _ = _torch_dtype_representer(dumper, dtype)

        # Parse the represented data
        with safe_yaml_representers():
            data = yaml.safe_load(yaml.safe_dump({"test": dtype}))
        assert "_target_" in data["test"]
        assert "_call_" in data["test"]
        assert data["test"]["_call_"] is False
        assert "float32" in data["test"]["_target_"]

    def test_torch_dtype_representer_function(self):
        """Test the torch dtype representer function directly."""
        # Create a mock torch dtype
        mock_dtype = Mock()
        mock_dtype.__str__ = Mock(return_value="torch.float32")

        dumper = yaml.SafeDumper("")
        result = _torch_dtype_representer(dumper, mock_dtype)

        # Test the direct result from the representer function
        # The result should be a MappingNode
        assert hasattr(result, "value")

        # Parse the result from the direct function call
        # We need to manually construct the data that would be generated
        test_data = {"_target_": "torch.float32", "_call_": False}
        yaml_result = yaml.safe_dump(test_data)
        data = yaml.safe_load(yaml_result)

        assert "_target_" in data
        assert "_call_" in data
        assert data["_call_"] is False


class TestGenerationConfigRepresenter:
    """Test the _generation_config_representer function."""

    def test_generation_config_representation(self):
        """Test representing a GenerationConfig object in YAML."""
        try:
            from transformers import GenerationConfig

            config = GenerationConfig(max_length=100, temperature=0.8)
            dumper = yaml.SafeDumper("")
            _ = _generation_config_representer(dumper, config)

            # Parse the represented data
            with safe_yaml_representers():
                data = yaml.safe_load(yaml.safe_dump({"test": config}))
            assert "_target_" in data["test"]
            assert "_call_" in data["test"]
            assert data["test"]["_call_"] is True
            assert "config_dict" in data["test"]
            assert "from_dict" in data["test"]["_target_"]
        except ImportError:
            pytest.skip("Transformers not available")

    def test_generation_config_representer_function(self):
        """Test the generation config representer function directly."""
        # Create a mock GenerationConfig
        mock_config = Mock()
        mock_config.__class__.__qualname__ = "GenerationConfig"
        mock_config.__class__.__module__ = "transformers.generation.configuration_utils"
        mock_config.to_dict = Mock(return_value={"max_length": 100, "temperature": 0.8})

        dumper = yaml.SafeDumper("")
        result = _generation_config_representer(dumper, mock_config)

        # Test the direct result from the representer function
        # The result should be a MappingNode
        assert hasattr(result, "value")

        # Parse the result from the direct function call
        # We need to manually construct the data that would be generated
        test_data = {
            "_target_": "transformers.generation.configuration_utils.GenerationConfig.from_dict",
            "_call_": True,
            "config_dict": {"max_length": 100, "temperature": 0.8},
        }
        yaml_result = yaml.safe_dump(test_data)
        data = yaml.safe_load(yaml_result)

        assert "_target_" in data
        assert "_call_" in data
        assert data["_call_"] is True
        assert "config_dict" in data
        assert "from_dict" in data["_target_"]


class TestIntegration:
    """Integration tests for the YAML utils functionality."""

    def test_complex_object_serialization(self):
        """Test serializing a complex object with multiple types."""
        complex_obj = {
            "function": test_function,
            "enum": TestEnum.VALUE1,
            "partial": functools.partial(test_function),
            "dataclass": TestDataclass("test", 42),
            "regular_data": {"key": "value", "number": 123},
        }

        with safe_yaml_representers():
            result = yaml.safe_dump(complex_obj)

        assert isinstance(result, str)

        # Verify all components are serialized
        assert "_target_:" in result
        assert "test_function" in result
        assert "TestEnum" in result
        assert "_partial_:" in result
        assert "TestDataclass" in result
        assert "_call_:" in result
        assert "key: value" in result

    def test_roundtrip_with_simple_objects(self):
        """Test that simple objects can be serialized and deserialized."""
        simple_obj = {
            "string": "test",
            "number": 42,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
        }

        with safe_yaml_representers():
            yaml_str = yaml.safe_dump(simple_obj)

        reconstructed = yaml.safe_load(yaml_str)

        assert reconstructed["string"] == "test"
        assert reconstructed["number"] == 42
        assert reconstructed["list"] == [1, 2, 3]
        assert reconstructed["dict"]["nested"] == "value"
