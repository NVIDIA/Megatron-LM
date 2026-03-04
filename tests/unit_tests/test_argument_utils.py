# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import signal
from argparse import ArgumentError, ArgumentParser
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Union

import pytest

from megatron.training.argument_utils import ArgumentGroupFactory, TypeInferenceError


@dataclass
class DummyConfig:
    """A dummy configuration for testing."""

    name: str = "default_name"
    """Name of the configuration"""

    count: int = 42
    """Number of items"""

    learning_rate: float = 0.001
    """Learning rate for training"""

    enabled: bool = False
    """Whether feature is enabled"""

    disabled_feature: bool = True
    """Feature that is disabled by default"""

    enum_setting: signal.Signals = signal.SIGTERM
    """Setting with enum type to test enum handling"""


@dataclass
class ConfigWithOptional:
    """Config with optional fields."""

    required_field: str = "required"
    """A required field"""

    optional_field: Optional[int] = None
    """An optional integer field"""

    optional_str: Optional[str] = "default"
    """An optional string with default"""

    int_new_form: int | None = None
    """Optional using new syntax"""

    str_new_form: str | None = "default"
    """Optional string using new syntax"""


@dataclass
class ConfigWithList:
    """Config with list fields."""

    tags: list[str] = field(default_factory=list)
    """List of tags"""

    numbers: list[int] = field(default_factory=lambda: [1, 2, 3])
    """List of numbers with default"""


@dataclass
class ConfigWithLiteral:
    """Config with Literal types."""

    mode: Literal["train", "eval", "test"] = "train"
    """Operating mode"""

    precision: Literal[16, 32] = 32
    """Precision level"""


class TestArgumentGroupFactoryBasic:
    """Test basic functionality of ArgumentGroupFactory."""

    def test_creates_argument_group(self):
        """Test that build_group creates an argument group."""
        parser = ArgumentParser()
        factory = ArgumentGroupFactory(DummyConfig)

        arg_group = factory.build_group(parser, title="Test Group")

        assert arg_group is not None
        assert arg_group.title == "Test Group"
        assert arg_group.description == DummyConfig.__doc__

    def test_all_fields_added(self):
        """Test that all dataclass fields are added as arguments."""
        parser = ArgumentParser()
        factory = ArgumentGroupFactory(DummyConfig)

        factory.build_group(parser, title="Test Group")

        # Parse empty args to get all defaults
        args = parser.parse_args([])

        # Check all fields exist
        assert hasattr(args, 'name')
        assert hasattr(args, 'count')
        assert hasattr(args, 'learning_rate')
        assert hasattr(args, 'enabled')
        assert hasattr(args, 'disabled_feature')

    def test_default_values_preserved(self):
        """Test that default values from dataclass are preserved."""
        parser = ArgumentParser()
        factory = ArgumentGroupFactory(DummyConfig)

        factory.build_group(parser, title="Test Group")
        args = parser.parse_args([])

        assert args.name == "default_name"
        assert args.count == 42
        assert args.learning_rate == 0.001
        assert args.enabled == False
        assert args.disabled_feature == True

    def test_argument_types(self):
        """Test that argument types are correctly inferred."""
        parser = ArgumentParser()
        factory = ArgumentGroupFactory(DummyConfig)

        factory.build_group(parser, title="Test Group")

        # Parse with actual values
        args = parser.parse_args(
            ['--name', 'test_name', '--count', '100', '--learning-rate', '0.01']
        )

        assert isinstance(args.name, str)
        assert args.name == 'test_name'
        assert isinstance(args.count, int)
        assert args.count == 100
        assert isinstance(args.learning_rate, float)
        assert args.learning_rate == 0.01

    def test_boolean_store_true(self):
        """Test that boolean fields with default False use store_true."""
        parser = ArgumentParser()
        factory = ArgumentGroupFactory(DummyConfig)

        factory.build_group(parser, title="Test Group")

        # Without flag, should be False
        args = parser.parse_args([])
        assert args.enabled == False

        # With flag, should be True
        args = parser.parse_args(['--enabled'])
        assert args.enabled == True

    def test_boolean_store_false(self):
        """Test that boolean fields with default True use store_false with no- prefix."""
        parser = ArgumentParser()
        factory = ArgumentGroupFactory(DummyConfig)

        factory.build_group(parser, title="Test Group")

        # Without flag, should be True
        args = parser.parse_args([])
        assert args.disabled_feature == True

        # With --no- flag, should be False
        args = parser.parse_args(['--no-disabled-feature'])
        assert args.disabled_feature == False

        # With --disable- flag, should also be False
        args = parser.parse_args(['--disable-disabled-feature'])
        assert args.disabled_feature == False

    def test_field_docstrings_as_help(self):
        """Test that field docstrings are extracted and used as help text."""
        parser = ArgumentParser()
        factory = ArgumentGroupFactory(DummyConfig)

        # Check that field_docstrings were extracted
        assert 'name' in factory.field_docstrings
        assert factory.field_docstrings['name'] == "Name of the configuration"
        assert factory.field_docstrings['count'] == "Number of items"
        assert factory.field_docstrings['learning_rate'] == "Learning rate for training"

    def test_enum_handling(self):
        """Test that enum types are handled correctly."""
        parser = ArgumentParser(exit_on_error=False)
        factory = ArgumentGroupFactory(DummyConfig)

        factory.build_group(parser, title="Test Group")

        args = parser.parse_args([])
        assert args.enum_setting == signal.SIGTERM

        # test a different valid enum value
        args = parser.parse_args(["--enum-setting", "SIGINT"])
        assert args.enum_setting == signal.SIGINT

        # test an invalid enum value
        with pytest.raises(KeyError, match="sigbar"):
            parser.parse_args(["--enum-setting", "sigbar"])


class TestArgumentGroupFactoryExclusion:
    """Test exclusion functionality."""

    def test_exclude_single_field(self):
        """Test excluding a single field."""
        parser = ArgumentParser()
        factory = ArgumentGroupFactory(DummyConfig, exclude=['count'])

        factory.build_group(parser, title="Test Group")
        args = parser.parse_args([])

        # Excluded field should not exist
        assert hasattr(args, 'name')
        assert not hasattr(args, 'count')
        assert hasattr(args, 'learning_rate')

    def test_exclude_multiple_fields(self):
        """Test excluding multiple fields."""
        parser = ArgumentParser()
        factory = ArgumentGroupFactory(DummyConfig, exclude=['count', 'learning_rate'])

        factory.build_group(parser, title="Test Group")
        args = parser.parse_args([])

        assert hasattr(args, 'name')
        assert not hasattr(args, 'count')
        assert not hasattr(args, 'learning_rate')
        assert hasattr(args, 'enabled')


class TestArgumentGroupFactoryOptional:
    """Test handling of Optional types."""

    def test_optional_fields(self):
        """Test that Optional fields are handled correctly."""
        parser = ArgumentParser()
        factory = ArgumentGroupFactory(ConfigWithOptional)

        factory.build_group(parser, title="Test Group")

        # Default values
        args = parser.parse_args([])
        assert args.required_field == "required"
        assert args.optional_field is None
        assert args.optional_str == "default"

        # Provided values
        args = parser.parse_args(
            ['--required-field', 'new_value', '--optional-field', '123', '--optional-str', 'custom']
        )
        assert args.required_field == "new_value"
        assert args.optional_field == 123
        assert args.optional_str == "custom"


class TestArgumentGroupFactoryList:
    """Test handling of list types."""

    def test_list_fields_with_default_factory(self):
        """Test that list fields use nargs='+'."""
        parser = ArgumentParser()
        factory = ArgumentGroupFactory(ConfigWithList)

        factory.build_group(parser, title="Test Group")

        # Default values
        args = parser.parse_args([])
        assert args.tags == []
        assert args.numbers == [1, 2, 3]

        # Provided values
        args = parser.parse_args(['--tags', 'tag1', 'tag2', 'tag3', '--numbers', '10', '20', '30'])
        assert args.tags == ['tag1', 'tag2', 'tag3']
        assert args.numbers == [10, 20, 30]


class TestArgumentGroupFactoryLiteral:
    """Test handling of Literal types."""

    def test_literal_fields_have_choices(self):
        """Test that Literal types create choice constraints."""
        parser = ArgumentParser()
        factory = ArgumentGroupFactory(ConfigWithLiteral)

        factory.build_group(parser, title="Test Group")

        # Default values
        args = parser.parse_args([])
        assert args.mode == "train"
        assert args.precision == 32

        # Valid choices
        args = parser.parse_args(['--mode', 'eval', '--precision', '16'])
        assert args.mode == "eval"
        assert args.precision == 16

    def test_literal_fields_reject_invalid_choices(self):
        """Test that invalid Literal choices are rejected."""
        parser = ArgumentParser()
        factory = ArgumentGroupFactory(ConfigWithLiteral)

        factory.build_group(parser, title="Test Group")

        # Invalid choice should raise error
        with pytest.raises(SystemExit):
            parser.parse_args(['--mode', 'invalid'])

        with pytest.raises(SystemExit):
            parser.parse_args(['--precision', '64'])


class TestArgumentGroupFactoryHelpers:
    """Test helper methods."""

    def test_format_arg_name_basic(self):
        """Test basic argument name formatting."""
        factory = ArgumentGroupFactory(DummyConfig)

        assert factory._format_arg_name("simple") == "--simple"
        assert factory._format_arg_name("with_underscore") == "--with-underscore"
        assert factory._format_arg_name("multiple_under_scores") == "--multiple-under-scores"

    def test_format_arg_name_with_prefix(self):
        """Test argument name formatting with prefix."""
        factory = ArgumentGroupFactory(DummyConfig)

        assert factory._format_arg_name("feature", prefix="no") == "--no-feature"
        assert factory._format_arg_name("feature", prefix="disable") == "--disable-feature"
        assert factory._format_arg_name("multi_word", prefix="no") == "--no-multi-word"

    def test_extract_type_primitive(self):
        """Test type extraction for primitive types."""
        factory = ArgumentGroupFactory(DummyConfig)

        assert factory._extract_type(int) == {"type": int}
        assert factory._extract_type(str) == {"type": str}
        assert factory._extract_type(float) == {"type": float}

    def test_extract_type_optional(self):
        """Test type extraction for Optional types."""
        factory = ArgumentGroupFactory(DummyConfig)

        result = factory._extract_type(Optional[int])
        assert result == {"type": int}

        result = factory._extract_type(Optional[str])
        assert result == {"type": str}

    def test_extract_type_list(self):
        """Test type extraction for list types."""
        factory = ArgumentGroupFactory(DummyConfig)

        result = factory._extract_type(list[int])
        assert result == {"type": int, "nargs": "+"}

        result = factory._extract_type(list[str])
        assert result == {"type": str, "nargs": "+"}

    def test_extract_type_literal(self):
        """Test type extraction for Literal types."""
        factory = ArgumentGroupFactory(DummyConfig)

        result = factory._extract_type(Literal["a", "b", "c"])
        assert result == {"type": str, "choices": ("a", "b", "c")}

        result = factory._extract_type(Literal[1, 2, 3])
        assert result == {"type": int, "choices": (1, 2, 3)}


@dataclass
class ConfigWithArgparseMeta:
    """Config with argparse_meta metadata for testing overrides."""

    custom_help: str = field(
        default="default_value",
        metadata={"argparse_meta": {"help": "Custom help text from metadata"}},
    )
    """Original help text"""

    custom_type: str = field(default="100", metadata={"argparse_meta": {"type": int}})
    """Field with type override"""

    custom_default: str = field(
        default="original_default", metadata={"argparse_meta": {"default": "overridden_default"}}
    )
    """Field with default override"""

    custom_choices: str = field(
        default="option1",
        metadata={"argparse_meta": {"choices": ["option1", "option2", "option3"]}},
    )
    """Field with choices override"""

    custom_dest: str = field(
        default="value", metadata={"argparse_meta": {"dest": "renamed_destination"}}
    )
    """Field with dest override"""

    custom_action: bool = field(
        default=False,
        metadata={"argparse_meta": {"action": "store_const", "const": "special_value"}},
    )
    """Field with custom action override"""

    multiple_overrides: int = field(
        default=42,
        metadata={
            "argparse_meta": {
                "type": str,
                "help": "Multiple overrides applied",
                "default": "999",
                "dest": "multi_override_dest",
            }
        },
    )
    """Field with multiple metadata overrides"""

    nargs_override: str = field(default="single", metadata={"argparse_meta": {"nargs": "?"}})
    """Field with nargs override"""


@dataclass
class ConfigWithUnsupportedCallables:
    """Config with argparse_meta metadata for testing overrides."""

    unsupported_type: Optional[Callable] = None
    """Cannot take a callable over CLI"""

    unsupported_with_metadata: Optional[Callable] = field(
        default=None, metadata={"argparse_meta": {"type": int, "choices": (0, 1, 2)}}
    )
    """This argument should be 0, 1, or 2. The appropriate
    Callable will be set by some other logic.
    """


@dataclass
class ConfigWithUnsupportedUnions:
    """Config with argparse_meta metadata for testing overrides."""

    unsupported_type: Union[int, str] = 0
    """Cannot infer type of a Union"""

    unsupported_with_metadata: Union[int, str] = field(
        default=0, metadata={"argparse_meta": {"type": str, "choices": ("foo", "bar")}}
    )
    """Metadata should take precedence over the exception caused by Union"""


class TestArgumentGroupFactoryArgparseMeta:
    """Test argparse_meta metadata override functionality."""

    def test_help_override(self):
        """Test that argparse_meta can override help text."""
        parser = ArgumentParser()
        factory = ArgumentGroupFactory(ConfigWithArgparseMeta)

        factory.build_group(parser, title="Test Group")

        # Find the action for this argument
        for action in parser._actions:
            if hasattr(action, 'dest') and action.dest == 'custom_help':
                assert action.help == "Custom help text from metadata"
                return

        pytest.fail("custom_help argument not found")

    def test_type_override(self):
        """Test that argparse_meta can override argument type."""
        parser = ArgumentParser()
        factory = ArgumentGroupFactory(ConfigWithArgparseMeta)

        factory.build_group(parser, title="Test Group")

        # Parse with integer value (metadata overrides type to int)
        args = parser.parse_args(['--custom-type', '42'])

        # Should be parsed as int, not str
        assert isinstance(args.custom_type, int)
        assert args.custom_type == 42

    def test_default_override(self):
        """Test that argparse_meta can override default value."""
        parser = ArgumentParser()
        factory = ArgumentGroupFactory(ConfigWithArgparseMeta)

        factory.build_group(parser, title="Test Group")

        # Parse with no arguments
        args = parser.parse_args([])

        # Should use metadata default, not field default
        assert args.custom_default == "overridden_default"

    def test_choices_override(self):
        """Test that argparse_meta can override choices."""
        parser = ArgumentParser()
        factory = ArgumentGroupFactory(ConfigWithArgparseMeta)

        factory.build_group(parser, title="Test Group")

        # Valid choice from metadata
        args = parser.parse_args(['--custom-choices', 'option2'])
        assert args.custom_choices == "option2"

        # Invalid choice should fail
        with pytest.raises(SystemExit):
            parser.parse_args(['--custom-choices', 'invalid_option'])

    def test_dest_override(self):
        """Test that argparse_meta can override destination name."""
        parser = ArgumentParser()
        factory = ArgumentGroupFactory(ConfigWithArgparseMeta)

        factory.build_group(parser, title="Test Group")

        args = parser.parse_args(['--custom-dest', 'test_value'])

        # Should be stored in renamed destination
        assert hasattr(args, 'renamed_destination')
        assert args.renamed_destination == "test_value"

    def test_action_override(self):
        """Test that argparse_meta can override action."""
        parser = ArgumentParser()
        factory = ArgumentGroupFactory(ConfigWithArgparseMeta)

        factory.build_group(parser, title="Test Group")

        # With custom action=store_const and const="special_value"
        args = parser.parse_args(['--custom-action'])
        assert args.custom_action == "special_value"

        # Without flag, should use default
        args = parser.parse_args([])
        assert args.custom_action == False

    def test_multiple_overrides(self):
        """Test that multiple argparse_meta overrides work together."""
        parser = ArgumentParser()
        factory = ArgumentGroupFactory(ConfigWithArgparseMeta)

        factory.build_group(parser, title="Test Group")

        # Parse with no arguments to check default override
        args = parser.parse_args([])

        # Check all overrides applied
        assert hasattr(args, 'multi_override_dest')
        assert args.multi_override_dest == "999"  # default override

        # Parse with value to check type override
        args = parser.parse_args(['--multiple-overrides', 'text_value'])
        assert isinstance(args.multi_override_dest, str)  # type override
        assert args.multi_override_dest == "text_value"

        # Check help override was applied
        for action in parser._actions:
            if hasattr(action, 'dest') and action.dest == 'multi_override_dest':
                assert action.help == "Multiple overrides applied"
                break

    def test_nargs_override(self):
        """Test that argparse_meta can override nargs."""
        parser = ArgumentParser()
        factory = ArgumentGroupFactory(ConfigWithArgparseMeta)

        factory.build_group(parser, title="Test Group")

        # With nargs='?', argument is optional
        args = parser.parse_args(['--nargs-override'])
        assert args.nargs_override is None  # No value provided with '?'

        # With value
        args = parser.parse_args(['--nargs-override', 'provided_value'])
        assert args.nargs_override == "provided_value"

        # Without flag at all, should use default
        args = parser.parse_args([])
        assert args.nargs_override == "single"

    def test_metadata_takes_precedence_over_inference(self):
        """Test that metadata has highest precedence over type inference."""
        parser = ArgumentParser()
        factory = ArgumentGroupFactory(ConfigWithArgparseMeta)

        # Build kwargs for custom_type field which is str but metadata says int
        from dataclasses import fields as dc_fields

        for f in dc_fields(ConfigWithArgparseMeta):
            if f.name == 'custom_type':
                kwargs = factory._build_argparse_kwargs_from_field(f)
                # Metadata type should override inferred type
                assert kwargs['type'] == int
                break

    def test_unhandled_unsupported_callables(self):
        """Test that an unsupported type produces a TypInferenceError."""
        parser = ArgumentParser()
        factory = ArgumentGroupFactory(
            ConfigWithUnsupportedCallables, exclude=["unsupported_with_metadata"]
        )

        with pytest.raises(TypeInferenceError, match="Unsupported type"):
            factory.build_group(parser, title="Test Group")

    def test_handled_unsupported_callables(self):
        """Test an attribute with an unsupported type that has type info in the metadata."""
        parser = ArgumentParser()
        factory = ArgumentGroupFactory(ConfigWithUnsupportedCallables, exclude=["unsupported_type"])

        factory.build_group(parser, title="Test Group")

        args = parser.parse_args(['--unsupported-with-metadata', '0'])
        assert args.unsupported_with_metadata == 0

    def test_unhandled_unsupported_unions(self):
        """Test that an unsupported type produces a TypInferenceError."""
        parser = ArgumentParser()
        factory = ArgumentGroupFactory(
            ConfigWithUnsupportedUnions, exclude=["unsupported_with_metadata"]
        )

        with pytest.raises(TypeInferenceError, match="Unions not supported by argparse"):
            factory.build_group(parser, title="Test Group")

    def test_handled_unsupported_unions(self):
        """Test an attribute with an unsupported type that has type info in the metadata."""
        parser = ArgumentParser(exit_on_error=False)
        factory = ArgumentGroupFactory(ConfigWithUnsupportedUnions, exclude=["unsupported_type"])

        factory.build_group(parser, title="Test Group")

        args = parser.parse_args(['--unsupported-with-metadata', 'foo'])
        assert args.unsupported_with_metadata == 'foo'

        with pytest.raises(ArgumentError, match="invalid choice"):
            args = parser.parse_args(['--unsupported-with-metadata', 'baz'])
