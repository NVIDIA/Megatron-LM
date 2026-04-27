# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from dataclasses import dataclass, field
from megatron.training.config.container import ConfigContainerBase


# Test dataclasses for testing
@dataclass
class SimpleDataclass:
    """Simple dataclass for testing."""

    name: str = "test"
    value: int = 42


@dataclass
class DataclassWithInitFalse:
    """Dataclass with init=False field for testing backward compatibility."""

    name: str = "test"
    value: int = 42
    computed_field: str = field(init=False, default="computed")

    def __post_init__(self):
        self.computed_field = f"computed_{self.name}"


@dataclass
class NestedDataclassWithInitFalse:
    """Nested dataclass with init=False field."""

    inner: DataclassWithInitFalse = None
    metadata: dict = field(default_factory=dict)
    cached_result: list = field(init=False, default_factory=list)


class TestBackwardCompatibility:
    """Test suite for backward compatibility functions."""

    def test_get_init_false_fields_with_init_false(self):
        """Test _get_init_false_fields correctly identifies init=False fields."""
        from megatron.training.config.utils import _get_init_false_fields

        result = _get_init_false_fields(DataclassWithInitFalse)
        assert "computed_field" in result
        assert "name" not in result
        assert "value" not in result

    def test_get_init_false_fields_no_init_false(self):
        """Test _get_init_false_fields returns empty set for normal dataclass."""
        from megatron.training.config.utils import _get_init_false_fields

        result = _get_init_false_fields(SimpleDataclass)
        assert result == frozenset()

    def test_get_init_false_fields_non_dataclass(self):
        """Test _get_init_false_fields returns empty set for non-dataclass."""
        from megatron.training.config.utils import _get_init_false_fields

        result = _get_init_false_fields(str)
        assert result == frozenset()

    def test_resolve_target_class_valid(self):
        """Test _resolve_target_class resolves valid class path."""
        from megatron.training.config.utils import _resolve_target_class

        result = _resolve_target_class(
            "megatron.bridge.training.utils.config_utils.ConfigContainerBase"
        )
        assert result is ConfigContainerBase

    def test_resolve_target_class_invalid(self):
        """Test _resolve_target_class returns None for invalid path."""
        from megatron.training.config.utils import _resolve_target_class

        result = _resolve_target_class("nonexistent.module.ClassName")
        assert result is None

    def test_resolve_target_class_malformed(self):
        """Test _resolve_target_class handles malformed paths gracefully."""
        from megatron.training.config.utils import _resolve_target_class

        result = _resolve_target_class("no_dots")
        assert result is None

    def test_sanitize_dataclass_config_removes_init_false_fields(self):
        """Test sanitize_dataclass_config removes init=False fields."""
        from megatron.training.config.utils import sanitize_dataclass_config

        config = {
            "_target_": "tests.unit_tests.training.utils.test_config_utils.DataclassWithInitFalse",
            "name": "test_name",
            "value": 123,
            "computed_field": "should_be_removed",
        }

        result = sanitize_dataclass_config(config)

        assert "name" in result
        assert "value" in result
        assert "_target_" in result
        assert "computed_field" not in result

    def test_sanitize_dataclass_config_preserves_normal_fields(self):
        """Test sanitize_dataclass_config preserves fields without init=False."""
        from megatron.training.config.utils import sanitize_dataclass_config

        config = {
            "_target_": "tests.unit_tests.training.utils.test_config_utils.SimpleDataclass",
            "name": "preserved",
            "value": 999,
        }

        result = sanitize_dataclass_config(config)

        assert result["name"] == "preserved"
        assert result["value"] == 999
        assert result["_target_"] == config["_target_"]

    def test_sanitize_dataclass_config_handles_nested_configs(self):
        """Test sanitize_dataclass_config recursively processes nested configs."""
        from megatron.training.config.utils import sanitize_dataclass_config

        config = {
            "_target_": "tests.unit_tests.training.utils.test_config_utils.NestedDataclassWithInitFalse",
            "inner": {
                "_target_": "tests.unit_tests.training.utils.test_config_utils.DataclassWithInitFalse",
                "name": "inner_test",
                "value": 42,
                "computed_field": "nested_computed_should_be_removed",
            },
            "metadata": {"key": "value"},
            "cached_result": ["should", "be", "removed"],
        }

        result = sanitize_dataclass_config(config)

        # Top-level init=False field removed
        assert "cached_result" not in result
        # Nested init=False field removed
        assert "computed_field" not in result["inner"]
        # Normal fields preserved
        assert result["inner"]["name"] == "inner_test"
        assert result["metadata"] == {"key": "value"}

    def test_sanitize_dataclass_config_handles_lists_of_configs(self):
        """Test sanitize_dataclass_config processes lists containing configs."""
        from megatron.training.config.utils import sanitize_dataclass_config

        config = {
            "_target_": "some.module.ListContainer",
            "items": [
                {
                    "_target_": "tests.unit_tests.training.utils.test_config_utils.DataclassWithInitFalse",
                    "name": "item1",
                    "computed_field": "remove_me",
                },
                {
                    "_target_": "tests.unit_tests.training.utils.test_config_utils.DataclassWithInitFalse",
                    "name": "item2",
                    "computed_field": "remove_me_too",
                },
            ],
        }

        result = sanitize_dataclass_config(config)

        assert "computed_field" not in result["items"][0]
        assert "computed_field" not in result["items"][1]
        assert result["items"][0]["name"] == "item1"
        assert result["items"][1]["name"] == "item2"

    def test_sanitize_dataclass_config_no_target(self):
        """Test sanitize_dataclass_config handles dicts without _target_."""
        from megatron.training.config.utils import sanitize_dataclass_config

        config = {"key": "value", "number": 42}
        result = sanitize_dataclass_config(config)

        assert result == config

    def test_sanitize_dataclass_config_non_dict_input(self):
        """Test sanitize_dataclass_config handles non-dict input."""
        from megatron.training.config.utils import sanitize_dataclass_config

        assert sanitize_dataclass_config("string") == "string"
        assert sanitize_dataclass_config(42) == 42
        assert sanitize_dataclass_config(None) is None

    def test_sanitize_dataclass_config_unresolvable_target(self):
        """Test sanitize_dataclass_config handles unresolvable _target_."""
        from megatron.training.config.utils import sanitize_dataclass_config

        config = {"_target_": "nonexistent.module.Class", "field1": "value1", "field2": "value2"}

        result = sanitize_dataclass_config(config)

        # All fields preserved when target can't be resolved
        assert result == config

    def test_sanitize_dataclass_config_sanitizes_model(self):
        """Test sanitize_dataclass_config sanitizes model section with init=False fields."""
        from megatron.training.config.utils import sanitize_dataclass_config

        run_config = {
            "model": {
                "_target_": "tests.unit_tests.training.utils.test_config_utils.DataclassWithInitFalse",
                "name": "model_name",
                "value": 100,
                "computed_field": "should_be_removed",
            },
            "training": {"lr": 0.001},
            "tokenizer": {"type": "sentencepiece"},
        }

        result = sanitize_dataclass_config(run_config)

        assert "computed_field" not in result["model"]
        assert result["model"]["name"] == "model_name"
        assert result["training"] == {"lr": 0.001}
        assert result["tokenizer"] == {"type": "sentencepiece"}

    def test_sanitize_dataclass_config_no_model_section(self):
        """Test sanitize_dataclass_config handles config without model section."""
        from megatron.training.config.utils import sanitize_dataclass_config

        run_config = {"training": {"lr": 0.001}}
        result = sanitize_dataclass_config(run_config)

        assert result == run_config

    def test_sanitize_dataclass_config_non_dict(self):
        """Test sanitize_dataclass_config handles non-dict input."""
        from megatron.training.config.utils import sanitize_dataclass_config

        assert sanitize_dataclass_config("string") == "string"
        assert sanitize_dataclass_config(None) is None

    def test_sanitize_dataclass_config_sanitizes_all_sections(self):
        """Test sanitize_dataclass_config sanitizes all sections, not just model."""
        from megatron.training.config.utils import sanitize_dataclass_config

        run_config = {
            "model": {
                "_target_": "tests.unit_tests.training.utils.test_config_utils.DataclassWithInitFalse",
                "name": "model_name",
                "computed_field": "should_be_removed_from_model",
            },
            "training": {
                "_target_": "tests.unit_tests.training.utils.test_config_utils.DataclassWithInitFalse",
                "name": "training_config",
                "computed_field": "should_be_removed_from_training",
            },
            "data": {
                "_target_": "tests.unit_tests.training.utils.test_config_utils.DataclassWithInitFalse",
                "name": "data_config",
                "computed_field": "should_be_removed_from_data",
            },
        }

        result = sanitize_dataclass_config(run_config)

        # All sections should have init=False fields removed
        assert "computed_field" not in result["model"]
        assert "computed_field" not in result["training"]
        assert "computed_field" not in result["data"]

        # Regular fields should be preserved
        assert result["model"]["name"] == "model_name"
        assert result["training"]["name"] == "training_config"
        assert result["data"]["name"] == "data_config"
