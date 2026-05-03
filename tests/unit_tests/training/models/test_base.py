# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from dataclasses import dataclass, field
from typing import Callable, ClassVar
from unittest.mock import Mock

import pytest

from megatron.training.models.base import ModelBuilder, ModelConfig, compose_hooks


# ---------------------------------------------------------------------------
# Dummy concrete implementations
# ---------------------------------------------------------------------------


@dataclass
class DummyModelConfig(ModelConfig):
    builder: ClassVar[str] = ""  # set dynamically below
    value: int = 42
    name: str = "test"


class DummyModelBuilder(ModelBuilder):
    def build_model(self, pg_collection, pre_process=None, post_process=None, vp_stage=None):
        pass

    def build_distributed_models(self, pg_collection, **kwargs):
        return []


# Set after both classes exist so __module__ is always the actual runtime path.
# get_builder_cls() calls importlib.import_module on the module portion of this
# string, so it must match what Python sees for this test module.
DummyModelConfig.builder = f"{DummyModelBuilder.__module__}.DummyModelBuilder"


# A plain nested dataclass (not a ModelConfig subclass) for testing nested
# serialization without depending on any real model classes.
@dataclass
class DummySubConfig:
    x: int = 1
    y: str = "sub"


def _dummy_callable() -> None:
    """Placeholder callable used as a field default in DummyNestedModelConfig."""


@dataclass
class DummyNestedModelConfig(ModelConfig):
    """ModelConfig subclass with a nested dataclass field and a callable field.

    Used to test nested serialization and callable-field exclusion without
    depending on MambaModelConfig or TransformerConfig.
    """

    builder: ClassVar[str] = ""  # set dynamically below
    sub: DummySubConfig = field(default_factory=DummySubConfig)
    fn_field: Callable = _dummy_callable
    extra: int = 0


DummyNestedModelConfig.builder = f"{DummyModelBuilder.__module__}.DummyModelBuilder"


# =============================================================================
# Section 1 — TestModelConfigDefaults
# =============================================================================


class TestModelConfigDefaults:
    """Base class field defaults and ClassVar on a concrete subclass."""

    def test_base_field_defaults(self):
        cfg = DummyModelConfig()
        assert cfg.restore_modelopt_state is False
        assert cfg.hf_model_id is None
        assert cfg.generation_config is None
        assert cfg.pre_wrap_hooks == []
        assert cfg.post_wrap_hooks == []

    def test_custom_fields_stored(self):
        hook1 = Mock()
        hook2 = Mock()
        cfg = DummyModelConfig(
            value=99, name="hello", pre_wrap_hooks=[hook1], post_wrap_hooks=[hook2]
        )
        assert cfg.value == 99
        assert cfg.name == "hello"
        assert cfg.pre_wrap_hooks[0] == hook1
        assert cfg.post_wrap_hooks[0] == hook2

    def test_builder_classvar_accessible(self):
        assert (
            DummyModelConfig.builder == "tests.unit_tests.models.common.test_base.DummyModelBuilder"
        )


# =============================================================================
# Section 2 — TestModelConfigGetBuilderCls
# =============================================================================


class TestModelConfigGetBuilderCls:
    """get_builder_cls() dynamically imports and returns the builder class named by the ClassVar."""

    def test_returns_correct_type(self):
        cfg = DummyModelConfig()
        result = cfg.get_builder_cls()
        assert result is DummyModelBuilder

    def test_return_is_class_not_instance(self):
        cfg = DummyModelConfig()
        result = cfg.get_builder_cls()
        assert isinstance(result, type)


# =============================================================================
# Section 3 — TestModelConfigToDict
# =============================================================================


class TestModelConfigToDict:
    """as_dict() serializes all non-callable, non-private dataclass fields, including nested dataclasses."""

    def test_target_key_present(self):
        cfg = DummyModelConfig()
        result = cfg.as_dict()
        assert result["_target_"] == f"{DummyModelConfig.__module__}.DummyModelConfig"

    def test_builder_key_present(self):
        cfg = DummyModelConfig()
        result = cfg.as_dict()
        assert result["_builder_"] == DummyModelConfig.builder

    def test_own_fields_serialized(self):
        cfg = DummyModelConfig(value=7, name="world")
        result = cfg.as_dict()
        assert result["value"] == 7
        assert result["name"] == "world"

    def test_base_class_fields_serialized(self):
        cfg = DummyModelConfig()
        result = cfg.as_dict()
        assert "restore_modelopt_state" in result
        assert "hf_model_id" in result
        assert "generation_config" in result

    def test_hook_lists_excluded(self):
        cfg = DummyModelConfig()
        result = cfg.as_dict()
        assert "pre_wrap_hooks" not in result
        assert "post_wrap_hooks" not in result

    def test_callable_field_excluded(self):
        cfg = DummyNestedModelConfig()
        result = cfg.as_dict()
        assert "fn_field" not in result

    def test_nested_dataclass_serialized_recursively(self):
        cfg = DummyNestedModelConfig()
        result = cfg.as_dict()
        assert isinstance(result["sub"], dict)
        assert "_target_" in result["sub"]
        assert result["sub"]["x"] == 1
        assert result["sub"]["y"] == "sub"


# =============================================================================
# Section 4 — TestModelConfigFromDict
# =============================================================================


class TestModelConfigFromDict:
    """from_dict() reconstructs configs from serialized dicts, handles nested dataclasses, and is robust to unknown keys."""

    def _dummy_dict(self, **overrides):
        d = {
            "_target_": f"{DummyModelConfig.__module__}.DummyModelConfig",
            "_builder_": DummyModelConfig.builder,
            "value": 10,
            "name": "from_dict_test",
        }
        d.update(overrides)
        return d

    def test_reconstructs_flat_config(self):
        d = self._dummy_dict(value=55, name="reconstructed")
        cfg = ModelConfig.from_dict(d)
        assert isinstance(cfg, DummyModelConfig)
        assert cfg.value == 55
        assert cfg.name == "reconstructed"

    def test_builder_restored(self):
        d = self._dummy_dict(_builder_="fake.builder.string")
        cfg = ModelConfig.from_dict(d)
        assert cfg.builder == "fake.builder.string"

    def test_ignores_unknown_fields(self):
        d = self._dummy_dict(unknown_field="should_be_ignored", another_unknown=123)
        cfg = ModelConfig.from_dict(d)
        assert isinstance(cfg, DummyModelConfig)
        assert cfg.value == 10
        assert cfg.name == "from_dict_test"
        assert not hasattr(cfg, "unknown_field")
        assert not hasattr(cfg, "another_unknown")

    def test_raises_if_target_missing(self):
        d = {"_builder_": DummyModelConfig.builder, "value": 1}
        with pytest.raises(ValueError):
            ModelConfig.from_dict(d)

    def test_round_trip_flat(self):
        original = DummyModelConfig(value=21, name="round_trip")
        cfg = ModelConfig.from_dict(original.as_dict())
        assert isinstance(cfg, DummyModelConfig)
        assert cfg.value == original.value
        assert cfg.name == original.name

    def test_round_trip_with_nested_dataclass(self):
        original = DummyNestedModelConfig(sub=DummySubConfig(x=7, y="nested"), extra=99)
        cfg = ModelConfig.from_dict(original.as_dict())
        assert isinstance(cfg, DummyNestedModelConfig)
        assert cfg.extra == 99
        assert isinstance(cfg.sub, DummySubConfig)
        assert cfg.sub.x == 7
        assert cfg.sub.y == "nested"


# =============================================================================
# Section 6 — TestComposeHooks
# =============================================================================


class TestComposeHooks:
    """compose_hooks() folds a list of hook functions into a single function, threading output through each in order."""

    def test_empty_list_is_identity(self):
        composed = compose_hooks([])
        sentinel = object()
        assert composed(sentinel) is sentinel

    def test_single_hook_equivalent(self):
        h = Mock(return_value="result_x")
        composed = compose_hooks([h])
        result = composed("x")
        h.assert_called_once_with("x")
        assert result == "result_x"

    def test_multiple_hooks_chained(self):
        h1 = Mock(return_value="after_h1")
        h2 = Mock(return_value="after_h2")
        composed = compose_hooks([h1, h2])
        result = composed("start")
        h1.assert_called_once_with("start")
        h2.assert_called_once_with("after_h1")
        assert result == "after_h2"

    def test_hooks_called_in_list_order(self):
        call_log = []

        def h1(x):
            call_log.append("h1")
            return x

        def h2(x):
            call_log.append("h2")
            return x

        def h3(x):
            call_log.append("h3")
            return x

        composed = compose_hooks([h1, h2, h3])
        composed("anything")
        assert call_log == ["h1", "h2", "h3"]

    def test_intermediate_values_threaded(self):
        def double(x):
            return x * 2

        def add_ten(x):
            return x + 10

        composed = compose_hooks([double, add_ten])
        assert composed(5) == 20  # 5 * 2 = 10, 10 + 10 = 20
