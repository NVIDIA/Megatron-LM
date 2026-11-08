# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, ClassVar
from unittest.mock import Mock

import pytest
import torch

from megatron.core.models.hybrid import MTPSplit, PipelineSplit
from megatron.core.models.hybrid.hybrid_layer_allocation import parse_hybrid_layer_config_list
from megatron.core.transformer.attention_layer_config import AttentionLayerConfig
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.config.instantiate_utils import (
    InstantiationException,
    InstantiationMode,
    instantiate,
)
from megatron.training.models.base import ModelBuilder, ModelConfig, compose_hooks
from megatron.training.models.hybrid import HybridModelConfig
from megatron.training.utils.checkpoint_utils import read_run_config


@pytest.fixture(autouse=True)
def _disable_allowlist():
    """Allow local test-module targets when from_dict resolves encoded callables."""
    from megatron.training.config.instantiate_utils import target_allowlist

    target_allowlist.disable()
    yield
    target_allowlist.enable()


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


def _dummy_callable(value=None, *, sub=None) -> None:
    """Placeholder callable used as a field default in DummyNestedModelConfig."""


class _CallbackOwner:
    """Owns a bound method, like the optimizer whose scale_loss setup attaches."""

    def scale_loss(self, loss):
        return loss


@dataclass
class DummyNestedModelConfig(ModelConfig):
    """ModelConfig subclass with nested-dataclass, callable, and TransformerConfig fields."""

    builder: ClassVar[str] = ""  # set dynamically below
    sub: DummySubConfig = field(default_factory=DummySubConfig)
    fn_field: Callable = _dummy_callable
    extra: int = 0
    transformer: TransformerConfig | None = None


DummyNestedModelConfig.builder = f"{DummyModelBuilder.__module__}.DummyModelBuilder"


# =============================================================================
# Section 1 — TestModelConfigDefaults
# =============================================================================


class TestModelConfigDefaults:
    """Base class field defaults and ClassVar on a concrete subclass."""

    def test_base_field_defaults(self):
        cfg = DummyModelConfig()
        assert cfg.restore_modelopt_state is False
        assert cfg.extra_checkpoint_metadata is None
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
        assert DummyModelConfig.builder == f"{DummyModelBuilder.__module__}.DummyModelBuilder"


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
    """as_dict() serializes non-private fields, nested dataclasses, and importable callables."""

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
        assert "extra_checkpoint_metadata" in result

    def test_runtime_fields_excluded(self):
        """Hooks and callbacks that each run attaches are never serialized, even when encodable."""
        runtime_fields = [
            "timers",
            "finalize_model_grads_func",
            "grad_scale_func",
            "moe_grad_scale_func",
            "mtp_grad_scale_func",
            "no_sync_func",
            "grad_sync_func",
            "param_sync_func",
        ]
        transformer = TransformerConfig(num_layers=2, hidden_size=128, num_attention_heads=4)
        for name in runtime_fields:
            setattr(transformer, name, _dummy_callable)
        transformer.no_sync_func = [_dummy_callable, _dummy_callable]  # one per VPP chunk
        cfg = DummyNestedModelConfig(
            transformer=transformer,
            pre_wrap_hooks=[_dummy_callable],
            post_wrap_hooks=[_dummy_callable],
        )

        result = cfg.as_dict()
        assert "pre_wrap_hooks" not in result and "post_wrap_hooks" not in result
        assert set(runtime_fields).isdisjoint(result["transformer"])
        # Setup asserts no_sync_func is unset before attaching DDP's under overlap_grad_reduce.
        assert ModelConfig.from_dict(result).transformer.no_sync_func is None

    def test_callable_field_encoded(self):
        """Module-level callables serialize as _target_/_call_ dicts (they used to be dropped)."""
        cfg = DummyNestedModelConfig()
        result = cfg.as_dict()
        assert result["fn_field"] == {
            "_target_": f"{_dummy_callable.__module__}.{_dummy_callable.__qualname__}",
            "_call_": False,
        }

    @pytest.mark.parametrize(
        "fn",
        [lambda: None, _CallbackOwner().scale_loss, torch.relu],
        ids=["lambda", "bound_method", "torch_builtin"],
    )
    def test_unencodable_callable_skipped(self, fn):
        """Callables whose import path does not resolve back to them are dropped from the dict."""
        assert "fn_field" not in DummyNestedModelConfig(fn_field=fn).as_dict()

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

    def test_raises_if_builder_missing(self):
        d = self._dummy_dict()
        del d["_builder_"]
        with pytest.raises(KeyError, match="_builder_"):
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
        # Encoded callable fields resolve back to the original function.
        assert cfg.fn_field is _dummy_callable

    def test_round_trip_partial_with_nested_dataclasses(self):
        from functools import partial

        import yaml

        from megatron.training.config.yaml_utils import safe_yaml_representers

        original = DummyNestedModelConfig(
            fn_field=partial(_dummy_callable, DummySubConfig(x=7), sub=DummySubConfig(x=9))
        )
        with safe_yaml_representers():
            saved = yaml.safe_load(yaml.safe_dump(original.as_dict()))
        restored = ModelConfig.from_dict(saved)
        assert restored.fn_field.func is _dummy_callable
        assert restored.fn_field.args == (DummySubConfig(x=7),)
        assert restored.fn_field.keywords == {'sub': DummySubConfig(x=9)}

    @pytest.mark.parametrize("container_type", [list, tuple])
    @pytest.mark.parametrize("saved_yaml", [False, True])
    @pytest.mark.parametrize("mode", [None, InstantiationMode.STRICT, InstantiationMode.LENIENT])
    def test_hybrid_config_list_round_trip_preserves_shared_mtp_template(
        self, run_config, tmp_path, container_type, saved_yaml, mode
    ):
        transformer = TransformerConfig(num_layers=2, hidden_size=128, num_attention_heads=4)
        first = AttentionLayerConfig.from_config(transformer)
        first.attention_dropout = 0.25
        second = AttentionLayerConfig.from_config(first)
        first.no_sync_func = [_dummy_callable]
        source_layers = container_type(
            [first, PipelineSplit, second, MTPSplit, first, MTPSplit, first]
        )
        original = HybridModelConfig(
            transformer=transformer, hybrid_layer_config_list=source_layers, vocab_size=128
        )
        run_config.model = original
        if saved_yaml:
            yaml_path = tmp_path / "hybrid_config.yaml"
            run_config.to_yaml(yaml_path)
            saved = read_run_config(str(yaml_path))
        else:
            saved = run_config.to_dict()
        unchanged = deepcopy(saved)

        if mode is None:
            restored = ModelConfig.from_dict(saved["model"])
        elif saved_yaml:
            restored = type(run_config).from_yaml(str(yaml_path), mode=mode).model
        else:
            restored = type(run_config).from_dict(saved, mode=mode).model
        layers = restored.hybrid_layer_config_list
        assert isinstance(restored, HybridModelConfig)
        assert isinstance(layers[0], AttentionLayerConfig)
        assert layers[0].attention_dropout == layers[2].attention_dropout == 0.25
        assert layers[0] is layers[4] is layers[6]
        assert layers[0] is not layers[2]
        assert layers[0] is not first
        assert layers[0].no_sync_func is None
        assert layers[1] is PipelineSplit
        assert layers[3] is layers[5] is MTPSplit
        parsed = parse_hybrid_layer_config_list(layers, expected_num_layers=2)
        assert parsed.mtp_num_depths == 2
        assert parsed.mtp_layer_config_list == (layers[0],)
        assert saved == unchanged
        assert original.hybrid_layer_config_list is source_layers
        assert first.no_sync_func == [_dummy_callable]
        assert first.mtp_num_layers is None

    @pytest.mark.parametrize("mode", [None, InstantiationMode.STRICT, InstantiationMode.LENIENT])
    def test_nested_models_have_independent_shared_config_ids(self, mode):
        left_sub = DummySubConfig(x=7)
        right_sub = DummySubConfig(x=7)
        left = DummyNestedModelConfig(sub=left_sub, extra_checkpoint_metadata={"shared": left_sub})
        right = DummyNestedModelConfig(
            sub=right_sub, extra_checkpoint_metadata={"shared": right_sub}
        )
        original = DummyModelConfig(extra_checkpoint_metadata={"left": left, "right": right})
        saved = original.as_dict()
        # Both nested serializers start their own ID sequence.
        metadata = saved["extra_checkpoint_metadata"]
        assert metadata["left"]["sub"]["_config_id_"] == metadata["right"]["sub"]["_config_id_"]

        if mode is None:
            restored = ModelConfig.from_dict(saved)
        else:
            restored = instantiate(saved, mode=mode)
        restored = restored.extra_checkpoint_metadata
        assert restored["left"].sub is restored["left"].extra_checkpoint_metadata["shared"]
        assert restored["right"].sub is restored["right"].extra_checkpoint_metadata["shared"]
        assert restored["left"].sub is not restored["right"].sub
        assert restored["left"].sub == restored["right"].sub == DummySubConfig(x=7)

    @pytest.mark.parametrize("mode", [None, InstantiationMode.STRICT, InstantiationMode.LENIENT])
    def test_conflicting_shared_config_definitions_are_rejected(self, mode):
        shared = DummySubConfig(x=7)
        original = DummyNestedModelConfig(sub=shared, extra_checkpoint_metadata={"shared": shared})
        saved = original.as_dict()
        saved["sub"]["x"] = 9

        exception = ValueError if mode is None else InstantiationException
        with pytest.raises(exception, match="Conflicting definitions for shared config"):
            if mode is None:
                ModelConfig.from_dict(saved)
            else:
                instantiate(saved, mode=mode)


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


def test_mimo_builder_config_round_trip():
    from examples.mimo.training.builder import MimoModelBuilder
    from megatron.training.models import MimoBuildConfig

    config = MimoBuildConfig(
        transformer=TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4),
        seq_length=16,
        vocab_size=128,
        mimo_llm_world_size=2,
        pre_wrap_hooks=[lambda models: models],
        post_wrap_hooks=[lambda models: models],
        _topology=object(),
    )
    serialized = config.as_dict()
    restored = ModelConfig.from_dict(serialized)
    assert isinstance(restored, MimoBuildConfig)
    assert restored.get_builder_cls() is MimoModelBuilder
    assert restored.mimo_llm_world_size == 2
    assert restored.pre_wrap_hooks == []
    assert restored.post_wrap_hooks == []
    assert restored._topology is None
    assert restored.as_dict() == serialized


@pytest.mark.parametrize("spec_name", ["hybrid_stack_spec", "gated_delta_product_stack_spec"])
@pytest.mark.parametrize("yaml_roundtrip", [False, True])
@pytest.mark.parametrize("customized", [False, True])
def test_hybrid_model_spec_round_trip(spec_name, yaml_roundtrip, customized):
    from copy import deepcopy

    import yaml

    from megatron.core.models.hybrid import hybrid_layer_specs
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear
    from megatron.core.transformer.spec_utils import ModuleSpec
    from megatron.training.config.yaml_utils import safe_yaml_representers
    from megatron.training.models import HybridModelConfig, ModelConfig

    spec = deepcopy(getattr(hybrid_layer_specs, spec_name))
    if customized:
        # Old YAML conversion dropped these fields and could select the default spec.
        spec.submodules.mlp_layer.submodules.mlp.keywords["submodules"].linear_fc1 = (
            ColumnParallelLinear
        )
    model_config = HybridModelConfig(
        transformer=TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4),
        hybrid_stack_spec=spec,
        hybrid_layer_pattern="MM",
        seq_length=16,
        vocab_size=128,
    )
    model_dict = model_config.as_dict()
    if yaml_roundtrip:
        with safe_yaml_representers():
            model_dict = yaml.safe_load(yaml.safe_dump(model_dict))
    restored = ModelConfig.from_dict(model_dict)
    assert isinstance(
        restored.hybrid_stack_spec.submodules.mtp_block_spec.submodules.layer_specs[0], ModuleSpec
    )
    assert (
        restored.hybrid_stack_spec.submodules.mlp_layer.submodules.mlp.keywords['submodules']
        == spec.submodules.mlp_layer.submodules.mlp.keywords['submodules']
    )
