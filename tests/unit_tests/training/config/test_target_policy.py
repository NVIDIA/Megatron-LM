# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Behavioral checks for the shared configuration target policy."""

import functools
from dataclasses import dataclass
from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf

from megatron.training.config import instantiate_utils as iu
from megatron.training.config.utils import sanitize_dataclass_config
from megatron.training.models.base import ModelConfig


@pytest.fixture(autouse=True)
def isolated_allowlist(monkeypatch):
    """Keep application registrations local to each test."""
    monkeypatch.setattr(iu, "target_allowlist", iu.TargetAllowlist())
    iu.target_allowlist.add_prefix(__name__ + ".")


@pytest.mark.parametrize(
    "target",
    [
        "torch.load",
        "torch.serialization.load",
        "torch.hub.load_state_dict_from_url",
        "torch.utils.model_zoo.load_url",
        "torch.package.PackageImporter",
        "torch.package.PackageImporter.load_pickle",
        "torch.package.package_importer.PackageImporter.load_pickle",
        "torch.save",
        "torch.set_default_dtype",
        "torch.nn.Module.load_state_dict",
        "megatron.training.config.instantiate_utils.target_allowlist.disable",
    ],
)
def test_reject_before_import(monkeypatch, target):
    """Nested, deferred and direct targets must fail before resolution or invocation."""
    locate = Mock(side_effect=AssertionError("target must not be imported"))
    monkeypatch.setattr(iu, "_locate", locate)
    for flags in ({}, {"_call_": False}, {"_partial_": True}):
        config = {"model": [{"_target_": target, **flags}]}
        with pytest.raises(iu.InstantiationException):
            iu.instantiate(config)
    locate.assert_not_called()


@pytest.mark.parametrize("entry", ["instantiate", "node", "sanitize", "model"])
def test_canonical_alias_cannot_execute(monkeypatch, entry):
    """An alias in a trusted package cannot hide the loader's canonical identity."""
    calls = []

    def loader(**kwargs):
        calls.append(kwargs)

    loader.__module__ = "torch.package.package_importer"
    loader.__qualname__ = "PackageImporter.load_pickle"
    from megatron.training.config import training_config

    monkeypatch.setattr(training_config, "config_alias", loader, raising=False)
    target = "megatron.training.config.training_config.config_alias"
    value = {"_target_": target}
    with pytest.raises(iu.InstantiationException, match="not allowed"):
        if entry == "instantiate":
            iu.instantiate({"tokenizer": value})
        elif entry == "node":
            iu.instantiate_node(OmegaConf.create({"model": [value]}))
        elif entry == "sanitize":
            sanitize_dataclass_config({"model": value})
        else:
            ModelConfig.from_dict({"_builder_": "unused", **value})
    assert calls == []


def test_unknown_torch_target_requires_exact_registration():
    """A package prefix alone does not opt into arbitrary torch functions."""
    iu.target_allowlist.add_prefix("torch.")
    with pytest.raises(iu.InstantiationException, match="supported torch configuration"):
        iu.instantiate({"_target_": "torch.get_default_dtype"})
    iu.target_allowlist.add_exact("torch.get_default_dtype")
    assert iu.instantiate({"_target_": "torch.get_default_dtype"}) is torch.get_default_dtype()


def test_known_loader_cannot_be_registered():
    """Exact registrations cannot override known prohibited targets."""
    iu.target_allowlist.add_exact("torch.load")
    with pytest.raises(iu.InstantiationException, match="not allowed"):
        iu.instantiate({"_target_": "torch.load", "_call_": False})


@dataclass
class NestedConfig:
    """Small legitimate nested configuration."""

    activation: object
    dtype: object


@dataclass
class ExampleModelConfig(ModelConfig):
    """Model metadata with nested configuration values."""

    nested: object = None
    builder = "megatron.training.models.base.ModelBuilder"


def test_valid_nested_model_roundtrip():
    """Function references, partials and builder metadata retain their behavior."""
    original = ExampleModelConfig(
        nested=NestedConfig(
            functools.partial(torch.nn.functional.gelu, approximate="tanh"), torch.bfloat16
        )
    )
    # as_dict encodes callables; the YAML representer persists dtype objects.
    import yaml

    from megatron.training.config.yaml_utils import safe_yaml_representers

    with safe_yaml_representers():
        serialized = yaml.safe_load(yaml.safe_dump(original.as_dict()))
    restored = ModelConfig.from_dict(serialized)
    assert type(restored) is ExampleModelConfig
    assert restored.builder == original.builder
    assert type(restored.nested) is NestedConfig
    assert restored.nested.dtype is torch.bfloat16
    x = torch.tensor([-2.0, 0.0, 1.0])
    torch.testing.assert_close(
        restored.nested.activation(x), original.nested.activation(x), rtol=0, atol=0
    )


def test_valid_module_and_function_references():
    """Constructors and non-callable values are distinct supported operations."""
    result = iu.instantiate(
        {
            "module": {"_target_": "torch.nn.Identity"},
            "activation": {"_target_": "torch.nn.functional.gelu", "_call_": False},
            "dtype": {"_target_": "torch.float32", "_call_": False},
            "partial": {"_target_": "torch.nn.functional.silu", "_partial_": True},
        }
    )
    assert isinstance(result["module"], torch.nn.Identity)
    assert result["activation"] is torch.nn.functional.gelu
    assert result["dtype"] is torch.float32
    x = torch.tensor([-1.0, 0.0, 2.0])
    assert result["module"](x) is x
    torch.testing.assert_close(result["partial"](x), torch.nn.functional.silu(x), rtol=0, atol=0)


@pytest.mark.parametrize(
    "target",
    [
        "megatron.training.config.instantiate_utils._locate",
        "megatron.training.config.instantiate_utils._call_target",
        "megatron.training.config.instantiate_utils.TargetAllowlist.disable",
        "megatron.training.config.instantiate_utils.target_allowlist",
    ],
)
def test_policy_implementation_is_not_a_config_target(target):
    with pytest.raises(iu.InstantiationException, match="not allowed"):
        iu.instantiate({"_target_": target, "_call_": False})
    assert iu.target_allowlist.enabled


def test_explicit_torch_function_registration():
    iu.target_allowlist.add_exact("torch.nn.functional.normalize")
    assert (
        iu.instantiate({"_target_": "torch.nn.functional.normalize", "_call_": False})
        is torch.nn.functional.normalize
    )


@pytest.mark.parametrize("builder", ["torch.load", "torch.nn.Identity"])
def test_builder_must_resolve_to_builder_class(builder):
    config = ExampleModelConfig()
    config.builder = builder
    with pytest.raises((iu.InstantiationException, TypeError)):
        config.get_builder_cls()


def test_nested_callable_field_keeps_its_path():
    config = {
        "_target_": __name__ + ".ExampleModelConfig",
        "_builder_": ExampleModelConfig.builder,
        "nested": {"preprocess_fn": [{"_target_": "torch.nn.functional.gelu", "_call_": False}]},
    }
    with pytest.raises(iu.InstantiationException, match="callable config field"):
        ModelConfig.from_dict(config)


@pytest.mark.parametrize("entry", ["dict", "yaml", "checkpoint"])
def test_configuration_readers_share_policy(tmp_path, entry):
    """All readers reject nested targets before compatibility hooks or construction."""
    import yaml

    from megatron.training.config.container import ConfigContainerBase
    from megatron.training.utils.checkpoint_utils import read_run_config

    config = {
        "_target_": "megatron.training.config.container.ConfigContainerBase",
        "value": {"_target_": "torch.package.PackageImporter.load_pickle"},
    }
    filename = tmp_path / "run_config.yaml"
    filename.write_text(yaml.safe_dump(config))
    with pytest.raises(iu.InstantiationException, match="not allowed"):
        if entry == "dict":
            ConfigContainerBase.from_dict(config)
        elif entry == "yaml":
            ConfigContainerBase.from_yaml(str(filename))
        else:
            read_run_config(str(filename))


@pytest.mark.parametrize("helper", ["import_module", "get_module", "build_module"])
def test_spec_dispatch_helpers_are_not_config_targets(helper):
    """Generic deferred import helpers cannot be selected by serialized input."""
    with pytest.raises(iu.InstantiationException):
        iu.instantiate(
            {"_target_": f"megatron.core.transformer.spec_utils.{helper}", "_call_": False}
        )


@pytest.mark.parametrize(
    "entry", ["instantiate", "partial", "positional", "model", "wrapped_partial"]
)
def test_deferred_spec_import_obeys_policy(entry):
    """A ModuleSpec cannot retain a prohibited import for later execution."""
    value = {
        "_target_": "megatron.core.transformer.spec_utils.ModuleSpec",
        "module": ["torch.package", "PackageImporter"],
    }
    with pytest.raises(iu.InstantiationException):
        if entry == "model":
            ModelConfig.from_dict(
                {
                    "_target_": __name__ + ".ExampleModelConfig",
                    "_builder_": "unused",
                    "nested": value,
                }
            )
        elif entry == "wrapped_partial":
            iu.instantiate(
                {
                    "_target_": "functools.partial",
                    "_args_": [{"_target_": value["_target_"], "_call_": False}],
                    "module": value["module"],
                }
            )
        else:
            if entry == "partial":
                value["_partial_"] = True
            if entry == "positional":
                value["_args_"] = [value.pop("module")]
            iu.instantiate(value)


def test_valid_deferred_spec_roundtrip():
    """Retain supported deferred imports and their numerical behavior."""
    from megatron.core.transformer.spec_utils import ModuleSpec, build_module

    original = ExampleModelConfig(nested=ModuleSpec(("torch.nn", "Identity")))
    restored = ModelConfig.from_dict(original.as_dict())
    assert tuple(restored.nested.module) == original.nested.module
    x = torch.tensor([1.0, -2.0])
    assert build_module(restored.nested)(x) is x


@pytest.mark.parametrize("inner_module", [None, ["torch.nn", "Identity"]])
def test_nested_partial_spec_cannot_override_validated_module(inner_module):
    """Apply partial keyword overrides before validating deferred module paths."""
    inner = {"_target_": "megatron.core.transformer.spec_utils.ModuleSpec", "_partial_": True}
    if inner_module is not None:
        inner["module"] = inner_module
    with pytest.raises(iu.InstantiationException):
        iu.instantiate(
            {
                "_target_": "functools.partial",
                "_args_": [inner],
                "module": ["torch.package", "PackageImporter"],
            }
        )
