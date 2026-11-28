# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Focused checks for configuration target validation."""

from unittest.mock import Mock

import pytest
import torch

from megatron.training.config import instantiate_utils as iu
from megatron.training.config import training_config


@pytest.fixture(autouse=True)
def isolated_allowlist(monkeypatch):
    monkeypatch.setattr(iu, "target_allowlist", iu.TargetAllowlist())


@pytest.mark.parametrize(
    "target",
    [
        "torch.load",
        "torch.hub.load_state_dict_from_url",
        "torch.utils.model_zoo.load_url",
        "torch.package.PackageImporter",
        "torch.package.PackageImporter.load_pickle",
    ],
)
def test_reject_before_import(monkeypatch, target):
    locate = Mock(side_effect=AssertionError("target must not be imported"))
    monkeypatch.setattr(iu, "_locate", locate)
    with pytest.raises(iu.InstantiationException):
        iu.instantiate({"model": {"_target_": target}})
    locate.assert_not_called()


def test_canonical_alias_is_rejected(monkeypatch):
    def loader():
        pytest.fail("loader must not be invoked")

    loader.__module__ = "torch.package.package_importer"
    loader.__qualname__ = "PackageImporter.load_pickle"
    monkeypatch.setattr(training_config, "config_alias", loader, raising=False)
    with pytest.raises(iu.InstantiationException):
        iu.instantiate({"_target_": "megatron.training.config.training_config.config_alias"})


def test_supported_configuration_values():
    result = iu.instantiate(
        {
            "module": {"_target_": "torch.nn.Identity"},
            "activation": {"_target_": "torch.nn.functional.gelu", "_call_": False},
            "dtype": {"_target_": "torch.float32", "_call_": False},
        }
    )
    assert isinstance(result["module"], torch.nn.Identity)
    assert result["activation"] is torch.nn.functional.gelu
    assert result["dtype"] is torch.float32


@pytest.mark.parametrize("model_path", [False, True])
def test_deferred_module_spec_validation(model_path):
    from megatron.core.transformer.spec_utils import ModuleSpec, build_module
    from megatron.training.models.base import ModelConfig

    spec_target = "megatron.core.transformer.spec_utils.ModuleSpec"

    def restore(spec):
        if model_path:
            return ModelConfig.from_dict(
                {
                    "_target_": "megatron.training.models.base.ModelConfig",
                    "_builder_": "megatron.training.models.base.ModelBuilder",
                    "extra_checkpoint_metadata": {"spec": spec},
                }
            ).extra_checkpoint_metadata["spec"]
        return iu.instantiate(spec)

    prohibited = ["torch.package", "PackageImporter"]
    for fields in (
        {"module": prohibited},
        {"module": prohibited, "_partial_": True},
        {
            "module": ["megatron.core.transformer.spec_utils", "ModuleSpec"],
            "params": {"module": prohibited},
        },
        {"module": {"torch.nn": "Identity"}},
    ):
        with pytest.raises(iu.InstantiationException):
            restore({"_target_": spec_target, **fields})

    spec = restore({"_target_": spec_target, "module": ["torch.nn", "Identity"]})
    assert isinstance(build_module(spec), torch.nn.Identity)
    factory = restore({"_target_": spec_target, "_partial_": True})
    assert factory.func is ModuleSpec
    assert isinstance(build_module(factory(module=torch.nn.Identity)), torch.nn.Identity)
