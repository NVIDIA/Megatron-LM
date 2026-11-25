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
