# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for ``LigerSpecProvider`` — the Liger-Kernel BackendSpecProvider."""

import pytest

pytest.importorskip("liger_kernel.megatron")

from liger_kernel.megatron import LigerMegatronRMSNorm

from megatron.core.extensions import liger_kernel_spec_provider as provider_module
from megatron.core.extensions.liger_kernel_spec_provider import LigerSpecProvider


def test_layer_norm_slot_returns_liger_rmsnorm():
    """rms_norm=True must route to LigerMegatronRMSNorm."""
    provider = LigerSpecProvider()
    assert provider.layer_norm(rms_norm=True) is LigerMegatronRMSNorm


def test_layer_norm_slot_falls_back_for_non_rmsnorm():
    """rms_norm=False must fall through to the LocalSpecProvider's LayerNorm builder."""
    from megatron.core.models.backends import LocalSpecProvider

    provider = LigerSpecProvider()
    expected = LocalSpecProvider().layer_norm(rms_norm=False)
    assert provider.layer_norm(rms_norm=False) is expected


def test_missing_package_raises(monkeypatch):
    """Instantiation must raise ImportError with a clear message when liger-kernel is absent."""
    monkeypatch.setattr(provider_module, "HAVE_LIGER", False)
    monkeypatch.setattr(provider_module, "LigerMegatronRMSNorm", None)

    with pytest.raises(ImportError, match="Liger-Kernel is required"):
        LigerSpecProvider()


def test_use_liger_flag_wires_through_gpt_layer_specs():
    """get_gpt_layer_local_submodules(use_liger=True) must produce a spec whose
    norm slots resolve to LigerMegatronRMSNorm."""
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules

    submodules = get_gpt_layer_local_submodules(
        normalization="RMSNorm", use_liger=True
    )
    assert submodules.input_layernorm is LigerMegatronRMSNorm
    assert submodules.pre_mlp_layernorm is LigerMegatronRMSNorm


def test_use_liger_is_mutually_exclusive_with_use_kitchen():
    """use_liger=True and use_kitchen=True must not be combined."""
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules

    with pytest.raises(AssertionError, match="mutually exclusive"):
        get_gpt_layer_local_submodules(
            normalization="RMSNorm", use_kitchen=True, use_liger=True
        )
