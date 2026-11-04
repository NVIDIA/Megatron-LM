# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Liger normalization selection and tensor contracts."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from megatron.core.extensions.liger_kernel_spec_provider import LigerSpecProvider
from megatron.core.models.backends import LocalSpecProvider, get_backend, get_backend_from_config
from megatron.core.ops.norm.torch_norm import WrappedTorchNorm


def _norm_type():
    pytest.importorskip("liger_kernel.megatron")
    from megatron.core.ops.norm.liger_norm import LigerRMSNorm

    return LigerRMSNorm


def _config(**overrides):
    options = dict(
        normalization="RMSNorm",
        layernorm_zero_centered_gamma=False,
        sequence_parallel=False,
        persist_layer_norm=False,
        memory_efficient_layer_norm=False,
    )
    return SimpleNamespace(**(options | overrides))


class TestLigerNormSelection:
    """Only an explicitly selected RMSNorm imports Liger."""

    def test_unselected_paths_do_not_import_liger(self):
        with patch("megatron.core.ops._backends.import_module") as load:
            assert LocalSpecProvider().layer_norm(rms_norm=True) is WrappedTorchNorm
            assert LigerSpecProvider().layer_norm() is LocalSpecProvider().layer_norm()
        load.assert_not_called()

    @pytest.mark.parametrize("for_qk", [False, True])
    @pytest.mark.parametrize("has_residual", [False, True])
    def test_rms_norm_slots(self, for_qk, has_residual):
        norm = _norm_type()
        assert (
            LigerSpecProvider().layer_norm(rms_norm=True, for_qk=for_qk, has_residual=has_residual)
            is norm
        )

    def test_selection_does_not_mutate_the_local_provider(self):
        _norm_type()
        local = LocalSpecProvider()
        before = local.layer_norm()
        LigerSpecProvider().layer_norm(rms_norm=True)
        assert local.layer_norm() is before
        assert local.layer_norm(rms_norm=True) is WrappedTorchNorm

    @pytest.mark.parametrize(
        "failure",
        [
            ModuleNotFoundError("No module named 'liger_kernel'", name="liger_kernel"),
            RuntimeError("incompatible native extension"),
        ],
    )
    def test_missing_or_broken_selected_library(self, failure):
        with patch("megatron.core.ops._backends.import_module", side_effect=failure):
            with pytest.raises(ImportError, match="Liger RMSNorm requires") as error:
                LigerSpecProvider().layer_norm(rms_norm=True)
        assert error.value.__cause__ is failure

    def test_missing_export(self):
        with patch("megatron.core.ops._backends.import_module", return_value=SimpleNamespace()):
            with pytest.raises(ImportError, match="LigerMegatronRMSNorm"):
                LigerSpecProvider().layer_norm(rms_norm=True)

    def test_factory_and_newer_slots(self):
        provider = get_backend_from_config(
            SimpleNamespace(transformer_impl="local", use_liger=True)
        )
        assert isinstance(provider, LigerSpecProvider)
        local = LocalSpecProvider()
        for slot in (
            "column_parallel_linear",
            "row_parallel_linear",
            "core_attention",
            "fuse_layernorm_and_linear",
            "column_parallel_layer_norm_linear",
            "activation_func",
            "mlp_module",
            "moe_router",
            "vocab_parallel_cross_entropy",
        ):
            assert getattr(provider, slot)() is getattr(local, slot)(), slot
        assert callable(provider.grouped_mlp_modules(moe_use_grouped_gemm=False))
        with pytest.raises(NotImplementedError, match="no non-parallel linear"):
            provider.linear()

    @pytest.mark.parametrize("backend", ["transformer_engine", "inference_optimized"])
    def test_unsupported_provider_fails_before_importing_liger(self, backend):
        with patch("megatron.core.ops._backends.import_module") as load:
            with pytest.raises(ValueError, match="use_liger requires"):
                get_backend(backend, use_liger=True)
        load.assert_not_called()

    def test_kitchen_is_mutually_exclusive(self):
        with pytest.raises(ValueError, match="cannot be combined with use_kitchen"):
            get_backend("local", use_liger=True, use_kitchen=True)


@pytest.mark.parametrize("zero_centered", [False, True])
@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_norm_interface_and_checkpoint(zero_centered, sequence_parallel):
    config = _config(
        layernorm_zero_centered_gamma=zero_centered, sequence_parallel=sequence_parallel
    )
    norm = _norm_type()(config=config, hidden_size=128, eps=1e-6)
    assert norm.returns_residual is False
    assert norm.weight.sequence_parallel is sequence_parallel
    assert list(norm.state_dict()) == ["weight"]
    torch.testing.assert_close(
        norm.weight, torch.full_like(norm.weight, 0.0 if zero_centered else 1.0)
    )
    restored = _norm_type()(config=config, hidden_size=128, eps=1e-6)
    restored.load_state_dict(norm.state_dict(), strict=True)
    torch.testing.assert_close(norm.weight, restored.weight)


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_default_device_is_preserved(device):
    norm_type = _norm_type()
    with torch.device(device):
        norm = norm_type(config=_config(), hidden_size=128, eps=1e-6)
    assert norm.weight.device.type == device


def test_non_rmsnorm_config_is_rejected():
    with pytest.raises(ValueError, match="normalization='RMSNorm'"):
        _norm_type()(config=_config(normalization="LayerNorm"), hidden_size=128, eps=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("zero_centered", [False, True])
@pytest.mark.parametrize("hidden_size", [123, 512])
def test_forward_backward_parity(dtype, zero_centered, hidden_size):
    norm = _norm_type()(
        config=_config(layernorm_zero_centered_gamma=zero_centered),
        hidden_size=hidden_size,
        eps=1e-6,
    ).to(device="cuda", dtype=dtype)
    torch.manual_seed(123)
    with torch.no_grad():
        norm.weight.add_(0.1 * torch.randn_like(norm.weight))
    x = torch.randn(7, 3, hidden_size, device="cuda", dtype=dtype).transpose(0, 1)
    x.requires_grad_(True)
    original = x.detach().clone()
    ref_x = x.detach().clone().requires_grad_(True)
    ref_weight = norm.weight.detach().clone().requires_grad_(True)
    normalized = (
        ref_x.float() * torch.rsqrt(ref_x.float().square().mean(-1, keepdim=True) + 1e-6)
    ).to(dtype)
    expected = normalized * (ref_weight + int(zero_centered))
    actual = norm(x)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    tolerance = dict(atol=2e-5, rtol=2e-5) if dtype == torch.float32 else dict(atol=0.1, rtol=0.03)
    torch.testing.assert_close(actual, expected, **tolerance)
    torch.testing.assert_close(x.grad, ref_x.grad, **tolerance)
    torch.testing.assert_close(norm.weight.grad, ref_weight.grad, **tolerance)
    torch.testing.assert_close(x, original, atol=0, rtol=0)
