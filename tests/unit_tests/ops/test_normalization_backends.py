# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Guard the automatic numerical contract independently of optional kernels."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core.ops.normalization import backends


@pytest.fixture
def supported_backend(monkeypatch):
    """Model the validated installation without importing the kernel."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (10, 3))
    monkeypatch.setattr(torch, "__version__", "2.11.0+cu130")
    versions = {"transformer_engine": "2.15.0+42b84005", "triton": "3.6.0"}
    monkeypatch.setattr(backends, "version", versions.__getitem__)
    for name in ("NVTE_NORM_FWD_USE_CUDNN", "NVTE_NORM_USE_CUDNN", "NVTE_USE_CUDNN_RMSNORM"):
        monkeypatch.delenv(name, raising=False)
    kernel = object()
    require = Mock(return_value=SimpleNamespace(fused_bf16_add_rmsnorm=kernel))
    monkeypatch.setattr(backends, "require", require)
    return kernel, require, versions


@pytest.mark.parametrize("hidden", [8, 128, 768, 2048, 2688, 4096, 8192])
def test_supported_backend_binds_kernel(supported_backend, hidden):
    kernel, require, _ = supported_backend
    assert backends.select_bf16_add_rmsnorm(hidden, torch.bfloat16, False) is kernel
    require.assert_called_once()


@pytest.mark.parametrize(
    "contract", ["width", "dtype", "fp32", "gpu", "te", "triton", "torch", "cudnn"]
)
def test_other_contracts_keep_native_path(supported_backend, monkeypatch, contract):
    _, require, versions = supported_backend
    width, dtype, fp32 = 2688, torch.bfloat16, False
    if contract == "width":
        width = 16384
    elif contract == "dtype":
        dtype = torch.float16
    elif contract == "fp32":
        fp32 = True
    elif contract == "gpu":
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    elif contract == "te":
        versions["transformer_engine"] = "2.16.0"
    elif contract == "triton":
        versions["triton"] = "3.7.0"
    elif contract == "torch":
        monkeypatch.setattr(torch, "__version__", "2.12.0")
    else:
        monkeypatch.setenv("NVTE_NORM_FWD_USE_CUDNN", "1")
    assert backends.select_bf16_add_rmsnorm(width, dtype, fp32) is None
    require.assert_not_called()


def test_selected_kernel_import_failure_is_explicit(supported_backend):
    _, require, _ = supported_backend
    require.side_effect = ImportError("BF16 residual addition and RMSNorm requires Triton")
    with pytest.raises(ImportError, match="requires Triton"):
        backends.select_bf16_add_rmsnorm(2688, torch.bfloat16, False)
