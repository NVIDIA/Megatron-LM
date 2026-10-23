# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Validate NVSHMEM's pre-initialization guards without GPU or native-library access."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from megatron.core.resharding.nvshmem_copy_service.core import gpu_resource_manager


@pytest.fixture
def nvshmem_runtime(monkeypatch):
    core = SimpleNamespace(
        get_version=Mock(return_value=SimpleNamespace(libnvshmem_version="3.7.1"))
    )
    monkeypatch.setattr(gpu_resource_manager, "nvshmem", SimpleNamespace(core=core), raising=False)
    monkeypatch.setattr(gpu_resource_manager, "HAVE_NVSHMEM", True)
    monkeypatch.setattr(gpu_resource_manager.dist, "is_initialized", lambda: False)
    monkeypatch.setenv("NVSHMEM_MAX_CTAS", "2")
    monkeypatch.delenv("MEGATRON_NVSHMEM_ALLOW_UNSUPPORTED_VERSION", raising=False)
    return core


@pytest.mark.parametrize("version", ["3.6.5", "3.7.1"])
def test_supported_loaded_runtime_reaches_distributed_guard(nvshmem_runtime, version):
    nvshmem_runtime.get_version.return_value.libnvshmem_version = version

    with pytest.raises(RuntimeError, match="torch.distributed must be initialized"):
        gpu_resource_manager.GPUResourceManager().init()

    nvshmem_runtime.get_version.assert_called_once_with()


@pytest.mark.parametrize("version", ["3.4.5", "3.6.4"])
def test_old_loaded_runtime_is_rejected(nvshmem_runtime, version):
    nvshmem_runtime.get_version.return_value.libnvshmem_version = version

    with pytest.raises(RuntimeError, match=f"Loaded NVSHMEM library {version} is older"):
        gpu_resource_manager.GPUResourceManager().init()


def test_unsupported_runtime_override_preserves_warning(nvshmem_runtime, monkeypatch, caplog):
    nvshmem_runtime.get_version.return_value.libnvshmem_version = "3.4.5"
    monkeypatch.setenv("MEGATRON_NVSHMEM_ALLOW_UNSUPPORTED_VERSION", "1")

    with pytest.raises(RuntimeError, match="torch.distributed must be initialized"):
        gpu_resource_manager.GPUResourceManager().init()

    assert "Loaded NVSHMEM library 3.4.5 is older" in caplog.text


def test_runtime_load_failure_is_not_silently_ignored(nvshmem_runtime):
    nvshmem_runtime.get_version.side_effect = RuntimeError("Cannot load libnvshmem_host.so")

    with pytest.raises(RuntimeError, match="Cannot load libnvshmem_host.so"):
        gpu_resource_manager.GPUResourceManager().init()


@pytest.mark.parametrize("max_ctas", [None, "4"])
def test_required_cta_limit_is_enforced(nvshmem_runtime, monkeypatch, max_ctas):
    if max_ctas is None:
        monkeypatch.delenv("NVSHMEM_MAX_CTAS")
    else:
        monkeypatch.setenv("NVSHMEM_MAX_CTAS", max_ctas)

    with pytest.raises(RuntimeError, match="NVSHMEM_MAX_CTAS must be set to '2'"):
        gpu_resource_manager.GPUResourceManager().init()


def test_missing_bindings_fail_before_runtime_query(nvshmem_runtime, monkeypatch):
    monkeypatch.setattr(gpu_resource_manager, "HAVE_NVSHMEM", False)

    with pytest.raises(RuntimeError, match="nvshmem.core is not available"):
        gpu_resource_manager.GPUResourceManager().init()

    nvshmem_runtime.get_version.assert_not_called()
