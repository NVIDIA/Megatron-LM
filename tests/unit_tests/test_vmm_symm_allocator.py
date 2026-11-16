# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the VMM symmetric allocator's runtime extension build."""

from pathlib import Path
from unittest.mock import Mock

import pytest

import megatron.core.allocator.vmm_symm_allocator as vmm_alloc


@pytest.fixture(autouse=True)
def reset_build_state(monkeypatch):
    """Isolate the allocator's successful and failed build caches."""
    monkeypatch.setattr(vmm_alloc, "_allocator", None)
    monkeypatch.setattr(vmm_alloc, "_build_error", None)


def test_build_uses_packaged_source(monkeypatch):
    """Avoid source writes before PyTorch's build lock and cache the allocator."""
    allocator = object()
    extension = Mock(get_vmm_allocator=Mock(return_value=allocator))
    load = Mock(return_value=extension)
    monkeypatch.setattr(vmm_alloc.torch.utils.cpp_extension, "load", load)
    monkeypatch.setattr(
        vmm_alloc.torch.utils.cpp_extension,
        "load_inline",
        lambda **_kwargs: pytest.fail("load_inline writes main.cpp before taking its build lock"),
    )

    vmm_alloc._build_vmm_allocator()
    vmm_alloc._build_vmm_allocator()

    assert vmm_alloc._allocator is allocator
    assert vmm_alloc._build_error is None
    load.assert_called_once()
    extension.get_vmm_allocator.assert_called_once_with()
    kwargs = load.call_args.kwargs
    source_path = Path(vmm_alloc.__file__).parent / "csrc" / "vmm_symm_allocator.cpp"
    assert kwargs["sources"] == [str(source_path)]
    assert source_path.is_file()
    assert not source_path.is_relative_to(Path(kwargs["build_directory"]))
    assert kwargs["with_cuda"] is True
    assert kwargs["extra_ldflags"] == ["-lcuda"]
    assert kwargs["is_python_module"] is True


def test_build_failure_is_cached(monkeypatch):
    """Preserve the build error and avoid retrying compilation on later calls."""
    build_error = RuntimeError("compiler failed")
    load = Mock(side_effect=build_error)
    monkeypatch.setattr(vmm_alloc.torch.utils.cpp_extension, "load", load)

    for _ in range(2):
        with pytest.raises(RuntimeError, match="requires nvcc and libcuda") as exc_info:
            vmm_alloc._build_vmm_allocator()
        assert exc_info.value.__cause__ is build_error

    assert vmm_alloc._allocator is None
    assert vmm_alloc._build_error is build_error
    load.assert_called_once()
