# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from unittest.mock import MagicMock, patch

import pytest
import torch

from megatron.core.inference.symmetric_memory import SymmetricMemoryBuffer, SymmetricMemoryManager


def _make_buffer_with_handle(num_bytes: int = 1024):
    """Return a SymmetricMemoryBuffer whose symm_buffer is a real CPU tensor.

    This skips the actual symmetric-memory rendezvous: we directly assign a
    fake handle and a uint8 buffer so we can exercise the allocation logic
    without GPUs or NCCL.
    """
    buf = SymmetricMemoryBuffer.__new__(SymmetricMemoryBuffer)
    buf.init_failure_reason = None
    buf.symm_buffer = torch.zeros(num_bytes, dtype=torch.uint8)
    buf.symm_mem_hdl = MagicMock(name="fake_symm_mem_handle")
    return buf


class TestSymmetricMemoryBufferInit:

    def test_init_failure_when_torch_symm_mem_missing(self):
        """When HAVE_TORCH_SYMM_MEM is False, init records the failure and leaves the buffer empty."""
        with patch("megatron.core.inference.symmetric_memory.HAVE_TORCH_SYMM_MEM", False), patch(
            "megatron.core.inference.symmetric_memory.HAVE_TRITON", True
        ):
            buf = SymmetricMemoryBuffer(size_in_mb=1, process_group=MagicMock())
        assert buf.symm_buffer is None
        assert buf.symm_mem_hdl is None
        assert "symmetric_memory" in buf.init_failure_reason

    def test_init_failure_when_triton_missing(self):
        """When HAVE_TRITON is False, init records that triton is unavailable."""
        with patch("megatron.core.inference.symmetric_memory.HAVE_TORCH_SYMM_MEM", True), patch(
            "megatron.core.inference.symmetric_memory.HAVE_TRITON", False
        ):
            buf = SymmetricMemoryBuffer(size_in_mb=1, process_group=MagicMock())
        assert buf.symm_buffer is None
        assert buf.symm_mem_hdl is None
        assert "triton" in buf.init_failure_reason.lower()

    def test_init_runtime_error_recorded(self):
        """A RuntimeError during rendezvous is captured as init_failure_reason."""
        fake_symm_mem = MagicMock()
        fake_symm_mem.enable_symm_mem_for_group = MagicMock()
        fake_symm_mem.empty.side_effect = RuntimeError("oom")
        with patch("megatron.core.inference.symmetric_memory.HAVE_TORCH_SYMM_MEM", True), patch(
            "megatron.core.inference.symmetric_memory.HAVE_TRITON", True
        ), patch("megatron.core.inference.symmetric_memory.symm_mem", fake_symm_mem):
            buf = SymmetricMemoryBuffer(size_in_mb=1, process_group=MagicMock())
        assert buf.symm_buffer is None
        assert buf.symm_mem_hdl is None
        assert "RuntimeError" in buf.init_failure_reason
        assert "oom" in buf.init_failure_reason


class TestSymmetricMemoryBufferAllocation:

    def test_can_allocate_returns_false_without_handle(self):
        """_can_allocate returns False when no symm_mem_hdl is present."""
        buf = SymmetricMemoryBuffer.__new__(SymmetricMemoryBuffer)
        buf.symm_mem_hdl = None
        buf.symm_buffer = None
        assert buf._can_allocate(numel=10, dtype=torch.float32) is False

    def test_can_allocate_true_when_buffer_large_enough(self):
        """_can_allocate returns True when bytes fit in the buffer."""
        buf = _make_buffer_with_handle(num_bytes=1024)
        # 100 floats = 400 bytes < 1024.
        assert buf._can_allocate(numel=100, dtype=torch.float32) is True

    def test_can_allocate_false_when_buffer_too_small(self):
        """_can_allocate returns False when the request exceeds buffer size."""
        buf = _make_buffer_with_handle(num_bytes=64)
        # 100 floats = 400 bytes > 64.
        assert buf._can_allocate(numel=100, dtype=torch.float32) is False

    def test_allocate_returns_view(self):
        """_allocate returns a view of the underlying buffer with the requested shape and dtype."""
        buf = _make_buffer_with_handle(num_bytes=64)
        out = buf._allocate(numel=4, dtype=torch.float32)
        assert out.numel() == 4
        assert out.dtype == torch.float32

    def test_maybe_get_tensor_returns_none_without_handle(self):
        """maybe_get_tensor signals unavailability when no symm_mem handle exists."""
        buf = SymmetricMemoryBuffer.__new__(SymmetricMemoryBuffer)
        buf.symm_mem_hdl = None
        buf.symm_buffer = None
        result = buf.maybe_get_tensor((2, 3), torch.float32)
        assert result == {"tensor": None, "handle": None}

    def test_maybe_get_tensor_returns_none_when_too_big(self):
        """maybe_get_tensor returns None when the requested tensor exceeds buffer size."""
        buf = _make_buffer_with_handle(num_bytes=16)  # only 16 bytes
        # (10, 10) floats = 400 bytes > 16.
        result = buf.maybe_get_tensor((10, 10), torch.float32)
        assert result["tensor"] is None
        assert result["handle"] is None

    def test_maybe_get_tensor_returns_shaped_tensor(self):
        """maybe_get_tensor returns a tensor of the requested shape on success."""
        buf = _make_buffer_with_handle(num_bytes=4096)
        result = buf.maybe_get_tensor((2, 3), torch.float32)
        assert result["tensor"] is not None
        assert result["tensor"].shape == (2, 3)
        assert result["tensor"].dtype == torch.float32
        assert result["handle"] is buf.symm_mem_hdl

    def test_maybe_get_tensors_returns_none_without_handle(self):
        """maybe_get_tensors returns the unavailable sentinel when no handle is present."""
        buf = SymmetricMemoryBuffer.__new__(SymmetricMemoryBuffer)
        buf.symm_mem_hdl = None
        buf.symm_buffer = None
        result = buf.maybe_get_tensors([(4, torch.float32)])
        assert result == {"handle": None, "tensors": None}

    def test_maybe_get_tensors_packs_with_alignment(self):
        """maybe_get_tensors packs multiple tensors with byte-aligned offsets."""
        buf = _make_buffer_with_handle(num_bytes=4096)
        # 4 fp32 = 16B, aligned to 16B; 1 int32 = 4B, aligned to 16B.
        result = buf.maybe_get_tensors(
            [(4, torch.float32), (1, torch.int32)], alignment=16
        )
        assert result["handle"] is buf.symm_mem_hdl
        tensors = result["tensors"]
        assert len(tensors) == 2
        offset_a = tensors[0][1]
        offset_b = tensors[1][1]
        # First tensor starts at offset 0; second is aligned past the first (16-byte boundary).
        assert offset_a == 0
        assert offset_b == 16

    def test_maybe_get_tensors_returns_none_when_total_exceeds_buffer(self):
        """maybe_get_tensors returns None when the aligned total does not fit."""
        buf = _make_buffer_with_handle(num_bytes=8)
        result = buf.maybe_get_tensors([(8, torch.float32)], alignment=16)
        assert result["handle"] is None
        assert result["tensors"] is None


class TestSymmetricMemoryManager:

    def setup_method(self):
        """Clear the class-level registry between tests."""
        SymmetricMemoryManager.destroy()

    def teardown_method(self):
        SymmetricMemoryManager.destroy()

    def test_get_buffer_creates_on_first_access(self):
        """get_buffer constructs a new buffer the first time a key is used."""
        with patch(
            "megatron.core.inference.symmetric_memory.SymmetricMemoryBuffer"
        ) as fake_cls:
            fake_cls.return_value = "BUF"
            out = SymmetricMemoryManager.get_buffer("tp", process_group=MagicMock(), size_mb=128)
            assert out == "BUF"
            fake_cls.assert_called_once()
            # The size_mb was forwarded.
            kwargs = fake_cls.call_args.kwargs
            assert kwargs["size_in_mb"] == 128

    def test_get_buffer_reuses_existing_entry(self):
        """Subsequent get_buffer calls return the same instance without recreating."""
        with patch(
            "megatron.core.inference.symmetric_memory.SymmetricMemoryBuffer"
        ) as fake_cls:
            fake_cls.return_value = "BUF"
            first = SymmetricMemoryManager.get_buffer("tp", process_group=MagicMock())
            second = SymmetricMemoryManager.get_buffer("tp")  # no process_group needed now
            assert first is second
            assert fake_cls.call_count == 1

    def test_get_buffer_requires_process_group_on_first_call(self):
        """The first call without process_group asserts."""
        with pytest.raises(AssertionError):
            SymmetricMemoryManager.get_buffer("ep")

    def test_get_buffer_uses_default_size(self):
        """When size_mb is omitted, the default is forwarded to SymmetricMemoryBuffer."""
        with patch(
            "megatron.core.inference.symmetric_memory.SymmetricMemoryBuffer"
        ) as fake_cls:
            fake_cls.return_value = "BUF"
            SymmetricMemoryManager.get_buffer("ep", process_group=MagicMock())
            kwargs = fake_cls.call_args.kwargs
            assert kwargs["size_in_mb"] == SymmetricMemoryManager._default_size_mb

    def test_destroy_specific_key(self):
        """destroy(key) removes only that buffer; others remain."""
        with patch(
            "megatron.core.inference.symmetric_memory.SymmetricMemoryBuffer", return_value="BUF"
        ):
            SymmetricMemoryManager.get_buffer("a", process_group=MagicMock())
            SymmetricMemoryManager.get_buffer("b", process_group=MagicMock())
        SymmetricMemoryManager.destroy("a")
        assert SymmetricMemoryManager.is_initialized("a") is False
        assert SymmetricMemoryManager.is_initialized("b") is True

    def test_destroy_all(self):
        """destroy() with no key wipes the entire registry."""
        with patch(
            "megatron.core.inference.symmetric_memory.SymmetricMemoryBuffer", return_value="BUF"
        ):
            SymmetricMemoryManager.get_buffer("a", process_group=MagicMock())
            SymmetricMemoryManager.get_buffer("b", process_group=MagicMock())
        SymmetricMemoryManager.destroy()
        assert SymmetricMemoryManager.is_initialized("a") is False
        assert SymmetricMemoryManager.is_initialized("b") is False

    def test_destroy_unknown_key_is_safe(self):
        """destroy(unknown) is a no-op rather than an error."""
        SymmetricMemoryManager.destroy("never_created")
        assert SymmetricMemoryManager.is_initialized("never_created") is False

    def test_is_initialized_default_false(self):
        """is_initialized returns False before any get_buffer call."""
        assert SymmetricMemoryManager.is_initialized("never") is False
