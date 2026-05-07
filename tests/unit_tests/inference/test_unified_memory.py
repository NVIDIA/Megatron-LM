# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from unittest.mock import MagicMock, patch

import pytest
import torch

from megatron.core.inference import unified_memory as um


class TestEnumsAndExceptions:

    def test_compilation_state_distinct(self):
        """CompilationState members are all distinct."""
        values = {
            um.CompilationState.UNATTEMPTED.value,
            um.CompilationState.FAILURE.value,
            um.CompilationState.SUCCESS.value,
        }
        assert len(values) == 3

    def test_unsupported_error_is_exception(self):
        """UnifiedMemoryUnsupportedError is an Exception subclass."""
        assert issubclass(um.UnifiedMemoryUnsupportedError, Exception)

    def test_compile_timeout_error_inherits_unsupported(self):
        """UnifiedMemoryCompileTimeoutError is a subclass of UnifiedMemoryUnsupportedError."""
        assert issubclass(um.UnifiedMemoryCompileTimeoutError, um.UnifiedMemoryUnsupportedError)

    def test_unsupported_error_can_be_raised(self):
        """The exception types can be raised and caught."""
        with pytest.raises(um.UnifiedMemoryUnsupportedError):
            raise um.UnifiedMemoryUnsupportedError("nope")


class TestPrefetchManagedTensorGuards:

    def test_none_tensor_short_circuits(self):
        """prefetch_managed_tensor(None, ...) is a silent no-op."""
        # No mocking needed — the function returns before touching the lib.
        um.prefetch_managed_tensor(None, device=0)

    def test_non_tensor_raises_type_error(self):
        """A non-tensor argument triggers TypeError."""
        with pytest.raises(TypeError, match="torch.Tensor"):
            um.prefetch_managed_tensor("not-a-tensor", device=0)

    def test_empty_tensor_short_circuits(self):
        """A 0-element tensor is a silent no-op (no lib call)."""
        empty = torch.empty(0)
        um.prefetch_managed_tensor(empty, device=0)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA tensor")
    def test_cpu_tensor_with_elements_raises_value_error(self):
        """Non-empty CPU tensors raise ValueError (must be CUDA)."""
        cpu_t = torch.zeros(4)  # CPU
        with pytest.raises(ValueError, match="CUDA tensor"):
            um.prefetch_managed_tensor(cpu_t, device=0)


class TestAdviseManagedTensorPreferredLocationGuards:

    def test_none_short_circuits(self):
        """A None tensor is a silent no-op."""
        um.advise_managed_tensor_preferred_location(None, device=0)

    def test_non_tensor_raises_type_error(self):
        """Non-tensor input raises TypeError."""
        with pytest.raises(TypeError, match="torch.Tensor"):
            um.advise_managed_tensor_preferred_location(42, device=0)

    def test_empty_tensor_short_circuits(self):
        """A 0-element tensor is a silent no-op."""
        um.advise_managed_tensor_preferred_location(torch.empty(0), device=0)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_cpu_tensor_with_elements_raises_value_error(self):
        """Non-empty CPU tensors raise ValueError."""
        with pytest.raises(ValueError, match="CUDA tensor"):
            um.advise_managed_tensor_preferred_location(torch.zeros(4), device=0)


class TestAdviseManagedTensorAccessedByGuards:

    def test_none_short_circuits(self):
        """A None tensor is a silent no-op."""
        um.advise_managed_tensor_accessed_by(None, device=0)

    def test_non_tensor_raises_type_error(self):
        """Non-tensor input raises TypeError."""
        with pytest.raises(TypeError, match="torch.Tensor"):
            um.advise_managed_tensor_accessed_by([1, 2, 3], device=0)

    def test_empty_tensor_short_circuits(self):
        """A 0-element tensor is a silent no-op."""
        um.advise_managed_tensor_accessed_by(torch.empty(0), device=0)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_cpu_tensor_with_elements_raises_value_error(self):
        """Non-empty CPU tensors raise ValueError."""
        with pytest.raises(ValueError, match="CUDA tensor"):
            um.advise_managed_tensor_accessed_by(torch.zeros(4), device=0)


class TestPrefetchManagedModuleParameters:

    def test_none_module_returns_zero(self):
        """A None module returns 0 bytes prefetched."""
        assert um.prefetch_managed_module_parameters(None, device=0) == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA tensors")
    def test_module_with_no_cuda_params_returns_zero(self):
        """A module whose parameters live on CPU contributes 0 bytes."""
        model = torch.nn.Linear(4, 8)  # CPU tensors by default
        assert um.prefetch_managed_module_parameters(model, device=0) == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA tensors")
    def test_module_with_cuda_params_calls_prefetch(self):
        """A module with CUDA params delegates to prefetch_managed_tensor."""
        model = torch.nn.Linear(4, 8).cuda()
        with patch.object(um, "prefetch_managed_tensor", return_value=None) as fake_prefetch:
            total = um.prefetch_managed_module_parameters(model, device=0)
        # Two parameters (weight + bias), both have content.
        assert fake_prefetch.call_count == 2
        # Total nbytes equals the sum of parameter nbytes.
        expected = sum(p.nbytes for p in model.parameters())
        assert total == expected

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA tensors")
    def test_dedup_by_data_ptr(self):
        """Parameters sharing storage are prefetched only once."""

        class _Shared(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.nn.Parameter(torch.zeros(4, device="cuda"))
                # b shares storage with a.
                self.b = torch.nn.Parameter(self.a.data, requires_grad=False)

        model = _Shared()
        with patch.object(um, "prefetch_managed_tensor", return_value=None) as fake_prefetch:
            um.prefetch_managed_module_parameters(model, device=0)
        # Even though there are two parameters, dedup means one prefetch call.
        assert fake_prefetch.call_count == 1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA tensors")
    def test_include_buffers_walks_buffers(self):
        """include_buffers=True also prefetches module buffers."""
        model = torch.nn.BatchNorm1d(4).cuda()
        # BatchNorm has running_mean / running_var as buffers and weight / bias as params.
        with patch.object(um, "prefetch_managed_tensor", return_value=None) as fake_prefetch:
            um.prefetch_managed_module_parameters(model, device=0, include_buffers=True)
        # 2 params + 2 buffers = 4 calls (num_batches_tracked is a 0-d 0-numel tensor on
        # some PyTorch versions; conservatively assert at least 4).
        assert fake_prefetch.call_count >= 4

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA tensors")
    def test_prefetch_error_raises_runtime_error(self):
        """A non-zero return from prefetch_managed_tensor turns into RuntimeError."""
        model = torch.nn.Linear(4, 8).cuda()
        with patch.object(um, "prefetch_managed_tensor", return_value=42):
            with pytest.raises(RuntimeError, match="cudaMemPrefetchAsync failed"):
                um.prefetch_managed_module_parameters(model, device=0)


class TestAdviseManagedModuleParametersPreferredLocation:

    def test_none_module_is_noop(self):
        """A None module is a silent no-op."""
        # Returns None implicitly.
        assert um.advise_managed_module_parameters_preferred_location(None, device=0) is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA tensors")
    def test_module_with_cuda_params_calls_advise(self):
        """A module with CUDA params delegates to advise_managed_tensor_preferred_location."""
        model = torch.nn.Linear(4, 8).cuda()
        with patch.object(
            um, "advise_managed_tensor_preferred_location", return_value=None
        ) as fake_advise:
            um.advise_managed_module_parameters_preferred_location(model, device=0)
        assert fake_advise.call_count == 2

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA tensors")
    def test_advise_error_raises_runtime_error(self):
        """A non-zero return from advise_managed_tensor_preferred_location raises RuntimeError."""
        model = torch.nn.Linear(4, 8).cuda()
        with patch.object(um, "advise_managed_tensor_preferred_location", return_value=99):
            with pytest.raises(RuntimeError, match="cudaMemAdvise"):
                um.advise_managed_module_parameters_preferred_location(model, device=0)


class TestCompileAllocator:

    def test_short_circuits_when_already_attempted(self):
        """compile_allocator does nothing when _compilation_state isn't UNATTEMPTED."""
        # Save original state.
        original_state = um._compilation_state
        try:
            um._compilation_state = um.CompilationState.SUCCESS
            # Should not change state.
            um.compile_allocator()
            assert um._compilation_state is um.CompilationState.SUCCESS
        finally:
            um._compilation_state = original_state

    def test_records_failure_when_mempool_unavailable(self):
        """Without _has_mem_pool, compilation transitions to FAILURE with a clear message."""
        original_state = um._compilation_state
        original_error = um._compilation_error
        try:
            um._compilation_state = um.CompilationState.UNATTEMPTED
            um._compilation_error = None
            with patch.object(um, "_has_mem_pool", False):
                um.compile_allocator()
            assert um._compilation_state is um.CompilationState.FAILURE
            assert "MemPool" in (um._compilation_error or "")
        finally:
            um._compilation_state = original_state
            um._compilation_error = original_error


class TestCreateUnifiedMempool:

    def test_raises_unsupported_when_compilation_failed(self):
        """create_unified_mempool raises UnifiedMemoryUnsupportedError when allocator failed to compile."""
        original_state = um._compilation_state
        try:
            um._compilation_state = um.CompilationState.FAILURE
            um._compilation_error = "test failure"
            with pytest.raises(um.UnifiedMemoryUnsupportedError, match="test failure"):
                um.create_unified_mempool()
        finally:
            um._compilation_state = original_state

    def test_raises_with_unknown_reason_when_no_error_recorded(self):
        """If _compilation_error is None, the message falls back to a generic phrase."""
        original_state = um._compilation_state
        original_error = um._compilation_error
        try:
            um._compilation_state = um.CompilationState.FAILURE
            um._compilation_error = None
            with pytest.raises(um.UnifiedMemoryUnsupportedError, match="Unknown reason"):
                um.create_unified_mempool()
        finally:
            um._compilation_state = original_state
            um._compilation_error = original_error
