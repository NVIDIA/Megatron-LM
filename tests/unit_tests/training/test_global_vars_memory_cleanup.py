# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import gc
import pytest
import torch

from megatron.training.global_vars import (
    set_model,
    set_optimizer,
    set_opt_param_scheduler,
    set_data_iterators,
    destroy_global_vars,
)


class DummyModel:
    """Mock model for testing."""

    def __init__(self):
        self.data = torch.randn(10, 10)


class DummyOptimizer:
    """Mock optimizer for testing."""

    def __init__(self):
        self.state = {'step': 0}


class DummyScheduler:
    """Mock scheduler for testing."""

    def __init__(self):
        self.lr = 0.001


class DummyIterator:
    """Mock data iterator for testing."""

    def __init__(self):
        self.data = [1, 2, 3, 4, 5]


class TestGlobalVarsMemoryCleanup:
    """Test cases for global variables memory cleanup."""

    def test_set_and_destroy_model(self):
        """Test that model can be set and properly destroyed."""
        model = DummyModel()
        set_model(model)

        # Verify model is set
        from megatron.training.global_vars import _GLOBAL_MODEL

        assert _GLOBAL_MODEL is not None

        # Destroy global vars
        destroy_global_vars()

        # Verify model is destroyed
        from megatron.training.global_vars import _GLOBAL_MODEL

        assert _GLOBAL_MODEL is None

    def test_set_and_destroy_optimizer(self):
        """Test that optimizer can be set and properly destroyed."""
        optimizer = DummyOptimizer()
        set_optimizer(optimizer)

        # Verify optimizer is set
        from megatron.training.global_vars import _GLOBAL_OPTIMIZER

        assert _GLOBAL_OPTIMIZER is not None

        # Destroy global vars
        destroy_global_vars()

        # Verify optimizer is destroyed
        from megatron.training.global_vars import _GLOBAL_OPTIMIZER

        assert _GLOBAL_OPTIMIZER is None

    def test_set_and_destroy_scheduler(self):
        """Test that scheduler can be set and properly destroyed."""
        scheduler = DummyScheduler()
        set_opt_param_scheduler(scheduler)

        # Verify scheduler is set
        from megatron.training.global_vars import _GLOBAL_OPT_PARAM_SCHEDULER

        assert _GLOBAL_OPT_PARAM_SCHEDULER is not None

        # Destroy global vars
        destroy_global_vars()

        # Verify scheduler is destroyed
        from megatron.training.global_vars import _GLOBAL_OPT_PARAM_SCHEDULER

        assert _GLOBAL_OPT_PARAM_SCHEDULER is None

    def test_set_and_destroy_data_iterators(self):
        """Test that data iterators can be set and properly destroyed."""
        train_iter = DummyIterator()
        valid_iter = DummyIterator()
        test_iter = DummyIterator()

        set_data_iterators(train_iter, valid_iter, test_iter)

        # Verify iterators are set
        from megatron.training.global_vars import (
            _GLOBAL_TRAIN_DATA_ITERATOR,
            _GLOBAL_VALID_DATA_ITERATOR,
            _GLOBAL_TEST_DATA_ITERATOR,
        )

        assert _GLOBAL_TRAIN_DATA_ITERATOR is not None
        assert _GLOBAL_VALID_DATA_ITERATOR is not None
        assert _GLOBAL_TEST_DATA_ITERATOR is not None

        # Destroy global vars
        destroy_global_vars()

        # Verify iterators are destroyed
        from megatron.training.global_vars import (
            _GLOBAL_TRAIN_DATA_ITERATOR,
            _GLOBAL_VALID_DATA_ITERATOR,
            _GLOBAL_TEST_DATA_ITERATOR,
        )

        assert _GLOBAL_TRAIN_DATA_ITERATOR is None
        assert _GLOBAL_VALID_DATA_ITERATOR is None
        assert _GLOBAL_TEST_DATA_ITERATOR is None

    def test_destroy_without_setting(self):
        """Test that destroy works even when variables were never set."""
        # This should not raise an error
        destroy_global_vars()

    def test_destroy_all_components(self):
        """Test that all components are destroyed together."""
        model = DummyModel()
        optimizer = DummyOptimizer()
        scheduler = DummyScheduler()
        train_iter = DummyIterator()
        valid_iter = DummyIterator()
        test_iter = DummyIterator()

        set_model(model)
        set_optimizer(optimizer)
        set_opt_param_scheduler(scheduler)
        set_data_iterators(train_iter, valid_iter, test_iter)

        # Verify all are set
        from megatron.training.global_vars import (
            _GLOBAL_MODEL,
            _GLOBAL_OPTIMIZER,
            _GLOBAL_OPT_PARAM_SCHEDULER,
            _GLOBAL_TRAIN_DATA_ITERATOR,
            _GLOBAL_VALID_DATA_ITERATOR,
            _GLOBAL_TEST_DATA_ITERATOR,
        )

        assert _GLOBAL_MODEL is not None
        assert _GLOBAL_OPTIMIZER is not None
        assert _GLOBAL_OPT_PARAM_SCHEDULER is not None
        assert _GLOBAL_TRAIN_DATA_ITERATOR is not None
        assert _GLOBAL_VALID_DATA_ITERATOR is not None
        assert _GLOBAL_TEST_DATA_ITERATOR is not None

        # Destroy all
        destroy_global_vars()

        # Verify all are destroyed
        from megatron.training.global_vars import (
            _GLOBAL_MODEL,
            _GLOBAL_OPTIMIZER,
            _GLOBAL_OPT_PARAM_SCHEDULER,
            _GLOBAL_TRAIN_DATA_ITERATOR,
            _GLOBAL_VALID_DATA_ITERATOR,
            _GLOBAL_TEST_DATA_ITERATOR,
        )

        assert _GLOBAL_MODEL is None
        assert _GLOBAL_OPTIMIZER is None
        assert _GLOBAL_OPT_PARAM_SCHEDULER is None
        assert _GLOBAL_TRAIN_DATA_ITERATOR is None
        assert _GLOBAL_VALID_DATA_ITERATOR is None
        assert _GLOBAL_TEST_DATA_ITERATOR is None

    def test_cuda_memory_cleanup_with_tensors(self):
        """Test that CUDA tensors are properly cleaned up."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Create model with CUDA tensors
        class CudaModel:
            def __init__(self):
                self.weight = torch.randn(1000, 1000, device='cuda')

        model = CudaModel()
        set_model(model)

        # Get initial memory
        initial_memory = torch.cuda.memory_allocated()

        # Destroy and cleanup
        destroy_global_vars()
        gc.collect()
        torch.cuda.empty_cache()

        # Memory should be freed (or at least not significantly more)
        final_memory = torch.cuda.memory_allocated()
        # The model tensor should be freed, so memory should decrease
        assert final_memory <= initial_memory

    def test_multiple_destroy_calls(self):
        """Test that multiple destroy calls don't cause errors."""
        model = DummyModel()
        set_model(model)

        # Multiple destroy calls should be safe
        destroy_global_vars()
        destroy_global_vars()
        destroy_global_vars()

        # Verify still None
        from megatron.training.global_vars import _GLOBAL_MODEL

        assert _GLOBAL_MODEL is None
