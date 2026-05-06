# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for DistributedOptimizer Adam-state offload / restore on checkpoint resume.

The resume cycle:
  load_state_dict() offloads Adam states (exp_avg, exp_avg_sq) to CPU pinned memory
  → _ensure_adam_states_on_gpu() moves them back to GPU just before the first optimizer
    step (after activations have been freed).

This tests the invariants without requiring the full DistributedOptimizer infrastructure
(buffers, DDP, process groups, etc.) by invoking the relevant methods on a minimal stub.
"""

import pytest
import torch
from torch.optim import Adam

from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestDistribOptimizerAdamStateOffload:

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_stub(self, optimizer, has_cpu_adam_states: bool = True):
        """Minimal stub exposing only the attributes consumed by the two methods under test."""

        class _Stub:
            pass

        stub = _Stub()
        stub.optimizer = optimizer
        stub._has_cpu_adam_states = has_cpu_adam_states
        return stub

    def _populated_adam(self):
        """Adam optimizer whose state is fully populated (one step performed on GPU)."""
        param = torch.nn.Parameter(torch.randn(8, 8, device='cuda'))
        optimizer = Adam([param], lr=1e-3)
        (param * param).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        return optimizer

    def _offload(self, optimizer):
        """Apply the same offload loop as DistributedOptimizer.load_state_dict().

        Returns the set of keys that were actually moved to CPU pinned memory.
        Note: some state tensors (e.g. 'step' in modern PyTorch Adam) are already
        on CPU and are not touched by the offload loop.
        """
        offloaded_keys: set = set()
        for param_state in optimizer.state.values():
            for key, val in list(param_state.items()):
                if isinstance(val, torch.Tensor) and val.is_cuda:
                    param_state[key] = val.cpu().pin_memory()
                    offloaded_keys.add(key)
        return offloaded_keys

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_offload_moves_cuda_states_to_cpu_pinned(self):
        """Offload loop must move all CUDA Adam state tensors to CPU pinned memory.

        Note: 'step' in modern PyTorch Adam is already CPU-resident; the offload
        loop does not touch it (is_cuda is False), so we only check the keys that
        were actually moved (exp_avg, exp_avg_sq, etc.).
        """
        optimizer = self._populated_adam()

        offloaded_keys = self._offload(optimizer)

        assert offloaded_keys, "No CUDA tensors found in Adam state — test setup broken"
        # exp_avg and exp_avg_sq are always CUDA tensors in standard Adam.
        assert 'exp_avg' in offloaded_keys
        assert 'exp_avg_sq' in offloaded_keys

        for param_state in optimizer.state.values():
            for key in offloaded_keys:
                val = param_state[key]
                assert not val.is_cuda, f"state['{key}'] still on GPU after offload"
                assert val.is_pinned(), f"state['{key}'] not pinned after offload"

    def test_ensure_adam_states_on_gpu_restores_and_clears_flag(self):
        """_ensure_adam_states_on_gpu() must move pinned states to GPU and clear the flag."""
        optimizer = self._populated_adam()
        offloaded_keys = self._offload(optimizer)

        stub = self._make_stub(optimizer, has_cpu_adam_states=True)
        DistributedOptimizer._ensure_adam_states_on_gpu(stub)
        torch.cuda.synchronize()

        assert not stub._has_cpu_adam_states, "_has_cpu_adam_states not cleared after restore"
        # Only the keys that were actually pinned (CUDA→CPU) should now be back on GPU.
        for param_state in optimizer.state.values():
            for key in offloaded_keys:
                val = param_state[key]
                assert val.is_cuda, f"state['{key}'] not back on GPU after restore"

    def test_ensure_adam_states_on_gpu_noop_when_flag_false(self):
        """_ensure_adam_states_on_gpu() must be a no-op on every step after the first."""
        optimizer = self._populated_adam()
        self._offload(optimizer)

        # Flag is False — as it would be on every step after the first restore.
        stub = self._make_stub(optimizer, has_cpu_adam_states=False)
        DistributedOptimizer._ensure_adam_states_on_gpu(stub)

        # States remain on CPU (not moved) because the early-return fired.
        for param_state in optimizer.state.values():
            for key, val in param_state.items():
                if isinstance(val, torch.Tensor):
                    assert not val.is_cuda, (
                        f"state['{key}'] moved to GPU despite _has_cpu_adam_states=False"
                    )

    def test_ensure_adam_states_on_gpu_ignores_non_pinned_cpu_tensors(self):
        """is_pinned() filter must leave intentionally-CPU-resident tensors untouched.

        This guards against accidentally GPU-ifying HybridDeviceOptimizer's CPU-resident
        optimizer states, which are not pinned.
        """
        optimizer = self._populated_adam()
        self._offload(optimizer)

        # Inject a non-pinned CPU tensor into the state (simulates HybridDeviceOptimizer).
        _SENTINEL = '__non_pinned_cpu_state__'
        first_state = next(iter(optimizer.state.values()))
        first_state[_SENTINEL] = torch.zeros(4)  # CPU, not pinned
        assert not first_state[_SENTINEL].is_pinned()

        stub = self._make_stub(optimizer, has_cpu_adam_states=True)
        DistributedOptimizer._ensure_adam_states_on_gpu(stub)
        torch.cuda.synchronize()

        # Non-pinned CPU tensor must remain on CPU.
        assert not first_state[_SENTINEL].is_cuda, (
            "Non-pinned CPU tensor was incorrectly moved to GPU by _ensure_adam_states_on_gpu"
        )
        # Offloaded (pinned) Adam tensors must be back on GPU.
        for key in ('exp_avg', 'exp_avg_sq'):
            assert first_state[key].is_cuda, f"Offloaded tensor '{key}' not restored to GPU"

        del first_state[_SENTINEL]
