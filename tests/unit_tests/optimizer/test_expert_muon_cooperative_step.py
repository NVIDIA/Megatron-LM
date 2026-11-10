# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# License for these changes: BSD-3-Clause.

"""Real public-class default dispatch regression; no optimizer kernels executed."""

import pytest

import megatron.core.optimizer.emerging_optimizers as muon


@pytest.mark.skipif(not muon.HAVE_EMERGING_OPTIMIZERS, reason="requires emerging_optimizers")
def test_default_step_preserves_real_adaptive_mro(monkeypatch):
    calls = []
    adaptive_marker, native_marker = object(), object()

    def adaptive_step(self, closure=None):
        calls.append(("adaptive", self, closure))
        return adaptive_marker

    def native_step(self, closure=None):
        calls.append(("native", self, closure))
        return native_marker

    monkeypatch.setattr(muon.AdaptiveMuon, "step", adaptive_step)
    monkeypatch.setattr(muon.OrthogonalizedOptimizer, "step", native_step)
    adaptive = object.__new__(muon.TensorParallelAdaptiveMuon)
    adaptive.expert_ns_batch_size = 1
    closure = lambda: None
    # Public actual class retains its original AdaptiveMuon entry.
    assert adaptive.step(closure) is adaptive_marker
    # The new TPM method must cooperate with the SAME actual C3 MRO.
    assert muon.TensorParallelMuon.step(adaptive, closure) is adaptive_marker
    native = object.__new__(muon.TensorParallelMuon)
    native.expert_ns_batch_size = 1
    assert native.step(closure) is native_marker
    assert [row[0] for row in calls] == ["adaptive", "adaptive", "native"]
    assert all(row[2] is closure for row in calls)
