# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Regression test for the GTP + CUDA-graph capture-step grad-norm bug.

Bug: create_cudagraphs() runs after finalize_model_grads, so main_grad already holds the finalized
(reduced + per-token-scaled) grads. Capturing the fwd warmup and the bwd graph *executes* GTP's
wgrad main_grad.add_ while recording it -- including the cascade add into a param's cross-graph
``next_w`` (which lives in another module) -- clobbering the finalized grads and spiking the step's
grad norm.

Fix: create_fwd_graph / create_bwd_graph snapshot the grads their capture touches via
``_backup_capture_grads`` and restore them after. This test exercises that helper pair directly:
the module's own params and their cross-graph ``next_w`` must survive a simulated capture clobber.
"""

import torch

from megatron.core.transformer.cuda_graphs import _backup_capture_grads, _restore_capture_grads


def _gtp_param(value: float, numel: int = 8) -> torch.nn.Parameter:
    """A param with a finalized (reduced + scaled) main_grad, flagged as a GTP weight."""
    p = torch.nn.Parameter(torch.zeros(numel, device="cuda"))
    p.is_gtp = True
    p.main_grad = torch.full((numel,), value, device="cuda")
    return p


class _Mod(torch.nn.Module):
    def __init__(self, weight: torch.nn.Parameter):
        super().__init__()
        self.weight = weight


class _StubRunner:
    """The ``base_module`` and ``gtp_remat`` attributes that ``_backup_capture_grads`` reads."""

    def __init__(self, base_module: torch.nn.Module, gtp_remat: bool = True):
        self.base_module = base_module
        self.gtp_remat = gtp_remat


class TestGTPCaptureGradSnapshot:
    def test_preserves_own_and_cross_graph_next_w(self):
        """Snapshot/restore must keep both the module's own grad and its cross-graph next_w grad
        (in another module) intact across a capture that clobbers them."""
        own = _gtp_param(0.0125)
        cross = _gtp_param(0.02)  # next_w lives in a different module/graph
        own.next_w = cross
        runner = _StubRunner(_Mod(own))

        backup = _backup_capture_grads(runner)
        own.main_grad.add_(410.0)  # simulate the capture-time main_grad.add_ clobber
        cross.main_grad.add_(99.0)
        _restore_capture_grads(backup)

        torch.testing.assert_close(own.main_grad, torch.full((8,), 0.0125, device="cuda"))
        torch.testing.assert_close(cross.main_grad, torch.full((8,), 0.02, device="cuda"))

    def test_routed_expert_next_w_via_weight_list(self):
        """A routed-expert next_w exposes its grad-bearing shards via ``weight_list`` (read directly,
        since the ``_weights`` property raises on non-leaders before capture)."""
        own = _gtp_param(0.0125)
        shard0, shard1 = _gtp_param(0.03), _gtp_param(0.04)
        routed = torch.nn.Parameter(torch.zeros(8, device="cuda"))  # leader wrapper (no own grad)
        routed.is_routed_expert = True
        routed.weight_list = [shard0, shard1]
        own.next_w = routed
        runner = _StubRunner(_Mod(own))

        backup = _backup_capture_grads(runner)
        shard0.main_grad.add_(50.0)
        shard1.main_grad.add_(60.0)
        _restore_capture_grads(backup)

        torch.testing.assert_close(shard0.main_grad, torch.full((8,), 0.03, device="cuda"))
        torch.testing.assert_close(shard1.main_grad, torch.full((8,), 0.04, device="cuda"))

    def test_non_gtp_backs_up_own_params_only(self):
        """Non-GTP runner: own params are snapshotted, but the GTP cross-graph next_w walk is
        skipped (the bwd capture doesn't touch main_grad on the non-GTP path)."""
        own = _gtp_param(0.0125)
        cross = _gtp_param(0.02)
        own.next_w = cross
        backup = _backup_capture_grads(_StubRunner(_Mod(own), gtp_remat=False))
        assert id(own) in backup
        assert id(cross) not in backup
