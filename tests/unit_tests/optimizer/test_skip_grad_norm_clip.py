# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Targeted tests for the Muon skip-grad-norm-clip PR (#5395, revision 937d8677d).

Covers the four reviewer findings:
  B1 (Bug1, critical): skip_grad_norm_clip is set on the *bf16 wrapper* members of
      LayerWiseDistributedOptimizer.chained_optimizers (not only the raw sub), for the
      results-empty (Muon-only) DIRECT path; Adam subs in a mix stay unflagged.
  B2 (Bug2, high): the flag is gated on isinstance(opt, OrthogonalizedOptimizer) so only
      the Muon family is flagged; SOAP/Lion keep clipping.
  B3 (Bug3, medium): ChainedOptimizer._get_grad_norm_skip_threshold() excludes flagged
      subs' grads so a Muon sub's huge grad cannot trip the skip threshold for an Adam sub.
  B4 (Bug4, low): should_clip is False for a Muon-only chain, so _compute_grad_norms_by_group
      (and its AllReduce) is not run; a clippable Adam chain still runs it.

Run: torchrun --nproc_per_node=2 -m pytest tests/unit_tests/optimizer/test_skip_grad_norm_clip.py
     with NVIDIA_PYTORCH_VERSION>25.05 set.
"""
import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging.version import Version

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.emerging_optimizers import HAVE_EMERGING_OPTIMIZERS
from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
from megatron.core.optimizer.optimizer import ChainedOptimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

if HAVE_EMERGING_OPTIMIZERS:
    from emerging_optimizers.orthogonalized_optimizers import OrthogonalizedOptimizer

pytestmark = [
    pytest.mark.skipif(
        Version(os.getenv('NVIDIA_PYTORCH_VERSION', "24.01")) <= Version("25.05"),
        reason="Skip emerging optimizer tests for LTS test",
    ),
    pytest.mark.skipif(
        not HAVE_EMERGING_OPTIMIZERS, reason="emerging_optimizers package is not installed"
    ),
    pytest.mark.skipif(
        int(os.getenv('WORLD_SIZE', '1')) == 1, reason="Multi-rank test requires WORLD_SIZE > 1"
    ),
]


class MuonOnly(nn.Module):
    """All-2D-weights (bias=False) so every managed param goes to Muon -> results empty."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 48, bias=False)
        self.fc2 = nn.Linear(48, 32, bias=False)
        self.fc3 = nn.Linear(32, 16, bias=False)

    def forward(self, x):
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))


class MuonAdamMix(nn.Module):
    """2D weights -> Muon, 1D biases -> Adam (a single bare LayerWise chain of [muon, adam])."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 48, bias=True)
        self.fc2 = nn.Linear(48, 16, bias=True)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def _inner(sub):
    """Unwrap a chained member to the actual torch optimizer (wrapper.optimizer or itself)."""
    raw = getattr(sub, 'optimizer', sub)
    return getattr(raw, 'optimizer', raw)


def _is_orthogonalizing(sub):
    return isinstance(_inner(sub), OrthogonalizedOptimizer)


@pytest.mark.skipif(
    int(os.getenv('WORLD_SIZE', '1')) == 1, reason="Multi-rank test requires WORLD_SIZE > 1"
)
class TestSkipGradNormClip:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        Utils.initialize_model_parallel()
        yield
        Utils.destroy_model_parallel()

    # ---- builders -----------------------------------------------------------------
    def _build(self, model, optimizer_name, use_layer_wise, clip_grad=1.0):
        model = model.bfloat16().cuda()
        model.requires_grad_(True)
        ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
        model = DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
        )
        model.broadcast_params()
        cfg = OptimizerConfig(
            optimizer=optimizer_name,
            lr=0.01,
            weight_decay=0.01,
            bf16=True,
            use_distributed_optimizer=False,
            clip_grad=clip_grad,
            muon_tp_mode="duplicated",
            use_layer_wise_distributed_optimizer=use_layer_wise,
        )
        pg = ProcessGroupCollection.use_mpu_process_groups()
        pg.dp_cp = parallel_state.get_data_parallel_group(with_context_parallel=True)
        pg.expt_dp = parallel_state.get_expert_data_parallel_group()
        opt = get_megatron_optimizer(cfg, [model], pg_collection=pg, use_gloo_process_groups=False)
        return model, opt

    @staticmethod
    def _forward_backward(model, batch=8, in_dim=64):
        x = torch.randn(batch, in_dim, dtype=torch.bfloat16, device='cuda')
        model(x).sum().backward()

    # ================================ B1 (Bug1) ================================
    def test_b1_muon_only_every_member_flagged(self):
        """Muon-only -> bare LayerWiseDistributedOptimizer; EVERY chained member (the bf16
        wrapper itself) and the container carry skip_grad_norm_clip is True."""
        _, opt = self._build(MuonOnly(), 'muon', use_layer_wise=True)
        assert isinstance(opt, LayerWiseDistributedOptimizer)
        # The DIRECT path returns the container; the container flag must be True (all-orthog).
        assert getattr(opt, 'skip_grad_norm_clip', False) is True
        assert len(opt.chained_optimizers) >= 1
        for i, sub in enumerate(opt.chained_optimizers):
            # bf16 re-wraps base optimizers in Float16OptimizerWithFloat16Params; the flag must
            # be visible on the WRAPPER (not only the raw sub it forwards to).
            assert getattr(sub, 'skip_grad_norm_clip', False) is True, (
                f"member {i} ({type(sub).__name__}) wrapping {type(_inner(sub)).__name__} "
                f"is not flagged"
            )
            assert _is_orthogonalizing(sub), f"member {i} should be a Muon-family optimizer"

    def test_b1_mix_adam_sub_not_flagged(self):
        """Muon+Adam mix -> the Adam sub must NOT be flagged; the Muon sub must be flagged;
        the container must NOT be flagged (not all base subs are orthogonalizing)."""
        _, opt = self._build(MuonAdamMix(), 'muon', use_layer_wise=True)
        assert isinstance(opt, ChainedOptimizer)
        flagged = {i: getattr(s, 'skip_grad_norm_clip', False) for i, s in enumerate(opt.chained_optimizers)}
        orthog = {i: _is_orthogonalizing(s) for i, s in enumerate(opt.chained_optimizers)}
        # exactly the orthogonalizing (Muon) subs are flagged
        for i, s in enumerate(opt.chained_optimizers):
            assert bool(flagged[i]) == orthog[i], (
                f"member {i} ({type(_inner(s)).__name__}): flagged={flagged[i]} orthog={orthog[i]}"
            )
        assert any(orthog.values()), "expected a Muon sub"
        assert not all(orthog.values()), "expected an Adam sub in the mix"
        # container must not claim skip when a non-orthogonalizing sub is present
        assert getattr(opt, 'skip_grad_norm_clip', False) is False

    # ================================ B2 (Bug2) ================================
    def test_b2_muon_flagged_lion_not(self):
        """isinstance(OrthogonalizedOptimizer) gate: muon flagged, lion (scalar) not flagged."""
        _, muon_opt = self._build(nn.Linear(64, 32, bias=False), 'muon', use_layer_wise=False)
        assert any(getattr(s, 'skip_grad_norm_clip', False) for s in muon_opt.chained_optimizers)
        for s in muon_opt.chained_optimizers:
            assert getattr(s, 'skip_grad_norm_clip', False) == _is_orthogonalizing(s)

        _, lion_opt = self._build(nn.Linear(64, 32, bias=False), 'lion', use_layer_wise=False)
        for s in lion_opt.chained_optimizers:
            assert getattr(s, 'skip_grad_norm_clip', False) is False, "Lion must keep clipping"
            assert not _is_orthogonalizing(s)

    # ================================ B3 (Bug3) ================================
    def test_b3_threshold_excludes_muon_grad(self):
        """_get_grad_norm_skip_threshold() excludes the flagged (Muon) sub's huge grad, so a
        well-behaved Adam sub is not wrongly skipped. Contrast with get_grad_norm() (full)."""
        model, opt = self._build(MuonAdamMix(), 'muon', use_layer_wise=True, clip_grad=1.0)
        assert isinstance(opt, ChainedOptimizer)
        # find the Adam sub and give it a finite skip threshold
        adam_subs = [s for s in opt.chained_optimizers if not _is_orthogonalizing(s)]
        muon_subs = [s for s in opt.chained_optimizers if _is_orthogonalizing(s)]
        assert adam_subs and muon_subs
        for s in adam_subs:
            s.config.grad_norm_skip_threshold = 10.0

        # populate grads, then make the Muon (2D) grads huge and the Adam (1D) grads tiny
        # we reach the model via the optimizer's param groups
        self._forward_backward(model)
        for p in model.parameters():
            g = p.main_grad if getattr(p, 'main_grad', None) is not None else p.grad
            if g is None:
                continue
            if p.dim() >= 2:
                g.fill_(1.0e5)   # Muon sub -> huge norm
            else:
                g.fill_(1.0e-4)  # Adam sub -> tiny norm

        opt.prepare_grads()
        full = float(opt.get_grad_norm())
        threshold_norm = float(opt._get_grad_norm_skip_threshold())
        if Utils.rank == 0:
            print(f"\n[B3] get_grad_norm()={full:.3e}  _get_grad_norm_skip_threshold()={threshold_norm:.3e}")
        # the skip-threshold norm must be far smaller than the full norm (Muon grad excluded)
        assert threshold_norm < full
        # and below the Adam sub's finite threshold (10) so the update is NOT skipped,
        # whereas the full global norm would have tripped it.
        assert threshold_norm < 10.0 < full

    def test_b3_update_not_skipped(self):
        """End-to-end: with a huge Muon grad and a finite Adam threshold, step() must NOT skip."""
        model, opt = self._build(MuonAdamMix(), 'muon', use_layer_wise=True, clip_grad=1.0)
        for s in opt.chained_optimizers:
            if not _is_orthogonalizing(s):
                s.config.grad_norm_skip_threshold = 10.0
        self._forward_backward(model)
        for p in model.parameters():
            g = p.main_grad if getattr(p, 'main_grad', None) is not None else p.grad
            if g is None:
                continue
            g.fill_(1.0e5 if p.dim() >= 2 else 1.0e-4)
        update_successful, grad_norm, _ = opt.step()
        assert update_successful is True, "update was wrongly skipped despite small non-Muon norm"

    # ================================ B4 (Bug4) ================================
    def test_b4_muon_only_skips_clip_norm_compute(self):
        """Muon-only chain: should_clip is False -> _compute_grad_norms_by_group not called."""
        model, opt = self._build(MuonOnly(), 'muon', use_layer_wise=True, clip_grad=1.0)
        self._forward_backward(model)
        calls = {'n': 0}
        orig = opt._compute_grad_norms_by_group

        def counting(*a, **k):
            calls['n'] += 1
            return orig(*a, **k)

        opt._compute_grad_norms_by_group = counting
        update_successful, _, _ = opt.step()
        assert calls['n'] == 0, "Muon-only chain should not compute per-group clip norms"
        assert update_successful is True

    def test_b4_adam_chain_runs_clip_norm_compute(self):
        """Clippable Adam chain: should_clip is True -> _compute_grad_norms_by_group is called."""
        model, opt = self._build(nn.Linear(64, 32, bias=True), 'adam', use_layer_wise=False, clip_grad=1.0)
        self._forward_backward(model)
        calls = {'n': 0}
        orig = opt._compute_grad_norms_by_group

        def counting(*a, **k):
            calls['n'] += 1
            return orig(*a, **k)

        opt._compute_grad_norms_by_group = counting
        opt.step()
        assert calls['n'] >= 1, "clippable Adam chain must compute clip norms"

    # ===== distributed-optimizer path: step() must succeed (was the f207dc2-fixed regression) =====
    def test_distopt_path_step_succeeds(self):
        """Muon LayerWise chained with an Adam DistributedOptimizer => non-shared grad-stats
        groups. _get_grad_norm_skip_threshold() must handle that (per-sub fallback) instead of
        asserting a shared group. Regressed in 937d8677d, fixed in f207dc2."""
        from megatron.training.training import wrap_model_chunks_with_ddp

        model = MuonAdamMix().bfloat16().cuda()
        model.requires_grad_(True)
        ddp_config = DistributedDataParallelConfig()  # use_distributed_optimizer=True (default)
        model = wrap_model_chunks_with_ddp(
            [model],
            TransformerConfig(num_attention_heads=1, num_layers=1),
            ddp_config,
            use_layer_wise_distributed_optimizer=True,
        )[0]
        cfg = OptimizerConfig(
            optimizer='muon', lr=0.01, weight_decay=0.01, bf16=True, clip_grad=1.0,
            muon_tp_mode="duplicated", use_layer_wise_distributed_optimizer=True,
        )
        pg = ProcessGroupCollection.use_mpu_process_groups()
        pg.dp_cp = parallel_state.get_data_parallel_group(with_context_parallel=True)
        pg.expt_dp = parallel_state.get_expert_data_parallel_group()
        opt = get_megatron_optimizer(cfg, [model], pg_collection=pg, use_gloo_process_groups=False)
        self._forward_backward(model)
        update_successful, _, _ = opt.step()
        assert update_successful is True
