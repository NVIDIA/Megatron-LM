# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Emerging-optimizers version-gate tests for the layer-sharded SYRK dispatch.

The batched-SYRK capability is detected by symbol (``triton_kernels.batched_tsyrk_ex``,
emerging-optimizers >= 0.5.0a0), and installs without it must fall back to baddbmm
for batched chunks while the baseline per-matrix path stays untouched. CI only ever
installs one emerging-optimizers, so these tests pin BOTH sides of the gate by
simulating symbol presence/absence, independent of the installed version.

Pure single-process tests: no distributed init, no GPU.
"""

import pytest
import torch

pytest.importorskip("emerging_optimizers", reason="requires emerging-optimizers")

from megatron.core.optimizer import layer_sharded_muon as lsm
from megatron.core.optimizer.layer_sharded_muon import LayerShardedMuon, _check_eo_version

try:
    # LayerShardedMuon.__init__ raises on emerging-optimizers < 0.3.0; turn that into
    # a module-level skip (same guard as test_layer_sharded_muon.py). Wrapping the
    # module import instead would guard nothing: layer_sharded_muon swallows its own
    # emerging-optimizers import failures behind HAVE_EMERGING_OPTIMIZERS, so the
    # import always succeeds and the ImportError only surfaces at construction.
    _check_eo_version()
except ImportError as _e:
    pytest.skip(str(_e), allow_module_level=True)


def _make_opt(**kwargs):
    p = torch.nn.Parameter(torch.randn(4, 4))
    return LayerShardedMuon([p], lr=0.1, gtp_group=None, **kwargs)


class TestBatchedSyrkSymbolDetection:
    def test_absent_symbol_reports_false(self, monkeypatch):
        monkeypatch.delattr(lsm.triton_kernels, "batched_tsyrk_ex", raising=False)
        assert lsm._has_batched_syrk() is False

    def test_present_symbol_reports_true(self, monkeypatch):
        monkeypatch.setattr(lsm.triton_kernels, "batched_tsyrk_ex", object(), raising=False)
        assert lsm._has_batched_syrk() is True


class TestInitDerivesBatchedSyrk:
    @pytest.mark.parametrize("has_symbol", [False, True])
    def test_baseline_never_arms_batched_syrk(self, monkeypatch, has_symbol):
        """use_syrk=False must yield _batched_syrk=False on every emerging-optimizers."""
        monkeypatch.setattr(lsm, "_has_batched_syrk", lambda: has_symbol)
        opt = _make_opt(use_syrk=False, ns_batch_size=1)
        assert opt.use_syrk is False
        assert opt._batched_syrk is False

    @pytest.mark.parametrize("has_symbol,expected", [(False, False), (True, True)])
    def test_syrk_arms_batched_only_with_symbol(self, monkeypatch, has_symbol, expected):
        # Bypass the CUDA/Triton/SM environment gate: this test pins the
        # version-capability derivation, not the hardware validation.
        monkeypatch.setattr(lsm, "_resolve_use_syrk", lambda flag: flag)
        monkeypatch.setattr(lsm, "_has_batched_syrk", lambda: has_symbol)
        opt = _make_opt(use_syrk=True, ns_batch_size=4)
        assert opt.use_syrk is True
        assert opt._batched_syrk is expected


class TestRunNsDispatch:
    """_run_ns must pass use_syrk per chunk: batched chunks follow _batched_syrk,
    unbatched chunks follow use_syrk — on both sides of the capability gate."""

    def _record_ns_calls(self, monkeypatch):
        calls = []

        def fake_newton_schulz(x, **kwargs):
            calls.append((x.dim(), kwargs["use_syrk"]))
            return x

        monkeypatch.setattr(lsm, "newton_schulz", fake_newton_schulz)
        return calls

    def _mats(self):
        # Two same-shape matrices (batchable) + one odd shape (never batched).
        return {0: torch.randn(4, 4), 1: torch.randn(4, 4), 2: torch.randn(4, 6)}

    def test_baseline_stays_2d_gemm_on_both_versions(self, monkeypatch):
        calls = self._record_ns_calls(monkeypatch)
        opt = _make_opt(use_syrk=False, ns_batch_size=1)
        out = opt._run_ns(self._mats())
        assert calls == [(2, False)] * 3, "baseline must be per-matrix 2-D, use_syrk=False"
        assert set(out) == {0, 1, 2}

    def test_old_eo_batched_falls_back_to_baddbmm(self, monkeypatch):
        calls = self._record_ns_calls(monkeypatch)
        opt = _make_opt(use_syrk=False, ns_batch_size=2)
        opt.use_syrk = True  # as if _resolve_use_syrk passed on real hardware
        opt._batched_syrk = False  # emerging-optimizers without batched_tsyrk_ex
        opt._run_ns(self._mats())
        assert sorted(calls) == [
            (2, True),
            (3, False),
        ], "old EO: unbatched chunk keeps SYRK, batched chunk must drop to use_syrk=False"

    def test_new_eo_batched_uses_syrk(self, monkeypatch):
        calls = self._record_ns_calls(monkeypatch)
        opt = _make_opt(use_syrk=False, ns_batch_size=2)
        opt.use_syrk = True
        opt._batched_syrk = True  # emerging-optimizers with batched_tsyrk_ex
        opt._run_ns(self._mats())
        assert sorted(calls) == [(2, True), (3, True)]

    def test_batch_of_one_is_2d_even_with_batching_enabled(self, monkeypatch):
        """ns_batch_size>1 with nothing to batch must preserve unbatched numerics."""
        calls = self._record_ns_calls(monkeypatch)
        opt = _make_opt(use_syrk=False, ns_batch_size=8)
        opt._run_ns({0: torch.randn(4, 4), 1: torch.randn(4, 6)})
        assert calls == [(2, False)] * 2
