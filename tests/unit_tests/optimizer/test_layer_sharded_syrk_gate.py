# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Constructor guard tests for LayerShardedMuon's Newton-Schulz configuration.

``_validate_ns_config`` rejects, at construction, every configuration the installed
stack cannot run (``ns_batch_size < 1``, batched Newton-Schulz below emerging-optimizers
0.3.0, and for ``use_syrk``: no CUDA device, Triton < 3.4.0, an SM emerging-optimizers
has not validated, ``ns_batch_size > 1`` without the batched SYRK kernel) instead of
downgrading. CI installs one emerging-optimizers on one GPU type, so every condition is
simulated here, on both sides, independent of the installed stack. ``_run_ns`` then
forwards ``use_syrk`` unchanged to every chunk.

Pure single-process tests: no distributed init, no GPU.
"""

import pytest
import torch

pytest.importorskip("emerging_optimizers", reason="requires emerging-optimizers")

from megatron.core.optimizer import emerging_optimizers as eo_mod
from megatron.core.optimizer import layer_sharded_muon as lsm
from megatron.core.optimizer.layer_sharded_muon import LayerShardedMuon

# No version skip on purpose: these tests pin guard/dispatch LOGIC and never execute a
# real Newton-Schulz (newton_schulz is mocked, step() is never called), so they must
# run — and give CI signal — on any installed emerging-optimizers. Constructions with
# ns_batch_size > 1 bypass the batched-NS version floor via _make_opt(monkeypatch, ...).


def _make_opt(monkeypatch=None, **kwargs):
    if monkeypatch is not None:
        monkeypatch.setattr(lsm, "is_emerging_optimizers_min_version", lambda v: True)
        monkeypatch.setattr(eo_mod, "is_emerging_optimizers_min_version", lambda v: True)
    p = torch.nn.Parameter(torch.randn(4, 4))
    return LayerShardedMuon([p], lr=0.1, gtp_remat_group=None, **kwargs)


def _simulate_hardware(monkeypatch, sm=(9, 0), triton_340=True):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: sm)
    monkeypatch.setattr(lsm.triton_kernels, "HAS_TRITON_340", triton_340, raising=False)


class TestValidateNsConfig:
    def test_rejects_ns_batch_size_below_one(self):
        with pytest.raises(ValueError, match="ns_batch_size must be at least 1"):
            lsm._validate_ns_config(False, ns_batch_size=0)

    def test_rejects_batched_ns_on_old_emerging_optimizers(self, monkeypatch):
        asked = []

        def fake_min_version(version, check_equality=True):
            asked.append(version)
            return False

        monkeypatch.setattr(lsm, "is_emerging_optimizers_min_version", fake_min_version)
        with pytest.raises(ValueError, match="batched Newton-Schulz"):
            lsm._validate_ns_config(False, ns_batch_size=4)
        assert asked == [lsm._BATCHED_NS_MIN_EO_VERSION]

    def test_unbatched_ns_needs_no_version_check(self, monkeypatch):
        """ns_batch_size=1 uses the 2-D API every release ships: no floor consulted."""
        monkeypatch.setattr(lsm, "is_emerging_optimizers_min_version", lambda v: False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        lsm._validate_ns_config(False, ns_batch_size=1)

    def test_use_syrk_false_skips_the_syrk_checks(self, monkeypatch):
        monkeypatch.setattr(lsm, "is_emerging_optimizers_min_version", lambda v: True)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        lsm._validate_ns_config(False, ns_batch_size=32)

    def test_rejects_without_cuda_device(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(ValueError, match="CUDA device"):
            lsm._validate_ns_config(True, ns_batch_size=1)

    def test_rejects_old_triton(self, monkeypatch):
        _simulate_hardware(monkeypatch, triton_340=False)
        with pytest.raises(ValueError, match="Triton >= 3.4.0"):
            lsm._validate_ns_config(True, ns_batch_size=1)

    def test_rejects_unvalidated_sm(self, monkeypatch):
        _simulate_hardware(monkeypatch, sm=(12, 0))
        with pytest.raises(ValueError, match=r"SM \(12, 0\)"):
            lsm._validate_ns_config(True, ns_batch_size=1)

    @pytest.mark.parametrize("sm", [(8, 0), (9, 0), (10, 0), (10, 3)])
    def test_accepts_validated_sms(self, monkeypatch, sm):
        _simulate_hardware(monkeypatch, sm=sm)
        lsm._validate_ns_config(True, ns_batch_size=1)

    def test_rejects_batched_syrk_on_old_emerging_optimizers(self, monkeypatch):
        _simulate_hardware(monkeypatch)
        asked = []

        def fake_min_version(version, check_equality=True):
            asked.append(version)
            return version == lsm._BATCHED_NS_MIN_EO_VERSION  # 0.3.x: batched NS, no batched SYRK

        monkeypatch.setattr(lsm, "is_emerging_optimizers_min_version", fake_min_version)
        with pytest.raises(ValueError, match="batched SYRK kernel"):
            lsm._validate_ns_config(True, ns_batch_size=4)
        assert asked == [lsm._BATCHED_NS_MIN_EO_VERSION, lsm._BATCHED_SYRK_MIN_EO_VERSION]

    def test_unbatched_syrk_needs_no_batched_kernel(self, monkeypatch):
        """2-D SYRK predates the batched kernel: ns_batch_size=1 must not consult it."""
        _simulate_hardware(monkeypatch)
        monkeypatch.setattr(lsm, "is_emerging_optimizers_min_version", lambda v: False)
        lsm._validate_ns_config(True, ns_batch_size=1)

    def test_accepts_batched_syrk_on_new_emerging_optimizers(self, monkeypatch):
        _simulate_hardware(monkeypatch)
        monkeypatch.setattr(lsm, "is_emerging_optimizers_min_version", lambda v: True)
        lsm._validate_ns_config(True, ns_batch_size=4)

    def test_constructor_runs_the_validation(self, monkeypatch):
        """No CUDA in this process: use_syrk=True must be rejected at construction."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(ValueError, match="CUDA device"):
            _make_opt(monkeypatch, use_syrk=True)

    def test_hardware_check_precedes_parent_version_gate(self, monkeypatch):
        """A stack that cannot run SYRK at all reports that first: no emerging-optimizers
        upgrade would help, so the parent's version gate must not mask that message."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(eo_mod, "is_emerging_optimizers_min_version", lambda v: False)
        with pytest.raises(ValueError, match="CUDA device"):
            _make_opt(use_syrk=True)


class TestRunNsDispatch:
    """_run_ns forwards use_syrk unchanged to every chunk: on a validated stack the
    batched (3-D) chunks reach newton_schulz's batched SYRK dispatch."""

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

    def test_baseline_stays_2d_gemm(self, monkeypatch):
        calls = self._record_ns_calls(monkeypatch)
        opt = _make_opt(use_syrk=False, ns_batch_size=1)
        out = opt._run_ns(self._mats())
        assert calls == [(2, False)] * 3, "baseline must be per-matrix 2-D, use_syrk=False"
        assert set(out) == {0, 1, 2}

    def test_use_syrk_reaches_batched_and_unbatched_chunks(self, monkeypatch):
        calls = self._record_ns_calls(monkeypatch)
        opt = _make_opt(monkeypatch, use_syrk=False, ns_batch_size=2)
        opt.use_syrk = True  # as if _validate_ns_config had passed on a validated stack
        opt._run_ns(self._mats())
        assert sorted(calls) == [(2, True), (3, True)]

    def test_batch_of_one_is_2d_even_with_batching_enabled(self, monkeypatch):
        """ns_batch_size>1 with nothing to batch must preserve unbatched numerics."""
        calls = self._record_ns_calls(monkeypatch)
        opt = _make_opt(monkeypatch, use_syrk=False, ns_batch_size=8)
        opt._run_ns({0: torch.randn(4, 4), 1: torch.randn(4, 6)})
        assert calls == [(2, False)] * 2


class TestConstructorGuards:
    def test_tp_mode_layer_sharded_is_rejected(self):
        """'layer_sharded' is the registry selector, not a class mode; direct-API
        misuse would otherwise fall silently into the parent's distributed
        branch — reject at construction like the other guards."""
        p = torch.nn.Parameter(torch.randn(4, 4))
        with pytest.raises(ValueError, match="registry-level selector"):
            LayerShardedMuon([p], lr=0.1, gtp_remat_group=None, tp_mode="layer_sharded")

    def test_split_qkv_is_rejected(self):
        """split_qkv would only apply on the fallback/degenerate paths, making
        the update rule depend on whether homes are set — reject at the class."""
        p = torch.nn.Parameter(torch.randn(4, 4))
        with pytest.raises(ValueError, match="split-QKV"):
            LayerShardedMuon([p], lr=0.1, gtp_remat_group=None, split_qkv=True)

    def test_ns_batch_size_below_one_is_rejected(self):
        """ns_batch_size is validated like the parent's num_ns_steps, not clamped."""
        p = torch.nn.Parameter(torch.randn(4, 4))
        with pytest.raises(ValueError, match="ns_batch_size"):
            LayerShardedMuon([p], lr=0.1, gtp_remat_group=None, ns_batch_size=0)
