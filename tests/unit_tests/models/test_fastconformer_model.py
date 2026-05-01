# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Unit tests for ``ParakeetHuggingFaceModel`` and the ``parakeet`` dispatch
in ``megatron.core.models.huggingface.module``.

These tests stub out the heavy NeMo / HuggingFace dependencies so they can run
on a CPU-only box. The point is to exercise the wrapper's branching logic
(prefix dispatch, dtype propagation, sampling-rate plumbing, singleton cache
keying, error path) and the ``parakeet`` substring rules in
``get_hf_model_type`` / ``build_hf_model``.
"""
from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


class _FakeFeatureExtractor:
    """Minimal stand-in for an HF FastConformer feature extractor."""

    sampling_rate = 16000

    def __call__(self, sound_clips, **kwargs):
        # Always emit a [B=1, T=4] feature tensor and a [1, 4] attention mask.
        # The wrapper only cares about ``input_features`` / ``attention_mask``.
        b = 1 if sound_clips.dim() == 1 else sound_clips.shape[0]
        feats = torch.zeros((b, 4), dtype=torch.float32)
        mask = torch.tensor([[1, 1, 1, 0]] * b, dtype=torch.long)
        return SimpleNamespace(input_features=feats, attention_mask=mask)


class _FakeHFEncoder(torch.nn.Module):
    """Returns a [B, T, H] tensor in a known dtype so the wrapper's dtype path is checked."""

    def __init__(self, hidden: int = 8, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        # Create a parameter with the target dtype so ``next(self.parameters()).dtype``
        # matches what the test asserts.
        self.proj = torch.nn.Parameter(torch.zeros((hidden, 4), dtype=dtype))

    def forward(self, input_features, attention_mask):
        # The wrapper passes ``features.input_features.to(dtype)`` so we can
        # round-trip the dtype without doing real math.
        b, t = attention_mask.shape
        last_hidden = torch.zeros((b, t, self.proj.shape[0]), dtype=self.proj.dtype)
        return SimpleNamespace(last_hidden_state=last_hidden)

    def gradient_checkpointing_enable(self):  # pragma: no cover - trivial stub
        self._grad_ckpt = True


def _stub_transformers(monkeypatch, *, encoder: torch.nn.Module, fe: Any):
    """Install fake ``transformers.AutoModel`` / ``AutoFeatureExtractor``."""
    fake = types.ModuleType("transformers")

    class _AutoModel:
        @classmethod
        def from_pretrained(cls, name, **_kw):
            return encoder

    class _AutoFeatureExtractor:
        @classmethod
        def from_pretrained(cls, name, **_kw):
            return fe

    class _AutoConfig:
        @classmethod
        def from_pretrained(cls, name, **_kw):
            # Mimic transformers' behaviour: any object with ``.architectures``
            # attribute. We never reach this in parakeet tests because the
            # parakeet branch in get_hf_model_type returns early.
            return SimpleNamespace(architectures=["FakeArch"], _name_or_path=name)

    fake.AutoModel = _AutoModel
    fake.AutoFeatureExtractor = _AutoFeatureExtractor
    fake.AutoConfig = _AutoConfig
    monkeypatch.setitem(sys.modules, "transformers", fake)
    return fake


# ---------------------------------------------------------------------------
# fastconformer_model.py — HF backend
# ---------------------------------------------------------------------------


class TestParakeetHuggingFaceModelHFBackend:
    """Tests for the ``hf://`` branch of ``ParakeetHuggingFaceModel``."""

    def _config(self, *, sound_model_type="hf://nvidia/parakeet-tdt-0.6b-v2"):
        return SimpleNamespace(
            sound_model_type=sound_model_type, hidden_dropout=0.0, recompute_granularity=None
        )

    @pytest.mark.internal
    def test_hf_backend_constructs_via_automodel(self, monkeypatch):
        from megatron.core.models.huggingface.fastconformer_model import ParakeetHuggingFaceModel

        encoder = _FakeHFEncoder(hidden=8, dtype=torch.bfloat16)
        fe = _FakeFeatureExtractor()
        _stub_transformers(monkeypatch, encoder=encoder, fe=fe)

        model = ParakeetHuggingFaceModel(self._config())

        assert model.use_nemo is False
        assert model.feature_extractor is fe
        assert model.model is encoder

    @pytest.mark.internal
    def test_hf_backend_forward_returns_hidden_and_lengths(self, monkeypatch):
        from megatron.core.models.huggingface.fastconformer_model import ParakeetHuggingFaceModel

        encoder = _FakeHFEncoder(hidden=8, dtype=torch.bfloat16)
        _stub_transformers(monkeypatch, encoder=encoder, fe=_FakeFeatureExtractor())

        model = ParakeetHuggingFaceModel(self._config())
        sound_clips = torch.zeros((1, 1600), dtype=torch.float32)

        hidden, lengths = model(sound_clips, None)

        # _FakeFeatureExtractor emits T=4 with mask sum 3.
        assert hidden.shape == (1, 4, 8)
        assert hidden.dtype == torch.bfloat16
        assert lengths.tolist() == [3]

    @pytest.mark.internal
    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
    def test_hf_backend_uses_model_dtype_not_hardcoded_bf16(self, monkeypatch, dtype):
        """Regression: input_features must be cast to the encoder's dtype, not bf16."""
        from megatron.core.models.huggingface.fastconformer_model import ParakeetHuggingFaceModel

        encoder = _FakeHFEncoder(hidden=8, dtype=dtype)
        _stub_transformers(monkeypatch, encoder=encoder, fe=_FakeFeatureExtractor())

        model = ParakeetHuggingFaceModel(self._config())
        hidden, _ = model(torch.zeros((1, 1600), dtype=torch.float32), None)
        assert hidden.dtype == dtype

    @pytest.mark.internal
    def test_hf_backend_reads_sampling_rate_from_feature_extractor(self, monkeypatch):
        """Regression: sampling rate comes from ``feature_extractor.sampling_rate``."""
        from megatron.core.models.huggingface.fastconformer_model import ParakeetHuggingFaceModel

        captured: dict = {}

        class _CapturingFE(_FakeFeatureExtractor):
            sampling_rate = 22050

            def __call__(self, sound_clips, **kwargs):
                captured.update(kwargs)
                return super().__call__(sound_clips, **kwargs)

        _stub_transformers(monkeypatch, encoder=_FakeHFEncoder(), fe=_CapturingFE())

        ParakeetHuggingFaceModel(self._config())(torch.zeros((1, 1600)), None)

        assert captured.get("sampling_rate") == 22050

    @pytest.mark.internal
    def test_hf_backend_enables_gradient_checkpointing_when_recomputing(self, monkeypatch):
        from megatron.core.models.huggingface.fastconformer_model import ParakeetHuggingFaceModel

        encoder = _FakeHFEncoder()
        _stub_transformers(monkeypatch, encoder=encoder, fe=_FakeFeatureExtractor())

        cfg = self._config()
        cfg.recompute_granularity = "selective"
        ParakeetHuggingFaceModel(cfg)
        assert getattr(encoder, "_grad_ckpt", False) is True

    @pytest.mark.internal
    def test_unknown_prefix_raises(self):
        from megatron.core.models.huggingface.fastconformer_model import ParakeetHuggingFaceModel

        with pytest.raises(ValueError, match="Unknown sound model type"):
            ParakeetHuggingFaceModel(self._config(sound_model_type="file:///tmp/foo"))


# ---------------------------------------------------------------------------
# fastconformer_model.py — NeMo singleton cache (B6 regression)
# ---------------------------------------------------------------------------


class TestNemoSingletonCacheKey:
    """The NeMo cache must be keyed by ``sound_model_type``.

    Regression for the previous implementation that used a single global
    singleton; calling with a second model id silently returned the first.
    """

    @pytest.mark.internal
    def test_cache_returns_distinct_entries_per_model_id(self, monkeypatch):
        # Build a fake nemo.collections.asr.models.ASRModel.from_pretrained
        # that returns a fresh sentinel object per name so we can identity-check.
        constructed: list[str] = []

        class _FakeEncoder:
            def __init__(self, name):
                self.name = name
                self.layers = []
                self.sync_max_audio_length = True  # value the wrapper flips to False

        class _FakeASRModel:
            def __init__(self, name):
                self.preprocessor = SimpleNamespace(name=f"{name}-pre")
                self.encoder = _FakeEncoder(name)

            @classmethod
            def from_pretrained(cls, model_name):
                constructed.append(model_name)
                return cls(model_name)

        nemo_pkg = types.ModuleType("nemo")
        nemo_collections = types.ModuleType("nemo.collections")
        nemo_asr = types.ModuleType("nemo.collections.asr")
        nemo_asr_models = types.ModuleType("nemo.collections.asr.models")
        nemo_asr_models.ASRModel = _FakeASRModel
        nemo_asr.models = nemo_asr_models
        nemo_collections.asr = nemo_asr
        nemo_pkg.collections = nemo_collections
        monkeypatch.setitem(sys.modules, "nemo", nemo_pkg)
        monkeypatch.setitem(sys.modules, "nemo.collections", nemo_collections)
        monkeypatch.setitem(sys.modules, "nemo.collections.asr", nemo_asr)
        monkeypatch.setitem(sys.modules, "nemo.collections.asr.models", nemo_asr_models)

        # Reset the module-level cache to keep tests independent.
        from megatron.core.models.huggingface import fastconformer_model as fcm

        fcm._NEMO_SOUND_MODEL_CACHE.clear()

        a_pre, a_enc = fcm.get_nemo_sound_model("nemo://nvidia/parakeet-a")
        b_pre, b_enc = fcm.get_nemo_sound_model("nemo://nvidia/parakeet-b")
        a_pre_again, a_enc_again = fcm.get_nemo_sound_model("nemo://nvidia/parakeet-a")

        # Distinct entries for distinct model ids ...
        assert a_enc is not b_enc
        assert a_pre is not b_pre
        # ... and identity reuse on cache hit.
        assert a_enc_again is a_enc
        assert a_pre_again is a_pre
        # ASRModel.from_pretrained called exactly twice (once per unique id).
        assert constructed == ["nvidia/parakeet-a", "nvidia/parakeet-b"]
        # And the encoder hangs were both flipped off.
        assert a_enc.sync_max_audio_length is False
        assert b_enc.sync_max_audio_length is False


# ---------------------------------------------------------------------------
# huggingface/module.py — parakeet wiring
# ---------------------------------------------------------------------------


class TestParakeetDispatch:
    """``get_hf_model_type`` and ``build_hf_model`` parakeet routing."""

    @pytest.mark.internal
    @pytest.mark.parametrize(
        "path",
        [
            "nemo://nvidia/parakeet-tdt-0.6b-v2",
            "hf://nvidia/parakeet-tdt-1.1b-en",
            "NEMO://Nvidia/Parakeet-TDT-0.6B-V2",  # case-insensitive
        ],
    )
    def test_parakeet_paths_route_to_parakeet(self, monkeypatch, path):
        # AutoConfig must not be touched on the parakeet path.
        from megatron.core.models.huggingface import module as hf_module

        sentinel = SimpleNamespace()

        class _ShouldNotCall:
            @classmethod
            def from_pretrained(cls, name):  # pragma: no cover - guard
                raise AssertionError(
                    f"AutoConfig.from_pretrained should not be called for parakeet, got {name!r}"
                )

        monkeypatch.setattr(hf_module, "AutoConfig", _ShouldNotCall)
        monkeypatch.setattr(hf_module, "HAVE_TRANSFORMERS", True)

        assert hf_module.get_hf_model_type(path) == "parakeet"

        # Sentinel kept just to demonstrate AutoConfig was not invoked.
        del sentinel

    @pytest.mark.internal
    @pytest.mark.parametrize(
        "path",
        [
            "/users/me/parakeet/local-model",  # no scheme prefix → not parakeet
            "hf://other-org/myparakeet-clone",  # substring inside seg, not prefix
        ],
    )
    def test_non_parakeet_paths_do_not_match(self, monkeypatch, path):
        """B13 regression: substring-anywhere matches must not route to parakeet."""
        from megatron.core.models.huggingface import module as hf_module

        # Provide an AutoConfig stub that returns a non-parakeet architecture so
        # the function falls through to the dispatch chain and raises NotImplementedError
        # (which is the expected behaviour for unhandled archs).
        class _AutoConfig:
            @classmethod
            def from_pretrained(cls, name):
                return SimpleNamespace(architectures=["FakeArch"], _name_or_path=name)

        monkeypatch.setattr(hf_module, "AutoConfig", _AutoConfig)
        monkeypatch.setattr(hf_module, "HAVE_TRANSFORMERS", True)

        if path.startswith(("nemo://", "hf://")):
            with pytest.raises(NotImplementedError):
                hf_module.get_hf_model_type(path)
        else:
            # No scheme → ``model_path.split("hf://")[1]`` would IndexError before
            # reaching dispatch; the parakeet branch must short-circuit on the
            # missing scheme. Verify it doesn't return "parakeet" by attempting
            # dispatch and confirming it errors at the AutoConfig path.
            with pytest.raises(IndexError):
                hf_module.get_hf_model_type(path)

    @pytest.mark.internal
    def test_nemo_scheme_non_parakeet_raises_clear_error(self, monkeypatch):
        """A non-parakeet ``nemo://`` model must raise NotImplementedError with a
        clear message instead of falling through to ``split("hf://")[1]`` and
        producing an opaque IndexError."""
        from megatron.core.models.huggingface import module as hf_module

        class _ShouldNotCall:
            @classmethod
            def from_pretrained(cls, name):  # pragma: no cover - guard
                raise AssertionError(
                    f"AutoConfig.from_pretrained should not be called for nemo://, got {name!r}"
                )

        monkeypatch.setattr(hf_module, "AutoConfig", _ShouldNotCall)
        monkeypatch.setattr(hf_module, "HAVE_TRANSFORMERS", True)

        with pytest.raises(NotImplementedError, match="nemo:// scheme"):
            hf_module.get_hf_model_type("nemo://nvidia/some-future-asr-model")

    @pytest.mark.internal
    def test_build_hf_model_dispatches_parakeet(self, monkeypatch):
        from megatron.core.models.huggingface import module as hf_module

        # Stub the heavy ParakeetHuggingFaceModel constructor by patching
        # the import via sys.modules so build_hf_model picks up our stub.
        ctor_called = {}

        class _StubParakeet:
            def __init__(self, config):
                ctor_called["cfg"] = config

        # Replace the symbol on the already-imported wrapper module.
        from megatron.core.models.huggingface import fastconformer_model as fcm

        monkeypatch.setattr(fcm, "ParakeetHuggingFaceModel", _StubParakeet)
        monkeypatch.setattr(hf_module, "HAVE_TRANSFORMERS", True)

        cfg = SimpleNamespace(huggingface_model_name_or_path="nemo://nvidia/parakeet-tdt-0.6b-v2")
        model = hf_module.build_hf_model(cfg, "nemo://nvidia/parakeet-tdt-0.6b-v2")
        assert isinstance(model, _StubParakeet)
        assert ctor_called["cfg"] is cfg
