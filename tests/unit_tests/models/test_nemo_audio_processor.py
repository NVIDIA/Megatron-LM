# Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the data-side NeMo audio processor.

Covers the cumulative-prefix slice primitives
(``num_frames_from_num_samples`` / ``num_embeddings_from_num_samples``),
``slice_range`` waveform cropping, and log-mel materialization. ``audio_ref`` is
duck-typed (a tiny ``SimpleNamespace`` stand-in) — the processor never imports
the data library's ``AudioRef`` type.
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from megatron.core.models.audio import audio_processor
from megatron.core.models.audio.audio_feature_config import (
    NemoAudioFeatureConfig,
    NemoTransformerAudioTokenEstimator,
)
from megatron.core.models.audio.audio_processor import NemoAudioProcessor


def _audio_ref(**kwargs):
    fields = dict(
        data=None,
        sample_rate=None,
        num_samples=None,
        slice_range=None,
        num_frames=None,
        feature_dim=None,
    )
    fields.update(kwargs)
    return SimpleNamespace(**fields)


def _processor():
    return NemoAudioProcessor(
        token_estimator=NemoTransformerAudioTokenEstimator(
            encoder_time_stride=4,
            stack_factor=2,
            pre_encode="conv",
        ),
        feature_config=NemoAudioFeatureConfig(
            sample_rate=16000,
            window_stride=0.01,
            n_window_stride=None,
            dither=0.0,
        ),
    )


# ---------------------------------------------------------------------------
# Slice primitives
# ---------------------------------------------------------------------------


def test_num_frames_from_num_samples_zero():
    assert _processor().num_frames_from_num_samples(0) == 0


def test_num_frames_from_num_samples_one_second():
    # 16000 samples @ hop=160.
    assert _processor().num_frames_from_num_samples(16000) == 100


def test_num_embeddings_from_num_samples_one_second():
    # 100 frames -> floor(100 / 4) encoder steps -> ceil(25 / 2) embeddings.
    assert _processor().num_embeddings_from_num_samples(16000) == 13


def test_num_embeddings_from_num_samples_zero():
    assert _processor().num_embeddings_from_num_samples(0) == 0


@pytest.mark.parametrize(
    "boundaries",
    [
        [0, 8000, 16000],  # 0.5s, 1.0s halves
        [0, 3200, 12800, 28800, 64000, 100000, 160000],  # arbitrary 10s split
    ],
)
def test_slice_contributions_sum_to_full_embedding_count(boundaries):
    """Cumulative-prefix invariant: contributions of disjoint slices sum to the
    unsliced total. This is the property that justifies the primitives."""
    p = _processor()
    total = p.num_embeddings_from_num_samples(boundaries[-1])
    cum = sum(
        p.num_embeddings_from_num_samples(e) - p.num_embeddings_from_num_samples(s)
        for s, e in zip(boundaries[:-1], boundaries[1:])
    )
    assert cum == total


@pytest.mark.parametrize(
    "boundaries",
    [
        [0, 8000, 16000],
        [0, 3200, 12800, 28800, 64000, 100000, 160000],
    ],
)
def test_slice_contributions_sum_to_full_frame_count(boundaries):
    p = _processor()
    total = p.num_frames_from_num_samples(boundaries[-1])
    cum = sum(
        p.num_frames_from_num_samples(e) - p.num_frames_from_num_samples(s)
        for s, e in zip(boundaries[:-1], boundaries[1:])
    )
    assert cum == total


def test_independent_slice_lengths_drift_from_total():
    """The naive ``f(end - start)`` approach is provably WRONG: ceil divisions
    mean per-slice-length counts do not sum to the unsliced total. Use the
    subtraction (cumulative-prefix) contract, not addition."""
    p = _processor()
    assert p.num_embeddings_from_num_samples(16000) == 13
    assert p.num_embeddings_from_num_samples(8000) == 6
    # 6 + 6 != 13.
    assert p.num_embeddings_from_num_samples(8000) + p.num_embeddings_from_num_samples(8000) != 13


# ---------------------------------------------------------------------------
# Waveform normalization / slicing
# ---------------------------------------------------------------------------


def test_normalize_waveform_crops_to_slice_range(monkeypatch):
    waveform = torch.arange(20, dtype=torch.float32)

    def fake_load_waveform(audio_spec):
        del audio_spec
        return waveform, 16000

    monkeypatch.setattr(audio_processor, "_load_waveform_from_spec", fake_load_waveform)
    # num_samples is the full source length; slice_range carries the crop window.
    audio = _audio_ref(
        data={"kind": "avdecoder"},
        sample_rate=16000,
        num_samples=20,
        slice_range=(5, 9),
    )

    cropped, decoded_sample_rate = audio_processor._normalize_mono_waveform(audio)

    assert decoded_sample_rate == 16000
    assert cropped.tolist() == [5.0, 6.0, 7.0, 8.0]


def test_infer_num_samples_uses_slice_range_length():
    # slice_range defines the effective length even though num_samples is the full source.
    audio = _audio_ref(data={"kind": "avdecoder"}, num_samples=20, slice_range=(5, 9))
    assert audio_processor._infer_num_samples(audio) == 4


# ---------------------------------------------------------------------------
# Materialization
# ---------------------------------------------------------------------------


def test_materialize_returns_time_major_log_mel():
    p = _processor()
    waveform = torch.zeros(16000, dtype=torch.float32)
    audio = _audio_ref(data=waveform, sample_rate=16000, num_samples=16000)

    log_mel, valid_frames = p.materialize(audio)

    assert valid_frames == 100
    assert log_mel.shape == (100, p.input_feature_dim)
    assert log_mel.dtype == torch.float32


def test_materialize_empty_waveform_yields_no_frames():
    p = _processor()
    audio = _audio_ref(data=torch.zeros(0, dtype=torch.float32), sample_rate=16000, num_samples=0)
    log_mel, valid_frames = p.materialize(audio)
    assert valid_frames == 0
    assert log_mel.shape == (0, p.input_feature_dim)
