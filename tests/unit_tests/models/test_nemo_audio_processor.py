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
            encoder_time_stride=4, stack_factor=2, pre_encode="conv"
        ),
        feature_config=NemoAudioFeatureConfig(
            sample_rate=16000, window_stride=0.01, n_window_stride=None, dither=0.0
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
    "boundaries", [[0, 8000, 16000], [0, 3200, 12800, 28800, 64000, 100000, 160000]]
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
        data={"kind": "avdecoder"}, sample_rate=16000, num_samples=20, slice_range=(5, 9)
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


# ---------------------------------------------------------------------------
# Public methods / properties
# ---------------------------------------------------------------------------


def test_processor_properties():
    p = _processor()
    assert p.sample_rate == 16000
    assert p.input_feature_dim == p._n_mels


def test_compute_num_frames_and_embeddings_from_waveform_ref():
    p = _processor()
    audio = _audio_ref(data=torch.zeros(16000, dtype=torch.float32), sample_rate=16000)
    # 16000 samples @ hop=160 -> 100 frames -> 13 embeddings (see slice-primitive tests).
    assert p.compute_num_frames(audio) == 100
    assert p.compute_num_embeddings(audio) == 13


def test_validate_sample_rate_mismatch_raises():
    p = _processor()
    audio = _audio_ref(data=torch.zeros(16000, dtype=torch.float32), sample_rate=8000)
    with pytest.raises(ValueError, match="Expected audio sample rate 16000"):
        p.compute_num_frames(audio)


# ---------------------------------------------------------------------------
# Lazy AV-decoder decode chain (duck-typed decoder fakes)
# ---------------------------------------------------------------------------


class _FakeAVData:
    def __init__(self, clips):
        self.audio_clips = clips


class _FakeDecoder:
    """Minimal AVDecoder-like stand-in exposing get_audio + sample-rate probes."""

    def __init__(self, clips, samples_per_second=None):
        self._clips = clips
        self._samples_per_second = samples_per_second

    def get_audio(self):
        return _FakeAVData(self._clips)

    def get_audio_samples_per_second(self):
        return self._samples_per_second


def test_audio_clip_to_float32_1d_float_is_unsqueezed():
    out = audio_processor._audio_clip_to_float32(torch.tensor([0.1, 0.2], dtype=torch.float32))
    assert out.shape == (1, 2)
    assert out.dtype == torch.float32


def test_audio_clip_to_float32_from_python_list():
    out = audio_processor._audio_clip_to_float32([0.0, 1.0, -1.0])
    assert out.shape == (1, 3)
    assert out.dtype == torch.float32


def test_audio_clip_to_float32_uint8_is_centered_and_scaled():
    out = audio_processor._audio_clip_to_float32(torch.tensor([0, 128, 255], dtype=torch.uint8))
    assert torch.allclose(out[0], torch.tensor([-1.0, 0.0, (255 - 128) / 128.0]))


def test_audio_clip_to_float32_int16_is_scaled_by_max():
    out = audio_processor._audio_clip_to_float32(torch.tensor([0, 32767], dtype=torch.int16))
    assert torch.allclose(out[0], torch.tensor([0.0, 1.0]))


def test_audio_clip_to_float32_rejects_bad_ndim():
    with pytest.raises(ValueError, match="Unsupported decoded audio clip shape"):
        audio_processor._audio_clip_to_float32(torch.zeros(2, 2, 2))


def test_audio_clip_to_float32_rejects_bad_dtype():
    with pytest.raises(ValueError, match="Unsupported decoded audio dtype"):
        audio_processor._audio_clip_to_float32(torch.tensor([True, False]))


def test_decoder_sample_rate_prefers_explicit():
    assert audio_processor._decoder_sample_rate(object(), 22050) == 22050


def test_decoder_sample_rate_from_samples_per_second():
    dec = _FakeDecoder(clips=[], samples_per_second=16000)
    assert audio_processor._decoder_sample_rate(dec, None) == 16000


def test_decoder_sample_rate_from_metadata():
    class _MetaDecoder:
        def get_metadata(self, **kwargs):
            del kwargs
            return SimpleNamespace(audio_sample_rate=8000)

    assert audio_processor._decoder_sample_rate(_MetaDecoder(), None) == 8000


def test_decoder_sample_rate_none_when_unavailable():
    assert audio_processor._decoder_sample_rate(object(), None) is None


def test_resolve_lazy_media_unwraps_get_and_sequence():
    dec = _FakeDecoder(clips=[])
    lazy = SimpleNamespace(get=lambda: [dec])
    assert audio_processor._resolve_lazy_media(lazy) is dec


def test_resolve_lazy_media_empty_sequence_raises():
    with pytest.raises(ValueError, match="empty sequence"):
        audio_processor._resolve_lazy_media([])


def test_decode_avdecoder_concatenates_clips():
    dec = _FakeDecoder(
        clips=[
            torch.tensor([0.0, 1.0], dtype=torch.float32),
            torch.tensor([2.0], dtype=torch.float32),
        ],
        samples_per_second=16000,
    )
    waveform, sr = audio_processor._decode_avdecoder(dec, "<test>")
    assert sr == 16000
    assert waveform.shape == (1, 3)
    assert waveform[0].tolist() == [0.0, 1.0, 2.0]


def test_decode_avdecoder_rejects_non_decoder():
    with pytest.raises(ValueError, match="Expected AVDecoder-like"):
        audio_processor._decode_avdecoder(object(), "<test>")


def test_decode_avdecoder_rejects_missing_clips():
    with pytest.raises(ValueError, match="did not contain audio clips"):
        audio_processor._decode_avdecoder(_FakeDecoder(clips=[]), "<test>")


def test_load_waveform_from_spec_dispatches_avdecoder():
    dec = _FakeDecoder(clips=[torch.tensor([0.5], dtype=torch.float32)], samples_per_second=16000)
    waveform, sr = audio_processor._load_waveform_from_spec(
        {"kind": "avdecoder", "decoder": dec, "sample_rate": 16000}
    )
    assert sr == 16000
    assert waveform.shape == (1, 1)


def test_load_waveform_from_spec_rejects_unknown_kind():
    with pytest.raises(ValueError, match="Unsupported audio kind"):
        audio_processor._load_waveform_from_spec({"kind": "wav"})


# ---------------------------------------------------------------------------
# Sample-rate / tolerance resolution
# ---------------------------------------------------------------------------


def test_resolve_sample_rate_prefers_audio_ref():
    audio = _audio_ref(sample_rate=16000, data={})
    assert audio_processor._resolve_sample_rate(audio, 8000) == 16000


def test_resolve_sample_rate_falls_back_to_decoded():
    audio = _audio_ref(sample_rate=None, data={})
    assert audio_processor._resolve_sample_rate(audio, 8000) == 8000


def test_resolve_sample_rate_falls_back_to_data_dict():
    audio = _audio_ref(sample_rate=None, data={"sampling_rate": 22050})
    assert audio_processor._resolve_sample_rate(audio, None) == 22050


def test_resolve_sample_rate_none_when_unknown():
    audio = _audio_ref(sample_rate=None, data=torch.zeros(1))
    assert audio_processor._resolve_sample_rate(audio, None) is None


def test_audio_num_sample_tolerance():
    audio = _audio_ref(sample_rate=16000, data={})
    # ceil(0.5 * 16000) = 8000 samples of allowed drift.
    assert audio_processor._audio_num_sample_tolerance(audio, None) == 8000


def test_audio_num_sample_tolerance_zero_without_sample_rate():
    audio = _audio_ref(sample_rate=None, data=torch.zeros(1))
    assert audio_processor._audio_num_sample_tolerance(audio, None) == 0


# ---------------------------------------------------------------------------
# Waveform normalization branches
# ---------------------------------------------------------------------------


def test_normalize_stereo_is_averaged_to_mono():
    data = torch.tensor([[0.0, 2.0], [2.0, 4.0]], dtype=torch.float32)  # [C=2, T=2]
    audio = _audio_ref(data=data, sample_rate=16000)
    waveform, _ = audio_processor._normalize_mono_waveform(audio)
    assert waveform.tolist() == [1.0, 3.0]


def test_normalize_pads_up_to_num_samples_within_tolerance():
    audio = _audio_ref(data=torch.ones(10, dtype=torch.float32), sample_rate=16000, num_samples=13)
    waveform, _ = audio_processor._normalize_mono_waveform(audio)
    assert waveform.shape[0] == 13
    assert waveform[10:].tolist() == [0.0, 0.0, 0.0]


def test_normalize_crops_down_to_num_samples():
    audio = _audio_ref(data=torch.arange(10, dtype=torch.float32), sample_rate=16000, num_samples=4)
    waveform, _ = audio_processor._normalize_mono_waveform(audio)
    assert waveform.tolist() == [0.0, 1.0, 2.0, 3.0]


def test_normalize_rejects_num_samples_beyond_tolerance():
    audio = _audio_ref(
        data=torch.ones(10, dtype=torch.float32), sample_rate=16000, num_samples=9000
    )
    with pytest.raises(ValueError, match="exceeds waveform length"):
        audio_processor._normalize_mono_waveform(audio)


def test_normalize_rejects_non_float32():
    audio = _audio_ref(data=torch.zeros(10, dtype=torch.float64), sample_rate=16000)
    with pytest.raises(ValueError, match="Expected raw float32 waveform"):
        audio_processor._normalize_mono_waveform(audio)


def test_normalize_rejects_bad_ndim():
    audio = _audio_ref(data=torch.zeros(2, 2, 2, dtype=torch.float32), sample_rate=16000)
    with pytest.raises(ValueError, match="Unsupported waveform shape"):
        audio_processor._normalize_mono_waveform(audio)


def test_normalize_rejects_unsupported_data_type():
    audio = _audio_ref(data="not-a-waveform", sample_rate=16000)
    with pytest.raises(ValueError, match="must be a raw float32 waveform"):
        audio_processor._normalize_mono_waveform(audio)


def test_normalize_rejects_bad_slice_range():
    audio = _audio_ref(
        data=torch.ones(10, dtype=torch.float32), sample_rate=16000, slice_range=(5, 2)
    )
    with pytest.raises(ValueError, match="slice_range must satisfy"):
        audio_processor._normalize_mono_waveform(audio)


# ---------------------------------------------------------------------------
# _infer_num_samples branches
# ---------------------------------------------------------------------------


def test_infer_num_samples_from_num_samples_field():
    audio = _audio_ref(num_samples=1234, data=torch.zeros(1, dtype=torch.float32))
    assert audio_processor._infer_num_samples(audio) == 1234


def test_infer_num_samples_from_1d_tensor():
    audio = _audio_ref(data=torch.zeros(500, dtype=torch.float32))
    assert audio_processor._infer_num_samples(audio) == 500


def test_infer_num_samples_from_2d_tensor_uses_time_dim():
    audio = _audio_ref(data=torch.zeros(2, 640, dtype=torch.float32))
    assert audio_processor._infer_num_samples(audio) == 640


def test_infer_num_samples_from_spec():
    dec = _FakeDecoder(clips=[torch.zeros(320, dtype=torch.float32)], samples_per_second=16000)
    audio = _audio_ref(data={"kind": "avdecoder", "decoder": dec})
    assert audio_processor._infer_num_samples(audio) == 320


def test_infer_num_samples_rejects_unsupported_data():
    audio = _audio_ref(data=42)
    with pytest.raises(ValueError, match="must be a raw float32 waveform"):
        audio_processor._infer_num_samples(audio)


def test_infer_num_samples_rejects_non_float32():
    audio = _audio_ref(data=torch.zeros(10, dtype=torch.int32))
    with pytest.raises(ValueError, match="Expected raw float32 waveform"):
        audio_processor._infer_num_samples(audio)
