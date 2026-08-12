# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""Waveform audio processor for the NeMo Transformer audio frontend.

``NemoAudioProcessor`` is the concrete, data-side feature extractor that the
multimodal data pipeline uses to (a) estimate how many encoder/projector
embeddings an audio clip expands to (for placeholder expansion and packing) and
(b) materialize log-mel features for the audio encoder. It composes the
model-frontend descriptors (``NemoAudioFeatureConfig`` and
``NemoTransformerAudioTokenEstimator``) with the vendored standalone log-mel
preprocessor.

The data pipeline depends on this only through a small duck-typed interface
(``compute_num_embeddings`` / ``compute_num_frames`` / ``materialize`` plus the
cumulative-prefix ``num_*_from_num_samples`` primitives), so an ``audio_ref`` is
treated structurally — this module has no dependency on the data library's
``AudioRef`` type.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from megatron.core.models.audio.audio_feature_config import (
    NemoAudioFeatureConfig,
    NemoTransformerAudioTokenEstimator,
)

_AUDIO_DURATION_MISMATCH_TOLERANCE_SECONDS = 0.5


def _load_waveform_from_spec(audio_spec: dict[str, Any]) -> tuple[torch.Tensor, int | None]:
    kind = audio_spec.get("kind")
    if kind == "avdecoder":
        return _decode_avdecoder(
            audio_spec["decoder"],
            audio_spec.get("source_name", "<avdecoder>"),
            sample_rate=audio_spec.get("sample_rate") or audio_spec.get("sampling_rate"),
        )
    raise ValueError(f"Unsupported audio kind {kind!r}")


def _resolve_lazy_media(media: Any) -> Any:
    if hasattr(media, "get") and not hasattr(media, "get_audio"):
        media = media.get()
    if isinstance(media, (list, tuple)):
        if not media:
            raise ValueError("Lazy audio media resolved to an empty sequence.")
        media = media[0]
    return media


def _audio_clip_to_float32(clip: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(clip):
        clip = torch.as_tensor(clip)
    if clip.ndim == 1:
        clip = clip.unsqueeze(0)
    elif clip.ndim != 2:
        raise ValueError(f"Unsupported decoded audio clip shape {tuple(clip.shape)}.")

    if clip.dtype.is_floating_point:
        return clip.to(torch.float32).contiguous()

    if clip.dtype == torch.uint8:
        return ((clip.to(torch.float32) - 128.0) / 128.0).contiguous()

    if clip.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        scale = float(torch.iinfo(clip.dtype).max)
        return (clip.to(torch.float32) / scale).contiguous()

    raise ValueError(f"Unsupported decoded audio dtype {clip.dtype}.")


def _decoder_sample_rate(decoder: Any, sample_rate: int | None) -> int | None:
    if sample_rate is not None:
        return int(sample_rate)

    if hasattr(decoder, "get_audio_samples_per_second"):
        return int(decoder.get_audio_samples_per_second())

    if hasattr(decoder, "get_metadata"):
        metadata = decoder.get_metadata(
            get_video=False,
            get_video_duration=False,
            get_video_frame_count=False,
            get_video_frame_size=False,
            get_audio=True,
            get_audio_duration=False,
        )
        audio_sample_rate = getattr(metadata, "audio_sample_rate", None)
        if audio_sample_rate is not None:
            return int(audio_sample_rate)

    return None


def _decode_avdecoder(
    decoder: Any, source_name: str, *, sample_rate: int | None = None
) -> tuple[torch.Tensor, int | None]:
    decoder = _resolve_lazy_media(decoder)
    if not hasattr(decoder, "get_audio"):
        raise ValueError(
            f"Expected AVDecoder-like audio media for {source_name}, "
            f"got {type(decoder).__name__}."
        )

    av_data = decoder.get_audio()
    clips = getattr(av_data, "audio_clips", None)
    if not clips:
        raise ValueError(f"Decoded audio {source_name!r} did not contain audio clips.")

    waveform = torch.cat([_audio_clip_to_float32(clip) for clip in clips], dim=-1)
    return waveform.contiguous(), _decoder_sample_rate(decoder, sample_rate)


def _resolve_sample_rate(audio_ref: Any, decoded_sample_rate: int | None) -> int | None:
    sample_rate = audio_ref.sample_rate
    if sample_rate is None:
        sample_rate = decoded_sample_rate
    if sample_rate is None and isinstance(audio_ref.data, dict):
        sample_rate = audio_ref.data.get("sample_rate") or audio_ref.data.get("sampling_rate")
    if sample_rate is None:
        return None
    return int(sample_rate)


def _audio_num_sample_tolerance(audio_ref: Any, decoded_sample_rate: int | None) -> int:
    sample_rate = _resolve_sample_rate(audio_ref, decoded_sample_rate)
    if sample_rate is None:
        return 0
    return int(_AUDIO_DURATION_MISMATCH_TOLERANCE_SECONDS * sample_rate + 0.999999)


def _normalize_mono_waveform(audio_ref: Any) -> tuple[torch.Tensor, int | None]:
    data = audio_ref.data
    decoded_sample_rate = None
    if torch.is_tensor(data):
        waveform = data
    elif isinstance(data, dict):
        waveform, decoded_sample_rate = _load_waveform_from_spec(data)
    else:
        raise ValueError(
            "AudioRef.data must be a raw float32 waveform tensor or a supported lazy "
            "audio spec for the Megatron multimodal audio path."
        )
    if waveform.dtype != torch.float32:
        raise ValueError(f"Expected raw float32 waveform tensor, got {waveform.dtype}.")

    if waveform.ndim == 1:
        pass
    elif waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    else:
        raise ValueError(
            f"Unsupported waveform shape {tuple(waveform.shape)}. Expected [T] or [C, T]."
        )

    # First reconcile the decoded waveform to the full source length (num_samples
    # always counts the un-sliced source), then crop to slice_range if set.
    if audio_ref.num_samples is not None:
        num_samples = int(audio_ref.num_samples)
        available_num_samples = int(waveform.shape[0])
        diff = num_samples - available_num_samples
        tolerance = _audio_num_sample_tolerance(audio_ref, decoded_sample_rate)
        if diff > tolerance:
            raise ValueError(
                f"AudioRef.num_samples={num_samples} exceeds waveform length "
                f"{available_num_samples} by {diff} samples, which is greater than "
                f"the allowed tolerance {tolerance}."
            )
        if diff > 0:
            waveform = F.pad(waveform, (0, diff))
        elif diff < 0:
            waveform = waveform[:num_samples]

    if audio_ref.slice_range is not None:
        start, end = int(audio_ref.slice_range[0]), int(audio_ref.slice_range[1])
        if start < 0 or end < start:
            raise ValueError(
                f"AudioRef.slice_range must satisfy 0 <= start <= end, got {(start, end)}"
            )
        waveform = waveform[start:end]

    return waveform.contiguous(), decoded_sample_rate


def _infer_num_samples(audio_ref: Any) -> int:
    # slice_range, when set, defines the effective length of this ref.
    if audio_ref.slice_range is not None:
        start, end = int(audio_ref.slice_range[0]), int(audio_ref.slice_range[1])
        return max(0, end - start)
    if audio_ref.num_samples is not None:
        return int(audio_ref.num_samples)

    data = audio_ref.data
    if torch.is_tensor(data):
        waveform = data
    elif isinstance(data, dict):
        waveform, _ = _load_waveform_from_spec(data)
    else:
        raise ValueError(
            "AudioRef.data must be a raw float32 waveform tensor or a supported lazy "
            "audio spec for the Megatron multimodal audio path."
        )
    if waveform.dtype != torch.float32:
        raise ValueError(f"Expected raw float32 waveform tensor, got {waveform.dtype}.")

    if waveform.ndim == 1:
        available_num_samples = int(waveform.shape[0])
    elif waveform.ndim == 2:
        available_num_samples = int(waveform.shape[-1])
    else:
        raise ValueError(
            f"Unsupported waveform shape {tuple(waveform.shape)}. Expected [T] or [C, T]."
        )

    return available_num_samples


class NemoAudioProcessor:
    """Waveform audio processor with a NeMo log-mel frontend."""

    def __init__(
        self,
        *,
        token_estimator: NemoTransformerAudioTokenEstimator,
        feature_config: NemoAudioFeatureConfig | None = None,
    ) -> None:
        # Lazy import keeps construction cheap and avoids importing the heavier
        # preprocessor module until a processor is actually built. The vendored
        # ``AudioToMelSpectrogramPreprocessor`` is a stdlib+PyTorch port of NeMo's
        # preprocessor; ``.eval()`` disables training-time dither and narrowband
        # augmentation (typical for a feature extractor inside the multimodal
        # pipeline; flip back via ``.train()`` if needed).
        from megatron.core.models.audio.nemo_audio_preprocessing import (
            AudioToMelSpectrogramPreprocessor,
        )

        self.token_estimator = token_estimator
        self.feature_config = feature_config or NemoAudioFeatureConfig()
        self._preprocessor = AudioToMelSpectrogramPreprocessor(
            **self.feature_config.to_nemo_kwargs()
        ).eval()
        # The vendored standalone AudioToMelSpectrogramPreprocessor exposes
        # win/hop lengths directly (no ``featurizer`` indirection).
        self._hop_length = int(self._preprocessor.hop_length)
        self._n_mels = int(self.feature_config.features)
        self._sample_rate = int(self.feature_config.sample_rate)

    @property
    def input_feature_dim(self) -> int:
        """Number of mel feature bins produced per frame (the encoder input dim)."""
        return self._n_mels

    @property
    def sample_rate(self) -> int:
        """Expected input waveform sample rate, in Hz."""
        return self._sample_rate

    def _validate_sample_rate(self, audio_ref: Any, decoded_sample_rate: int | None = None) -> None:
        sample_rate = audio_ref.sample_rate
        if sample_rate is None:
            sample_rate = decoded_sample_rate
        if sample_rate is not None and int(sample_rate) != self.sample_rate:
            raise ValueError(
                f"Expected audio sample rate {self.sample_rate}, got {sample_rate}. "
                "Resample raw waveforms to the encoder sample rate before packing."
            )

    def _compute_num_frames_from_num_samples(self, num_samples: int) -> int:
        if num_samples < 0:
            raise ValueError(f"num_samples must be >= 0, got {num_samples}")
        if num_samples == 0:
            return 0
        return int(num_samples // self._hop_length)

    def compute_num_frames(self, audio_ref: Any) -> int:
        """Number of feature frames the clip described by ``audio_ref`` expands to."""
        self._validate_sample_rate(audio_ref)
        return self._compute_num_frames_from_num_samples(_infer_num_samples(audio_ref))

    def num_frames_from_num_samples(self, num_samples: int) -> int:
        """Pure frame-count math for an audio prefix of ``num_samples`` samples.

        Slice math:
        ``frames_in([s, e)) = num_frames_from_num_samples(e) - num_frames_from_num_samples(s)``.
        """
        return self._compute_num_frames_from_num_samples(num_samples)

    def num_embeddings_from_num_samples(self, num_samples: int) -> int:
        """Pure embedding-count math for an audio prefix of ``num_samples`` samples.

        Slice math: ``embeds_in([s, e))`` =
        ``num_embeddings_from_num_samples(e) - num_embeddings_from_num_samples(s)``.
        """
        return self.token_estimator.estimate_from_num_frames(
            self._compute_num_frames_from_num_samples(num_samples)
        )

    def compute_num_embeddings(self, audio_ref: Any) -> int:
        """Number of encoder/projector embeddings the clip in ``audio_ref`` expands to."""
        self._validate_sample_rate(audio_ref)
        return self.token_estimator.estimate_from_num_frames(self.compute_num_frames(audio_ref))

    def materialize(self, audio_ref: Any) -> tuple[torch.Tensor, int]:
        """Decode ``audio_ref`` and return its ``(T, n_mels)`` log-mel features and frame count."""
        waveform, decoded_sample_rate = _normalize_mono_waveform(audio_ref)
        self._validate_sample_rate(audio_ref, decoded_sample_rate)

        num_samples = waveform.shape[0]
        num_frames = self._compute_num_frames_from_num_samples(num_samples)
        if num_frames == 0:
            return (
                torch.empty(
                    (0, self.input_feature_dim), dtype=torch.float32, device=waveform.device
                ),
                0,
            )

        batched = waveform.unsqueeze(0)
        lengths = torch.tensor([num_samples], dtype=torch.long, device=waveform.device)
        mels, out_lengths = self._preprocessor(batched, lengths)

        # mels: (1, n_mels, T_frames) -> (T_frames, n_mels), trimmed to valid frames.
        valid_frames = int(out_lengths[0].item())
        if valid_frames == 0:
            return (
                torch.empty(
                    (0, self.input_feature_dim), dtype=torch.float32, device=waveform.device
                ),
                0,
            )
        log_mel = mels[0, :, :valid_frames].transpose(0, 1).contiguous()
        return log_mel.to(torch.float32), valid_frames
