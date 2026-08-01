# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Host-side log-mel front-end for the Parakeet audio tower.

The whole pipeline runs in fp32 -- pre-emphasis, STFT, mel projection, log, and normalization.
Downcasting anywhere before the tower changes the normalized features enough to shift token
content.

`subsampling_output_length` is the contract point: the host token count, the device-side
per-clip trim, and the dummy-input sizing all call it, and they must agree exactly or the
prompt's `<so_embedding>` slot count will not match the encoder's row count.
"""

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, List, Sequence, Tuple

import torch

from megatron.core.inference.multimodal.nemotron_omni.config import SoundConfig

# Log-domain floor. 2**-24, matching the reference implementation exactly; a different guard
# shifts every mel bin in near-silent frames.
LOG_ZERO_GUARD_VALUE = 2.0**-24
NORMALIZE_EPSILON = 1e-5


@dataclass
class ProcessedAudio:
    """Mel features for a batch of clips, grouped back to items by `clips_per_item`."""

    mel_features: torch.Tensor
    """`[total_clips, max_frames, num_mel_bins]`, normalized."""

    attention_mask: torch.Tensor
    """`[total_clips, max_frames]` bool, True on valid frames."""

    clips_per_item: List[int]
    """Clips belonging to each input audio item, in order."""

    token_counts: List[int]
    """Encoder row count per input audio item."""


@lru_cache(maxsize=8)
def _hann_window(win_length: int, device: str) -> torch.Tensor:
    """Symmetric (non-periodic) Hann window."""
    return torch.hann_window(win_length, periodic=False, device=device)


@lru_cache(maxsize=8)
def _mel_filters(num_mel_bins: int, sampling_rate: int, n_fft: int, device: str) -> torch.Tensor:
    """Slaney-normalized, Slaney-scaled mel filterbank, `[num_mel_bins, n_fft // 2 + 1]`.

    Built here rather than pulled from `transformers.audio_utils` so the audio front-end does
    not add a transformers import to the host path, and cross-checked against the filterbank
    vendored at `megatron/core/models/audio/nemo_audio_preprocessing.py`.
    """

    def hz_to_mel(freq: float) -> float:
        # Slaney scale: linear below 1 kHz, logarithmic above.
        min_log_hz = 1000.0
        min_log_mel = min_log_hz / (200.0 / 3)
        logstep = math.log(6.4) / 27.0
        if freq >= min_log_hz:
            return min_log_mel + math.log(freq / min_log_hz) / logstep
        return freq / (200.0 / 3)

    def mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
        min_log_hz = 1000.0
        min_log_mel = min_log_hz / (200.0 / 3)
        logstep = math.log(6.4) / 27.0
        linear = mel * (200.0 / 3)
        log_region = min_log_hz * torch.exp(logstep * (mel - min_log_mel))
        return torch.where(mel >= min_log_mel, log_region, linear)

    num_frequency_bins = n_fft // 2 + 1
    max_frequency = sampling_rate / 2.0
    mel_points = torch.linspace(
        hz_to_mel(0.0), hz_to_mel(max_frequency), num_mel_bins + 2, dtype=torch.float64
    )
    hz_points = mel_to_hz(mel_points)
    fft_freqs = torch.linspace(0.0, max_frequency, num_frequency_bins, dtype=torch.float64)

    filters = torch.zeros(num_mel_bins, num_frequency_bins, dtype=torch.float64)
    diff = hz_points[1:] - hz_points[:-1]
    ramps = hz_points.unsqueeze(-1) - fft_freqs.unsqueeze(0)
    for i in range(num_mel_bins):
        lower = -ramps[i] / diff[i]
        upper = ramps[i + 2] / diff[i + 1]
        filters[i] = torch.clamp(torch.minimum(lower, upper), min=0.0)

    # Slaney normalization: unit area per filter.
    enorm = 2.0 / (hz_points[2 : num_mel_bins + 2] - hz_points[:num_mel_bins])
    filters *= enorm.unsqueeze(-1)

    return filters.to(device=device, dtype=torch.float32)


class ParakeetAudioProcessor:
    """Splits audio into clips and produces normalized log-mel features."""

    def __init__(self, config: SoundConfig) -> None:
        self.config = config
        self._clip_target_samples = int(round(config.clip_seconds * config.sampling_rate))
        # A trailing fragment shorter than this is padded up rather than dropped, so a 30.05 s
        # input still produces two clips.
        self._tail_min_samples = int(round(0.1 * config.sampling_rate))

    def subsampling_output_length(self, num_frames: int) -> int:
        """Encoder rows produced from `num_frames` mel frames.

        The subsampling stack is `log2(subsampling_factor)` strided convolutions, each with
        stride 2, kernel `subsampling_conv_kernel_size`, and symmetric padding
        `(kernel - 1) // 2`. Applied sequentially, since flooring at each layer is not the same
        as one division by the total factor.
        """
        cfg = self.config
        kernel = cfg.subsampling_conv_kernel_size
        padding = (kernel - 1) // 2
        num_layers = int(round(math.log2(cfg.subsampling_factor)))

        length = num_frames
        for _ in range(num_layers):
            length = (length + 2 * padding - kernel) // 2 + 1
        return max(int(length), 0)

    def clip_sizes(self, num_samples: int) -> List[int]:
        """Split an audio length into 30 s clips plus a padded-up tail."""
        num_samples = max(num_samples, self._tail_min_samples)
        num_full_clips, remainder = divmod(num_samples, self._clip_target_samples)
        sizes = [self._clip_target_samples] * num_full_clips
        if remainder > 0:
            sizes.append(max(remainder, self._tail_min_samples))
        return sizes

    def token_count(self, num_samples: int) -> int:
        """Encoder rows an audio item of `num_samples` will produce.

        Floored at 1: an audio item always occupies at least one placeholder slot, so a
        near-silent fragment cannot produce a zero-length span.
        """
        total = 0
        for clip_size in self.clip_sizes(num_samples):
            total += self.subsampling_output_length(clip_size // self.config.hop_length)
        return max(1, total)

    def _split(self, waveform: torch.Tensor) -> List[torch.Tensor]:
        """Cut one waveform into clips, right-padding the tail to its nominal size."""
        assert waveform.ndim == 1, f"expected mono 1-D audio, got shape {tuple(waveform.shape)}"
        sizes = self.clip_sizes(int(waveform.shape[0]))
        target_len = sum(sizes)
        if waveform.shape[0] < target_len:
            waveform = torch.nn.functional.pad(waveform, (0, target_len - waveform.shape[0]))

        clips = []
        offset = 0
        for size in sizes:
            clips.append(waveform[offset : offset + size])
            offset += size
        return clips

    def _preemphasis(self, waveforms: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """First-order high-pass, `x[t] - 0.97 * x[t-1]`, zeroed past each clip's valid length.

        The masking matters: without it the padded tail leaks a non-zero sample at the
        boundary, which the STFT then spreads across the final frames.
        """
        time_mask = torch.arange(waveforms.shape[1], device=waveforms.device).unsqueeze(
            0
        ) < lengths.unsqueeze(1)
        emphasized = torch.cat(
            [waveforms[:, :1], waveforms[:, 1:] - self.config.preemphasis * waveforms[:, :-1]],
            dim=1,
        )
        return emphasized.masked_fill(~time_mask, 0.0)

    def _log_mel(self, waveforms: torch.Tensor) -> torch.Tensor:
        """STFT power spectrum through the mel filterbank, in the log domain."""
        cfg = self.config
        device = str(waveforms.device)
        stft = torch.stft(
            waveforms,
            cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            window=_hann_window(cfg.win_length, device),
            return_complex=True,
            pad_mode="constant",
        )
        magnitudes = stft.real.square() + stft.imag.square()
        mel_spec = _mel_filters(cfg.num_mel_bins, cfg.sampling_rate, cfg.n_fft, device) @ magnitudes
        return torch.log(mel_spec + LOG_ZERO_GUARD_VALUE).permute(0, 2, 1)

    def _normalize(
        self, mel_features: torch.Tensor, sample_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-clip mean/variance normalization over valid frames only.

        Uses the *sample* variance (divide by `n - 1`), not the population variance. The
        difference is small but systematic, and it propagates through the whole tower.
        """
        cfg = self.config
        # Frame count implied by torch.stft's centered padding.
        frame_lengths = torch.floor_divide(
            sample_lengths + (cfg.n_fft // 2) * 2 - cfg.n_fft, cfg.hop_length
        )
        attention_mask = (
            torch.arange(mel_features.shape[1], device=mel_features.device)[None, :]
            < frame_lengths[:, None]
        )
        mask = attention_mask.unsqueeze(-1)
        valid_counts = attention_mask.sum(dim=1)

        masked = mel_features * mask
        mean = (masked.sum(dim=1) / valid_counts.unsqueeze(-1)).unsqueeze(1)
        variance = ((masked - mean) ** 2 * mask).sum(dim=1) / (valid_counts - 1).unsqueeze(-1)
        std = torch.sqrt(variance).unsqueeze(1)
        return (mel_features - mean) / (std + NORMALIZE_EPSILON) * mask, attention_mask

    def process(self, audios: Sequence[Any], device: str = "cpu") -> ProcessedAudio:
        """Turn raw waveforms into padded, normalized log-mel features.

        Args:
            audios (Sequence[Any]): One entry per audio item: a 1-D waveform, or a
                `(waveform, sample_rate)` pair. Multi-channel input is averaged to mono.
            device (str): Device to run the front-end on.

        Return:
            (ProcessedAudio) Clip-batched features plus the per-item grouping and token counts.
        """
        waveforms: List[torch.Tensor] = []
        for audio in audios:
            if isinstance(audio, tuple):
                audio, sample_rate = audio
                assert sample_rate == self.config.sampling_rate, (
                    f"audio must be resampled to {self.config.sampling_rate} Hz on the host, "
                    f"got {sample_rate}"
                )
            tensor = torch.as_tensor(audio, device=device, dtype=torch.float32)
            if tensor.ndim > 1:
                tensor = tensor.mean(-1)
            waveforms.append(tensor)

        clips: List[torch.Tensor] = []
        clips_per_item: List[int] = []
        token_counts: List[int] = []
        for waveform in waveforms:
            item_clips = self._split(waveform)
            clips.extend(item_clips)
            clips_per_item.append(len(item_clips))
            token_counts.append(self.token_count(int(waveform.shape[0])))

        sample_lengths = torch.tensor(
            [clip.shape[0] for clip in clips], dtype=torch.long, device=device
        )
        max_length = int(sample_lengths.max())
        padded = torch.zeros(len(clips), max_length, dtype=torch.float32, device=device)
        for i, clip in enumerate(clips):
            padded[i, : clip.shape[0]] = clip

        padded = self._preemphasis(padded, sample_lengths)
        mel_features = self._log_mel(padded)
        mel_features, attention_mask = self._normalize(mel_features, sample_lengths)

        return ProcessedAudio(
            mel_features=mel_features,
            attention_mask=attention_mask,
            clips_per_item=clips_per_item,
            token_counts=token_counts,
        )
