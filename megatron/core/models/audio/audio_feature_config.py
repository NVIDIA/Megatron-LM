# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""Audio frontend descriptors for the NeMo Transformer audio encoder.

These are the model-side configuration and token-count primitives that describe
the NeMo audio frontend:

* ``NemoAudioFeatureConfig`` mirrors the NeMo ``AudioToMelSpectrogramPreprocessor``
  keyword arguments consumed by
  ``megatron.core.models.audio.nemo_audio_preprocessing_standalone``.
* ``NemoTransformerAudioTokenEstimator`` is the pure frame-count -> expanded-token
  math implied by the encoder's pre-encode/subsampling configuration.

They are deliberately dependency-free (stdlib + ``math`` only) so the audio model
package carries no data-loader dependency. The data pipeline's waveform processor
(``NemoAudioProcessor``) composes these and is injected into the dataloader.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Dict


def ceil_div(value: int, divisor: int) -> int:
    if divisor <= 0:
        raise ValueError(f"divisor must be > 0, got {divisor}")
    return (value + divisor - 1) // divisor


@dataclass
class NemoAudioFeatureConfig:
    """Mirrors NeMo ``AudioToMelSpectrogramPreprocessor.__init__`` keyword args.

    Defaults match the published NeMo ``transformer_stacking`` YAML
    (Slaney mel, ``per_feature`` normalize, 0.97 pre-emphasis, ``log(x + 2**-24)``).
    ``NemoAudioProcessor`` forwards these verbatim (via ``to_nemo_kwargs``) to
    the vendored standalone ``AudioToMelSpectrogramPreprocessor`` in
    ``megatron.core.models.audio.nemo_audio_preprocessing_standalone``.
    """

    sample_rate: int = 16000
    window_size: float = 0.025
    window_stride: float = 0.01
    n_window_size: int | None = None
    n_window_stride: int | None = None
    window: str = "hann"
    normalize: str | None = "per_feature"
    n_fft: int | None = 512
    preemph: float | None = 0.97
    features: int = 128
    lowfreq: float = 0.0
    highfreq: float | None = None
    log: bool = True
    log_zero_guard_type: str = "add"
    log_zero_guard_value: Any = 2 ** -24
    dither: float = 1e-5
    pad_to: int = 0
    frame_splicing: int = 1
    exact_pad: bool = False
    pad_value: float = 0.0
    mag_power: float = 2.0
    nb_augmentation_prob: float = 0.0
    nb_max_freq: int = 4000
    mel_norm: str = "slaney"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NemoAudioFeatureConfig":
        valid = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in valid}
        return cls(**kwargs)

    def to_nemo_kwargs(self) -> Dict[str, Any]:
        """Returns the kwarg dict consumable by ``AudioToMelSpectrogramPreprocessor``."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True)
class NemoTransformerAudioTokenEstimator:
    """Pure math for NeMo TransformerEncoder expanded-token counts.

    Conv-style pre-encoders update lengths with floor division after each
    strided convolution. The stacking pre-encoder pads the batch tensor to a
    multiple of ``encoder_time_stride``, then keeps each sample's partial final
    stack as one output token, so stacking counts are per-sample ceil divisions.

    ``encoder_time_stride`` is required and has no default: it must be derived
    from the loaded encoder config (``NemoTransformerAudioConfig.encoder_time_stride``,
    which depends on ``pre_encode`` and ``subsampling_factor``). Hard-coding a
    default here would silently lie for any checkpoint whose encoder downsamples
    by something other than that constant -- e.g. a ``transformer_stacking``
    encoder with ``subsampling_factor=8``. Construct via the provider
    (``examples/multimodal/v3/energon_multimodal_provider.py``) or pass the value
    explicitly.
    """

    encoder_time_stride: int
    stack_factor: int = 2
    pre_encode: str = "conv"

    def _estimate_encoder_steps(
        self,
        num_frames: int,
        padded_num_frames: int | None = None,
    ) -> int:
        if num_frames < 0:
            raise ValueError(f"num_frames must be >= 0, got {num_frames}")
        if padded_num_frames is not None and padded_num_frames < num_frames:
            raise ValueError(
                f"padded_num_frames={padded_num_frames} must be >= num_frames={num_frames}"
            )

        if self.pre_encode in ("conv", "depth_conv"):
            return num_frames // self.encoder_time_stride

        if self.pre_encode == "stacking":
            return ceil_div(num_frames, self.encoder_time_stride)

        raise ValueError(
            f"Unsupported Nemo TransformerEncoder pre_encode={self.pre_encode!r}; "
            "expected 'conv', 'depth_conv', or 'stacking'."
        )

    def estimate(self, num_frames: int, padded_num_frames: int | None = None) -> int:
        encoder_steps = self._estimate_encoder_steps(num_frames, padded_num_frames)
        return math.ceil(encoder_steps / self.stack_factor)

    def estimate_from_num_frames(
        self,
        num_frames: int,
        padded_num_frames: int | None = None,
    ) -> int:
        return self.estimate(num_frames, padded_num_frames)

    def __call__(self, num_frames: int, padded_num_frames: int | None = None) -> int:
        return self.estimate(num_frames, padded_num_frames)
