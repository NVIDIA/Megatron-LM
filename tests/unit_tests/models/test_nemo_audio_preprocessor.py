# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Smoke tests for the vendored NeMo preprocessor wrapper.

The implementation lives in ``nemo_audio_preprocessing_standalone`` and uses
only stdlib + PyTorch, so these tests do not require ``nemo_toolkit``.
"""

import torch

from megatron.core.models.audio.audio_feature_config import NemoAudioFeatureConfig
from megatron.core.models.audio.nemo_audio_preprocessing_standalone import (
    AudioToMelSpectrogramPreprocessor,
)


def _build(config: NemoAudioFeatureConfig) -> AudioToMelSpectrogramPreprocessor:
    return AudioToMelSpectrogramPreprocessor(**config.to_nemo_kwargs()).eval()


class TestVendoredNemoPreprocessor:
    def test_forward_shape_and_seq_len(self):
        config = NemoAudioFeatureConfig(
            sample_rate=16000,
            window_size=0.025,
            window_stride=0.01,
            features=64,
            n_fft=512,
            normalize="per_feature",
            preemph=0.97,
            log=True,
            dither=0.0,
            pad_to=0,
        )
        preproc = _build(config)

        torch.manual_seed(0)
        sample_rate = 16000
        durations = [0.5, 1.0]
        wave = torch.zeros(2, int(max(durations) * sample_rate), dtype=torch.float32)
        for i, dur in enumerate(durations):
            wave[i, : int(dur * sample_rate)] = torch.randn(int(dur * sample_rate))
        lengths = torch.tensor([int(d * sample_rate) for d in durations], dtype=torch.long)

        mels, out_len = preproc(wave, lengths)

        assert mels.dim() == 3
        assert mels.shape[0] == 2
        assert mels.shape[1] == 64
        # Re-use the preprocessor's own get_seq_len so the test doesn't bake in
        # the exact STFT framing formula.
        for i in range(2):
            expected = int(preproc.get_seq_len(lengths[i].float()).item())
            assert int(out_len[i].item()) == expected

    def test_no_preemph_no_normalize(self):
        config = NemoAudioFeatureConfig(
            features=32, normalize=None, preemph=None, dither=0.0, log=False, pad_to=0
        )
        preproc = _build(config)

        wave = torch.randn(1, 16000, dtype=torch.float32)
        mels, out_len = preproc(wave, torch.tensor([16000], dtype=torch.long))
        assert mels.shape[0] == 1
        assert mels.shape[1] == 32
        assert int(out_len[0].item()) > 0
