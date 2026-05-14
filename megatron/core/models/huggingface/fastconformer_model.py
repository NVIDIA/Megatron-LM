# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
import torch

from megatron.core.models.huggingface import HuggingFaceModule

# NeMo model loading is slow, so cache the (preprocessor, encoder) tuple per
# `sound_model_type`. Keying by model id avoids returning a stale cached encoder
# when the same process constructs more than one Parakeet variant.
_NEMO_SOUND_MODEL_CACHE: dict[str, tuple] = {}


def get_nemo_sound_model(sound_model_type):
    """Load (and cache) a NeMo ASR encoder + preprocessor for the given ``nemo://`` model id."""
    if sound_model_type not in _NEMO_SOUND_MODEL_CACHE:
        import nemo.collections.asr as nemo_asr

        asr_model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=sound_model_type.split("nemo://")[1]
        )
        # Avoid hangs from an unnecessary max-seq-len NCCL sync in some edge cases.
        asr_model.encoder.sync_max_audio_length = False
        for layer in asr_model.encoder.layers:
            layer.self_attn.use_pytorch_sdpa = True
        _NEMO_SOUND_MODEL_CACHE[sound_model_type] = (asr_model.preprocessor, asr_model.encoder)
    return _NEMO_SOUND_MODEL_CACHE[sound_model_type]


class ParakeetHuggingFaceModel(HuggingFaceModule):
    """Wrapper for Parakeet sound encoders.

    Supports two backends, selected by ``config.sound_model_type`` prefix:

    - ``nemo://<model_name>`` loads a NeMo ASR encoder + preprocessor.
    - ``hf://<model_name>`` loads the upstream Hugging Face FastConformer model
      via ``transformers.AutoModel`` / ``AutoFeatureExtractor``.
    """

    def __init__(self, config):
        super().__init__(config)

        self.use_nemo = config.sound_model_type.startswith("nemo://")
        if self.use_nemo:
            self.feature_extractor, self.model = get_nemo_sound_model(config.sound_model_type)

            for module in self.model.modules():
                if module.__class__.__name__.lower() == "dropout":
                    module.p = config.hidden_dropout

            if config.recompute_granularity is not None:
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                    checkpoint_wrapper,
                )

                self.model = checkpoint_wrapper(self.model)
        elif config.sound_model_type.startswith("hf://"):
            from transformers import AutoFeatureExtractor, AutoModel

            sound_model_type = config.sound_model_type.split("hf://")[1]
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(sound_model_type)
            self.model = AutoModel.from_pretrained(sound_model_type)

            if config.recompute_granularity is not None:
                self.model.gradient_checkpointing_enable()
        else:
            raise ValueError(f"Unknown sound model type: {config.sound_model_type}")

    def _model_dtype(self) -> torch.dtype:
        """Return the dtype of the encoder's first parameter (defaults to bf16)."""
        for param in self.model.parameters():
            return param.dtype
        return torch.bfloat16

    def _sampling_rate(self) -> int:
        """Return the sampling rate the feature extractor expects (default 16 kHz)."""
        return int(getattr(self.feature_extractor, "sampling_rate", 16000))

    def forward(self, *args, **kwargs):
        """Forward pass returning (hidden_states, lengths).

        Args:
            args[0]: Sound clips tensor.
            args[1]: Sound length tensor (used by NeMo backend; ignored for HF).
        """
        if self.use_nemo:
            features = self.feature_extractor(input_signal=args[0], length=args[1])
            y = self.model(audio_signal=features[0], length=features[1])
            # NeMo encoder returns [B, H, T]; LLaVA expects [B, T, H].
            return y[0].permute(0, 2, 1), y[1]
        else:
            # HF feature extractor expects audio as the first arg only,
            # not (audio, length) as in NeMo.
            sound_clips = args[0]
            features = self.feature_extractor(
                sound_clips,
                **kwargs,
                return_tensors="pt",
                sampling_rate=self._sampling_rate(),
                return_attention_mask=True,
            )
            y = self.model(features.input_features.to(self._model_dtype()), features.attention_mask)
            lengths = features.attention_mask.sum(dim=-1).to(y.last_hidden_state.device)
            return y.last_hidden_state, lengths
