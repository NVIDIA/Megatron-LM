# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
import torch

from megatron.core.models.huggingface import HuggingFaceModule


# Nemo model loading is very slow so do this global thing to cache it.
_NEMO_SOUND_MODEL_SINGLETON = None

def get_nemo_sound_model(sound_model_type):
    global _NEMO_SOUND_MODEL_SINGLETON
    if _NEMO_SOUND_MODEL_SINGLETON is None:
        import nemo.collections.asr as nemo_asr
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=sound_model_type.split("nemo://")[1])
        asr_model.encoder.sync_max_audio_length = False  # fix: hanging on unnecessary max seq len sync via NCCL in some edge cases
        for layer in asr_model.encoder.layers:
            layer.self_attn.use_pytorch_sdpa = True
        _NEMO_SOUND_MODEL_SINGLETON = (asr_model.preprocessor, asr_model.encoder)
    return _NEMO_SOUND_MODEL_SINGLETON


class ParakeetHuggingFaceModel(HuggingFaceModule):
    """
    Wrapper for Parakeet based on original Nemo implementation or HF FastConformer model
    """

    def __init__(self, config):
        super().__init__(config)

        self.use_nemo = config.sound_model_type.startswith("nemo://")
        if config.sound_model_type.startswith("nemo://"):
            self.feature_extractor, self.model = get_nemo_sound_model(config.sound_model_type)

            for module in self.model.modules():
                if module.__class__.__name__.lower() == "dropout":
                    module.p = config.hidden_dropout

            if config.recompute_granularity is not None:
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
                self.model = checkpoint_wrapper(self.model)
        elif config.sound_model_type.startswith("hf://"):
            sound_model_type = config.sound_model_type.split("hf://")[1]

            from megatron.core.models.huggingface.fastconformer.modeling_fastconformer import FastConformerModel
            from megatron.core.models.huggingface.fastconformer.feature_extraction_fastconformer import FastConformerFeatureExtractor

            self.feature_extractor = FastConformerFeatureExtractor.from_pretrained(sound_model_type)
            self.model = FastConformerModel.from_pretrained(sound_model_type)

            if config.recompute_granularity is not None:
                self.model.gradient_checkpointing_enable()
        else:
            raise ValueError(f"Unknown sound model type: {config.sound_model_type}")

    def forward(self, *args, **kwargs):
        """Forward function"""
        if self.use_nemo:
            features = self.feature_extractor(input_signal=args[0], length=args[1])
            y = self.model(audio_signal=features[0], length=features[1])
            return y[0].permute(0, 2, 1), y[1]
        else:
            # HF feature extractor expects audio as first arg only, not (audio, length) like NeMo
            # args[0] is sound_clips tensor, args[1] is sound_length (skipping it for HF)
            sound_clips = args[0]
            features = self.feature_extractor(sound_clips, **kwargs, return_tensors="pt", sampling_rate=16000, return_attention_mask=True)
            y = self.model(features.input_features.to(torch.bfloat16), features.attention_mask)
            lengths = features.attention_mask.sum(dim=-1).to(y.last_hidden_state.device)
            return y.last_hidden_state, lengths
