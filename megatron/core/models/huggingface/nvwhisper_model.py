# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
from megatron.core.models.huggingface import HuggingFaceModule
import torch
from transformers import Qwen2AudioEncoder

class NVWhisperHuggingFaceModel(HuggingFaceModule):
    """
    Wrapper for NVWhisper based on HF Qwen2AudioEncoder
    """

    def __init__(self, config):
        super().__init__(config)
        # TODO(jbarker): This is a hack to load the model from a local directory.
        # We should load from an openly available source.
        model = Qwen2AudioEncoder.from_pretrained(config.sound_model_type.split("hf://")[1])
        self.model = model.cuda().to(torch.bfloat16)

        if config.recompute_granularity is not None:
            self.model.gradient_checkpointing_enable()

    def forward(self, *args, **kwargs):
        """Forward function"""
        x = self.model(*args, **kwargs)

        return x['last_hidden_state']
