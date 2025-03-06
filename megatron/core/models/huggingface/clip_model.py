# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

from transformers import AutoModel

from megatron.core.models.huggingface import HuggingFaceModule


class SiglipHuggingFaceModel(HuggingFaceModule):
    """
    Wrapper for Siglip HuggingFace models.
    """

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_pretrained(config.vision_model_type.split("hf://")[1])

    def forward(self, *args, **kwargs):
        """Siglip forward."""
        x = self.model(*args, **kwargs)
        x = x["last_hidden_state"]

        return x
