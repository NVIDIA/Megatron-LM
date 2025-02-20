# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

from transformers import AutoModel

from megatron.core.models.huggingface import HuggingFaceModule


class ClipHuggingFaceModel(HuggingFaceModule):
    """
    Wrapper for CLIP HuggingFace models
    """

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_pretrained(config.huggingface_model_name_or_path)

    def forward(self, *args, **kwargs):
        """Forward function"""
        x = self.model(*args, **kwargs)
        x = x['last_hidden_state']

        return x
