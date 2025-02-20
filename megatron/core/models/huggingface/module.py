# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

from transformers import AutoConfig, AutoModel

from megatron.core.transformer.module import MegatronModule


class HuggingFaceModule(MegatronModule):
    """
    Basic module for huggingface
    """

    def __init__(self, config):
        super().__init__(config=config)

    def set_input_tensor(self, input_tensor):
        """Dummy function for set_input_tensor"""
        self.input_tensor = input_tensor


class AutoHuggingFaceModel(HuggingFaceModule):
    """
    Wrapper for HuggingFace AutoModel
    """

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_pretrained(config.huggingface_model_name_or_path)

    def forward(self, *args, **kwargs):
        """Forward function"""
        return self.model(*args, **kwargs)


def build_hf_model(config):
    """Builds huggingface wrapper model given config"""
    hf_config = AutoConfig.from_pretrained(config.huggingface_model_name_or_path)

    if "qwen" in hf_config.model_type:
        from megatron.core.models.huggingface.qwen_model import QwenHuggingFaceModel

        model = QwenHuggingFaceModel(config)
    elif "vit" in hf_config.model_type:
        from megatron.core.models.huggingface.clip_model import ClipHuggingFaceModel

        model = ClipHuggingFaceModel(config)
    else:
        raise NotImplementedError(f"Huggingface model type {hf_config.model_type} is not supported")

    return model
