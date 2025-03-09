# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

from transformers import AutoConfig, AutoModel

from megatron.core.transformer.module import MegatronModule


class HuggingFaceModule(MegatronModule):
    """
    Basic module for huggingface.
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


def get_hf_model_type(model_path):
    """Get the Huggingface model type."""
    hf_config = AutoConfig.from_pretrained(model_path.split("hf://")[1])
    model_type = hf_config.architectures[0].lower()

    if "qwen" in model_type:
        return "qwen"
    elif "siglip" in model_type:
        return "siglip"
    else:
        raise NotImplementedError(f"unsupported huggingface model {model_type}")


def build_hf_model(config, model_path):
    """Builds Huggingface wrapper model given config and model path."""
    model_type = get_hf_model_type(model_path)

    if "qwen" in model_type:
        from megatron.core.models.huggingface.qwen_model import QwenHuggingFaceModel

        model = QwenHuggingFaceModel(config)
    elif "siglip" in model_type:
        from megatron.core.models.huggingface.clip_model import SiglipHuggingFaceModel

        model = SiglipHuggingFaceModel(config)
    else:
        raise NotImplementedError(f"unsupported huggingface model {config.hf_config}")

    return model
