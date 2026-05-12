# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import torch

from megatron.core.transformer.module import MegatronModule

try:
    from transformers import AutoConfig, AutoModel

    HAVE_TRANSFORMERS = True
except ImportError:
    HAVE_TRANSFORMERS = False


class HuggingFaceModule(MegatronModule):
    """
    Basic module for huggingface.
    """

    def __init__(self, config):
        super().__init__(config=config)

    def set_input_tensor(self, input_tensor):
        """Dummy function for set_input_tensor"""
        self.input_tensor = input_tensor

    def __setattr__(self, name: str, value):
        """
        Set average_gradients_across_tp_domain attribute true on all params so that during
        finalize_model_grads an all-reduce is performed on this module’s gradients across
        tensor parallel ranks. This keeps replicated weights synchronized and prevents drift
        due to non determinism in HF models producing slightly different grads in replicated
        models on the same inputs.
        """
        super().__setattr__(name, value)

        if isinstance(value, torch.nn.Module):
            for param in value.parameters(recurse=True):
                setattr(param, "average_gradients_across_tp_domain", True)


class AutoHuggingFaceModel(HuggingFaceModule):
    """
    Wrapper for HuggingFace AutoModel
    """

    def __init__(self, config):
        if not HAVE_TRANSFORMERS:
            raise ImportError(
                "transformers is required for AutoHuggingFaceModel, "
                "please install it with `pip install transformers`"
            )

        super().__init__(config)
        self.model = AutoModel.from_pretrained(config.huggingface_model_name_or_path)

    def forward(self, *args, **kwargs):
        """Forward function"""
        return self.model(*args, **kwargs)


def get_hf_model_type(model_path):
    """Get the Huggingface model type."""

    if not HAVE_TRANSFORMERS:
        raise ImportError(
            "transformers is required for get_hf_model_type, "
            "please install it with `pip install transformers`"
        )

    # Parakeet is a special case: its model id may be `nemo://...`, which
    # AutoConfig cannot resolve, so detect it from the prefix. Require the
    # `nemo://` or `hf://` scheme so unrelated local paths that happen to
    # contain "parakeet" (e.g. a user directory) don't get misrouted.
    lowered = model_path.lower()
    if lowered.startswith(("nemo://", "hf://")):
        model_id = lowered.split("://", 1)[1]
        # Match a path segment whose name begins with "parakeet" (e.g.
        # `nvidia/parakeet-tdt-0.6b-v2`). Substring-anywhere matches like
        # `myparakeet-clone` are intentionally rejected.
        if any(seg.startswith("parakeet") for seg in model_id.split("/")):
            return "parakeet"
        # Any other `nemo://` model can't be resolved by AutoConfig below;
        # raise a clear error rather than letting `split("hf://")[1]` raise
        # an IndexError with no context.
        if lowered.startswith("nemo://"):
            raise NotImplementedError(
                f"nemo:// scheme is currently only supported for parakeet models, "
                f"got {model_path}"
            )

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
    elif "parakeet" in model_type:
        from megatron.core.models.huggingface.fastconformer_model import ParakeetHuggingFaceModel

        model = ParakeetHuggingFaceModel(config)
    else:
        raise NotImplementedError(f"unsupported huggingface model {config.hf_config}")

    return model
