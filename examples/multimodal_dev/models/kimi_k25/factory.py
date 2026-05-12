# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Factory functions for Kimi K2.5 VL model construction.

Encapsulates all Kimi K2.5-specific logic needed by ``pretrain_multimodal.py``
so that the training entry point remains model-agnostic.
"""

import os

from megatron.core.models.gpt.gpt_model import GPTModel

from examples.multimodal_dev.models.kimi_k25.configuration import (
    KIMI_K25_IMAGE_TOKEN_ID,
    KIMI_K25_VOCAB_SIZE,
    get_kimi_k25_language_config,
)
from examples.multimodal_dev.models.kimi_k25.model import KimiK25VLModel
from examples.multimodal_dev.models.kimi_k25.specs import get_kimi_k25_language_spec


# Default HF model path
_DEFAULT_HF_MODEL_PATH = "moonshotai/Kimi-K2.5"


def _get_hf_model_path() -> str:
    env_path = os.environ.get("KIMI_K25_HF_MODEL_PATH")
    if env_path:
        return env_path
    try:
        from megatron.training import get_args

        path = getattr(get_args(), "hf_model_path", None)
        if path:
            return path
    except (ModuleNotFoundError, AssertionError):
        pass
    return _DEFAULT_HF_MODEL_PATH


def post_language_config(language_config, args):
    """Apply Kimi K2.5-specific settings to the language config.

    For Kimi K2.5, the language config is an MLATransformerConfig built
    entirely from ``configuration.py``.  The CLI-provided
    ``TransformerConfig`` from ``core_transformer_config_from_args`` is
    NOT used — we override it completely with the MLA config and only
    propagate parallelism/precision fields from args.
    """
    pass


def build_model(args, language_config, vision_config, **kwargs):
    """Build a complete Kimi K2.5 VL model instance.

    Because Kimi K2.5 uses MLATransformerConfig (not the standard
    TransformerConfig from CLI args), we build the language config from
    scratch using the variant dict and only inherit parallelism/precision
    settings from the CLI-provided config.

    Args:
        args: Megatron parsed arguments.
        language_config: ``TransformerConfig`` from CLI args (used only
            for parallelism/precision fields).
        vision_config: Ignored — Kimi vision encoder is loaded from HF.
        **kwargs: Extra keyword arguments.

    Returns:
        A :class:`KimiK25VLModel` instance.
    """
    variant = getattr(args, "model_variant", "proxy") or "proxy"
    hf_model_path = _get_hf_model_path()

    # Build the real MLATransformerConfig from the variant dict
    mla_config = get_kimi_k25_language_config(variant=variant)

    # Propagate parallelism and precision from CLI args
    mla_config.bf16 = language_config.bf16
    mla_config.fp16 = language_config.fp16
    mla_config.tensor_model_parallel_size = language_config.tensor_model_parallel_size
    mla_config.pipeline_model_parallel_size = language_config.pipeline_model_parallel_size
    mla_config.expert_model_parallel_size = language_config.expert_model_parallel_size
    mla_config.sequence_parallel = language_config.sequence_parallel
    mla_config.context_parallel_size = language_config.context_parallel_size
    if hasattr(language_config, 'use_distributed_optimizer'):
        mla_config.use_distributed_optimizer = language_config.use_distributed_optimizer
    if hasattr(language_config, 'init_model_with_meta_device'):
        mla_config.init_model_with_meta_device = language_config.init_model_with_meta_device

    # Build language spec and model
    layer_spec = get_kimi_k25_language_spec(mla_config)

    seq_length = getattr(args, "seq_length", 4096)

    language_model = GPTModel(
        config=mla_config,
        transformer_layer_spec=layer_spec,
        vocab_size=KIMI_K25_VOCAB_SIZE,
        max_sequence_length=seq_length,
        pre_process=True,
        post_process=True,
        position_embedding_type="rope",
        scatter_embedding_sequence_parallel=False,
    )

    model = KimiK25VLModel(
        language_model=language_model,
        hf_model_path=hf_model_path,
        media_placeholder_token_id=getattr(
            args, "image_token_id", KIMI_K25_IMAGE_TOKEN_ID,
        ),
        freeze_vision_model=True,
        freeze_vision_projection=True,
    )

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("*" * 80)
    print(f"KimiK25VLModel [variant={variant}]")
    print(f"  Total params:     {total / 1e9:.3f}B")
    print(f"  Trainable params: {trainable / 1e9:.3f}B")
    print(f"  Vision frozen:    True")
    print(f"  HF model path:    {hf_model_path}")
    print("*" * 80)

    return model


def get_kimi_k25_vision_config_stub(num_layers_override=None):
    """Stub vision config — Kimi vision encoder is loaded from HF.

    Returns a minimal TransformerConfig to satisfy the registry interface.
    The actual vision encoder (MoonViT3d) is loaded dynamically from HF
    inside KimiK25VLModel, so this config is unused.
    """
    from megatron.core.transformer.transformer_config import TransformerConfig

    return TransformerConfig(
        num_layers=1,
        hidden_size=1152,
        num_attention_heads=16,
        bf16=False,
    )
