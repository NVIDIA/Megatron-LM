# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import torch


def clip_qk(model, log_only=False) -> float:
    """
    Clip the QK attention logits to the threshold, recommended for Muon optimizer.

    Args:
        model: The model to clip the QK attention logits, a list of model chunks.
        log_only: Whether to only log the max attention logit, without updating the weights.

    Returns:
        The maximum attention logit, a float.
    """

    log_max_attention_logit = 0
    for model_chunk in model:
        for transformer_layer in model_chunk.module.module.decoder.layers:
            if hasattr(transformer_layer.self_attention, 'clip_qk'):
                log_max_attention_logit = max(
                    log_max_attention_logit,
                    torch.max(
                        transformer_layer.self_attention.core_attention.current_max_attn_logits
                    ).item(),
                )
                if not log_only:
                    transformer_layer.self_attention.clip_qk()

    return log_max_attention_logit
