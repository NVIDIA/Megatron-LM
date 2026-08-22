# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core import parallel_state
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.transformer.attention import Attention


def _iter_qk_clip_modules(model_chunk):
    """Yield attention modules that need QK clipping.

    Preserve the legacy traversal for other model types. HybridModel may own
    nested MTP stacks, so discover its attention modules recursively.
    """

    model_module = model_chunk.module.module
    if isinstance(model_module, HybridModel):
        yield from (
            module
            for module in model_module.modules()
            if isinstance(module, Attention) and hasattr(module, 'clip_qk')
        )
        return

    for transformer_layer in model_module.decoder.layers:
        if hasattr(transformer_layer.self_attention, 'clip_qk'):
            yield transformer_layer.self_attention


def clip_qk(model, log_max_only=False) -> float:
    """
    Clips QK attention logits to prevent numerical instability.

    Args:
        model (List[MegatronModule]): Model chunks containing attention layers.
        log_max_only (bool): If True, only computes max logit without clipping.

    Returns:
        float: The maximum QK logit value across all chunks.
    """

    with torch.no_grad():
        log_max_attention_logit = 0
        for model_chunk in model:
            for attention in _iter_qk_clip_modules(model_chunk):
                current_max_attn_logits = attention.core_attention.current_max_attn_logits
                if current_max_attn_logits is None:
                    continue
                torch.distributed.all_reduce(
                    current_max_attn_logits,
                    op=torch.distributed.ReduceOp.MAX,
                    group=parallel_state.get_data_parallel_group(with_context_parallel=True),
                )
                log_max_attention_logit = max(
                    log_max_attention_logit, torch.max(current_max_attn_logits).item()
                )
                if not log_max_only:
                    attention.clip_qk()
                else:
                    # When qk-clip is disabled, clip_qk() is not called and
                    # would otherwise never reset current_max_attn_logits.
                    # Reset it here so the logged value reflects the current
                    # step and stale references are not retained.
                    attention.core_attention.current_max_attn_logits = None

    return log_max_attention_logit
