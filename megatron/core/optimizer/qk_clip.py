# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core import parallel_state


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
            for transformer_layer in model_chunk.module.module.decoder.layers:
                if hasattr(transformer_layer.self_attention, 'clip_qk'):
                    torch.distributed.all_reduce(
                        transformer_layer.self_attention.core_attention.current_max_attn_logits,
                        op=torch.distributed.ReduceOp.MAX,
                        group=parallel_state.get_data_parallel_group(with_context_parallel=True),
                    )
                    log_max_attention_logit = max(
                        log_max_attention_logit,
                        torch.max(
                            transformer_layer.self_attention.core_attention.current_max_attn_logits
                        ).item(),
                    )
                    if not log_max_only:
                        transformer_layer.self_attention.clip_qk()

    return log_max_attention_logit
