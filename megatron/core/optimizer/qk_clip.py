import torch

def clip_qk(model) -> float:
    """
    Clip the QK attention logits to the threshold, recommended for Muon optimizer.
    
    Args:
        model: The model to clip the QK attention logits, a list of model chunks.

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
                    ).item()
                )
                transformer_layer.self_attention.clip_qk()

    return log_max_attention_logit