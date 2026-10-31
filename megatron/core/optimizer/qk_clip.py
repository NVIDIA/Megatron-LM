# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.process_groups_config import resolve_process_groups


def clip_qk(model, log_max_only=False, *, dp_cp_group=None) -> float:
    """
    Clips QK attention logits to prevent numerical instability.

    Args:
        model (List[MegatronModule]): Model chunks containing attention layers.
        log_max_only (bool): If True, only computes max logit without clipping.
        dp_cp_group (torch.distributed.ProcessGroup, optional): Group over which the maximum
            logits are reduced: the model's data- and context-parallel group including its
            GTP-remat peers (``pg_collection.dp_cp_gtp_remat``). Omitting it is deprecated:
            ``clip_qk`` then warns and uses the global
            ``parallel_state.get_data_parallel_group(with_context_parallel=True)``.

    Returns:
        float: The maximum QK logit value across all chunks.
    """

    with torch.no_grad():
        log_max_attention_logit = 0
        for model_chunk in model:
            for transformer_layer in model_chunk.module.module.decoder.layers:
                if hasattr(transformer_layer.self_attention, 'clip_qk'):
                    if (
                        transformer_layer.self_attention.core_attention.current_max_attn_logits
                        is None
                    ):
                        continue
                    if dp_cp_group is None:
                        # Only a model with logits to reduce needs the group, so only such a
                        # model warns when the caller omits it.
                        dp_cp_group = resolve_process_groups(
                            None,
                            owner="clip_qk",
                            required=("dp_cp_gtp_remat",),
                            argument="dp_cp_group",
                            deprecated_in="0.21",
                            removed_in="0.23",
                        ).dp_cp_gtp_remat
                    torch.distributed.all_reduce(
                        transformer_layer.self_attention.core_attention.current_max_attn_logits,
                        op=torch.distributed.ReduceOp.MAX,
                        group=dp_cp_group,
                    )
                    log_max_attention_logit = max(
                        log_max_attention_logit,
                        torch.max(
                            transformer_layer.self_attention.core_attention.current_max_attn_logits
                        ).item(),
                    )
                    if not log_max_only:
                        transformer_layer.self_attention.clip_qk()
                    else:
                        # When qk-clip is disabled, clip_qk() is not called and
                        # would otherwise never reset current_max_attn_logits.
                        # Reset it here so the logged value reflects the current
                        # step and stale references are not retained.
                        transformer_layer.self_attention.core_attention.current_max_attn_logits = (
                            None
                        )

    return log_max_attention_logit
