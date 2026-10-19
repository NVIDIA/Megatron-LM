# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DS4.1 shifted mHC state, composed from unmodified Core operations."""
import torch
from torch import nn

from megatron.core.transformer.hyper_connection import HyperConnectionModule, native_h_post_bda
from megatron.lite.model.deepseek_v41.lite.native import reject_native_recompute


class DS41Block(nn.Module):
    """Carry the preceding sublayer's pre-mix across layer/stage boundaries.

    Core's complete TransformerLayer uses the current sublayer's pre-mix;
    DS4.1 shifts it by one sublayer. Only this model-level composition differs.
    The pair (hidden, pre_mix) must remain intact across pipeline boundaries.
    Core dev 7b150acf owns compute_mappings, aggregation and residual mixing.
    """

    def __init__(self, config, layer_number, attention, ffn):
        super().__init__()
        reject_native_recompute(config.recompute_granularity, config.recompute_method)
        self.attention, self.ffn = attention, ffn
        self.attn_mixes = HyperConnectionModule(config, layer_number)
        self.ffn_mixes = HyperConnectionModule(config, layer_number)
        self.attn_mixes.norm_eps = self.ffn_mixes.norm_eps = config.layernorm_epsilon
        self.attn_norm = nn.RMSNorm(
            config.hidden_size, eps=config.layernorm_epsilon, dtype=config.params_dtype
        )
        self.ffn_norm = nn.RMSNorm(
            config.hidden_size, eps=config.layernorm_epsilon, dtype=config.params_dtype
        )
        self.copies, self.width = config.num_residual_streams, config.hidden_size

    def forward(
        self,
        hidden,
        pre_mix,
        *,
        packed_seq_params,
        input_ids=None,
        source_state=None,
        recompute=False,
    ):
        if recompute and torch.is_grad_enabled():
            if source_state is None:
                raise ValueError("DS4.1 layer recomputation requires explicit source_state")
            return source_state.checkpoint(
                lambda h, pre: self._forward(
                    h, pre, packed_seq_params=packed_seq_params, input_ids=input_ids
                ),
                hidden,
                pre_mix,
            )
        return self._forward(
            hidden, pre_mix, packed_seq_params=packed_seq_params, input_ids=input_ids
        )

    def _forward(self, hidden, pre_mix, *, packed_seq_params, input_ids=None):
        shape = hidden.shape
        flat = hidden.reshape(*shape[:2], -1)
        dtype = hidden.dtype
        # Release mixing coefficients and reductions are FP32; only the
        # sublayer input/residual carrier is rounded back to BF16.
        attn_pre, post, comb = self.attn_mixes.compute_mappings(flat.float())
        x = self.attn_mixes.aggregate(flat.float(), pre_mix.float()).to(dtype)
        x = self.attn_norm(x)
        x, bias = self.attention(x, attention_mask=None, packed_seq_params=packed_seq_params)
        hidden = native_h_post_bda(
            comb, hidden.float(), post, x.float(), None if bias is None else bias.float()
        ).to(dtype)
        flat = hidden.reshape(*shape[:2], -1)
        next_pre, post, comb = self.ffn_mixes.compute_mappings(flat.float())
        x = self.ffn_mixes.aggregate(flat.float(), attn_pre).to(dtype)
        x = self.ffn_norm(x)
        x, bias = self.ffn(x, input_ids=input_ids, packed_seq_params=packed_seq_params)
        hidden = native_h_post_bda(
            comb, hidden.float(), post, x.float(), None if bias is None else bias.float()
        ).to(dtype)
        return hidden, next_pre
