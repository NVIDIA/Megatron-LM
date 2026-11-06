# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DS4.1 shifted mHC state, composed from unmodified Core operations."""
import torch
from torch import nn

from megatron.core.transformer.hyper_connection import HyperConnectionModule, native_h_post_bda
from megatron.lite.model.deepseek_v41.lite.components.config import reject_native_recompute


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
        modes = (
            {"full"}
            if recompute is True
            else set(recompute.split(",") if isinstance(recompute, str) else (recompute or ()))
        )
        if modes and torch.is_grad_enabled() and source_state is None:
            raise ValueError("DS4.1 layer recomputation requires explicit source_state")
        if "full" in modes and torch.is_grad_enabled():
            return source_state.checkpoint(
                lambda h, pre: self._forward(
                    h, pre, packed_seq_params=packed_seq_params, input_ids=input_ids
                ),
                hidden,
                pre_mix,
            )
        return self._forward(
            hidden,
            pre_mix,
            packed_seq_params=packed_seq_params,
            input_ids=input_ids,
            source_state=source_state,
            modes=modes,
        )

    def _forward(
        self, hidden, pre_mix, *, packed_seq_params, input_ids=None, source_state=None, modes=()
    ):
        for kind in ("attn", "moe"):

            def step(h, pre, kind=kind):
                return self._sublayer(h, pre, kind, packed_seq_params, input_ids)

            hidden, pre_mix = (
                source_state.checkpoint(step, hidden, pre_mix)
                if kind in modes and torch.is_grad_enabled()
                else step(hidden, pre_mix)
            )
        return hidden, pre_mix

    def _sublayer(self, hidden, pre_mix, kind, packed_seq_params, input_ids):
        mixes = self.attn_mixes if kind == "attn" else self.ffn_mixes
        norm = self.attn_norm if kind == "attn" else self.ffn_norm
        flat, dtype = hidden.flatten(2), hidden.dtype
        # DS4.1 shifts pre-mixes by one sublayer; Core owns all mixing math.
        next_pre, post, comb = mixes.compute_mappings(flat.float())
        x = norm(mixes.aggregate(flat.float(), pre_mix.float()).to(dtype))
        x, bias = (
            self.attention(x, attention_mask=None, packed_seq_params=packed_seq_params)
            if kind == "attn"
            else self.ffn(x, input_ids=input_ids, packed_seq_params=packed_seq_params)
        )
        hidden = native_h_post_bda(
            comb, hidden.float(), post, x.float(), None if bias is None else bias.float()
        ).to(dtype)
        return hidden, next_pre
