# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MiniMax-M3 language model (text tower) for Megatron Lite.

Decoder layer: ``x + attn(norm(x))`` then ``x + ffn(norm(x))`` with (1 + w) RMSNorms,
MiniMax sparse attention (Magi-MSA ``MSAV1``) and either a dense clamped
quick-GEGLU FFN or a sigmoid top-k MoE with one shared expert.

Token layout: one packed microbatch per forward (micro-batch 1). The embedding
is computed for the full packed row and then dispatched to this CP rank's
tokens with the microbatch's Magi-MSA runtime key, so every layer, RoPE and the
loss see the dispatched order; the embedding gradient is scaled by 1/CP so the
sum over CP ranks is the full-sequence gradient. Tensors are ``[S, 1, H]``.

Loss: per-token cross-entropy masked and summed over this rank's tokens
(``loss_sum``); the protocol divides it by this rank's token count (``num_tokens``).
The indexer KL enters the backward only.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from megatron.lite.model.minimax_m3.config import MiniMaxM3Config
from megatron.lite.primitive import transformer_engine as te
from megatron.lite.primitive.kernels.quick_geglu import clamped_quick_geglu
from megatron.lite.primitive.modules.attention.msa import MsaDenseAttention, MsaV1Attention
from megatron.lite.primitive.modules.dispatcher import TokenDispatcher
from megatron.lite.primitive.modules.experts import Experts
from megatron.lite.primitive.modules.router import SigmoidTopKRouter
from megatron.lite.primitive.ops.cross_entropy import vocab_parallel_cross_entropy
from megatron.lite.primitive.parallel import (
    ParallelState,
    VocabParallelEmbedding,
    VocabParallelOutput,
    build_pipeline_chunk_layout,
)


def _rms_norm(config: MiniMaxM3Config) -> nn.Module:
    return te.RMSNorm(
        config.hidden_size,
        eps=config.rms_norm_eps,
        zero_centered_gamma=True,
        params_dtype=torch.bfloat16,
    )


def _linear(in_features: int, out_features: int) -> nn.Module:
    return te.Linear(in_features, out_features, bias=False, params_dtype=torch.bfloat16)


class _ScaleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


class GatedMLP(nn.Module):
    """Dense FFN / shared expert: ``fc2(swigluoai(fc1(x)))`` with ``fc1 = [gate; up]``."""

    def __init__(self, config: MiniMaxM3Config, ffn_size: int):
        super().__init__()
        self.limit = config.swiglu_limit
        self.fc1 = _linear(config.hidden_size, 2 * ffn_size)
        self.fc2 = _linear(ffn_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = clamped_quick_geglu(self.fc1(x), limit=self.limit, linear_offset=1.0)
        return self.fc2(h)


class MoE(nn.Module):
    """Shared expert + sigmoid top-k routed experts (``routed + shared``).

    The shared expert runs first, then the router, then dispatch; this order fixes
    the autograd accumulation order of the three gradients into the input.
    """

    def __init__(self, config: MiniMaxM3Config, ps: ParallelState, *, use_deepep: bool = False):
        super().__init__()
        self.router = SigmoidTopKRouter(
            config,
            ps,
            compute_aux_loss=bool(config.aux_loss_alpha),
            router_dtype=torch.float32,
            expert_bias_persistent=True,
            router_te_gemm=False,
            gate_weight_fp32=True,
        )
        self.experts = Experts(config, ps)
        self.dispatcher = TokenDispatcher(
            config.num_experts, config.hidden_size, ps, use_deepep=use_deepep
        )
        self.shared_expert = GatedMLP(config, config.shared_intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.shared_expert(x)
        flat = x.view(-1, x.size(-1))
        scores, indices = self.router(flat)
        dispatched, tokens_per_expert, probs = self.dispatcher.dispatch(flat, scores, indices)
        del scores, indices
        self.dispatcher.wait_dispatch_event()
        expert_out = self.experts(
            dispatched,
            tokens_per_expert,
            probs,
            tokens_per_expert_list=getattr(self.dispatcher, "_local_tpe_list", None),
        )
        routed = self.dispatcher.combine(expert_out)
        return routed.view(x.shape) + shared


class MiniMaxM3Layer(nn.Module):
    def __init__(
        self,
        config: MiniMaxM3Config,
        ps: ParallelState,
        layer_idx: int,
        *,
        use_deepep: bool = False,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = _rms_norm(config)
        heads = dict(
            rotary_dim=config.rotary_dim,
            norm_eps=config.rms_norm_eps,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
        )
        if config.is_sparse_attention_layer(layer_idx):
            self.self_attention: nn.Module = MsaV1Attention(
                config.hidden_size,
                num_index_heads=config.sparse_num_index_heads,
                index_dim=config.sparse_index_dim,
                **heads,
            )
        else:
            self.self_attention = MsaDenseAttention(config.hidden_size, **heads)
        self.pre_mlp_layernorm = _rms_norm(config)
        if config.is_moe_layer(layer_idx):
            self.mlp: nn.Module = MoE(config, ps, use_deepep=use_deepep)
        else:
            self.mlp = GatedMLP(config, config.dense_intermediate_size)

    def forward(self, x: torch.Tensor, *, runtime_key: Any, rotary_freqs: torch.Tensor):
        x = x + self.self_attention(
            self.input_layernorm(x), runtime_key=runtime_key, rotary_freqs=rotary_freqs
        )
        return x + self.mlp(self.pre_mlp_layernorm(x))


class MiniMaxM3Model(nn.Module):
    """One pipeline chunk of the MiniMax-M3 text model.

    ``forward`` takes the full packed ``input_ids`` (embedding stage) or the
    previous stage's hidden states, plus this microbatch's Magi-MSA ``runtime_key``,
    dispatched RoPE angles, and (last stage) dispatched ``labels`` / ``loss_mask``.
    """

    def __init__(
        self,
        config: MiniMaxM3Config,
        ps: ParallelState,
        *,
        deterministic: bool = False,
        use_deepep: bool = False,
    ):
        super().__init__()
        self.config = config
        self.ps = ps
        self._input_tensor: torch.Tensor | None = None
        layout = build_pipeline_chunk_layout(config.num_hidden_layers, ps)
        self.layer_indices = layout.layer_indices
        self.pre_process = layout.has_embed
        self.post_process = layout.has_head
        self.share_embeddings_and_output_weights = False

        self.embed: VocabParallelEmbedding | None = None
        if layout.has_embed:
            self.embed = VocabParallelEmbedding(
                config.vocab_size, config.hidden_size, ps, deterministic=deterministic
            )
        self.layers = nn.ModuleList(
            [MiniMaxM3Layer(config, ps, idx, use_deepep=use_deepep) for idx in self.layer_indices]
        )
        self.norm: nn.Module | None = None
        self.head: VocabParallelOutput | None = None
        if layout.has_head:
            self.norm = _rms_norm(config)
            self.head = VocabParallelOutput(config.vocab_size, config.hidden_size, ps)

    def set_input_tensor(self, input_tensor):
        if isinstance(input_tensor, list):
            if len(input_tensor) > 1:
                raise ValueError("MiniMaxM3Model expects a single pipeline input tensor.")
            input_tensor = input_tensor[0] if input_tensor else None
        self._input_tensor = input_tensor

    def embed_and_dispatch(self, input_ids: torch.Tensor, dispatch) -> torch.Tensor:
        """Embed the full packed row ``[1, T]`` and keep this CP rank's tokens ``[S, 1, H]``."""
        assert self.embed is not None
        h = self.embed(input_ids)  # [T, 1, H]
        if self.ps.cp_size > 1:
            h = _ScaleGrad.apply(h, 1.0 / self.ps.cp_size)
        return dispatch(h[:, 0]).clone().unsqueeze(1)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        runtime_key: Any,
        rotary_freqs: torch.Tensor,
        dispatch=None,
        labels: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
    ) -> dict:
        if self.embed is not None:
            h = self.embed_and_dispatch(input_ids, dispatch)
        else:
            h = self._input_tensor
            assert h is not None
        for layer in self.layers:
            h = layer(h, runtime_key=runtime_key, rotary_freqs=rotary_freqs)
        output: dict[str, Any] = {"hidden_states": h}
        if self.head is None:
            return output
        assert self.norm is not None
        logits = self.head(self.norm(h))  # [S, 1, V] bf16
        if labels is None:
            output["logits"] = self.head.gather(logits)
            return output
        output.update(self.loss(logits, labels, loss_mask))
        return output

    def loss(self, logits: torch.Tensor, labels: torch.Tensor, loss_mask: torch.Tensor) -> dict:
        """``labels``/``loss_mask``: dispatched ``[S]``; masked labels must already be -100."""
        token_loss = vocab_parallel_cross_entropy(
            logits, labels.view(-1, 1), self.ps.tp_group
        ).view(-1)
        mask = loss_mask.view(-1).float()
        return {
            "token_loss": token_loss,
            "loss_sum": torch.sum(token_loss * mask),
            "num_tokens": mask.sum().detach(),
        }


__all__ = ["GatedMLP", "MiniMaxM3Layer", "MiniMaxM3Model", "MoE"]
