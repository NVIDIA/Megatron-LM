# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MiniMax-M3 lite native model (text / LM).

Composition (see ``skills/model-compose/minimax-m3.md``):
* layers 0-2: ``GQAttention`` (full causal, TE core) + dense clamped-SwiGLU MLP
* layers 3-59: ``MSAttention`` (MiniMax Sparse Attention primitive) + MoE
  (``SigmoidTopKRouter`` with expert bias, ``Experts`` grouped GEMM, one shared expert)
* every RMSNorm is Gemma-style (``zero_centered_gamma=True``); the attention input
  norm is fused into the qkv ``LayerNormLinear``, the dense-MLP input norm into
  ``gate_up``; MoE layers keep an explicit ``mlp_norm``.

Reference: HF ``transformers.models.minimax_m3_vl`` (5.16.1).

Attention runs on MagiAttention (MSA extension + msa_v1 kernels for sparse layers, native
``calc_attn`` for dense layers) over packed documents described by ``magi_ctx``; attention TP=1.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from megatron.lite.model.minimax_m3.config import MiniMaxM3Config
from megatron.lite.primitive import transformer_engine as te
from megatron.lite.primitive.modules.attention.msa import MSAttention
from megatron.lite.primitive.modules.dispatcher import TokenDispatcher
from megatron.lite.primitive.modules.experts import Experts, swiglu_with_probs
from megatron.lite.primitive.modules.gqa import GQAttention
from megatron.lite.primitive.modules.router import SigmoidTopKRouter
from megatron.lite.primitive.ops.cross_entropy import vocab_parallel_cross_entropy
from megatron.lite.primitive.ops.logprob import vocab_parallel_entropy
from megatron.lite.primitive.parallel import (
    ColumnParallelLinear,
    ParallelState,
    RowParallelLinear,
    VocabParallelEmbedding,
    VocabParallelOutput,
    build_pipeline_chunk_layout,
    scatter_to_sequence_parallel,
)


def m3_activation(y: torch.Tensor, config: MiniMaxM3Config) -> torch.Tensor:
    """Clamped SwiGLU ``(up + 1) * gate * sigmoid(1.702 gate)`` (HF ``swigluoai``)."""
    return swiglu_with_probs(y, None, config.swiglu_limit, config.swiglu_alpha, config.swiglu_up_offset)


class M3DenseMLP(nn.Module):
    """Dense MLP with the pre-MLP RMSNorm fused into ``gate_up`` (layers 0-2)."""

    def __init__(self, config: MiniMaxM3Config, ps: ParallelState):
        super().__init__()
        self.config = config
        self.gate_up = ColumnParallelLinear(
            config.hidden_size,
            config.dense_intermediate_size * 2,
            ps,
            bias=False,
            normalization="RMSNorm",
            eps=config.rms_norm_eps,
            zero_centered_gamma=True,
        )
        self.down = RowParallelLinear(config.dense_intermediate_size, config.hidden_size, ps, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(m3_activation(self.gate_up(x), self.config))


class M3SharedExpert(nn.Module):
    """Single shared expert, no gate (unlike Qwen)."""

    def __init__(self, config: MiniMaxM3Config, ps: ParallelState):
        super().__init__()
        self.config = config
        ffn = config.shared_intermediate_size
        self.gate_up = ColumnParallelLinear(config.hidden_size, ffn * 2, ps, bias=False)
        self.down = RowParallelLinear(ffn, config.hidden_size, ps, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(m3_activation(self.gate_up(x), self.config))


class M3MoELayer(nn.Module):
    """sigmoid router (+ expert bias, renormalised top-k, x routed_scaling_factor) + routed experts + shared expert."""

    def __init__(
        self,
        config: MiniMaxM3Config,
        ps: ParallelState,
        *,
        moe_dispatcher: str = "alltoall",
        moe_hybridep_num_sms: int | None = None,
        moe_hybridep_fused_permute: bool = False,
        moe_act_recompute: bool = False,
        moe_permute_fusion: bool | None = None,
    ):
        super().__init__()
        self.router = SigmoidTopKRouter(
            config,
            ps,
            compute_aux_loss=False,
            router_dtype=torch.float32,
            gate_fp32=True,  # as mcore moe_router_dtype=fp32
            expert_bias_persistent=True,  # loaded from e_score_correction_bias, updated outside the optimizer
        )
        self.experts = Experts(config, ps, fp8=False, moe_act_recompute=moe_act_recompute)
        self.dispatcher = TokenDispatcher(
            config.num_experts,
            config.hidden_size,
            ps,
            dispatch_backend=moe_dispatcher,
            moe_permute_fusion=moe_permute_fusion,
            hybridep_num_sms=moe_hybridep_num_sms,
            hybridep_fused_permute=moe_hybridep_fused_permute,
        )
        self.shared_expert = M3SharedExpert(config, ps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape
        x_2d = x.reshape(-1, x.size(-1))
        shared_out = self.shared_expert(x_2d)
        scores, indices = self.router(x_2d)
        dispatched, tpe, permuted_probs = self.dispatcher.dispatch(x_2d, scores, indices)
        del scores, indices
        self.dispatcher.wait_dispatch_event()
        expert_out = self.experts(
            dispatched,
            tpe,
            permuted_probs,
            tokens_per_expert_list=getattr(self.dispatcher, "_local_tpe_list", None),
        )
        routed_out = self.dispatcher.combine(expert_out)
        out = routed_out.view(input_shape) + shared_out.view(input_shape)
        return out.to(x.dtype)


class MiniMaxM3Layer(nn.Module):
    def __init__(
        self,
        config: MiniMaxM3Config,
        ps: ParallelState,
        layer_idx: int,
        *,
        moe_dispatcher: str = "alltoall",
        moe_hybridep_num_sms: int | None = None,
        moe_hybridep_fused_permute: bool = False,
        moe_act_recompute: bool = False,
        moe_permute_fusion: bool = True,
        rope_fusion: bool = False,
        msa_index_input_norm: str = "recompute",
        msa_indexer_trainable: bool = False,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_sparse_attention = config.is_sparse_attention_layer(layer_idx)
        self.is_moe = config.is_moe_layer(layer_idx)
        attn_kwargs = dict(
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
            rotary_percent=config.partial_rotary_factor,
            zero_centered_gamma=True,
            qkv_layout="flat",
            rope_fusion=rope_fusion,
        )
        if self.is_sparse_attention:
            self.attn: nn.Module = MSAttention(
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                ps,
                index_n_heads=config.index_n_heads,
                index_head_dim=config.index_head_dim,
                block_size=config.index_block_size,
                topk_blocks=config.index_topk_blocks,
                local_blocks=config.index_local_blocks,
                index_input_norm=msa_index_input_norm,
                indexer_trainable=msa_indexer_trainable,
                **attn_kwargs,
            )
        else:
            self.attn = GQAttention(
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                ps,
                use_thd=False,
                output_gate=False,
                attention_backend="magi",
                **attn_kwargs,
            )
        if self.is_moe:
            self.mlp_norm: nn.Module | None = te.RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps, zero_centered_gamma=True
            )
            self.moe: M3MoELayer | None = M3MoELayer(
                config,
                ps,
                moe_dispatcher=moe_dispatcher,
                moe_hybridep_num_sms=moe_hybridep_num_sms,
                moe_hybridep_fused_permute=moe_hybridep_fused_permute,
                moe_act_recompute=moe_act_recompute,
                moe_permute_fusion=moe_permute_fusion,
            )
            self.mlp: M3DenseMLP | None = None
        else:
            self.mlp_norm = None
            self.moe = None
            self.mlp = M3DenseMLP(config, ps)

    def forward(self, x: torch.Tensor, magi_ctx):
        if self.is_sparse_attention:
            x = x + self.attn(x, magi_ctx=magi_ctx)
        else:
            # dense layers: generic GQAttention magi backend on the same dispatch layout
            x = x + self.attn(
                x, position_ids=magi_ctx.position_ids, packed_seq_params=magi_ctx.dense_packed_seq_params()
            )
        if self.moe is not None:
            assert self.mlp_norm is not None
            x = x + self.moe(self.mlp_norm(x))
        else:
            assert self.mlp is not None
            x = x + self.mlp(x)
        return x


def _temperature_to_float(temperature: float | torch.Tensor) -> float:
    return float(temperature.item()) if isinstance(temperature, torch.Tensor) else float(temperature)


class MiniMaxM3Model(nn.Module):
    def __init__(
        self,
        config: MiniMaxM3Config,
        train_config,
        ps: ParallelState,
        *,
        vpp_chunk_id: int | None = None,
    ):
        super().__init__()
        self.config = config
        self.train_config = train_config
        self.ps = ps
        self._input_tensor: torch.Tensor | None = None
        layout = build_pipeline_chunk_layout(config.num_hidden_layers, ps, train_config.vpp, vpp_chunk_id)
        self.layer_indices = layout.layer_indices
        self.pre_process = layout.has_embed
        self.post_process = layout.has_head
        self.share_embeddings_and_output_weights = False

        self.embed: VocabParallelEmbedding | None = None
        if layout.has_embed:
            self.embed = VocabParallelEmbedding(config.vocab_size, config.hidden_size, ps)

        recompute_modules = getattr(train_config, "recompute_modules", [])
        moe_act_recompute = "moe_act" in recompute_modules and "moe" not in recompute_modules
        self.layers = nn.ModuleList(
            [
                MiniMaxM3Layer(
                    config,
                    ps,
                    idx,
                    moe_dispatcher=train_config.moe_dispatcher,
                    moe_hybridep_num_sms=train_config.moe_hybridep_num_sms,
                    moe_hybridep_fused_permute=train_config.moe_hybridep_fused_permute,
                    moe_act_recompute=moe_act_recompute,
                    moe_permute_fusion=getattr(train_config, "moe_permute_fusion", True),
                    rope_fusion=getattr(train_config, "rope_fusion", False),
                    msa_index_input_norm=getattr(train_config, "msa_index_input_norm", "recompute"),
                    msa_indexer_trainable=getattr(train_config, "msa_indexer_trainable", False),
                )
                for idx in self.layer_indices
            ]
        )
        self.norm: nn.Module | None = None
        self.head: VocabParallelOutput | None = None
        if layout.has_head:
            self.norm = te.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, zero_centered_gamma=True)
            self.head = VocabParallelOutput(config.vocab_size, config.hidden_size, ps)

    def _reduce_loss(self, token_loss: torch.Tensor, loss_mask: torch.Tensor | None) -> torch.Tensor:
        """Mean over the supervised tokens of the whole sequence, also under context parallel.

        Each CP rank holds a token shard whose supervised-token count differs, so a local mean averaged by
        DDP would weight shards unequally. Normalise by the CP-global count and sum the shares with an
        autograd all-reduce: every rank reports the true global mean, and the backward pass scales the
        local gradient by cp_size so the DDP average over dp x cp gives the correct per-token gradient.
        """
        if loss_mask is None:
            mask_sb = torch.ones_like(token_loss)
        else:
            mask_sb = loss_mask.transpose(0, 1).to(dtype=token_loss.dtype)
        num_tokens = mask_sb.sum()
        if self.ps.cp_size > 1:
            num_tokens = num_tokens.clone()
            torch.distributed.all_reduce(num_tokens, group=self.ps.cp_group)
        share = (token_loss * mask_sb).sum() / num_tokens.clamp_min(1.0)
        if self.ps.cp_size > 1:
            from torch.distributed.nn.functional import all_reduce

            share = all_reduce(share, group=self.ps.cp_group)
        return share

    def set_input_tensor(self, input_tensor):
        if isinstance(input_tensor, list):
            if len(input_tensor) > 1:
                raise ValueError("MiniMaxM3Model expects a single pipeline input tensor.")
            input_tensor = input_tensor[0] if input_tensor else None
        self._input_tensor = input_tensor

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        temperature: float | torch.Tensor = 1.0,
        use_fused_kernels: bool = False,
        calculate_entropy: bool = False,
        magi_ctx=None,
    ) -> dict:
        if use_fused_kernels:
            raise NotImplementedError("fused linear cross-entropy is not wired for MiniMax-M3 yet")
        if magi_ctx is None:
            raise ValueError("MiniMaxM3Model requires magi_ctx (built by the protocol forward step)")
        if self.embed is not None:
            assert input_ids is not None
            h = self.embed(input_ids)  # [S, B, H]
        else:
            h = hidden_states if hidden_states is not None else self._input_tensor
            assert h is not None
        if self.embed is not None:
            h = scatter_to_sequence_parallel(h, self.ps)
        for layer in self.layers:
            h = layer(h, magi_ctx=magi_ctx)

        output = {"hidden_states": h}
        if self.head is not None:
            assert self.norm is not None
            hidden_for_head = self.norm(h)
            logits = self.head(hidden_for_head)
            if labels is not None:
                t = _temperature_to_float(temperature)
                if t != 1.0:
                    logits = logits / t
                labels_sb = labels.transpose(0, 1).contiguous()
                loss = vocab_parallel_cross_entropy(logits, labels_sb, self.ps.tp_group)
                output["loss"] = self._reduce_loss(loss, loss_mask)
                output["log_probs"] = (-loss).transpose(0, 1).contiguous()
                if calculate_entropy:
                    output["entropy"] = vocab_parallel_entropy(logits, self.ps.tp_group).transpose(0, 1).contiguous()
            else:
                output["logits"] = self.head.gather(logits).transpose(0, 1).contiguous()
        return output
