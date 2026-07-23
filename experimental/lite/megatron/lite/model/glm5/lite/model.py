# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""GLM-5 (deepseek_v3_2) lite native model.

This model is a near-verbatim clone of the Kimi-K2 (deepseek_v3) lite model
(``megatron/lite/model/kimi_k2/lite/model.py``).  It inherits ALL of Kimi's
Megatron plumbing unchanged: SBHD ``[S, B, H]`` layout, sequence-parallel
scatter/gather, ``VocabParallelEmbedding`` / ``VocabParallelOutput``,
``share_embeddings_and_output_weights``, ``set_input_tensor``,
``build_pipeline_chunk_layout``-based pipeline boundaries, MTP, and the
dist-opt / distckpt integration.

The ONLY functional difference from Kimi is the attention module: where Kimi
uses ``MultiLatentAttention`` (MLA), GLM-5 uses Dynamic Sparse Attention (DSA).
The DSA primitive is hard-wired batch-first ``[B, S, H]`` and expects explicit
``cos`` / ``sin`` / ``position_ids`` + ``packed_seq_params``, so it is wrapped
by ``Glm5DSAAttention`` which transposes ``[S, B, H] -> [B, S, H]`` before DSA
and back after, and builds rotary embeddings from protocol-provided positions.
surrounding Kimi skeleton therefore stays byte-for-byte SBHD and untouched.

NOTE: DSA is NOT tensor-parallel-capable, so GLM-5 is a documented TP=1 special
case (the protocol gate raises for TP>1 / ETP>1).  VPP / PP / EP / CP work,
inherited from Kimi.
"""

from __future__ import annotations

import inspect
import os
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te

from megatron.lite.model.glm5.config import Glm5Config
from megatron.lite.primitive.kernels.swiglu import bias_swiglu_impl
from megatron.lite.primitive.modules.attention import (
    DSAIndexShareState,
    DynamicSparseAttention,
    build_rotary_embeddings,
    dsa,
)
from megatron.lite.primitive.modules.dispatcher import TokenDispatcher
from megatron.lite.primitive.modules.experts import Experts
from megatron.lite.primitive.modules.moe import MoEAuxLossAutoScaler
from megatron.lite.primitive.modules.mtp import MTPLossAutoScaler
from megatron.lite.primitive.ops.cross_entropy import vocab_parallel_cross_entropy
from megatron.lite.primitive.ops.linear_cross_entropy import linear_cross_entropy
from megatron.lite.primitive.ops.logprob import vocab_parallel_entropy
from megatron.lite.primitive.ops.sp_ops import ReduceScatterDim0
from megatron.lite.primitive.parallel import (
    ColumnParallelLinear,
    ParallelState,
    RowParallelLinear,
    VanillaColumnParallelLinear,
    VocabParallelEmbedding,
    VocabParallelOutput,
    build_pipeline_chunk_layout,
    gather_from_sequence_parallel,
    roll_packed_thd_left,
    scatter_to_sequence_parallel,
)
from megatron.lite.primitive.parallel.cp import contiguous_position_ids_for_cp
from megatron.lite.primitive.utils import build_fp8_recipe
from megatron.lite.primitive.utils.moe import (
    compute_routing_scores_for_aux_loss,
    router_gating_linear,
    switch_load_balancing_loss_func,
    topk_routing_with_score_function,
)

# -- GLM-5 ONLY: SP-grad parameter suffixes for the DSA attention (Kimi's MLA
# suffixes -- linear_q_down_proj / linear_kv_down_proj / *_up_proj.linear.* --
# are replaced by the DSA module's parameter names).  These tag the
# layernorm-fronted columns whose grads must be all-reduced across SP ranks.
# (At GLM-5's enforced TP=1 this list is unused, but is kept structurally so
# the model stays a faithful Kimi clone.)
_SP_GRAD_SUFFIXES: tuple[str, ...] = (
    ".input_layernorm.weight",
    ".self_attention.q_a_proj.weight",
    ".self_attention.q_a_layernorm.weight",
    ".self_attention.kv_a_proj_with_mqa.weight",
    ".self_attention.kv_a_layernorm.weight",
    ".mlp_norm.weight",
    ".mlp.gate_up.linear.layer_norm_weight",
    ".moe.router.gate.weight",
    ".norm.weight",
)


def _collect_sp_grad_params(model: nn.Module) -> list[nn.Parameter]:
    return [
        param
        for name, param in model.named_parameters()
        if any(name.endswith(suffix) for suffix in _SP_GRAD_SUFFIXES) or name == "norm.weight"
    ]


def _swiglu(x: torch.Tensor) -> torch.Tensor:
    if x.is_cuda:
        return bias_swiglu_impl(x, None, False, False)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return F.silu(x1) * x2


def _reduce_scatter_to_sequence_parallel(x: torch.Tensor, ps: ParallelState) -> torch.Tensor:
    if ps.tp_size == 1:
        return x
    return ReduceScatterDim0.apply(x, ps.tp_size, ps.tp_rank, ps.tp_group)


def _ordered_topk_from_routing_map(
    probs_dense: torch.Tensor, routing_map: torch.Tensor, topk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    expert_ids = torch.arange(
        probs_dense.size(-1), device=probs_dense.device, dtype=torch.long
    ).expand_as(routing_map)
    masked_ids = torch.where(
        routing_map, expert_ids, torch.full_like(expert_ids, probs_dense.size(-1))
    )
    topk_indices = torch.sort(masked_ids, dim=-1).values[:, :topk]
    topk_scores = torch.gather(probs_dense, dim=-1, index=topk_indices)
    return topk_scores, topk_indices


def _router_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    router_dtype: torch.dtype,
) -> torch.Tensor:
    if x.is_cuda:
        return router_gating_linear(x, weight, bias, router_dtype)
    return F.linear(
        x.to(router_dtype),
        weight.to(router_dtype),
        None if bias is None else bias.to(router_dtype),
    )


def _topk_routing_supports_groups() -> bool:
    params = inspect.signature(topk_routing_with_score_function).parameters
    return "num_groups" in params and "group_topk" in params


# -- GLM-5 ONLY: DSA attention wrapper.  Holds ``DynamicSparseAttention`` and
# adapts it to Kimi's SBHD-in / SBHD-out attention contract.  Everything outside
# this class (norms, residual, MoE, MTP, embed, head, SP scatter/gather) is
# identical to Kimi.
class Glm5DSAAttention(nn.Module):
    """SBHD ``[S, B, H]`` shim around the batch-first DSA primitive.

    Kimi's decoder layer calls ``self.self_attention(x_sbhd, packed_seq_params=)``
    where ``x_sbhd`` is ``[S, B, H]``.  DSA is hard-wired ``[B, S, H]`` and needs
    explicit ``cos`` / ``sin`` / ``position_ids``.  This wrapper:
      1. transposes ``[S, B, H] -> [B, S, H]``,
      2. consumes explicit packed/CP ``position_ids`` and builds rotary
         ``cos`` / ``sin``,
      3. runs DSA,
      4. transposes the ``[B, S, H]`` output back to ``[S, B, H]``.
    The Kimi skeleton therefore never observes the batch-first interior.
    """

    def __init__(
        self,
        config: Glm5Config,
        ps: ParallelState,
        layer_idx: int,
        *,
        dsa_cp_mode: str = "native",
        dsa_indexer_loss_coeff: float = 0.0,
        dsa_indexer_use_sparse_loss: bool = False,
        calculate_per_token_loss: bool = False,
    ):
        super().__init__()
        self.ps = ps
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.rope_theta = config.rope_theta
        self.self_attention = DynamicSparseAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            index_n_heads=config.index_n_heads,
            index_head_dim=config.index_head_dim,
            index_topk=config.index_topk,
            rms_norm_eps=config.rms_norm_eps,
            rope_interleaved=config.rope_interleave,
            latent_rms_norm_eps=config.latent_rms_norm_eps,
            indexer_layer_norm_eps=config.indexer_layer_norm_eps,
            indexer_rope_interleaved=config.indexer_rope_interleave,
            indexer_rope_first=config.indexer_rope_first,
            indexer_use_hadamard=config.indexer_use_hadamard,
            layer_number=layer_idx + 1,
            index_topk_freq=config.index_topk_freq,
            index_skip_topk_offset=config.index_skip_topk_offset,
            indexer_type=config.dsa_indexer_type(layer_idx),
            indexer_loss_coeff=dsa_indexer_loss_coeff,
            indexer_use_sparse_loss=dsa_indexer_use_sparse_loss,
            calculate_per_token_loss=calculate_per_token_loss,
            cp_size=ps.cp_size,
            cp_rank=ps.cp_rank,
            cp_group=ps.cp_group,
            cp_mode=dsa_cp_mode,
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        packed_seq_params=None,
        dsa_index_share_state: DSAIndexShareState | None = None,
    ) -> torch.Tensor:
        # Kimi feeds SBHD [S, B, H]; DSA needs batch-first [B, S, H].
        x_bsh = x.transpose(0, 1).contiguous()
        batch, seq_len, _ = x_bsh.shape
        if position_ids is None:
            if packed_seq_params is not None:
                raise ValueError(
                    "GLM5 packed DSA requires explicit per-sequence position_ids."
                )
            if self.ps.cp_size > 1:
                position_ids = contiguous_position_ids_for_cp(
                    seq_len * self.ps.cp_size,
                    self.ps.cp_rank,
                    self.ps.cp_size,
                    x_bsh.device,
                )
            else:
                position_ids = torch.arange(
                    seq_len, device=x_bsh.device, dtype=torch.long
                ).unsqueeze(0)
        position_ids = position_ids.to(device=x_bsh.device, dtype=torch.long)
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        if position_ids.shape[-1] != seq_len:
            raise ValueError(
                "GLM5 DSA position_ids must match the local sequence length, "
                f"got {tuple(position_ids.shape)} for local length {seq_len}."
            )
        if position_ids.shape[0] == 1 and batch > 1:
            position_ids = position_ids.expand(batch, -1)
        elif position_ids.shape[0] != batch:
            raise ValueError(
                "GLM5 DSA position_ids batch dimension must be 1 or match hidden states, "
                f"got {tuple(position_ids.shape)} for batch {batch}."
            )
        cos, sin = build_rotary_embeddings(
            position_ids=position_ids,
            dim=self.qk_rope_head_dim,
            rope_theta=self.rope_theta,
            dtype=x_bsh.dtype,
        )
        out_bsh = self.self_attention(
            x_bsh,
            cos=cos,
            sin=sin,
            position_ids=position_ids,
            attention_mask=None,
            packed_seq_params=packed_seq_params,
            index_share_state=dsa_index_share_state,
        )
        # Back to SBHD [S, B, H] for the Kimi skeleton.
        return out_bsh.transpose(0, 1).contiguous()


class Glm5SigmoidTopKRouter(nn.Module):
    """GLM-5 sigmoid router with group-limited routing and persistent expert bias.

    Byte-identical to Kimi's ``KimiK2SigmoidTopKRouter`` except for the config
    type and that GLM-5 has no ``aux_loss_alpha`` HF field -- the aux coefficient
    therefore defaults to 0 (aux loss contributes 0).
    """

    def __init__(
        self,
        config: Glm5Config,
        ps: ParallelState,
        *,
        router_bias_rate: float = 0.0,
        compute_aux_loss: bool = True,
        use_pre_softmax: bool = False,
        moe_router_fusion: bool = False,
    ):
        super().__init__()
        if router_bias_rate > 0:
            raise NotImplementedError(
                "GLM-5 expert-bias EMA update is not implemented in lite yet."
            )
        self.topk = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts
        # GLM-5 has no aux_loss_alpha HF field; default the coefficient to 0.
        self.aux_loss_coeff = getattr(config, "aux_loss_alpha", 0.0)
        self.scaling_factor = config.routed_scaling_factor
        self.num_groups = config.n_group if (config.n_group and config.n_group > 1) else None
        self.group_topk = config.topk_group if self.num_groups is not None else None
        self.router_bias_rate = router_bias_rate
        self.compute_aux_loss = compute_aux_loss
        self.use_pre_softmax = use_pre_softmax
        self.moe_router_fusion = moe_router_fusion

        self.gate = nn.Linear(config.hidden_size, config.n_routed_experts, bias=False)
        self.register_buffer(
            "expert_bias",
            torch.zeros(config.n_routed_experts, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "local_tokens_per_expert",
            torch.zeros(config.n_routed_experts, dtype=torch.float32),
            persistent=False,
        )
        self._aux_loss_group = ps.tp_group if ps.tp_size > 1 else None

    def _apply(self, fn):
        super()._apply(fn)
        self.expert_bias.data = self.expert_bias.data.float()
        self.local_tokens_per_expert.data = self.local_tokens_per_expert.data.float()
        return self

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = _router_linear(x, self.gate.weight, None, torch.float32)
        logits = logits.view(-1, self.num_experts)
        num_tokens = logits.size(0)
        routing_kwargs = {}
        if self.num_groups is not None and self.group_topk is not None:
            if not _topk_routing_supports_groups():
                raise NotImplementedError(
                    "topk_routing_with_score_function does not support group-limited routing."
                )
            routing_kwargs = dict(num_groups=self.num_groups, group_topk=self.group_topk)
        probs_dense, routing_map = topk_routing_with_score_function(
            logits,
            self.topk,
            use_pre_softmax=self.use_pre_softmax,
            score_function="sigmoid",
            expert_bias=self.expert_bias.to(logits.dtype),
            scaling_factor=(self.scaling_factor or None),
            fused=self.moe_router_fusion,
            **routing_kwargs,
        )
        if torch.is_grad_enabled():
            with torch.no_grad():
                self.local_tokens_per_expert += routing_map.sum(dim=0)
        topk_scores, topk_indices = _ordered_topk_from_routing_map(
            probs_dense, routing_map, self.topk
        )
        topk_scores = topk_scores.to(logits.dtype)

        if self.compute_aux_loss and self.training and torch.is_grad_enabled():
            _, aux_scores = compute_routing_scores_for_aux_loss(
                logits, self.topk, score_function="sigmoid", fused=self.moe_router_fusion
            )
            tokens_per_expert = routing_map.sum(dim=0).to(torch.int64)
            total_num_tokens = num_tokens
            if self._aux_loss_group is not None:
                dist.all_reduce(tokens_per_expert, group=self._aux_loss_group)
                total_num_tokens = num_tokens * dist.get_world_size(group=self._aux_loss_group)
            aux_loss = switch_load_balancing_loss_func(
                aux_scores,
                tokens_per_expert,
                total_num_tokens,
                self.topk,
                self.num_experts,
                self.aux_loss_coeff,
                fused=False,
            )
            topk_scores = MoEAuxLossAutoScaler.apply(topk_scores, aux_loss)

        return topk_scores, topk_indices


class DenseMLP(nn.Module):
    def __init__(self, config: Glm5Config, ps: ParallelState):
        super().__init__()
        self.gate_up = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size * 2,
            ps,
            bias=False,
            normalization="RMSNorm",
            eps=config.rms_norm_eps,
        )
        self.down = RowParallelLinear(config.intermediate_size, config.hidden_size, ps, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(_swiglu(self.gate_up(x)))


class SharedExpert(nn.Module):
    def __init__(self, config: Glm5Config, ps: ParallelState):
        super().__init__()
        self.ps = ps
        # GLM-5 has no shared_expert_intermediate_size property; compute it from
        # n_shared_experts * moe_intermediate_size (matches Kimi's property).
        ffn = config.n_shared_experts * config.moe_intermediate_size
        self.gate_up = _LocalLinear(config.hidden_size, ffn * 2 // ps.tp_size)
        self.down = _LocalLinear(ffn // ps.tp_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze_batch = x.dim() == 2
        if squeeze_batch:
            x = x.unsqueeze(1)
        full_x = gather_from_sequence_parallel(x, self.ps)
        partial_out = self.down(_swiglu(self.gate_up(full_x)))
        out = _reduce_scatter_to_sequence_parallel(partial_out, self.ps)
        return out.squeeze(1) if squeeze_batch else out


class _LocalLinear(nn.Module):
    """TE linear without built-in TP collectives; weight remains TP-local."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = te.Linear(
            in_features,
            out_features,
            bias=False,
            params_dtype=torch.bfloat16,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MoELayer(nn.Module):
    def __init__(
        self,
        config: Glm5Config,
        ps: ParallelState,
        *,
        use_deepep: bool,
        router_bias_rate: float,
        fp8: bool,
        moe_act_recompute: bool,
    ):
        super().__init__()
        if fp8:
            raise NotImplementedError("GLM-5 lite MoE fp8 training is not implemented yet.")
        self.router = Glm5SigmoidTopKRouter(
            config,
            ps,
            router_bias_rate=router_bias_rate,
            compute_aux_loss=True,
            use_pre_softmax=True,
        )
        self.experts = Experts(
            config,
            ps,
            fp8=fp8,
            moe_act_recompute=moe_act_recompute,
        )
        self.dispatcher = TokenDispatcher(
            config.num_experts,
            config.hidden_size,
            ps,
            use_deepep=use_deepep,
        )
        self.shared_expert = SharedExpert(config, ps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape

        flat_x = x.view(-1, x.size(-1))
        scores, indices = self.router(flat_x)
        dispatched, tpe, permuted_probs = self.dispatcher.dispatch(flat_x, scores, indices)
        del scores, indices
        self.dispatcher.wait_dispatch_event()
        expert_out = self.experts(
            dispatched,
            tpe,
            permuted_probs,
            tokens_per_expert_list=getattr(self.dispatcher, "_local_tpe_list", None),
        )
        routed_out = self.dispatcher.combine(expert_out)
        shared_out = self.shared_expert(x)
        output = routed_out.view(input_shape)
        output += shared_out
        return output.to(x.dtype)


class Glm5Layer(nn.Module):
    def __init__(
        self,
        config: Glm5Config,
        ps: ParallelState,
        layer_idx: int,
        *,
        use_deepep: bool = False,
        router_bias_rate: float = 0.0,
        fp8: bool = False,
        moe_act_recompute: bool = False,
        use_thd: bool = False,
        dsa_cp_mode: str = "native",
        dsa_indexer_loss_coeff: float = 0.0,
        dsa_indexer_use_sparse_loss: bool = False,
        calculate_per_token_loss: bool = False,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = te.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # GLM-5 ONLY: DSA attention (Kimi builds MultiLatentAttention here).
        # The wrapper preserves the SBHD self_attention(x, packed_seq_params=)
        # contract so this layer's forward stays identical to Kimi's.
        del use_thd  # DSA derives its own THD handling from packed_seq_params.
        self.self_attention = Glm5DSAAttention(
            config,
            ps,
            layer_idx,
            dsa_cp_mode=dsa_cp_mode,
            dsa_indexer_loss_coeff=dsa_indexer_loss_coeff,
            dsa_indexer_use_sparse_loss=dsa_indexer_use_sparse_loss,
            calculate_per_token_loss=calculate_per_token_loss,
        )
        if config.is_moe_layer(layer_idx):
            self.mlp_norm: nn.Module | None = te.RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.moe: MoELayer | None = MoELayer(
                config,
                ps,
                use_deepep=use_deepep,
                router_bias_rate=router_bias_rate,
                fp8=fp8,
                moe_act_recompute=moe_act_recompute,
            )
            self.mlp: DenseMLP | None = None
        else:
            self.mlp_norm = None
            self.moe = None
            self.mlp = DenseMLP(config, ps)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        packed_seq_params=None,
        dsa_index_share_state: DSAIndexShareState | None = None,
    ) -> torch.Tensor:
        x = x + self.self_attention(
            self.input_layernorm(x),
            position_ids=position_ids,
            packed_seq_params=packed_seq_params,
            dsa_index_share_state=dsa_index_share_state,
        )
        if self.moe is not None:
            assert self.mlp_norm is not None
            mlp_input = self.mlp_norm(x)
            return x + self.moe(mlp_input)
        assert self.mlp is not None
        return x + self.mlp(x)


def _roll_mtp_left(
    tensor: torch.Tensor,
    *,
    packed_seq_params=None,
    dims: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if packed_seq_params is not None:
        return roll_packed_thd_left(tensor, packed_seq_params=packed_seq_params, dims=dims)
    dim = dims if dims >= 0 else tensor.dim() + dims
    rolled = torch.roll(tensor, shifts=-1, dims=dim)
    rolled.select(dim, -1).zero_()
    return rolled, rolled.sum()


class Glm5MTPLayer(nn.Module):
    def __init__(
        self,
        config: Glm5Config,
        ps: ParallelState,
        layer_idx: int,
        *,
        embedding: VocabParallelEmbedding,
        use_deepep: bool,
        router_bias_rate: float,
        fp8: bool,
        moe_act_recompute: bool,
        use_thd: bool,
        detach_encoder: bool,
        dsa_cp_mode: str,
        dsa_indexer_loss_coeff: float,
        dsa_indexer_use_sparse_loss: bool,
        calculate_per_token_loss: bool,
    ):
        super().__init__()
        self.ps = ps
        object.__setattr__(self, "embedding", embedding)
        self.detach_encoder = detach_encoder
        self.enorm = te.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = te.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = VanillaColumnParallelLinear(
            config.hidden_size * 2,
            config.hidden_size,
            ps,
            sp=ps.tp_size > 1,
            gather_output=True,
        )
        self.transformer_layer = Glm5Layer(
            config,
            ps,
            config.num_hidden_layers + layer_idx,
            use_deepep=use_deepep,
            router_bias_rate=router_bias_rate,
            fp8=fp8,
            moe_act_recompute=moe_act_recompute,
            use_thd=use_thd,
            dsa_cp_mode=dsa_cp_mode,
            dsa_indexer_loss_coeff=dsa_indexer_loss_coeff,
            dsa_indexer_use_sparse_loss=dsa_indexer_use_sparse_loss,
            calculate_per_token_loss=calculate_per_token_loss,
        )
        self.final_layernorm = te.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None,
        hidden_states: torch.Tensor,
        rotary_position_ids: torch.Tensor | None = None,
        packed_seq_params=None,
        dsa_index_share_state: DSAIndexShareState | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        attention_position_ids = (
            rotary_position_ids if rotary_position_ids is not None else position_ids
        )
        input_ids, _ = _roll_mtp_left(input_ids, packed_seq_params=packed_seq_params, dims=-1)
        if position_ids is not None:
            position_ids, _ = _roll_mtp_left(
                position_ids, packed_seq_params=packed_seq_params, dims=-1
            )
        decoder_input = self.embedding(input_ids)
        decoder_input = scatter_to_sequence_parallel(decoder_input, self.ps)

        if self.detach_encoder:
            decoder_input = decoder_input.detach()
            hidden_states = hidden_states.detach()

        decoder_input = self.enorm(decoder_input)
        hidden_states = self.hnorm(hidden_states)
        hidden_states = torch.cat((decoder_input, hidden_states), dim=-1)
        hidden_states = self.eh_proj(hidden_states)
        hidden_states = scatter_to_sequence_parallel(hidden_states, self.ps)
        hidden_states = self.transformer_layer(
            hidden_states,
            position_ids=attention_position_ids,
            packed_seq_params=packed_seq_params,
            dsa_index_share_state=dsa_index_share_state,
        )
        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states, input_ids, position_ids


class Glm5MTPBlock(nn.Module):
    def __init__(
        self,
        config: Glm5Config,
        ps: ParallelState,
        *,
        embedding: VocabParallelEmbedding,
        use_deepep: bool,
        router_bias_rate: float,
        fp8: bool,
        moe_act_recompute: bool,
        use_thd: bool,
        detach_encoder: bool,
        repeated_layer: bool,
        dsa_cp_mode: str,
        dsa_indexer_loss_coeff: float,
        dsa_indexer_use_sparse_loss: bool,
        calculate_per_token_loss: bool,
    ):
        super().__init__()
        self.num_layers = config.num_nextn_predict_layers
        self.repeated_layer = repeated_layer
        layers_to_build = 1 if repeated_layer else self.num_layers
        self.layers = nn.ModuleList(
            [
                Glm5MTPLayer(
                    config,
                    ps,
                    idx,
                    embedding=embedding,
                    use_deepep=use_deepep,
                    router_bias_rate=router_bias_rate,
                    fp8=fp8,
                    moe_act_recompute=moe_act_recompute,
                    use_thd=use_thd,
                    detach_encoder=detach_encoder,
                    dsa_cp_mode=dsa_cp_mode,
                    dsa_indexer_loss_coeff=dsa_indexer_loss_coeff,
                    dsa_indexer_use_sparse_loss=dsa_indexer_use_sparse_loss,
                    calculate_per_token_loss=calculate_per_token_loss,
                )
                for idx in range(layers_to_build)
            ]
        )

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None,
        hidden_states: torch.Tensor,
        packed_seq_params=None,
        dsa_index_share_state: DSAIndexShareState | None = None,
    ) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        rotary_position_ids = position_ids
        for depth in range(self.num_layers):
            layer = self.layers[0] if self.repeated_layer else self.layers[depth]
            hidden_states, input_ids, position_ids = layer(
                input_ids=input_ids,
                position_ids=position_ids,
                hidden_states=hidden_states,
                rotary_position_ids=rotary_position_ids,
                packed_seq_params=packed_seq_params,
                dsa_index_share_state=dsa_index_share_state,
            )
            outputs.append(hidden_states)
        return outputs


def _temperature_to_float(temperature: float | torch.Tensor) -> float:
    if isinstance(temperature, torch.Tensor):
        if temperature.numel() != 1:
            raise ValueError("Glm5Model supports scalar temperature only.")
        return float(temperature.detach().float().item())
    return float(temperature)


def _apply_attention_backend_override(backend: str | None) -> None:
    if backend in (None, "flash"):
        backend = "fused"
    env = {
        "auto": ("1", "1", "1"),
        "flash": ("1", "0", "0"),
        "fused": ("0", "1", "0"),
        "unfused": ("0", "0", "1"),
        "local": ("0", "0", "1"),
    }.get(backend)
    if env is None:
        raise ValueError(
            "attention_backend_override must be one of "
            "{'auto', 'flash', 'fused', 'unfused', 'local'}"
        )
    (
        os.environ["NVTE_FLASH_ATTN"],
        os.environ["NVTE_FUSED_ATTN"],
        os.environ["NVTE_UNFUSED_ATTN"],
    ) = env


def _dsa_index_share_decoder_layer_groups(config: Glm5Config) -> list[list[int]] | None:
    if not config.uses_dsa_index_share:
        return None

    groups: list[list[int]] = []
    current: list[int] = []
    current_source: int | None = None
    for layer_idx in range(config.num_hidden_layers):
        if config.dsa_indexer_type(layer_idx) == "shared":
            source_idx = config.dsa_indexer_source_layer(layer_idx)
        else:
            source_idx = layer_idx
        if current and source_idx != current_source:
            groups.append(current)
            current = []
        current.append(layer_idx)
        current_source = source_idx
    if current:
        groups.append(current)
    return groups


class Glm5Model(nn.Module):
    def __init__(
        self,
        config: Glm5Config,
        train_config,
        ps: ParallelState,
        *,
        vpp_chunk_id: int | None = None,
        router_bias_rate: float = 0.0,
        use_thd: bool = False,
        hf_path: str = "",
        attention_backend_override: str | None = None,
        dsa_cp_mode: str = "native",
        dsa_indexer_loss_coeff: float = 0.0,
        dsa_indexer_use_sparse_loss: bool = False,
        calculate_per_token_loss: bool = False,
        mtp_enable: bool = False,
        mtp_enable_train: bool = False,
        mtp_detach_encoder: bool = False,
    ):
        super().__init__()
        del hf_path
        _apply_attention_backend_override(attention_backend_override)
        self.config = config
        self.train_config = train_config
        self.ps = ps
        self._input_tensor: torch.Tensor | None = None
        self.mtp_enable_train = bool(mtp_enable and mtp_enable_train)
        self.mtp_loss_scaling_factor = config.mtp_loss_scaling_factor

        layout = build_pipeline_chunk_layout(
            config.num_hidden_layers,
            ps,
            train_config.vpp,
            vpp_chunk_id,
            num_mtp_layers=config.num_nextn_predict_layers if mtp_enable else 0,
            decoder_layer_groups=_dsa_index_share_decoder_layer_groups(config),
        )
        self.layer_indices = layout.layer_indices
        self.pre_process = layout.has_embed
        self.post_process = layout.has_head
        local_dsa_layer_indices = list(self.layer_indices)
        if layout.has_mtp:
            mtp_layers_to_build = (
                1 if config.mtp_use_repeated_layer else config.num_nextn_predict_layers
            )
            local_dsa_layer_indices.extend(
                range(
                    config.num_hidden_layers,
                    config.num_hidden_layers + mtp_layers_to_build,
                )
            )
        dsa.validate_dsa_index_share_pipeline_split(
            local_dsa_layer_indices,
            topk_freq=config.index_topk_freq,
            skip_topk_offset=config.index_skip_topk_offset,
            indexer_types=config.indexer_types,
        )
        # GLM-5 does not tie embeddings (no tie_word_embeddings HF field); the
        # attribute is preserved for the dist-opt / distckpt interface.
        self.share_embeddings_and_output_weights = bool(
            getattr(config, "tie_word_embeddings", False)
        )
        self.vision_model: nn.Module | None = None

        self.embed: VocabParallelEmbedding | None = None
        if layout.has_embed:
            self.embed = VocabParallelEmbedding(config.vocab_size, config.hidden_size, ps)

        recompute_modules = getattr(train_config, "recompute_modules", [])
        offload_modules = getattr(train_config, "offload_modules", [])
        self._retain_index_share_for_recompute = bool(
            {"full", "core_attn", "self_attn", "dsa"}
            & set([*recompute_modules, *offload_modules])
        )
        moe_act_recompute = "moe_act" in recompute_modules and "moe" not in recompute_modules
        self.layers = nn.ModuleList(
            [
                Glm5Layer(
                    config,
                    ps,
                    idx,
                    use_deepep=train_config.use_deepep,
                    router_bias_rate=router_bias_rate,
                    fp8=train_config.fp8,
                    moe_act_recompute=moe_act_recompute,
                    use_thd=use_thd,
                    dsa_cp_mode=dsa_cp_mode,
                    dsa_indexer_loss_coeff=dsa_indexer_loss_coeff,
                    dsa_indexer_use_sparse_loss=dsa_indexer_use_sparse_loss,
                    calculate_per_token_loss=calculate_per_token_loss,
                )
                for idx in self.layer_indices
            ]
        )

        self.norm: nn.Module | None = None
        self.head: VocabParallelOutput | None = None
        if layout.has_head:
            self.norm = te.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.head = VocabParallelOutput(config.vocab_size, config.hidden_size, ps)

        self.mtp_embed: VocabParallelEmbedding | None = None
        self.mtp: Glm5MTPBlock | None = None
        if mtp_enable and config.num_nextn_predict_layers > 0 and layout.has_mtp:
            mtp_embedding = self.embed
            if mtp_embedding is None:
                mtp_embedding = VocabParallelEmbedding(config.vocab_size, config.hidden_size, ps)
                self.mtp_embed = mtp_embedding
            self.mtp = Glm5MTPBlock(
                config,
                ps,
                embedding=mtp_embedding,
                use_deepep=train_config.use_deepep,
                router_bias_rate=router_bias_rate,
                fp8=train_config.fp8,
                moe_act_recompute=moe_act_recompute,
                use_thd=use_thd,
                detach_encoder=mtp_detach_encoder,
                repeated_layer=config.mtp_use_repeated_layer,
                dsa_cp_mode=dsa_cp_mode,
                dsa_indexer_loss_coeff=dsa_indexer_loss_coeff,
                dsa_indexer_use_sparse_loss=dsa_indexer_use_sparse_loss,
                calculate_per_token_loss=calculate_per_token_loss,
            )

        self.sp_params: list[nn.Parameter] = []
        if ps.tp_size > 1:
            self.sp_params = _collect_sp_grad_params(self)

    def set_input_tensor(self, input_tensor):
        if isinstance(input_tensor, list):
            if len(input_tensor) > 1:
                raise ValueError("Glm5Model expects a single pipeline input tensor.")
            input_tensor = input_tensor[0] if input_tensor else None
        self._input_tensor = input_tensor

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        packed_seq_params=None,
        labels: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        temperature: float | torch.Tensor = 1.0,
        use_fused_kernels: bool = False,
        calculate_entropy: bool = False,
    ) -> dict:
        if self.embed is not None:
            assert input_ids is not None
            h = self.embed(input_ids)
        else:
            if hidden_states is None:
                hidden_states = self._input_tensor
            assert hidden_states is not None
            h = hidden_states

        fp8_ctx = (
            te.fp8_autocast(enabled=True, fp8_recipe=build_fp8_recipe(self.train_config))
            if self.train_config.fp8
            else nullcontext()
        )
        with fp8_ctx:
            dsa_index_share_state = (
                DSAIndexShareState(
                    retain_for_recompute=self._retain_index_share_for_recompute
                )
                if self.config.uses_dsa_index_share
                else None
            )
            if self.embed is not None:
                h = scatter_to_sequence_parallel(h, self.ps)
            for layer in self.layers:
                h = layer(
                    h,
                    position_ids=position_ids,
                    packed_seq_params=packed_seq_params,
                    dsa_index_share_state=dsa_index_share_state,
                )

        output = {"hidden_states": h}
        if self.head is not None:
            assert self.norm is not None
            hidden_for_head = self.norm(h)
            mtp_hidden_states = self._apply_mtp(
                hidden_for_head,
                input_ids=input_ids,
                position_ids=position_ids,
                packed_seq_params=packed_seq_params,
                dsa_index_share_state=dsa_index_share_state,
            )
            if mtp_hidden_states is not None:
                output["mtp_hidden_states"] = mtp_hidden_states
            if labels is not None:
                temperature_value = _temperature_to_float(temperature)
                mtp_result = self._apply_mtp_loss(
                    hidden_for_head,
                    mtp_hidden_states=mtp_hidden_states,
                    labels=labels,
                    loss_mask=loss_mask,
                    packed_seq_params=packed_seq_params,
                    temperature=temperature_value,
                    use_fused_kernels=use_fused_kernels,
                )
                if mtp_result is not None:
                    hidden_for_head, mtp_loss = mtp_result
                    output["mtp_loss"] = mtp_loss
                labels_sb = labels.transpose(0, 1).contiguous()
                if use_fused_kernels:
                    hidden_full = gather_from_sequence_parallel(hidden_for_head, self.ps)
                    log_probs, entropy = linear_cross_entropy(
                        hidden_full,
                        self._head_weight_for_fused_ce(hidden_full),
                        labels_sb,
                        temperature_value,
                        self.ps.tp_group,
                    )
                    output["loss"] = (-log_probs).mean()
                    output["log_probs"] = log_probs.transpose(0, 1).contiguous()
                    if calculate_entropy:
                        output["entropy"] = entropy.transpose(0, 1).contiguous()
                else:
                    logits = self.head(hidden_for_head)
                    if temperature_value != 1.0:
                        logits = logits / temperature_value
                    loss = vocab_parallel_cross_entropy(logits, labels_sb, self.ps.tp_group)
                    output["loss"] = loss.mean()
                    output["log_probs"] = (-loss).transpose(0, 1).contiguous()
                    if calculate_entropy:
                        entropy = vocab_parallel_entropy(logits, self.ps.tp_group)
                        output["entropy"] = entropy.transpose(0, 1).contiguous()
            else:
                logits = self.head(hidden_for_head)
                output["logits"] = self.head.gather(logits).transpose(0, 1).contiguous()
                if mtp_hidden_states is not None:
                    output["mtp_logits"] = [
                        self.head.gather(self.head(mtp_hidden)).transpose(0, 1).contiguous()
                        for mtp_hidden in mtp_hidden_states
                    ]
        if dsa_index_share_state is not None:
            dsa_index_share_state.finish_forward()
        return output

    def _apply_mtp(
        self,
        hidden_states: torch.Tensor,
        *,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        packed_seq_params,
        dsa_index_share_state: DSAIndexShareState | None,
    ) -> list[torch.Tensor] | None:
        if self.mtp is None:
            return None
        if input_ids is None:
            if self.mtp_enable_train:
                raise ValueError("MTP training requires input_ids.")
            return None
        return self.mtp(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden_states,
            packed_seq_params=packed_seq_params,
            dsa_index_share_state=dsa_index_share_state,
        )

    def _apply_mtp_loss(
        self,
        hidden_states: torch.Tensor,
        *,
        mtp_hidden_states: list[torch.Tensor] | None,
        labels: torch.Tensor,
        loss_mask: torch.Tensor | None,
        packed_seq_params,
        temperature: float,
        use_fused_kernels: bool,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if mtp_hidden_states is None:
            return None
        if not self.mtp_enable_train:
            return None
        if loss_mask is None:
            mtp_loss_mask = torch.ones_like(labels, dtype=torch.float32)
        else:
            mtp_loss_mask = loss_mask.to(dtype=torch.float32).clone()
        mtp_labels = labels.clone()

        mtp_loss_values = []
        for mtp_hidden in mtp_hidden_states:
            mtp_labels, _ = _roll_mtp_left(
                mtp_labels,
                packed_seq_params=packed_seq_params,
                dims=-1,
            )
            mtp_loss_mask, num_tokens = _roll_mtp_left(
                mtp_loss_mask,
                packed_seq_params=packed_seq_params,
                dims=-1,
            )
            labels_sb = mtp_labels.transpose(0, 1).contiguous()
            mask_sb = mtp_loss_mask.transpose(0, 1).contiguous()

            if use_fused_kernels:
                mtp_hidden_full = gather_from_sequence_parallel(mtp_hidden, self.ps)
                log_probs, _entropy = linear_cross_entropy(
                    mtp_hidden_full,
                    self._head_weight_for_fused_ce(mtp_hidden_full),
                    labels_sb,
                    temperature,
                    self.ps.tp_group,
                )
                token_loss = -log_probs
            else:
                logits = self.head(mtp_hidden)
                if temperature != 1.0:
                    logits = logits / temperature
                token_loss = vocab_parallel_cross_entropy(logits, labels_sb, self.ps.tp_group)

            token_loss = token_loss * mask_sb.to(dtype=token_loss.dtype)
            num_tokens = num_tokens.to(dtype=token_loss.dtype).clamp_min(1.0)
            mtp_loss_values.append(token_loss.sum() / num_tokens)

            mtp_loss_scale = self.mtp_loss_scaling_factor / max(len(mtp_hidden_states), 1)
            hidden_states = MTPLossAutoScaler.apply(
                hidden_states,
                mtp_loss_scale * token_loss / num_tokens,
            )

        if not mtp_loss_values:
            return None
        return (
            hidden_states,
            torch.stack([loss.detach().float() for loss in mtp_loss_values]).mean(),
        )

    def _head_weight_for_fused_ce(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert self.head is not None
        weight = self.head.col.linear.weight
        return (
            weight if weight.dtype == hidden_states.dtype else weight.to(dtype=hidden_states.dtype)
        )


__all__ = [
    "DenseMLP",
    "DynamicSparseAttention",
    "Glm5DSAAttention",
    "Glm5Layer",
    "Glm5MTPBlock",
    "Glm5MTPLayer",
    "Glm5Model",
    "Glm5SigmoidTopKRouter",
    "MoELayer",
    "MTPLossAutoScaler",
    "SharedExpert",
]
