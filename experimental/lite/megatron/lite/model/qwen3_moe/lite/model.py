# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Native Qwen3MoE: TransformerLayer + Qwen3MoEModel.

Attention and MoE come from primitive/modules; this file only
defines the model-specific composition (Layer stacking, PP layout,
loss computation).
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn

from megatron.lite.model.qwen3_moe.config import Qwen3MoEConfig
from megatron.lite.primitive import transformer_engine as te
from megatron.lite.primitive.modules.dispatcher import TokenDispatcher
from megatron.lite.primitive.modules.experts import Experts
from megatron.lite.primitive.modules.gqa import GQAttention
from megatron.lite.primitive.modules.lora import LoraConfig
from megatron.lite.primitive.modules.moe_ep_chunk_overlap import EPChunkExecution
from megatron.lite.primitive.modules.moe_ep_chunk_overlap_policy import (
    validate_ep_chunk_overlap_config,
)
from megatron.lite.primitive.modules.router import TopKRouter
from megatron.lite.primitive.ops.cross_entropy import vocab_parallel_cross_entropy
from megatron.lite.primitive.ops.linear_cross_entropy import linear_cross_entropy
from megatron.lite.primitive.ops.logprob import vocab_parallel_entropy
from megatron.lite.primitive.parallel import (
    ParallelState,
    VanillaColumnParallelLinear,
    VocabParallelEmbedding,
    VocabParallelOutput,
    build_pipeline_chunk_layout,
    gather_from_sequence_parallel,
    roll_packed_thd_left,
    scatter_to_sequence_parallel,
)
from megatron.lite.primitive.utils import build_fp8_recipe

# ---------------------------------------------------------------------------
# MoE Layer (thin assembly over megatron.lite.primitive.modules)
# ---------------------------------------------------------------------------


def validate_chunked_ep_mtp(*, enable_ep_chunk_overlap: bool, mtp_enable: bool) -> None:
    """Reject the unqualified MTP composition before allocation."""
    if enable_ep_chunk_overlap and mtp_enable:
        raise ValueError("ChunkedEP with MTP is unsupported; disable MTP or ChunkedEP.")


class _Qwen3TransformerLayerFullRecomputeFunction(torch.autograd.Function):
    """Full layer checkpoint with ChunkedEP owning only MoE recomputation."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        position_ids: torch.Tensor | None,
        packed_seq_params,
        layer: "TransformerLayer",
        park_chunked_ep_after_backward: bool,
        *params: torch.Tensor,
    ):
        # ``Function.apply`` must make the initial whole-layer pass graph-free:
        # saving a post-attention residual or the MLP norm output would restore
        # the 48-layer activation growth that full recompute is meant to remove.
        if torch.is_grad_enabled():
            raise RuntimeError("Qwen3 full-recompute layer forward must run with grad disabled")
        if torch.is_tensor(position_ids) and position_ids.requires_grad:
            raise RuntimeError("Qwen3 full-recompute position_ids must not require gradients")
        ctx.param_ids = tuple(id(param) for param in params)
        ctx.cpu_rng_state = torch.get_rng_state()
        ctx.cuda_device = x.device if x.is_cuda and torch.cuda.is_initialized() else None
        ctx.cuda_rng_state = (
            torch.cuda.get_rng_state(ctx.cuda_device) if ctx.cuda_device is not None else None
        )
        del params
        ctx.layer = layer
        ctx.park_chunked_ep_after_backward = park_chunked_ep_after_backward
        ctx.position_ids = position_ids
        ctx.packed_seq_params = packed_seq_params
        ctx.save_for_backward(x.detach())
        return layer._ep_chunk_full_recompute_forward(
            x, position_ids=position_ids, packed_seq_params=packed_seq_params
        ).detach()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x_saved,) = ctx.saved_tensors
        layer = ctx.layer
        x = x_saved.detach().requires_grad_(True)
        attention_params = tuple(layer.attn.parameters())
        norm_params = tuple(layer.mlp_norm.parameters())
        router_params = tuple(layer.moe.router.parameters())
        expert_params = tuple(layer.moe.experts.parameters())
        if ctx.param_ids != tuple(
            id(param) for param in (*attention_params, *norm_params, *router_params, *expert_params)
        ):
            raise RuntimeError("Qwen3 full-recompute parameter order changed")
        current_cpu_rng_state = torch.get_rng_state()
        current_cuda_rng_state = (
            torch.cuda.get_rng_state(ctx.cuda_device) if ctx.cuda_device is not None else None
        )
        try:
            torch.set_rng_state(ctx.cpu_rng_state)
            if ctx.cuda_rng_state is not None:
                torch.cuda.set_rng_state(ctx.cuda_rng_state, ctx.cuda_device)
            with torch.enable_grad():
                attention_out = layer.attn(
                    x, position_ids=ctx.position_ids, packed_seq_params=ctx.packed_seq_params
                )
                residual = x + attention_out
                norm_out = layer.mlp_norm(residual)
                assert (
                    layer.moe.chunked_ep is not None and layer.moe.chunked_ep.fused_op is not None
                )
                grad_norm, router_grads, expert_grads = (
                    layer.moe.chunked_ep.fused_op.forward_backward(norm_out, grad_output)
                )
                differentiable = (x, *attention_params, *norm_params)
                required = tuple(value for value in differentiable if value.requires_grad)
                required_grads = torch.autograd.grad(
                    (norm_out, residual), required, (grad_norm, grad_output), allow_unused=True
                )
                if ctx.park_chunked_ep_after_backward:
                    layer.moe.chunked_ep.finish_backward(x)
        finally:
            torch.set_rng_state(current_cpu_rng_state)
            if current_cuda_rng_state is not None:
                torch.cuda.set_rng_state(current_cuda_rng_state, ctx.cuda_device)

        grads_by_id = {
            id(value): grad for value, grad in zip(required, required_grads, strict=True)
        }
        grad_x = grads_by_id.get(id(x))
        param_grads = (
            *[grads_by_id.get(id(param)) for param in attention_params],
            *[grads_by_id.get(id(param)) for param in norm_params],
            *router_grads,
            *expert_grads,
        )
        if len(param_grads) != len(ctx.param_ids):
            raise RuntimeError("Qwen3 full-recompute parameter gradient order changed")
        return (grad_x, None, None, None, None, *param_grads)


class MoELayer(nn.Module):
    def __init__(
        self,
        config: Qwen3MoEConfig,
        ps: ParallelState,
        *,
        use_deepep: bool = True,
        router_bias_rate: float = 0.0,
        fp8: bool = False,
        moe_act_recompute: bool = False,
        enable_ep_chunk_overlap: bool = False,
        ep_chunk_max_token_rows_per_rank: int | None = None,
        ep_chunk_count: int = 2,
        ep_chunk_full_recompute: bool = False,
        lora_config: LoraConfig | dict | None = None,
    ):
        super().__init__()
        validate_qwen3_ep_chunk_recompute_composition(
            enable_ep_chunk_overlap=enable_ep_chunk_overlap,
            ep_chunk_full_recompute=ep_chunk_full_recompute,
            recompute_modules=[],
        )
        validate_ep_chunk_overlap_config(
            enable_ep_chunk_overlap,
            use_deepep=use_deepep,
            ep_size=ps.ep_size,
            topk=config.num_experts_per_tok,
            max_token_rows_per_rank=ep_chunk_max_token_rows_per_rank,
            chunk_count=ep_chunk_count,
        )
        # Match Qwen3-MoE's `load_balancing_type="none"` setting: no aux loss.
        self.router = TopKRouter(
            config, ps, router_bias_rate=router_bias_rate, compute_aux_loss=False
        )
        self.experts = Experts(
            config,
            ps,
            fp8=fp8,
            moe_act_recompute=moe_act_recompute,
            delay_wgrad_compute=enable_ep_chunk_overlap,
            lora_config=lora_config,
        )
        self.ep_chunk_full_recompute = ep_chunk_full_recompute
        self.chunked_ep = (
            EPChunkExecution(
                router=self.router,
                experts=self.experts,
                dispatcher_factory=lambda _slot: TokenDispatcher(
                    config.num_experts, config.hidden_size, ps, use_deepep=True
                ),
                max_input_rows=ep_chunk_max_token_rows_per_rank,
                hidden_size=config.hidden_size,
                expert_intermediate_size=getattr(config, "moe_intermediate_size", None),
                topk=config.num_experts_per_tok,
                ep_size=ps.ep_size,
                ep_group=ps.tp_ep_group,
                chunk_count=ep_chunk_count,
                retain_backward=not ep_chunk_full_recompute,
            )
            if enable_ep_chunk_overlap
            else None
        )
        self.dispatcher = (
            None
            if self.chunked_ep
            else TokenDispatcher(config.num_experts, config.hidden_size, ps, use_deepep=use_deepep)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chunked_ep is not None:
            return self.chunked_ep.forward_op(x)

        assert self.dispatcher is not None
        input_shape = x.shape
        if x.dim() == 3:
            x_2d = x.view(-1, x.size(-1))
        else:
            x_2d = x

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
        del dispatched, tpe, permuted_probs
        combined = self.dispatcher.combine(expert_out)
        del expert_out

        return combined.view(input_shape).to(x.dtype)


# ---------------------------------------------------------------------------
# Transformer Layer + Model
# ---------------------------------------------------------------------------

_SP_GRAD_SUFFIXES: tuple[str, ...] = (
    ".attn.qkv.linear.layer_norm_weight",
    ".mlp_norm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
    ".moe.router.gate.weight",
    ".enorm.weight",
    ".hnorm.weight",
    ".final_layernorm.weight",
)


def validate_qwen3_ep_chunk_recompute_composition(
    *,
    enable_ep_chunk_overlap: bool,
    ep_chunk_full_recompute: bool,
    recompute_modules: list[str] | tuple[str, ...],
) -> None:
    """Validate Qwen3 recompute composition without leaking policy to primitives."""
    if ep_chunk_full_recompute and not enable_ep_chunk_overlap:
        raise ValueError("ep_chunk_full_recompute=True requires enable_ep_chunk_overlap=True")
    if (
        enable_ep_chunk_overlap
        and not ep_chunk_full_recompute
        and any(module in {"moe", "full"} for module in recompute_modules)
    ):
        raise ValueError(
            "normal ChunkedEP conflicts with outer MoE recompute; enable "
            "ep_chunk_full_recompute or remove moe/full recompute"
        )


def _qwen3_moe_act_recompute_requested(
    recompute_modules: list[str], *, ep_chunk_full_recompute: bool
) -> bool:
    """Keep recompute-policy interpretation in the Qwen composition layer."""
    return (
        "moe_act" in recompute_modules
        and "moe" not in recompute_modules
        and not ep_chunk_full_recompute
    )


def _collect_sp_grad_params(model: nn.Module) -> list[nn.Parameter]:
    """Collect non-TP-sharded params needing coalesced all_reduce after backward."""
    params = []
    for name, p in model.named_parameters():
        if any(name.endswith(s) for s in _SP_GRAD_SUFFIXES) or name == "norm.weight":
            params.append(p)
    return params


class TransformerLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3MoEConfig,
        ps: ParallelState,
        layer_idx: int,
        *,
        use_deepep: bool = True,
        router_bias_rate: float = 0.0,
        fp8: bool = False,
        moe_act_recompute: bool = False,
        use_thd: bool = False,
        lora_config: LoraConfig | dict | None = None,
        enable_ep_chunk_overlap: bool = False,
        ep_chunk_max_token_rows_per_rank: int | None = None,
        ep_chunk_count: int = 2,
        ep_chunk_full_recompute: bool = False,
        attention_backend: str = "te",
    ):
        super().__init__()
        self.layer_idx = layer_idx

        # Declaration order follows MC's TransformerLayer (self_attention →
        # pre_mlp_layernorm → mlp). `named_parameters()` iterates in
        # declaration order, and MC's `DistributedDataParallel` lays out
        # gradient buckets by that order; mismatching it changes fp32 master
        # shard layouts and breaks bitwise alignment from step 1 onwards.
        self.attn = GQAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            ps=ps,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
            use_thd=use_thd,
            qkv_layout="mcore",
            lora_config=lora_config,
            attention_backend=attention_backend,
        )
        self.mlp_norm = te.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.moe = MoELayer(
            config,
            ps,
            use_deepep=use_deepep,
            router_bias_rate=router_bias_rate,
            fp8=fp8,
            moe_act_recompute=moe_act_recompute,
            enable_ep_chunk_overlap=enable_ep_chunk_overlap,
            ep_chunk_max_token_rows_per_rank=ep_chunk_max_token_rows_per_rank,
            ep_chunk_count=ep_chunk_count,
            ep_chunk_full_recompute=ep_chunk_full_recompute,
            lora_config=lora_config,
        )

    def _ep_chunk_full_recompute_forward(
        self, x: torch.Tensor, *, position_ids: torch.Tensor | None, packed_seq_params
    ) -> torch.Tensor:
        """Run the graph-free initial pass for the composition-owned checkpoint."""
        residual = x
        h = self.attn(x, position_ids=position_ids, packed_seq_params=packed_seq_params)
        x = residual + h
        residual = x
        h = self.mlp_norm(x)
        assert self.moe.chunked_ep is not None
        moe_out = self.moe.chunked_ep.forward_op(h)
        return residual + moe_out

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        packed_seq_params=None,
        park_chunked_ep_after_backward: bool = False,
    ) -> torch.Tensor:
        if self.moe.ep_chunk_full_recompute and torch.is_grad_enabled():
            params = (
                *tuple(self.attn.parameters()),
                *tuple(self.mlp_norm.parameters()),
                *tuple(self.moe.router.parameters()),
                *tuple(self.moe.experts.parameters()),
            )
            return _Qwen3TransformerLayerFullRecomputeFunction.apply(
                x, position_ids, packed_seq_params, self, park_chunked_ep_after_backward, *params
            )
        residual = x
        h = self.attn(x, position_ids=position_ids, packed_seq_params=packed_seq_params)
        x = residual + h

        residual = x
        h = self.mlp_norm(x)
        moe_out = self.moe(h)
        x = residual + moe_out

        return x


class MTPLossAutoScaler(torch.autograd.Function):
    """Attach MTP loss gradients to the main LM hidden state."""

    main_loss_backward_scale: float = 1.0

    @staticmethod
    def forward(ctx, output: torch.Tensor, mtp_loss: torch.Tensor):
        ctx.save_for_backward(mtp_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (mtp_loss,) = ctx.saved_tensors
        scaled_mtp_grad = torch.ones_like(mtp_loss) * MTPLossAutoScaler.main_loss_backward_scale
        return grad_output, scaled_mtp_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor | float) -> None:
        if isinstance(scale, torch.Tensor):
            scale = float(scale.detach().float().item())
        MTPLossAutoScaler.main_loss_backward_scale = float(scale)


class MultiTokenPredictionLayer(nn.Module):
    """MCore-style MTP layer for the THD SFT lite path."""

    def __init__(
        self,
        config: Qwen3MoEConfig,
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
        lora_config: LoraConfig | dict | None,
    ):
        super().__init__()
        self.ps = ps
        self.embedding = embedding
        self.detach_encoder = detach_encoder
        self.enorm = te.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = te.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = VanillaColumnParallelLinear(
            config.hidden_size * 2, config.hidden_size, ps, sp=ps.tp_size > 1, gather_output=True
        )
        self.transformer_layer = TransformerLayer(
            config,
            ps,
            config.num_hidden_layers + layer_idx,
            use_deepep=use_deepep,
            router_bias_rate=router_bias_rate,
            fp8=fp8,
            moe_act_recompute=moe_act_recompute,
            use_thd=use_thd,
            lora_config=lora_config,
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        attention_position_ids = (
            rotary_position_ids if rotary_position_ids is not None else position_ids
        )
        input_ids, _ = roll_packed_thd_left(input_ids, packed_seq_params=packed_seq_params, dims=-1)
        if position_ids is not None:
            position_ids, _ = roll_packed_thd_left(
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
            hidden_states, position_ids=attention_position_ids, packed_seq_params=packed_seq_params
        )
        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states, input_ids, position_ids


class MultiTokenPredictionBlock(nn.Module):
    def __init__(
        self,
        config: Qwen3MoEConfig,
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
        lora_config: LoraConfig | dict | None,
    ):
        super().__init__()
        self.num_layers = config.num_nextn_predict_layers
        self.repeated_layer = repeated_layer
        layers_to_build = 1 if repeated_layer else self.num_layers
        self.layers = nn.ModuleList(
            [
                MultiTokenPredictionLayer(
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
                    lora_config=lora_config,
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
            )
            outputs.append(hidden_states)
        return outputs


def _temperature_to_float(temperature: float | torch.Tensor) -> float:
    if isinstance(temperature, torch.Tensor):
        if temperature.numel() != 1:
            raise ValueError(
                "Megatron Lite fused/MTP SFT currently supports scalar temperature only."
            )
        return float(temperature.detach().float().item())
    return float(temperature)


class Qwen3MoEModel(nn.Module):
    def __init__(
        self,
        config: Qwen3MoEConfig,
        ps: ParallelState,
        vpp: int | None = None,
        vpp_chunk_id: int | None = None,
        *,
        use_deepep: bool = False,
        fp8: bool = False,
        recompute_modules: list[str] | None = None,
        router_bias_rate: float = 0.0,
        use_thd: bool = False,
        mtp_enable: bool = False,
        mtp_enable_train: bool = False,
        mtp_detach_encoder: bool = False,
        lora_config: LoraConfig | dict | None = None,
        enable_ep_chunk_overlap: bool = False,
        ep_chunk_max_token_rows_per_rank: int | None = None,
        ep_chunk_count: int = 2,
        ep_chunk_full_recompute: bool = False,
        attention_backend: str = "te",
    ):
        super().__init__()
        validate_chunked_ep_mtp(
            enable_ep_chunk_overlap=enable_ep_chunk_overlap, mtp_enable=mtp_enable
        )
        validate_qwen3_ep_chunk_recompute_composition(
            enable_ep_chunk_overlap=enable_ep_chunk_overlap,
            ep_chunk_full_recompute=ep_chunk_full_recompute,
            recompute_modules=recompute_modules or [],
        )
        self.config = config
        self.ps = ps
        self.fp8 = fp8
        self.mtp_enable_train = bool(mtp_enable and mtp_enable_train)
        self.mtp_loss_scaling_factor = config.mtp_loss_scaling_factor
        # The backend name is the model's entire attention-backend knowledge.
        # Backend tuning is deliberately not configurable here: policy lives in
        # the backend primitive (see resolve_magi_attention_config for magi).
        self.attention_backend = attention_backend
        self._input_tensor: torch.Tensor | None = None
        layout = build_pipeline_chunk_layout(config.num_hidden_layers, ps, vpp, vpp_chunk_id)
        self.layer_indices = layout.layer_indices
        has_embed = layout.has_embed
        has_head = layout.has_head
        self.pre_process = has_embed
        self.post_process = has_head
        self.share_embeddings_and_output_weights = False

        self.embed: VocabParallelEmbedding | None = None
        if has_embed:
            self.embed = VocabParallelEmbedding(config.vocab_size, config.hidden_size, ps)

        _recompute = recompute_modules or []
        moe_act_recompute = _qwen3_moe_act_recompute_requested(
            _recompute, ep_chunk_full_recompute=ep_chunk_full_recompute
        )
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    config,
                    ps,
                    idx,
                    use_deepep=use_deepep,
                    router_bias_rate=router_bias_rate,
                    fp8=fp8,
                    moe_act_recompute=moe_act_recompute,
                    use_thd=use_thd,
                    lora_config=lora_config,
                    attention_backend=attention_backend,
                    enable_ep_chunk_overlap=enable_ep_chunk_overlap,
                    ep_chunk_max_token_rows_per_rank=ep_chunk_max_token_rows_per_rank,
                    ep_chunk_count=ep_chunk_count,
                    ep_chunk_full_recompute=ep_chunk_full_recompute,
                )
                for idx in self.layer_indices
            ]
        )

        self.norm: nn.Module | None = None
        self.head: VocabParallelOutput | None = None
        if has_head:
            self.norm = te.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.head = VocabParallelOutput(config.vocab_size, config.hidden_size, ps)

        self.mtp_embed: VocabParallelEmbedding | None = None
        self.mtp: MultiTokenPredictionBlock | None = None
        if mtp_enable and config.num_nextn_predict_layers > 0 and self.head is not None:
            mtp_embedding = self.embed
            if mtp_embedding is None:
                mtp_embedding = VocabParallelEmbedding(config.vocab_size, config.hidden_size, ps)
                self.mtp_embed = mtp_embedding
            self.mtp = MultiTokenPredictionBlock(
                config,
                ps,
                embedding=mtp_embedding,
                use_deepep=use_deepep,
                router_bias_rate=router_bias_rate,
                fp8=fp8,
                moe_act_recompute=moe_act_recompute,
                use_thd=use_thd,
                detach_encoder=mtp_detach_encoder,
                repeated_layer=config.mtp_use_repeated_layer,
                lora_config=lora_config,
            )

        self.sp_params: list[nn.Parameter] = []
        if ps.tp_size > 1:
            self.sp_params = _collect_sp_grad_params(self)

    def set_attention_backend(self, attention_backend: str) -> None:
        """Hot-swap the attention backend on a built model.

        Both backends are parameter-free, so the swap leaves parameters,
        buffers, and optimizer state untouched: te-trained checkpoints resume
        under magi and vice versa (only TE's ``_extra_state`` metadata keys
        follow the te backend). The batch protocol re-reads
        ``self.attention_backend`` for every microbatch, so the swap takes
        effect on the next step. Only switch at step boundaries.
        """
        if attention_backend == "magi":
            if self.ps.cp_size <= 1:
                raise ValueError("MagiAttention requires context parallel size CP>1.")
            if self.ps.pp_size != 1:
                raise ValueError("MagiAttention supports PP=1 only.")
            if self.config.num_nextn_predict_layers > 0:
                raise ValueError("MagiAttention does not currently support MTP.")
        for layer in self.layers:
            layer.attn.set_attention_backend(attention_backend)
        self.attention_backend = attention_backend

    def set_input_tensor(self, input_tensor):
        if isinstance(input_tensor, list):
            if len(input_tensor) > 1:
                raise ValueError("Qwen3MoEModel expects a single pipeline input tensor.")
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
        return_log_probs: bool = True,
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
            te.fp8_autocast(enabled=True, fp8_recipe=build_fp8_recipe())
            if self.fp8
            else nullcontext()
        )

        with fp8_ctx:
            if self.embed is not None:
                h = scatter_to_sequence_parallel(h, self.ps)
            final_local_backward_chunked_ep = next(
                (
                    layer
                    for layer in self.layers
                    if layer.moe.chunked_ep is not None
                    and layer.moe.chunked_ep.fused_op is not None
                ),
                None,
            )
            for layer in self.layers:
                h = layer(
                    h,
                    position_ids=position_ids,
                    packed_seq_params=packed_seq_params,
                    park_chunked_ep_after_backward=layer is final_local_backward_chunked_ep,
                )
            # Head path is SP-aware: norm runs on SP-sharded [S/tp, B, H] and
            # head's internal all-gather happens inside VocabParallelOutput.
            # Mirrors MC GPTModel's final_layernorm → output_layer(sp=True).

        output = {"hidden_states": h}

        def reset_forward_chunked_ep() -> None:
            """Park the shared forward arena after all Qwen3 MoE consumers."""
            for layer in self.layers:
                if layer.moe.chunked_ep is not None:
                    layer.moe.chunked_ep.finish_forward(h)
                    break

        if self.head is not None:
            hidden_for_head = self.norm(h)

            if labels is not None:
                temperature_value = _temperature_to_float(temperature)
                mtp_result = self._apply_mtp_loss(
                    hidden_for_head,
                    input_ids=input_ids,
                    position_ids=position_ids,
                    labels=labels,
                    loss_mask=loss_mask,
                    packed_seq_params=packed_seq_params,
                    temperature=temperature_value,
                    use_fused_kernels=use_fused_kernels,
                )
                if mtp_result is not None:
                    hidden_for_head, mtp_loss = mtp_result
                    output["mtp_loss"] = mtp_loss
                reset_forward_chunked_ep()
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
                    token_loss = -log_probs
                    output["loss"] = token_loss.mean()
                    if return_log_probs:
                        output["log_probs"] = log_probs.transpose(0, 1).contiguous()
                    if calculate_entropy:
                        output["entropy"] = entropy.transpose(0, 1).contiguous()
                else:
                    logits = self.head(hidden_for_head)
                    if temperature_value != 1.0:
                        logits = logits / temperature_value
                    token_loss = vocab_parallel_cross_entropy(logits, labels_sb, self.ps.tp_group)
                    output["loss"] = token_loss.mean()
                    if return_log_probs:
                        output["log_probs"] = (-token_loss).transpose(0, 1).contiguous()
                    if calculate_entropy:
                        entropy = vocab_parallel_entropy(logits, self.ps.tp_group)
                        output["entropy"] = entropy.transpose(0, 1).contiguous()

            if labels is None:
                reset_forward_chunked_ep()
                logits = self.head(hidden_for_head)
                output["logits"] = self.head.gather(logits)
        else:
            reset_forward_chunked_ep()

        return output

    def _apply_mtp_loss(
        self,
        hidden_states: torch.Tensor,
        *,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        labels: torch.Tensor,
        loss_mask: torch.Tensor | None,
        packed_seq_params,
        temperature: float,
        use_fused_kernels: bool,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.mtp is None:
            return None
        if not self.mtp_enable_train:
            return None
        if input_ids is None:
            raise ValueError("MTP training requires input_ids.")
        if loss_mask is None:
            loss_mask = torch.ones_like(labels, dtype=torch.float32)
        else:
            loss_mask = loss_mask.to(dtype=torch.float32)

        mtp_hidden_states = self.mtp(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden_states,
            packed_seq_params=packed_seq_params,
        )

        mtp_labels = labels.clone()
        mtp_loss_mask = loss_mask.clone()
        mtp_loss_values = []
        for mtp_hidden in mtp_hidden_states:
            mtp_labels, _ = roll_packed_thd_left(
                mtp_labels, packed_seq_params=packed_seq_params, dims=-1
            )
            mtp_loss_mask, num_tokens = roll_packed_thd_left(
                mtp_loss_mask, packed_seq_params=packed_seq_params, dims=-1
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
                hidden_states, mtp_loss_scale * token_loss / num_tokens
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
        if weight.dtype == hidden_states.dtype:
            return weight
        return weight.to(dtype=hidden_states.dtype)


__all__ = [
    "MoELayer",
    "MTPLossAutoScaler",
    "MultiTokenPredictionBlock",
    "MultiTokenPredictionLayer",
    "Qwen3MoEModel",
    "TransformerLayer",
]
