# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Kimi K2 lite native model."""

from __future__ import annotations

import os
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te

from megatron.lite.model.kimi_k2.config import KimiK2Config
from megatron.lite.primitive.modules.attention import MultiLatentAttention
from megatron.lite.primitive.modules.dispatcher import TokenDispatcher
from megatron.lite.primitive.modules.experts import Experts
from megatron.lite.primitive.modules.mtp import MTPLossAutoScaler
from megatron.lite.primitive.modules.router import SigmoidTopKRouter
from megatron.lite.primitive.ops.cross_entropy import vocab_parallel_cross_entropy
from megatron.lite.primitive.ops.linear_cross_entropy import linear_cross_entropy
from megatron.lite.primitive.ops.logprob import vocab_parallel_entropy
from megatron.lite.primitive.ops.sp_ops import ReduceScatterDim0
from megatron.lite.primitive.kernels.swiglu import bias_swiglu_impl
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
from megatron.lite.primitive.utils import build_fp8_recipe

_SP_GRAD_SUFFIXES: tuple[str, ...] = (
    ".input_layernorm.weight",
    ".self_attention.linear_q_down_proj.weight",
    ".self_attention.linear_q_up_proj.linear.layer_norm_weight",
    ".self_attention.linear_kv_down_proj.weight",
    ".self_attention.linear_kv_up_proj.linear.layer_norm_weight",
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


class DenseMLP(nn.Module):
    def __init__(self, config: KimiK2Config, ps: ParallelState):
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
    def __init__(self, config: KimiK2Config, ps: ParallelState):
        super().__init__()
        self.ps = ps
        ffn = config.shared_expert_intermediate_size
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
        config: KimiK2Config,
        ps: ParallelState,
        *,
        use_deepep: bool,
        router_bias_rate: float,
        fp8: bool,
        moe_act_recompute: bool,
    ):
        super().__init__()
        if fp8:
            raise NotImplementedError("Kimi K2 lite MoE fp8 training is not implemented yet.")
        self.router = SigmoidTopKRouter(
            config,
            ps,
            router_bias_rate=router_bias_rate,
            compute_aux_loss=True,
            use_pre_softmax=True,
            router_dtype=torch.float32,
            expert_bias_persistent=True,
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


class KimiK2Layer(nn.Module):
    def __init__(
        self,
        config: KimiK2Config,
        ps: ParallelState,
        layer_idx: int,
        *,
        use_deepep: bool = False,
        router_bias_rate: float = 0.0,
        fp8: bool = False,
        moe_act_recompute: bool = False,
        use_thd: bool = False,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = te.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attention = MultiLatentAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            ps=ps,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
            use_thd=use_thd,
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

    def forward(self, x: torch.Tensor, packed_seq_params=None) -> torch.Tensor:
        x = x + self.self_attention(self.input_layernorm(x), packed_seq_params=packed_seq_params)
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


class KimiK2MTPLayer(nn.Module):
    def __init__(
        self,
        config: KimiK2Config,
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
        self.transformer_layer = KimiK2Layer(
            config,
            ps,
            config.num_hidden_layers + layer_idx,
            use_deepep=use_deepep,
            router_bias_rate=router_bias_rate,
            fp8=fp8,
            moe_act_recompute=moe_act_recompute,
            use_thd=use_thd,
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
        del rotary_position_ids
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
        hidden_states = self.transformer_layer(hidden_states, packed_seq_params=packed_seq_params)
        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states, input_ids, position_ids


class KimiK2MTPBlock(nn.Module):
    def __init__(
        self,
        config: KimiK2Config,
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
    ):
        super().__init__()
        self.num_layers = config.num_nextn_predict_layers
        self.repeated_layer = repeated_layer
        layers_to_build = 1 if repeated_layer else self.num_layers
        self.layers = nn.ModuleList(
            [
                KimiK2MTPLayer(
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
            raise ValueError("KimiK2Model supports scalar temperature only.")
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


class KimiK2Model(nn.Module):
    def __init__(
        self,
        config: KimiK2Config,
        train_config,
        ps: ParallelState,
        *,
        vpp_chunk_id: int | None = None,
        router_bias_rate: float = 0.0,
        use_thd: bool = False,
        hf_path: str = "",
        attention_backend_override: str | None = None,
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
        )
        self.layer_indices = layout.layer_indices
        self.pre_process = layout.has_embed
        self.post_process = layout.has_head
        self.share_embeddings_and_output_weights = bool(config.tie_word_embeddings)
        self.vision_model: nn.Module | None = None

        self.embed: VocabParallelEmbedding | None = None
        if layout.has_embed:
            self.embed = VocabParallelEmbedding(config.vocab_size, config.hidden_size, ps)

        recompute_modules = getattr(train_config, "recompute_modules", [])
        moe_act_recompute = "moe_act" in recompute_modules and "moe" not in recompute_modules
        self.layers = nn.ModuleList(
            [
                KimiK2Layer(
                    config,
                    ps,
                    idx,
                    use_deepep=train_config.use_deepep,
                    router_bias_rate=router_bias_rate,
                    fp8=train_config.fp8,
                    moe_act_recompute=moe_act_recompute,
                    use_thd=use_thd,
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
        self.mtp: KimiK2MTPBlock | None = None
        if mtp_enable and config.num_nextn_predict_layers > 0 and layout.has_mtp:
            mtp_embedding = self.embed
            if mtp_embedding is None:
                mtp_embedding = VocabParallelEmbedding(config.vocab_size, config.hidden_size, ps)
                self.mtp_embed = mtp_embedding
            self.mtp = KimiK2MTPBlock(
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
            )

        self.sp_params: list[nn.Parameter] = []
        if ps.tp_size > 1:
            self.sp_params = _collect_sp_grad_params(self)

    def set_input_tensor(self, input_tensor):
        if isinstance(input_tensor, list):
            if len(input_tensor) > 1:
                raise ValueError("KimiK2Model expects a single pipeline input tensor.")
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
            if self.embed is not None:
                h = scatter_to_sequence_parallel(h, self.ps)
            for layer in self.layers:
                h = layer(h, packed_seq_params=packed_seq_params)

        output = {"hidden_states": h}
        if self.head is not None:
            assert self.norm is not None
            hidden_for_head = self.norm(h)
            mtp_hidden_states = self._apply_mtp(
                hidden_for_head,
                input_ids=input_ids,
                position_ids=position_ids,
                packed_seq_params=packed_seq_params,
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
        return output

    def _apply_mtp(
        self,
        hidden_states: torch.Tensor,
        *,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        packed_seq_params,
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
    "KimiK2MTPBlock",
    "KimiK2MTPLayer",
    "KimiK2Layer",
    "KimiK2Model",
    "MoELayer",
    "MTPLossAutoScaler",
    "MultiLatentAttention",
    "SharedExpert",
]
