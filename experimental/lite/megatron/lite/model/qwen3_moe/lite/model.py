"""Native Qwen3MoE: TransformerLayer + Qwen3MoEModel.

Attention and MoE come from primitive/modules; this file only
defines the model-specific composition (Layer stacking, PP layout,
loss computation).
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn
import transformer_engine.pytorch as te

from megatron.lite.model.qwen3_moe.config import Qwen3MoEConfig
from megatron.lite.primitive.modules.dispatcher import TokenDispatcher
from megatron.lite.primitive.modules.experts import Experts
from megatron.lite.primitive.modules.gqa import GQAttention
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
    ):
        super().__init__()
        # Mirror bridge's `load_balancing_type="none"` for Qwen3-MoE: no aux loss.
        self.router = TopKRouter(
            config, ps,
            router_bias_rate=router_bias_rate,
            compute_aux_loss=False,
        )
        self.experts = Experts(config, ps, fp8=fp8, moe_act_recompute=moe_act_recompute)
        self.dispatcher = TokenDispatcher(
            config.num_experts, config.hidden_size, ps, use_deepep=use_deepep,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    ):
        super().__init__()
        self.layer_idx = layer_idx

        # Declaration order follows MC's TransformerLayer (self_attention →
        # pre_mlp_layernorm → mlp). `named_parameters()` iterates in
        # declaration order, and MC's `DistributedDataParallel` lays out
        # gradient buckets by that order; mismatching it would make step-0
        # fp32 master shard layouts diverge from the MC reference path and break
        # bitwise alignment from step 1 onwards.
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
        )
        self.mlp_norm = te.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.moe = MoELayer(
            config, ps,
            use_deepep=use_deepep,
            router_bias_rate=router_bias_rate,
            fp8=fp8,
            moe_act_recompute=moe_act_recompute,
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        packed_seq_params=None,
    ) -> torch.Tensor:
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
    ):
        super().__init__()
        self.ps = ps
        self.embedding = embedding
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
        self.transformer_layer = TransformerLayer(
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
        attention_position_ids = rotary_position_ids if rotary_position_ids is not None else position_ids
        input_ids, _ = roll_packed_thd_left(
            input_ids,
            packed_seq_params=packed_seq_params,
            dims=-1,
        )
        if position_ids is not None:
            position_ids, _ = roll_packed_thd_left(
                position_ids,
                packed_seq_params=packed_seq_params,
                dims=-1,
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
            raise ValueError("Megatron Lite fused/MTP SFT currently supports scalar temperature only.")
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
    ):
        super().__init__()
        self.config = config
        self.ps = ps
        self.fp8 = fp8
        self.mtp_enable_train = bool(mtp_enable and mtp_enable_train)
        self.mtp_loss_scaling_factor = config.mtp_loss_scaling_factor
        self._input_tensor: torch.Tensor | None = None
        layout = build_pipeline_chunk_layout(
            config.num_hidden_layers, ps, vpp, vpp_chunk_id,
        )
        self.layer_indices = layout.layer_indices
        has_embed = layout.has_embed
        has_head = layout.has_head

        self.embed: VocabParallelEmbedding | None = None
        if has_embed:
            self.embed = VocabParallelEmbedding(config.vocab_size, config.hidden_size, ps)

        _recompute = recompute_modules or []
        moe_act_recompute = "moe_act" in _recompute and "moe" not in _recompute
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    config, ps, idx,
                    use_deepep=use_deepep,
                    router_bias_rate=router_bias_rate,
                    fp8=fp8,
                    moe_act_recompute=moe_act_recompute,
                    use_thd=use_thd,
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
            )

        self.sp_params: list[nn.Parameter] = []
        if ps.tp_size > 1:
            self.sp_params = _collect_sp_grad_params(self)

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
            for layer in self.layers:
                h = layer(h, position_ids=position_ids, packed_seq_params=packed_seq_params)
            # Head path is SP-aware: norm runs on SP-sharded [S/tp, B, H] and
            # head's internal all-gather happens inside VocabParallelOutput.
            # Mirrors MC GPTModel's final_layernorm → output_layer(sp=True).

        output = {"hidden_states": h}

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
                    token_loss = vocab_parallel_cross_entropy(logits, labels_sb, self.ps.tp_group)
                    output["loss"] = token_loss.mean()
                    output["log_probs"] = (-token_loss).transpose(0, 1).contiguous()
                    if calculate_entropy:
                        entropy = vocab_parallel_entropy(logits, self.ps.tp_group)
                        output["entropy"] = entropy.transpose(0, 1).contiguous()

            if labels is None:
                logits = self.head(hidden_for_head)
                output["logits"] = self.head.gather(logits)

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
                mtp_labels,
                packed_seq_params=packed_seq_params,
                dims=-1,
            )
            mtp_loss_mask, num_tokens = roll_packed_thd_left(
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
            hidden_states = MTPLossAutoScaler.apply(hidden_states, mtp_loss_scale * token_loss / num_tokens)

        if not mtp_loss_values:
            return None
        return hidden_states, torch.stack([loss.detach().float() for loss in mtp_loss_values]).mean()

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
