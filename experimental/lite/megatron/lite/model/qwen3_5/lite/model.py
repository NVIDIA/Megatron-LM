"""Qwen3.5 lite model: uses shared MCoreFullAttn/MCoreGDN/MCoreMoELayer ops."""

from __future__ import annotations

import os
from contextlib import nullcontext

import torch
import torch.nn as nn
import transformer_engine.pytorch as te

from megatron.lite.model.qwen3_5.config import Qwen35Config
from megatron.lite.model.qwen3_5.mc_ops import (
    MCoreContext,
    MCoreFullAttn,
    MCoreGDN,
    MCoreMoELayer,
    _build_vision_model,
    build_mcore_context,
)
from megatron.lite.primitive.ops.cross_entropy import vocab_parallel_cross_entropy
from megatron.lite.primitive.ops.linear_cross_entropy import linear_cross_entropy
from megatron.lite.primitive.ops.logprob import vocab_parallel_entropy
from megatron.lite.primitive.parallel import (
    ColumnParallelLinear,
    ParallelState,
    VocabParallelEmbedding,
    VocabParallelOutput,
    build_pipeline_chunk_layout,
    gather_from_sequence_parallel,
    roll_packed_thd_left,
    scatter_to_sequence_parallel,
)
from megatron.lite.primitive.utils import build_fp8_recipe

# ---------------------------------------------------------------------------
# SP gradient suffixes — MC naming: self_attention / pre_mlp_layernorm / mlp
# ---------------------------------------------------------------------------

_SP_GRAD_SUFFIXES: tuple[str, ...] = (
    ".self_attention.linear_qkv.layer_norm_weight",
    ".self_attention.q_layernorm.weight",
    ".self_attention.k_layernorm.weight",
    ".self_attention.in_proj.layer_norm_weight",
    ".self_attention.norm.weight",
    ".pre_mlp_layernorm.weight",
    ".mlp.router.weight",
    ".mlp.shared_experts.gate_weight",
    ".enorm.weight",
    ".hnorm.weight",
    ".final_layernorm.weight",
)

def _collect_sp_grad_params(model: nn.Module) -> list[nn.Parameter]:
    params = []
    for name, param in model.named_parameters():
        if any(name.endswith(s) for s in _SP_GRAD_SUFFIXES) or name == "norm.weight":
            params.append(param)
    return params


# ---------------------------------------------------------------------------
# Qwen35Layer
# ---------------------------------------------------------------------------


class Qwen35Layer(nn.Module):
    """Mixed attention layer: MCoreGDN or MCoreFullAttn + MCoreMoELayer."""

    def __init__(
        self,
        config: Qwen35Config,
        train_config,
        ps: ParallelState,
        layer_idx: int,
        *,
        mcore_context: MCoreContext,
        use_deepep: bool = True,
        router_bias_rate: float = 0.0,
        fp8: bool = False,
        moe_act_recompute: bool = False,
        use_thd: bool = False,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        layer_type = config.layer_type_at(layer_idx)
        self._layer_type = layer_type
        is_mtp_layer = layer_idx >= config.num_hidden_layers
        mtp_layer_number = layer_idx - config.num_hidden_layers + 1 if is_mtp_layer else None

        if layer_type == "full_attention":
            self.self_attention: nn.Module = MCoreFullAttn(
                config,
                train_config,
                ps,
                layer_idx,
                mcore_context,
                layer_number=mtp_layer_number,
            )
        else:
            self.self_attention = MCoreGDN(
                config,
                train_config,
                ps,
                layer_idx,
                mcore_context,
                layer_number=mtp_layer_number,
            )

        self.pre_mlp_layernorm = te.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, zero_centered_gamma=True
        )
        self.mlp: nn.Module = MCoreMoELayer(
            config,
            train_config,
            ps,
            layer_idx,
            mcore_context,
            layer_number=mtp_layer_number,
            is_mtp_layer=is_mtp_layer,
        )

        # D3: verify mc_ops classes are used (not reverted to lite-specific impls)
        assert isinstance(self.self_attention, MCoreFullAttn | MCoreGDN), (
            f"Qwen35Layer[{layer_idx}] self_attention is {type(self.self_attention).__name__}, "
            f"expected MCoreFullAttn or MCoreGDN"
        )
        assert isinstance(self.mlp, MCoreMoELayer), (
            f"Qwen35Layer[{layer_idx}] mlp is {type(self.mlp).__name__}, expected MCoreMoELayer"
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        packed_seq_params=None,
    ) -> torch.Tensor:
        residual = x
        if self._layer_type == "full_attention":
            h = self.self_attention(x, position_ids=position_ids, packed_seq_params=packed_seq_params)
        else:
            h = self.self_attention(x, packed_seq_params=packed_seq_params)
        x = residual + h

        residual = x
        h = self.pre_mlp_layernorm(x)
        h = self.mlp(h)
        x = residual + h
        return x


class MTPLossAutoScaler(torch.autograd.Function):
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


class Qwen35MTPLayer(nn.Module):
    def __init__(
        self,
        config: Qwen35Config,
        train_config,
        ps: ParallelState,
        layer_idx: int,
        *,
        mcore_context: MCoreContext,
        embedding: VocabParallelEmbedding,
        router_bias_rate: float,
        use_thd: bool,
        detach_encoder: bool,
    ):
        super().__init__()
        self.ps = ps
        self.layer_idx = layer_idx
        self.embedding = embedding
        self.detach_encoder = detach_encoder
        self.enorm = te.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, zero_centered_gamma=True
        )
        self.hnorm = te.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, zero_centered_gamma=True
        )
        self.eh_proj = ColumnParallelLinear(
            config.hidden_size * 2,
            config.hidden_size,
            ps,
            gather_output=True,
            sequence_parallel=ps.tp_size > 1,
        )
        self.transformer_layer = Qwen35Layer(
            config,
            train_config,
            ps,
            config.num_hidden_layers + layer_idx,
            mcore_context=mcore_context,
            use_deepep=train_config.use_deepep,
            router_bias_rate=router_bias_rate,
            fp8=train_config.fp8,
            moe_act_recompute=(
                "moe_act" in train_config.recompute_modules
                and "moe" not in train_config.recompute_modules
            ),
            use_thd=use_thd,
        )
        self.final_layernorm = te.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, zero_centered_gamma=True
        )

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


class Qwen35MTPBlock(nn.Module):
    def __init__(
        self,
        config: Qwen35Config,
        train_config,
        ps: ParallelState,
        *,
        mcore_context: MCoreContext,
        embedding: VocabParallelEmbedding,
        router_bias_rate: float,
        use_thd: bool,
        detach_encoder: bool,
    ):
        super().__init__()
        self.num_layers = config.num_nextn_predict_layers
        layers_to_build = 1 if config.mtp_use_repeated_layer else self.num_layers
        self.repeated_layer = config.mtp_use_repeated_layer
        self.layers = nn.ModuleList(
            [
                Qwen35MTPLayer(
                    config,
                    train_config,
                    ps,
                    idx,
                    mcore_context=mcore_context,
                    embedding=embedding,
                    router_bias_rate=router_bias_rate,
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
                rotary_position_ids=rotary_position_ids,
                hidden_states=hidden_states,
                packed_seq_params=packed_seq_params,
            )
            outputs.append(hidden_states)
        return outputs


def _temperature_to_float(temperature: float | torch.Tensor) -> float:
    if isinstance(temperature, torch.Tensor):
        if temperature.numel() != 1:
            raise ValueError("Qwen35Model fused/MTP SFT supports scalar temperature only.")
        return float(temperature.detach().float().item())
    return float(temperature)


def _ensure_mrope_position_ids(position_ids: torch.Tensor | None) -> torch.Tensor | None:
    if position_ids is None:
        return None
    if position_ids.dim() == 2:
        return position_ids.unsqueeze(0).expand(3, -1, -1).contiguous()
    if position_ids.dim() == 3:
        if position_ids.shape[0] == 1:
            return position_ids.expand(3, -1, -1).contiguous()
        if position_ids.shape[0] == 3:
            return position_ids
    raise ValueError(
        "Qwen3.5 multimodal RoPE expects position_ids with shape (B, S), "
        "(1, B, S), or (3, B, S)."
    )


# ---------------------------------------------------------------------------
# Qwen35Model
# ---------------------------------------------------------------------------


class Qwen35Model(nn.Module):
    """Qwen3.5 lite: Embedding → N×Qwen35Layer → RMSNorm → Output (PP-aware)."""

    def __init__(
        self,
        config: Qwen35Config,
        train_config,
        ps: ParallelState,
        *,
        vpp_chunk_id: int | None = None,
        router_bias_rate: float = 0.001,
        use_thd: bool = False,
        hf_path: str = "",
        attention_backend_override: str | None = None,
        mtp_enable: bool = False,
        mtp_enable_train: bool = False,
        mtp_detach_encoder: bool = False,
    ):
        super().__init__()
        self.config = config
        self.train_config = train_config
        self.ps = ps
        self.mtp_enable_train = bool(mtp_enable and mtp_enable_train)
        self.mtp_loss_scaling_factor = config.mtp_loss_scaling_factor
        self._input_tensor: torch.Tensor | None = None

        layout = build_pipeline_chunk_layout(
            config.num_hidden_layers, ps, train_config.vpp, vpp_chunk_id,
        )
        self.layer_indices = layout.layer_indices
        has_embed = layout.has_embed
        has_head = layout.has_head
        self.pre_process = has_embed
        self.post_process = has_head
        self.share_embeddings_and_output_weights = False

        use_deepep = train_config.use_deepep
        fp8 = train_config.fp8
        deterministic_embedding = getattr(train_config, "deterministic_embedding", None)
        if deterministic_embedding is None:
            deterministic_embedding = bool(getattr(train_config, "deterministic", False))
        moe_act_recompute = (
            "moe_act" in train_config.recompute_modules
            and "moe" not in train_config.recompute_modules
        )
        mcore_context = build_mcore_context(
            config, train_config, hf_path=hf_path,
            attention_backend_override=attention_backend_override,
            vp_stage=vpp_chunk_id,
        )
        self.mbridge_bridge = mcore_context.bridge

        # vision_model registered before embed to match Qwen3_5VLModel param order
        if has_embed:
            self.vision_model: nn.Module | None = _build_vision_model(self.mbridge_bridge)
        else:
            self.vision_model = None

        self.embed: VocabParallelEmbedding | None = None
        if has_embed:
            self.embed = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                ps,
                deterministic=deterministic_embedding,
            )

        self.layers = nn.ModuleList(
            [
                Qwen35Layer(
                    config,
                    train_config,
                    ps,
                    idx,
                    mcore_context=mcore_context,
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
            self.norm = te.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, zero_centered_gamma=True)
            self.head = VocabParallelOutput(config.vocab_size, config.hidden_size, ps)

        self.mtp_embed: VocabParallelEmbedding | None = None
        self.mtp: Qwen35MTPBlock | None = None
        if mtp_enable and config.num_nextn_predict_layers > 0 and self.head is not None:
            mtp_embedding = self.embed
            if mtp_embedding is None:
                mtp_embedding = VocabParallelEmbedding(
                    config.vocab_size,
                    config.hidden_size,
                    ps,
                    deterministic=deterministic_embedding,
                )
                self.mtp_embed = mtp_embedding
            self.mtp = Qwen35MTPBlock(
                config,
                train_config,
                ps,
                mcore_context=mcore_context,
                embedding=mtp_embedding,
                router_bias_rate=router_bias_rate,
                use_thd=use_thd,
                detach_encoder=mtp_detach_encoder,
            )

        self.sp_params: list[nn.Parameter] = []
        if ps.tp_size > 1:
            self.sp_params = _collect_sp_grad_params(self)

    def set_input_tensor(self, input_tensor):
        if isinstance(input_tensor, list):
            if len(input_tensor) > 1:
                raise ValueError("Qwen35Model expects a single pipeline input tensor.")
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

        if packed_seq_params is not None:
            assert position_ids is not None, (
                "THD path requires caller-supplied position_ids with shape (3, 1, T)"
            )
        elif position_ids is None and input_ids is not None:
            from megatron.core import parallel_state as mpu  # pyright: ignore[reportMissingImports]
            if mpu.is_initialized() and mpu.get_context_parallel_world_size() > 1:
                raise ValueError(
                    "CP>1 requires caller-supplied FULL position_ids (3, B, S); "
                    "cannot auto-build from CP-sliced input_ids."
                )
            _b, _s = input_ids.shape
            _pos = torch.arange(_s, device=input_ids.device, dtype=torch.long)
            position_ids = _pos.unsqueeze(0).unsqueeze(0).expand(3, _b, _s).contiguous()
        position_ids = _ensure_mrope_position_ids(position_ids)

        fp8_ctx = (
            te.fp8_autocast(enabled=True, fp8_recipe=build_fp8_recipe(self.train_config))
            if self.train_config.fp8
            else nullcontext()
        )

        with fp8_ctx:
            if self.embed is not None:
                h = scatter_to_sequence_parallel(h, self.ps)
            for layer in self.layers:
                h = layer(h, position_ids=position_ids, packed_seq_params=packed_seq_params)

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
                        self.head.col.linear.weight,
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
            if labels is None:
                logits = self.head(hidden_for_head)
                output["logits"] = self.head.gather(logits).transpose(0, 1).contiguous()
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
        if self.mtp is None or not self.mtp_enable_train:
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
                    self.head.col.linear.weight,
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
        mtp_loss = torch.stack([loss.detach().float() for loss in mtp_loss_values]).mean()
        return hidden_states, mtp_loss


__all__ = ["MTPLossAutoScaler", "Qwen35Layer", "Qwen35MTPBlock", "Qwen35MTPLayer", "Qwen35Model"]
