# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Qwen3.5 lite native model.

This implementation keeps the lightweight qwen3_moe/lite composition style
and does not wrap Megatron-Core layer modules. It still reuses Megatron Lite
parallel/TE primitives and small Megatron atomic RoPE helpers where those are
already used by other native Megatron Lite modules.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn
import transformer_engine.pytorch as te

from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl
from megatron.lite.model.qwen3_5.config import Qwen35Config
from megatron.lite.primitive.modules.dispatcher import TokenDispatcher
from megatron.lite.primitive.modules.experts import Experts
from megatron.lite.primitive.modules.gated_delta_net import GatedDeltaNet
from megatron.lite.primitive.modules.gqa import GQAttention as FullAttention
from megatron.lite.primitive.modules.gqa import split_grouped_qkvg as _split_grouped_qkvg
from megatron.lite.primitive.modules.mrope import MultimodalRotaryEmbedding as Qwen35MRoPE
from megatron.lite.primitive.modules.mtp import MTPBlock, MTPDecoderLayer, MTPLossAutoScaler
from megatron.lite.primitive.modules.router import TopKRouter
from megatron.lite.primitive.ops.cross_entropy import vocab_parallel_cross_entropy
from megatron.lite.primitive.ops.linear_cross_entropy import linear_cross_entropy
from megatron.lite.primitive.ops.logprob import vocab_parallel_entropy
from megatron.lite.primitive.parallel import (
    ColumnParallelLinear,
    ParallelState,
    RowParallelLinear,
    VocabParallelEmbedding,
    VocabParallelOutput,
    build_pipeline_chunk_layout,
    gather_from_sequence_parallel,
    roll_packed_thd_left,
    scatter_to_sequence_parallel,
)
from megatron.lite.primitive.utils import build_fp8_recipe

_SP_GRAD_SUFFIXES: tuple[str, ...] = (
    ".full_attn.qkv.linear.layer_norm_weight",
    ".full_attn.q_norm.weight",
    ".full_attn.k_norm.weight",
    ".linear_attn.in_proj.linear.layer_norm_weight",
    ".linear_attn.norm.weight",
    ".mlp_norm.weight",
    ".moe.router.gate.weight",
    ".moe.shared_expert.shared_gate.weight",
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


def _swiglu(x: torch.Tensor) -> torch.Tensor:
    return bias_swiglu_impl(x, bias=None)


def _qwen_mrope_section(config: Qwen35Config) -> list[int]:
    section = getattr(config, "mrope_section", None)
    if section is not None:
        return list(section)
    rotary_half = max(int(config.rotary_dim // 2), 1)
    base = rotary_half // 3
    return [base, base, rotary_half - 2 * base]


class SharedExpert(nn.Module):
    _stream: torch.cuda.Stream | None = None

    class _CopyToTPRegion(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, group):
            ctx.group = group
            return x

        @staticmethod
        def backward(ctx, grad):
            group = ctx.group
            if group is not None and dist.get_world_size(group) > 1:
                dist.all_reduce(grad, group=group)
            return grad, None

    class _PlainTELinear(nn.Module):
        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.linear = te.Linear(
                in_features,
                out_features,
                bias=False,
                params_dtype=torch.bfloat16,
                parallel_mode=None,
                sequence_parallel=False,
                tp_group=None,
                tp_size=1,
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x)

    def __init__(
        self, config: Qwen35Config, ps: ParallelState, *, use_plain_te_linear: bool = False
    ):
        super().__init__()
        ffn = config.shared_expert_intermediate_size
        if use_plain_te_linear and ps.tp_size == 1:
            self.gate_up = self._PlainTELinear(config.hidden_size, ffn * 2)
            self.down = self._PlainTELinear(ffn, config.hidden_size)
        else:
            self.gate_up = ColumnParallelLinear(config.hidden_size, ffn * 2, ps, bias=False)
            self.down = RowParallelLinear(ffn, config.hidden_size, ps, bias=False)
        self.shared_gate = nn.Linear(config.hidden_size, 1, bias=False)
        self.tp_group = ps.tp_group
        self.use_mcore_overlap_graph = bool(use_plain_te_linear and ps.tp_size == 1)

    @staticmethod
    def _get_stream() -> torch.cuda.Stream:
        if SharedExpert._stream is None:
            SharedExpert._stream = torch.cuda.Stream()
        return SharedExpert._stream

    @staticmethod
    def _set_grad_fn_sequence_sr(tensor: torch.Tensor) -> None:
        grad_fn = getattr(tensor, "grad_fn", None)
        if grad_fn is not None and hasattr(grad_fn, "_set_sequence_nr"):
            grad_fn._set_sequence_nr(torch.iinfo(torch.int).max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_val = self.shared_gate(x).sigmoid()
        fc1_input = x
        if self.use_mcore_overlap_graph:
            fc1_input = self._CopyToTPRegion.apply(x, self.tp_group)
            self._set_grad_fn_sequence_sr(fc1_input)
        output = self.down(_swiglu(self.gate_up(fc1_input)))
        if self.use_mcore_overlap_graph:
            self._set_grad_fn_sequence_sr(output)
        return output * gate_val


class MoELayer(nn.Module):
    def __init__(
        self,
        config: Qwen35Config,
        ps: ParallelState,
        *,
        use_deepep: bool,
        router_bias_rate: float,
        fp8: bool,
        moe_act_recompute: bool,
        router_dtype: torch.dtype | None = None,
        preserve_3d_graph: bool = False,
        shared_expert_plain_te: bool = False,
    ):
        super().__init__()
        if fp8:
            raise NotImplementedError("lite qwen35 MoE fp8 is not implemented yet.")
        self.router = TopKRouter(
            config,
            ps,
            router_bias_rate=router_bias_rate,
            compute_aux_loss=True,
            router_dtype=router_dtype,
        )
        self.experts = Experts(config, ps, fp8=fp8, moe_act_recompute=moe_act_recompute)
        self.dispatcher = TokenDispatcher(
            config.num_experts, config.hidden_size, ps, use_deepep=use_deepep
        )
        self.shared_expert = SharedExpert(config, ps, use_plain_te_linear=shared_expert_plain_te)
        self.preserve_3d_graph = bool(preserve_3d_graph)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape
        x_2d = x.reshape(-1, x.size(-1))
        shared_input = x_2d.view(input_shape) if self.preserve_3d_graph else x_2d
        router_input = x if self.preserve_3d_graph else x_2d

        shared_out = None
        side_stream = None
        if x_2d.is_cuda:
            side_stream = SharedExpert._get_stream()
            side_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side_stream):
                shared_out = self.shared_expert(shared_input)
        scores, indices = self.router(router_input)
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

        if shared_out is None:
            shared_out = self.shared_expert(shared_input)
        else:
            assert side_stream is not None
            torch.cuda.current_stream().wait_stream(side_stream)
        output = routed_out.view(input_shape)
        if shared_out.shape != input_shape:
            shared_out = shared_out.view(input_shape)
        output += shared_out
        return output.to(x.dtype)


class Qwen35Layer(nn.Module):
    def __init__(
        self,
        config: Qwen35Config,
        ps: ParallelState,
        layer_idx: int,
        *,
        use_deepep: bool = False,
        router_bias_rate: float = 0.0,
        fp8: bool = False,
        moe_act_recompute: bool = False,
        use_thd: bool = False,
        deterministic: bool = False,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self._layer_type = config.layer_type_at(layer_idx)
        self.full_attn: FullAttention | None = None
        self.linear_attn: GatedDeltaNet | None = None
        if self._layer_type == "full_attention":
            self.full_attn = FullAttention(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                ps=ps,
                rms_norm_eps=config.rms_norm_eps,
                rope_theta=config.rope_theta,
                rotary_percent=config.partial_rotary_factor,
                use_thd=use_thd,
                output_gate=True,
                zero_centered_gamma=True,
                qkv_layout="mcore",
                mrope_section=_qwen_mrope_section(config),
            )
        else:
            self.linear_attn = GatedDeltaNet(
                hidden_size=config.hidden_size,
                linear_num_key_heads=config.linear_num_key_heads,
                linear_key_head_dim=config.linear_key_head_dim,
                linear_num_value_heads=config.linear_num_value_heads,
                linear_value_head_dim=config.linear_value_head_dim,
                linear_conv_kernel_dim=config.linear_conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
                ps=ps,
                deterministic=deterministic,
            )
        self.mlp_norm = te.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, zero_centered_gamma=True
        )
        self.moe = MoELayer(
            config,
            ps,
            use_deepep=use_deepep,
            router_bias_rate=router_bias_rate,
            fp8=fp8,
            moe_act_recompute=moe_act_recompute,
            router_dtype=torch.float32 if deterministic else None,
            preserve_3d_graph=deterministic,
            shared_expert_plain_te=deterministic,
        )

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor | None = None, packed_seq_params=None
    ) -> torch.Tensor:
        residual = x
        if self.full_attn is not None:
            h = self.full_attn(x, position_ids=position_ids, packed_seq_params=packed_seq_params)
        else:
            assert self.linear_attn is not None
            h = self.linear_attn(x, position_ids=position_ids, packed_seq_params=packed_seq_params)
        x = residual + h
        residual = x
        x = residual + self.moe(self.mlp_norm(x))
        return x


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
    raise ValueError("Qwen3.5 MRoPE expects position_ids shape (B,S), (1,B,S), or (3,B,S).")


class Qwen35Model(nn.Module):
    def __init__(
        self,
        config: Qwen35Config,
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
        mount_vision_model: bool = False,
    ):
        super().__init__()
        del attention_backend_override
        self.config = config
        self.train_config = train_config
        self.ps = ps
        self.mtp_enable_train = bool(mtp_enable and mtp_enable_train)
        self.mtp_loss_scaling_factor = config.mtp_loss_scaling_factor
        self._input_tensor: torch.Tensor | None = None

        layout = build_pipeline_chunk_layout(
            config.num_hidden_layers, ps, train_config.vpp, vpp_chunk_id
        )
        self.layer_indices = layout.layer_indices
        has_embed = layout.has_embed
        has_head = layout.has_head
        self.pre_process = has_embed
        self.post_process = has_head
        self.share_embeddings_and_output_weights = False
        self.vision_model: nn.Module | None = (
            _build_native_vision_model(hf_path) if has_embed and mount_vision_model else None
        )

        self.embed: VocabParallelEmbedding | None = None
        if has_embed:
            self.embed = VocabParallelEmbedding(config.vocab_size, config.hidden_size, ps)

        recompute_modules = getattr(train_config, "recompute_modules", [])
        moe_act_recompute = "moe_act" in recompute_modules and "moe" not in recompute_modules
        self.layers = nn.ModuleList(
            [
                Qwen35Layer(
                    config,
                    ps,
                    idx,
                    use_deepep=train_config.use_deepep,
                    router_bias_rate=router_bias_rate,
                    fp8=train_config.fp8,
                    moe_act_recompute=moe_act_recompute,
                    use_thd=use_thd,
                    deterministic=getattr(train_config, "deterministic", False),
                )
                for idx in self.layer_indices
            ]
        )

        self.norm: nn.Module | None = None
        self.head: VocabParallelOutput | None = None
        if has_head:
            self.norm = te.RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps, zero_centered_gamma=True
            )
            self.head = VocabParallelOutput(config.vocab_size, config.hidden_size, ps)

        self.mtp_embed: VocabParallelEmbedding | None = None
        self.mtp: MTPBlock | None = None
        if mtp_enable and config.num_nextn_predict_layers > 0 and self.head is not None:
            mtp_embedding = self.embed
            if mtp_embedding is None:
                mtp_embedding = VocabParallelEmbedding(config.vocab_size, config.hidden_size, ps)
                self.mtp_embed = mtp_embedding

            def make_mtp_layer(layer_idx: int) -> MTPDecoderLayer:
                return MTPDecoderLayer(
                    hidden_size=config.hidden_size,
                    rms_norm_eps=config.rms_norm_eps,
                    ps=ps,
                    embedding=mtp_embedding,
                    transformer_layer=Qwen35Layer(
                        config,
                        ps,
                        config.num_hidden_layers + layer_idx,
                        use_deepep=train_config.use_deepep,
                        router_bias_rate=router_bias_rate,
                        fp8=train_config.fp8,
                        moe_act_recompute=moe_act_recompute,
                        use_thd=use_thd,
                        deterministic=getattr(train_config, "deterministic", False),
                    ),
                    detach_encoder=mtp_detach_encoder,
                )

            self.mtp = MTPBlock(
                num_layers=config.num_nextn_predict_layers,
                repeated_layer=config.mtp_use_repeated_layer,
                layer_factory=make_mtp_layer,
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
            assert position_ids is not None, "THD path requires caller-supplied MRoPE position_ids."
        elif position_ids is None and input_ids is not None:
            if self.ps.cp_size > 1:
                raise ValueError("CP>1 requires caller-supplied FULL position_ids (3,B,S).")
            batch, seq = input_ids.shape
            pos = torch.arange(seq, device=input_ids.device, dtype=torch.long)
            position_ids = pos.unsqueeze(0).unsqueeze(0).expand(3, batch, seq).contiguous()
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
            assert self.norm is not None
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
                    loss = vocab_parallel_cross_entropy(logits, labels_sb, self.ps.tp_group)
                    output["loss"] = loss.mean()
                    output["log_probs"] = (-loss).transpose(0, 1).contiguous()
                    if calculate_entropy:
                        entropy = vocab_parallel_entropy(logits, self.ps.tp_group)
                        output["entropy"] = entropy.transpose(0, 1).contiguous()
            else:
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
                assert self.head is not None
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
        return (
            weight if weight.dtype == hidden_states.dtype else weight.to(dtype=hidden_states.dtype)
        )


def _iter_auto_model_class_refs(hf_config) -> list[str]:
    auto_map = getattr(hf_config, "auto_map", None) or {}
    preferred_keys = (
        "AutoModelForCausalLM",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "AutoModel",
    )
    refs: list[str] = []
    for key in preferred_keys:
        ref = auto_map.get(key)
        if isinstance(ref, (list, tuple)):
            ref = ref[0] if ref else None
        if isinstance(ref, str) and ref not in refs:
            refs.append(ref)
    return refs


def _resolve_hf_vision_cls(hf_config, hf_path: str) -> type:
    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
    except ImportError as exc:
        raise ImportError(
            "mount_vision_model=True requires transformers with dynamic module support."
        ) from exc

    errors: list[str] = []
    for class_ref in _iter_auto_model_class_refs(hf_config):
        try:
            model_cls = get_class_from_dynamic_module(class_ref, hf_path, trust_remote_code=True)
        except Exception as exc:
            errors.append(f"{class_ref}: {exc}")
            continue
        vision_cls = getattr(model_cls, "HfVisionClass", None)
        if vision_cls is not None:
            return vision_cls
        errors.append(f"{class_ref}: missing HfVisionClass")
    detail = "; ".join(errors) if errors else "config auto_map has no supported AutoModel entry"
    raise RuntimeError(f"Cannot resolve native HF vision class for Qwen3.5: {detail}.")


def _hook_fp32_rotary_emb(module: nn.Module) -> None:
    for submodule in module.modules():
        if hasattr(submodule, "inv_freq") and submodule.inv_freq is not None:
            submodule._inv_freq_fp32_original = submodule.inv_freq.detach().clone().float()

            def _hook(mod, args):
                del args
                if hasattr(mod, "_inv_freq_fp32_original"):
                    mod.inv_freq = mod._inv_freq_fp32_original.to(device=mod.inv_freq.device)

            submodule.register_forward_pre_hook(_hook)


def _hook_vision_params_avg_grad_across_tp(module: nn.Module) -> None:
    for param in module.parameters(recurse=True):
        param.average_gradients_across_tp_domain = True  # type: ignore[assignment]


def _build_native_vision_model(hf_path: str) -> nn.Module:
    if not hf_path:
        raise ValueError("mount_vision_model requires hf_path.")
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise ImportError("mount_vision_model=True requires transformers.") from exc

    hf_config = AutoConfig.from_pretrained(hf_path, trust_remote_code=True)
    vision_config = getattr(hf_config, "vision_config", None)
    if vision_config is None:
        raise RuntimeError("HF config does not expose vision_config; cannot build vision_model.")
    hf_vision_cls = _resolve_hf_vision_cls(hf_config, hf_path)
    if hasattr(hf_vision_cls, "_from_config"):
        vision = hf_vision_cls._from_config(vision_config)
    else:
        vision = hf_vision_cls(vision_config)
    _hook_fp32_rotary_emb(vision)
    _hook_vision_params_avg_grad_across_tp(vision)
    return vision.to(torch.bfloat16)


__all__ = [
    "FullAttention",
    "GatedDeltaNet",
    "MoELayer",
    "MTPLossAutoScaler",
    "Qwen35MRoPE",
    "Qwen35Layer",
    "Qwen35Model",
    "SharedExpert",
    "_split_grouped_qkvg",
]
