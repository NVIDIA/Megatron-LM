# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Connect hybrid inference residual additions to the following native RMSNorm.

The graph boundary carries two explicit tensors: the rounded residual and
the normalized input for the next layer. No tensor is exchanged through a
Python cache. The existing layer objects continue to own weights and graphs.
"""

from dataclasses import dataclass
from typing import Callable

import torch

from megatron.core.inference.utils import InferenceMode
from megatron.core.ops.normalization.backends import select_bf16_add_rmsnorm
from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.ssm.mamba_mixer import MambaMixer
from megatron.core.tensor_observation import observe_layer_residuals
from megatron.core.tensor_parallel.inference_layers import (
    InferenceLayerNormColumnParallelLinear,
    _te_rms_norm_kernel,
)
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
    get_batch_invariant_backend,
    is_batch_invariant_mode_enabled,
)
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_layer import MoETransformerLayer, TransformerLayer


@dataclass(frozen=True)
class _Norm:
    """Reference an existing weight owner without registering duplicate parameters."""

    owner: torch.nn.Module
    weight_name: str

    @property
    def weight(self):
        """Resolve the current weight, including a checkpoint refit."""
        return getattr(self.owner, self.weight_name)

    @property
    def eps(self):
        """Return the owning normalization's epsilon."""
        return self.owner.eps

    def normalize(self, hidden_states):
        """Apply the first norm, which has no preceding residual addition."""
        return _te_rms_norm_kernel(hidden_states, self.weight, self.eps)


@dataclass(frozen=True)
class _LayerPlan:
    """Execute one native branch and fuse its tail with the next input norm."""

    kind: str
    input_norm: _Norm
    next_norm: _Norm | None
    kernel: Callable

    def forward(
        self,
        layer,
        hidden_states,
        inference_normalized_input=None,
        inference_context=None,
        attention_mask=None,
        rotary_pos_emb=None,
        sequence_len_offset=None,
        padding_mask=None,
        input_ids=None,
        **kwargs,
    ):
        """Return (rounded residual, next normalized input) for eager or graph use."""
        if layer.training or torch.is_grad_enabled() or not InferenceMode.is_active():
            raise RuntimeError("Fused residual normalization requires inference mode")
        normalized = inference_normalized_input
        if normalized is None:
            normalized = self.input_norm.normalize(hidden_states)
        if self.kind == "mamba":
            output, bias = layer.mixer(
                normalized, inference_context=inference_context, inference_input_is_normalized=True
            )
        elif self.kind == "attention":
            output, bias = layer.self_attention(
                normalized,
                attention_mask=attention_mask,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                sequence_len_offset=sequence_len_offset,
                inference_input_is_normalized=True,
            )
        else:
            moe_kwargs = {"input_ids": input_ids} if input_ids is not None else {}
            output, bias = layer.mlp(normalized, padding_mask=padding_mask, **moe_kwargs)
        if bias is not None:
            raise RuntimeError("Fused residual normalization requires a bias-free branch")
        if self.next_norm is None:
            return output.add_(hidden_states), None
        normalized, residual = self.kernel(
            output, hidden_states, self.next_norm.weight, self.next_norm.eps, inplace=True
        )
        return residual, normalized


@dataclass(frozen=True)
class InferenceResidualNormPlan:
    """A stack-local execution plan; original modules retain all state ownership."""

    layers: tuple
    plans: tuple[_LayerPlan, ...]
    final_norm: _Norm | None

    def can_run(self, hidden_states, *, training, inference_context, packed_seq_params):
        """Check call-local boundaries before changing any execution behavior."""
        return (
            not training
            and all(not layer.training for layer in self.layers)
            and not torch.is_grad_enabled()
            and InferenceMode.is_active()
            and inference_context is not None
            and packed_seq_params is None
            and hidden_states.is_cuda
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.is_contiguous()
            and (
                not is_batch_invariant_mode_enabled()
                or get_batch_invariant_backend() == "te_native"
            )
            and all(
                plan.input_norm.weight.dtype == torch.bfloat16
                and plan.input_norm.weight.device == hidden_states.device
                and plan.input_norm.weight.is_contiguous()
                for plan in self.plans
            )
            and (
                self.final_norm is None
                or (
                    self.final_norm.weight.dtype == torch.bfloat16
                    and self.final_norm.weight.device == hidden_states.device
                    and self.final_norm.weight.is_contiguous()
                )
            )
        )

    def forward(self, hidden_states, **kwargs):
        """Carry both tensor outputs through existing per-layer graph managers."""
        normalized = None
        for layer, plan in zip(self.layers, self.plans):
            residual = hidden_states
            layer_kwargs = dict(
                hidden_states=hidden_states,
                inference_context=kwargs.get("inference_context"),
                inference_fuse_residual_norm=True,
                inference_normalized_input=normalized,
            )
            if plan.kind != "mamba":
                layer_kwargs.update(kwargs)
            hidden_states, normalized = layer(**layer_kwargs)
            observe_layer_residuals(layer, residual, hidden_states)
        return normalized if self.final_norm is not None else hidden_states


def _standalone_norm(module):
    """Recognize the plain TE contract, excluding norm subclasses and residual fusion."""
    if (
        type(module).__name__ == "RMSNorm"
        and type(module).__module__.startswith("transformer_engine.")
        and not module.zero_centered_gamma
        and not module.returns_residual
    ):
        return _Norm(module, "weight")
    return None


def _layer_input_norm(layer):
    """Recognize ordinary single-branch layers with an explicit input norm owner."""
    if type(layer) is MambaLayer and type(layer.mixer) is MambaMixer:
        projection = layer.mixer.in_proj
        if (
            isinstance(layer.norm, IdentityOp)
            and type(projection) is InferenceLayerNormColumnParallelLinear
        ):
            return "mamba", _Norm(projection, "layer_norm_weight")
    if type(layer) in (TransformerLayer, MoETransformerLayer):
        if not isinstance(layer.cross_attention, IdentityOp):
            return None
        if (
            type(layer.self_attention) is SelfAttention
            and isinstance(layer.input_layernorm, IdentityOp)
            and isinstance(layer.mlp, IdentityOp)
            and type(layer.self_attention.linear_qkv) is InferenceLayerNormColumnParallelLinear
        ):
            return "attention", _Norm(layer.self_attention.linear_qkv, "layer_norm_weight")
        if isinstance(layer.self_attention, IdentityOp) and isinstance(layer.mlp, MoELayer):
            norm = _standalone_norm(layer.pre_mlp_layernorm)
            if norm is not None:
                return "moe", norm
    return None


def _supports_config(config):
    """Exclude numerical and execution policies handled by the native layer path."""
    return not (
        config.transformer_impl != "inference_optimized"
        or config.normalization != "RMSNorm"
        or config.layernorm_zero_centered_gamma
        or config.add_bias_linear
        or config.add_qkv_bias
        or config.hidden_dropout != 0
        or config.fp8
        or config.fp4
        or config.enable_mhc_connections
        or config.wide_residual is not None
        or config.moe_shortcut_connection
        or config.context_parallel_size != 1
        or config.inference_fuse_tp_communication
        or config.cuda_graph_impl not in (None, "none", "local")
        or config.cuda_graph_modules
        or config.fine_grained_activation_offloading
        or config.recompute_granularity is not None
    )


def build_inference_residual_norm_plan(layers, config, final_norm=None):
    """Bind the default optimization only for supported ordinary hybrid layers."""
    if (
        not layers
        or not _supports_config(config)
        or any(
            not _supports_config(layer.config)
            or layer.config.fp32_residual_connection
            or layer.config.params_dtype != config.params_dtype
            or layer.config.hidden_size != config.hidden_size
            for layer in layers
        )
    ):
        return None
    kernel = select_bf16_add_rmsnorm(
        config.hidden_size, config.params_dtype, config.fp32_residual_connection
    )
    if kernel is None:
        return None
    inputs = [_layer_input_norm(layer) for layer in layers]
    if any(item is None for item in inputs):
        return None
    final = _standalone_norm(final_norm) if final_norm is not None else None
    if final_norm is not None and final is None:
        return None
    plans = tuple(
        _LayerPlan(kind, norm, inputs[i + 1][1] if i + 1 < len(inputs) else final, kernel)
        for i, (kind, norm) in enumerate(inputs)
    )
    for layer, plan in zip(layers, plans):
        layer._inference_residual_norm = plan
        layer.inference_cudagraph_io_map = {"hidden_states": 0, "inference_normalized_input": 1}
    return InferenceResidualNormPlan(tuple(layers), plans, final)
