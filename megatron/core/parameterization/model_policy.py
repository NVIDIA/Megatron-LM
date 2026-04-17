# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from torch import Tensor

from megatron.core.utils import init_method_normal, mup_scaled_init_method_normal, scaled_init_method_normal

from .spec import ResolvedScalingContext, build_resolved_scaling_context


@dataclass(frozen=True)
class ResolvedModelPolicy:
    context: ResolvedScalingContext

    @property
    def enabled(self) -> bool:
        return self.context.enabled

    @property
    def residual_branch_multiplier(self) -> float:
        return self.context.depth_mult**self.context.residual_branch_depth_power

    @property
    def dense_block_out_proj_init_multiplier(self) -> float:
        return self.context.depth_mult**self.context.block_out_proj_init_depth_power

    def resolve_attention_softmax_scale(
        self, *, softmax_scale: Optional[float], kv_channels: int
    ) -> Optional[float]:
        if softmax_scale is not None or not self.enabled:
            return softmax_scale
        base_head_scale = (
            1.0
            if self.context.references.base_head_dim is None
            else self.context.references.base_head_dim**0.5
        )
        return base_head_scale / (kv_channels**self.context.attention_scale_power)

    def build_hidden_init_method(self, *, init_method_std: float):
        if not self.enabled:
            return init_method_normal(init_method_std)
        return init_method_normal(init_method_std / math.sqrt(self.context.width_mult))

    def build_default_output_layer_init_method(
        self, *, init_method_std: float, num_layers: int, is_hybrid_model: bool
    ):
        multiplier = 2.0 if not is_hybrid_model else 1.0
        if self.enabled:
            return mup_scaled_init_method_normal(
                init_method_std,
                num_layers,
                self.context.width_mult,
                multiplier=multiplier,
            )
        return scaled_init_method_normal(
            init_method_std,
            num_layers,
            multiplier=multiplier,
        )

    def dense_block_output_init_method(
        self,
        *,
        default_init_method,
        init_method_std: float,
        num_layers: int,
        is_hybrid_model: bool,
        output_layer_init_method_is_user_provided: bool,
        apply_depth_hook: bool = True,
    ):
        if output_layer_init_method_is_user_provided:
            return default_init_method
        if not apply_depth_hook:
            return default_init_method
        if self.dense_block_out_proj_init_multiplier == 1.0:
            return default_init_method

        multiplier = 2.0 if not is_hybrid_model else 1.0
        std = init_method_std / math.sqrt(multiplier * num_layers)
        if self.enabled:
            std = std / math.sqrt(self.context.width_mult)
        std = std * self.dense_block_out_proj_init_multiplier
        return functools.partial(torch.nn.init.normal_, mean=0.0, std=std)

    def output_layer_init_method(
        self,
        *,
        share_embeddings_and_output_weights: bool,
        default_init_method,
        embedding_init_method,
    ):
        if self.enabled and not share_embeddings_and_output_weights:
            return embedding_init_method
        return default_init_method

    def mark_embedding_class_parameters(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        if not self.enabled:
            return
        for param in parameters:
            param.is_embedding_parameter = True

    def scale_embedding_activations(self, embeddings: Tensor) -> Tensor:
        if not self.enabled or self.context.embedding_mult == 1.0:
            return embeddings
        return embeddings * self.context.embedding_mult

    def scale_output_logits(self, logits: Tensor) -> Tensor:
        if not self.enabled or self.context.output_mult == 1.0:
            return logits
        return logits * self.context.output_mult

    def scale_residual_branch_output(
        self, output_with_bias: tuple[Tensor, Tensor | None]
    ) -> tuple[Tensor, Tensor | None]:
        if self.residual_branch_multiplier == 1.0:
            return output_with_bias

        output, bias = output_with_bias
        scaled_output = output * self.residual_branch_multiplier
        scaled_bias = None if bias is None else bias * self.residual_branch_multiplier
        return scaled_output, scaled_bias


def build_resolved_model_policy(config) -> ResolvedModelPolicy:
    cached = getattr(config, '_resolved_model_policy', None)
    if cached is not None:
        return cached

    policy = ResolvedModelPolicy(build_resolved_scaling_context(config))
    setattr(config, '_resolved_model_policy', policy)
    return policy
