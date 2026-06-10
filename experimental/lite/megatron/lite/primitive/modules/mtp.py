# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Multi-token prediction primitives."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import transformer_engine.pytorch as te

from megatron.lite.primitive.parallel import (
    ParallelState,
    VanillaColumnParallelLinear,
    VocabParallelEmbedding,
    roll_packed_thd_left,
    scatter_to_sequence_parallel,
)

__all__ = ["MTPBlock", "MTPDecoderLayer", "MTPLossAutoScaler"]


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


class MTPDecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        rms_norm_eps: float,
        ps: ParallelState,
        embedding: VocabParallelEmbedding,
        transformer_layer: nn.Module,
        detach_encoder: bool,
    ):
        super().__init__()
        self.ps = ps
        self.embedding = embedding
        self.detach_encoder = detach_encoder
        self.enorm = te.RMSNorm(hidden_size, eps=rms_norm_eps, zero_centered_gamma=True)
        self.hnorm = te.RMSNorm(hidden_size, eps=rms_norm_eps, zero_centered_gamma=True)
        self.eh_proj = VanillaColumnParallelLinear(
            hidden_size * 2, hidden_size, ps, sp=ps.tp_size > 1, gather_output=True
        )
        self.transformer_layer = transformer_layer
        self.final_layernorm = te.RMSNorm(hidden_size, eps=rms_norm_eps, zero_centered_gamma=True)

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
        decoder_input = scatter_to_sequence_parallel(self.embedding(input_ids), self.ps)
        if self.detach_encoder:
            decoder_input = decoder_input.detach()
            hidden_states = hidden_states.detach()
        decoder_input = self.enorm(decoder_input)
        hidden_states = self.hnorm(hidden_states)
        hidden_states = torch.cat((decoder_input, hidden_states), dim=-1)
        hidden_states = scatter_to_sequence_parallel(self.eh_proj(hidden_states), self.ps)
        hidden_states = self.transformer_layer(
            hidden_states, position_ids=attention_position_ids, packed_seq_params=packed_seq_params
        )
        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states, input_ids, position_ids


class MTPBlock(nn.Module):
    def __init__(
        self,
        *,
        num_layers: int,
        repeated_layer: bool,
        layer_factory: Callable[[int], MTPDecoderLayer],
    ):
        super().__init__()
        self.num_layers = num_layers
        self.repeated_layer = repeated_layer
        layers_to_build = 1 if repeated_layer else num_layers
        self.layers = nn.ModuleList([layer_factory(idx) for idx in range(layers_to_build)])

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
