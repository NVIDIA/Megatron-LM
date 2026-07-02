# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig

from .packed_audio import PackedAudioEmbeddings


class AudioProjection(MegatronModule):
    """Stack audio embeddings in time and project them into LM hidden size."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        projector_type: str,
        input_size: int,
        stack_factor: int = 1,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        super().__init__(config=config)
        if stack_factor < 1:
            raise ValueError(f"stack_factor must be >= 1, got {stack_factor}")

        self.input_size = input_size
        self.stack_factor = stack_factor
        self.output_size = config.hidden_size
        self.projector = MultimodalProjector(
            config=config,
            submodules=submodules,
            projector_type=projector_type,
            input_size=input_size * stack_factor,
            tp_group=tp_group,
            pg_collection=pg_collection,
        )

    def _stack_features(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        if hidden_size != self.input_size:
            raise ValueError(f"Expected hidden size {self.input_size}, got {hidden_size}")

        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.bool, device=hidden_states.device)
            hidden_states = hidden_states * attention_mask.unsqueeze(-1).to(hidden_states.dtype)

        pad = (-seq_len) % self.stack_factor
        if pad:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad))
            if attention_mask is not None:
                attention_mask = F.pad(attention_mask, (0, pad), value=False)

        stacked_seq_len = hidden_states.shape[1] // self.stack_factor
        hidden_states = hidden_states.reshape(
            batch_size, stacked_seq_len, self.stack_factor * hidden_size
        )

        output_mask = None
        if attention_mask is not None:
            output_mask = attention_mask.reshape(
                batch_size, stacked_seq_len, self.stack_factor
            ).any(dim=-1)

        return hidden_states, output_mask

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        stacked_states, output_mask = self._stack_features(hidden_states, attention_mask)
        stacked_states = stacked_states.permute(1, 0, 2).contiguous()
        projected_states = self.projector(stacked_states)
        return projected_states, output_mask

    def forward_packed(self, packed_states: PackedAudioEmbeddings) -> PackedAudioEmbeddings:
        if self.stack_factor != 1:
            raise NotImplementedError(
                "Packed audio projection currently supports stack_factor == 1 only"
            )
        hidden_states = packed_states.embeddings
        if hidden_states.ndim != 2:
            raise ValueError(
                f"Expected packed audio embeddings [Ttotal, H], got {tuple(hidden_states.shape)}"
            )
        if hidden_states.shape[-1] != self.input_size:
            raise ValueError(
                f"Expected hidden size {self.input_size}, got {hidden_states.shape[-1]}"
            )

        projected_states = self.projector(hidden_states.unsqueeze(1)).squeeze(1)
        return PackedAudioEmbeddings(
            embeddings=projected_states,
            lengths=packed_states.lengths.to(dtype=torch.int32, device=projected_states.device),
        )

    def estimate_flops(
        self,
        output_seq_lengths: torch.Tensor,
        include_backward: Optional[bool] = None,
        input_requires_grad: Optional[bool] = None,
        count_padded: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Estimate FLOPs for the audio projection over projected audio lengths.

        ``output_seq_lengths`` is in the projected/stacked audio-token space
        (the same space as ``audio_embeds_seq_lengths``). By default the
        estimate counts the padded tensor shape used by ``forward``.
        """
        if not torch.is_tensor(output_seq_lengths):
            output_seq_lengths = torch.tensor(output_seq_lengths, dtype=torch.long)
        device = output_seq_lengths.device
        zero = torch.zeros((), dtype=torch.float64, device=device)
        if output_seq_lengths.numel() == 0:
            return {"projection_forward": zero, "projection_train": zero}

        output_seq_lengths = output_seq_lengths.to(dtype=torch.long).clamp(min=0)
        if count_padded:
            token_count = output_seq_lengths.max().to(dtype=torch.float64) * float(
                output_seq_lengths.numel()
            )
        else:
            token_count = output_seq_lengths.to(dtype=torch.float64).sum()

        input_size = int(self.input_size) * int(self.stack_factor)
        output_size = int(self.output_size)
        projector_type = self.projector.projector_type

        if projector_type == "affine":
            projection_forward = 2.0 * token_count * input_size * output_size
        elif projector_type == "mlp":
            if self.config.ffn_hidden_size is None:
                raise ValueError("Audio projection MLP requires config.ffn_hidden_size")
            ffn_hidden_size = int(self.config.ffn_hidden_size)
            fc1_output_size = ffn_hidden_size * (
                2 if getattr(self.config, "gated_linear_unit", False) else 1
            )
            projection_forward = (
                2.0 * token_count * input_size * fc1_output_size
                + 2.0 * token_count * ffn_hidden_size * output_size
            )
        else:
            raise ValueError(f"Unsupported audio projection type {projector_type!r}")

        params_require_grad = any(p.requires_grad for p in self.parameters())
        if input_requires_grad is None:
            input_requires_grad = params_require_grad
        if include_backward is None:
            include_backward = self.training and (params_require_grad or input_requires_grad)

        backward_factor = 1.0
        if include_backward:
            if params_require_grad:
                backward_factor += 1.0
            if input_requires_grad:
                backward_factor += 1.0

        return {
            "projection_forward": projection_forward,
            "projection_train": projection_forward * backward_factor,
        }
