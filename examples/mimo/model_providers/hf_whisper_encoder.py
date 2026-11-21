# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn.functional as F


def scale_whisper_tp_grad(grad, group):
    """Average a duplicated HF Whisper gradient across the tensor-parallel group.

    The weights are not tagged for tensor parallelism, so DDP will not divide
    this gradient. The return value is the SUM reduced by the group size.
    """
    torch.distributed.all_reduce(grad, op=torch.distributed.ReduceOp.SUM, group=group)
    return grad / torch.distributed.get_world_size(group)


class HFWhisperEncoderWrapper(torch.nn.Module):
    """Whisper audio encoder wrapper.

    A full 30 s mel still goes through ``WhisperEncoder.forward``. A row with
    ``feature_lengths`` is cropped before the encoder layers, so the padded
    tail is not in the convolution or the attention.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        finetune: bool = False,
        encoder=None,
    ):
        super().__init__()
        if encoder is None:
            from transformers import WhisperModel

            self.encoder = WhisperModel.from_pretrained(model_name).encoder
        else:
            self.encoder = encoder
        self.finetune = finetune
        if finetune:
            for name in (
                "dropout",
                "attention_dropout",
                "activation_dropout",
                "encoder_layerdrop",
            ):
                if float(getattr(self.encoder.config, name, 0) or 0) != 0:
                    raise ValueError(f"{name} must be 0 to finetune the Whisper encoder")
        else:
            self.encoder.eval()
            self.encoder.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.finetune:
            self.encoder.eval()
        return self

    def forward(self, input_features, seq_lengths=None, feature_lengths=None):
        grad_ctx = nullcontext() if self.finetune else torch.no_grad()
        with grad_ctx:
            if seq_lengths is None and feature_lengths is None:
                return self.encoder(input_features).last_hidden_state
            if feature_lengths is None:
                raise ValueError(
                    "seq_lengths requires feature_lengths; 2 * seq_lengths is not invertible"
                )
            if feature_lengths.ndim == 0:
                feature_lengths = feature_lengths.unsqueeze(0)
            if seq_lengths is not None and seq_lengths.ndim == 0:
                seq_lengths = seq_lengths.unsqueeze(0)
            rows = []
            for i in range(input_features.shape[0]):
                mel = int(feature_lengths[i])
                time = int(input_features.shape[-1])
                if not 1 <= mel <= time:
                    raise ValueError(f"feature length {mel} is outside 1..{time}")
                hidden = self._encode_cropped(input_features[i, :, :mel])
                if seq_lengths is not None and hidden.shape[0] != int(seq_lengths[i]):
                    raise ValueError(
                        f"seq_lengths[{i}]={int(seq_lengths[i])} does not match "
                        f"conv length {hidden.shape[0]}"
                    )
                rows.append(hidden)
            return torch.cat(rows, dim=0)

    def _encode_cropped(self, features: torch.Tensor) -> torch.Tensor:
        hidden = features.unsqueeze(0)
        hidden = F.gelu(self.encoder.conv1(hidden))
        hidden = F.gelu(self.encoder.conv2(hidden))
        hidden = hidden.permute(0, 2, 1)
        positions = self.encoder.embed_positions(
            torch.arange(hidden.shape[1], device=hidden.device)
        )
        hidden = hidden + positions.to(dtype=hidden.dtype)
        dropout_p = float(getattr(self.encoder.dropout, "p", 0.0))
        hidden = F.dropout(hidden, p=dropout_p, training=self.encoder.training)
        for layer in self.encoder.layers:
            hidden = layer(hidden, None)
            if isinstance(hidden, tuple):
                hidden = hidden[0]
        hidden = self.encoder.layer_norm(hidden)
        return hidden.squeeze(0)
