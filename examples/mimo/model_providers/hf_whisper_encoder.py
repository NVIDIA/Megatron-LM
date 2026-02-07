# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import torch
from transformers import WhisperConfig, WhisperModel

class HFWhisperEncoderWrapper(torch.nn.Module):
    """Whisper audio encoder wrapper that extracts last_hidden_state."""

    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = WhisperModel.from_pretrained(model_name).encoder

    def forward(self, input_features, seq_lengths=None):
        '''
        input_features: torch.Tensor
            input audio features
        seq_lengths: torch.Tensor
            the number of audio tokens corresponding to non-padded audio frames
            we only get the embeddings for the non-padded audio frames
        '''
        with torch.no_grad():
            hidden = self.encoder(input_features).last_hidden_state  # [b, s, h]
            if seq_lengths is not None:
                seq_len = hidden.shape[1]
                mask = torch.arange(seq_len, device=hidden.device)[None, :] < seq_lengths[:, None]
                hidden = hidden[mask]
            return hidden