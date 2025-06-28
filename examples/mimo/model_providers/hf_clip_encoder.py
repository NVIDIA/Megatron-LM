# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import torch
from transformers import CLIPVisionModel


class HFCLIPEncoderWrapper(torch.nn.Module):
    """CLIP encoder wrapper that extracts last_hidden_state."""

    def __init__(self):
        super().__init__()
        self.encoder = CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14-336')
        self.encoder.eval()
        self.feature_layer_index = -2

    def forward(self, **encoder_inputs):
        '''
        input_features: dict
            pixel_values: torch.Tensor
                shape: (B, 3, 336, 336)
        '''
        # Process through encoder and extract last_hidden_state
        with torch.no_grad():
            last_hidden_state = self.encoder(**encoder_inputs, output_hidden_states=True)

            # -1 index is image features
            image_features = last_hidden_state[-1]
            # select last but second layer
            image_features = image_features[self.feature_layer_index]
            # drop cls token
            image_features = image_features[:, 1:, :]
            return image_features
