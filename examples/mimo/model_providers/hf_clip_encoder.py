# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import torch
from transformers import CLIPVisionModel, LlavaNextVideoConfig
from transformers.models.llava_next_video.modeling_llava_next_video import (
    LlavaNextVideoPooler,
)


class HFCLIPEncoderWrapper(torch.nn.Module):
    """CLIP encoder wrapper that extracts last_hidden_state."""

    def __init__(self, feature_layer_index=-2, is_video_input: bool = False):
        """Initialize the HFCLIPEncoderWrapper.

        Args:
            feature_layer_index (int): Index of the feature layer to extract from the encoder's hidden states.
                                       Default is -2 (second to last layer).
            is_video_input (bool): If True, expects video input and applies vision resampler.
        """
        super().__init__()
        self.encoder = CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14-336')
        self.encoder.eval()
        self.feature_layer_index = feature_layer_index
        self.is_video_input = is_video_input
        if self.is_video_input:
            config = LlavaNextVideoConfig()
            self.vision_resampler = LlavaNextVideoPooler(config)

    def forward(self, pixel_values: torch.Tensor):
        """Input: (B, F, 3, 336, 336) if video, else (B, 3, 336, 336) or (num_frames, 3, 336, 336)."""
        # Process through encoder and extract last_hidden_state
        with torch.no_grad():
            if self.is_video_input:
                batch_size, frames, channels, height, width = pixel_values.shape
                pixel_values = pixel_values.reshape(batch_size * frames, channels, height, width)
            

            last_hidden_state = self.encoder(pixel_values, output_hidden_states=True)
            # -1 index is image features
            image_features = last_hidden_state[-1]
            # select last but second layer
            image_features = image_features[self.feature_layer_index]
            # drop cls token
            image_features = image_features[:, 1:, :]
            if self.is_video_input:
                image_features = self.vision_resampler(image_features)
            return image_features