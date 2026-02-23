# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""HuggingFace Qwen3.5 vision encoder wrapper for MIMO.

Wraps the Qwen3.5-MoE vision model (ViT + PatchMerger) from HuggingFace
Transformers so it can be plugged into the MIMO encoder slot.

The vision model includes:
  - 3D PatchEmbed (patch_size=16, temporal_patch_size=2)
  - 27-layer ViT with rotary position embeddings
  - PatchMerger (spatial_merge_size=2) that projects to out_hidden_size

Since the merger already outputs at out_hidden_size (4096 for 397B, matching
the decoder hidden_size), no additional projection is needed.
"""

import torch
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeVisionModel


class HFQwen35VisionEncoderWrapper(torch.nn.Module):
    """Qwen3.5 vision encoder that wraps the HF Qwen3_5MoeVisionModel.

    The encoder outputs at out_hidden_size (default 4096), which is designed
    to match the language decoder hidden_size directly.

    Args:
        pretrained_model_name: HuggingFace model name to load config from.
        load_pretrained_weights: If True, attempt to load pretrained vision weights.
    """

    def __init__(
        self,
        pretrained_model_name: str = "Qwen/Qwen3.5-397B-A17B",
        load_pretrained_weights: bool = False,
    ):
        super().__init__()
        full_config = AutoConfig.from_pretrained(pretrained_model_name, trust_remote_code=True)
        vision_config = full_config.vision_config

        self.encoder = Qwen3_5MoeVisionModel(vision_config)
        self.spatial_merge_size = vision_config.spatial_merge_size

        if load_pretrained_weights:
            self._load_pretrained_vision_weights(pretrained_model_name)

        self.encoder.eval()

    def _load_pretrained_vision_weights(self, model_name: str):
        """Load only the vision model weights from the full checkpoint."""
        from transformers import Qwen3_5MoeForConditionalGeneration

        full_model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
        vision_state_dict = {
            k.removeprefix("visual."): v
            for k, v in full_model.state_dict().items()
            if k.startswith("visual.")
        }
        self.encoder.load_state_dict(vision_state_dict, strict=False)
        del full_model

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """Encode pixel values through the Qwen3.5 ViT + PatchMerger.

        Args:
            pixel_values: Flattened patches for 3D Conv PatchEmbed.
                Shape (total_patches, patch_dim) or (batch, num_patches, patch_dim)
                when coming from a DataLoader with default collation.
            grid_thw: Temporal/height/width grid for each image or video.
                Shape (num_images, 3) or (batch, num_images_per_sample, 3).

        Returns:
            Merged embeddings at out_hidden_size, shape (total_merged_tokens, out_hidden_size).
        """
        if pixel_values.ndim == 3:
            pixel_values = pixel_values.reshape(-1, pixel_values.shape[-1])
        if grid_thw.ndim == 3:
            grid_thw = grid_thw.reshape(-1, grid_thw.shape[-1])

        target_dtype = self.encoder.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.to(dtype=target_dtype)

        with torch.no_grad():
            output = self.encoder(pixel_values, grid_thw=grid_thw, return_dict=True)
        return output.pooler_output
