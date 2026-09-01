# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Synthetic DeepSeek-V4-Flash-Vision samples for launch and smoke tests."""

import math

import torch
from torch.utils.data import Dataset

from examples.multimodal_dev.models.deepseek_v4.configuration import (
    DEEPSEEK_V4_VOCAB_SIZE,
    VISION_DOWNSAMPLE_RATIO,
    VISION_PATCH_SIZE,
    build_image_block,
)


class MockDeepSeekV4VisionDataset(Dataset):
    """Generate one-image samples using the official synthetic-token N-layout."""

    def __init__(
        self,
        num_samples: int = 1000,
        seq_length: int = 1024,
        image_size: int = 224,
        vocab_size: int = DEEPSEEK_V4_VOCAB_SIZE,
    ) -> None:
        if image_size % VISION_PATCH_SIZE:
            raise ValueError(
                f"image_size={image_size} must be divisible by patch size {VISION_PATCH_SIZE}."
            )
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.image_size = image_size
        self.vocab_size = vocab_size
        self.n_vit_h = image_size // VISION_PATCH_SIZE
        self.n_vit_w = image_size // VISION_PATCH_SIZE
        self.n_llm_h = math.ceil(self.n_vit_h / VISION_DOWNSAMPLE_RATIO)
        self.n_llm_w = math.ceil(self.n_vit_w / VISION_DOWNSAMPLE_RATIO)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        del index
        # The compression-prefix length depends on the image block start modulo four.
        tentative_start = max(1, self.seq_length // 3)
        image_types, _ = build_image_block(self.n_llm_h, self.n_llm_w, tentative_start)
        if image_types.numel() >= self.seq_length:
            raise ValueError(
                f"seq_length={self.seq_length} is too short for a {image_types.numel()}-token "
                "image block."
            )
        prefix_length = min(tentative_start, self.seq_length - image_types.numel() - 1)
        # Rebuild because changing start position may change compression padding.
        image_types, _ = build_image_block(self.n_llm_h, self.n_llm_w, prefix_length)
        suffix_length = self.seq_length - prefix_length - image_types.numel()
        if suffix_length < 0:
            raise ValueError("seq_length became too short after image-block alignment padding.")

        text = torch.randint(1, self.vocab_size, (prefix_length + suffix_length,), dtype=torch.long)
        input_ids = torch.cat(
            (text[:prefix_length], image_types + self.vocab_size, text[prefix_length:])
        )
        labels = torch.empty_like(input_ids)
        labels[:-1] = input_ids[1:]
        labels[-1] = 0
        loss_mask = (labels < self.vocab_size).float()
        loss_mask[-1] = 0
        # Synthetic image IDs live immediately above the decoder vocabulary and
        # therefore cannot be gathered from the language-model output logits.
        labels[loss_mask == 0] = -100

        patch_dim = 3 * VISION_PATCH_SIZE**2
        pixel_values = torch.randn(self.n_vit_h * self.n_vit_w, patch_dim, dtype=torch.float32)
        image_grid_thw = torch.tensor([[1, self.n_vit_h, self.n_vit_w]], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Return synthetic train, validation, and test datasets."""
    from megatron.training import get_args

    args = get_args()
    kwargs = {
        "seq_length": getattr(args, "total_seq_length", args.seq_length),
        "image_size": getattr(args, "image_size", 224),
        "vocab_size": DEEPSEEK_V4_VOCAB_SIZE,
    }
    return tuple(
        MockDeepSeekV4VisionDataset(num_samples=num_samples, **kwargs)
        for num_samples in train_val_test_num_samples
    )
