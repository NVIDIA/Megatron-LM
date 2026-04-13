# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Mock data provider for Kimi K2.5 VL training.

Generates synthetic image-text data matching the KimiK25VLModel forward
signature: input_ids, labels, loss_mask, pixel_values, image_grid_thw.

MoonViT3d vision geometry (224 px input):
  Conv2d patch_size=14 → grid = 16×16 patches per image
  PatchMergerMLP merge → 64 merged patches per image (placeholder tokens)
  pixel_values: (total_patches, C*pH*pW) = (256, 588) flattened for TP broadcast
"""

from typing import Dict

import torch
from torch.utils.data import Dataset

from examples.multimodal_dev.models.kimi_k25.configuration import (
    KIMI_K25_IMAGE_TOKEN_ID,
    KIMI_K25_VOCAB_SIZE,
    VISION_HEIGHT_PATCHES,
    VISION_MERGED_PATCHES,
    VISION_PER_PATCH_DIM,
    VISION_TEMPORAL,
    VISION_TOTAL_RAW_PATCHES,
    VISION_WIDTH_PATCHES,
)


class MockKimiK25VLDataset(Dataset):
    """Synthetic image-text dataset for Kimi K2.5 VL training."""

    def __init__(
        self,
        size: int = 10000,
        seq_len: int = 4096,
        pad_token_id: int = 0,
        image_token_id: int = KIMI_K25_IMAGE_TOKEN_ID,
        vocab_size: int = KIMI_K25_VOCAB_SIZE,
    ):
        self.size = size
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id
        self.image_token_id = image_token_id
        self._text_vocab_upper = min(vocab_size, image_token_id)

        if seq_len < VISION_MERGED_PATCHES:
            raise ValueError(
                f"seq_len ({seq_len}) must be >= merged_patches ({VISION_MERGED_PATCHES})"
            )

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict:
        # Pre-expanded: N placeholder tokens per image
        image_tokens = torch.full(
            (VISION_MERGED_PATCHES,), self.image_token_id, dtype=torch.long,
        )
        num_text = self.seq_len - VISION_MERGED_PATCHES
        text_tokens = torch.randint(
            1, self._text_vocab_upper, (num_text,), dtype=torch.long,
        )
        input_ids = torch.cat([image_tokens, text_tokens])

        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = self.pad_token_id
        labels[input_ids == self.image_token_id] = -100

        loss_mask = torch.ones(self.seq_len, dtype=torch.float32)
        loss_mask[input_ids == self.pad_token_id] = 0.0
        loss_mask[input_ids == self.image_token_id] = 0.0

        # Flat pixel values for TP broadcast (reshape to 4D in model)
        pixel_values = torch.zeros(
            VISION_TOTAL_RAW_PATCHES, VISION_PER_PATCH_DIM, dtype=torch.float32,
        )
        grid_thw = torch.tensor(
            [[VISION_TEMPORAL, VISION_HEIGHT_PATCHES, VISION_WIDTH_PATCHES]],
            dtype=torch.long,
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": grid_thw,
        }


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Provide mock train / val / test datasets."""
    from megatron.training import get_args

    args = get_args()
    seq_len = getattr(args, "total_seq_length", None) or getattr(args, "seq_length", 4096)
    pad_token_id = getattr(args, "pad_token_id", 0)
    image_token_id = getattr(args, "image_token_id", KIMI_K25_IMAGE_TOKEN_ID)

    kwargs = dict(
        seq_len=seq_len,
        pad_token_id=pad_token_id,
        image_token_id=image_token_id,
    )

    train_ds = MockKimiK25VLDataset(
        size=train_val_test_num_samples[0], **kwargs,
    )
    val_ds = MockKimiK25VLDataset(
        size=max(train_val_test_num_samples[1], 100), **kwargs,
    )
    test_ds = MockKimiK25VLDataset(
        size=max(train_val_test_num_samples[2], 100), **kwargs,
    )

    return train_ds, val_ds, test_ds
