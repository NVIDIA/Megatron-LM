# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Mock data provider for Qwen3.5-VL MIMO training.

Generates synthetic image-text data in the exact format expected by
``model_provider_qwen35_vlm`` (encoder key ``"qwen35_vision"`` under
modality ``"images"``).

Vision geometry (224 px input with Qwen3.5-VL ViT config):
  in_channels=3, temporal_patch_size=2, patch_size=16, spatial_merge_size=2
  → grid_thw = [[1, 14, 14]]          (T=1, H=14, W=14 patch grid)
  → total raw patches  = 196          (1 × 14 × 14)
  → merged patches per image = 49     (1 × 7 × 7 after spatial merge=2)
  → per_patch_dim = 1536              (3 × 2 × 16 × 16)

Modality input format (what MimoModel.forward() expects):
  modality_inputs = {
      "images": {
          "qwen35_vision": {
              "pixel_values": Tensor[total_patches, 1536],
              "grid_thw":     Tensor[num_images, 3],
          }
      }
  }
"""

from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from model_providers.qwen35 import QWEN35_VL_IMAGE_TOKEN_ID, QWEN35_VL_VOCAB_SIZE

# Vision geometry constants (for 224-px input with Qwen3.5-VL ViT config)
_TEMPORAL = 1
_HEIGHT_PATCHES = 14        # 224 // patch_size (16)
_WIDTH_PATCHES = 14         # 224 // patch_size (16)
_MERGE_SIZE = 2
_IN_CHANNELS = 3
_TEMPORAL_PATCH_SIZE = 2
_PATCH_SIZE = 16

# Derived constants
_TOTAL_RAW_PATCHES = _TEMPORAL * _HEIGHT_PATCHES * _WIDTH_PATCHES           # 196
_MERGED_PATCHES = (
    _TEMPORAL
    * (_HEIGHT_PATCHES // _MERGE_SIZE)
    * (_WIDTH_PATCHES // _MERGE_SIZE)
)  # 49
_PER_PATCH_DIM = _IN_CHANNELS * _TEMPORAL_PATCH_SIZE * _PATCH_SIZE * _PATCH_SIZE  # 1536


class MockQwen35VLDataset(Dataset):
    """Synthetic image-text dataset for Qwen3.5-VL MIMO training.

    Each sample contains one image (49 merged image-token placeholders)
    followed by random text tokens to fill the requested sequence length.
    """

    def __init__(
        self,
        size: int = 10000,
        seq_len: int = 4096,
        pad_token_id: int = 0,
        image_token_id: int = QWEN35_VL_IMAGE_TOKEN_ID,
        vocab_size: int = QWEN35_VL_VOCAB_SIZE,
    ):
        self.size = size
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id
        self.image_token_id = image_token_id
        # Safe upper bound for random text token IDs (avoid image/special tokens)
        self._text_vocab_upper = min(vocab_size, image_token_id)

        if seq_len < _MERGED_PATCHES:
            raise ValueError(
                f"seq_len ({seq_len}) must be >= merged_patches_per_image "
                f"({_MERGED_PATCHES}). Increase --total-seq-length."
            )

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict:
        # Build input_ids: image placeholder tokens followed by random text
        image_tokens = torch.full(
            (_MERGED_PATCHES,), self.image_token_id, dtype=torch.long
        )
        num_text = self.seq_len - _MERGED_PATCHES
        text_tokens = torch.randint(
            low=1,
            high=self._text_vocab_upper,
            size=(num_text,),
            dtype=torch.long,
        )
        input_ids = torch.cat([image_tokens, text_tokens], dim=0)  # [seq_len]

        # Labels: shift by one; image placeholder positions are ignored
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = self.pad_token_id
        labels[input_ids == self.image_token_id] = -100

        # Loss mask: 0 for image placeholders and padding, 1 elsewhere
        loss_mask = torch.ones(self.seq_len, dtype=torch.float32)
        loss_mask[input_ids == self.pad_token_id] = 0.0
        loss_mask[input_ids == self.image_token_id] = 0.0

        # pixel_values: zero-filled flat patches [total_raw_patches, per_patch_dim]
        pixel_values = torch.zeros(
            _TOTAL_RAW_PATCHES, _PER_PATCH_DIM, dtype=torch.float32
        )

        # grid_thw: [num_images=1, 3] with (T, H, W) in patch-grid units
        grid_thw = torch.tensor(
            [[_TEMPORAL, _HEIGHT_PATCHES, _WIDTH_PATCHES]], dtype=torch.long
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "modality_inputs": {
                "images": {
                    "qwen35_vision": {
                        "pixel_values": pixel_values,
                        "grid_thw": grid_thw,
                    }
                }
            },
        }


def _collate_fn(batch: List[Dict]) -> Dict:
    """Collate a list of samples into a batched dict for MIMO training."""
    input_ids = torch.stack([s["input_ids"] for s in batch])     # [B, S]
    labels = torch.stack([s["labels"] for s in batch])           # [B, S]
    loss_mask = torch.stack([s["loss_mask"] for s in batch])     # [B, S]

    # pixel_values: [B × 196, 1536]  (all images from the batch, concatenated)
    pixel_values = torch.cat(
        [s["modality_inputs"]["images"]["qwen35_vision"]["pixel_values"] for s in batch],
        dim=0,
    )
    # grid_thw: [B, 3]  (one row per image, one image per sample)
    grid_thw = torch.cat(
        [s["modality_inputs"]["images"]["qwen35_vision"]["grid_thw"] for s in batch],
        dim=0,
    )

    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_mask": loss_mask,
        "modality_inputs": {
            "images": {
                "qwen35_vision": {
                    "pixel_values": pixel_values,
                    "grid_thw": grid_thw,
                }
            }
        },
    }


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Megatron-compatible dataset provider for qwen35_vlm mock data.

    Registered as ``"qwen35_vlm"`` in ``_DATASET_PROVIDERS`` in
    ``examples/mimo/train.py``.
    """
    from megatron.core import mpu
    from megatron.training import get_args

    args = get_args()
    # Prefer total_seq_length (MIMO-specific arg) over seq_length (GPT arg)
    seq_len = getattr(args, "total_seq_length", None) or getattr(args, "seq_length", 4096)
    pad_token_id = getattr(args, "pad_token_id", 0)
    image_token_id = getattr(args, "image_token_id", QWEN35_VL_IMAGE_TOKEN_ID)

    if mpu.get_tensor_model_parallel_rank() == 0:
        train_dataset = MockQwen35VLDataset(
            size=train_val_test_num_samples[0],
            seq_len=seq_len,
            pad_token_id=pad_token_id,
            image_token_id=image_token_id,
        )
        valid_dataset = MockQwen35VLDataset(
            size=max(train_val_test_num_samples[1], 100),
            seq_len=seq_len,
            pad_token_id=pad_token_id,
            image_token_id=image_token_id,
        )
        test_dataset = None
    else:
        train_dataset = None
        valid_dataset = None
        test_dataset = None

    return train_dataset, valid_dataset, test_dataset
