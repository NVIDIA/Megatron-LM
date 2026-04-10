# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Mock dataset for multimodal_dev end-to-end testing.

Generates synthetic image + text data.  Each sample has random text
tokens with image-token placeholders, random pixel values sized for the
vision encoder, 3D MRoPE position IDs, and shifted labels.
"""

import torch
from torch.utils.data import Dataset

from examples.multimodal_dev.models.qwen35_vl.configuration import (
    QWEN35_VL_IMAGE_TOKEN_ID,
    QWEN35_VL_VIDEO_TOKEN_ID,
    QWEN35_VL_VISION_START_TOKEN_ID,
)
from examples.multimodal_dev.models.qwen35_vl.mrope import get_rope_index


class MockQwen35VLDataset(Dataset):
    """Synthetic Qwen3.5-VL training samples.

    Args:
        num_samples: Number of samples.
        seq_length: Total sequence length (text + image tokens).
        image_seq_length: Number of image tokens per sample.
        vocab_size: Vocabulary size for random text tokens.
        image_token_id: Token ID for image placeholders.
        video_token_id: Token ID for video placeholders.
        vision_start_token_id: Token ID marking start of a vision region.
        image_size: Image height and width in pixels.
        patch_size: Spatial patch size.
        temporal_patch_size: Temporal patch size.
        spatial_merge_size: Spatial merge factor.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        seq_length: int = 1024,
        image_seq_length: int = 256,
        vocab_size: int = 248320,
        image_token_id: int = QWEN35_VL_IMAGE_TOKEN_ID,
        video_token_id: int = QWEN35_VL_VIDEO_TOKEN_ID,
        vision_start_token_id: int = QWEN35_VL_VISION_START_TOKEN_ID,
        image_size: int = 224,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        spatial_merge_size: int = 2,
    ):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.image_size = image_size
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.spatial_merge_size = spatial_merge_size

        h_patches = image_size // patch_size
        w_patches = image_size // patch_size
        t_patches = temporal_patch_size
        self.grid_thw = torch.tensor([[t_patches, h_patches, w_patches]])

        self.num_merged_tokens = (
            t_patches
            * (h_patches // spatial_merge_size)
            * (w_patches // spatial_merge_size)
        )
        self.image_seq_length = min(
            image_seq_length, self.num_merged_tokens,
        )
        self.total_patches = t_patches * h_patches * w_patches

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Reserve 1 slot for the vision_start sentinel before image tokens.
        text_length = self.seq_length - self.image_seq_length - 1
        text_tokens = torch.randint(
            1, self.vocab_size, (text_length,), dtype=torch.long,
        )
        special_ids = {
            self.image_token_id,
            self.video_token_id,
            self.vision_start_token_id,
        }
        for sid in special_ids:
            text_tokens[text_tokens == sid] = 1

        prefix_len = text_length // 2
        suffix_len = text_length - prefix_len
        input_ids = torch.cat([
            text_tokens[:prefix_len],
            torch.tensor(
                [self.vision_start_token_id], dtype=torch.long,
            ),
            torch.full(
                (self.image_seq_length,),
                self.image_token_id,
                dtype=torch.long,
            ),
            text_tokens[prefix_len: prefix_len + suffix_len],
        ])

        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = 0

        loss_mask = (input_ids != self.image_token_id).float()
        loss_mask[-1] = 0

        pixel_dim = (
            3
            * self.temporal_patch_size
            * self.patch_size
            * self.patch_size
        )
        pixel_values = torch.randn(self.total_patches, pixel_dim)

        image_grid_thw = self.grid_thw.clone()

        position_ids, _ = get_rope_index(
            spatial_merge_size=self.spatial_merge_size,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            input_ids=input_ids.unsqueeze(0),
            image_grid_thw=image_grid_thw,
        )
        position_ids = position_ids.squeeze(1)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }


def mock_collate_fn(batch):
    """Collate: handles position_ids ``[3, S]`` stacking."""
    result = {}
    keys = batch[0].keys()
    for key in keys:
        tensors = [sample[key] for sample in batch]
        if key == "position_ids":
            result[key] = torch.stack(tensors, dim=1)
        elif key == "image_grid_thw":
            result[key] = torch.cat(tensors, dim=0)
        elif key == "pixel_values":
            result[key] = torch.cat(tensors, dim=0)
        else:
            result[key] = torch.stack(tensors, dim=0)
    return result


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Provide mock train / val / test datasets."""
    from megatron.training import get_args

    args = get_args()
    kwargs = dict(
        seq_length=getattr(args, "total_seq_length", 1024),
        image_seq_length=getattr(args, "image_seq_length", 256),
        vocab_size=getattr(args, "padded_vocab_size", 248320),
        image_token_id=getattr(args, "image_token_id", 248056),
        image_size=getattr(args, "image_size", 224),
    )

    train_ds = MockQwen35VLDataset(
        num_samples=train_val_test_num_samples[0], **kwargs,
    )
    val_ds = MockQwen35VLDataset(
        num_samples=train_val_test_num_samples[1], **kwargs,
    )
    test_ds = MockQwen35VLDataset(
        num_samples=train_val_test_num_samples[2], **kwargs,
    )

    return train_ds, val_ds, test_ds
