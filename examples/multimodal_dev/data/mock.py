# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Mock dataset for multimodal_dev end-to-end testing.

Generates synthetic image + text data.  Each sample has random text
tokens with image-token placeholders, random pixel values sized for the
vision encoder, 3D MRoPE position IDs, and shifted labels.
"""

import hashlib

import torch
from torch.utils.data import Dataset

from examples.multimodal_dev.models.qwen35_vl.configuration import (
    QWEN35_VL_IMAGE_TOKEN_ID,
    QWEN35_VL_VIDEO_TOKEN_ID,
    QWEN35_VL_VISION_START_TOKEN_ID,
)
from examples.multimodal_dev.models.qwen35_vl.mrope import get_rope_index


def _sample_seed(seed: int, split: str, idx: int) -> int:
    """Derive a per-sample seed from its coordinates.

    Hashing rather than adding: ``seed + idx`` overlaps by construction, since ``idx=1`` under
    seed ``S`` is ``idx=0`` under seed ``S+1``, and each split would replay the previous one
    shifted by one index. blake2b is stable across processes and Python versions, unlike the
    built-in ``hash``, which is randomized per interpreter. The 8-byte digest matches the
    range ``torch.Generator.manual_seed`` accepts.
    """
    key = f"{seed}:{split}:{idx}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")


class MockQwen35VLDataset(Dataset):
    """Synthetic Qwen3.5-VL training samples.

    Args:
        num_samples: Number of samples.
        seq_length: Total sequence length (text + image tokens).
        image_seq_length: Image-token budget. It must be at least the merged-token
            count implied by the image geometry; the emitted count is derived from
            that geometry.
        vocab_size: Vocabulary size for random text tokens.
        image_token_id: Token ID for image placeholders.
        video_token_id: Token ID for video placeholders.
        vision_start_token_id: Token ID marking start of a vision region.
        image_size: Image height and width in pixels.
        patch_size: Spatial patch size.
        temporal_patch_size: Temporal patch size.
        spatial_merge_size: Spatial merge factor.
        seed: Base seed for sample generation.
        split: Identifies this dataset among datasets sharing ``seed`` (e.g. "train",
            "valid", "test"), so the splits draw separate content instead of the same
            samples. Each sample is derived from ``(seed, split, idx)``, so a given
            index always yields the same sample.
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
        seed: int = 1234,
        split: str = "train",
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
        self.seed = seed
        self.split = split

        h_patches = image_size // patch_size
        w_patches = image_size // patch_size
        t_patches = temporal_patch_size
        self.grid_thw = torch.tensor([[t_patches, h_patches, w_patches]])

        self.num_merged_tokens = (
            t_patches
            * (h_patches // spatial_merge_size)
            * (w_patches // spatial_merge_size)
        )
        # Reject rather than clamp: a smaller budget yields an image span that disagrees with
        # grid_thw, and that only surfaces later as a shape mismatch in the masked position-ID
        # assignment inside get_rope_index.
        if image_seq_length < self.num_merged_tokens:
            raise ValueError(
                f"image_seq_length={image_seq_length} is smaller than the "
                f"{self.num_merged_tokens} merged vision tokens implied by "
                f"image_size={image_size}, patch_size={patch_size}, "
                f"temporal_patch_size={temporal_patch_size}, "
                f"spatial_merge_size={spatial_merge_size}. Shrink the image instead of the "
                "token budget, or raise image_seq_length to at least the merged-token count."
            )
        self.image_seq_length = self.num_merged_tokens
        self.total_patches = t_patches * h_patches * w_patches

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Bounds matter now that ``idx`` determines content: an out-of-range index would
        # otherwise hash to a plausible sample from outside the dataset instead of failing.
        idx = int(idx)
        if not 0 <= idx < self.num_samples:
            raise IndexError(f"index {idx} out of range for {self.num_samples} samples")

        # Derive the sample from its index. Ranks share the CPU seed unless
        # data_parallel_random_init is set, so ambient-RNG draws collapse a global batch's
        # distinct samples to the per-rank microbatch count — a number that changes with the
        # parallel layout, so two configurations being compared see different data.
        generator = torch.Generator().manual_seed(
            _sample_seed(self.seed, self.split, idx)
        )

        # Reserve 1 slot for the vision_start sentinel before image tokens.
        text_length = self.seq_length - self.image_seq_length - 1
        text_tokens = torch.randint(
            1, self.vocab_size, (text_length,), dtype=torch.long, generator=generator,
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
        pixel_values = torch.randn(
            self.total_patches, pixel_dim, generator=generator,
        )

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
            "cu_seqlens": torch.tensor([0, self.seq_length], dtype=torch.int32),
            "cu_seqlens_padded": torch.tensor(
                [0, self.seq_length], dtype=torch.int32,
            ),
            "max_seqlen": torch.tensor(self.seq_length, dtype=torch.int32),
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
        seed=getattr(args, "seed", 1234),
        seq_length=getattr(args, "total_seq_length", 1024),
        image_seq_length=getattr(args, "image_seq_length", 256),
        vocab_size=getattr(args, "padded_vocab_size", 248320),
        image_token_id=getattr(args, "image_token_id", 248056),
        image_size=getattr(args, "image_size", 224),
    )

    train_ds = MockQwen35VLDataset(
        num_samples=train_val_test_num_samples[0], split="train", **kwargs,
    )
    val_ds = MockQwen35VLDataset(
        num_samples=train_val_test_num_samples[1], split="valid", **kwargs,
    )
    test_ds = MockQwen35VLDataset(
        num_samples=train_val_test_num_samples[2], split="test", **kwargs,
    )

    return train_ds, val_ds, test_ds
