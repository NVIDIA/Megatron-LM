# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""
Simple mock data module for testing MIMO with image-text (VLM) models.

This module provides basic synthetic data generation for testing Vision Language Models
within the MIMO framework.
"""

from typing import Callable, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset


def create_mock_image(image_size: int = 336) -> torch.Tensor:
    """
    Create a simple mock image (all zeros).

    Args:
        image_size: Size of the square image

    Returns:
        Tensor of shape [3, H, W] with all zeros
    """
    return torch.zeros(3, image_size, image_size)


def create_mock_caption() -> str:
    """
    Create a simple mock caption.

    Returns:
        A simple caption string
    """
    return "This is an image."


class MockVLMDataset(Dataset):
    """Simple dataset of mock image-text pairs for VLM testing."""

    def __init__(
        self,
        size: int = 10000,
        image_size: int = 336,
        seq_len: int = 512,
        image_seq_length: int = 32,
        vocab_size: int = 256,
        tokenizer: Optional[Callable] = None,
        pad_token_id: int = 0,
        image_token_id: int = 32000,
    ):
        """
        Initialize the mock VLM dataset.

        Args:
            size: Number of examples in the dataset
            image_size: Size of the square images
            seq_len: Total length of the token sequence (image + text)
            image_seq_length: Number of image tokens to pad
            vocab_size: Size of the vocabulary for tokenization
            tokenizer: Optional tokenizer function
            pad_token_id: ID for padding token
            image_token_id: ID for image placeholder token
        """
        self.size = size
        self.image_size = image_size
        self.seq_len = seq_len
        self.image_seq_length = image_seq_length
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer

        # Special token IDs
        self.pad_token_id = pad_token_id
        self.image_token_id = image_token_id

        if self.seq_len < self.image_seq_length:
            raise ValueError(
                f"seq_len ({self.seq_len}) must be >= image_seq_length ({self.image_seq_length})."
            )

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return self.size

    def __getitem__(self, idx: int) -> Dict:
        """
        Get an item from the dataset.

        Args:
            idx: Index of the item (ignored, all items are identical)

        Returns:
            Dictionary containing:
            - images: Tensor of shape [C, H, W]
            - input_ids: Tokenized caption with image token
            - labels: Shifted input_ids for language modeling
            - loss_mask: Mask for loss calculation
            - position_ids: Position IDs for the tokens
        """
        # Create a zero image
        image = create_mock_image(self.image_size)

        # Generate random token sequence for this sample.
        input_ids = self._mock_tokenize()

        # Create labels (shifted input_ids)
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = self.pad_token_id  # Padding for the last position

        # Set labels for image tokens to -100 (ignored in loss calculation)
        labels[input_ids == self.image_token_id] = -100

        # Create loss mask (1 for tokens to calculate loss on, 0 for others)
        loss_mask = torch.ones_like(input_ids).float()
        loss_mask[input_ids == self.pad_token_id] = 0.0  # Don't calculate loss on padding
        loss_mask[input_ids == self.image_token_id] = 0.0  # Don't calculate loss on image tokens

        # Create position IDs (just sequential integers)
        position_ids = torch.arange(len(input_ids), dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "modality_inputs": {
                "clip_encoder": {
                    "images": image,
                }
            },
        }

    def _mock_tokenize(self) -> torch.Tensor:
        """
        Generate a mock token sequence consisting of ``image_seq_length`` image tokens followed by
        randomly generated text tokens such that the total sequence length equals
        ``self.seq_len``.

        Returns:
            torch.Tensor: Tensor of token IDs of shape ``[seq_len]``.
        """

        # Image placeholder tokens ─ placed at the beginning of the sequence to mimic
        # the layout produced by many VLM tokenizers.
        image_tokens = torch.full(
            (self.image_seq_length,), self.image_token_id, dtype=torch.long
        )

        # Random text tokens drawn uniformly in ``[1, vocab_size)`` (we reserve ``0`` for pad).
        num_text_tokens = self.seq_len - self.image_seq_length
        text_tokens = torch.randint(
            low=1,
            high=self.vocab_size,
            size=(num_text_tokens,),
            dtype=torch.long,
        )

        # Concatenate to form the full sequence.
        token_ids = torch.cat((image_tokens, text_tokens), dim=0)

        return token_ids


def get_mock_vlm_dataloader(
    batch_size: int = 8,
    dataset_size: int = 100,
    image_size: int = 224,
    seq_len: int = 77,
    image_seq_length: int = 32,
    num_workers: int = 0,
    pad_token_id: int = 0,
    image_token_id: int = 50000,
) -> DataLoader:
    """
    Create a DataLoader for mock VLM data.

    Args:
        batch_size: Batch size
        dataset_size: Size of the dataset
        image_size: Size of the square images
        seq_len: Total length of the token sequence (image + text)
        image_seq_length: Number of image tokens to pad
        num_workers: Number of worker processes for data loading
        pad_token_id: ID for padding token
        image_token_id: ID for image placeholder token

    Returns:
        DataLoader for the mock VLM dataset
    """
    dataset = MockVLMDataset(
        size=dataset_size,
        image_size=image_size,
        seq_len=seq_len,
        image_seq_length=image_seq_length,
        pad_token_id=pad_token_id,
        image_token_id=image_token_id,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda batch: _collate_fn(batch),
    )

    return dataloader


def _collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for the DataLoader.

    Args:
        batch: List of dictionaries from the dataset

    Returns:
        Dictionary of batched tensors
    """
    images = torch.stack([item["images"] for item in batch])
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    loss_mask = torch.stack([item["loss_mask"] for item in batch])
    position_ids = torch.stack([item["position_ids"] for item in batch])

    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
        "modality_inputs": {
            "clip_encoder": {
                "images": images,
            }
        },
    }


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Provide datasets for training, validation, and testing."""
    from megatron.core import mpu
    from megatron.training import get_args

    args = get_args()

    # Print some info to confirm args are available
    print(f"Creating datasets with batch size: {args.micro_batch_size}")
    print(f"Image size: {args.image_size}")
    print(f"Image sequence length: {args.image_seq_length}")
    print(f"Total sequence length: {args.total_seq_length}")

    # Only build dataset on tensor parallel rank 0
    if mpu.get_tensor_model_parallel_rank() == 0:

        from examples.mimo.data.mock import MockVLMDataset

        train_dataset = MockVLMDataset(
            size=train_val_test_num_samples[0],
            image_size=args.image_size,
            seq_len=args.total_seq_length,
            image_seq_length=args.image_seq_length,
            pad_token_id=args.pad_token_id,
            image_token_id=args.image_token_id,
        )

        # Use the same dataset type for validation
        valid_dataset = MockVLMDataset(
            size=train_val_test_num_samples[1] if train_val_test_num_samples[1] > 0 else 100,
            image_size=args.image_size,
            seq_len=args.total_seq_length,
            image_seq_length=args.image_seq_length,
            pad_token_id=args.pad_token_id,
            image_token_id=args.image_token_id,
        )

        # No test dataset for now
        test_dataset = None
    else:
        train_dataset = None
        valid_dataset = None
        test_dataset = None

    return train_dataset, valid_dataset, test_dataset

if __name__ == "__main__":
    print("\nCreating mock VLM dataloader...")
    dataloader = get_mock_vlm_dataloader(batch_size=4, dataset_size=10)

    print(f"DataLoader has {len(dataloader)} batches")

    for batch in dataloader:
        print("\nBatch from dataloader:")
        for key, tensor in batch.items():
            print(f"  {key}: {tensor.shape}")
        break
