# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
# Qwen3-VL multimodal dataset for QAT training.
"""
Qwen3-VL multimodal dataset that loads JSONL files with images.

This dataset reads JSONL files containing:
- text: Conversation text with <image> token markers
- images: List of relative image paths

And returns:
- tokens: Tokenized input_ids
- labels: Target labels for causal LM
- loss_mask: Mask for loss computation
- position_ids: Position IDs
- pixel_values: Preprocessed image tensors
- image_grid_thw: Grid dimensions for variable resolution
"""

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    from transformers import AutoProcessor
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


def qwen3vl_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for Qwen3-VL dataset.

    Handles variable-length pixel_values by concatenating instead of stacking.
    Pads text tokens to the same length within a batch.

    Args:
        batch: List of sample dictionaries from __getitem__

    Returns:
        Collated batch dictionary
    """
    if not batch:
        return {}

    # Find max sequence length in batch
    max_seq_len = max(sample["tokens"].size(0) for sample in batch)

    # Initialize output tensors
    batch_size = len(batch)
    tokens = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    labels = torch.full((batch_size, max_seq_len), -100, dtype=torch.long)
    loss_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.float32)
    position_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)

    # Collect pixel_values and grid_thw (variable length)
    all_pixel_values = []
    all_grid_thw = []
    has_images = False

    for i, sample in enumerate(batch):
        seq_len = sample["tokens"].size(0)

        # Copy tokens (left-padded or truncated)
        tokens[i, :seq_len] = sample["tokens"][:max_seq_len]
        labels[i, :seq_len] = sample["labels"][:max_seq_len]
        loss_mask[i, :seq_len] = sample["loss_mask"][:max_seq_len]
        position_ids[i, :seq_len] = sample["position_ids"][:max_seq_len]

        # Collect pixel_values if present
        if "pixel_values" in sample and sample["pixel_values"] is not None:
            has_images = True
            all_pixel_values.append(sample["pixel_values"])

            if "image_grid_thw" in sample and sample["image_grid_thw"] is not None:
                all_grid_thw.append(sample["image_grid_thw"])

    result = {
        "tokens": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
    }

    # Concatenate pixel_values along first dimension (variable num patches)
    if has_images and all_pixel_values:
        # Concatenate all pixel_values: [total_patches, hidden_dim]
        result["pixel_values"] = torch.cat(all_pixel_values, dim=0)

        # Stack grid_thw if available: [batch_size, 3] or [num_images, 3]
        if all_grid_thw:
            result["image_grid_thw"] = torch.cat(all_grid_thw, dim=0)

    return result


@dataclass
class Qwen3VLDatasetConfig:
    """Configuration for Qwen3-VL multimodal dataset."""

    # JSONL file paths
    jsonl_paths: List[str] = None

    # Image settings
    image_base_dir: str = None  # Base directory for relative image paths
    img_h: int = 384
    img_w: int = 384

    # Tokenizer/Processor
    processor_name: str = "Qwen/Qwen3-VL-8B-Instruct"

    # Sequence settings
    sequence_length: int = 4096

    # Random seed
    random_seed: int = 42


class Qwen3VLDataset(Dataset):
    """Qwen3-VL multimodal dataset for QAT training.

    Loads JSONL files containing text and image paths, processes images
    using Qwen3-VL processor, and returns batches suitable for training.
    """

    def __init__(
        self,
        config: Qwen3VLDatasetConfig,
        split: str = "train",
    ):
        """Initialize the dataset.

        Args:
            config: Dataset configuration
            split: Data split ("train", "valid", or "test")
        """
        super().__init__()

        if not HAS_TRANSFORMERS:
            raise ImportError("transformers is required for Qwen3VLDataset")

        self.config = config
        self.split = split
        self.sequence_length = config.sequence_length

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            config.processor_name,
            trust_remote_code=True,
        )
        self.tokenizer = self.processor.tokenizer

        # Set image token
        self.image_token = "<|image_pad|>"
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)

        # Cache frequently used tokenizer lookups (Fix 1: avoid per-sample calls)
        self._im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self._im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self._assistant_tokens = self.tokenizer.encode("assistant", add_special_tokens=False)
        self._newline_id = self.tokenizer.convert_tokens_to_ids("\n")

        # Set custom collate function for variable-length pixel_values
        self.collate_fn = qwen3vl_collate_fn

        # HF dataset cache: (hf_name, hf_config, split) -> loaded dataset
        self._hf_datasets = {}

        # Directory listing cache for image path resolution (Fix 2)
        self._dir_listing_cache = {}

        # Load all samples from JSONL files
        self.samples = []
        if config.jsonl_paths:
            for jsonl_path in config.jsonl_paths:
                if os.path.exists(jsonl_path):
                    self._load_jsonl(jsonl_path)

        # Pre-resolve image paths (Fix 2: avoid per-sample filesystem stat calls)
        self._resolved_paths = {}
        self._pre_resolve_image_paths()

        print(f"[Qwen3VLDataset] Loaded {len(self.samples)} samples for {split}")

        # Random state
        self.rng = np.random.RandomState(config.random_seed)

    def _load_jsonl(self, jsonl_path: str):
        """Load samples from a JSONL file."""
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    # Store jsonl directory for relative image paths
                    sample['_jsonl_dir'] = os.path.dirname(jsonl_path)
                    self.samples.append(sample)
                except json.JSONDecodeError:
                    continue

    def __len__(self) -> int:
        return len(self.samples)

    def _cached_listdir(self, dir_path: str) -> List[str]:
        """Cached os.listdir to avoid repeated filesystem calls."""
        if dir_path not in self._dir_listing_cache:
            try:
                self._dir_listing_cache[dir_path] = os.listdir(dir_path)
            except OSError:
                self._dir_listing_cache[dir_path] = []
        return self._dir_listing_cache[dir_path]

    def _pre_resolve_image_paths(self):
        """Pre-resolve all image paths during init to avoid per-sample stat calls."""
        for sample_idx, sample in enumerate(self.samples):
            image_paths = sample.get('images', [])
            jsonl_dir = sample.get('_jsonl_dir', '')
            for img_path in image_paths:
                cache_key = (img_path, jsonl_dir)
                if cache_key not in self._resolved_paths:
                    self._resolved_paths[cache_key] = self._resolve_image_path_uncached(
                        img_path, jsonl_dir
                    )

    def _resolve_image_path_uncached(self, image_path: str, jsonl_dir: str) -> str:
        """Resolve relative image path to absolute path (uncached implementation).

        Handles multiple directory structures:
        - Direct relative paths from JSONL directory
        - source_images/ subdirectory
        - source_images/extracted/<archive_name>/ subdirectory
        """
        if os.path.isabs(image_path):
            return image_path

        # Try relative to image_base_dir first
        if self.config.image_base_dir:
            full_path = os.path.join(self.config.image_base_dir, image_path)
            if os.path.exists(full_path):
                return full_path

        # Try relative to JSONL directory
        full_path = os.path.join(jsonl_dir, image_path)
        if os.path.exists(full_path):
            return full_path

        # Try relative to source_images subdirectory
        full_path = os.path.join(jsonl_dir, "source_images", image_path)
        if os.path.exists(full_path):
            return full_path

        # Try relative to source_images/extracted/ directly
        extracted_dir = os.path.join(jsonl_dir, "source_images", "extracted")
        if os.path.isdir(extracted_dir):
            full_path = os.path.join(extracted_dir, image_path)
            if os.path.exists(full_path):
                return full_path

            # Try with archive subdirectory prefix (use cached listdir)
            for subdir in self._cached_listdir(extracted_dir):
                full_path = os.path.join(extracted_dir, subdir, image_path)
                if os.path.exists(full_path):
                    return full_path

        return image_path  # Return as-is, let downstream handle error

    def _resolve_image_path(self, image_path: str, jsonl_dir: str) -> str:
        """Resolve relative image path to absolute path (cached).

        Returns pre-resolved path from init, falling back to uncached resolution.
        """
        cache_key = (image_path, jsonl_dir)
        resolved = self._resolved_paths.get(cache_key)
        if resolved is not None:
            return resolved
        # Fallback for paths not seen during init
        resolved = self._resolve_image_path_uncached(image_path, jsonl_dir)
        self._resolved_paths[cache_key] = resolved
        return resolved

    def _load_image(self, image_path: str) -> Optional[Image.Image]:
        """Load an image from path."""
        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            return None

    def _load_image_from_hf(self, sample: Dict[str, Any], image_path: str) -> Optional[Image.Image]:
        """Load image from HuggingFace dataset when local file not found.

        Uses hf_image_source metadata saved during download to lazily load
        images from the source HF dataset. Loaded images are cached to disk.

        Args:
            sample: The JSONL sample dict containing hf_image_source metadata
            image_path: The relative image path from the sample

        Returns:
            PIL Image if found, None otherwise
        """
        hf_source = sample.get('hf_image_source')
        if not hf_source:
            return None

        hf_name = hf_source['hf_name']
        hf_config = hf_source.get('hf_config')
        hf_split = hf_source.get('hf_split', 'train')
        image_column = hf_source.get('image_column', 'image')

        key = (hf_name, hf_config, hf_split)

        # Lazy-load the HF dataset
        if key not in self._hf_datasets:
            try:
                from datasets import load_dataset
                print(f"[Qwen3VLDataset] Loading HF dataset: {hf_name} "
                      f"(config={hf_config}, split={hf_split})")
                self._hf_datasets[key] = load_dataset(hf_name, hf_config, split=hf_split)
            except Exception as e:
                print(f"[Qwen3VLDataset] Failed to load HF dataset {hf_name}: {e}")
                self._hf_datasets[key] = None
                return None

        ds = self._hf_datasets[key]
        if ds is None:
            return None

        # Use hf_image_index if available (pre-computed during download)
        hf_idx = sample.get('hf_image_index')
        if hf_idx is not None and 0 <= hf_idx < len(ds):
            return self._extract_hf_image(ds, hf_idx, image_column, image_path)

        # Fallback: try to match by extracting numeric ID from filename
        basename = os.path.splitext(os.path.basename(image_path))[0]
        # Try to interpret basename as an integer index
        try:
            idx = int(basename)
            if 0 <= idx < len(ds):
                return self._extract_hf_image(ds, idx, image_column, image_path)
        except ValueError:
            pass

        return None

    def _extract_hf_image(
        self, ds, idx: int, image_column: str, image_path: str
    ) -> Optional[Image.Image]:
        """Extract a PIL image from an HF dataset row.

        Args:
            ds: The loaded HF dataset
            idx: Row index
            image_column: Column name containing the image(s)
            image_path: Original image path (used for multi-image matching)

        Returns:
            PIL Image if found, None otherwise
        """
        try:
            row = ds[idx]
            img_data = row.get(image_column)
            if img_data is None:
                return None

            # Handle list of images (e.g., "images" column)
            if isinstance(img_data, list):
                if len(img_data) == 0:
                    return None
                # Use first image by default; could refine with path-based matching
                img = img_data[0]
            else:
                img = img_data

            # img should be a PIL Image from the HF dataset
            if isinstance(img, Image.Image):
                return img.convert('RGB')

            return None
        except Exception as e:
            print(f"[Qwen3VLDataset] Failed to extract HF image at index {idx}: {e}")
            return None

    def _parse_conversation(self, text: str) -> List[Dict[str, Any]]:
        """Parse text into conversation format for Qwen3-VL processor.

        Converts "User: <image> ...\n\nAssistant: ..." format to chat messages.

        Args:
            text: Raw text from JSONL with User/Assistant format

        Returns:
            List of message dicts with role and content
        """
        messages = []
        current_role = None
        current_content = []

        lines = text.split('\n')
        for line in lines:
            line_stripped = line.strip()

            # Check for role markers
            if line_stripped.startswith('User:'):
                if current_role and current_content:
                    messages.append({
                        "role": current_role,
                        "content": '\n'.join(current_content).strip()
                    })
                current_role = "user"
                current_content = [line_stripped[5:].strip()]  # Remove "User:"
            elif line_stripped.startswith('Assistant:'):
                if current_role and current_content:
                    messages.append({
                        "role": current_role,
                        "content": '\n'.join(current_content).strip()
                    })
                current_role = "assistant"
                current_content = [line_stripped[10:].strip()]  # Remove "Assistant:"
            elif current_role:
                current_content.append(line)

        # Add final message
        if current_role and current_content:
            messages.append({
                "role": current_role,
                "content": '\n'.join(current_content).strip()
            })

        return messages

    def _process_sample(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Process a single sample into model inputs.

        Args:
            sample: Dictionary containing 'text' and 'images' fields

        Returns:
            Dictionary with tokens, labels, loss_mask, position_ids,
            pixel_values, and image_grid_thw
        """
        text = sample.get('text', '')
        image_paths = sample.get('images', [])
        jsonl_dir = sample.get('_jsonl_dir', '')

        # Load images (with HF dataset fallback and disk caching)
        images = []
        for img_path in image_paths:
            full_path = self._resolve_image_path(img_path, jsonl_dir)
            img = self._load_image(full_path)

            # Fallback: try loading from HF dataset
            if img is None and sample.get('hf_image_source'):
                img = self._load_image_from_hf(sample, img_path)
                # Cache to disk for subsequent runs
                if img is not None:
                    cache_path = os.path.join(jsonl_dir, "source_images", img_path)
                    try:
                        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                        img.save(cache_path)
                    except Exception:
                        pass  # Non-fatal: image is still usable in memory

            if img is not None:
                images.append(img)

        # Parse conversation and build messages for Qwen3-VL
        messages = self._parse_conversation(text)

        # Convert messages to format expected by Qwen3-VL processor
        # Replace <image> with actual image content markers
        conversation = []
        img_idx = 0
        for msg in messages:
            content = msg['content']
            # Replace <image> markers with proper image content
            if '<image>' in content and images:
                parts = content.split('<image>')
                content_list = []
                for i, part in enumerate(parts):
                    if part:
                        content_list.append({"type": "text", "text": part})
                    if i < len(parts) - 1 and img_idx < len(images):
                        content_list.append({"type": "image", "image": images[img_idx]})
                        img_idx += 1
                conversation.append({"role": msg['role'], "content": content_list})
            else:
                conversation.append({"role": msg['role'], "content": msg['content']})

        # Apply chat template and process
        try:
            formatted_text = self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False,
            )

            # Process with Qwen3-VL processor
            if images:
                inputs = self.processor(
                    text=[formatted_text],
                    images=images,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=self.sequence_length,
                    truncation=True,
                )
            else:
                inputs = self.tokenizer(
                    formatted_text,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=self.sequence_length,
                    truncation=True,
                )
        except Exception as e:
            # Fallback: process text directly
            if images:
                inputs = self.processor(
                    text=[text],
                    images=images,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=self.sequence_length,
                    truncation=True,
                )
            else:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=self.sequence_length,
                    truncation=True,
                )

        # Extract tensors (remove batch dimension)
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).squeeze(0)

        # Create labels (shift by 1 for causal LM)
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100  # Ignore last position

        # Create loss mask (1 for ASSISTANT tokens only)
        # For VLM, we only compute loss on assistant responses, not user prompts/system messages
        # Find assistant response regions by looking for assistant header tokens
        loss_mask = torch.zeros(len(input_ids), dtype=torch.float32)

        # Qwen3-VL uses <|im_start|>assistant as the header for assistant turns
        # We need to find these regions and set loss_mask=1 for assistant content
        input_ids_list = input_ids.tolist()

        # Get special token IDs
        im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        assistant_token_id = self.tokenizer.convert_tokens_to_ids("assistant")
        newline_id = self.tokenizer.convert_tokens_to_ids("\n")

        # Find assistant regions: <|im_start|>assistant\n ... <|im_end|>
        i = 0
        while i < len(input_ids_list) - 2:
            # Check for <|im_start|>assistant pattern
            if (input_ids_list[i] == im_start_id and
                i + 1 < len(input_ids_list) and input_ids_list[i + 1] == assistant_token_id):
                # Skip the header: <|im_start|>assistant\n
                start_idx = i + 2  # Skip <|im_start|> and assistant
                if start_idx < len(input_ids_list) and input_ids_list[start_idx] == newline_id:
                    start_idx += 1  # Skip newline

                # Find the end: <|im_end|>
                end_idx = start_idx
                while end_idx < len(input_ids_list) and input_ids_list[end_idx] != im_end_id:
                    end_idx += 1

                # Set loss_mask=1 for assistant content (excluding <|im_end|>)
                loss_mask[start_idx:end_idx] = 1.0
                i = end_idx + 1
            else:
                i += 1

        # Mask out padding tokens and -100 labels
        loss_mask[labels == -100] = 0.0
        if self.tokenizer.pad_token_id is not None:
            loss_mask[input_ids == self.tokenizer.pad_token_id] = 0.0

        # Position IDs
        position_ids = torch.arange(len(input_ids), dtype=torch.long)

        # Build output
        output = {
            "tokens": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
        }

        # Add image data if present
        if images and "pixel_values" in inputs:
            output["pixel_values"] = inputs["pixel_values"].squeeze(0)

            if "image_grid_thw" in inputs:
                grid_thw = inputs["image_grid_thw"].squeeze(0)
                # Ensure grid_thw is always 2D [num_images, 3]
                if grid_thw.dim() == 1:
                    grid_thw = grid_thw.unsqueeze(0)
                output["image_grid_thw"] = grid_thw

        return output

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with model inputs
        """
        sample = self.samples[idx]
        return self._process_sample(sample)


class Qwen3VLDatasetBuilder:
    """Builder for Qwen3-VL datasets from datablend configuration."""

    def __init__(
        self,
        config: Qwen3VLDatasetConfig,
        blend_path: str,
        train_val_test_num_samples: List[int],
    ):
        """Initialize the builder.

        Args:
            config: Dataset configuration
            blend_path: Path to datablend JSON file
            train_val_test_num_samples: Number of samples for [train, val, test]
        """
        self.config = config
        self.blend_path = blend_path
        self.train_val_test_num_samples = train_val_test_num_samples

        # Load blend configuration
        with open(blend_path, 'r') as f:
            self.blend_config = json.load(f)

    def _get_jsonl_paths(self, split: str) -> List[str]:
        """Get JSONL paths for a given split.

        The datablend JSON points to preprocessed binary files like:
        ./mlm_output/.../chartqa_cot_train_text_document

        We convert these to JSONL paths:
        ./mlm_output/.../chartqa_cot_train.jsonl
        """
        split_key = "valid" if split == "valid" else split
        blend_data = self.blend_config.get(split_key, [])

        jsonl_paths = []
        # blend_data format: [weight, path, weight, path, ...]
        for i in range(1, len(blend_data), 2):
            binary_path = blend_data[i]
            # Convert binary path to JSONL path
            # e.g., .../chartqa_cot_train_text_document -> .../chartqa_cot_train.jsonl
            jsonl_path = binary_path.replace("_text_document", ".jsonl")
            # Handle path relative to preprocessed -> parent
            jsonl_path = jsonl_path.replace("/preprocessed/", "/")
            if os.path.exists(jsonl_path):
                jsonl_paths.append(jsonl_path)
            else:
                print(f"[Qwen3VLDatasetBuilder] JSONL not found: {jsonl_path}")

        return jsonl_paths

    def build(self) -> tuple:
        """Build train, validation, and test datasets.

        Returns:
            Tuple of (train_ds, valid_ds, test_ds)
        """
        datasets = []

        for split in ["train", "valid", "test"]:
            jsonl_paths = self._get_jsonl_paths(split)

            if jsonl_paths:
                config = Qwen3VLDatasetConfig(
                    jsonl_paths=jsonl_paths,
                    image_base_dir=self.config.image_base_dir,
                    img_h=self.config.img_h,
                    img_w=self.config.img_w,
                    processor_name=self.config.processor_name,
                    sequence_length=self.config.sequence_length,
                    random_seed=self.config.random_seed,
                )
                ds = Qwen3VLDataset(config, split=split)
            else:
                ds = None

            datasets.append(ds)

        return tuple(datasets)
