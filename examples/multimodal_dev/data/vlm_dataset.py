# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Simple VLM dataset for multimodal_dev training.

Single-turn image-text dataset using a HuggingFace ``AutoProcessor`` for
tokenization and image preprocessing.  Currently supports CORD-V2 (receipt
OCR).  No multi-turn support — each sample is one image + question →
answer pair.

Because Megatron's DataLoader uses ``default_collate`` (no custom collate
function), all images are resized to a fixed resolution so that
``pixel_values`` has a consistent shape across samples.

Usage::

    torchrun ... pretrain_multimodal.py \\
        --model-arch qwen35_vl --dataset-provider cord_v2 \\
        --hf-processor-path Qwen/Qwen2.5-VL-7B-Instruct \\
        --image-size 448 --seq-length 2048
"""

import json
import logging
import random
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CORD-V2 helpers
# ---------------------------------------------------------------------------

def _json2token(obj, sort_json_key=True):
    """Convert a JSON object to a token-sequence string (Donut format)."""
    if isinstance(obj, dict):
        if len(obj) == 1 and "text_sequence" in obj:
            return obj["text_sequence"]
        output = ""
        keys = sorted(obj.keys(), reverse=True) if sort_json_key else obj.keys()
        for k in keys:
            output += f"<s_{k}>" + _json2token(obj[k], sort_json_key) + f"</s_{k}>"
        return output
    if isinstance(obj, list):
        return "<sep/>".join(_json2token(item, sort_json_key) for item in obj)
    return str(obj)


def load_cord_v2(split="train"):
    """Load CORD-V2 and return a list of ``{image, question, answer}`` dicts."""
    from datasets import load_dataset

    ds = load_dataset("naver-clova-ix/cord-v2", split=split)
    rng = random.Random(42)
    examples = []
    for ex in ds:
        gt = json.loads(ex["ground_truth"])
        gt_jsons = gt.get("gt_parses") or [gt["gt_parse"]]
        text = rng.choice(
            [_json2token(g, sort_json_key=True) for g in gt_jsons]
        )
        examples.append(
            {"image": ex["image"], "question": "Describe this image.", "answer": text}
        )
    return examples


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CordV2VLMDataset(Dataset):
    """Single-turn VLM dataset backed by CORD-V2.

    Each sample is tokenized by the HF ``AutoProcessor`` and resized to a
    fixed resolution so that ``pixel_values`` has consistent shape across
    samples (required by ``default_collate``).

    Args:
        examples: Output of :func:`load_cord_v2`.
        processor: ``AutoProcessor`` instance.
        seq_length: Pad / truncate ``input_ids`` to this length.
        image_size: Resize all images to ``(image_size, image_size)``.
        image_token_id: Token ID for image placeholders.
        target_length: Virtual dataset length (repeats examples if needed).
    """

    def __init__(
        self,
        examples: List[Dict],
        processor,
        seq_length: int = 2048,
        image_size: int = 448,
        image_token_id: Optional[int] = None,
        target_length: Optional[int] = None,
    ):
        self.examples = examples
        self.processor = processor
        self.seq_length = seq_length
        self.image_size = (image_size, image_size)
        self._length = target_length if target_length else len(examples)

        tok = processor.tokenizer
        self.pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else 0

        # Resolve image token ID
        if image_token_id is not None:
            self.image_token_id = image_token_id
        else:
            vocab = tok.get_vocab()
            for candidate in ("<|image_pad|>", "<|placeholder|>"):
                if candidate in vocab:
                    self.image_token_id = vocab[candidate]
                    break
            else:
                self.image_token_id = None

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx % len(self.examples)]

        # Fixed-resolution image
        image = example["image"].convert("RGB").resize(self.image_size)

        # Build single-turn conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": example["question"]},
                ],
            },
            {
                "role": "assistant",
                "content": example["answer"],
            },
        ]

        # Tokenize + extract pixel values via HF processor
        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False,
        )
        batch = self.processor(
            text=[text], images=[image], return_tensors="pt",
        )

        input_ids = batch["input_ids"].squeeze(0)
        pixel_values = batch["pixel_values"]       # [total_patches, pixel_dim]
        image_grid_thw = batch["image_grid_thw"]   # [1, 3]

        # Pad or truncate input_ids to seq_length
        cur_len = len(input_ids)
        valid_len = min(cur_len, self.seq_length)
        if cur_len > self.seq_length:
            logger.warning(
                "Truncating input_ids from %d to %d tokens (seq_length). "
                "Consider increasing --seq-length to avoid dropping tokens.",
                cur_len,
                self.seq_length,
            )
            input_ids = input_ids[: self.seq_length]
        else:
            pad = torch.full(
                (self.seq_length - cur_len,),
                self.pad_token_id,
                dtype=input_ids.dtype,
            )
            input_ids = torch.cat([input_ids, pad])

        # Labels: shifted next-token prediction targets
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100

        # Loss mask: 1 for trainable positions, 0 for padding / image tokens
        loss_mask = torch.ones(self.seq_length, dtype=torch.float32)
        loss_mask[input_ids == self.pad_token_id] = 0.0
        if self.image_token_id is not None:
            loss_mask[input_ids == self.image_token_id] = 0.0
        loss_mask[-1] = 0.0

        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            # Provide per-sample cu_seqlens so packed-sequence path can
            # consume data-side lengths directly (aligned with other
            # sequence packing code paths).
            "cu_seqlens": torch.tensor([0, valid_len], dtype=torch.int32),
            "cu_seqlens_padded": torch.tensor([0, valid_len], dtype=torch.int32),
            "max_seqlen": torch.tensor(valid_len, dtype=torch.int32),
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }


# ---------------------------------------------------------------------------
# Megatron dataset provider interface
# ---------------------------------------------------------------------------

def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Provide CORD-V2 train / val / test datasets.

    Requires ``--hf-processor-path`` to point to a HuggingFace VL model
    (e.g. ``Qwen/Qwen2.5-VL-7B-Instruct``) whose processor handles
    tokenization and image preprocessing.
    """
    from transformers import AutoProcessor

    from megatron.training import get_args

    args = get_args()

    processor_path = getattr(args, "hf_processor_path", None)
    if processor_path is None:
        raise ValueError(
            "cord_v2 dataset requires --hf-processor-path "
            "(e.g. Qwen/Qwen2.5-VL-7B-Instruct)"
        )
    processor = AutoProcessor.from_pretrained(
        processor_path, trust_remote_code=True,
    )

    seq_length = (
        getattr(args, "total_seq_length", None)
        or getattr(args, "seq_length", 2048)
    )
    image_size = getattr(args, "image_size", 448)
    image_token_id = getattr(args, "image_token_id", None)

    # Load real data
    train_examples = load_cord_v2(split="train")
    val_examples = load_cord_v2(split="validation")
    test_examples = load_cord_v2(split="test")

    def _make(examples, num_samples):
        return CordV2VLMDataset(
            examples=examples,
            processor=processor,
            seq_length=seq_length,
            image_size=image_size,
            image_token_id=image_token_id,
            target_length=num_samples,
        )

    train_ds = _make(train_examples, train_val_test_num_samples[0])
    val_ds = _make(val_examples, max(train_val_test_num_samples[1], 1))
    test_ds = _make(test_examples, max(train_val_test_num_samples[2], 1))

    return train_ds, val_ds, test_ds
