# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Simple VLM dataset for multimodal_dev training.

Single-turn image-text dataset using a HuggingFace ``AutoProcessor`` for
tokenization and image preprocessing.  Currently supports CORD-V2 (receipt
OCR).  No multi-turn support — each sample is one image + question →
answer pair.

Each image is preprocessed via ``qwen_vl_utils.process_vision_info`` and
fed to the processor with Qwen-VL's recommended ``min_pixels`` /
``max_pixels`` budget, so the per-sample patch grid varies with aspect
ratio.  The run script must therefore pass ``--use-vanilla-collate-fn``
(which the example launcher does) so the dataloader does not try to stack
variable-shape tensors.

Usage::

    torchrun ... pretrain_multimodal.py \\
        --model-arch qwen35_vl --dataset-provider cord_v2 \\
        --hf-processor-path Qwen/Qwen3.5-397B-A17B \\
        --total-seq-length 4096 --use-vanilla-collate-fn
"""

import json
import logging
import random
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

try:
    from qwen_vl_utils import process_vision_info
    HAVE_QWEN_VL_UTILS = True
except ImportError:
    HAVE_QWEN_VL_UTILS = False

logger = logging.getLogger(__name__)

# Qwen-VL recommended pixel-budget range; lets the processor pick a
# per-image patch grid that respects aspect ratio.
_QWEN_VL_MIN_PIXELS = 256 * 28 * 28   # 200_704
_QWEN_VL_MAX_PIXELS = 1280 * 28 * 28  # 1_003_520


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

    Each sample is tokenized by the HF ``AutoProcessor`` and the image is
    handed to the processor with Qwen-VL's dynamic-resolution budget
    (``min_pixels`` / ``max_pixels``); the per-image patch grid varies with
    aspect ratio.

    Args:
        examples: Output of :func:`load_cord_v2`.
        processor: ``AutoProcessor`` instance.
        seq_length: End-truncate ``input_ids`` at this length.
        image_token_id: Token ID for image placeholders.
        target_length: Virtual dataset length (repeats examples if needed).

    NOTE:
        For the Qwen3.5-VL processor, the temporal patch dimension is
        always 2 (the processor duplicates a single frame so the 3D conv
        behaves like a 2D conv on one image) — ``image_grid_thw`` therefore
        has shape ``[num_images, 3]`` with ``T=2`` per image.
        ``pixel_values`` has shape ``[total_patches, 3 * T * P * P]`` where
        ``P`` is the processor's patch size.
    """

    def __init__(
        self,
        examples: List[Dict],
        processor,
        seq_length: int = 2048,
        image_token_id: Optional[int] = None,
        target_length: Optional[int] = None,
    ):
        if not HAVE_QWEN_VL_UTILS:
            raise ImportError(
                "qwen_vl_utils is required for Qwen3.5-VL preprocessing. "
                "Install with `pip install qwen-vl-utils`.",
            )
        self.examples = examples
        self.processor = processor
        self.seq_length = seq_length
        self._length = target_length if target_length else len(examples)
        tok = processor.tokenizer
        # Falling back to 0 is unsafe: token 0 is a real vocab token in many
        # tokenizers (incl. Qwen) and would be silently masked.  Prefer EOS,
        # and require at least one of pad/eos to be set.
        if tok.pad_token_id is not None:
            self.pad_token_id = int(tok.pad_token_id)
        elif tok.eos_token_id is not None:
            self.pad_token_id = int(tok.eos_token_id)
        else:
            raise ValueError(
                "Tokenizer has neither pad_token_id nor eos_token_id; "
                "cannot derive a safe pad id for loss masking.",
            )

        # Resolve image token ID.  Vision embeddings are scattered into
        # positions equal to this id by the model, so a wrong id silently
        # breaks training — fail loudly rather than return None.
        if image_token_id is not None:
            self.image_token_id = int(image_token_id)
        else:
            vocab = tok.get_vocab()
            for candidate in ("<|image_pad|>", "<|placeholder|>"):
                if candidate in vocab:
                    self.image_token_id = int(vocab[candidate])
                    break
            else:
                raise ValueError(
                    "Could not resolve image token id from tokenizer "
                    f"({type(tok).__name__}); pass --image-token-id "
                    "explicitly.",
                )

        # Structural tokens that must never appear as a loss target:
        # pad, image, plus everything the tokenizer registered as special
        # (im_start/im_end, vision_start/vision_end, video_pad, endoftext...).
        # Mirrors megatron-bridge's extract_skipped_token_ids convention.
        skipped: set = set(int(x) for x in (tok.all_special_ids or []))
        skipped.add(self.pad_token_id)
        skipped.add(self.image_token_id)
        self.skipped_token_ids = torch.tensor(
            sorted(skipped), dtype=torch.long,
        )

    def __len__(self) -> int:
        return self._length

    def _mark_assistant_span(
        self,
        input_ids_list: List[int],
        asst_text: str,
        loss_mask: torch.Tensor,
    ) -> bool:
        """Find ``asst_text`` as a contiguous token span in ``input_ids_list``
        and set ``loss_mask`` to 1 over those positions.

        Substring tokenization is sensitive to surrounding whitespace and
        BPE merge boundaries, so we try a few common variants.  Returns
        True if a span was found.
        """
        tokenizer = self.processor.tokenizer
        n = len(input_ids_list)
        variants = (
            asst_text,
            asst_text + "\n",
            asst_text.strip(),
            asst_text.strip() + "\n",
        )
        for variant in variants:
            span_tokens = tokenizer(
                variant, add_special_tokens=False,
            )["input_ids"]
            m = len(span_tokens)
            if m == 0 or m > n:
                continue
            # Backward search: rightmost match = the actual assistant turn.
            for start in range(n - m, -1, -1):
                if input_ids_list[start : start + m] == span_tokens:
                    loss_mask[start : start + m] = 1.0
                    return True
        return False

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx % len(self.examples)]

        # Conversation schema must include the actual image object inside
        # the content so the chat template + process_vision_info can extract
        # it (matches megatron-bridge's qwen2_5_collate_fn convention; also
        # used by the Qwen3-VL processor).
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": example["image"]},
                    {"type": "text", "text": example["question"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": example["answer"]}],
            },
        ]

        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False,
        )
        images, _ = process_vision_info(conversation)
        batch = self.processor(
            text=[text],
            images=images,
            return_tensors="pt",
            min_pixels=_QWEN_VL_MIN_PIXELS,
            max_pixels=_QWEN_VL_MAX_PIXELS,
        )

        input_ids = batch["input_ids"].squeeze(0)
        pixel_values = batch["pixel_values"].to(torch.bfloat16)
        image_grid_thw = batch["image_grid_thw"]   # [num_images, 3]

        # End-truncate so the model never sees more than seq_length tokens.
        # Qwen-VL chat template puts the image at the user-turn start and the
        # assistant answer trails at the end, so end-truncation preserves the
        # image_pad block for normal-sized images; if the image grid alone
        # already exceeds seq_length, the model will fail loudly at the
        # masked_scatter step.
        if input_ids.shape[0] > self.seq_length:
            logger.warning(
                "Sample idx=%d has %d tokens > seq_length=%d; truncating.",
                idx, input_ids.shape[0], self.seq_length,
            )
            input_ids = input_ids[: self.seq_length]

        # SFT loss mask: start fully masked, then unmask only the assistant
        # answer span found via substring token search (mirrors
        # megatron-bridge's create_multiturn_loss_mask_by_search).  The user
        # turn, chat-template tags, and image tokens stay masked.
        loss_mask = torch.zeros_like(input_ids, dtype=torch.float32)
        found = self._mark_assistant_span(
            input_ids.tolist(), example["answer"], loss_mask,
        )
        if not found:
            logger.warning(
                "Assistant span not located for example idx=%d; "
                "loss_mask will be all-zero for this sample.",
                idx,
            )

        # Shifted next-token labels: labels[i] is the target for position i.
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100

        # Mask structural tokens on the *labels* (the prediction targets),
        # not on input_ids — matches the next-token timeline.
        labels[torch.isin(labels, self.skipped_token_ids)] = -100

        # Shift loss_mask left by one so position i decides whether to learn
        # input_ids[i] -> labels[i] (== input_ids[i+1]).  Last position is
        # never trained (no next token to predict).
        loss_mask = torch.cat(
            [loss_mask[1:], torch.zeros(1, dtype=loss_mask.dtype)],
        )

        # Enforce label = -100 wherever we won't compute loss.
        labels[loss_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }


# ---------------------------------------------------------------------------
# Megatron dataset provider interface
# ---------------------------------------------------------------------------

def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Provide CORD-V2 train / val / test datasets.

    Requires ``--hf-processor-path`` to point to a HuggingFace VL model
    (e.g. ``Qwen/Qwen3.5-397B-A17B``) whose processor handles tokenization
    and image preprocessing.
    """
    from transformers import AutoProcessor

    from megatron.training import get_args

    args = get_args()

    processor_path = getattr(args, "hf_processor_path", None)
    if processor_path is None:
        raise ValueError(
            "cord_v2 dataset requires --hf-processor-path "
            "(e.g. Qwen/Qwen3.5-397B-A17B)"
        )
    processor = AutoProcessor.from_pretrained(
        processor_path, trust_remote_code=True,
    )

    seq_length = (
        getattr(args, "total_seq_length", None)
        or getattr(args, "seq_length", 2048)
    )
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
            image_token_id=image_token_id,
            target_length=num_samples,
        )

    # MegatronPretrainingSampler asserts total_samples > 0, so val/test
    # datasets must have non-zero length even when eval is disabled.
    train_ds = _make(train_examples, train_val_test_num_samples[0])
    val_ds = _make(val_examples, max(train_val_test_num_samples[1], 1))
    test_ds = _make(test_examples, max(train_val_test_num_samples[2], 1))

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen3.5-397B-A17B", trust_remote_code=True,
    )
    examples = load_cord_v2(split="train")
    dataset = CordV2VLMDataset(
        examples=examples,
        processor=processor,
        image_token_id=248056,
    )
    print(dataset[0])
