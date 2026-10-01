# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Real-data SFT pipeline for DeepSeek-V4-Flash-Vision.

This module turns image-text conversations into the exact token / pixel layout
consumed by :class:`DeepSeekV4VisionModel`:

* images are resized, padded, normalized and patchified exactly like the
  official ``inference/image_processor.py`` of
  ``deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`` (aspect-ratio-aware resize under
  a 384-token budget, gray padding, ``(x - 0.5) / 0.5`` normalization, 14x14
  RGB patches in ``(C, P, P)`` order);
* conversations are rendered with the official non-thinking chat template
  (``<bos><｜User｜>...<｜Assistant｜></think>answer<eos>``);
* each ``<｜deepseek_image｜>`` placeholder is expanded into the synthetic
  N-layout image block (``vocab_size + type``) via ``build_image_block``, whose
  compression padding depends on the block's start position;
* loss is computed only on assistant turns (content plus the closing EOS).

Two sources are provided:

``cord_v2``
    CORD-V2 receipt parsing from the Hugging Face hub (same question/answer
    construction as the Qwen3.5-VL ``cord_v2`` provider).
``jsonl``
    A local JSONL file passed with ``--dataset-path``. Each line is
    ``{"messages": [...]}`` using OpenAI/Anthropic-style content blocks::

        {"messages": [
            {"role": "user", "content": [
                {"type": "image", "url": "/data/img/0001.jpg"},
                {"type": "text", "text": "What is the total?"}]},
            {"role": "assistant", "content": "12.50"}]}

    Image blocks accept ``url`` (path, http(s) or data URL), ``data``
    (base64), ``source`` (Anthropic style), ``path`` or an in-memory ``image``
    (``PIL.Image``). Relative paths resolve against the JSONL file's folder.
"""

import base64
import io
import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from examples.multimodal_dev.models.deepseek_v4.configuration import (
    COMPRESS_PAD_TO,
    DEEPSEEK_V4_VOCAB_SIZE,
    VISION_DOWNSAMPLE_RATIO,
    VISION_MAX_IMAGE_TOKENS,
    VISION_MAX_WH_RATIO,
    VISION_MIN_PIXELS,
    VISION_PATCH_SIZE,
    build_image_block,
)

logger = logging.getLogger(__name__)

# Official DeepSeek-V4 chat-template strings (encoding/encoding_dsv4.py).
BOS_TOKEN = "<｜begin▁of▁sentence｜>"
EOS_TOKEN = "<｜end▁of▁sentence｜>"
USER_SP_TOKEN = "<｜User｜>"
ASSISTANT_SP_TOKEN = "<｜Assistant｜>"
THINKING_END_TOKEN = "</think>"
IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"

IGNORE_INDEX = -100
PAD_GRAY = (127, 127, 127)


# ---------------------------------------------------------------------------
# Image preprocessing (port of the official inference/image_processor.py)
# ---------------------------------------------------------------------------


def grid_tokens(
    best_height: int, best_width: int, patch_size: int, downsample_ratio: int
) -> tuple[int, int, int]:
    """Return ``(n_llm_h, n_llm_w, num_tokens)`` for one aligned image grid.

    ``num_tokens`` counts the N-layout block including START/END, row newlines,
    odd-row padding and the final alignment padding, but excluding the
    position-dependent compression prefix.
    """
    n_llm_h = math.ceil((best_height // patch_size) / downsample_ratio)
    n_llm_w = math.ceil((best_width // patch_size) / downsample_ratio)
    num_tokens = n_llm_h * (n_llm_w + 1) + 2
    if n_llm_h % 2 == 1:
        num_tokens += n_llm_w + 1
    num_tokens += (n_llm_h + 1) // 2 * (n_llm_w + 1) % 2 * 2
    return n_llm_h, n_llm_w, num_tokens


def solve_resize_ratio(
    height: float, width: float, patch_size: int, downsample_ratio: int, max_n_token: int
) -> tuple[int, int, int, int, int]:
    """Largest aspect-preserving grid whose block fits in ``max_n_token``."""
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    if max_w_float < 1.0:
        max_w = 1
        max_h = (max_n_token - 2) // (max_w + 1)
        if max_h % 2 == 1:
            max_h -= 1
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    elif max_h_float < 2.0:
        max_h = 2
        max_w = ((max_n_token - 2) // max_h) - 1
        assert max_w > 1
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    else:
        max_w = math.floor(max_w_float)
        max_h = math.floor(max_h_float)
        if max_h % 2 == 1:
            max_h -= 1
        beta = min(
            max_w * patch_size * downsample_ratio / width,
            max_h * patch_size * downsample_ratio / height,
        )
        best_width = math.floor(width * beta / patch_size) * patch_size
        best_height = math.floor(height * beta / patch_size) * patch_size
    n_llm_h, n_llm_w, num_tokens = grid_tokens(
        best_height, best_width, patch_size, downsample_ratio
    )
    return n_llm_h, n_llm_w, best_height, best_width, num_tokens


def safe_resize(
    height: float,
    width: float,
    best_height: int,
    best_width: int,
    patch_size: int,
    downsample_ratio: int,
    max_n_token: int,
) -> tuple[int, int, int, int]:
    """Shrink the target grid until the worst-case image block fits the budget."""
    max_n_token -= COMPRESS_PAD_TO - 1
    n_llm_h, n_llm_w, num_tokens = grid_tokens(
        best_height, best_width, patch_size, downsample_ratio
    )
    budget = max_n_token
    while num_tokens > max_n_token:
        n_llm_h, n_llm_w, best_height, best_width, num_tokens = solve_resize_ratio(
            height, width, patch_size, downsample_ratio, budget
        )
        budget -= 1
    return n_llm_h, n_llm_w, best_height, best_width


@dataclass
class ProcessedImage:
    """ViT patches and grid sizes for one image."""

    patches: torch.Tensor  # [n_vit_h * n_vit_w, 3 * P * P]
    n_vit_h: int
    n_vit_w: int
    n_llm_h: int
    n_llm_w: int


def load_image_bytes(record: dict[str, Any], base_dir: str | None = None) -> bytes:
    """Load image bytes from raw/base64 data, an Anthropic source, URL, or path."""
    data = record.get("data")
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    if isinstance(data, str):
        return base64.b64decode(data)

    source = record.get("source")
    if isinstance(source, dict):
        if source.get("data") is not None:
            return base64.b64decode(source["data"])
        if source.get("url"):
            return load_image_bytes({"url": source["url"]}, base_dir)

    url = record.get("url") or record.get("path") or record.get("image_url")
    if isinstance(url, dict):  # OpenAI {"image_url": {"url": ...}}
        url = url.get("url")
    if isinstance(url, str) and url:
        if url.startswith("data:"):
            header, _, payload = url.partition(",")
            if ";base64" not in header:
                raise ValueError(f"Unsupported data URL encoding: {header}")
            return base64.b64decode(payload)
        if url.startswith(("http://", "https://")):
            from urllib.request import urlopen

            with urlopen(url, timeout=30) as response:
                return response.read()
        if base_dir is not None and not os.path.isabs(url):
            url = os.path.join(base_dir, url)
        with open(url, "rb") as file:
            return file.read()

    raise ValueError(f"Cannot load image from record with keys {sorted(record.keys())}")


def _to_pil_rgb(record: Any, base_dir: str | None = None):
    from PIL import Image

    if isinstance(record, Image.Image):
        return record.convert("RGB")
    if isinstance(record, dict) and isinstance(record.get("image"), Image.Image):
        return record["image"].convert("RGB")
    if isinstance(record, dict) and isinstance(record.get("image"), str):
        record = {"url": record["image"]}
    if isinstance(record, str):
        record = {"url": record}
    with Image.open(io.BytesIO(load_image_bytes(record, base_dir))) as source:
        return source.convert("RGB")


def process_image(
    record: Any,
    *,
    patch_size: int = VISION_PATCH_SIZE,
    downsample_ratio: int = VISION_DOWNSAMPLE_RATIO,
    max_n_token: int = VISION_MAX_IMAGE_TOKENS,
    min_pixels: int = VISION_MIN_PIXELS,
    max_wh_ratio: float | None = VISION_MAX_WH_RATIO,
    base_dir: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> ProcessedImage:
    """Resize, pad, normalize and patchify one image like the official processor."""
    from PIL import ImageOps

    image = _to_pil_rgb(record, base_dir)
    p = patch_size
    width, height = image.size
    if max_wh_ratio is not None and width > height * max_wh_ratio:
        width = height * max_wh_ratio
    if 0 < width * height < min_pixels:
        ratio = (min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / p) * p
    best_height = math.ceil(height / p) * p
    n_llm_h, n_llm_w, best_height, best_width = safe_resize(
        height, width, best_height, best_width, p, downsample_ratio, max_n_token
    )
    n_vit_h, n_vit_w = best_height // p, best_width // p
    if max_wh_ratio is not None and image.width >= max_wh_ratio * image.height:
        image = image.resize((best_width, best_height))
    else:
        image = ImageOps.pad(image, (best_width, best_height), color=PAD_GRAY)
    x = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255
    x = ((x - 0.5) / 0.5).to(dtype)
    patches = (
        x.reshape(3, n_vit_h, p, n_vit_w, p)
        .permute(1, 3, 0, 2, 4)
        .reshape(n_vit_h * n_vit_w, 3 * p * p)
    )
    return ProcessedImage(patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w)


# ---------------------------------------------------------------------------
# Chat rendering
# ---------------------------------------------------------------------------

# A rendered conversation is a list of segments. Each segment is either
# ("text", str, trainable) or ("image", record, False).
Segment = tuple[str, Any, bool]


def _content_blocks(content: Any) -> list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        blocks = []
        for block in content:
            if isinstance(block, str):
                blocks.append({"type": "text", "text": block})
            elif isinstance(block, dict):
                blocks.append(block)
            else:
                raise TypeError(f"Unsupported content block {type(block).__name__}.")
        return blocks
    raise TypeError(f"Unsupported message content {type(content).__name__}.")


def _is_image_block(block: dict[str, Any]) -> bool:
    return block.get("type") in ("image", "image_url", "input_image")


def render_conversation(messages: Sequence[dict[str, Any]]) -> list[Segment]:
    """Render messages with the official non-thinking DeepSeek-V4 chat template.

    ``<bos>`` then, per round, ``<｜User｜>{user}<｜Assistant｜></think>`` and
    ``{assistant}<eos>``. Only assistant text and its EOS are trainable.
    Images are only allowed in user turns.
    """
    segments: list[Segment] = [("text", BOS_TOKEN, False)]
    if not messages:
        raise ValueError("Conversation has no messages.")
    expect = "user"
    for message in messages:
        role = message.get("role")
        if role == "system":
            raise NotImplementedError(
                "System prompts are not rendered by this pipeline; fold them into the first "
                "user turn or render with the official encoding_dsv4.encode_messages."
            )
        if role != expect:
            raise ValueError(f"Expected a '{expect}' turn but found '{role}'.")
        blocks = _content_blocks(message.get("content"))
        if role == "user":
            segments.append(("text", USER_SP_TOKEN, False))
            for block in blocks:
                if _is_image_block(block):
                    segments.append(("image", block, False))
                elif block.get("type", "text") == "text":
                    segments.append(("text", block.get("text", ""), False))
                else:
                    raise ValueError(f"Unsupported user content block type {block.get('type')}.")
            segments.append(("text", ASSISTANT_SP_TOKEN + THINKING_END_TOKEN, False))
            expect = "assistant"
        else:
            if any(_is_image_block(block) for block in blocks):
                raise ValueError("Images in assistant turns are not supported.")
            text = "".join(block.get("text", "") for block in blocks)
            segments.append(("text", text + EOS_TOKEN, True))
            expect = "user"
    if expect != "user":
        raise ValueError("Conversation must end with an assistant turn.")
    return segments


def segments_to_prompt(segments: Sequence[Segment]) -> str:
    """Return the prompt string (images as placeholders), e.g. for template checks."""
    return "".join(IMAGE_PLACEHOLDER if kind == "image" else value for kind, value, _ in segments)


# ---------------------------------------------------------------------------
# Sample construction
# ---------------------------------------------------------------------------


class SampleTooLongError(ValueError):
    """Raised when a sample cannot fit without cutting an image block."""


def build_sample(
    messages: Sequence[dict[str, Any]],
    encode: Callable[[str], list[int]],
    *,
    seq_length: int,
    vocab_size: int = DEEPSEEK_V4_VOCAB_SIZE,
    base_dir: str | None = None,
    image_kwargs: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    """Tokenize one conversation into a DeepSeek-V4-Vision training sample.

    ``encode`` must map a string to token IDs without adding BOS/EOS and must
    recognize the chat-template special tokens. Text is tokenized per segment;
    every segment boundary sits next to a special token, so this matches
    tokenizing the whole prompt at once.

    Returns ``input_ids``, ``labels`` (already shifted, ``-100`` where ignored),
    ``loss_mask``, ``pixel_values`` ``[total_patches, 3*P*P]`` and
    ``image_grid_thw`` ``[num_images, 3]``.
    """
    image_kwargs = dict(image_kwargs or {})
    tokens: list[int] = []
    trainable: list[bool] = []
    patches: list[torch.Tensor] = []
    grids: list[list[int]] = []

    for kind, value, is_trainable in render_conversation(messages):
        if kind == "text":
            ids = encode(value) if value else []
            tokens.extend(ids)
            trainable.extend([is_trainable] * len(ids))
            continue
        image = process_image(value, base_dir=base_dir, **image_kwargs)
        types, _ = build_image_block(image.n_llm_h, image.n_llm_w, len(tokens))
        tokens.extend((types + vocab_size).tolist())
        trainable.extend([False] * types.numel())
        patches.append(image.patches)
        grids.append([1, image.n_vit_h, image.n_vit_w])
        if len(tokens) > seq_length:
            raise SampleTooLongError(
                f"Image block ends at token {len(tokens)} > seq_length={seq_length}."
            )

    if not patches:
        raise ValueError("Conversation contains no images; DeepSeek-V4-Vision SFT needs one.")

    if len(tokens) > seq_length:
        # Image blocks are never cut (checked above); only trailing text is truncated.
        tokens = tokens[:seq_length]
        trainable = trainable[:seq_length]

    input_ids = torch.tensor(tokens, dtype=torch.long)
    trainable_t = torch.tensor(trainable, dtype=torch.bool)
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    labels[:-1] = input_ids[1:]
    # Position i learns token i+1 only when token i+1 belongs to an assistant turn.
    loss_mask = torch.zeros(input_ids.shape, dtype=torch.float32)
    loss_mask[:-1] = trainable_t[1:].float()
    # Synthetic image IDs live above the vocabulary and can never be targets.
    loss_mask[labels >= vocab_size] = 0
    labels[loss_mask == 0] = IGNORE_INDEX

    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_mask": loss_mask,
        "pixel_values": torch.cat(patches, dim=0),
        "image_grid_thw": torch.tensor(grids, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def _make_encoder(tokenizer) -> Callable[[str], list[int]]:
    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    return encode


def check_tokenizer(tokenizer) -> None:
    """Fail fast if the tokenizer does not know the DeepSeek-V4 special tokens."""
    encode = _make_encoder(tokenizer)
    for token in (BOS_TOKEN, EOS_TOKEN, USER_SP_TOKEN, ASSISTANT_SP_TOKEN, IMAGE_PLACEHOLDER):
        ids = encode(token)
        if len(ids) != 1:
            raise ValueError(
                f"Tokenizer splits special token {token!r} into {ids}; use the "
                "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp tokenizer."
            )


class DeepSeekV4VLDataset(Dataset):
    """Conversations -> DeepSeek-V4-Vision samples.

    Samples that cannot fit ``seq_length`` without cutting an image block are
    skipped by moving to the next conversation; the number of skips is logged.
    """

    def __init__(
        self,
        conversations: Sequence[dict[str, Any]],
        tokenizer,
        seq_length: int,
        target_length: int | None = None,
        vocab_size: int = DEEPSEEK_V4_VOCAB_SIZE,
        base_dir: str | None = None,
        max_skips: int = 64,
    ) -> None:
        if not conversations:
            raise ValueError("No conversations to train on.")
        check_tokenizer(tokenizer)
        self.conversations = conversations
        self.encode = _make_encoder(tokenizer)
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.base_dir = base_dir
        self.max_skips = max_skips
        self._length = target_length if target_length else len(conversations)
        self.num_skipped = 0

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        for attempt in range(self.max_skips + 1):
            conversation = self.conversations[(idx + attempt) % len(self.conversations)]
            try:
                return build_sample(
                    conversation["messages"],
                    self.encode,
                    seq_length=self.seq_length,
                    vocab_size=self.vocab_size,
                    base_dir=conversation.get("base_dir", self.base_dir),
                )
            except SampleTooLongError as error:
                self.num_skipped += 1
                logger.warning("Skipping sample idx=%d: %s", idx + attempt, error)
        raise RuntimeError(
            f"{self.max_skips + 1} consecutive samples exceed seq_length={self.seq_length}; "
            "increase --total-seq-length."
        )


def load_cord_v2_conversations(split: str = "train") -> list[dict[str, Any]]:
    """CORD-V2 receipts as single-turn image -> parse conversations."""
    from examples.multimodal_dev.data.cord_v2 import load_cord_v2

    return [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": example["image"]},
                        {"type": "text", "text": example["question"]},
                    ],
                },
                {"role": "assistant", "content": example["answer"]},
            ]
        }
        for example in load_cord_v2(split=split)
    ]


def load_jsonl_conversations(path: str) -> list[dict[str, Any]]:
    """Read ``{"messages": [...]}`` lines; image paths resolve relative to ``path``."""
    base_dir = os.path.dirname(os.path.abspath(path))
    conversations = []
    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "messages" not in record:
                raise ValueError(f"{path}:{line_number} has no 'messages' field.")
            record.setdefault("base_dir", base_dir)
            conversations.append(record)
    return conversations


def _split_paths(dataset_path: str) -> tuple[str, str | None, str | None]:
    """``train.jsonl[,valid.jsonl[,test.jsonl]]``."""
    parts = [part.strip() for part in dataset_path.split(",") if part.strip()]
    if not 1 <= len(parts) <= 3:
        raise ValueError("--dataset-path expects 1-3 comma-separated JSONL files.")
    parts += [None] * (3 - len(parts))
    return parts[0], parts[1], parts[2]


def _load_tokenizer(args):
    from transformers import AutoTokenizer

    path = getattr(args, "hf_processor_path", None) or getattr(args, "tokenizer_model", None)
    if path is None:
        raise ValueError(
            "Set --tokenizer-model (or --hf-processor-path) to the DeepSeek-V4-Flash-Vision "
            "tokenizer, e.g. deepseek-ai/DeepSeek-V4-Flash-Vision-Exp."
        )
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def _build(splits, train_val_test_num_samples):
    from megatron.training import get_args

    args = get_args()
    tokenizer = _load_tokenizer(args)
    seq_length = getattr(args, "total_seq_length", None) or args.seq_length
    datasets = []
    for index, (conversations, num_samples) in enumerate(zip(splits, train_val_test_num_samples)):
        # MegatronPretrainingSampler asserts total_samples > 0 even with eval disabled.
        target = num_samples if index == 0 else max(num_samples, 1)
        datasets.append(
            DeepSeekV4VLDataset(conversations, tokenizer, seq_length, target_length=target)
        )
    return tuple(datasets)


def cord_v2_datasets_provider(train_val_test_num_samples):
    """CORD-V2 train / validation / test datasets for DeepSeek-V4-Vision."""
    splits = [load_cord_v2_conversations(split) for split in ("train", "validation", "test")]
    return _build(splits, train_val_test_num_samples)


def jsonl_datasets_provider(train_val_test_num_samples):
    """Local JSONL datasets from ``--dataset-path train[,valid[,test]]``."""
    from megatron.training import get_args

    dataset_path = getattr(get_args(), "dataset_path", None)
    if not dataset_path:
        raise ValueError("--dataset-provider jsonl requires --dataset-path.")
    train_path, valid_path, test_path = _split_paths(dataset_path)
    train = load_jsonl_conversations(train_path)
    valid = load_jsonl_conversations(valid_path) if valid_path else train
    test = load_jsonl_conversations(test_path) if test_path else valid
    return _build((train, valid, test), train_val_test_num_samples)
