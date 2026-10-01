# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the DeepSeek-V4-Flash-Vision real-data SFT pipeline."""

import json
import math
import re

import pytest
import torch
from PIL import Image

from examples.multimodal_dev.data.deepseek_v4_vl import (
    ASSISTANT_SP_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    IGNORE_INDEX,
    IMAGE_PLACEHOLDER,
    THINKING_END_TOKEN,
    USER_SP_TOKEN,
    DeepSeekV4VLDataset,
    SampleTooLongError,
    build_sample,
    load_jsonl_conversations,
    process_image,
    render_conversation,
    safe_resize,
    segments_to_prompt,
)
from examples.multimodal_dev.models.deepseek_v4.configuration import (
    COMPRESS_PAD_TO,
    DEEPSEEK_V4_VOCAB_SIZE,
    IMAGE,
    IMAGE_END,
    IMAGE_START,
    VISION_DOWNSAMPLE_RATIO,
    VISION_MAX_IMAGE_TOKENS,
    VISION_PATCH_SIZE,
    build_image_block,
    get_deepseek_v4_vision_config,
)

V = DEEPSEEK_V4_VOCAB_SIZE
P = VISION_PATCH_SIZE


# ---------------------------------------------------------------------------
# Reference copy of the official inference/image_processor.py sizing logic
# (deepseek-ai/DeepSeek-V4-Flash-Vision-Exp, MIT). Kept verbatim for parity.
# ---------------------------------------------------------------------------


def _ref_grid_tokens(best_height, best_width, patch_size, downsample_ratio):
    n_llm_h = math.ceil((best_height // patch_size) / downsample_ratio)
    n_llm_w = math.ceil((best_width // patch_size) / downsample_ratio)
    num_tokens = n_llm_h * (n_llm_w + 1) + 2
    if n_llm_h % 2 == 1:
        num_tokens += n_llm_w + 1
    num_tokens += (n_llm_h + 1) // 2 * (n_llm_w + 1) % 2 * 2
    return n_llm_h, n_llm_w, num_tokens


def _ref_solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token):
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
    n_llm_h, n_llm_w, num_tokens = _ref_grid_tokens(
        best_height, best_width, patch_size, downsample_ratio
    )
    return n_llm_h, n_llm_w, best_height, best_width, num_tokens


def _ref_safe_resize(height, width, best_height, best_width, patch_size, downsample_ratio, max_n):
    max_n -= COMPRESS_PAD_TO - 1
    n_llm_h, n_llm_w, num_tokens = _ref_grid_tokens(
        best_height, best_width, patch_size, downsample_ratio
    )
    budget = max_n
    while num_tokens > max_n:
        n_llm_h, n_llm_w, best_height, best_width, num_tokens = _ref_solve_resize_ratio(
            height, width, patch_size, downsample_ratio, budget
        )
        budget -= 1
    return n_llm_h, n_llm_w, best_height, best_width


def _ref_target(width, height, min_pixels=147_456, ratio_cap=8):
    if width > height * ratio_cap:
        width = height * ratio_cap
    if 0 < width * height < min_pixels:
        ratio = (min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / P) * P
    best_height = math.ceil(height / P) * P
    return _ref_safe_resize(
        height, width, best_height, best_width, P, VISION_DOWNSAMPLE_RATIO, VISION_MAX_IMAGE_TOKENS
    )


# ---------------------------------------------------------------------------
# Fake tokenizer: special tokens map to single IDs, other characters to IDs.
# ---------------------------------------------------------------------------


class FakeTokenizer:
    SPECIAL = {
        BOS_TOKEN: 0,
        EOS_TOKEN: 1,
        USER_SP_TOKEN: 2,
        ASSISTANT_SP_TOKEN: 3,
        THINKING_END_TOKEN: 4,
        IMAGE_PLACEHOLDER: 5,
    }
    _pattern = re.compile("(" + "|".join(re.escape(token) for token in SPECIAL) + ")")

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        ids = []
        for piece in self._pattern.split(text):
            if piece in self.SPECIAL:
                ids.append(self.SPECIAL[piece])
            else:
                ids.extend(1000 + ord(char) % 5000 for char in piece)
        return ids


def _encode(text):
    return FakeTokenizer().encode(text, add_special_tokens=False)


def _image(width, height, color=(200, 30, 60)):
    return Image.new("RGB", (width, height), color)


def _conversation(images, question="Read it.", answer="OK"):
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": question})
    return [{"role": "user", "content": content}, {"role": "assistant", "content": answer}]


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

SIZES = [
    (224, 224),
    (640, 480),
    (480, 640),
    (1920, 1080),
    (1080, 1920),
    (100, 30),
    (30, 100),
    (4000, 200),
    (200, 4000),
    (5000, 300),
    (17, 17),
    (3024, 4032),
    (768, 768),
    (1000, 120),
    (64, 2048),
    (2048, 64),
    (377, 911),
    (911, 377),
    (1, 1),
    (8000, 1000),
]


@pytest.mark.parametrize("width,height", SIZES)
def test_resize_matches_official_and_fits_budget(width, height):
    n_llm_h, n_llm_w, best_h, best_w = _ref_target(width, height)
    image = process_image(_image(width, height))

    assert (image.n_llm_h, image.n_llm_w) == (n_llm_h, n_llm_w)
    assert (image.n_vit_h * P, image.n_vit_w * P) == (best_h, best_w)
    assert image.n_llm_h == math.ceil(image.n_vit_h / VISION_DOWNSAMPLE_RATIO)
    assert image.n_llm_w == math.ceil(image.n_vit_w / VISION_DOWNSAMPLE_RATIO)
    assert image.patches.shape == (image.n_vit_h * image.n_vit_w, 3 * P * P)
    assert image.patches.dtype == torch.bfloat16
    # Worst-case block (any start position) never exceeds the 384-token budget.
    for start in range(COMPRESS_PAD_TO):
        types, _ = build_image_block(image.n_llm_h, image.n_llm_w, start)
        assert types.numel() <= VISION_MAX_IMAGE_TOKENS


def test_safe_resize_is_the_official_function():
    for height, width in [(480, 640), (2000, 100), (30, 30)]:
        args = (height, width, math.ceil(height / P) * P, math.ceil(width / P) * P, P, 3, 384)
        assert safe_resize(*args) == _ref_safe_resize(*args)


def test_patches_are_chw_ordered_row_major_crops():
    # Single-colour channels make every pixel's normalized value predictable.
    image = process_image(_image(224, 224, color=(255, 0, 127)))
    patch = image.patches[0].float().view(3, P, P)
    assert torch.allclose(patch[0], torch.ones(P, P))
    assert torch.allclose(patch[1], -torch.ones(P, P))
    assert torch.allclose(patch[2], torch.full((P, P), 127 / 255 * 2 - 1), atol=1e-2)

    # Left half white, right half black: the first patch row must switch halfway.
    canvas = Image.new("RGB", (448, 448), (0, 0, 0))
    canvas.paste((255, 255, 255), (0, 0, 224, 448))
    image = process_image(canvas)
    first_row = image.patches.float().view(image.n_vit_h, image.n_vit_w, 3, P, P)[0, :, 0]
    means = first_row.mean(dim=(1, 2))
    assert means[0] > 0.9 and means[-1] < -0.9
    assert torch.all(means[:-1] >= means[1:])  # monotone white -> black across columns


def test_letterbox_padding_is_gray():
    # 400x380 fits the budget at its ceil-to-patch size 406x392, so ImageOps.pad scales it to
    # 406x385 and centres it vertically: the top three pixel rows are gray letterbox.
    image = process_image(_image(400, 380, color=(0, 0, 0)))
    assert (image.n_vit_h, image.n_vit_w) == (28, 29)
    grid = image.patches.float().view(image.n_vit_h, image.n_vit_w, 3, P, P)
    gray = 127 / 255 * 2 - 1
    assert torch.allclose(
        grid[0, :, :, 0, :], torch.full_like(grid[0, :, :, 0, :], gray), atol=1e-2
    )
    assert torch.allclose(grid[1:-1], -torch.ones_like(grid[1:-1]))


# ---------------------------------------------------------------------------
# Chat rendering and sample construction
# ---------------------------------------------------------------------------


def test_render_matches_official_non_thinking_template():
    messages = [
        {"role": "user", "content": [{"type": "image", "url": "a.png"}, "Q1"]},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": [{"type": "text", "text": "A2"}]},
    ]
    prompt = segments_to_prompt(render_conversation(messages))
    assert prompt == (
        f"{BOS_TOKEN}{USER_SP_TOKEN}{IMAGE_PLACEHOLDER}Q1{ASSISTANT_SP_TOKEN}{THINKING_END_TOKEN}"
        f"A1{EOS_TOKEN}{USER_SP_TOKEN}Q2{ASSISTANT_SP_TOKEN}{THINKING_END_TOKEN}A2{EOS_TOKEN}"
    )


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "assistant", "content": "x"}],
        [{"role": "user", "content": "x"}],
        [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
    ],
)
def test_render_rejects_malformed_conversations(messages):
    with pytest.raises((ValueError, NotImplementedError)):
        render_conversation(messages)


def _expected_tokens(messages, sizes):
    """Expand placeholders the way the official prepare_vl_inputs does."""
    prompt_ids = _encode(segments_to_prompt(render_conversation(messages)))
    tokens = []
    size_iter = iter(sizes)
    for token in prompt_ids:
        if token != FakeTokenizer.SPECIAL[IMAGE_PLACEHOLDER]:
            tokens.append(token)
            continue
        n_llm_h, n_llm_w, _, _ = _ref_target(*next(size_iter))
        types, _ = build_image_block(n_llm_h, n_llm_w, len(tokens))
        tokens += (types + V).tolist()
    return tokens


def test_sample_matches_official_placeholder_expansion():
    sizes = [(640, 480), (300, 900)]
    messages = _conversation([_image(*size) for size in sizes], answer="Total 12.50")
    sample = build_sample(messages, _encode, seq_length=4096)

    assert sample["input_ids"].tolist() == _expected_tokens(messages, sizes)
    ids = sample["input_ids"]
    assert (ids == V + IMAGE_START).sum() == 2 and (ids == V + IMAGE_END).sum() == 2
    grid = sample["image_grid_thw"]
    assert grid.shape == (2, 3) and torch.all(grid[:, 0] == 1)
    assert sample["pixel_values"].shape[0] == int((grid[:, 1] * grid[:, 2]).sum())
    num_image_rows = sum(
        math.ceil(h / VISION_DOWNSAMPLE_RATIO) * math.ceil(w / VISION_DOWNSAMPLE_RATIO)
        for h, w in grid[:, 1:].tolist()
    )
    assert int((ids == V + IMAGE).sum()) == num_image_rows


def test_loss_only_on_assistant_text_and_eos():
    messages = _conversation([_image(224, 224)], answer="AB")
    sample = build_sample(messages, _encode, seq_length=1024)
    ids, labels, mask = sample["input_ids"], sample["labels"], sample["loss_mask"]

    targets = labels[mask.bool()].tolist()
    assert targets == _encode("AB") + [FakeTokenizer.SPECIAL[EOS_TOKEN]]
    assert torch.equal(labels[mask.bool()], ids[1:][mask[:-1].bool()])
    assert torch.all(labels[~mask.bool()] == IGNORE_INDEX)
    assert torch.all(labels < V)
    assert mask[-1] == 0


def test_multi_turn_trains_every_assistant_turn():
    messages = _conversation([_image(224, 224)], answer="A1") + [
        {"role": "user", "content": "more?"},
        {"role": "assistant", "content": "A2"},
    ]
    sample = build_sample(messages, _encode, seq_length=1024)
    eos = FakeTokenizer.SPECIAL[EOS_TOKEN]
    targets = sample["labels"][sample["loss_mask"].bool()].tolist()
    assert targets == _encode("A1") + [eos] + _encode("A2") + [eos]


def test_trailing_text_truncates_but_image_blocks_never_split():
    messages = _conversation([_image(224, 224)], answer="x" * 500)
    full = build_sample(messages, _encode, seq_length=10_000)
    image_end = int((full["input_ids"] == V + IMAGE_END).nonzero()[-1])

    cut = build_sample(messages, _encode, seq_length=image_end + 20)
    assert cut["input_ids"].numel() == image_end + 20
    assert torch.equal(cut["input_ids"], full["input_ids"][: image_end + 20])

    with pytest.raises(SampleTooLongError):
        build_sample(messages, _encode, seq_length=image_end)


def test_dataset_skips_samples_that_cannot_fit():
    short = {"messages": _conversation([_image(56, 56)], answer="ok")}
    huge = {"messages": _conversation([_image(1000, 1000)] * 3, answer="ok")}
    dataset = DeepSeekV4VLDataset([huge, short], FakeTokenizer(), seq_length=400)
    sample = dataset[0]
    assert dataset.num_skipped == 1
    assert sample["image_grid_thw"].shape[0] == 1


def test_jsonl_loading_resolves_relative_image_paths(tmp_path):
    (tmp_path / "img").mkdir()
    _image(320, 240).save(tmp_path / "img" / "a.png")
    record = {
        "messages": [
            {"role": "user", "content": [{"type": "image", "url": "img/a.png"}, "What?"]},
            {"role": "assistant", "content": "A cat."},
        ]
    }
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(record) + "\n\n")

    conversations = load_jsonl_conversations(str(path))
    dataset = DeepSeekV4VLDataset(conversations, FakeTokenizer(), seq_length=1024)
    sample = dataset[0]
    assert sample["image_grid_thw"].tolist() == [[1, *[s // P for s in _ref_target(320, 240)[2:]]]]


def test_samples_drive_the_vision_encoder_and_scatter_contract():
    """Encoder output rows must line up 1:1 with IMAGE tokens in the sample."""
    from examples.multimodal_dev.models.deepseek_v4.vision_encoder import DeepSeekV4VisionEncoder

    torch.manual_seed(0)
    config = get_deepseek_v4_vision_config(num_layers_override=1)
    config.params_dtype = torch.float32
    config.vision_out_hidden_size = 64
    encoder = DeepSeekV4VisionEncoder(config).eval()

    messages = _conversation([_image(448, 336), _image(200, 700)])
    sample = build_sample(messages, _encode, seq_length=4096)
    with torch.no_grad():
        rows = encoder(sample["pixel_values"].float(), sample["image_grid_thw"])
    assert rows.shape == (int((sample["input_ids"] == V + IMAGE).sum()), 64)
    assert torch.isfinite(rows).all()


def test_vanilla_collated_batch_survives_pack_or_pad(monkeypatch):
    """Variable-size samples go through the BSHD padding path used in training."""
    from examples.multimodal_dev import forward_step

    monkeypatch.setattr(forward_step.mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(forward_step.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(forward_step.mpu, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(forward_step, "broadcast_data_batch", lambda data, device: data)

    batch = [
        build_sample(_conversation([_image(224, 224)]), _encode, seq_length=2048),
        build_sample(
            _conversation([_image(900, 300)], answer="long" * 20), _encode, seq_length=2048
        ),
    ]
    lengths = [sample["input_ids"].numel() for sample in batch]
    out = forward_step.pack_or_pad_batch(batch, use_packed_sequence=False, seq_length=2048)

    assert out["input_ids"].shape == (2, max(lengths))
    assert out["pixel_values"].shape[0] == sum(
        int((g[:, 1] * g[:, 2]).sum()) for g in [s["image_grid_thw"] for s in batch]
    )
    assert out["image_grid_thw"].shape == (2, 3)
    assert out["padding_mask"][0, lengths[0] :].all()
